"""Slow pick of any coloured object, choosing the detector to match.

Each `color-detector-*` service on the machine handles exactly one colour, so
the colour you ask for selects which detector to query. Position comes from the
point cloud rather than `det-to-segment`, because that segmenter is wired to a
single detector and so cannot see blue or yellow.

    .venv/bin/python multi_pick_box.py --list              # every colour, all objects
    .venv/bin/python multi_pick_box.py --locate blue       # locate only, no motion
    .venv/bin/python multi_pick_box.py --dry-run blue      # full plan, no motion
    .venv/bin/python multi_pick_box.py blue                # execute

Filtering is in four stages, each needed:

  1. bbox size — the colour detectors return large background regions
     (blue's biggest hit was 409x719 at the image corner)
  2. 3D workspace bounds — rejects anything not physically on the table
  3. drop-box exclusion — ignores objects already sitting in the green box
  4. frame persistence — the detectors drop and invent blobs between frames
"""
import asyncio
import sys

import numpy as np

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from tutorial import connect, goto_saved_pose, _decode_depth
from pc_slow_pick import parse_point_cloud
from slow_pick import interpolate
from seg_pick import (
    APPROACH_CLEARANCE_MM,
    GRASP_OFFSET_X,
    GRASP_OFFSET_Y,
    APPLY_OFFSET,
    MIN_GRASP_Z,
    WORKING_GRASP_Z,
    collision_limits,
    creep_to,
)
from slow_pick import GRIPPER_NAME, gripper_pose_in_world, wait_until_stopped

# Colour -> the vision service configured for it. Each service detects exactly
# one colour, so the requested colour picks the detector.
DETECTORS = {
    "orange": "color-detector-orange",
    "yellow": "color-detector-yellow",
    "blue": "color-detector-blue",
}

# The green box is the drop target, not something to pick. It is taller than
# the objects (measured 97.3 mm above the table, stated 100 mm) and its
# detector returns one large blob rather than a compact cube, so it needs its
# own size window and its own locate path.
DROP_DETECTOR = "color-detector-green"
DROP_BOX_MIN_PX, DROP_BOX_MAX_PX = 80, 500

# Clearance above the green box's rim before opening the gripper. The carried
# object hangs below the gripper origin, so this must exceed the object height.
DROP_CLEARANCE_MM = 90.0

# Half-width of a square exclusion zone around the drop box centre. A colour
# detector can still fire on an object already sitting inside the box (seen
# through the open top), and picking that up again would be wrong — either
# it's already placed, or the arm would be reaching down inside the box walls.
# Not yet measured against the physical box; tune if real objects near the
# box's edge get excluded, or box-interior hits keep slipping through.
BOX_EXCLUSION_RADIUS_MM = 90.0

# Stage 1: plausible on-screen size for a table-top object at ~680 mm.
MIN_BOX_PX, MAX_BOX_PX = 25, 110

# Stage 2: the physical workspace. Anything outside this is background.
#
# The y bounds are deliberately tight. Carpet on the floor reads as blue and
# was passing a +/-400 mm window: false candidates clustered at y = +365..380
# while every real object measured sits between y = -80 and +51. The table
# itself does not extend past about +/-200 mm in y at the working distance.
WORKSPACE = dict(x=(200.0, 700.0), y=(-200.0, 200.0), z=(20.0, 150.0))

# Two candidate positions within this distance are treated as the same object
# across frames.
# Velocity/acceleration hints for the smooth path. Higher than the stepped
# mode's crawl, but still well short of the arm's default. Honoured only if the
# driver reads them; the motion service itself has no speed argument.
SMOOTH_EXTRA = {
    "max_vel_degs_per_sec": 25.0,
    "max_acc_degs_per_sec2": 25.0,
}

CLUSTER_RADIUS_MM = 40.0

TOP_FACE_FRACTION = 0.125
MIN_3D_POINTS = 30


def in_workspace(p):
    return (WORKSPACE['x'][0] < p.x < WORKSPACE['x'][1]
            and WORKSPACE['y'][0] < p.y < WORKSPACE['y'][1]
            and WORKSPACE['z'][0] < p.z < WORKSPACE['z'][1])


def in_drop_box(p, drop):
    if drop is None:
        return False
    return (abs(p.x - drop.x) < BOX_EXCLUSION_RADIUS_MM
            and abs(p.y - drop.y) < BOX_EXCLUSION_RADIUS_MM)


async def candidates(machine, cam, detector, drop=None):
    """On-table objects this detector sees, as (world Pose, w, h).

    Applies the size and workspace filters; persistence is handled by the
    caller sampling several frames. When `drop` (the green box's world pose)
    is given, candidates that fall inside it are dropped too — they're
    already-placed objects, not something to pick.
    """
    detections = await detector.get_detections_from_camera("cam")
    sized = [
        d for d in detections
        if MIN_BOX_PX <= (d.x_max - d.x_min) <= MAX_BOX_PX
        and MIN_BOX_PX <= (d.y_max - d.y_min) <= MAX_BOX_PX
    ]
    if not sized:
        return []

    raw, _ = await cam.get_point_cloud()
    cloud = parse_point_cloud(raw)

    out = []
    for d in sized:
        patch = cloud[d.y_min:d.y_max, d.x_min:d.x_max].ravel()
        valid = patch[np.isfinite(patch['z']) & (patch['z'] != 0)]
        if valid.size < MIN_3D_POINTS:
            continue
        depths = valid['z']
        take = max(20, int(valid.size * TOP_FACE_FRACTION))
        nearest = np.argsort(depths)[:take]
        pif = await machine.transform_pose(
            PoseInFrame(
                reference_frame="cam",
                pose=Pose(
                    x=float(valid['x'][nearest].mean() * 1000),
                    y=float(valid['y'][nearest].mean() * 1000),
                    z=float(valid['z'][nearest].mean() * 1000),
                    o_z=1,
                ),
            ),
            "world",
        )
        if in_workspace(pif.pose) and not in_drop_box(pif.pose, drop):
            out.append((pif.pose, d.x_max - d.x_min, d.y_max - d.y_min))
    return out


async def locate_colour(machine, cam, colour, frames=8, min_hits=4, drop=None):
    """Average the position of the single on-table object of `colour`.

    Returns None when the object is not seen consistently, or when more than
    one candidate survives — picking arbitrarily between two would be worse
    than refusing. `drop`, the green box's world pose, excludes objects
    already sitting inside it.
    """
    service = DETECTORS.get(colour)
    if service is None:
        print(f"  no detector configured for '{colour}'. "
              f"Known: {', '.join(sorted(DETECTORS))}")
        return None

    detector = VisionClient.from_robot(machine, service)

    # Collect every candidate from every frame, then keep the position that
    # recurs most often. Transient noise (carpet, reflections) moves around
    # between frames; a real object does not. This is more robust than
    # requiring exactly one candidate per frame, which a single noisy frame
    # would otherwise veto.
    seen = []
    for _ in range(frames):
        for pose, _w, _h in await candidates(machine, cam, detector, drop):
            seen.append(pose)
        await asyncio.sleep(0.12)

    if not seen:
        print(f"  '{colour}' not detected in any of {frames} frames")
        return None

    clusters = []
    for pose in seen:
        for c in clusters:
            if (abs(pose.x - c[0].x) < CLUSTER_RADIUS_MM
                    and abs(pose.y - c[0].y) < CLUSTER_RADIUS_MM):
                c.append(pose)
                break
        else:
            clusters.append([pose])
    clusters.sort(key=len, reverse=True)

    best = clusters[0]
    if len(clusters) > 1:
        others = ", ".join(
            f"({c[0].x:.0f},{c[0].y:.0f}) x{len(c)}" for c in clusters[1:4]
        )
        print(f"    ignoring {len(clusters) - 1} less consistent "
              f"cluster(s): {others}")

    if len(best) < min_hits:
        print(f"  '{colour}' best cluster seen in only {len(best)}/{frames} "
              f"frames (need {min_hits}) — too unstable to pick")
        return None

    xs = [p.x for p in best]
    ys = [p.y for p in best]
    zs = [p.z for p in best]

    def mean(v):
        return sum(v) / len(v)

    def std(v):
        mu = mean(v)
        return (sum((a - mu) ** 2 for a in v) / len(v)) ** 0.5

    srt = sorted(zs)
    n = len(srt)
    med_z = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2

    print(f"  '{colour}' via {service}: best cluster seen "
          f"{len(xs)}/{frames} frames")
    print(f"    x={mean(xs):.1f}+/-{std(xs):.1f}  "
          f"y={mean(ys):.1f}+/-{std(ys):.1f}  z={med_z:.1f}+/-{std(zs):.1f}")

    x, y = mean(xs), mean(ys)
    if APPLY_OFFSET:
        x += GRASP_OFFSET_X
        y += GRASP_OFFSET_Y
        print(f"    + offset ({GRASP_OFFSET_X:+.2f}, {GRASP_OFFSET_Y:+.2f}) "
              f"-> grasp at ({x:.1f}, {y:.1f})")

    grasp_z = max(med_z, MIN_GRASP_Z)
    if grasp_z > med_z:
        print(f"    top face z={med_z:.1f} is below the verified grasp height "
              f"— clamping to {MIN_GRASP_Z:.2f}")

    return Pose(x=x, y=y, z=grasp_z, o_z=1, theta=0)


async def move_smooth(machine, motion, arm, target, label, min_z,
                      dry_run=False, assume_at=None):
    """Move to `target` as one continuous motion instead of stepped waypoints.

    The straight-line path is still checked against the collision floor before
    anything is commanded — the motion service plans its own trajectory, which
    need not be a straight line, but a straight line dipping below the floor is
    a reliable sign the target itself is unsafe.
    """
    current = assume_at or await gripper_pose_in_world(machine)
    waypoints, distance = interpolate(current, target, step_mm=10.0)
    print(f"  {label}: {distance:.0f} mm, single continuous move")

    below = [w for w in waypoints if w.z < min_z]
    if below:
        lowest = min(w.z for w in below)
        raise RuntimeError(
            f"{label} path dips to z={lowest:.1f}, below the floor "
            f"z={min_z:.1f} — refusing to move."
        )

    if dry_run:
        print(f"    [dry-run] path clear of z={min_z:.1f} "
              f"(lowest {min(w.z for w in waypoints):.1f})")
        return target

    print(f"    -> ({target.x:.1f}, {target.y:.1f}, {target.z:.1f})", flush=True)
    await motion.move(
        component_name=GRIPPER_NAME,
        destination=PoseInFrame(reference_frame="world", pose=target),
        extra=SMOOTH_EXTRA,
    )
    await wait_until_stopped(arm)
    return target


async def locate_drop_box(machine, cam, frames=5):
    """Find the green drop box's top face in world coordinates.

    Separate from locate_colour: the box is large and its detector returns a
    single wide blob, so the compact-object size window does not apply.
    """
    detector = VisionClient.from_robot(machine, DROP_DETECTOR)
    xs, ys, zs = [], [], []

    for _ in range(frames):
        detections = [
            d for d in await detector.get_detections_from_camera("cam")
            if DROP_BOX_MIN_PX <= (d.x_max - d.x_min) <= DROP_BOX_MAX_PX
            and DROP_BOX_MIN_PX <= (d.y_max - d.y_min) <= DROP_BOX_MAX_PX
        ]
        if not detections:
            await asyncio.sleep(0.12)
            continue

        d = max(detections,
                key=lambda x: (x.x_max - x.x_min) * (x.y_max - x.y_min))
        # Depth map, not point cloud: the cloud is the same data already
        # deprojected but is 14.7 MB / ~1250 ms against 1.8 MB / ~120 ms.
        # Five frames of it made this a ~6 s stall.
        images, _ = await cam.get_images(filter_source_names=["depth"])
        depth = _decode_depth(images[0].data)
        intr = (await cam.get_properties()).intrinsic_parameters
        patch = depth[d.y_min:d.y_max, d.x_min:d.x_max]
        valid = patch[patch > 0]
        if valid.size < 100:
            await asyncio.sleep(0.12)
            continue

        cutoff = np.percentile(valid, 12.5)
        mask = (patch > 0) & (patch <= cutoff)
        rows, cols = np.nonzero(mask)
        if cols.size == 0:
            await asyncio.sleep(0.12)
            continue
        z_mm = float(patch[mask].mean())
        px = float(cols.mean()) + d.x_min
        py = float(rows.mean()) + d.y_min
        pif = await machine.transform_pose(
            PoseInFrame(
                reference_frame="cam",
                pose=Pose(
                    x=(px - intr.center_x_px) * z_mm / intr.focal_x_px,
                    y=(py - intr.center_y_px) * z_mm / intr.focal_y_px,
                    z=z_mm,
                    o_z=1,
                ),
            ),
            "world",
        )
        xs.append(pif.pose.x)
        ys.append(pif.pose.y)
        zs.append(pif.pose.z)
        await asyncio.sleep(0.12)

    if not xs:
        print("  green drop box not found")
        return None

    def mean(v):
        return sum(v) / len(v)

    print(f"  drop box: ({mean(xs):.1f}, {mean(ys):.1f}, {mean(zs):.1f}) "
          f"from {len(xs)} frames")
    return Pose(x=mean(xs), y=mean(ys), z=mean(zs), o_z=1, theta=0)


async def main(argv):
    list_only = '--list' in argv
    locate_only = '--locate' in argv
    dry_run = '--dry-run' in argv
    stepped = '--slow' in argv
    no_drop = '--no-drop' in argv
    colour = next((a for a in argv[1:] if not a.startswith('-')), None)
    move = creep_to if stepped else move_smooth

    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")
        cam = Camera.from_robot(machine, "cam")
        motion = MotionClient.from_robot(machine, "builtin")

        min_z, why = await collision_limits(machine)
        print(f"grasp floor: z={min_z:.1f}  ({why})")
        print()

        if list_only:
            print("on-table objects, by colour (1 frame each):")
            for name, service in sorted(DETECTORS.items()):
                detector = VisionClient.from_robot(machine, service)
                found = await candidates(machine, cam, detector)
                if not found:
                    print(f"  {name:8s} none")
                for pose, w, h in found:
                    print(f"  {name:8s} world=({pose.x:7.1f}, {pose.y:7.1f}, "
                          f"{pose.z:6.1f})  {w}x{h}px")
            return

        if colour is None:
            print(f"usage: multi_pick_box.py [--list|--locate|--dry-run] "
                  f"[--slow] [--no-drop] <{('|'.join(sorted(DETECTORS)))}>")
            print("  --slow     use the old stepped crawl instead of smooth motion")
            print("  --no-drop  lift and hold instead of dropping in the green box")
            return

        if not (dry_run or locate_only):
            print("moving to top-pose to look...")
            await goto_saved_pose(machine, "top-pose")
            await wait_until_stopped(arm)
            await asyncio.sleep(0.5)
        else:
            print("(looking from the arm's current pose — not moving it)")

        # Find the drop box first, while the arm is still at top-pose and can
        # see the whole table (after the pick it is down at the object and
        # the box may be out of frame) — and so its position is known before
        # locating the object, to exclude anything already sitting in it.
        drop = None
        if not no_drop:
            print("locating drop box...")
            drop = await locate_drop_box(machine, cam)
            if drop is None:
                print("  no drop box — will lift and hold instead of dropping")

        print(f"locating '{colour}'...")
        target = await locate_colour(machine, cam, colour, drop=drop)
        if target is None:
            return

        down = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
        above = Pose(x=target.x, y=target.y,
                     z=target.z + APPROACH_CLEARANCE_MM, **down)
        grasp = Pose(x=target.x, y=target.y, z=target.z, **down)

        print()
        print(f"  grasp z = {grasp.z:7.1f}")
        print(f"  hover z = {above.z:7.1f}")

        if grasp.z < min_z:
            print(f"\n  REFUSED: grasp z={grasp.z:.1f} is below the floor "
                  f"z={min_z:.1f}.")
            return

        if locate_only:
            print("\nlocate-only: stopping here.")
            return

        print("\nopening gripper...")
        if not dry_run:
            await gripper.open()

        print("\n[1/3] hover above object")
        at = await move(machine, motion, arm, above, "approach", min_z,
                            dry_run)

        print("\n[2/3] descend onto object")
        at = await move(machine, motion, arm, grasp, "descend", min_z,
                            dry_run, assume_at=at if dry_run else None)

        print("\nclosing gripper...")
        if not dry_run:
            grabbed = await gripper.grab()
            print("  grabbed." if grabbed
                  else "  WARNING: grab() reported nothing grasped.")

        print("\n[3/4] lift clear" if drop else "\n[3/3] lift clear")
        at = await move(machine, motion, arm, above, "lift", min_z,
                        dry_run, assume_at=at if dry_run else None)

        if drop is None:
            print("\ndone — object lifted, held above the pick point.")
            return

        # Carry to the drop box and release above its rim. The object hangs
        # below the gripper, so release high enough that it clears the wall.
        over_box = Pose(x=drop.x, y=drop.y,
                        z=drop.z + DROP_CLEARANCE_MM, **down)
        print("\n[4/4] carry to drop box")
        print(f"  releasing at z={over_box.z:.1f} "
              f"({DROP_CLEARANCE_MM:.0f} mm above the box rim at {drop.z:.1f})")
        await move(machine, motion, arm, over_box, "carry", min_z,
                   dry_run, assume_at=at if dry_run else None)

        print("\nreleasing...")
        if not dry_run:
            await gripper.open()
            await asyncio.sleep(0.5)

        print("\ndone — object dropped in the green box.")


if __name__ == '__main__':
    asyncio.run(main(sys.argv))
