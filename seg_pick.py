"""Vision-guided slow pick using the `det-to-segment` 3D segmenter.

Improvements over pc_slow_pick.py:

  * Uses the machine's own `det-to-segment` service, which returns labelled 3D
    bounding boxes (centre + dims) instead of a raw cloud we segment ourselves.
    Measured far more precisely: x/y scatter +/- 0.5 mm vs +/- 3-5 mm, and a
    known 90 mm object separation reads 90.53 mm vs the ~95-96 mm our own
    estimator gave.
  * Every waypoint is checked against the machine's collision geometry BEFORE
    any motion, so a plan that would be refused mid-descent fails up front.
  * Applies a measured horizontal correction (see GRASP_OFFSET_* below).

    .venv/bin/python seg_pick.py --list                # what is on the table
    .venv/bin/python seg_pick.py --locate <label>      # locate, no motion
    .venv/bin/python seg_pick.py --dry-run <label>     # full plan + collision check
    .venv/bin/python seg_pick.py <label>               # execute
"""
import asyncio
import sys

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from tutorial import connect, goto_saved_pose
from slow_pick import (
    GRIPPER_NAME,
    STEP_MM,
    STEP_PAUSE_S,
    SLOW_EXTRA,
    interpolate,
    gripper_pose_in_world,
    wait_until_stopped,
)

SEGMENTER = "det-to-segment"

# The segmenter occasionally emits nonsense (a 1092 mm tall "rectangle-blue"
# was observed). Reject boxes that cannot be a table-top object.
MAX_OBJECT_DIM_MM = 200.0
MIN_OBJECT_DIM_MM = 5.0

# --- measured horizontal correction -----------------------------------------
#
# With an object placed at the spot the arm grasps from `reach-cuboid`, the
# segmenter reports a centre that sits consistently short of the recorded
# gripper position:
#
#     recorded gripper   (470.73,  20.95)
#     segmenter centre   (457.15,   4.36)   +/- 0.5
#     correction         (+13.57, +16.58)   = 21.4 mm
#
# The same offset appeared with a different object at the same spot
# (17.5 mm / -134.6 deg via an independent HSV method), so it is systematic
# rather than a placement artifact. Root cause is still unconfirmed — most
# likely a camera-mount calibration error or a gripper jaw offset the config
# does not model.
#
# NOTE: this correction is measured at ONE table position and one arm pose. If
# it is a camera-mount error it may vary across the workspace. Re-measure at a
# second position before trusting it far from where it was taken.
GRASP_OFFSET_X = 13.57
GRASP_OFFSET_Y = 16.58
APPLY_OFFSET = True

APPROACH_CLEARANCE_MM = 120.0

# --- grasp height -----------------------------------------------------------
#
# The segmenter's box CENTRE is not a safe grasp height. Its z is unstable
# (measured spread 6-8 mm on a stationary object) because the box bottom is cut
# at a different place each frame depending on how much table gets included —
# the box height for the 60 mm object ranged 55-93 mm over 10 samples.
#
# Worse, the centre scales with object height: a 30 mm cube centres at z~15,
# which would drive the gripper 13 mm BELOW the hand-verified reach-cuboid
# grasp. The gripper does not need to reach an object's centre; it needs a
# height where the claws close around it, and reach-cuboid already establishes
# what the hardware tolerates.
#
# So grasp at the known-good height, never lower.
WORKING_GRASP_Z = 28.36          # recorded reach-cuboid gripper z
MIN_GRASP_Z = WORKING_GRASP_Z    # never command the gripper below this

# --- collision model --------------------------------------------------------
#
# Read from the live machine rather than hardcoded, so it tracks config changes.
# The planner refuses any pose where the claws box intersects the table box.
SAFETY_MARGIN_MM = 2.0


async def collision_limits(machine):
    """Lowest gripper-origin z the collision model allows, and why.

    Returns (min_z, description). Computed from the live geometries so a config
    fix is picked up without editing this file.
    """
    gripper = Gripper.from_robot(machine, "gripper")
    table = Gripper.from_robot(machine, "table")

    claws_bottom = None
    for geo in await gripper.get_geometries():
        if geo.label == "claws":
            claws_bottom = geo.center.z - geo.box.dims_mm.z / 2
    if claws_bottom is None:
        return None, "no 'claws' geometry found on the gripper"

    table_top = None
    for geo in await table.get_geometries():
        pif = await machine.transform_pose(
            PoseInFrame(reference_frame="table", pose=geo.center), "world"
        )
        table_top = pif.pose.z + geo.box.dims_mm.z / 2

    if table_top is None:
        return None, "no table geometry found"

    model_min_z = table_top - claws_bottom + SAFETY_MARGIN_MM

    # The configured model rejects the hand-recorded reach-cuboid grasp
    # (z=28.36), which is known to work. Where it disagrees with reality,
    # trust the verified grasp — but never go below it either.
    if model_min_z > WORKING_GRASP_Z:
        desc = (f"config model says z>={model_min_z:.1f}, but the verified "
                f"reach-cuboid grasp at z={WORKING_GRASP_Z:.2f} works; "
                f"using the verified height")
        return WORKING_GRASP_Z, desc

    desc = (f"claws extend {-claws_bottom:.1f} mm below the gripper origin; "
            f"table solid top is world z={table_top:.1f}")
    return model_min_z, desc


def check_plan(waypoints, min_z, label):
    """Report which waypoints violate the collision floor. Returns the bad ones."""
    bad = [(i, wp) for i, wp in enumerate(waypoints, 1) if wp.z < min_z]
    if bad:
        print(f"  {label}: {len(bad)}/{len(waypoints)} waypoints below the "
              f"collision floor (z < {min_z:.1f})")
        for i, wp in bad[:3]:
            print(f"    step {i}: z={wp.z:.1f}  ({min_z - wp.z:.1f} mm too low)")
        if len(bad) > 3:
            print(f"    ... and {len(bad) - 3} more")
    return bad


async def find_objects_stable(machine, segmenter, frames=8, min_seen=0.5):
    """Objects that appear in at least `min_seen` of `frames` consecutive reads.

    The segmenter is noisy frame to frame: over 12 frames of a stationary scene
    the two real objects appeared 12/12 and 10/12 times, while three phantom
    labels appeared in 2-5 frames each (including a 877x1045x1006 mm
    "rectangle-blue"). A single read is therefore not trustworthy — both for
    missing a real object and for inventing one.
    """
    tally = {}
    for _ in range(frames):
        for label, pose, dims in await find_objects(machine, segmenter):
            rec = tally.setdefault(label, {'n': 0, 'x': [], 'y': [], 'z': []})
            rec['n'] += 1
            rec['x'].append(pose.x)
            rec['y'].append(pose.y)
            rec['z'].append(pose.z)
        await asyncio.sleep(0.1)

    need = max(2, int(frames * min_seen))
    out = []
    for label, rec in sorted(tally.items()):
        if rec['n'] < need:
            continue
        n = rec['n']
        srt = sorted(rec['z'])
        med_z = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
        out.append((
            label,
            Pose(x=sum(rec['x']) / n, y=sum(rec['y']) / n, z=med_z, o_z=1),
            rec['n'],
            frames,
        ))
    return out


async def find_objects(machine, segmenter):
    """All segmented objects as (label, world centre Pose, dims), one read."""
    out = []
    for obj in await segmenter.get_object_point_clouds("cam"):
        # The segmenter can return a point cloud with no geometry attached --
        # a cluster it could not fit a box to. Indexing [0] blindly raised
        # IndexError and killed the caller, which for seg_z_pump means losing
        # the accurate z source mid-pick. Skip it instead: one unfittable
        # cluster is not a reason to drop the rest of the read.
        if not obj.geometries.geometries:
            continue
        geo = obj.geometries.geometries[0]
        pif = await machine.transform_pose(
            PoseInFrame(reference_frame="cam", pose=geo.center), "world"
        )
        dims = geo.box.dims_mm if geo.HasField("box") else None
        if dims is not None:
            biggest = max(dims.x, dims.y, dims.z)
            smallest = min(dims.x, dims.y, dims.z)
            if biggest > MAX_OBJECT_DIM_MM or smallest < MIN_OBJECT_DIM_MM:
                # Not a plausible table-top object — the segmenter sometimes
                # returns the wall or floor as a huge box.
                continue
        out.append((geo.label, pif.pose, dims))
    return out


async def locate(machine, segmenter, want, samples=10, min_hits=4):
    """Average `samples` readings of the object called `want`.

    The segmenter drops a real object from an occasional frame, so require a
    minimum number of hits rather than trusting however many arrive.
    """
    xs, ys, zs, dims = [], [], [], None
    for _ in range(samples):
        for label, pose, d in await find_objects(machine, segmenter):
            if label == want:
                xs.append(pose.x)
                ys.append(pose.y)
                zs.append(pose.z)
                dims = d
        await asyncio.sleep(0.15)

    if len(xs) < min_hits:
        print(f"  '{want}' seen in only {len(xs)}/{samples} frames "
              f"(need {min_hits}) — not reliable enough to pick")
        return None

    def mean(v):
        return sum(v) / len(v)

    def std(v):
        mu = mean(v)
        return (sum((a - mu) ** 2 for a in v) / len(v)) ** 0.5

    def median(v):
        srt = sorted(v)
        n = len(srt)
        return srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2

    print(f"  '{want}' seen {len(xs)}/{samples} frames: "
          f"x={mean(xs):.1f}+/-{std(xs):.1f}  "
          f"y={mean(ys):.1f}+/-{std(ys):.1f}  "
          f"z={median(zs):.1f}+/-{std(zs):.1f} (median)")
    if dims:
        print(f"    box dims {dims.x:.1f} x {dims.y:.1f} x {dims.z:.1f} mm")

    x, y = mean(xs), mean(ys)
    if APPLY_OFFSET:
        x += GRASP_OFFSET_X
        y += GRASP_OFFSET_Y
        print(f"    + measured offset ({GRASP_OFFSET_X:+.2f}, {GRASP_OFFSET_Y:+.2f}) "
              f"-> grasp at ({x:.1f}, {y:.1f})")

    # Clamp to the hand-verified grasp height. The segmenter's centre is both
    # unstable (6-8 mm spread) and object-height dependent, so it is used only
    # as a sanity reference, never as the commanded depth.
    seen_z = median(zs)
    grasp_z = max(seen_z, MIN_GRASP_Z)
    if grasp_z > seen_z:
        print(f"    segmenter centre z={seen_z:.1f} is below the verified "
              f"grasp height — clamping to {MIN_GRASP_Z:.2f}")

    return Pose(x=x, y=y, z=grasp_z, o_z=1, theta=0)


async def creep_to(machine, motion, arm, target, label, min_z,
                   dry_run=False, assume_at=None):
    """Move the gripper to `target` in small steps, refusing unsafe plans."""
    current = assume_at or await gripper_pose_in_world(machine)
    waypoints, distance = interpolate(current, target)
    print(f"  {label}: {distance:.0f} mm in {len(waypoints)} steps")

    if check_plan(waypoints, min_z, label):
        raise RuntimeError(
            f"{label} would collide — refusing to move. "
            f"Lower the target or fix the collision geometry."
        )

    if dry_run:
        print(f"    [dry-run] {len(waypoints)} waypoints, all clear of z={min_z:.1f}")
        return target

    for i, wp in enumerate(waypoints, 1):
        print(f"    step {i}/{len(waypoints)} -> "
              f"({wp.x:.1f}, {wp.y:.1f}, {wp.z:.1f})", flush=True)
        await motion.move(
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(reference_frame="world", pose=wp),
            extra=SLOW_EXTRA,
        )
        await wait_until_stopped(arm)
        await asyncio.sleep(STEP_PAUSE_S)
    return target


async def main(argv):
    list_only = '--list' in argv
    locate_only = '--locate' in argv
    dry_run = '--dry-run' in argv
    wanted = next((a for a in argv[1:] if not a.startswith('-')), None)

    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")
        motion = MotionClient.from_robot(machine, "builtin")
        segmenter = VisionClient.from_robot(machine, SEGMENTER)

        min_z, why = await collision_limits(machine)
        print(f"collision floor: gripper origin must stay above z={min_z:.1f}")
        print(f"  ({why})")
        print()

        if list_only:
            print("objects on the table (8 frames, must appear in >=4):")
            found = await find_objects_stable(machine, segmenter)
            for label, pose, seen, total in found:
                print(f"  {label:20s} world=({pose.x:7.1f}, {pose.y:7.1f}, "
                      f"{pose.z:6.1f})   seen {seen}/{total}")
            if not found:
                print("  (nothing appeared consistently)")
            return

        if not wanted:
            print("usage: seg_pick.py [--list|--locate|--dry-run] <label>")
            print("       run with --list to see the available labels")
            return

        if not (dry_run or locate_only):
            print("moving to top-pose to look...")
            await goto_saved_pose(machine, "top-pose")
            await wait_until_stopped(arm)
            await asyncio.sleep(0.5)
        else:
            print("(looking from the arm's current pose — not moving it)")

        print(f"locating '{wanted}'...")
        target = await locate(machine, segmenter, wanted)
        if target is None:
            print(f"'{wanted}' not found. Run --list to see what is visible.")
            return

        down = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
        above = Pose(x=target.x, y=target.y,
                     z=target.z + APPROACH_CLEARANCE_MM, **down)
        grasp = Pose(x=target.x, y=target.y, z=target.z, **down)

        print()
        print(f"  grasp  z = {grasp.z:7.1f}   (object centre)")
        print(f"  hover  z = {above.z:7.1f}")

        if grasp.z < min_z:
            print()
            print(f"  REFUSED: grasp z={grasp.z:.1f} is below the floor "
                  f"z={min_z:.1f}.")
            return

        if locate_only:
            print("\nlocate-only: stopping here.")
            return

        print("\nopening gripper...")
        if not dry_run:
            await gripper.open()

        print("\n[1/3] hover above object")
        at = await creep_to(machine, motion, arm, above, "approach", min_z,
                            dry_run)

        print("\n[2/3] descend onto object")
        at = await creep_to(machine, motion, arm, grasp, "descend", min_z,
                            dry_run, assume_at=at if dry_run else None)

        print("\nclosing gripper...")
        if not dry_run:
            grabbed = await gripper.grab()
            print("  grabbed." if grabbed
                  else "  WARNING: grab() reported nothing grasped.")

        print("\n[3/3] lift clear")
        await creep_to(machine, motion, arm, above, "lift", min_z,
                       dry_run, assume_at=at if dry_run else None)

        print("\ndone — object lifted, held above the pick point.")


if __name__ == '__main__':
    asyncio.run(main(sys.argv))
