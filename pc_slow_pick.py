"""Slow pick using the POINT CLOUD to find the object's top face.

Differs from slow_pick.py in how the target is located:

  slow_pick.py     deprojects the detection pixel through the depth image.
                   The depth image does not resolve a 60 mm object here — it
                   returns the TABLE (verified: 1 mm step across the bbox).

  this script      segments the point cloud by colour in 3D and takes the
                   nearest points as the object's top face (verified: 37 mm
                   step, and the top face lands 59.9 mm above the depth-image
                   table reading, against a measured 60 mm object).

Motion is unchanged: small waypoints with pauses, each gated on the arm
reporting it has stopped. The motion helpers are imported from slow_pick.py.

    .venv/bin/python pc_slow_pick.py --dry-run   # locate + plan, never moves
    .venv/bin/python pc_slow_pick.py --locate    # locate only, print and exit
    .venv/bin/python pc_slow_pick.py             # full slow pick

The point cloud is ~14.7 MB per frame, so it is fetched once per locate, never
in a loop.
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

from tutorial import TARGET_LABEL, connect, goto_saved_pose
from slow_pick import (
    SLOW_EXTRA,
    creep_to,
    gripper_pose_in_world,
    wait_until_stopped,
)

# --- point cloud segmentation ----------------------------------------------

PC_WIDTH, PC_HEIGHT = 1280, 720
PC_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4')])

# Orange mask applied to the cloud's own RGB. Looser than a strict hue window
# because it only has to separate the object from the table inside a bbox the
# detector has already localised.
RGB_MIN_R = 110
RGB_MIN_R_MINUS_G = 35
RGB_MIN_R_MINUS_B = 35

# Fraction of the orange points, nearest-first by depth, taken as the top face.
# An eighth is enough to average down noise while excluding table points that
# bleed in around the object's edges.
TOP_FACE_FRACTION = 0.125
MIN_ORANGE_POINTS = 100

# Measured with calipers. Used only to place the grasp relative to the top face.
OBJECT_HEIGHT_MM = 60.0

# Grasp at the object's mid-height. The manually recorded reach-cuboid pose puts
# the gripper 53% of the way down from the top, so half is the right target.
GRASP_BELOW_TOP_MM = OBJECT_HEIGHT_MM / 2

# Hover height above the top face before descending.
APPROACH_CLEARANCE_MM = 120.0


def parse_point_cloud(raw):
    """Parse a pointcloud/pcd buffer into a (720, 1280) structured array.

    The cloud is one point per pixel in row-major order, so it can be indexed
    with the same coordinates as the colour image. Coordinates are in METRES
    in the camera frame.
    """
    header_start = raw.index(b'DATA ')
    after = raw[header_start:]
    body = after[after.index(b'\n') + 1:]
    count = PC_WIDTH * PC_HEIGHT
    points = np.frombuffer(body[:count * PC_DTYPE.itemsize], dtype=PC_DTYPE)
    return points.reshape(PC_HEIGHT, PC_WIDTH)


def orange_mask(points):
    """Boolean mask of orange-ish points, from the cloud's own RGB channel."""
    rgb = points['rgb']
    r = ((rgb >> 16) & 255).astype(np.int16)
    g = ((rgb >> 8) & 255).astype(np.int16)
    b = (rgb & 255).astype(np.int16)
    return (
        (r > RGB_MIN_R)
        & (r - g > RGB_MIN_R_MINUS_G)
        & (r - b > RGB_MIN_R_MINUS_B)
    )


async def locate_top_face(machine, cam, detector, verbose=True):
    """Find the orange object's TOP FACE centre as a world-frame Pose.

    Returns None if nothing is detected or too few 3D points survive masking.
    """
    detections = await detector.get_detections_from_camera("cam")
    matches = [d for d in detections if d.class_name == TARGET_LABEL]
    if not matches:
        seen = {d.class_name for d in detections}
        print(f"  no '{TARGET_LABEL}' detected"
              + (f" (saw: {', '.join(sorted(seen))})" if seen else ""))
        return None

    best = max(matches, key=lambda d: (d.confidence,
                                       (d.x_max - d.x_min) * (d.y_max - d.y_min)))

    raw, _ = await cam.get_point_cloud()
    cloud = parse_point_cloud(raw)

    patch = cloud[best.y_min:best.y_max, best.x_min:best.x_max].ravel()
    valid = patch[np.isfinite(patch['z']) & (patch['z'] != 0)]
    if valid.size == 0:
        print("  detection has no valid 3D points")
        return None

    obj = valid[orange_mask(valid)]
    if obj.size < MIN_ORANGE_POINTS:
        print(f"  only {obj.size} orange 3D points "
              f"(need {MIN_ORANGE_POINTS}) — cannot locate reliably")
        return None

    # Nearest points by depth = the top face. Table points bleeding in around
    # the object's edges sit further away and are excluded by this cut.
    depths = obj['z']
    take = max(50, int(obj.size * TOP_FACE_FRACTION))
    nearest = np.argsort(depths)[:take]

    x_m = float(obj['x'][nearest].mean())
    y_m = float(obj['y'][nearest].mean())
    z_m = float(obj['z'][nearest].mean())

    in_world = await machine.transform_pose(
        PoseInFrame(
            reference_frame="cam",
            pose=Pose(x=x_m * 1000, y=y_m * 1000, z=z_m * 1000, o_z=1, theta=0),
        ),
        "world",
    )
    top = in_world.pose

    if verbose:
        spread = (depths.max() - depths.min()) * 1000
        print(f"  '{TARGET_LABEL}' conf={best.confidence:.2f}  "
              f"{obj.size} orange pts, z spread {spread:.1f} mm "
              f"(object is {OBJECT_HEIGHT_MM:.0f} mm)")
        print(f"  top face -> world ({top.x:.1f}, {top.y:.1f}, {top.z:.1f})")
        if spread < OBJECT_HEIGHT_MM * 0.5:
            print(f"  WARNING: z spread {spread:.1f} mm is much less than the "
                  f"object height — the cloud may not be resolving it")
    return top


async def sample_top_face(machine, cam, detector, n=5):
    """Average several locate_top_face readings; report the scatter.

    Per-sample scatter on the cloud method is a few mm, so averaging is worth
    the extra frames before committing the arm to a position.
    """
    xs, ys, zs = [], [], []
    for i in range(n):
        p = await locate_top_face(machine, cam, detector, verbose=(i == 0))
        if p is not None:
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)
        await asyncio.sleep(0.15)

    if not xs:
        return None

    def mean(v):
        return sum(v) / len(v)

    def std(v):
        m = mean(v)
        return (sum((a - m) ** 2 for a in v) / len(v)) ** 0.5

    print(f"  {len(xs)}/{n} samples: "
          f"x={mean(xs):.1f}+/-{std(xs):.1f}  "
          f"y={mean(ys):.1f}+/-{std(ys):.1f}  "
          f"z={mean(zs):.1f}+/-{std(zs):.1f}")
    return Pose(x=mean(xs), y=mean(ys), z=mean(zs), o_z=1, theta=0)


async def main(dry_run=False, locate_only=False):
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")
        cam = Camera.from_robot(machine, "cam")
        detector = VisionClient.from_robot(machine, "color-detector")
        motion = MotionClient.from_robot(machine, "builtin")

        if dry_run:
            print("=== DRY RUN — the arm will not be commanded ===\n")
        if locate_only:
            print("=== LOCATE ONLY — the arm will not be commanded ===\n")

        # The camera is on the arm, so the cloud is only valid for the pose it
        # was captured at. Settle at top-pose first, then measure.
        if not (dry_run or locate_only):
            print("moving to top-pose to look...")
            await goto_saved_pose(machine, "top-pose")
            await wait_until_stopped(arm)
            await asyncio.sleep(0.5)
        else:
            print("(using the arm's current pose to look — not moving it)")

        print("locating target via point cloud...")
        top = await sample_top_face(machine, cam, detector)
        if top is None:
            print("nothing to pick — stopping.")
            return

        down = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
        above = Pose(x=top.x, y=top.y,
                     z=top.z + APPROACH_CLEARANCE_MM, **down)
        grasp = Pose(x=top.x, y=top.y,
                     z=top.z - GRASP_BELOW_TOP_MM, **down)

        print()
        print(f"  object top face   z = {top.z:7.1f}")
        print(f"  grasp target      z = {grasp.z:7.1f}  "
              f"({GRASP_BELOW_TOP_MM:.0f} mm below the top face)")
        print(f"  hover             z = {above.z:7.1f}")

        if locate_only:
            print("\nlocate-only: stopping here.")
            return

        print("\nopening gripper...")
        if not dry_run:
            await gripper.open()

        print("\n[1/3] hover above object")
        at = await creep_to(machine, motion, arm, above, "approach", dry_run)

        print("\n[2/3] descend onto object")
        at = await creep_to(machine, motion, arm, grasp, "descend", dry_run,
                            assume_at=at if dry_run else None)

        print("\nclosing gripper...")
        if not dry_run:
            grabbed = await gripper.grab()
            print("  grabbed." if grabbed
                  else "  WARNING: grab() reported nothing grasped.")

        print("\n[3/3] lift clear")
        await creep_to(machine, motion, arm, above, "lift", dry_run,
                       assume_at=at if dry_run else None)

        print("\ndone — object lifted, held above the pick point.")


if __name__ == '__main__':
    asyncio.run(main(
        dry_run='--dry-run' in sys.argv,
        locate_only='--locate' in sys.argv,
    ))
