"""Slowly approach the detected orange object and pick it up.

Separate from tutorial.py so the fast path stays untouched. Movement is made
slow in two independent ways, because `extra` speed hints are driver-specific
and this arm's driver may ignore them:

  1. The cartesian approach is split into many small waypoints (STEP_MM apart)
     with a pause between each, so the arm creeps rather than sweeps.
  2. Velocity/acceleration hints are passed via `extra` as well, in case the
     driver does honour them. Harmless if ignored.

Every motion is preceded by a printed target and gated on the arm reporting
it has stopped, so you can follow along and hit Ctrl-C between steps.

    .venv/bin/python slow_pick.py            # full run
    .venv/bin/python slow_pick.py --dry-run  # plan and print, never move
"""
import asyncio
import sys

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.utils import dict_to_struct
from viam.proto.service.motion import MoveRequest
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from tutorial import (
    APPROACH_CLEARANCE_MM,
    GRASP_DEPTH_MM,
    connect,
    goto_saved_pose,
    locate_target,
)

# The motion service takes component_name as a STRING, not a ResourceName,
# despite what Gripper.get_resource_name() returns. Passing the object raises
# "TypeError: bad argument type for built-in operation" inside MoveRequest.
GRIPPER_NAME = "gripper"

# --- speed knobs ------------------------------------------------------------

# Distance between intermediate waypoints on the cartesian approach. Smaller
# = slower and smoother, at the cost of more round-trips to the motion service.
STEP_MM = 15.0

# Pause after each waypoint, seconds. This is the main slowness dial.
STEP_PAUSE_S = 0.6

# Passed through `extra`; honoured only if the arm driver reads them.
SLOW_EXTRA = {
    "max_vel_degs_per_sec": 10.0,
    "max_acc_degs_per_sec2": 10.0,
}

# How long to wait for the arm to report it has stopped before moving on.
SETTLE_TIMEOUT_S = 30.0


async def wait_until_stopped(arm, timeout=SETTLE_TIMEOUT_S):
    """Block until the arm reports it is no longer moving."""
    waited = 0.0
    while waited < timeout:
        if not await arm.is_moving():
            return True
        await asyncio.sleep(0.1)
        waited += 0.1
    print("    WARNING: arm still moving after settle timeout")
    return False


def _lerp(a, b, t):
    return a + (b - a) * t


def interpolate(start, end, step_mm=STEP_MM):
    """Waypoints from `start` to `end`, no more than step_mm apart.

    Orientation is taken from `end` throughout: these approaches hold a fixed
    tool orientation, so there is nothing to interpolate.
    """
    dx, dy, dz = end.x - start.x, end.y - start.y, end.z - start.z
    distance = (dx * dx + dy * dy + dz * dz) ** 0.5
    steps = max(1, int(distance / step_mm))
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        points.append(Pose(
            x=_lerp(start.x, end.x, t),
            y=_lerp(start.y, end.y, t),
            z=_lerp(start.z, end.z, t),
            o_x=end.o_x, o_y=end.o_y, o_z=end.o_z, theta=end.theta,
        ))
    return points, distance


async def gripper_pose_in_world(machine):
    """Where the gripper origin currently is, in world coordinates.

    NOT arm.get_end_position() — that is the wrist, a fixed 150 mm behind the
    gripper. Waypoints are commanded in the gripper frame, so the start of the
    interpolation has to be measured in the same frame or the first step jumps.
    """
    pif = await machine.transform_pose(
        PoseInFrame(reference_frame="gripper", pose=Pose(o_z=1)), "world"
    )
    return pif.pose


async def creep_to(machine, motion, arm, target, label, dry_run=False,
                   assume_at=None):
    """Move the gripper to `target` in small steps, settling after each.

    `assume_at` lets a dry run chain legs together: without it every leg would
    measure from the arm's real position and report the wrong distance.
    Returns the pose the gripper ends at.
    """
    current = assume_at or await gripper_pose_in_world(machine)
    waypoints, distance = interpolate(current, target)
    print(f"  {label}: {distance:.0f} mm in {len(waypoints)} steps "
          f"(~{STEP_MM:.0f} mm each, {STEP_PAUSE_S}s pause)")
    print(f"    from ({current.x:.1f}, {current.y:.1f}, {current.z:.1f})")
    print(f"      to ({target.x:.1f}, {target.y:.1f}, {target.z:.1f})")

    if dry_run:
        # Build the request we would have sent for the first waypoint, without
        # sending it. A dry run that skips this misses argument-type errors
        # that only surface on a real move.
        MoveRequest(
            name="builtin",
            component_name=GRIPPER_NAME,
            destination=PoseInFrame(reference_frame="world", pose=waypoints[0]),
            extra=dict_to_struct(SLOW_EXTRA),
        )

        # Only show the first and last few — a 40-step leg is not worth printing
        # in full, and the endpoints are what need checking.
        for i, wp in enumerate(waypoints, 1):
            if i <= 3 or i > len(waypoints) - 2:
                print(f"    [dry-run] step {i}/{len(waypoints)} -> "
                      f"({wp.x:.1f}, {wp.y:.1f}, {wp.z:.1f})")
            elif i == 4:
                print(f"    [dry-run] ... {len(waypoints) - 5} more steps ...")
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


async def main(dry_run=False):
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")
        cam = Camera.from_robot(machine, "cam")
        detector = VisionClient.from_robot(machine, "color-detector")
        motion = MotionClient.from_robot(machine, "builtin")

        if dry_run:
            print("=== DRY RUN — the arm will not be commanded ===\n")

        # Survey from overhead. The camera rides on the arm, so the detection
        # is only valid for the pose it was captured at — hence locating here
        # and immediately transforming to world coordinates.
        print("moving to top-pose to look...")
        if not dry_run:
            await goto_saved_pose(machine, "top-pose")
            await wait_until_stopped(arm)
            await asyncio.sleep(0.5)

        print("locating target...")
        target = await locate_target(machine, cam, detector)
        if target is None:
            print("nothing to pick — stopping here, arm left at top-pose.")
            return

        down = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
        above = Pose(x=target.x, y=target.y,
                     z=target.z + APPROACH_CLEARANCE_MM, **down)
        grasp = Pose(x=target.x, y=target.y,
                     z=target.z - GRASP_DEPTH_MM, **down)

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

        print("\ndone — object lifted. Arm left holding above the pick point.")


if __name__ == '__main__':
    asyncio.run(main(dry_run='--dry-run' in sys.argv))
