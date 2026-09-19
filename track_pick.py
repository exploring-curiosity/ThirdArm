"""Acquire an object in one frame, then track it continuously while approaching.

Two phases:

  ACQUIRE  one detection frame at top-pose (~300 ms) establishes which object
           we are going for and roughly where it is. No 8-frame averaging —
           that cost 3 s and is unnecessary when tracking will refine it.

  TRACK    live3d's loop runs continuously from then on. The target position is
           refined on every frame, so the approach corrects itself as the arm
           moves and the object can be moved by hand mid-approach.

The camera is wrist-mounted, so the view changes constantly during the approach.
The tracker re-detects each frame rather than dead-reckoning, and the motion
target is re-issued whenever the object has moved beyond RETARGET_MM.

x/y and z come from different sensors at different rates, and are kept apart:

  x/y   live3d blob tracking, every colour frame (~15 Hz). Precise, and fast
        enough that the object can be nudged by hand mid-approach.
  z     det-to-segment, in its own task (~1.3 Hz). The raw depth map does not
        resolve object height at this range at all — see the SEG_MATCH_MM
        comment below — so z is taken from the segmenter's box, top face =
        centre + dims.z/2. That uses each object's own measured height, so
        objects of different heights work with nothing hardcoded.

Neither stream waits on the other. The tracking loop never awaits a segmenter
read; it just uses the most recent z published to the side.

The grab is gated rather than automatic. Arriving at the grasp pose is not by
itself a reason to close the claws: if the object has moved since the last
retarget, or has slid out of the wrist camera's view during the descent, the
arm retreats BACKOFF_MM upward to regain the view and descends again, up to
MAX_BACKOFFS times before giving up without grabbing.

    python track_pick.py --watch yellow      # track only, never move the arm
    python track_pick.py --dry-run yellow    # plan, print retargets, no motion
    python track_pick.py yellow              # track and pick
"""
import asyncio
import sys
import time

import cv2
import numpy as np

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.media.video import CameraMimeType
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from tutorial import connect, goto_saved_pose, _decode_depth
from live3d import COLOURS, find_blobs, depth_pump, pose_pump
from seg_pick import (
    APPROACH_CLEARANCE_MM,
    GRASP_OFFSET_X,
    GRASP_OFFSET_Y,
    MIN_GRASP_Z,
    SEGMENTER,
    collision_limits,
    find_objects,
)
from slow_pick import GRIPPER_NAME, gripper_pose_in_world, wait_until_stopped
from overhead import Overhead, open_lenovo, grab_async as grab_overhead
from multi_pick import WORKSPACE, SMOOTH_EXTRA
from multi_pick_box import (
    BOX_EXCLUSION_RADIUS_MM,
    DROP_CLEARANCE_MM,
    in_drop_box,
    locate_drop_box,
)

# Re-issue the motion goal when the tracked position has drifted this far from
# what the arm is currently driving towards. Smaller means more responsive but
# more motion commands; the planner needs time to accept each one.
RETARGET_MM = 12.0

# --slow halves the velocity hints and widens the retarget threshold. Tracking
# still runs at full rate; only the arm moves more gently, and it commits to a
# goal for longer instead of chasing every small refinement.
SLOW_EXTRA = {
    "max_vel_degs_per_sec": 8.0,
    "max_acc_degs_per_sec2": 8.0,
}
SLOW_RETARGET_MM = 25.0

# The tracked estimate is smoothed to stop a single noisy frame yanking the
# target. x/y are precise so they follow quickly; z is noisy so it lags more.
ALPHA_XY = 0.5
ALPHA_Z = 0.25

DEPTH_PATCH = 6
LOST_AFTER_S = 1.5

# --- z from the segmenter ---------------------------------------------------
#
# The raw depth map cannot resolve these objects. Measured at top-pose (~672 mm
# range) the step between an object's top face and the table beside it came out
# at -22.5 to +17.5 mm across five blobs, several of them negative, i.e. the
# object reading as FURTHER than the table it stands on. Averaging 12 frames
# drove a 60 mm object's step to -0.1 mm, so this is not noise that integrates
# away: at this range the sensor is returning the table plane through the
# object. That is why z had to be clamped to MIN_GRASP_Z to be safe at all.
#
# det-to-segment does resolve them, because it segments a point cloud into
# labelled boxes rather than reading one pixel patch. It costs ~1.3 Hz, far too
# slow to gate the tracking loop on — so it runs in its own task and publishes
# z to the side, while live3d's blob tracking keeps x/y at full rate.
#
# The segmenter reports a box CENTRE. The gripper needs the TOP FACE, which is
# centre + dims.z/2. That derivation carries the object's own measured height,
# so objects of different heights each get their own correct grasp z with
# nothing hardcoded.
SEG_MATCH_MM = 70.0      # how close a segment must be in x/y to be our object
ALPHA_SEG_Z = 0.4        # smoothing on the segmenter's z
SEG_STALE_S = 4.0        # ignore a segmenter z older than this

# How long to let the pumps finish their in-flight RPC and exit on their own
# before cancelling them. One depth read is the slowest at ~170 ms.
SHUTDOWN_GRACE_S = 1.5

# --- grab gate --------------------------------------------------------------
#
# The descend leg used to close the gripper as soon as the arm stopped, with no
# check that it stopped anywhere useful. Two ways that goes wrong:
#
#   * the object is nudged (or tracking refines) after the last retarget, so
#     the claws close beside it;
#   * the object leaves the frame during descent — the camera is wrist-mounted
#     and ends up very close, so a blob that filled the view can slide out of
#     it — leaving the target coasting on a stale position.
#
# So gate the grab: only close when the object is currently visible AND the
# gripper is within GRAB_TOLERANCE_MM of it in x/y. Otherwise back off upward,
# where the wider view reacquires it, and descend again.
# Blob-size limits are calibrated for the ~670 mm survey view. Apparent size
# goes as 1/range, so as the wrist camera descends the object outgrows them and
# find_blobs discards it — the tracker goes blind at exactly the wrong moment.
# Scale the ceiling by how far the camera has closed in, with headroom.
SURVEY_RANGE_MM = 670.0
MAX_BLOB_SCALE = 6.0        # cap, so a wall filling the view is still rejected

GRAB_TOLERANCE_MM = 8.0     # max x/y error between gripper and target to grab
GRAB_FRESH_S = 0.4          # target must have been seen this recently
# An object that is merely out of view, but was agreed to be in the right place
# before the view was lost, is still safe to grab: it is not going anywhere on
# its own. Only distrust a fix once it is properly stale.
BLIND_GRAB_S = 3.0          # grab on a last-good fix up to this old
BACKOFF_MM = 45.0           # how far up to retreat to regain the view
BACKOFF_SETTLE_S = 0.35     # let the view stabilise after retreating
MAX_BACKOFFS = 3            # give up rather than bouncing forever

# --- overhead recovery ------------------------------------------------------
#
# Retreating upward to re-look does not work. The wrist camera regains the view
# at distance, descends, and loses the object again at the same range — the arm
# just bounces until it gives up.
#
# The fixed Lenovo camera does not have that problem: it never moves, so its
# view of the object cannot degrade as the arm closes in. It has no depth and
# is less accurate than the wrist camera, so it never sets a grasp; it answers
# only "roughly where is the object now?", which is all a recovery needs. The
# arm hovers over that spot at survey height, where the wrist camera can
# reacquire properly and the normal tracking loop takes over again.
OVERHEAD_HOVER_MM = 220.0   # height to hover at while reacquiring
OVERHEAD_SETTLE_S = 0.6     # let the wrist view settle after the hover move
OVERHEAD_REACQUIRE_S = 2.0  # how long to give the wrist camera to find it


class Target:
    """A tracked object's position: x/y tracked live, z fed from the side.

    The two axes come from different sensors at different rates and are kept
    apart deliberately. x/y update on every colour frame (~15 Hz) because blob
    centroids are precise and the object can be nudged mid-approach. z comes
    from the segmenter task whenever it manages a read (~1.3 Hz), because the
    depth map does not resolve object height at all at this range.

    `self.z` remains the depth-map estimate so the code still works with the
    segmenter absent; `seg_z` shadows it when a fresh segment is available.
    """

    def __init__(self, colour, xyz):
        self.colour = colour
        self.x, self.y, self.z = xyz
        self.last_seen = time.monotonic()
        self.updates = 1
        self.seg_z = None          # top-face z from det-to-segment
        self.seg_h = None          # the segment's own measured height
        self.seg_at = 0.0          # when that z last arrived
        self.seg_n = 0

    def update(self, xyz):
        x, y, z = xyz
        self.x += ALPHA_XY * (x - self.x)
        self.y += ALPHA_XY * (y - self.y)
        self.z += ALPHA_Z * (z - self.z)
        self.last_seen = time.monotonic()
        self.updates += 1

    def update_seg_z(self, top_z, height):
        """A new top-face height from the segmenter task."""
        if self.seg_z is None:
            self.seg_z = top_z
        else:
            self.seg_z += ALPHA_SEG_Z * (top_z - self.seg_z)
        self.seg_h = height
        self.seg_at = time.monotonic()
        self.seg_n += 1

    @property
    def seg_fresh(self):
        return (self.seg_z is not None
                and time.monotonic() - self.seg_at < SEG_STALE_S)

    @property
    def best_z(self):
        """Top-face z to grasp at, and where it came from.

        Prefers the segmenter, which measures the object's own height. Falls
        back to the depth-map z, which at this range is really the table plane
        and so gets clamped to the hand-verified height.
        """
        if self.seg_fresh:
            return self.seg_z, "seg"
        return self.z, "depth"

    @property
    def age(self):
        return time.monotonic() - self.last_seen

    @property
    def lost(self):
        return self.age > LOST_AFTER_S

    def grasp_pose(self):
        """Where to send the gripper: x/y from live tracking plus the
        calibrated offset, z from whichever source is trustworthy.

        The clamp stays as a floor in both cases — it is the deepest the
        hardware is known to tolerate, so it guards a bad segment as well as a
        missing one. A taller object segments above it and is unaffected.
        """
        z, _ = self.best_z
        return Pose(
            x=self.x + GRASP_OFFSET_X,
            y=self.y + GRASP_OFFSET_Y,
            z=max(z, MIN_GRASP_Z),
            o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0,
        )


async def seg_z_pump(machine, segmenter, target_ref, state):
    """Keep the target's z fed from det-to-segment, independently of tracking.

    Runs as its own task: one `get_object_point_clouds` read takes ~750 ms, so
    awaiting it in the tracking loop would drop that loop from ~15 Hz to ~1.3.
    Here it simply publishes whenever it has something, and the loop reads the
    latest value without ever blocking on it.

    Matching is by x/y proximity to the live-tracked position rather than by
    the segmenter's label, because a segment arrives up to a second stale and
    its label is not reliable enough to key on when several objects share a
    colour.
    """
    while not state["stop"]:
        target = target_ref.get("t")
        if target is None:
            await asyncio.sleep(0.05)
            continue
        try:
            objects = await find_objects(machine, segmenter)
        except Exception as exc:                      # noqa: BLE001
            # On the way out this is just the channel closing; don't sit out
            # the back-off sleep, or shutdown waits on a pump with nothing
            # left to do.
            if state["stop"]:
                return
            state["seg_err"] = str(exc)
            await asyncio.sleep(0.5)
            continue

        best, best_d = None, SEG_MATCH_MM
        for label, pose, dims in objects:
            if dims is None:
                continue
            d = ((pose.x - target.x) ** 2 + (pose.y - target.y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = (pose, dims), d

        if best is not None:
            pose, dims = best
            # centre -> top face, using the segment's own measured height.
            target.update_seg_z(pose.z + dims.z / 2.0, dims.z)
        state["seg_reads"] = state.get("seg_reads", 0) + 1
        await asyncio.sleep(0)


def observe(hsv, depth, colour, intr, R, T, rejects=None, drop=None,
            max_scale=1.0):
    """All world-frame positions of `colour` in this frame. Cheap, local.

    `rejects` collects blobs that were found but failed a filter, so a failed
    acquisition can say why instead of just reporting nothing.

    `drop` is the green box's world pose. A colour detector still fires on an
    object already sitting in the box (seen through its open top), and going
    back for that would either re-pick something already placed or drive the
    gripper down inside the box walls.

    `max_scale` relaxes the blob-size ceiling as the camera closes in — see
    find_blobs. Without it the object is discarded for being too big once the
    camera is within ~350 mm of it.
    """
    out = []
    for cx, cy, bx, by, bw, bh in find_blobs(hsv, colour, max_scale):
        patch = depth[
            max(0, cy - DEPTH_PATCH):cy + DEPTH_PATCH + 1,
            max(0, cx - DEPTH_PATCH):cx + DEPTH_PATCH + 1,
        ]
        valid = patch[patch > 0]
        if valid.size < 20:
            if rejects is not None:
                rejects.append((None, "no depth at blob centre"))
            continue
        z_mm = float(np.percentile(valid, 25))
        cam = np.array([
            (cx - intr.center_x_px) * z_mm / intr.focal_x_px,
            (cy - intr.center_y_px) * z_mm / intr.focal_y_px,
            z_mm,
        ])
        w = R @ cam + T
        if not (WORKSPACE["x"][0] < w[0] < WORKSPACE["x"][1]
                and WORKSPACE["y"][0] < w[1] < WORKSPACE["y"][1]
                and WORKSPACE["z"][0] < w[2] < WORKSPACE["z"][1]):
            if rejects is not None:
                axis = ("x" if not WORKSPACE["x"][0] < w[0] < WORKSPACE["x"][1]
                        else "y" if not WORKSPACE["y"][0] < w[1] < WORKSPACE["y"][1]
                        else "z")
                rejects.append(((float(w[0]), float(w[1]), float(w[2])),
                                f"outside workspace {axis}"))
            continue
        pos = Pose(x=float(w[0]), y=float(w[1]), z=float(w[2]), o_z=1)
        if in_drop_box(pos, drop):
            if rejects is not None:
                rejects.append(((pos.x, pos.y, pos.z),
                                "already in the drop box"))
            continue
        out.append(((float(w[0]), float(w[1]), float(w[2])), (bx, by, bw, bh)))
    return out


def nearest(observations, target):
    """The observation closest to the current target — keeps the same object
    when several of one colour are visible."""
    if not observations:
        return None
    if target is None:
        return max(observations, key=lambda o: o[1][2] * o[1][3])
    return min(
        observations,
        key=lambda o: (o[0][0] - target.x) ** 2 + (o[0][1] - target.y) ** 2,
    )


async def main(argv):
    watch = "--watch" in argv
    dry_run = "--dry-run" in argv
    no_box = "--no-box" in argv
    no_overhead = "--no-overhead" in argv
    slow = "--slow" in argv
    colour = next((a for a in argv[1:] if not a.startswith("-")), None)

    if colour not in COLOURS:
        print(f"usage: track_pick.py [--watch|--dry-run] [--slow] [--no-box] "
              f"<{'|'.join(sorted(COLOURS))}>")
        print("  --slow    move gently (8 deg/s hints, 25 mm retarget threshold)")
        print("  --no-box  skip the drop-box lookup and exclude nothing")
        print("  --no-overhead  do not use the Lenovo camera for recovery")
        return

    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        arm = Arm.from_robot(machine, "arm")
        gripper = Gripper.from_robot(machine, "gripper")
        motion = MotionClient.from_robot(machine, "builtin")
        segmenter = VisionClient.from_robot(machine, SEGMENTER)
        intr = (await cam.get_properties()).intrinsic_parameters

        min_z, why = await collision_limits(machine)
        print(f"grasp floor z={min_z:.1f}")

        # The overhead camera is optional: without it the pick still runs, it
        # just has no recovery when the wrist camera loses the object.
        overhead, oh_cap = None, None
        if not no_overhead:
            overhead = Overhead.load()
            if not overhead.ready:
                print("overhead camera NOT calibrated — no recovery available")
                print("  run: python overhead.py --calibrate")
                overhead = None
            else:
                oh_cap = open_lenovo()
                if oh_cap is None:
                    print("overhead camera could not be opened — "
                          "no recovery available")
                    overhead = None
                else:
                    print("overhead camera ready for recovery")
        move_extra = SLOW_EXTRA if slow else SMOOTH_EXTRA
        retarget_mm = SLOW_RETARGET_MM if slow else RETARGET_MM
        if slow:
            print(f"slow mode: {SLOW_EXTRA['max_vel_degs_per_sec']:.0f} deg/s, "
                  f"retarget at {retarget_mm:.0f} mm")

        state = {"depth": None, "R": None, "T": None, "stop": False,
                 "depth_n": 0, "seg_reads": 0, "seg_err": None}
        # The segmenter task needs the target, which does not exist until
        # acquisition. Hand it a box to read from rather than restarting it.
        target_ref = {"t": None}
        pumps = [
            asyncio.create_task(depth_pump(cam, state)),
            asyncio.create_task(pose_pump(machine, state)),
            asyncio.create_task(seg_z_pump(machine, segmenter, target_ref,
                                           state)),
        ]

        try:
            if not (watch or dry_run):
                print("moving to top-pose...")
                await goto_saved_pose(machine, "top-pose")
                await wait_until_stopped(arm)

            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if state["depth"] is not None and state["R"] is not None:
                    break
                await asyncio.sleep(0.005)
            if state["depth"] is None or state["R"] is None:
                print("no depth or pose")
                return

            # Find the drop box before acquiring, so anything already sitting
            # in it is excluded from the very first frame rather than being
            # acquired and then rejected mid-approach.
            drop = None
            if not no_box:
                print("locating drop box...")
                drop = await locate_drop_box(machine, cam)
                if drop is None:
                    print("  no drop box found — nothing will be excluded")
                else:
                    print(f"  excluding +/-{BOX_EXCLUSION_RADIUS_MM:.0f} mm "
                          f"around ({drop.x:.1f}, {drop.y:.1f})")

            # --- ACQUIRE: a single frame is enough to start ---
            t_acq = time.monotonic()
            target = None
            for _ in range(40):
                images, _ = await cam.get_images(filter_source_names=["color"])
                jpeg = next((i for i in images
                             if i.mime_type == CameraMimeType.JPEG), None)
                if jpeg is None:
                    continue
                bgr = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8),
                                   cv2.IMREAD_COLOR)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                rejects = []
                obs = observe(hsv, state["depth"], colour, intr,
                              state["R"], state["T"], rejects, drop)
                if obs:
                    xyz, _ = nearest(obs, None)
                    target = Target(colour, xyz)
                    break

            if target is None:
                print(f"could not acquire '{colour}'")
                if rejects:
                    print(f"  {len(rejects)} blob(s) found but rejected:")
                    for pos, why in rejects[:4]:
                        where = (f"({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})"
                                 if pos else "")
                        print(f"    {where:28s} {why}")
                    print(f"  workspace is x{WORKSPACE['x']} y{WORKSPACE['y']} "
                          f"z{WORKSPACE['z']}")
                    print("  the arm may not be at top-pose, or the objects "
                          "have moved outside it")
                else:
                    print(f"  no {colour} blobs detected at all")
                return

            target_ref["t"] = target
            print(f"acquired '{colour}' in "
                  f"{(time.monotonic() - t_acq) * 1000:.0f} ms at "
                  f"({target.x:.1f}, {target.y:.1f}, {target.z:.1f})")
            print("  x/y tracked live; z fed asynchronously by "
                  f"{SEGMENTER}")

            if not (watch or dry_run):
                await gripper.open()

            # --- TRACK: refine continuously, retarget when it drifts ---
            commanded = None
            phase = "approach"
            backoffs = 0
            frames = 0
            t0 = time.monotonic()
            last_log = 0.0

            while True:
                images, _ = await cam.get_images(filter_source_names=["color"])
                jpeg = next((i for i in images
                             if i.mime_type == CameraMimeType.JPEG), None)
                if jpeg is None:
                    continue
                bgr = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8),
                                   cv2.IMREAD_COLOR)
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                frames += 1

                # How close is the camera to the object right now? T is the
                # camera origin in world coordinates, so this is the true
                # viewing range, not the gripper height.
                cam_range = SURVEY_RANGE_MM
                if state["T"] is not None:
                    cam_range = max(
                        50.0,
                        ((state["T"][0] - target.x) ** 2
                         + (state["T"][1] - target.y) ** 2
                         + (state["T"][2] - target.z) ** 2) ** 0.5,
                    )
                blob_scale = min(MAX_BLOB_SCALE,
                                 max(1.0, SURVEY_RANGE_MM / cam_range))

                obs = observe(hsv, state["depth"], colour, intr,
                              state["R"], state["T"], None, drop,
                              max_scale=blob_scale)
                hit = nearest(obs, target)
                if hit is not None:
                    target.update(hit[0])

                grasp = target.grasp_pose()
                goal = (Pose(x=grasp.x, y=grasp.y,
                             z=grasp.z + APPROACH_CLEARANCE_MM,
                             o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
                        if phase == "approach" else grasp)

                moved = commanded is None or max(
                    abs(goal.x - commanded.x),
                    abs(goal.y - commanded.y),
                    abs(goal.z - commanded.z),
                ) > retarget_mm

                elapsed = time.monotonic() - t0
                if elapsed - last_log > 0.4:
                    last_log = elapsed
                    seen = "seen" if target.age < 0.2 else f"lost {target.age:.1f}s"
                    z, src = target.best_z
                    hgt = (f" h={target.seg_h:.0f}" if target.seg_h else "")
                    print(f"\r{frames/elapsed:5.1f} Hz  {phase:8s} "
                          f"target({target.x:6.1f},{target.y:6.1f},{z:5.1f}) "
                          f"z:{src}{hgt} n={target.seg_n} {seen}    ",
                          end="", flush=True)

                if watch:
                    if target.lost:
                        print("\n  target lost")
                        return
                    await asyncio.sleep(0)
                    continue

                if moved and goal.z >= min_z:
                    if dry_run:
                        print(f"\n  [dry-run] retarget {phase} -> "
                              f"({goal.x:.1f}, {goal.y:.1f}, {goal.z:.1f})")
                    else:
                        await motion.move(
                            component_name=GRIPPER_NAME,
                            destination=PoseInFrame(reference_frame="world",
                                                    pose=goal),
                            extra=move_extra,
                        )
                    commanded = goal

                if not dry_run and not await arm.is_moving():
                    if phase == "approach":
                        phase = "descend"
                        commanded = None
                    else:
                        # Arrived at the grasp pose. Decide between three
                        # cases — they are NOT the same problem:
                        #
                        #   aligned      -> grab
                        #   drifted      -> the object really moved; the arm
                        #                   is in the wrong place, so back off
                        #                   and re-approach
                        #   out of view  -> the object stopped being visible
                        #                   but never moved. Backing off here
                        #                   is pointless: it regains the view,
                        #                   re-descends, loses it at the same
                        #                   distance and bounces until it
                        #                   gives up. If the position was
                        #                   agreed before the view was lost,
                        #                   trust it and grab.
                        here = await gripper_pose_in_world(machine)
                        err = max(abs(here.x - grasp.x),
                                  abs(here.y - grasp.y))
                        fresh = target.age < GRAB_FRESH_S
                        aligned = err <= GRAB_TOLERANCE_MM

                        if aligned and (fresh or target.age < BLIND_GRAB_S):
                            if not fresh:
                                print(f"\n  object out of view for "
                                      f"{target.age:.1f}s but still aligned "
                                      f"({err:.1f} mm) — grabbing on the last "
                                      f"good fix")
                            break

                        if backoffs >= MAX_BACKOFFS:
                            why = (f"lost for {target.age:.1f}s" if not fresh
                                   else f"x/y off by {err:.1f} mm")
                            print(f"\n  giving up after {backoffs} retries "
                                  f"({why}) — not grabbing")
                            return

                        backoffs += 1
                        why = (f"lost for {target.age:.1f}s" if not fresh
                               else f"x/y off by {err:.1f} mm")

                        # Ask the overhead camera where the object is now. It
                        # sees the whole table from a fixed mount, so unlike
                        # the wrist camera it still has a view here.
                        seen = None
                        if overhead is not None and overhead.ready:
                            frame = await grab_overhead(oh_cap)
                            if frame is not None:
                                hits = overhead.locate(frame, colour)
                                if hits:
                                    # Nearest to where we believed it was, so
                                    # another object of the same colour on the
                                    # table does not steal the recovery.
                                    (wx, wy), _, _ = min(
                                        hits,
                                        key=lambda o: (o[0][0] - target.x) ** 2
                                        + (o[0][1] - target.y) ** 2,
                                    )
                                    seen = (wx, wy)

                        if seen is None:
                            print(f"\n  {why} — overhead camera cannot see "
                                  f"the {colour} object either "
                                  f"({backoffs}/{MAX_BACKOFFS})")
                            # Nothing better to go on: rise to survey height
                            # over the last known spot and try the wrist
                            # camera again from there.
                            hover_x, hover_y = target.x, target.y
                        else:
                            moved_mm = ((seen[0] - target.x) ** 2
                                        + (seen[1] - target.y) ** 2) ** 0.5
                            print(f"\n  {why} — overhead camera puts it at "
                                  f"({seen[0]:.1f}, {seen[1]:.1f}), "
                                  f"{moved_mm:.0f} mm away "
                                  f"({backoffs}/{MAX_BACKOFFS})")
                            hover_x, hover_y = seen
                            # Adopt it as the working position so the hover is
                            # centred on the object and the wrist camera has
                            # the best chance of picking it straight back up.
                            target.x, target.y = seen

                        # Hover above it at survey height, where the wrist
                        # camera resolves the object properly again.
                        hover_z = max(min_z, OVERHEAD_HOVER_MM)
                        print(f"    hovering at ({hover_x:.1f}, "
                              f"{hover_y:.1f}, {hover_z:.1f}) to reacquire")
                        await motion.move(
                            component_name=GRIPPER_NAME,
                            destination=PoseInFrame(
                                reference_frame="world",
                                pose=Pose(x=hover_x + GRASP_OFFSET_X,
                                          y=hover_y + GRASP_OFFSET_Y,
                                          z=hover_z,
                                          o_x=0.0, o_y=0.0, o_z=-1.0,
                                          theta=0.0),
                            ),
                            extra=move_extra,
                        )
                        await wait_until_stopped(arm)
                        await asyncio.sleep(OVERHEAD_SETTLE_S)

                        # Give the wrist camera a moment to find it from up
                        # here before resuming, so the descent restarts from a
                        # properly tracked position rather than the overhead
                        # camera's rougher one.
                        t_re = time.monotonic()
                        while time.monotonic() - t_re < OVERHEAD_REACQUIRE_S:
                            imgs, _ = await cam.get_images(
                                filter_source_names=["color"])
                            jp = next((i for i in imgs if i.mime_type
                                       == CameraMimeType.JPEG), None)
                            if jp is not None:
                                h2 = cv2.cvtColor(
                                    cv2.imdecode(
                                        np.frombuffer(jp.data, np.uint8),
                                        cv2.IMREAD_COLOR),
                                    cv2.COLOR_BGR2HSV)
                                o2 = observe(h2, state["depth"], colour, intr,
                                             state["R"], state["T"], None,
                                             drop)
                                h3 = nearest(o2, target)
                                if h3 is not None:
                                    target.update(h3[0])
                                    print("    wrist camera reacquired it")
                                    break
                            await asyncio.sleep(0.05)

                        # Restart the approach from above rather than dropping
                        # straight back down, so the descent re-runs with
                        # tracking live from the start.
                        phase = "approach"
                        commanded = None
                elif dry_run and frames > 60:
                    break

                await asyncio.sleep(0)

            if watch or dry_run:
                return

            print("\nclosing gripper...")
            grabbed = await gripper.grab()
            print("  grabbed." if grabbed
                  else "  WARNING: grab() reported nothing grasped.")

            lift_z, _ = target.best_z
            lift = Pose(x=target.x + GRASP_OFFSET_X, y=target.y + GRASP_OFFSET_Y,
                        z=max(lift_z, MIN_GRASP_Z) + APPROACH_CLEARANCE_MM,
                        o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
            await motion.move(
                component_name=GRIPPER_NAME,
                destination=PoseInFrame(reference_frame="world", pose=lift),
                extra=move_extra,
            )

            if drop is None:
                print("done — object lifted, held above the pick point.")
                return

            # Carry to the box and release above its rim. The object hangs
            # below the gripper origin, so release high enough to clear the
            # wall rather than dragging it across the edge.
            over_box = Pose(x=drop.x, y=drop.y,
                            z=drop.z + DROP_CLEARANCE_MM,
                            o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
            print(f"carrying to drop box, releasing at z={over_box.z:.1f} "
                  f"({DROP_CLEARANCE_MM:.0f} mm above the rim at {drop.z:.1f})")
            await motion.move(
                component_name=GRIPPER_NAME,
                destination=PoseInFrame(reference_frame="world", pose=over_box),
                extra=move_extra,
            )
            await wait_until_stopped(arm)

            print("releasing...")
            await gripper.open()
            await asyncio.sleep(0.5)
            print("done — object dropped in the green box.")

        finally:
            # Shut the pumps down gracefully before the `async with` closes the
            # channel underneath them. Each pump is usually parked inside an
            # RPC (get_images, the gathered transform_pose calls,
            # get_object_point_clouds); cancelling one does not wait for that
            # gRPC stream to unwind, so if the channel closes first the Rust
            # layer logs "error deserializing message: channel closed" once per
            # stream still in flight.
            #
            # So set the stop flag and give each pump a moment to notice it at
            # the top of its loop and return on its own. Only cancel what is
            # still running after that, which leaves nothing mid-RPC in the
            # normal case.
            if oh_cap is not None:
                oh_cap.release()
            state["stop"] = True
            done, pending = await asyncio.wait(pumps, timeout=SHUTDOWN_GRACE_S)
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
