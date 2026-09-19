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

Arriving at the grasp pose closes the claws. There used to be a gate here that
retreated upward and re-approached whenever the object appeared to have moved,
but it judged movement from the WRIST camera, which cannot see the object at
grab distance: target x/y froze at its last value, the alignment error was
measured against that stale guess, and the arm bounced up and down before
grabbing empty space. That machinery is gone. --motion now means "refuse to
grab unless aligned" rather than "retry".

Re-introducing it needs a sensor that can still see the object while the
gripper is on top of it -- the overhead camera, once calibrated.

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
import sam_observe
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
# Re-command the pose when the wrist angle changes by more than this, even if
# the position has not moved. Without it a pure rotation is never issued.
RETARGET_DEG = 3.0

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
# After a disturbance the object is somewhere else by definition, so matching
# has to reach further or the segmenter drops it exactly when it matters.
SEG_MATCH_DISTURBED_MM = 250.0
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

# Wrist angle used when nothing has measured a grasp angle. 0.0 is what every
# grasp in this file used before SAM could supply one, so an un-surveyed pick
# behaves exactly as it always has. Measured on the arm: commanded theta is
# achieved to within 0.04 deg, and joint 6 moves -1 deg per +1 deg of theta.
DEFAULT_GRASP_THETA = 0.0

# How close SAM's mask centroid and the blob centroid must land before the two
# detectors are agreed to be looking at the same physical object.
#
# MEASURED 2026-09-19 over three trials: when both detectors saw the same
# object the gap was 3.0-8.5 mm. When SAM saw an object in the parts bin and
# the nearest blob of that colour was a DIFFERENT object on the table, the gap
# was 195-266 mm. There is no overlap, so this threshold has wide margin on
# both sides. It exists because the table and the bin can hold several objects
# of one colour, and applying SAM's grasp angle to the wrong one would twist
# the wrist for an object the arm is not approaching.
SAM_MATCH_MM = 40.0

# The wrist must already be at the grasp angle before the descent begins.
# Rotating at the bottom would sweep the jaws THROUGH the object; rotating on
# the way up would twist whatever is held.
# A single observation landing this far from the smoothed estimate means the
# object was disturbed, not that the detector is jittering. Tracking is stable
# to +/-0.3 mm in steady state, so this is ~65x the noise floor.
MOVED_MM = 20.0
# How long a disturbance keeps mattering. Long enough that a jump seen during
# the descent still blocks the grab at the bottom.
MOVED_WINDOW_S = 3.0

GRAB_TOLERANCE_MM = 8.0     # max x/y error between gripper and target to grab
GRAB_FRESH_S = 0.4          # target must have been seen this recently
# An object that is merely out of view, but was agreed to be in the right place
# before the view was lost, is still safe to grab: it is not going anywhere on
# its own. Only distrust a fix once it is properly stale.
BLIND_GRAB_S = 3.0          # grab on a last-good fix up to this old
BACKOFF_MM = 45.0           # unused: see the note on motion detection above
BACKOFF_SETTLE_S = 0.35     # let the view stabilise after retreating
MAX_BACKOFFS = 3            # unused: see the note on motion detection above

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
# Reacquiring after a disturbance: rise until the object is visible again
# rather than to a fixed height, because how far the arm must retreat depends
# on the object's size and where it ended up. A fixed 220 mm was too low --
# find_blobs' size limits are calibrated for the survey range (~670 mm), so at
# 220 mm the object subtends ~3x the expected pixels and is rejected as too
# big, leaving the tracker unable to re-acquire the very object it retreated
# to find.
REACQUIRE_STEP_MM = 80.0     # how much higher to try on each attempt
REACQUIRE_LOOK_S = 1.0       # how long to look before climbing further
# Ceiling for the climb: the height the arm acquires from. Measured at
# top-pose, gripper z = 542 mm, and acquisition demonstrably works there, so
# there is nothing to gain by climbing past it. (WORKSPACE["z"] is NOT the
# bound to use -- its 250 mm limit describes where OBJECTS may be, not where
# the arm may go, and it would cap the climb below the useful range.)
REACQUIRE_MAX_Z = 542.0
OVERHEAD_SETTLE_S = 0.6     # let the wrist view settle after the hover move
# (the reacquire climb uses REACQUIRE_LOOK_S per step instead)


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

    def __init__(self, colour, xyz, grasp_theta=None):
        self.colour = colour
        self.x, self.y, self.z = xyz
        # World-frame wrist angle for the jaws, from a SAM survey. None means
        # no measurement, and every grasp then uses DEFAULT_GRASP_THETA -- the
        # behaviour this code had before the angle existed at all.
        self.grasp_theta = grasp_theta
        self.last_seen = time.monotonic()
        self.updates = 1
        self.seg_z = None          # top-face z from det-to-segment
        self.seg_h = None          # the segment's own measured height
        self.seg_at = 0.0          # when that z last arrived
        self.seg_n = 0
        self.jump_mm = 0.0         # size of the last overhead-confirmed move
        self.moved_at = 0.0        # when the overhead camera last saw it move
        self.overhead_mm = 0.0     # last overhead-to-overhead step
        self.overhead_xy = None    # the overhead's own previous reading
        self.seg_x = None          # x/y from the point-cloud segmenter
        self.seg_y = None
        self.seg_xy_at = 0.0

    def update(self, xyz):
        x, y, z = xyz
        # NOTE: this does NOT set moved_at. The wrist camera cannot tell "the
        # object moved" from "I lost sight of it" -- both look like the
        # observation jumping -- and judging movement here made the arm back
        # off from objects that never moved. Only the overhead camera, which
        # keeps the whole table in view throughout the descent, is allowed to
        # declare a disturbance. See mark_moved / overhead_watch.
        self.jump_mm = ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5
        self.x += ALPHA_XY * (x - self.x)
        self.y += ALPHA_XY * (y - self.y)
        self.z += ALPHA_Z * (z - self.z)
        self.last_seen = time.monotonic()
        self.updates += 1

    def mark_moved(self, wx, wy):
        """The overhead camera says the object is now at (wx, wy).

        This is the ONLY way a disturbance is declared. The overhead camera is
        fixed, sees the whole table, and keeps seeing the object while the
        gripper is on top of it -- so a change it reports is a change in the
        world, not a change in visibility.
        """
        # Compare the overhead against ITS OWN previous reading, not against
        # self.x/self.y.
        #
        # self.x/y comes from the point cloud; this comes from the overhead
        # homography. They are different sensors with different errors (the
        # homography's own residual is up to 5.2 mm, and the two disagree by
        # more than that in places). Comparing across them meant a standing
        # sensor disagreement larger than MOVED_MM read as "the object is
        # moving" on every single frame -- so a completely stationary object
        # stayed permanently `disturbed` and the arm backed off, re-approached
        # and backed off again without anything having moved.
        #
        # Overhead-to-overhead cancels that offset entirely: whatever the
        # homography's bias is at that spot, it is the same in both readings,
        # so only real displacement survives.
        prev = self.overhead_xy
        self.overhead_xy = (wx, wy)
        if prev is None:
            self.overhead_mm = 0.0
            return False
        d = ((wx - prev[0]) ** 2 + (wy - prev[1]) ** 2) ** 0.5
        self.overhead_mm = d
        if d > MOVED_MM:
            self.jump_mm = d
            self.moved_at = time.monotonic()
            return True
        return False

    @property
    def disturbed(self):
        """True if the object jumped recently enough to distrust the descent.

        Kept as a time window rather than a flag so a disturbance seen a
        moment ago still stops a grab that is about to happen.
        """
        return time.monotonic() - self.moved_at < MOVED_WINDOW_S

    def update_seg_xy(self, x, y):
        """An x/y measured by the point-cloud segmenter.

        Stored rather than applied. During normal tracking the blob centroid
        is both faster (16 Hz vs ~1.3 Hz) and finer, so it stays in charge.
        This is the fallback for the one case the blob path cannot handle:
        the object was disturbed and the wrist camera can no longer resolve
        it, where a ~1 Hz measurement from the point cloud beats a stale
        estimate or the overhead camera's plane homography.
        """
        self.seg_x, self.seg_y = float(x), float(y)
        self.seg_xy_at = time.monotonic()

    @property
    def seg_xy_fresh(self):
        return (self.seg_x is not None
                and time.monotonic() - self.seg_xy_at < SEG_STALE_S)

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
            o_x=0.0, o_y=0.0, o_z=-1.0, theta=self.theta,
        )

    @property
    def theta(self):
        """Wrist angle for the grasp, in world degrees.

        Falls back to DEFAULT_GRASP_THETA when no survey measured an angle, so
        an un-surveyed pick behaves exactly as it did before.
        """
        return (DEFAULT_GRASP_THETA if self.grasp_theta is None
                else self.grasp_theta)


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

        # A disturbed object has moved, so the usual tight match radius would
        # reject the very segment that shows where it went.
        best, best_d = None, (SEG_MATCH_DISTURBED_MM if target.disturbed
                              else SEG_MATCH_MM)
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
            # The point cloud also knows WHERE the object is, not just how
            # tall it is, and it is the accurate source. Publish x/y too so a
            # disturbed target can be re-seated on a real measurement rather
            # than on the overhead camera's rougher homography estimate.
            target.update_seg_xy(pose.x, pose.y)
        state["seg_reads"] = state.get("seg_reads", 0) + 1
        await asyncio.sleep(0)


# How near an overhead hit must be to the target to be treated as the SAME
# object rather than a different block of the same colour. Generous, because
# the overhead homography is the rough source (2.6 mm mean, 5.2 mm max
# residual on the calibration set) and the target may itself be mid-move.
OVERHEAD_SAME_MM = 120.0
OVERHEAD_PERIOD_S = 0.15    # ~7 Hz; the capture itself costs ~60 ms


async def overhead_watch_pump(oh, cap, target_ref, state):
    """Watch the object from the fixed camera and declare real movement.

    This pump exists because the wrist camera cannot tell "the object moved"
    apart from "I stopped being able to see it". Both present as the
    observation jumping, and judging movement from the wrist made the arm
    abandon objects that had never moved -- it would lose the view on the way
    down, read that as a disturbance, and back off to re-look, repeatedly.

    The overhead camera has neither failure: it is bolted in place, so its
    view cannot degrade as the arm descends, and it sees the whole table, so
    the object stays in frame throughout. A change it reports is a change in
    the world. It is NOT accurate enough to grasp on -- that still comes from
    the point cloud -- it only answers "did the thing move?".

    Tracking is by proximity to the last overhead fix, not to the target's
    own estimate, so that a second block of the same colour sitting elsewhere
    on the table cannot be mistaken for this one having jumped to it.
    """
    last = None                 # last confirmed overhead fix for THIS object
    announced = False           # already printed the current disturbance
    others = False              # have we ever seen another block of this
                                # colour elsewhere? If so, a lone hit is not
                                # automatically ours.
    while not state["stop"]:
        target = target_ref.get("t")
        if target is None:
            last = None
            others = False
            await asyncio.sleep(0.1)
            continue
        try:
            frame = await grab_overhead(cap)
            if frame is None:
                await asyncio.sleep(OVERHEAD_PERIOD_S)
                continue
            # locate() is OpenCV work on a full frame: blocking, and long
            # enough to starve the SDK's 1 s keepalive if awaited inline.
            hits = await asyncio.to_thread(oh.locate, frame, target.colour)
        except Exception as exc:                      # noqa: BLE001
            if state["stop"]:
                return
            state["oh_err"] = str(exc)
            await asyncio.sleep(0.5)
            continue

        state["oh_reads"] = state.get("oh_reads", 0) + 1
        if not hits:
            # Nothing seen. Say nothing: silence is not evidence of movement,
            # and the whole point of this pump is not to guess.
            state["oh_seen"] = False
            await asyncio.sleep(OVERHEAD_PERIOD_S)
            continue

        anchor = last if last is not None else (target.x, target.y)

        def nearest(radius):
            b, bd = None, radius
            for (wx, wy), _px, _bbox in hits:
                d = ((wx - anchor[0]) ** 2 + (wy - anchor[1]) ** 2) ** 0.5
                if d < bd:
                    b, bd = (wx, wy), d
            return b

        best = nearest(OVERHEAD_SAME_MM)
        if best is None and len(hits) == 1 and not others:
            # Nothing near where the object was, exactly one block of this
            # colour in view, and no OTHER same-coloured block seen recently
            # somewhere else. That block is ours, moved further than the match
            # radius.
            #
            # Both halves of this condition are load-bearing:
            #   - without it, a big push went undetected -- the object's new
            #     position failed the very identity check meant to guard
            #     against decoys, and a big push is the case that matters.
            #   - without `not others`, the gripper occluding our block leaves
            #     a decoy as the only hit, and adopting it reports a huge
            #     phantom move. Measured: a decoy at (600, 400) produced a
            #     spurious "moved 476 mm".
            # `others` remembers where same-coloured blocks were seen while
            # ours was still visible, so a lone survivor is only trusted when
            # there was never anything else to confuse it with.
            best = hits[0][0]

        if best is None:
            # Either nothing plausible, or several same-coloured blocks and no
            # way to tell which is ours -- most likely the gripper is covering
            # it. Not evidence of movement; say nothing.
            state["oh_seen"] = False
            await asyncio.sleep(OVERHEAD_PERIOD_S)
            continue

        state["oh_seen"] = True
        state["oh_xy"] = best
        last = best
        # Anything else in view right now is a decoy we must remember, so that
        # if our object later disappears under the gripper we do not mistake
        # that decoy for it.
        if any(((wx - best[0]) ** 2 + (wy - best[1]) ** 2) ** 0.5
               > OVERHEAD_SAME_MM for (wx, wy), _px, _b in hits):
            others = True
        moved = target.mark_moved(*best)
        # mark_moved compares against the target's own estimate, which only
        # catches up once the wrist re-acquires -- so a single push keeps
        # testing true at pump rate. Report the transition, not the state,
        # or one nudge floods the log with the same line ~7x a second.
        if moved and not announced:
            print(f"  overhead: {target.colour} moved {target.jump_mm:.0f} mm "
                  f"-> ({best[0]:.0f}, {best[1]:.0f})")
        announced = moved
        await asyncio.sleep(OVERHEAD_PERIOD_S)


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


def nearest_to(observations, xy, max_mm):
    """The observation closest to `xy`, or None if none is within `max_mm`.

    This is how a SAM-chosen object is identified in the blob detector's own
    terms. SAM decides WHICH object to grasp and at what angle; the blob
    detector decides WHERE it is. With two blocks of one colour on the table
    -- or one on the table and one in the bin -- those two decisions have to
    refer to the same physical object, and the only thing linking them is
    approximate agreement in position.

    Returning None rather than a best guess is deliberate: if no blob is near
    what SAM segmented, the two detectors disagree about the scene, and
    grasping at SAM's angle would then twist the wrist for an object the arm
    is not actually approaching.
    """
    # Always a 2-tuple, so callers can unpack unconditionally. Returning a
    # bare None here would crash the unpack at the one call site.
    if not observations:
        return None, float("inf")
    best = min(observations,
               key=lambda o: (o[0][0] - xy[0]) ** 2 + (o[0][1] - xy[1]) ** 2)
    d = ((best[0][0] - xy[0]) ** 2 + (best[0][1] - xy[1]) ** 2) ** 0.5
    return (best, d) if d <= max_mm else (None, d)



async def run_pick(machine, arm, gripper, motion, cam, segmenter, target,
                   colour, intr, state, drop, min_z, move_extra, retarget_mm,
                   overhead=None, oh_cap=None, *, watch=False, dry_run=False,
                   no_motion=False):
    """Track the acquired object, grasp it, and drop it in the box.

    Split out of main() so a caller that ALREADY has a robot session, a loaded
    SAM model, camera pumps and a drop-box fix can run a pick without building
    any of it again. web_gui is such a caller: it holds all of this open
    continuously, and shelling out to a fresh `track_pick.py` made it reload
    SAM (~860M params) and re-locate the drop box on every pick, while two
    processes fought over a machine that serves one camera client reliably.

    Everything this needs is passed in: there are no module-level or closure
    dependencies, so the caller owns the session and its lifetime.

    Acquisition is deliberately NOT here. The caller supplies `target`
    already acquired, because a long-running service has better information
    than a cold start does -- it has been watching the table all along.
    """
    # --- TRACK: refine continuously, retarget when it drifts ---
    commanded = None
    phase = "approach"
    frames = 0
    t0 = time.monotonic()
    last_log = 0.0
    backoffs = 0        # overhead re-approach attempts this pick

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
            # Just track it. The mid-descent stop that used to live here
            # reacted to target.disturbed, and nothing sets that any more:
            # the overhead motion watch is gone, so there is no signal that
            # the object moved and no reason to interrupt a descent.
            target.update(hit[0])

        grasp = target.grasp_pose()
        goal = (Pose(x=grasp.x, y=grasp.y,
                     z=grasp.z + APPROACH_CLEARANCE_MM,
                     o_x=0.0, o_y=0.0, o_z=-1.0, theta=target.theta)
                if phase == "approach" else grasp)

        # Position AND wrist angle. Comparing only x/y/z meant a
        # goal that differed solely in theta looked unchanged, so no
        # move was issued and the wrist never rotated to the grasp
        # angle -- the measured angle reached the Pose and then went
        # nowhere.
        turned = (commanded is not None
                  and abs(goal.theta - commanded.theta)
                  > RETARGET_DEG)
        moved = commanded is None or turned or max(
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
                      f"({goal.x:.1f}, {goal.y:.1f}, {goal.z:.1f}) "
                      f"theta={goal.theta:+.1f}")
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

                # `err` is the gripper against its own target, so it
                # says whether the arm ARRIVED, not whether the object
                # moved: when the object is nudged, tracking follows
                # it and the arm is retargeted, so both move together
                # and err stays small. target.disturbed is the check
                # that actually sees a disturbance.
                # A stale wrist fix is NOT a reason to back off while
                # the overhead camera can still see the object sitting
                # where we think it is. That was the bug: the wrist
                # loses the view on the way down, `age` grows past
                # BLIND_GRAB_S, and the arm abandoned a block that had
                # never moved. The overhead vouches for it instead.
                # If the arm is aligned, GRAB. Do not retreat because the
                # wrist stopped resolving the object on the way down.
                #
                # Nothing declares the object moved any more -- the overhead
                # motion watch was removed -- so losing the view is the only
                # thing `age` can mean, and it is expected: the object fills
                # the frame and falls outside the blob size window as the
                # gripper closes in. Retreating then re-approaching regains
                # the view at distance, loses it again at the same range, and
                # bounces until the attempt is abandoned, which is what
                # "retracting after committing" was.
                if aligned:
                    if not fresh:
                        print(f"\n  object out of the wrist view for "
                              f"{target.age:.1f}s but aligned to "
                              f"{err:.1f} mm — grabbing")
                    break

                # --- object-motion detection ---
                #
                # The wrist camera cannot see the object at grab
                # distance, so it alone cannot tell "the object moved"
                # from "I stopped being able to see it". Judging that
                # from the wrist made the arm bounce up and down and
                # then grab empty space.
                #
                # The overhead camera can see it: fixed mount, whole
                # table, calibrated to 2.6 mm. So it supplies the new
                # APPROXIMATE x/y after a disturbance, which is enough
                # to hover over the object again and let the wrist
                # camera re-acquire it properly.
                #
                # z is NOT taken from here. It keeps coming from
                # det-to-segment via seg_z_pump and Target.best_z,
                # exactly as before: the overhead camera maps the
                # table PLANE, so it has no per-object height at all.
                if no_motion:
                    print(f"\n  at the grasp pose "
                          f"(x/y off by {err:.1f} mm, last seen "
                          f"{target.age:.1f}s ago) — grabbing "
                          f"(--no-motion)")
                    break

                if backoffs >= MAX_BACKOFFS:
                    why = (f"moved {target.jump_mm:.0f} mm"
                           if target.disturbed else
                           f"lost for {target.age:.1f}s" if not fresh
                           else f"x/y off by {err:.1f} mm")
                    print(f"\n  giving up after {backoffs} retries "
                          f"({why}) — not grabbing")
                    return

                backoffs += 1
                why = (f"object moved {target.jump_mm:.0f} mm"
                       if target.disturbed else
                       f"lost for {target.age:.1f}s" if not fresh
                       else f"x/y off by {err:.1f} mm")

                # Ask the overhead camera where the object is now. It
                # sees the whole table from a fixed mount, so unlike
                # the wrist camera it still has a view here.
                # Use the fix overhead_watch_pump already holds
                # rather than grabbing and locating again here. Two
                # reasons: the pump's fix tracks THIS object across
                # frames (a fresh lookup only knows "nearest", which
                # a decoy can win), and locate() is blocking OpenCV
                # work -- running it inline on the event loop is what
                # produced the earlier "Deadline exceeded" against the
                # SDK's 1 s keepalive.
                seen = state.get("oh_xy") if state.get("oh_seen") \
                    else None

                # Prefer the point cloud. It is the accurate source
                # -- the same measurement z already comes from -- and
                # it reports a real 3D position rather than a plane
                # homography. The overhead camera is the fallback for
                # when the segmenter has not managed a read (it runs
                # at ~1.3 Hz, so right after a disturbance it often
                # has not yet).
                src = None
                if target.seg_xy_fresh:
                    seen = (target.seg_x, target.seg_y)
                    src = "point cloud"
                elif seen is not None:
                    src = "overhead camera"

                if seen is None:
                    print(f"\n  {why} — neither the point cloud nor "
                          f"the overhead camera can place the "
                          f"{colour} object "
                          f"({backoffs}/{MAX_BACKOFFS})")
                    # Nothing better to go on: rise to survey height
                    # over the last known spot and try the wrist
                    # camera again from there.
                    hover_x, hover_y = target.x, target.y
                else:
                    moved_mm = ((seen[0] - target.x) ** 2
                                + (seen[1] - target.y) ** 2) ** 0.5
                    print(f"\n  {why} — {src} puts it at "
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
                # Rise until the object is actually visible again,
                # rather than to a fixed height. How far the arm has
                # to retreat depends on the object's size and where
                # it ended up, neither of which is known in advance;
                # a hardcoded height is either too low to re-acquire
                # or wastes time climbing past the point it could
                # have seen it. Climb in steps and stop at the first
                # step that gives a clean fix.
                here_now = await gripper_pose_in_world(machine)
                z_try = max(min_z, here_now.z)
                regained = False
                # Loop on the step COUNT, not on z_try: once z_try
                # is clamped to the ceiling, `z_try <= MAX` stays true
                # forever and the arm re-looks at the same height
                # until something kills it.
                steps = int(np.ceil((REACQUIRE_MAX_Z - z_try)
                                    / REACQUIRE_STEP_MM)) + 1
                at_ceiling = False
                for _ in range(max(1, steps)):
                    if regained or at_ceiling:
                        break
                    z_try = min(z_try + REACQUIRE_STEP_MM,
                                REACQUIRE_MAX_Z)
                    at_ceiling = z_try >= REACQUIRE_MAX_Z
                    print(f"    rising to z={z_try:.0f} to look "
                          f"again")
                    await motion.move(
                        component_name=GRIPPER_NAME,
                        destination=PoseInFrame(
                            reference_frame="world",
                            pose=Pose(x=hover_x + GRASP_OFFSET_X,
                                      y=hover_y + GRASP_OFFSET_Y,
                                      z=z_try,
                                      o_x=0.0, o_y=0.0, o_z=-1.0,
                                      # Nothing is held and the
                                      # object is being re-found, so
                                      # the grasp angle does not
                                      # apply here.
                                      theta=DEFAULT_GRASP_THETA),
                        ),
                        extra=move_extra,
                    )
                    await wait_until_stopped(arm)
                    await asyncio.sleep(OVERHEAD_SETTLE_S)

                    # "Full clarity" means the blob detector accepts
                    # it at THIS range: find_blobs' size window is
                    # scaled by how far the camera is from the table,
                    # so a blob that passes here is one the tracker
                    # can actually follow down.
                    scale_here = min(MAX_BLOB_SCALE,
                                     max(1.0, SURVEY_RANGE_MM
                                         / max(z_try, 1.0)))
                    t_re = time.monotonic()
                    while time.monotonic() - t_re < REACQUIRE_LOOK_S:
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
                            o2 = observe(h2, state["depth"], colour,
                                         intr, state["R"], state["T"],
                                         None, drop,
                                         max_scale=scale_here)
                            h3 = nearest(o2, target)
                            if h3 is not None:
                                target.update(h3[0])
                                print(f"    reacquired at z="
                                      f"{z_try:.0f} "
                                      f"({target.x:.1f}, "
                                      f"{target.y:.1f})")
                                regained = True
                                break
                        await asyncio.sleep(0.05)

                if not regained:
                    print(f"    still not visible at z="
                          f"{REACQUIRE_MAX_Z:.0f} — descending on "
                          f"the last known position")

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
                o_x=0.0, o_y=0.0, o_z=-1.0, theta=target.theta)
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
    # Released into an open box, so the carried object's orientation
    # does not matter here; returning the wrist to neutral keeps it
    # away from its rotation limit before the next pick.
    over_box = Pose(x=drop.x, y=drop.y,
                    z=drop.z + DROP_CLEARANCE_MM,
                    o_x=0.0, o_y=0.0, o_z=-1.0,
                    theta=DEFAULT_GRASP_THETA)
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



async def main(argv):
    watch = "--watch" in argv
    dry_run = "--dry-run" in argv
    no_box = "--no-box" in argv
    no_overhead = "--no-overhead" in argv
    slow = "--slow" in argv
    use_sam = "--sam" in argv
    # Object-motion detection is ON when the overhead camera is calibrated:
    # it is what supplies the new approximate x/y after a disturbance. It was
    # disabled while the only sensor was the wrist camera, which is blind at
    # grab distance and made the arm bounce and grab empty space.
    # --no-motion forces the old behaviour: grab at the commanded pose
    # whatever the alignment says.
    no_motion = "--no-motion" in argv
    colour = next((a for a in argv[1:] if not a.startswith("-")), None)

    # Colour is required for the HSV tracker, but optional with --sam: SAM
    # finds objects without being told what colour to look for, which is the
    # whole point of using it.
    if colour is not None and colour not in COLOURS:
        print(f"unknown colour '{colour}'")
        colour = "INVALID"
    if colour == "INVALID" or (colour is None and not use_sam):
        print(f"usage: track_pick.py [--watch|--dry-run] [--slow] [--no-box] "
              f"[--sam] <{'|'.join(sorted(COLOURS))}>")
        print("  --no-motion  grab at the commanded pose without checking "
              "alignment (disables overhead recovery)")
        print("  --sam     use SAM 3 to choose the object and measure the")
        print("            wrist grasp angle; colour then optional")
        print("  --slow    move gently (8 deg/s hints, 25 mm retarget threshold)")
        print("  --no-box  skip the drop-box lookup and exclude nothing")
        print("  --no-overhead  do not use the Lenovo camera for recovery")
        return

    # Load SAM before opening the robot session: the model takes ~5 s to build
    # and there is no reason to hold a connection open through it.
    sam = sam_observe.load() if use_sam else None

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
                 "depth_n": 0, "seg_reads": 0, "seg_err": None,
                 "oh_reads": 0, "oh_err": None, "oh_seen": False,
                 "oh_xy": None}
        # The segmenter task needs the target, which does not exist until
        # acquisition. Hand it a box to read from rather than restarting it.
        target_ref = {"t": None}
        pumps = [
            asyncio.create_task(depth_pump(cam, state)),
            asyncio.create_task(pose_pump(machine, state)),
            asyncio.create_task(seg_z_pump(machine, segmenter, target_ref,
                                           state)),
        ]
        # The overhead camera owns the "has it moved?" decision. Without it
        # there is no disturbance signal at all, which is the safe default:
        # the arm commits to its fix instead of backing off on a guess.
        if overhead is not None and oh_cap is not None:
            pumps.append(asyncio.create_task(
                overhead_watch_pump(overhead, oh_cap, target_ref, state)))
        else:
            print("no overhead watch — object movement will NOT be detected")

        try:
            if not (watch or dry_run):
                # NOTE: this goes through the arm-position-saver switch, which
                # takes no speed argument, so it runs at the module's own
                # speed even under --slow. Only the motion.move calls below
                # honour move_extra. It is a stored pose the arm has driven
                # many times, but it is the one move --slow does not govern.
                if slow:
                    print("  (top-pose runs at the position-saver's own "
                          "speed; --slow governs the tracked motions)")
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

            # --- ACQUIRE ---
            #
            # With --sam, one SAM 3 survey chooses the object and measures the
            # wrist angle to grasp it at. That costs ~2-3 s, which is why it
            # runs ONCE here rather than in the tracking loop below: a rigid
            # object's grasp angle does not change while the camera closes in,
            # so it only has to be measured once. The fast colour tracker then
            # keeps x/y locked through the descent as before.
            t_acq = time.monotonic()
            target = None

            # SAM contributes ONE thing: the wrist angle, plus the colour of
            # whichever object it judged most graspable. It deliberately does
            # NOT supply x/y/z.
            #
            # Position stays on the two validated sources, unchanged: x/y from
            # the 2D blob centroid deprojected through the depth map (the only
            # measurement verified to +/-0.3 mm, at 16 Hz), and z from
            # det-to-segment's point cloud via seg_z_pump. SAM's mask centroid
            # is a fourth, unvalidated estimate of the same quantity, and
            # seeding the target from it would quietly replace a measurement
            # that was checked with one that was not.
            #
            # Keeping the sources separate is also what makes the system
            # robust to any one detector failing: SAM can miss an object
            # entirely (observed) without moving the arm anywhere wrong,
            # because it never had a vote on where the arm goes.
            sam_theta = None
            sam_xy = None
            if sam is not None:
                images, _ = await cam.get_images(filter_source_names=["color"])
                jpeg = next((i for i in images
                             if i.mime_type == CameraMimeType.JPEG), None)
                if jpeg is not None:
                    bgr = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8),
                                       cv2.IMREAD_COLOR)
                    s_rejects = []
                    # In a thread: the survey is ~2.8 s of blocking torch
                    # work, and running it on the event loop starves the
                    # SDK's 1-second keepalive ping, which then reports
                    # "Deadline exceeded" and drops the connection. Same
                    # failure the blocking cv2 capture caused in overhead.py.
                    found, dt = await asyncio.to_thread(
                        sam_observe.survey,
                        bgr, state["depth"], intr, state["R"], state["T"], sam,
                        workspace=WORKSPACE, drop=drop, in_drop_box=in_drop_box,
                        rejects=s_rejects)
                    print(f"  SAM survey: {len(found)} graspable in {dt:.1f}s "
                          f"({len(s_rejects)} rejected)")

                    usable = [o for o in found if o["feasible"]]
                    if colour:
                        usable = [o for o in usable
                                  if sam_observe.colour_of(o, bgr) == colour]
                    for o in usable:
                        th = sam_observe.image_deg_to_world_theta(
                            o["grasp_deg"], state["R"])
                        x, y, z = o["xyz_approx"]
                        print(f"    {o['name']:<12} ~({x:.0f},{y:.0f},{z:.0f}) "
                              f"grasp {o['grasp_deg']:.0f}deg image -> "
                              f"{th:+.0f}deg world, jaws {o['grasp_width']:.0f}px")

                    # The blob tracker follows an object by colour, so SAM's
                    # choice has to be expressible as one.
                    named = [(o, colour or sam_observe.colour_of(o, bgr))
                             for o in usable]
                    named = [(o, c) for o, c in named if c is not None]
                    if len(named) < len(usable):
                        print(f"    {len(usable) - len(named)} object(s) "
                              f"skipped: no trackable colour")

                    if named:
                        pick, colour = named[0]
                        sam_theta = sam_observe.image_deg_to_world_theta(
                            pick["grasp_deg"], state["R"])
                        # Remember WHERE SAM saw it, so the blob detector can
                        # be asked for the same object rather than for any
                        # blob of that colour.
                        sam_xy = pick["xyz_approx"][:2]
                        print(f"  SAM: grasp {pick['name']} ({colour}) at "
                              f"theta={sam_theta:+.1f} deg "
                              f"— position still from blob+depth")
                    else:
                        print("  SAM found nothing graspable")
                        if colour is None:
                            print("  no colour given and SAM found nothing "
                                  "— name a colour, or move the objects "
                                  "into view")
                            return
                        print("  continuing with colour tracking, theta "
                              f"{DEFAULT_GRASP_THETA:+.0f}")

            # Position comes from here and only here, with or without SAM.
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
                if not obs:
                    continue

                if sam_xy is None:
                    # No survey: the largest blob of the named colour, as
                    # this code has always done.
                    xyz, _ = nearest(obs, None)
                else:
                    # A survey chose a specific object. Take the blob at that
                    # position, so the angle and the position describe the
                    # same physical object even when several blocks share a
                    # colour -- on the table or in the bin.
                    hit, gap = nearest_to(obs, sam_xy, SAM_MATCH_MM)
                    if hit is None:
                        print(f"  SAM's object has no matching {colour} blob "
                              f"(nearest {gap:.0f} mm away, limit "
                              f"{SAM_MATCH_MM:.0f}) — the detectors disagree; "
                              f"grasping at theta {DEFAULT_GRASP_THETA:+.0f} "
                              f"instead")
                        sam_theta = None
                        xyz, _ = nearest(obs, None)
                    else:
                        xyz, _ = hit
                        print(f"  blob confirms SAM's object "
                              f"({gap:.1f} mm apart)")

                # xyz is the blob centroid deprojected through the depth map;
                # sam_theta is the only thing SAM contributes.
                target = Target(colour, xyz, sam_theta)
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
            print(f"  theta {target.theta:+.1f} deg from "
                  f"{'SAM' if target.grasp_theta is not None else 'default'}")
            print("  x/y tracked live; z fed asynchronously by "
                  f"{SEGMENTER}")

            if not (watch or dry_run):
                await gripper.open()

            await run_pick(
                machine, arm, gripper, motion, cam, segmenter, target,
                colour, intr, state, drop, min_z, move_extra, retarget_mm,
                overhead, oh_cap, watch=watch, dry_run=dry_run,
                no_motion=no_motion)
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
