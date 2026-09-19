"""Web control interface for the arm.

Runs the perception stack as a service -- depth, camera pose, the point-cloud
segmenter and SAM all refresh continuously in the background -- and serves it
as a web page. Nothing is computed on demand: by the time you click a button,
the grasp angle and 3D geometry are already there.

    python web_gui.py                 # then open http://localhost:8752
    python web_gui.py --no-sam        # skip the 860M-param model
    python web_gui.py --port 9000

The front end is a React app in ui/, built by Vite into static/. Rebuild it
with:

    cd ui && npm run build        # or npm run dev for hot reload on :5173

Server side is http.server from the standard library rather than a web
framework, so this adds no Python dependency to a project that already pins
15. Frames go out as MJPEG (a multipart stream of JPEGs) which the browser
decodes natively in an <img>; state goes out as JSON the page polls. Frames
deliberately do NOT travel through the JSON channel -- base64 in a React state
update several times a second is what makes a page feel slow.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.services.motion import MotionClient
from viam.proto.common import Pose
from viam.media.video import CameraMimeType
from viam.services.vision import VisionClient

from tutorial import connect, goto_saved_pose
from slow_pick import wait_until_stopped, gripper_pose_in_world
from live3d import COLOURS, depth_pump, pose_pump, find_blobs
from seg_pick import find_objects, SEGMENTER
from multi_pick_box import (locate_drop_box, DROP_DETECTORS,
                            BOX_EXCLUSION_X_MM, BOX_EXCLUSION_Y_MM)
from overhead import Overhead, open_lenovo, grab_async, overhead_blobs
from calib_gui import world_to_wrist_px
from multi_pick import WORKSPACE

# How near a colour blob must be to a point-cloud segment for them to be the
# same physical object. The same scale as the tracker's own SEG_MATCH_MM:
# both compare a wrist-derived x/y against a point-cloud centre.
SEG_COLOUR_MATCH_MM = 70.0

# Used only when the point cloud has no segment for an object. The table
# plane is the safe assumption: MIN_GRASP_Z clamps the descent anyway.
# A point-cloud segment older than this is not used for grasp height: the
# object it described may already have been picked.
SEG_FRESH_S = 2.0
DEFAULT_TOP_Z = 40.0
DEFAULT_OBJ_H = 40.0

# Which drop box each object colour is sorted into. The box is found by its
# OWN colour in the overhead view, where both boxes are visible and already
# classified as containers by size.
DROP_FOR = {"orange": "green", "yellow": "blue"}

# How long the wrist must be still before a frame is trusted for geometry.
# SAM's grasp angle and the drop box's rim both come from a single frame, and
# a frame grabbed mid-move is blurred.
STILL_S = 0.4
# How far a point-cloud centre must shift before the scene counts as changed
# and SAM is re-run. Above the segmenter's own jitter, below a real nudge.
SCENE_MOVED_MM = 15.0
# Re-run SAM this often even on a still scene, so a missed change cannot
# leave the angles stale. Measured: one grid-13 pass is ~10 s on this
# machine (MPS, 169 point prompts), so this effectively means "run
# continuously" -- the pass itself is the rate limit, not this number.
SAM_MAX_AGE_S = 3.0
# How often to refresh the wrist panel while a pick owns the camera. Slow on
# purpose: the pick's own tracking loop is the priority for that stream.
WRIST_PAUSED_S = 0.5
# Overhead motion watch. The period is what the USB camera and the blob
# detector sustain; MOTION_MM sits well above the ~2.6 mm calibration
# residual so table jitter is never reported as a move.
MOTION_PERIOD_S = 0.15
MOTION_MM = 20.0
from track_pick import (run_pick, Target, collision_limits, SLOW_EXTRA,
                        SLOW_RETARGET_MM, SAM_MATCH_MM, seg_z_pump)

STATIC_ROOT = Path(__file__).parent / "static"

PORT = 8752
PANEL_W, PANEL_H = 960, 720
CLOUD_W, CLOUD_H = 640, 720
WRIST_W, WRIST_H = 1280, 720
CLICK_SNAP_MM = 60.0

PLOT_X = (150.0, 800.0)
PLOT_Y = (-450.0, 450.0)

C_BLOB = (255, 255, 255)
C_SEG = (120, 255, 120)
C_OVER = (0, 170, 255)
C_SAM = (0, 255, 255)
C_BOX = (255, 220, 0)
C_SEL = (255, 0, 255)

# Everything the render threads and the HTTP handlers share. Guarded by a lock
# because http.server dispatches each request on its own thread while the
# asyncio loop writes from another.
HUB = {
    "overhead": None, "wrist": None, "cloud": None,
    "status": {}, "lock": threading.Lock(),
    "command": None, "log": [],
}


def publish(name, frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        with HUB["lock"]:
            HUB[name] = buf.tobytes()


def log(msg):
    print("  " + msg, flush=True)
    with HUB["lock"]:
        HUB["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
        del HUB["log"][:-40]




class Handler(BaseHTTPRequestHandler):
    # http.server logs every request to stderr, which drowns the robot output.
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]

        if path.startswith("/stream/"):
            return self._stream(path.rsplit("/", 1)[-1])

        if path == "/api/status":
            with HUB["lock"]:
                body = json.dumps(dict(HUB["status"], log=list(HUB["log"])))
            return self._send(200, body.encode(), "application/json")

        # Anything else is the React bundle. Unknown paths fall back to
        # index.html so client-side routing keeps working.
        root = STATIC_ROOT
        rel = path.lstrip("/") or "index.html"
        target = root / rel
        if not target.is_file():
            target = root / "index.html"
        if not target.is_file():
            return self._send(
                503,
                b"UI not built. Run:  cd ui && npm install && npm run build",
                "text/plain")
        ctype = {
            ".html": "text/html", ".js": "text/javascript",
            ".css": "text/css", ".svg": "image/svg+xml",
            ".json": "application/json", ".map": "application/json",
        }.get(target.suffix, "application/octet-stream")
        return self._send(200, target.read_bytes(), ctype)

    def do_POST(self):
        if self.path != "/api/command":
            return self._send(404, b"no", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            cmd = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, b'{"error":"bad json"}', "application/json")
        with HUB["lock"]:
            HUB["command"] = cmd
        return self._send(200, b'{"ok":true}', "application/json")

    def _stream(self, name):
        """MJPEG: an endless multipart response, one JPEG per part."""
        if name not in ("overhead", "wrist", "cloud"):
            return self._send(404, b"no such view", "text/plain")
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                with HUB["lock"]:
                    buf = HUB[name]
                if buf:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     + f"Content-Length: {len(buf)}\r\n\r\n"
                                     .encode() + buf + b"\r\n")
                time.sleep(0.07)
        except (BrokenPipeError, ConnectionResetError):
            pass            # the tab was closed; nothing to clean up


def plot_px(w):
    """World x/y -> pixel in the top-down plot.

    World +x runs away from the base and is drawn upward; world +y runs to the
    side and is drawn rightward, so the plot reads as looking down at the
    table from above the arm.
    """
    fx = (w[1] - PLOT_Y[0]) / (PLOT_Y[1] - PLOT_Y[0])
    fy = 1.0 - (w[0] - PLOT_X[0]) / (PLOT_X[1] - PLOT_X[0])
    return int(fx * CLOUD_W), int(fy * CLOUD_H)


def draw_cloud(segs, blobs, drop, sel, gripper):
    c = np.full((CLOUD_H, CLOUD_W, 3), 22, np.uint8)
    for mm in range(int(PLOT_Y[0]), int(PLOT_Y[1]) + 1, 100):
        x, _ = plot_px((PLOT_X[0], mm))
        cv2.line(c, (x, 0), (x, CLOUD_H), (42, 46, 56), 1)
        if mm % 200 == 0:
            cv2.putText(c, f"y{mm}", (x + 3, CLOUD_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 160, 180), 1)
    for mm in range(int(PLOT_X[0]), int(PLOT_X[1]) + 1, 100):
        _, y = plot_px((mm, PLOT_Y[0]))
        y = min(y, CLOUD_H - 2)   # x=150 lands exactly on the edge
        cv2.line(c, (0, y), (CLOUD_W, y), (42, 46, 56), 1)
        if mm % 200 == 0:
            cv2.putText(c, f"x{mm}", (6, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 160, 180), 1)

    if drop is not None:
        a = plot_px((drop.x - BOX_EXCLUSION_X_MM, drop.y - BOX_EXCLUSION_Y_MM))
        b = plot_px((drop.x + BOX_EXCLUSION_X_MM, drop.y + BOX_EXCLUSION_Y_MM))
        cv2.rectangle(c, a, b, C_BOX, 2)
        cv2.putText(c, "DROP BOX", (min(a[0], b[0]) + 6, min(a[1], b[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_BOX, 1)

    for label, pose, dims in segs:
        if dims is not None:
            a = plot_px((pose.x - dims.x / 2, pose.y - dims.y / 2))
            b = plot_px((pose.x + dims.x / 2, pose.y + dims.y / 2))
            cv2.rectangle(c, a, b, C_SEG, 2)
            cv2.putText(c, f"{label} h{dims.z:.0f}",
                        (min(a[0], b[0]), min(a[1], b[1]) - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_SEG, 1)
        cv2.circle(c, plot_px((pose.x, pose.y)), 4, C_SEG, -1)

    for b in blobs:
        cv2.circle(c, plot_px(b), 5, C_BLOB, 1)

    if gripper is not None:
        q = plot_px((gripper.x, gripper.y))
        cv2.drawMarker(c, q, (90, 90, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(c, "arm", (q[0] + 10, q[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 255), 1)

    if sel is not None:
        cv2.circle(c, plot_px(sel), 14, C_SEL, 2)
    return c


def mark(panel, q, colour, label, r=9):
    if q is None:
        return
    cv2.circle(panel, q, r, colour, 2)
    cv2.line(panel, (q[0] - r - 5, q[1]), (q[0] + r + 5, q[1]), colour, 1)
    cv2.line(panel, (q[0], q[1] - r - 5), (q[0], q[1] + r + 5), colour, 1)
    if label:
        cv2.putText(panel, label, (q[0] + 12, q[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)


def _overhead_all(oh, frame):
    """Every colour's world x/y as the overhead camera sees it right now.

    Restricted to the arm's workspace. The homography is fitted over the
    table, so a blob outside that region -- a reflection, something on the
    floor, the edge of the bin -- projects to an arbitrary world coordinate.
    Measured: spurious 'orange' detections at y = -457..-506 mm, beyond the
    -450 workspace limit, produced motion messages of 130-438 mm on a
    completely still table.
    """
    xlo, xhi = WORKSPACE["x"]
    ylo, yhi = WORKSPACE["y"]
    out = {}
    for name in COLOURS:
        pts = [w for w, _px, _b in oh.locate(frame, name)
               if xlo <= w[0] <= xhi and ylo <= w[1] <= yhi]
        if pts:
            out[name] = pts
    return out


def _cancel_pick(live, log):
    """Stop an in-process pick. Returns True if one was running."""
    t = live.get("task")
    if t is None or t.done():
        return False
    t.cancel()
    live["task"] = None
    log("pick stopped")
    return True


def _seg_for(colour, live):
    """Where to pick a `colour` object, and how tall it is.

    Returns (x, y, top_z, height) or None.

    The point cloud is preferred -- it measures the object's actual top face,
    which the depth map cannot -- but it is NOT required. det-to-segment
    clusters intermittently and often returns one object while the table has
    several, so requiring a segment meant "no yellow object in the point
    cloud" while the colour detector was showing five blobs. The blob's own
    world position comes from the same depth map through the same transform,
    so it is a valid fallback; only the height falls back to a default.
    """
    blobs = live.get("blobs", {}).get(colour, [])
    if not blobs:
        return None

    boxes = [(b.x, b.y) for b in live.get("boxes", {}).values()]
    if live.get("drop") is not None:
        boxes.append((live["drop"].x, live["drop"].y))

    def in_a_box(x, y):
        return any(abs(x - bx) <= BOX_EXCLUSION_X_MM
                   and abs(y - by) <= BOX_EXCLUSION_Y_MM
                   for bx, by in boxes)

    blobs = [(x, y) for x, y in blobs if not in_a_box(x, y)]
    if not blobs:
        return None

    # Only trust a RECENT segment set. live["segs"] holds the last non-empty
    # result so an intermittent read does not flicker the display to zero --
    # but a pick removes an object while its segment lingers, and that stale
    # segment then supplied the next target's grasp height. Measured: two
    # consecutive picks of the same object reported z=35 then z=19, the
    # second being the departed object's segment rather than the new one.
    segs = (live["segs"]
            if time.monotonic() - live.get("seg_at", 0.0) < SEG_FRESH_S
            else [])

    # Nearest point-cloud segment to each blob, if there is one.
    best = None
    for bx, by in blobs:
        seg = None
        seg_d = SEG_COLOUR_MATCH_MM
        for _label, pose, dims in segs:
            if dims is None or in_a_box(pose.x, pose.y):
                continue
            d = ((pose.x - bx) ** 2 + (pose.y - by) ** 2) ** 0.5
            if d < seg_d:
                seg, seg_d = (pose, dims), d
        if seg is not None:
            pose, dims = seg
            cand = (pose.x, pose.y, pose.z + dims.z / 2.0, dims.z, True)
        else:
            cand = (bx, by, DEFAULT_TOP_Z, DEFAULT_OBJ_H, False)
        # Prefer a blob backed by the point cloud.
        if best is None or (cand[4] and not best[4]):
            best = cand
    return best[:4] if best else None


def _sam_theta_for(target, live):
    """SAM's grasp angle for the object at `target`, or None.

    Matched by position so the angle belongs to the SAME block the point
    cloud picked. An unmatched SAM result is discarded rather than used: a
    confident angle from the wrong object rotates the wrist wrongly, which is
    worse than falling back to the neutral default.
    """
    best, best_d = None, SAM_MATCH_MM
    for o in live["sam"]:
        x, y, _z = o["xyz_approx"]
        d = ((x - target.x) ** 2 + (y - target.y) ** 2) ** 0.5
        if d < best_d and o.get("world_theta") is not None:
            best, best_d = o, d
    if best is None:
        return None, "no SAM match — neutral wrist"
    return best["world_theta"], f"SAM, {best_d:.0f} mm away"


# Colours the automatic sort handles, in the order it tries them.
SORT_COLOURS = ("orange", "yellow")
# Give up on a colour after this many consecutive failures, so one
# ungraspable object cannot spin the loop forever.
SORT_MAX_FAILS = 3
# Longest to wait for a fresh measurement after returning to top-pose.
# The segmenter runs at ~1.3 Hz, so this allows several attempts.
SORT_SETTLE_S = 4.0


def _best_target(live):
    """The pickable object with the most confident SAM grasp angle.

    Returns (colour, target, theta, why) or None.

    "Most confident" is SAM's own mask score for the object its angle came
    from. An object SAM never segmented has no angle at all, so it sorts last
    -- it can still be picked, just with a neutral wrist.
    """
    best = None
    for colour in SORT_COLOURS:
        found = _seg_for(colour, live)
        if found is None:
            continue
        fx, fy, ftop, fh = found
        t = Target(colour, (fx, fy, ftop))
        t.update_seg_z(ftop, fh)
        t.update_seg_xy(fx, fy)
        theta, why = _sam_theta_for(t, live)
        score = 0.0
        if theta is not None:
            for o in live.get("sam", []):
                ox, oy, _oz = o["xyz_approx"]
                if ((ox - fx) ** 2 + (oy - fy) ** 2) ** 0.5 < SAM_MATCH_MM:
                    score = max(score, float(o.get("score", 0.0)))
        if best is None or score > best[0]:
            best = (score, colour, t, theta, why)
    if best is None:
        return None
    _score, colour, t, theta, why = best
    return colour, t, theta, why


async def _do_pick(colour, machine, arm, cam, intr, rig, live, log,
                   target=None, theta=None, why=None):
    """One pick: grasp `colour` and drop it in that colour's box."""
    if target is None:
        found = _seg_for(colour, live)
        if found is None:
            log(f"no {colour} object outside the drop boxes — nothing to pick")
            return False
        fx, fy, ftop, fh = found
        target = Target(colour, (fx, fy, ftop))
        target.update_seg_z(ftop, fh)
        target.update_seg_xy(fx, fy)
        theta, why = _sam_theta_for(target, live)
    target.grasp_theta = theta
    log(f"picking {colour} at ({target.x:.0f}, {target.y:.0f}) "
        f"z={target.best_z[0]:.0f} theta={target.theta:+.0f} ({why})")

    want = DROP_FOR.get(colour, "green")
    drop_for = live.get("boxes", {}).get(want) or live["drop"]
    if drop_for is not None:
        log(f"{colour} -> {want} box ({drop_for.x:.0f}, {drop_for.y:.0f})")

    live["paused"] = True
    # seg_z_pump feeds the target's z and its x/y from the point cloud.
    target_ref = {"t": target}
    seg_task = asyncio.create_task(
        seg_z_pump(machine, rig.segmenter, target_ref, rig.state))
    ok = False
    try:
        await run_pick(machine, arm, rig.gripper, rig.motion, cam,
                       rig.segmenter, target, colour, intr, rig.state,
                       drop_for, rig.min_z, SLOW_EXTRA, SLOW_RETARGET_MM,
                       rig.oh, rig.cap)
        log("pick complete")
        ok = True
    except asyncio.CancelledError:
        log("pick stopped")
        raise
    except Exception as exc:                                # noqa: BLE001
        log(f"pick failed: {str(exc)[:120]}")
    finally:
        seg_task.cancel()
        # Back to the survey pose, then re-measure. A pick changes the scene:
        # one object is gone and the rest may have shifted, so SAM's angles
        # are stale -- and they can only be recomputed from top-pose, where
        # the wrist sees the whole table.
        try:
            await goto_saved_pose(machine, "top-pose")
            await wait_until_stopped(arm)
        except Exception as e:                              # noqa: BLE001
            log(f"return to top-pose: {str(e)[:60]}")
        live["paused"] = False
        live["sam_dirty"] = True
        live["still_since"] = 0.0
        # The picked object is gone; its segment and blob must not survive
        # into the next round's measurements.
        live["segs"] = []
        live["seg_at"] = 0.0
        live["blobs"] = {}
        log("back at top-pose — re-measuring")
    return ok


async def _sort_loop(machine, arm, cam, intr, rig, live, log):
    """Clear the table: pick, drop, re-measure, repeat until empty.

    Each round picks the object SAM is most confident about rather than a
    fixed colour order, because a confident angle is the one most likely to
    produce a successful grasp -- and every pick declutters the scene for the
    next measurement.
    """
    log("sorting — clearing the table")
    n, fails = 0, 0
    try:
        while fails < SORT_MAX_FAILS:
            # Wait for a FRESH measurement of the decluttered table, rather
            # than a fixed sleep: a blind delay either picks on stale data or
            # wastes time, and after a pick the previous segments have been
            # cleared so there is a definite thing to wait for.
            t0 = time.monotonic()
            while time.monotonic() - t0 < SORT_SETTLE_S:
                if (live.get("blobs")
                        and time.monotonic() - live.get("seg_at", 0.0)
                        < SEG_FRESH_S):
                    break
                await asyncio.sleep(0.1)
            choice = _best_target(live)
            if choice is None:
                log(f"table clear — {n} object(s) sorted")
                return
            colour, target, theta, why = choice
            if await _do_pick(colour, machine, arm, cam, intr, rig, live, log,
                              target, theta, why):
                n += 1
                fails = 0
            else:
                fails += 1
                log(f"failed ({fails}/{SORT_MAX_FAILS})")
        log(f"stopping after {SORT_MAX_FAILS} failures — {n} sorted")
    except asyncio.CancelledError:
        log(f"sort stopped — {n} object(s) sorted")
        raise


async def service(argv):
    use_sam = "--no-sam" not in argv
    sam = None
    if use_sam:
        import sam_observe
        log("loading SAM 3 ...")
        sam = sam_observe.load()
        log("SAM ready")

    oh = Overhead.load()
    log(f"overhead {'calibrated' if oh.ready else 'NOT CALIBRATED'}")
    cap = open_lenovo(None)
    if cap is None:
        log("overhead camera failed to open")
        return

    try:
        async with await connect() as machine:
            cam = Camera.from_robot(machine, "cam")
            arm = Arm.from_robot(machine, "arm")
            segmenter = VisionClient.from_robot(machine, SEGMENTER)
            gripper = Gripper.from_robot(machine, "gripper")
            motion = MotionClient.from_robot(machine, "builtin")
            intr = (await cam.get_properties()).intrinsic_parameters
            # Once, at startup -- not per pick. This is a fixed property of
            # the workspace, and probing it on every pick was part of the
            # cold-start cost that made each button press slow.
            min_z, _why = await collision_limits(machine)
            log(f"grasp floor z={min_z:.1f}")

            state = {"depth": None, "R": None, "T": None, "stop": False,
                     "depth_n": 0, "frame": None}
            live = {"segs": [], "seg_at": 0.0, "sam": [], "sam_at": 0.0,
                    "drop": None, "sel": None, "task": None,
                    "blobs": {}, "blobs_at": 0.0,
                    # Event-driven recompute: SAM re-runs when the scene
                    # changes, not on a timer. sam_scene is the object layout
                    # it last ran on; sam_dirty forces a pass regardless.
                    "sam_scene": [], "sam_dirty": True, "sam_blobs": [],
                    "drop_at": 0.0, "still_since": 0.0,
                    "motion_xy": {}, "motion_at": 0.0,
                    "sam_rejects": [], "blob_px": [], "sam_seeds": 0,
                    "boxes": {},
                    # While a pick runs, this service must stop touching the
                    # camera: the machine only serves one client well.
                    "paused": False}

            # Everything a pick needs, built ONCE. A pick is an action on
            # this, not a program that assembles its own world: the whole
            # cost the old subprocess paid per click -- loading SAM,
            # re-locating the drop box, opening a second session -- was the
            # cost of not having somewhere to keep these.
            rig = SimpleNamespace(gripper=gripper, motion=motion,
                                  segmenter=segmenter, state=state,
                                  min_z=min_z, cap=cap, oh=oh)

            async def seg_loop():
                # Intermittent by nature: consecutive reads of a stationary
                # scene alternate between finding the object and finding
                # nothing. Hold the last non-empty result and report its age
                # rather than flickering to zero -- an empty read is not
                # evidence the object left.
                while not state["stop"]:
                    if live["paused"]:
                        await asyncio.sleep(0.2)
                        continue
                    try:
                        got = await find_objects(machine, segmenter)
                        if got:
                            live["segs"] = got
                            live["seg_at"] = time.monotonic()
                    except Exception as e:                  # noqa: BLE001
                        log(f"segmenter: {str(e)[:60]}")
                    await asyncio.sleep(0.05)

            async def still_enough():
                """True when the wrist has been stationary long enough.

                SAM and the drop-box finder both measure GEOMETRY from a
                single frame, so a frame taken mid-move is motion-blurred and
                its angle is wrong. Waiting costs nothing: neither answer can
                change while the arm is the only thing moving.
                """
                if await arm.is_moving():
                    live["still_since"] = 0.0
                    return False
                now = time.monotonic()
                if not live["still_since"]:
                    live["still_since"] = now
                return now - live["still_since"] >= STILL_S

            def scene_moved():
                # Colour blobs first: det-to-segment often reports one object
                # when the table has several, so a layout change is invisible
                # to the point cloud alone.
                now_b = sorted((round(x / 10.0), round(y / 10.0))
                               for pts in live.get("blobs", {}).values()
                               for x, y in pts)
                if now_b != live.get("sam_blobs", []):
                    return True
                """Has anything on the table shifted since the last SAM pass?

                Compares the point cloud's object centres against the set SAM
                last ran on. This is what makes a recompute EVENT-DRIVEN: SAM
                re-runs because the scene changed, not because a timer fired.
                A rigid object's grasp angle cannot change while nothing
                touches it, so re-running on a static scene is pure waste --
                and it is 2-3 s of GPU per pass.
                """
                now = [(p_.x, p_.y) for _l, p_, d in live["segs"] if d]
                was = live["sam_scene"]
                if len(now) != len(was):
                    return True
                for (x, y) in now:
                    if all(((x - a) ** 2 + (y - b) ** 2) ** 0.5 > SCENE_MOVED_MM
                           for a, b in was):
                        return True
                return False

            async def sam_loop():
                # SAM is loaded ONCE, at startup, and stays resident. This
                # loop only ever calls survey() on the warm model -- there is
                # no reload path here at all.
                #
                # It runs when the scene changes and the wrist is still,
                # rather than continuously: a pass costs 2-3 s of GPU, and
                # repeating it on an unchanged scene produces the same
                # numbers at the cost of everything else the loop could do.
                import sam_observe
                while not state["stop"]:
                    if live["paused"] or sam is None \
                            or state["frame"] is None or state["R"] is None:
                        await asyncio.sleep(0.25)
                        continue
                    stale = time.monotonic() - live["sam_at"] > SAM_MAX_AGE_S
                    if not (live["sam_dirty"] or scene_moved() or stale):
                        await asyncio.sleep(0.05)
                        continue
                    if not await still_enough():
                        await asyncio.sleep(0.05)
                        continue
                    try:
                        R, T = state["R"], state["T"]
                        scene = [(p_.x, p_.y)
                                 for _l, p_, d in live["segs"] if d]
                        rej = []
                        seeds = list(live["blob_px"])
                        found, _ = await asyncio.to_thread(
                            sam_observe.survey, state["frame"], state["depth"],
                            intr, R, T, sam, workspace=WORKSPACE,
                            rejects=rej,
                            extra_points=seeds)
                        for o in found:
                            # Named world_theta, not theta: this is the wrist
                            # angle in WORLD terms, already converted from
                            # SAM's image-plane grasp_deg. The dict also
                            # carries image-plane fields, and picking the
                            # wrong one rotates the wrist wrongly.
                            o["world_theta"] = \
                                sam_observe.image_deg_to_world_theta(
                                    o["grasp_deg"], R)
                        live["sam"] = found
                        live["sam_at"] = time.monotonic()
                        live["sam_scene"] = scene
                        live["sam_dirty"] = False
                        live["sam_blobs"] = sorted(
                            (round(x / 10.0), round(y / 10.0))
                            for pts in live.get("blobs", {}).values()
                            for x, y in pts)
                        live["sam_rejects"] = [
                            (xyz, why) for xyz, why in rej][:40]
                        live["sam_seeds"] = len(seeds)
                        # Deliberately NOT logged. A pass finishes every few
                        # seconds and the count is already on the status bar
                        # with its age; printing it only pushed the events
                        # that matter -- motion, picks, errors -- off screen.
                    except Exception as e:                  # noqa: BLE001
                        log(f"sam: {str(e)[:60]}")
                    # Yield only. A pass costs seconds; an extra idle sleep
                    # here is pure added latency on the next one.
                    await asyncio.sleep(0)

            async def find_boxes():
                """Locate every drop box with its own colour detector."""
                out = {}
                for name in DROP_DETECTORS:
                    # Breathe between detectors: two back-to-back 5-frame
                    # lookups starved the keepalive and dropped the channel.
                    await asyncio.sleep(0.3)
                    try:
                        pose = await locate_drop_box(machine, cam, which=name,
                                                     frames=3)
                    except Exception as e:                  # noqa: BLE001
                        log(f"drop box {name}: {str(e)[:50]}")
                        continue
                    if pose is not None:
                        out[name] = pose
                        log(f"drop box {name} ({pose.x:.0f}, {pose.y:.0f})")
                return out

            async def box_loop():
                """Locate the drop box once, then leave it alone.

                Movement tracking is deliberately OFF. The box does not move
                on its own, and re-measuring it cost more than it was worth:
                a reading taken from anywhere but the survey pose sees only
                part of the rim and reports a box that "moved" when it had
                not -- which then drove the arm back to top-pose to check.

                To re-measure after actually moving the box, press the box
                button in the UI ("box" command), which runs the same
                measurement on demand.
                """
                while not state["stop"]:
                    if live["drop"] is not None:
                        return              # found it; nothing more to do
                    if (live["paused"] or live["task"] is not None
                            or not await still_enough()):
                        await asyncio.sleep(0.5)
                        continue
                    try:
                        seen = await locate_drop_box(machine, cam)
                    except Exception as e:                  # noqa: BLE001
                        log(f"drop box: {str(e)[:60]}")
                        await asyncio.sleep(1.0)
                        continue
                    live["drop_at"] = time.monotonic()
                    if seen is None:
                        await asyncio.sleep(1.0)
                        continue
                    live["drop"] = seen
                    boxes = await find_boxes()
                    boxes.setdefault("green", seen)
                    live["boxes"] = boxes
                    return
            # depth_pump and pose_pump poll the camera unconditionally, so
            # they are stopped and restarted around a pick rather than being
            # taught to pause; live["cam_pumps"] holds them for that.
            live["cam_pumps"] = [asyncio.create_task(depth_pump(cam, state)),
                                 asyncio.create_task(pose_pump(machine, state))]
            pumps = live["cam_pumps"] + [
                asyncio.create_task(seg_loop()),
                asyncio.create_task(sam_loop()),
                asyncio.create_task(box_loop())]
            try:
                log("moving to top-pose")
                await goto_saved_pose(machine, "top-pose")
                await wait_until_stopped(arm)
                t0 = time.monotonic()
                while time.monotonic() - t0 < 10:
                    if state["depth"] is not None and state["R"] is not None:
                        break
                    await asyncio.sleep(0.01)
                # The drop box is NOT located here any more: box_loop owns
                # it, finds it within its first cycle, and keeps re-checking
                # it. Doing it here as well would measure the same box twice
                # at startup.

                await render_loop(machine, cam, arm, intr, oh, cap,
                                  state, live, sam is not None, rig)
            finally:
                state["stop"] = True
                if live["task"] is not None:
                    live["task"].cancel()
                done, pending = await asyncio.wait(pumps, timeout=1.5)
                for t in pending:
                    t.cancel()
    finally:
        cap.release()


async def resume(cam, machine, state, live):
    """Take the camera back after a pick finishes.

    The pumps were cancelled rather than paused, so they are recreated here.
    Depth and pose must be live again before the render loop touches the
    camera, or the first frames come back empty.
    """
    if not live.get("paused"):
        return
    live["cam_pumps"] = [asyncio.create_task(depth_pump(cam, state)),
                         asyncio.create_task(pose_pump(machine, state))]
    live["paused"] = False
    log("perception resumed")


async def render_loop(machine, cam, arm, intr, oh, cap, state, live, has_sam, rig):
    osc = wsc = (1.0, 1.0)
    while True:
        # One bad frame must not take the service down. A renderer is
        # the least important thing here -- it draws what the loops
        # already computed -- but before this, any exception inside it
        # killed the whole process: a single missing dict key ended the
        # session, camera streams and all, with the arm left where it
        # stood. Log it, drop the frame, keep serving.
        try:
            if live["paused"]:
                # A pick owns the robot camera. The overhead camera is a plain USB
                # device with no such contention, so it keeps streaming -- which
                # is also the view that still shows the arm working.
                raw = await grab_async(cap)
                if raw is not None:
                    publish("overhead", cv2.resize(raw, (PANEL_W, PANEL_H)))
                # Keep the wrist panel live too. It used to freeze on its
                # last pre-pick frame for the WHOLE pick -- exactly when
                # watching the approach matters most.
                #
                # This needs its own colour read: depth_pump fetches only the
                # depth source, and state["frame"] is written by the unpaused
                # path below, so during a pick it is stale by definition.
                # Rate-limited, because run_pick is reading the same camera
                # and this must stay a bystander.
                now = time.monotonic()
                if now - live.get("wrist_pub_at", 0.0) > WRIST_PAUSED_S:
                    live["wrist_pub_at"] = now
                    try:
                        imgs, _ = await cam.get_images(
                            filter_source_names=["color"])
                        jp = next((i for i in imgs
                                   if i.mime_type == CameraMimeType.JPEG),
                                  None)
                        if jp is not None:
                            fr = cv2.imdecode(
                                np.frombuffer(jp.data, np.uint8),
                                cv2.IMREAD_COLOR)
                            state["frame"] = fr
                            publish("wrist",
                                    cv2.resize(fr, (PANEL_W, PANEL_H)))
                    except Exception:                       # noqa: BLE001
                        pass        # the pick owns this camera; never fight it
                with HUB["lock"]:
                    HUB["status"] = dict(HUB.get("status", {}), busy=True,
                                         paused=True)
                    cmd, HUB["command"] = HUB["command"], None
                if cmd and cmd.get("action") in ("stop", "clear"):
                    if cmd["action"] == "stop":
                        _cancel_pick(live, log)
                        await arm.stop()
                        log("arm stopped")
                        await resume(cam, machine, state, live)
                    else:
                        live["sel"] = None
                await asyncio.sleep(0.05)
                continue

            R, T, depth = state["R"], state["T"], state["depth"]
            if R is None or depth is None:
                await asyncio.sleep(0.05)
                continue

            raw = await grab_async(cap)
            if raw is None:
                break
            ov = cv2.resize(raw, (PANEL_W, PANEL_H))
            osc = (PANEL_W / raw.shape[1], PANEL_H / raw.shape[0])
            ohsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
            oh_world = []
            for name in COLOURS:
                for cx, cy, bx, by, bw, bh in overhead_blobs(ohsv, name):
                    w = oh.to_world(cx, cy) if oh.ready else None
                    if w:
                        oh_world.append(w)
                    cv2.rectangle(ov, (int(bx*osc[0]), int(by*osc[1])),
                                  (int((bx+bw)*osc[0]), int((by+bh)*osc[1])),
                                  COLOURS[name][2], 2)

            images, _ = await cam.get_images(filter_source_names=["color"])
            jpeg = next((i for i in images
                         if i.mime_type == CameraMimeType.JPEG), None)
            if jpeg is None:
                continue
            full = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8),
                                cv2.IMREAD_COLOR)
            state["frame"] = full
            wr = cv2.resize(full, (PANEL_W, PANEL_H))
            wsc = (PANEL_W / WRIST_W, PANEL_H / WRIST_H)

            whsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
            blob_world = []
            by_colour = {}
            blob_px = []
            for name in COLOURS:
                for cx, cy, bx, by, bw, bh in find_blobs(whsv, name):
                    patch = depth[max(0, cy-6):cy+7, max(0, cx-6):cx+7]
                    good = patch[patch > 0]
                    if good.size < 20:
                        continue
                    z = float(np.percentile(good, 25))
                    c = np.array([(cx-intr.center_x_px)*z/intr.focal_x_px,
                                  (cy-intr.center_y_px)*z/intr.focal_y_px, z])
                    w = R @ c + T
                    if not (WORKSPACE["z"][0] < w[2] < WORKSPACE["z"][1]):
                        continue
                    blob_world.append((float(w[0]), float(w[1])))
                    by_colour.setdefault(name, []).append(
                        (float(w[0]), float(w[1])))
                    # The PIXEL of every colour object we can see, for SAM to
                    # prompt directly. A 13x13 grid steps 74 px and misses
                    # anything narrower, which is why the orange block had no
                    # grasp axis while the power strip did.
                    blob_px.append((int(cx), int(cy)))
                    cv2.rectangle(wr, (int(bx*wsc[0]), int(by*wsc[1])),
                                  (int((bx+bw)*wsc[0]), int((by+bh)*wsc[1])),
                                  COLOURS[name][2], 2)

            def wpx(w):
                p = world_to_wrist_px(w, intr, R, T)
                if p is None:
                    return None
                q = (int(p[0]*wsc[0]), int(p[1]*wsc[1]))
                return q if 0 <= q[0] < PANEL_W and 0 <= q[1] < PANEL_H else None

            def opx(w):
                if not oh.ready:
                    return None
                v = np.linalg.inv(oh.H) @ np.array([w[0], w[1], 1.0])
                if abs(v[2]) < 1e-9:
                    return None
                q = (int(v[0]/v[2]*osc[0]), int(v[1]/v[2]*osc[1]))
                return q if 0 <= q[0] < PANEL_W and 0 <= q[1] < PANEL_H else None

            for w in blob_world:
                mark(wr, wpx(w), C_BLOB, f"blob {w[0]:.0f},{w[1]:.0f}")
                mark(ov, opx(w), C_BLOB, "blob")
            for label, pose, dims in live["segs"]:
                w = (pose.x, pose.y)
                h = f" h{dims.z:.0f}" if dims else ""
                mark(wr, wpx(w), C_SEG, f"{label} {w[0]:.0f},{w[1]:.0f}{h}")
                mark(ov, opx(w), C_SEG, f"cloud{h}")
            for w in oh_world:
                mark(wr, wpx(w), C_OVER, "overhead")
            if live["drop"] is not None:
                d = live["drop"]
                for panel, q in ((wr, wpx((d.x, d.y))), (ov, opx((d.x, d.y)))):
                    if q:
                        cv2.circle(panel, q, 17, C_BOX, 3)
                        cv2.putText(panel, "DROP BOX", (q[0]+22, q[1]+5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_BOX, 2)
            for o in live["sam"]:
                cx, cy = o["centroid"]
                a = np.deg2rad(o["grasp_deg"])
                L = 75
                cv2.line(wr, (int((cx-np.cos(a)*L)*wsc[0]),
                              int((cy-np.sin(a)*L)*wsc[1])),
                         (int((cx+np.cos(a)*L)*wsc[0]),
                          int((cy+np.sin(a)*L)*wsc[1])), C_SAM, 3)
                # world_theta, matching what sam_loop writes and what a pick
                # reads. .get(), not [..]: an overlay must never be the thing
                # that takes the service down, and a SAM dict that predates the
                # conversion simply has no angle to show yet.
                th = o.get("world_theta")
                cv2.putText(wr,
                            f"{o['name']} th{th:+.0f}" if th is not None
                            else f"{o['name']} th?",
                            (int(cx*wsc[0])+10, int(cy*wsc[1])+22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_SAM, 2)
                mark(ov, opx(o["xyz_approx"][:2]), C_SAM, "SAM")
            if live["sel"] is not None:
                for panel, q in ((wr, wpx(live["sel"])), (ov, opx(live["sel"]))):
                    if q:
                        cv2.circle(panel, q, 22, C_SEL, 3)

            g = await gripper_pose_in_world(machine)
            publish("overhead", ov)
            publish("wrist", wr)
            # The wrist detector's colour verdicts, keyed by colour. The point
            # cloud has accurate positions but its labels are not colour names,
            # so a colour pick needs both: the cloud for WHERE, this for WHICH.
            live["blobs"] = by_colour
            live["blobs_at"] = time.monotonic()
            # Pixel seeds for SAM's next pass, so an object smaller than the
            # grid spacing still gets prompted.
            live["blob_px"] = blob_px

            publish("cloud", draw_cloud(live["segs"], blob_world, live["drop"],
                                        live["sel"], g))

            now = time.monotonic()
            with HUB["lock"]:
                HUB["status"] = {
                    "gripper": f"{g.x:.0f}, {g.y:.0f}, {g.z:.0f}  θ{g.theta:+.0f}",
                    "blob": len(blob_world),
                    "seg": len(live["segs"]),
                    "seg_age": f"{now - live['seg_at']:.1f}" if live["seg_at"] else "-",
                    "overhead": len(oh_world),
                    "sam": len(live["sam"]),
                    "sam_age": f"{now - live['sam_at']:.1f}" if live["sam_at"] else "-",
                "sam_rejects": [f"{w}" for _xyz, w in live["sam_rejects"]][:12],
                "sam_seeds": live.get("sam_seeds", 0),
                "blob_px": len(live.get("blob_px", [])),
                    "busy": (live["task"] is not None
                                     and not live["task"].done()),
                    "selected": (f"{live['sel'][0]:.0f}, {live['sel'][1]:.0f}"
                                 if live["sel"] else None),
                }
                cmd, HUB["command"] = HUB["command"], None

            if cmd:
                await handle(cmd, machine, cam, arm, intr, oh, depth, R, T,
                             osc, wsc, live, blob_world, oh_world, has_sam, rig)
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:                        # noqa: BLE001
            now = time.monotonic()
            if now - live.get('render_err_at', 0.0) > 5.0:
                live['render_err_at'] = now
                log(f'render: {type(exc).__name__}: {str(exc)[:80]}')
            await asyncio.sleep(0.1)


async def handle(cmd, machine, cam, arm, intr, oh, depth, R, T,
                 osc, wsc, live, blob_world, oh_world, has_sam, rig):
    """Act on one command from the UI.

    `rig` carries the long-lived robot handles the service built once at
    startup -- gripper, motion, segmenter, the shared pump state, the grasp
    floor and the overhead capture. A pick uses these directly instead of
    constructing anything, which is the whole point of running as a service.
    """
    act = cmd.get("action")

    if act == "select":
        # The click arrives as a fraction of the panel, because the browser
        # lays the image out responsively and its pixel coordinates mean
        # nothing here.
        fx, fy = float(cmd.get("x", 0)), float(cmd.get("y", 0))
        wsel = None
        if cmd.get("panel") == "overhead" and oh.ready:
            wsel = oh.to_world(fx * PANEL_W / osc[0], fy * PANEL_H / osc[1])
        elif cmd.get("panel") == "wrist":
            px, py = int(fx * WRIST_W), int(fy * WRIST_H)
            patch = depth[max(0, py-6):py+7, max(0, px-6):px+7]
            good = patch[patch > 0]
            if good.size >= 20:
                z = float(np.percentile(good, 25))
                c = np.array([(px-intr.center_x_px)*z/intr.focal_x_px,
                              (py-intr.center_y_px)*z/intr.focal_y_px, z])
                v = R @ c + T
                wsel = (float(v[0]), float(v[1]))
        if wsel is None:
            log("no depth at that point")
            return
        cands = ([("blob", b) for b in blob_world]
                 + [("cloud", (p.x, p.y)) for _, p, _ in live["segs"]]
                 + [("overhead", w) for w in oh_world])
        if cands:
            src, best = min(cands, key=lambda c: (c[1][0]-wsel[0])**2
                            + (c[1][1]-wsel[1])**2)
            d = ((best[0]-wsel[0])**2 + (best[1]-wsel[1])**2) ** 0.5
            if d <= CLICK_SNAP_MM:
                live["sel"] = best
                log(f"selected {src} ({best[0]:.0f}, {best[1]:.0f})")
                return
        live["sel"] = wsel
        log(f"selected empty table ({wsel[0]:.0f}, {wsel[1]:.0f})")

    elif act == "clear":
        live["sel"] = None
        log("selection cleared")

    elif act == "stop":
        _cancel_pick(live, log)
        await arm.stop()
        log("arm stopped")

    elif act == "home":
        log("moving to top-pose")
        await goto_saved_pose(machine, "top-pose")
        await wait_until_stopped(arm)
        log("at top-pose")

    elif act == "box":
        live["drop"] = await locate_drop_box(machine, cam)
        log(f"drop box ({live['drop'].x:.0f}, {live['drop'].y:.0f})"
            if live["drop"] else "drop box not found")

    elif act == "pick":
        colour = cmd.get("colour")
        if colour not in COLOURS:
            log(f"unknown colour {colour!r}")
            return
        if live["task"] is not None and not live["task"].done():
            log("a pick is already running")
            return
        live["task"] = asyncio.create_task(
            _do_pick(colour, machine, arm, cam, intr, rig, live, log))

    elif act == "sort":
        # Clear the table automatically: repeatedly pick whichever object SAM
        # is most confident about, drop it in its colour's box, return to
        # top-pose, re-measure, repeat until nothing is left.
        if live["task"] is not None and not live["task"].done():
            log("already running")
            return
        live["task"] = asyncio.create_task(
            _sort_loop(machine, arm, cam, intr, rig, live, log))



def main(argv):
    port = PORT
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"\n  arm control:  http://localhost:{port}\n", flush=True)
    try:
        asyncio.run(service(argv))
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main(sys.argv)
