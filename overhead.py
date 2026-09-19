"""The second (Lenovo) camera: a fixed overhead view of the whole workspace.

Why this exists
---------------
The wrist camera goes blind during a descent. Widening the blob-size limits
(see live3d.find_blobs max_scale) keeps the object tracked much closer in, but
the wrist view is still fundamentally narrow at grasp range and sees nothing at
all once the gripper is right on top of the object.

Backing off to re-look did not work: it regains the view, re-descends, and
loses it again at the same distance.

This camera does not move, so its view never degrades as the arm approaches. It
cannot grasp-align on its own — it has no depth, and its accuracy is well below
the wrist camera's — but it can always answer "roughly where is the object
now?", which is what a recovery needs.

Coordinates
-----------
Mounted looking down at the table, hand-measured relative to the arm:

    ~510 mm above the table
    ~150 mm forward of the arm  (+x)
    ~100 mm to the right        (-y)

That approximation only bootstraps the mapping. The real mapping is solved by
calibrate(), which matches objects this camera sees against the same objects
located by the wrist camera from top-pose, then fits pixel -> world directly.
Run `python overhead.py --calibrate` and it is saved to overhead_calib.json.

A homography is the right model here: every object of interest sits on one flat
table, and a homography is the exact mapping between a plane and its image
under a pinhole camera. It absorbs the mounting position, tilt and focal length
together, so none of those has to be measured individually. It is only valid
for points on that plane — objects of different heights project slightly off,
which is acceptable for "hover near here" and is why this never sets the grasp.

    python overhead.py              # BOTH cameras side by side
    python overhead.py --calibrate  # solve the mapping against top-pose
    python cams.py                  # all cameras side by side, to set which is which
    python overhead.py --index 0    # force a device if detection picks wrong
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.media.video import CameraMimeType

from tutorial import connect, goto_saved_pose, _decode_depth
from live3d import COLOURS, find_blobs, depth_pump, pose_pump

CALIB_PATH = Path(__file__).with_name("overhead_calib.json")

# Hand-measured mounting pose, relative to the arm. Used only to sanity-check a
# solved calibration and as a fallback scale before one exists.
MOUNT_HEIGHT_MM = 510.0
MOUNT_FORWARD_MM = 150.0
MOUNT_RIGHT_MM = -100.0

# The Lenovo is 1920x1080. Work at half size: the blob limits in live3d are
# tuned for a 1280x720 wrist frame, and downscaling keeps blobs in a similar
# pixel range as well as making every frame cheaper.
WORK_W, WORK_H = 960, 540

# This camera sees the whole room, not just the table. Objects here are much
# smaller in pixels than in the wrist view, and the room contains large
# coloured distractions (the green box, clothing, furniture).
OH_MIN_AREA, OH_MAX_AREA = 150, 40000
OH_MIN_SIDE, OH_MAX_SIDE = 10, 260

# Only this fraction of the frame, measured from the top, is off-table room:
# people, laptops, the wall. Skin tone in particular reads as orange in HSV and
# produced two false "orange objects" from a hand and forearm on the first
# live frame. Everything above this line is ignored.
#
# This is a crude crop rather than a real table mask, which is the right level
# of effort: the mapping is only ever used to hover towards a lost object, and
# a calibrated homography already rejects anything landing outside the
# workspace bounds. Expressed as a fraction so it survives a resolution change.
ROI_TOP_FRAC = 0.42


CAMERA_FILE = Path(__file__).with_name("overhead_device.json")

# Substring that identifies the overhead webcam by its macOS device name.
# Matched case-insensitively against AVFoundation's device list.
CAMERA_NAME = "lenovo"

# Never open this one. The MacBook's built-in camera faces the room, not the
# workspace, and selecting it silently would look like a broken detector.
EXCLUDE_NAME = "macbook"


def open_lenovo(index=None):
    """Open the overhead camera at the index recorded in overhead_device.json.

    No auto-detection. Identifying it programmatically was tried three ways
    and each failed: a fixed index breaks on replug, a brightness heuristic
    chose the laptop camera once the room lights came up, and AVFoundation's
    device order turned out not to match OpenCV's index order. Looking at the
    two feeds settles it in a second, so:

        python cams.py            # side by side
        python cams.py --set 0    # whichever one shows the table
    """
    if index is None:
        if not CAMERA_FILE.exists():
            print("  no overhead camera set — run: python cams.py --set N")
            print("  (python cams.py shows all cameras side by side)")
            return None
        try:
            index = json.loads(CAMERA_FILE.read_text())["index"]
        except Exception:                             # noqa: BLE001
            print(f"  {CAMERA_FILE.name} is unreadable — "
                  f"run: python cams.py --set N")
            return None

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"  camera index {index} would not open — "
              f"run: python cams.py --set N")
        return None
    print(f"  overhead camera: index {index}")
    return cap


def grab(cap):
    """One frame from the webcam, downscaled to the working size.

    The capture keeps an internal buffer, so the newest frame is not
    necessarily the next one read. During calibration the arm has just moved
    and settled, and a stale frame would carry the arm mid-motion across the
    object, so flush before taking the one that counts.

    BLOCKING: measured at ~200 ms per call. Never call this directly from a
    coroutine — see grab_async.
    """
    for _ in range(4):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        return None
    return cv2.resize(frame, (WORK_W, WORK_H))


async def grab_async(cap):
    """grab() on a worker thread, so the event loop keeps running.

    This matters more than it looks. cv2 frame capture blocks for ~200 ms, and
    calling it straight from the calibration loop starved the event loop badly
    enough that the Viam connection's keepalives never went out — the gRPC
    channel dropped before the arm had even moved, and the run hung with no
    error beyond "channel closed".
    """
    return await asyncio.to_thread(grab, cap)


def overhead_blobs(hsv, colour):
    """Colour blobs in the overhead view, with its own size limits.

    live3d.find_blobs is calibrated for the wrist camera at survey range. This
    camera is further away and wider, so its plausible sizes are different.
    """
    lo, hi, _ = COLOURS[colour]
    mask = cv2.inRange(hsv, lo, hi)
    # Blank the off-table part of the frame before looking for components, so
    # a hand resting at the edge cannot merge with a real object.
    mask[:int(mask.shape[0] * ROI_TOP_FRAC), :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if not (OH_MIN_AREA < area < OH_MAX_AREA):
            continue
        if not (OH_MIN_SIDE < w < OH_MAX_SIDE
                and OH_MIN_SIDE < h < OH_MAX_SIDE):
            continue
        out.append((int(centroids[k][0]), int(centroids[k][1]), x, y, w, h))
    return out


class Overhead:
    """Pixel -> world mapping for the fixed overhead camera."""

    def __init__(self, H=None):
        self.H = H          # 3x3 homography, table plane -> world x/y

    @property
    def ready(self):
        return self.H is not None

    @classmethod
    def load(cls):
        if not CALIB_PATH.exists():
            return cls(None)
        data = json.loads(CALIB_PATH.read_text())
        return cls(np.array(data["H"], dtype=np.float64))

    def save(self, residual, points):
        CALIB_PATH.write_text(json.dumps({
            "H": self.H.tolist(),
            "residual_mm": residual,
            "points": points,
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))

    def to_world(self, px, py):
        """Map a pixel to world x/y on the table plane."""
        if not self.ready:
            return None
        v = self.H @ np.array([px, py, 1.0])
        if abs(v[2]) < 1e-9:
            return None
        return float(v[0] / v[2]), float(v[1] / v[2])

    def locate(self, frame, colour):
        """World x/y of every `colour` object the overhead camera can see."""
        if not self.ready:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out = []
        for cx, cy, bx, by, bw, bh in overhead_blobs(hsv, colour):
            w = self.to_world(cx, cy)
            if w is not None:
                out.append((w, (cx, cy), (bx, by, bw, bh)))
        return out


async def wrist_observations(machine, cam, intr, state, colour):
    """World positions of `colour` as seen by the WRIST camera right now.

    This is the reference the overhead camera is calibrated against, so it
    deliberately uses the wrist pipeline — which is the accurate one.
    """
    images, _ = await cam.get_images(filter_source_names=["color"])
    jpeg = next((i for i in images if i.mime_type == CameraMimeType.JPEG), None)
    if jpeg is None:
        return []
    bgr = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8), cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    depth, R, T = state["depth"], state["R"], state["T"]
    out = []
    for cx, cy, bx, by, bw, bh in find_blobs(hsv, colour):
        patch = depth[max(0, cy - 6):cy + 7, max(0, cx - 6):cx + 7]
        valid = patch[patch > 0]
        if valid.size < 20:
            continue
        z = float(np.percentile(valid, 25))
        cam_xyz = np.array([
            (cx - intr.center_x_px) * z / intr.focal_x_px,
            (cy - intr.center_y_px) * z / intr.focal_y_px,
            z,
        ])
        w = R @ cam_xyz + T
        out.append((float(w[0]), float(w[1]), float(w[2])))
    return out


async def calibrate(colour="orange", want=6, index=None):
    """Solve pixel -> world by moving one object and watching both cameras.

    Procedure: the arm sits at top-pose so the wrist camera can see the table.
    You move one object around; at each new resting place the object's world
    position (wrist camera, accurate) is paired with its overhead pixel. A
    homography needs 4 such pairs and is over-determined past that, so
    collecting ~6 well-spread ones and fitting with RANSAC both solves the
    mapping and reveals how good it is.

    Pairs are only taken when the object is STATIONARY and the hand has left
    the frame, since a moving object would pair a pixel with a world position
    sampled a moment later.
    """
    print(f"calibrating the overhead camera against the wrist camera\n"
          f"  colour: {colour}\n")

    cap = open_lenovo(index)
    if cap is None:
        print("could not open the overhead camera")
        return
    try:
        async with await connect() as machine:
            cam = Camera.from_robot(machine, "cam")
            arm = Arm.from_robot(machine, "arm")
            intr = (await cam.get_properties()).intrinsic_parameters

            state = {"depth": None, "R": None, "T": None, "stop": False,
                     "depth_n": 0}
            pumps = [asyncio.create_task(depth_pump(cam, state)),
                     asyncio.create_task(pose_pump(machine, state))]
            try:
                print("moving to top-pose so the wrist camera sees the table...")
                await goto_saved_pose(machine, "top-pose")
                while await arm.is_moving():
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.5)

                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if state["depth"] is not None and state["R"] is not None:
                        break
                    await asyncio.sleep(0.01)
                if state["depth"] is None:
                    print("no depth from the wrist camera")
                    return

                print(f"\nMove the {colour} object to a new spot on the table,")
                print("then take your hand away. Repeat until collected.")
                print("The arm will not move. Ctrl-C to stop early.\n")

                pairs = []
                last = None
                stable_since = None

                while len(pairs) < want:
                    frame = await grab_async(cap)
                    if frame is None:
                        await asyncio.sleep(0.05)
                        continue
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    blobs = overhead_blobs(hsv, colour)
                    if len(blobs) != 1:
                        stable_since = None
                        await asyncio.sleep(0.1)
                        continue

                    cx, cy = blobs[0][0], blobs[0][1]
                    if last is not None and abs(cx - last[0]) < 4 \
                            and abs(cy - last[1]) < 4:
                        if stable_since is None:
                            stable_since = time.monotonic()
                    else:
                        stable_since = None
                    last = (cx, cy)

                    # Hold still for a moment before trusting the pair, so a
                    # hand still withdrawing does not get measured.
                    if stable_since is None \
                            or time.monotonic() - stable_since < 1.2:
                        await asyncio.sleep(0.1)
                        continue

                    seen = await wrist_observations(machine, cam, intr,
                                                    state, colour)
                    if len(seen) != 1:
                        print(f"  wrist sees {len(seen)} {colour} objects — "
                              f"need exactly 1, adjust and wait")
                        stable_since = None
                        await asyncio.sleep(0.6)
                        continue

                    wx, wy, wz = seen[0]
                    # Reject a repeat of a spot already recorded: a homography
                    # needs spread, and four clustered points fit anything.
                    if any((wx - p[2]) ** 2 + (wy - p[3]) ** 2 < 60 ** 2
                           for p in pairs):
                        await asyncio.sleep(0.3)
                        continue

                    pairs.append((cx, cy, wx, wy))
                    print(f"  [{len(pairs)}/{want}] px({cx:4d},{cy:4d}) "
                          f"-> world({wx:7.1f},{wy:7.1f})")
                    stable_since = None
                    await asyncio.sleep(0.5)

                if len(pairs) < 4:
                    print(f"\nonly {len(pairs)} pairs — need at least 4")
                    return

                src = np.array([[p[0], p[1]] for p in pairs], dtype=np.float64)
                dst = np.array([[p[2], p[3]] for p in pairs], dtype=np.float64)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 15.0)
                if H is None:
                    print("\nhomography did not converge — points too close "
                          "together or mismatched")
                    return

                oh = Overhead(H)
                errs = []
                for cx, cy, wx, wy in pairs:
                    gx, gy = oh.to_world(cx, cy)
                    errs.append(((gx - wx) ** 2 + (gy - wy) ** 2) ** 0.5)
                residual = float(np.mean(errs))

                print(f"\nsolved from {len(pairs)} pairs, "
                      f"{int(mask.sum())} inliers")
                print(f"  mean error {residual:.1f} mm, "
                      f"worst {max(errs):.1f} mm")
                oh.save(residual, pairs)
                print(f"  saved to {CALIB_PATH.name}")
                if residual > 40:
                    print("  WARNING: this is loose. Good enough to hover "
                          "towards, but re-run with more spread-out points "
                          "if recovery keeps missing.")
            finally:
                state["stop"] = True
                done, pending = await asyncio.wait(pumps, timeout=1.5)
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
    finally:
        cap.release()


async def view(index=None):
    """Both cameras side by side: overhead (Lenovo) and wrist (arm).

    Showing them together is the honest way to check this setup. The two
    disagree in useful ways — the wrist camera is accurate but loses the
    object up close, the overhead one is rough but never loses it — and
    seeing both at once makes it obvious which is which, whether they are
    looking at the same object, and whether the calibration is any good.
    """
    oh = Overhead.load()
    if oh.ready:
        print(f"loaded calibration from {CALIB_PATH.name}")
    else:
        print("NO CALIBRATION — run: python overhead.py --calibrate")
        print("overhead detections will show in pixels only")

    cap = open_lenovo(index)
    if cap is None:
        return

    print("press q to quit")
    try:
        async with await connect() as machine:
            cam = Camera.from_robot(machine, "cam")
            intr = (await cam.get_properties()).intrinsic_parameters
            state = {"depth": None, "R": None, "T": None, "stop": False,
                     "depth_n": 0}
            pumps = [asyncio.create_task(depth_pump(cam, state)),
                     asyncio.create_task(pose_pump(machine, state))]
            try:
                while True:
                    frame = await grab_async(cap)
                    if frame is None:
                        break

                    # --- overhead panel ---
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    oh_seen = {}
                    for colour in COLOURS:
                        _, _, bgr_colour = COLOURS[colour]
                        for cx, cy, bx, by, bw, bh in overhead_blobs(hsv,
                                                                     colour):
                            cv2.rectangle(frame, (bx, by),
                                          (bx + bw, by + bh), bgr_colour, 2)
                            w = oh.to_world(cx, cy) if oh.ready else None
                            if w:
                                oh_seen[colour] = w
                            label = (f"{colour} {w[0]:.0f},{w[1]:.0f}" if w
                                     else f"{colour} px{cx},{cy}")
                            cv2.putText(frame, label, (bx, max(12, by - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        bgr_colour, 2)
                    # Show where the ROI cuts off, so an object that falls
                    # outside it is visibly outside rather than mysteriously
                    # undetected.
                    roi_y = int(frame.shape[0] * ROI_TOP_FRAC)
                    cv2.line(frame, (0, roi_y), (frame.shape[1], roi_y),
                             (60, 60, 60), 1)
                    cv2.putText(frame, "OVERHEAD (Lenovo)", (12, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

                    # --- wrist panel ---
                    images, _ = await cam.get_images(
                        filter_source_names=["color"])
                    jpeg = next((i for i in images
                                 if i.mime_type == CameraMimeType.JPEG), None)
                    if jpeg is None:
                        continue
                    wrist = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8),
                                         cv2.IMREAD_COLOR)
                    whsv = cv2.cvtColor(wrist, cv2.COLOR_BGR2HSV)
                    for colour in COLOURS:
                        _, _, bgr_colour = COLOURS[colour]
                        for cx, cy, bx, by, bw, bh in find_blobs(whsv, colour):
                            cv2.rectangle(wrist, (bx, by),
                                          (bx + bw, by + bh), bgr_colour, 2)
                            label = colour
                            if state["depth"] is not None \
                                    and state["R"] is not None:
                                patch = state["depth"][
                                    max(0, cy - 6):cy + 7,
                                    max(0, cx - 6):cx + 7]
                                valid = patch[patch > 0]
                                if valid.size >= 20:
                                    z = float(np.percentile(valid, 25))
                                    c3 = np.array([
                                        (cx - intr.center_x_px) * z
                                        / intr.focal_x_px,
                                        (cy - intr.center_y_px) * z
                                        / intr.focal_y_px, z])
                                    wv = state["R"] @ c3 + state["T"]
                                    label = (f"{colour} {wv[0]:.0f},"
                                             f"{wv[1]:.0f}")
                                    # Where the two cameras disagree, say by
                                    # how much — that is the calibration
                                    # quality, live.
                                    if colour in oh_seen:
                                        d = ((oh_seen[colour][0] - wv[0]) ** 2
                                             + (oh_seen[colour][1] - wv[1])
                                             ** 2) ** 0.5
                                        label += f"  d={d:.0f}mm"
                            cv2.putText(wrist, label, (bx, max(12, by - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        bgr_colour, 2)
                    cv2.putText(wrist, "WRIST (arm)", (12, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

                    # Match heights so hstack works whatever each camera gives.
                    h = 540
                    left = cv2.resize(frame, (int(frame.shape[1] * h
                                                  / frame.shape[0]), h))
                    right = cv2.resize(wrist, (int(wrist.shape[1] * h
                                                   / wrist.shape[0]), h))
                    cv2.imshow("overhead | wrist",
                               np.hstack([left, right]))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            finally:
                state["stop"] = True
                done, pending = await asyncio.wait(pumps, timeout=1.5)
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _cli_index(argv):
    """--index N forces a device, bypassing auto-detection."""
    if "--index" in argv:
        return int(argv[argv.index("--index") + 1])
    return None


if __name__ == "__main__":
    forced = _cli_index(sys.argv)
    if forced is not None:
        print(f"forcing camera index {forced}")
    if "--calibrate" in sys.argv:
        colour = next((a for a in sys.argv[1:]
                       if not a.startswith("-")
                       and not a.isdigit()), "orange")
        asyncio.run(calibrate(colour, index=forced))
    else:
        asyncio.run(view(index=forced))
