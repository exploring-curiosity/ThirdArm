"""Side-by-side calibration GUI: overhead and wrist, with a capture button.

Replaces the "move the object, wait, hope" flow of `overhead.py --calibrate`
with something you can see. Both cameras are live, every pair collected so far
is drawn in BOTH views, and a pair is only recorded when you press Capture --
so a frame with your hand in it, or a wrist mis-detection, never gets stored
by accident.

Why the existing points are drawn in both views
-----------------------------------------------
A pair is only meaningful if the two cameras were looking at the SAME object.
Drawing each stored pair at its overhead pixel and, independently, at the
pixel its world position projects to in the wrist image makes a bad pair
obvious: the two marks land on different things.

    python calib_gui.py            # orange (default)
    python calib_gui.py yellow

Keys:  SPACE / c  capture      u  undo last      s  solve and save
       r          refit+report q  quit
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import cv2
import numpy as np

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.media.video import CameraMimeType

from tutorial import connect, goto_saved_pose
from slow_pick import wait_until_stopped
from live3d import COLOURS, depth_pump, pose_pump, find_blobs
from overhead import (
    CALIB_PATH, PAIRS_PATH, Overhead, open_lenovo, grab_async,
    overhead_blobs, wrist_observations, WORK_W, WORK_H,
)

# A pair is only trustworthy when both cameras see exactly one object and they
# agree about where it is. 40 mm is well beyond the 1-2 mm a good pair shows
# and well under the 300+ mm a mismatched one does.
AGREE_MM = 40.0

# Points closer together than this add nothing: a homography needs spread.
MIN_SEPARATION_MM = 60.0


def load_pairs(colour):
    if not PAIRS_PATH.exists():
        return []
    try:
        d = json.loads(PAIRS_PATH.read_text())
        return [tuple(p) for p in d["pairs"]] if d.get("colour") == colour else []
    except Exception:                                      # noqa: BLE001
        return []


def save_pairs(colour, pairs):
    PAIRS_PATH.write_text(json.dumps(
        {"colour": colour, "pairs": [list(p) for p in pairs]}, indent=2))


def fit(pairs):
    """Homography plus per-pair residuals, or (None, []) if under-determined."""
    if len(pairs) < 4:
        return None, []
    src = np.array([[p[0], p[1]] for p in pairs], np.float32)
    dst = np.array([[p[2], p[3]] for p in pairs], np.float32)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 15.0)
    if H is None:
        return None, []
    res = []
    for cx, cy, wx, wy in pairs:
        v = H @ np.array([cx, cy, 1.0])
        res.append((((v[0] / v[2]) - wx) ** 2
                    + ((v[1] / v[2]) - wy) ** 2) ** 0.5)
    return H, res


def world_to_wrist_px(w, intr, R, T):
    """Project a world x/y on the table plane into wrist-camera pixels.

    Inverse of the deprojection wrist_observations does. The table plane is
    z = 0 in world, which is where the calibration objects sit.
    """
    cam = R.T @ (np.array([w[0], w[1], 0.0]) - T)
    if cam[2] <= 1.0:
        return None
    return (int(cam[0] * intr.focal_x_px / cam[2] + intr.center_x_px),
            int(cam[1] * intr.focal_y_px / cam[2] + intr.center_y_px))


BTN = (18, 470, 250, 524)          # x0, y0, x1, y1 of the Capture button


def draw_button(panel, armed, msg, msg_col):
    x0, y0, x1, y1 = BTN
    col = (40, 160, 40) if armed else (70, 70, 70)
    cv2.rectangle(panel, (x0, y0), (x1, y1), col, -1)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (255, 255, 255), 2)
    cv2.putText(panel, "CAPTURE" if armed else "not ready",
                (x0 + 16, y1 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2)
    cv2.putText(panel, msg, (x1 + 14, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, msg_col, 1)


async def main(argv):
    colour = next((a for a in argv[1:] if not a.startswith("-")), "orange")
    if colour not in COLOURS:
        print(f"unknown colour '{colour}'; one of {sorted(COLOURS)}")
        return

    pairs = load_pairs(colour)
    print(f"calibration GUI — colour: {colour}")
    print(f"  {len(pairs)} pair(s) already collected")
    print("  SPACE/c capture   u undo   s save   r report   q quit")

    cap = open_lenovo(None)
    if cap is None:
        print("could not open the overhead camera")
        return

    click = {"hit": False}

    def on_mouse(event, x, y, flags, _p):
        if event == cv2.EVENT_LBUTTONDOWN:
            # The button lives on the overhead panel, which is drawn first.
            if BTN[0] <= x <= BTN[2] and BTN[1] <= y <= BTN[3]:
                click["hit"] = True

    win = f"calibration — {colour}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

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
                print("moving to top-pose ...")
                await goto_saved_pose(machine, "top-pose")
                await wait_until_stopped(arm)
                t0 = time.monotonic()
                while time.monotonic() - t0 < 10:
                    if state["depth"] is not None and state["R"] is not None:
                        break
                    await asyncio.sleep(0.01)

                msg, msg_col = "", (200, 200, 200)
                while True:
                    frame = await grab_async(cap)
                    if frame is None:
                        break
                    R, T = state["R"], state["T"]

                    # ---------- overhead panel ----------
                    oh_panel = frame.copy()
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    blobs = list(overhead_blobs(hsv, colour))
                    for cx, cy, bx, by, bw, bh in blobs:
                        cv2.rectangle(oh_panel, (bx, by), (bx+bw, by+bh),
                                      (0, 255, 255), 2)
                        cv2.circle(oh_panel, (cx, cy), 5, (0, 0, 255), -1)

                    # Every stored pair, at the pixel it was captured at.
                    for i, (px, py, wx, wy) in enumerate(pairs, 1):
                        cv2.circle(oh_panel, (int(px), int(py)), 7,
                                   (255, 120, 0), -1)
                        cv2.putText(oh_panel, f"{i}", (int(px)+10, int(py)-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 200, 120), 2)
                    if len(pairs) >= 3:
                        hull = cv2.convexHull(
                            np.array([[int(p[0]), int(p[1])] for p in pairs],
                                     np.int32))
                        cv2.polylines(oh_panel, [hull], True, (255, 120, 0), 1)

                    cv2.putText(oh_panel, "OVERHEAD", (12, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                    # ---------- wrist panel ----------
                    images, _ = await cam.get_images(
                        filter_source_names=["color"])
                    jpeg = next((i for i in images
                                 if i.mime_type == CameraMimeType.JPEG), None)
                    if jpeg is None:
                        continue
                    wr = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8),
                                      cv2.IMREAD_COLOR)
                    wr = cv2.resize(wr, (WORK_W, WORK_H))
                    sx, sy = WORK_W / 1280.0, WORK_H / 720.0

                    whsv = cv2.cvtColor(wr, cv2.COLOR_BGR2HSV)
                    wblobs = list(find_blobs(whsv, colour))
                    for cx, cy, bx, by, bw, bh in wblobs:
                        cv2.rectangle(wr, (bx, by), (bx+bw, by+bh),
                                      (0, 255, 255), 2)

                    seen = await wrist_observations(machine, cam, intr,
                                                    state, colour)
                    # The same stored pairs, projected into the wrist view.
                    # A pair whose two marks land on different objects is bad.
                    if R is not None:
                        for i, (px, py, wx, wy) in enumerate(pairs, 1):
                            p = world_to_wrist_px((wx, wy), intr, R, T)
                            if p is None:
                                continue
                            q = (int(p[0]*sx), int(p[1]*sy))
                            if 0 <= q[0] < WORK_W and 0 <= q[1] < WORK_H:
                                cv2.circle(wr, q, 7, (255, 120, 0), -1)
                                cv2.putText(wr, f"{i}", (q[0]+10, q[1]-6),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (255, 200, 120), 2)
                    cv2.putText(wr, "WRIST", (12, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                    # ---------- can we capture? ----------
                    armed, why = False, ""
                    if len(blobs) != 1:
                        why = f"overhead sees {len(blobs)} {colour} (need 1)"
                    elif len(seen) != 1:
                        why = f"wrist sees {len(seen)} {colour} (need 1)"
                    else:
                        w = None
                        oh = Overhead.load()
                        cand = (seen[0][0], seen[0][1])
                        near = [p for p in pairs
                                if (p[2]-cand[0])**2 + (p[3]-cand[1])**2
                                < MIN_SEPARATION_MM**2]
                        if near:
                            why = f"too close to pair {pairs.index(near[0])+1}"
                        else:
                            armed = True
                            why = (f"ready: world "
                                   f"({cand[0]:.0f}, {cand[1]:.0f})")
                    draw_button(oh_panel, armed, why,
                                (120, 255, 120) if armed else (120, 120, 255))

                    H, res = fit(pairs)
                    hud = (f"{len(pairs)} pairs" +
                           (f"   residual {np.mean(res):.1f} mm "
                            f"(max {np.max(res):.1f})" if res else
                            "   need 4+ to fit"))
                    canvas = np.hstack([oh_panel, wr])
                    cv2.putText(canvas, hud, (12, canvas.shape[0]-14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    if msg:
                        cv2.putText(canvas, msg, (12, canvas.shape[0]-42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_col, 2)
                    cv2.imshow(win, canvas)

                    k = cv2.waitKey(1) & 0xFF
                    take = click["hit"] or k in (ord(' '), ord('c'))
                    click["hit"] = False

                    if k == ord('q'):
                        break
                    if take:
                        if not armed:
                            msg, msg_col = f"cannot capture: {why}", (120,120,255)
                        else:
                            cx, cy = blobs[0][0], blobs[0][1]
                            wx, wy = seen[0][0], seen[0][1]
                            pairs.append((cx, cy, wx, wy))
                            save_pairs(colour, pairs)
                            msg = (f"captured #{len(pairs)}: px({cx},{cy}) "
                                   f"-> world({wx:.0f},{wy:.0f})")
                            msg_col = (120, 255, 120)
                            print("  " + msg)
                    elif k == ord('u') and pairs:
                        gone = pairs.pop()
                        save_pairs(colour, pairs)
                        msg = f"removed pair {len(pairs)+1} {gone[2]:.0f},{gone[3]:.0f}"
                        msg_col = (120, 200, 255)
                        print("  " + msg)
                    elif k == ord('r'):
                        if res:
                            print(f"\n  {len(pairs)} pairs, mean "
                                  f"{np.mean(res):.1f} mm:")
                            for i, r in enumerate(res, 1):
                                flag = "   <-- BAD" if r > 20 else ""
                                print(f"    pair {i}: {r:6.1f} mm{flag}")
                        else:
                            print("  need 4+ pairs to fit")
                    elif k == ord('s'):
                        if H is None:
                            msg, msg_col = "need 4+ pairs to solve", (120,120,255)
                        else:
                            CALIB_PATH.write_text(json.dumps({
                                "H": H.tolist(),
                                "residual_mm": float(np.mean(res)),
                                "points": [list(p) for p in pairs],
                                "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }, indent=2))
                            msg = (f"saved {len(pairs)} pairs, residual "
                                   f"{np.mean(res):.1f} mm")
                            msg_col = (120, 255, 120)
                            print("  " + msg)
            finally:
                state["stop"] = True
                done, pending = await asyncio.wait(pumps, timeout=1.5)
                for p in pending:
                    p.cancel()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
