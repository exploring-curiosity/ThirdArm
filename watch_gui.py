"""Live view of everything the system believes, in both cameras at once.

One window, two panels. The point is to see the sources DISAGREE: each
measurement is drawn in its own colour in both views, so when the arm goes to
the wrong place it is obvious which sensor sent it there.

    WHITE   blob centroid + depth  -- x/y, 16 Hz, the tracking source
    GREEN   point-cloud segmenter  -- the accurate source, ~1.3 Hz, x/y AND z
    ORANGE  overhead camera        -- homography on the table plane, x/y only
    YELLOW  SAM grasp axis         -- the direction the jaws will close

Everything is projected into BOTH panels, so a segment the overhead camera
disagrees with shows up as two marks on different objects.

    python watch_gui.py              # orange
    python watch_gui.py yellow --sam

Click either panel to select an object; the click is resolved to a world
position and snapped to the nearest detection, so it works in either view.

Keys:  s  SAM survey   p  report the selection   x  clear   f  fullscreen
       q  quit
"""

from __future__ import annotations

import asyncio
import sys
import time

import cv2
import numpy as np

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.media.video import CameraMimeType
from viam.services.vision import VisionClient

from tutorial import connect, goto_saved_pose
from slow_pick import wait_until_stopped, gripper_pose_in_world
from live3d import COLOURS, depth_pump, pose_pump, find_blobs
from seg_pick import find_objects, SEGMENTER
from overhead import (
    Overhead, open_lenovo, grab_async, overhead_blobs, WORK_W, WORK_H,
)
from calib_gui import world_to_wrist_px
from multi_pick import WORKSPACE

WRIST_W, WRIST_H = 1280, 720

# A click this far from a detected object still selects that object. Beyond
# it the click is taken at face value, so an empty patch of table can be
# chosen deliberately rather than silently snapping somewhere else.
CLICK_SNAP_MM = 60.0

C_BLOB = (255, 255, 255)
C_SEG = (120, 255, 120)
C_OVER = (0, 170, 255)
C_SAM = (0, 255, 255)


def wrist_px(w, intr, R, T, scale):
    """World x/y (table plane) -> pixel in the downscaled wrist panel."""
    p = world_to_wrist_px(w, intr, R, T)
    if p is None:
        return None
    q = (int(p[0] * scale[0]), int(p[1] * scale[1]))
    if 0 <= q[0] < WORK_W and 0 <= q[1] < WORK_H:
        return q
    return None


def over_px(w, oh):
    """World x/y -> pixel in the overhead panel, via the inverse homography."""
    if not oh.ready:
        return None
    v = np.linalg.inv(oh.H) @ np.array([w[0], w[1], 1.0])
    if abs(v[2]) < 1e-9:
        return None
    q = (int(v[0] / v[2]), int(v[1] / v[2]))
    if 0 <= q[0] < WORK_W and 0 <= q[1] < WORK_H:
        return q
    return None


def mark(panel, q, colour, label, r=7):
    if q is None:
        return
    cv2.circle(panel, q, r, colour, 2)
    cv2.line(panel, (q[0] - r - 4, q[1]), (q[0] + r + 4, q[1]), colour, 1)
    cv2.line(panel, (q[0], q[1] - r - 4), (q[0], q[1] + r + 4), colour, 1)
    if label:
        cv2.putText(panel, label, (q[0] + 11, q[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)


async def main(argv):
    colour = next((a for a in argv[1:] if not a.startswith("-")), "orange")
    use_sam = "--sam" in argv
    if colour not in COLOURS:
        print(f"unknown colour '{colour}'; one of {sorted(COLOURS)}")
        return

    sam = None
    if use_sam:
        import sam_observe
        sam = sam_observe.load()

    oh = Overhead.load()
    print(f"overhead calibration: "
          f"{'ready' if oh.ready else 'MISSING — run calib_gui.py'}")
    cap = open_lenovo(None)
    if cap is None:
        return
    print("keys:  s  SAM survey      q  quit")

    win = f"watch — {colour}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Click anywhere on either panel to choose the object to pick. The click
    # is resolved to a world position, so it does not matter which panel it
    # lands in -- the two views are just two ways of looking at the same
    # table.
    picked = {"world": None, "at": 0.0, "panel": None, "px": None}

    def on_mouse(event, x, y, flags, _p):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Panels are stacked horizontally, overhead first, each WORK_W wide;
        # the canvas is then scaled to the screen, so undo that first.
        sx = view["canvas_w"] / max(1, view["shown_w"])
        sy = view["canvas_h"] / max(1, view["shown_h"])
        cx, cy = int(x * sx), int(y * sy)
        if cx < WORK_W:
            picked["panel"], picked["px"] = "overhead", (cx, cy)
        else:
            picked["panel"], picked["px"] = "wrist", (cx - WORK_W, cy)
        picked["at"] = time.monotonic()

    view = {"canvas_w": WORK_W * 2, "canvas_h": WORK_H,
            "shown_w": WORK_W * 2, "shown_h": WORK_H}
    cv2.setMouseCallback(win, on_mouse)

    try:
        async with await connect() as machine:
            cam = Camera.from_robot(machine, "cam")
            arm = Arm.from_robot(machine, "arm")
            segmenter = VisionClient.from_robot(machine, SEGMENTER)
            intr = (await cam.get_properties()).intrinsic_parameters
            state = {"depth": None, "R": None, "T": None, "stop": False,
                     "depth_n": 0}
            pumps = [asyncio.create_task(depth_pump(cam, state)),
                     asyncio.create_task(pose_pump(machine, state))]

            # The point-cloud segmenter is slow (~750 ms), so it runs on its
            # own schedule and publishes into `shared` rather than gating the
            # display loop.
            shared = {"segs": [], "at": 0.0, "err": None}

            async def seg_loop():
                while not state["stop"]:
                    try:
                        objs = await find_objects(machine, segmenter)
                        shared["segs"] = objs
                        shared["at"] = time.monotonic()
                        shared["err"] = None
                    except Exception as e:                  # noqa: BLE001
                        shared["err"] = str(e)[:50]
                    await asyncio.sleep(0.1)

            pumps.append(asyncio.create_task(seg_loop()))
            try:
                print("moving to top-pose ...")
                await goto_saved_pose(machine, "top-pose")
                await wait_until_stopped(arm)
                t0 = time.monotonic()
                while time.monotonic() - t0 < 10:
                    if state["depth"] is not None and state["R"] is not None:
                        break
                    await asyncio.sleep(0.01)

                sam_objs, sam_at = [], 0.0
                while True:
                    R, T, depth = state["R"], state["T"], state["depth"]
                    if R is None or depth is None:
                        await asyncio.sleep(0.05)
                        continue

                    # ---------------- overhead panel ----------------
                    frame = await grab_async(cap)
                    if frame is None:
                        break
                    ov = frame.copy()
                    ohsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    oh_world = []
                    for cx, cy, bx, by, bw, bh in overhead_blobs(ohsv, colour):
                        cv2.rectangle(ov, (bx, by), (bx+bw, by+bh), C_OVER, 2)
                        w = oh.to_world(cx, cy) if oh.ready else None
                        if w:
                            oh_world.append(w)
                            cv2.putText(ov, f"{w[0]:.0f},{w[1]:.0f}",
                                        (bx, max(12, by-6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                        C_OVER, 1)

                    # ---------------- wrist panel ----------------
                    images, _ = await cam.get_images(
                        filter_source_names=["color"])
                    jpeg = next((i for i in images
                                 if i.mime_type == CameraMimeType.JPEG), None)
                    if jpeg is None:
                        continue
                    full = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8),
                                        cv2.IMREAD_COLOR)
                    wr = cv2.resize(full, (WORK_W, WORK_H))
                    scale = (WORK_W / WRIST_W, WORK_H / WRIST_H)

                    whsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
                    blob_world = []
                    for cx, cy, bx, by, bw, bh in find_blobs(whsv, colour):
                        patch = depth[max(0, cy-6):cy+7, max(0, cx-6):cx+7]
                        valid = patch[patch > 0]
                        if valid.size < 20:
                            continue
                        z = float(np.percentile(valid, 25))
                        c = np.array([
                            (cx - intr.center_x_px) * z / intr.focal_x_px,
                            (cy - intr.center_y_px) * z / intr.focal_y_px, z])
                        w = R @ c + T
                        if not (WORKSPACE["z"][0] < w[2] < WORKSPACE["z"][1]):
                            continue
                        blob_world.append((float(w[0]), float(w[1]),
                                           float(w[2])))
                        cv2.rectangle(wr,
                                      (int(bx*scale[0]), int(by*scale[1])),
                                      (int((bx+bw)*scale[0]),
                                       int((by+bh)*scale[1])), C_BLOB, 2)

                    # ---------------- overlay every source on both ----------
                    for w in blob_world:
                        lab = f"blob {w[0]:.0f},{w[1]:.0f},{w[2]:.0f}"
                        mark(wr, wrist_px(w, intr, R, T, scale), C_BLOB, lab)
                        mark(ov, over_px(w, oh), C_BLOB, lab)
                    seg_world = []
                    for label, pose, dims in shared["segs"]:
                        w = (pose.x, pose.y)
                        seg_world.append((w, dims))
                        top = pose.z + (dims.z / 2.0 if dims else 0.0)
                        lab = (f"seg {w[0]:.0f},{w[1]:.0f} top{top:.0f}"
                               + (f" h{dims.z:.0f}" if dims else ""))
                        mark(wr, wrist_px(w, intr, R, T, scale), C_SEG, lab)
                        mark(ov, over_px(w, oh), C_SEG, lab)
                    for w in oh_world:
                        mark(wr, wrist_px(w, intr, R, T, scale), C_OVER, "overhead")

                    # SAM grasp axes, drawn where SAM measured them.
                    for o in sam_objs:
                        cx, cy = o["centroid"]
                        gd = np.deg2rad(o["grasp_deg"])
                        L = 55
                        a = (int((cx-np.cos(gd)*L)*scale[0]),
                             int((cy-np.sin(gd)*L)*scale[1]))
                        b = (int((cx+np.cos(gd)*L)*scale[0]),
                             int((cy+np.sin(gd)*L)*scale[1]))
                        cv2.line(wr, a, b, C_SAM, 3)
                        cv2.putText(wr, f"{o['theta']:+.0f}d",
                                    (int(cx*scale[0])+8, int(cy*scale[1])+18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_SAM, 2)
                        w = o["xyz_approx"][:2]
                        mark(ov, over_px(w, oh), C_SAM, "SAM")

                    # ---------------- HUD ----------------
                    g = await gripper_pose_in_world(machine)
                    age = time.monotonic() - shared["at"]
                    cv2.putText(ov, "OVERHEAD", (12, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(wr, "WRIST", (12, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    canvas = np.hstack([ov, wr])
                    hud = (f"gripper({g.x:.0f},{g.y:.0f},{g.z:.0f}) "
                           f"th{g.theta:+.0f}   blob {len(blob_world)}   "
                           f"seg {len(shared['segs'])} ({age:.1f}s ago)   "
                           f"overhead {len(oh_world)}")
                    cv2.putText(canvas, hud, (12, canvas.shape[0]-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
                    key = ("WHITE blob+depth   GREEN point cloud   "
                           "ORANGE overhead   YELLOW SAM grasp")
                    cv2.putText(canvas, key, (12, canvas.shape[0]-14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    if shared["err"]:
                        cv2.putText(canvas, f"segmenter: {shared['err']}",
                                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0,0,255), 2)
                    # ---------------- click -> object ----------------
                    if picked["px"] is not None:
                        px, py = picked["px"]
                        wsel = None
                        if picked["panel"] == "overhead" and oh.ready:
                            wsel = oh.to_world(px, py)
                        elif picked["panel"] == "wrist":
                            # Invert the wrist projection through the depth
                            # map, so the click lands on the real surface
                            # rather than an assumed plane.
                            fx = int(px / scale[0])
                            fy = int(py / scale[1])
                            patch = depth[max(0, fy-6):fy+7, max(0, fx-6):fx+7]
                            good = patch[patch > 0]
                            if good.size >= 20:
                                zc = float(np.percentile(good, 25))
                                c = np.array([
                                    (fx - intr.center_x_px) * zc / intr.focal_x_px,
                                    (fy - intr.center_y_px) * zc / intr.focal_y_px,
                                    zc])
                                wv = R @ c + T
                                wsel = (float(wv[0]), float(wv[1]))
                        picked["px"] = None
                        if wsel is not None:
                            # Snap to the nearest thing actually detected, so
                            # a slightly-off click still selects an object
                            # rather than bare table.
                            cands = ([("blob", (b[0], b[1])) for b in blob_world]
                                     + [("seg", w) for w, _ in seg_world]
                                     + [("overhead", w) for w in oh_world])
                            if cands:
                                src, best = min(
                                    cands,
                                    key=lambda c: (c[1][0]-wsel[0])**2
                                    + (c[1][1]-wsel[1])**2)
                                d = ((best[0]-wsel[0])**2
                                     + (best[1]-wsel[1])**2) ** 0.5
                                if d <= CLICK_SNAP_MM:
                                    picked["world"] = best
                                    print(f"  selected {src} at "
                                          f"({best[0]:.1f}, {best[1]:.1f}) "
                                          f"[{d:.0f} mm from the click]")
                                else:
                                    picked["world"] = wsel
                                    print(f"  selected empty table at "
                                          f"({wsel[0]:.1f}, {wsel[1]:.1f}) "
                                          f"— nearest object {d:.0f} mm away")
                            else:
                                picked["world"] = wsel

                    # Draw the current selection in both panels.
                    if picked["world"] is not None:
                        wsel = picked["world"]
                        for panel, q in ((wr, wrist_px(wsel, intr, R, T, scale)),
                                         (ov, over_px(wsel, oh))):
                            if q is not None:
                                cv2.circle(panel, q, 18, (255, 0, 255), 3)
                                cv2.putText(panel, "SELECTED",
                                            (q[0]+22, q[1]+5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                            (255, 0, 255), 2)
                        canvas = np.hstack([ov, wr])
                        cv2.putText(canvas, hud, (12, canvas.shape[0]-40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0,255,255), 2)
                        cv2.putText(canvas, key, (12, canvas.shape[0]-14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (200,200,200), 1)
                        cv2.putText(canvas,
                                    f"SELECTED ({wsel[0]:.0f}, {wsel[1]:.0f})"
                                    f"   p = pick it   x = clear",
                                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 0, 255), 2)

                    view["canvas_w"], view["canvas_h"] = (canvas.shape[1],
                                                          canvas.shape[0])
                    try:
                        _, _, ww, wh = cv2.getWindowImageRect(win)
                        if ww > 0 and wh > 0:
                            view["shown_w"], view["shown_h"] = ww, wh
                    except Exception:                       # noqa: BLE001
                        pass
                    cv2.imshow(win, canvas)

                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('q'):
                        break
                    if k == ord('x'):
                        picked["world"] = None
                        print("  selection cleared")
                    if k == ord('p') and picked["world"] is not None:
                        wsel = picked["world"]
                        print(f"\n  PICK requested at "
                              f"({wsel[0]:.1f}, {wsel[1]:.1f})")
                        print(f"  run:  python track_pick.py --sam --slow "
                              f"{colour}")
                        print("  (this viewer does not move the arm)")
                    if k == ord('f'):
                        full_now = cv2.getWindowProperty(
                            win, cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty(
                            win, cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_NORMAL if full_now == cv2.WINDOW_FULLSCREEN
                            else cv2.WINDOW_FULLSCREEN)
                    if k == ord('s') and sam is not None:
                        import sam_observe
                        found, dt = await asyncio.to_thread(
                            sam_observe.survey, full, depth, intr, R, T, sam,
                            workspace=WORKSPACE)
                        for o in found:
                            o["theta"] = sam_observe.image_deg_to_world_theta(
                                o["grasp_deg"], R)
                        sam_objs, sam_at = found, time.monotonic()
                        print(f"  SAM: {len(found)} objects in {dt:.1f}s")
                        for o in found:
                            x, y, _ = o["xyz_approx"]
                            print(f"    {o['name']:<16} ~({x:.0f},{y:.0f}) "
                                  f"grasp {o['grasp_deg']:.0f}deg -> "
                                  f"theta {o['theta']:+.0f}")
            finally:
                state["stop"] = True
                done, pending = await asyncio.wait(pumps, timeout=1.5)
                for t in pending:
                    t.cancel()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
