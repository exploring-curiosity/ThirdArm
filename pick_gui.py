"""Control interface for the arm: see what every sensor sees, and drive a pick.

Not a viewer. Three panels, a button column, and the arm under your control:

    OVERHEAD   fixed camera, whole table, calibrated homography
    WRIST      eye-in-hand, the tracking source
    POINT CLOUD a live top-down plot of the segmenter's 3D boxes, drawn in
               world millimetres beside the camera views so the geometry the
               arm actually reasons about is visible rather than implied

Every measurement is projected into BOTH camera panels, so sources that
disagree show up as two marks on different objects:

    WHITE   blob centroid + depth   x/y at ~16 Hz, what tracking follows
    GREEN   point-cloud segmenter   the accurate source: x/y AND a real height
    ORANGE  overhead camera         plane homography, x/y only
    YELLOW  SAM grasp axis          the direction the jaws will close
    CYAN    drop box                where a picked object is released

Buttons run the pick for a colour; clicking a panel selects one object.

    python pick_gui.py            # loads SAM at startup
    python pick_gui.py --no-sam   # skip it if you only want tracking
"""

from __future__ import annotations

import asyncio
import subprocess
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
from multi_pick_box import locate_drop_box, BOX_EXCLUSION_X_MM, BOX_EXCLUSION_Y_MM
from overhead import Overhead, open_lenovo, grab_async, overhead_blobs
from calib_gui import world_to_wrist_px
from multi_pick import WORKSPACE

PANEL_W, PANEL_H = 800, 600          # each camera panel
CLOUD_W = 520                        # the 3D/top-down plot beside them
BAR_H = 92                           # button strip along the bottom
WRIST_W, WRIST_H = 1280, 720

C_BLOB = (255, 255, 255)
C_SEG = (120, 255, 120)
C_OVER = (0, 170, 255)
C_SAM = (0, 255, 255)
C_BOX = (255, 220, 0)
C_SEL = (255, 0, 255)

CLICK_SNAP_MM = 60.0

# The plot covers the reachable table, so a point outside it is outside the
# arm's world rather than merely off-screen.
PLOT_X = (150.0, 800.0)
PLOT_Y = (-450.0, 450.0)


def plot_px(w):
    """World x/y -> pixel in the point-cloud plot.

    World +x runs away from the base and is drawn upward; world +y runs to the
    side and is drawn rightward, so the plot reads like looking down at the
    table from above the arm.
    """
    fx = (w[1] - PLOT_Y[0]) / (PLOT_Y[1] - PLOT_Y[0])
    fy = 1.0 - (w[0] - PLOT_X[0]) / (PLOT_X[1] - PLOT_X[0])
    return int(fx * CLOUD_W), int(fy * PANEL_H)


def draw_cloud(segs, blobs, drop, sel, gripper):
    """Top-down plot of the segmenter's boxes, in world millimetres."""
    c = np.full((PANEL_H, CLOUD_W, 3), 24, np.uint8)

    for mm in range(int(PLOT_Y[0]), int(PLOT_Y[1]) + 1, 100):
        x, _ = plot_px((PLOT_X[0], mm))
        cv2.line(c, (x, 0), (x, PANEL_H), (44, 44, 44), 1)
        if mm % 200 == 0:
            cv2.putText(c, f"{mm}", (x + 3, PANEL_H - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
    for mm in range(int(PLOT_X[0]), int(PLOT_X[1]) + 1, 100):
        _, y = plot_px((mm, PLOT_Y[0]))
        cv2.line(c, (0, y), (CLOUD_W, y), (44, 44, 44), 1)
        if mm % 200 == 0:
            cv2.putText(c, f"x{mm}", (4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)

    if drop is not None:
        a = plot_px((drop.x - BOX_EXCLUSION_X_MM, drop.y - BOX_EXCLUSION_Y_MM))
        b = plot_px((drop.x + BOX_EXCLUSION_X_MM, drop.y + BOX_EXCLUSION_Y_MM))
        cv2.rectangle(c, a, b, C_BOX, 2)
        cv2.putText(c, "DROP", (min(a[0], b[0]) + 6, min(a[1], b[1]) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_BOX, 1)

    # Each segment as a real footprint, so size is visible not just position.
    for label, pose, dims in segs:
        w = (pose.x, pose.y)
        if dims is not None:
            a = plot_px((pose.x - dims.x / 2, pose.y - dims.y / 2))
            b = plot_px((pose.x + dims.x / 2, pose.y + dims.y / 2))
            cv2.rectangle(c, a, b, C_SEG, 2)
            top = pose.z + dims.z / 2.0
            cv2.putText(c, f"h{dims.z:.0f} top{top:.0f}",
                        (min(a[0], b[0]), min(a[1], b[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_SEG, 1)
        q = plot_px(w)
        cv2.circle(c, q, 4, C_SEG, -1)

    for b in blobs:
        cv2.circle(c, plot_px(b), 5, C_BLOB, 1)

    if gripper is not None:
        q = plot_px((gripper.x, gripper.y))
        cv2.drawMarker(c, q, (80, 80, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(c, "arm", (q[0] + 9, q[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 255), 1)

    if sel is not None:
        cv2.circle(c, plot_px(sel), 13, C_SEL, 2)

    cv2.putText(c, "POINT CLOUD (top-down, mm)", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
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


def build_buttons(width):
    """Button strip: one PICK per colour, plus the actions."""
    labels = ([(f"PICK {c.upper()}", f"pick:{c}") for c in sorted(COLOURS)]
              + [("SAM SURVEY", "sam"), ("FIND BOX", "box"),
                 ("STOP", "stop"), ("CLEAR", "clear"), ("QUIT", "quit")])
    n = len(labels)
    pad, y0, y1 = 8, 10, BAR_H - 12
    bw = (width - pad * (n + 1)) // n
    out = []
    for i, (text, action) in enumerate(labels):
        x0 = pad + i * (bw + pad)
        out.append((text, action, (x0, y0, x0 + bw, y1)))
    return out


def draw_buttons(bar, buttons, busy, sel_colour):
    for text, action, (x0, y0, x1, y1) in buttons:
        if action.startswith("pick:"):
            name = action.split(":")[1]
            base = COLOURS[name][2]
            col = tuple(int(v * (0.35 if busy else 1.0)) for v in base)
        elif action == "stop":
            col = (40, 40, 200)
        elif action == "quit":
            col = (60, 60, 60)
        else:
            col = (90, 90, 90)
        cv2.rectangle(bar, (x0, y0), (x1, y1), col, -1)
        edge = (255, 255, 255) if action == f"pick:{sel_colour}" else (25, 25, 25)
        cv2.rectangle(bar, (x0, y0), (x1, y1), edge, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(bar, text, (x0 + (x1 - x0 - tw) // 2,
                                y0 + (y1 - y0 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)
    return bar


async def run_pick(colour, sel, use_sam, log):
    """Launch track_pick as its own process.

    Deliberately a subprocess rather than an in-process call: track_pick owns
    the arm, opens its own robot session and its own camera pumps, and running
    it inside this GUI's event loop would mean two clients fighting over the
    same resources. A separate process also means the STOP button can end the
    pick without taking the interface down with it.
    """
    cmd = [sys.executable, "-u", "track_pick.py", "--slow"]
    if use_sam:
        cmd.append("--sam")
    cmd.append(colour)
    log(f"$ {' '.join(cmd[2:])}")
    return await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


async def main(argv):
    use_sam = "--no-sam" not in argv
    sam = None
    if use_sam:
        import sam_observe
        sam = sam_observe.load()

    oh = Overhead.load()
    print(f"overhead: {'calibrated' if oh.ready else 'NOT CALIBRATED'}")
    cap = open_lenovo(None)
    if cap is None:
        return

    win = "arm control"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, PANEL_W * 2 + CLOUD_W, PANEL_H + BAR_H)

    canvas_w = PANEL_W * 2 + CLOUD_W
    buttons = build_buttons(canvas_w)
    ui = {"click": None, "shown": (canvas_w, PANEL_H + BAR_H)}

    def on_mouse(event, x, y, flags, _p):
        if event == cv2.EVENT_LBUTTONDOWN:
            sx = canvas_w / max(1, ui["shown"][0])
            sy = (PANEL_H + BAR_H) / max(1, ui["shown"][1])
            ui["click"] = (int(x * sx), int(y * sy))

    cv2.setMouseCallback(win, on_mouse)

    lines = []
    def log(msg):
        print("  " + msg)
        lines.append(msg)
        del lines[:-6]

    try:
        async with await connect() as machine:
            cam = Camera.from_robot(machine, "cam")
            arm = Arm.from_robot(machine, "arm")
            segmenter = VisionClient.from_robot(machine, SEGMENTER)
            intr = (await cam.get_properties()).intrinsic_parameters
            state = {"depth": None, "R": None, "T": None, "stop": False,
                     "depth_n": 0, "frame": None}
            shared = {"segs": [], "at": 0.0, "err": None,
                      "sam": [], "sam_at": 0.0, "sam_ms": 0.0,
                      "sam_err": None}
            proc = {"p": None}
            drop = {"pose": None}
            sel = {"world": None, "colour": None}
            sam_objs = []

            async def sam_loop():
                """Keep a SAM survey fresh in the background.

                SAM takes 2-3 s per pass, so it cannot run in the render loop
                and does not need to: a rigid object's grasp angle does not
                change while nothing touches it. Running it continuously on a
                worker thread means the grasp axis is simply always on screen
                and always current to within a few seconds, with no button to
                press and no stall when it runs.
                """
                import sam_observe
                while not state["stop"]:
                    if sam is None or state["frame"] is None \
                            or state["R"] is None:
                        await asyncio.sleep(0.2)
                        continue
                    try:
                        frame = state["frame"]
                        R, T = state["R"], state["T"]
                        found, dt = await asyncio.to_thread(
                            sam_observe.survey, frame, state["depth"], intr,
                            R, T, sam, workspace=WORKSPACE)
                        for o in found:
                            o["theta"] = \
                                sam_observe.image_deg_to_world_theta(
                                    o["grasp_deg"], R)
                        shared["sam"] = found
                        shared["sam_at"] = time.monotonic()
                        shared["sam_ms"] = dt * 1000.0
                    except Exception as e:                  # noqa: BLE001
                        shared["sam_err"] = str(e)[:60]
                    await asyncio.sleep(0.1)

            async def seg_loop():
                # The segmenter is intermittent: consecutive reads of a
                # stationary scene alternate between finding the object and
                # finding nothing (measured 0,0,1,0,1 over five reads). Show
                # the last non-empty result and say how old it is, rather than
                # flickering the display to zero on every empty frame. An
                # empty read is not evidence the object is gone.
                while not state["stop"]:
                    try:
                        got = await find_objects(machine, segmenter)
                        if got:
                            shared["segs"] = got
                            shared["at"] = time.monotonic()
                        shared["err"] = None
                    except Exception as e:                  # noqa: BLE001
                        shared["err"] = str(e)[:60]
                    await asyncio.sleep(0.05)

            async def drain():
                """Surface the pick subprocess's output in the interface."""
                while not state["stop"]:
                    p = proc["p"]
                    if p is not None and p.stdout is not None:
                        try:
                            raw = await asyncio.wait_for(p.stdout.readline(),
                                                         timeout=0.3)
                            if raw:
                                t = raw.decode(errors="replace").strip()
                                if t and not t.startswith(("Warning", "  warn")):
                                    log(t[:110])
                            elif p.returncode is not None:
                                log(f"pick finished (exit {p.returncode})")
                                proc["p"] = None
                        except asyncio.TimeoutError:
                            pass
                    else:
                        await asyncio.sleep(0.1)

            pumps = [asyncio.create_task(depth_pump(cam, state)),
                     asyncio.create_task(pose_pump(machine, state)),
                     asyncio.create_task(seg_loop()),
                     asyncio.create_task(sam_loop()),
                     asyncio.create_task(drain())]
            try:
                log("moving to top-pose")
                await goto_saved_pose(machine, "top-pose")
                await wait_until_stopped(arm)
                t0 = time.monotonic()
                while time.monotonic() - t0 < 10:
                    if state["depth"] is not None and state["R"] is not None:
                        break
                    await asyncio.sleep(0.01)
                try:
                    drop["pose"] = await locate_drop_box(machine, cam)
                    if drop["pose"]:
                        log(f"drop box ({drop['pose'].x:.0f}, "
                            f"{drop['pose'].y:.0f})")
                except Exception as e:                      # noqa: BLE001
                    log(f"drop box lookup failed: {str(e)[:50]}")

                while True:
                    busy = proc["p"] is not None
                    R, T, depth = state["R"], state["T"], state["depth"]
                    if R is None or depth is None:
                        await asyncio.sleep(0.05)
                        continue

                    # ---------- overhead ----------
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
                            cv2.rectangle(
                                ov, (int(bx*osc[0]), int(by*osc[1])),
                                (int((bx+bw)*osc[0]), int((by+bh)*osc[1])),
                                COLOURS[name][2], 2)

                    # ---------- wrist ----------
                    images, _ = await cam.get_images(
                        filter_source_names=["color"])
                    jpeg = next((i for i in images
                                 if i.mime_type == CameraMimeType.JPEG), None)
                    if jpeg is None:
                        continue
                    full = cv2.imdecode(np.frombuffer(jpeg.data, np.uint8),
                                        cv2.IMREAD_COLOR)
                    wr = cv2.resize(full, (PANEL_W, PANEL_H))
                    wsc = (PANEL_W / WRIST_W, PANEL_H / WRIST_H)

                    whsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
                    blob_world = []
                    for name in COLOURS:
                        for cx, cy, bx, by, bw, bh in find_blobs(whsv, name):
                            patch = depth[max(0, cy-6):cy+7, max(0, cx-6):cx+7]
                            good = patch[patch > 0]
                            if good.size < 20:
                                continue
                            z = float(np.percentile(good, 25))
                            c = np.array([
                                (cx-intr.center_x_px)*z/intr.focal_x_px,
                                (cy-intr.center_y_px)*z/intr.focal_y_px, z])
                            w = R @ c + T
                            if not (WORKSPACE["z"][0] < w[2] < WORKSPACE["z"][1]):
                                continue
                            blob_world.append((float(w[0]), float(w[1])))
                            cv2.rectangle(
                                wr, (int(bx*wsc[0]), int(by*wsc[1])),
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
                    for label, pose, dims in shared["segs"]:
                        w = (pose.x, pose.y)
                        h = f" h{dims.z:.0f}" if dims else ""
                        mark(wr, wpx(w), C_SEG, f"seg {w[0]:.0f},{w[1]:.0f}{h}")
                        mark(ov, opx(w), C_SEG, f"seg{h}")
                    for w in oh_world:
                        mark(wr, wpx(w), C_OVER, "overhead")
                    if drop["pose"] is not None:
                        d = drop["pose"]
                        for panel, q in ((wr, wpx((d.x, d.y))),
                                         (ov, opx((d.x, d.y)))):
                            if q:
                                cv2.circle(panel, q, 16, C_BOX, 3)
                                cv2.putText(panel, "DROP BOX", (q[0]+20, q[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            C_BOX, 2)
                    for o in sam_objs:
                        cx, cy = o["centroid"]
                        a = np.deg2rad(o["grasp_deg"])
                        L = 70
                        p1 = (int((cx-np.cos(a)*L)*wsc[0]),
                              int((cy-np.sin(a)*L)*wsc[1]))
                        p2 = (int((cx+np.cos(a)*L)*wsc[0]),
                              int((cy+np.sin(a)*L)*wsc[1]))
                        cv2.line(wr, p1, p2, C_SAM, 3)
                        cv2.putText(wr, f"{o['name']} th{o['theta']:+.0f}",
                                    (int(cx*wsc[0])+10, int(cy*wsc[1])+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_SAM, 2)
                        mark(ov, opx(o["xyz_approx"][:2]), C_SAM, "SAM")

                    if sel["world"] is not None:
                        for panel, q in ((wr, wpx(sel["world"])),
                                         (ov, opx(sel["world"]))):
                            if q:
                                cv2.circle(panel, q, 22, C_SEL, 3)

                    g = await gripper_pose_in_world(machine)
                    cloud = draw_cloud(shared["segs"], blob_world,
                                       drop["pose"], sel["world"], g)

                    cv2.putText(ov, "OVERHEAD", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(wr, "WRIST", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    top = np.hstack([ov, wr, cloud])
                    bar = np.full((BAR_H, canvas_w, 3), 18, np.uint8)
                    draw_buttons(bar, buttons, busy, sel["colour"])
                    canvas = np.vstack([top, bar])

                    age = time.monotonic() - shared["at"]
                    hud = (f"gripper({g.x:.0f},{g.y:.0f},{g.z:.0f}) "
                           f"th{g.theta:+.0f}  |  blob {len(blob_world)}  "
                           f"seg {len(shared['segs'])} ({age:.1f}s)  "
                           f"overhead {len(oh_world)}  "
                           f"{'PICKING' if busy else 'idle'}")
                    cv2.putText(canvas, hud, (12, PANEL_H - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    cv2.putText(canvas,
                                "WHITE blob  GREEN cloud  ORANGE overhead  "
                                "YELLOW SAM  CYAN drop",
                                (12, PANEL_H - 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190,190,190), 1)
                    for i, t in enumerate(lines[-5:]):
                        cv2.putText(canvas, t, (PANEL_W*2 + 12, PANEL_H - 96 + i*18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (200, 200, 120), 1)
                    if shared["err"]:
                        cv2.putText(canvas, f"segmenter: {shared['err']}",
                                    (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)

                    try:
                        _, _, ww, wh = cv2.getWindowImageRect(win)
                        if ww > 0 and wh > 0:
                            ui["shown"] = (ww, wh)
                    except Exception:                       # noqa: BLE001
                        pass
                    cv2.imshow(win, canvas)
                    k = cv2.waitKey(1) & 0xFF

                    # ---------- input ----------
                    act = None
                    if ui["click"] is not None:
                        cx, cy = ui["click"]
                        ui["click"] = None
                        if cy >= PANEL_H:
                            by = cy - PANEL_H
                            for text, action, (x0, y0, x1, y1) in buttons:
                                if x0 <= cx <= x1 and y0 <= by <= y1:
                                    act = action
                        else:
                            wsel = None
                            if cx < PANEL_W and oh.ready:
                                wsel = oh.to_world(cx / osc[0], cy / osc[1])
                            elif PANEL_W <= cx < PANEL_W * 2:
                                fx = int((cx - PANEL_W) / wsc[0])
                                fy = int(cy / wsc[1])
                                patch = depth[max(0, fy-6):fy+7,
                                              max(0, fx-6):fx+7]
                                good = patch[patch > 0]
                                if good.size >= 20:
                                    z = float(np.percentile(good, 25))
                                    c = np.array([
                                        (fx-intr.center_x_px)*z/intr.focal_x_px,
                                        (fy-intr.center_y_px)*z/intr.focal_y_px,
                                        z])
                                    v = R @ c + T
                                    wsel = (float(v[0]), float(v[1]))
                            if wsel is not None:
                                cands = ([("blob", b) for b in blob_world]
                                         + [("seg", (p.x, p.y))
                                            for _, p, _ in shared["segs"]]
                                         + [("overhead", w) for w in oh_world])
                                if cands:
                                    src, best = min(
                                        cands,
                                        key=lambda c: (c[1][0]-wsel[0])**2
                                        + (c[1][1]-wsel[1])**2)
                                    d = ((best[0]-wsel[0])**2
                                         + (best[1]-wsel[1])**2)**0.5
                                    if d <= CLICK_SNAP_MM:
                                        sel["world"] = best
                                        log(f"selected {src} "
                                            f"({best[0]:.0f},{best[1]:.0f})")
                                    else:
                                        sel["world"] = wsel
                                        log(f"empty table "
                                            f"({wsel[0]:.0f},{wsel[1]:.0f})")
                                else:
                                    sel["world"] = wsel

                    if k == ord('q'):
                        act = "quit"

                    if act == "quit":
                        break
                    if act == "clear":
                        sel["world"], sel["colour"] = None, None
                        sam_objs = []
                        log("cleared")
                    if act == "stop":
                        if proc["p"] is not None:
                            proc["p"].terminate()
                            log("pick stopped")
                            proc["p"] = None
                        await arm.stop()
                        log("arm stopped")
                    if act == "box":
                        drop["pose"] = await locate_drop_box(machine, cam)
                        log(f"drop box "
                            f"({drop['pose'].x:.0f},{drop['pose'].y:.0f})"
                            if drop["pose"] else "drop box not found")
                    if act == "sam" and sam is not None:
                        import sam_observe
                        found, dt = await asyncio.to_thread(
                            sam_observe.survey, full, depth, intr, R, T, sam,
                            workspace=WORKSPACE)
                        for o in found:
                            o["theta"] = sam_observe.image_deg_to_world_theta(
                                o["grasp_deg"], R)
                        sam_objs = found
                        log(f"SAM: {len(found)} objects in {dt:.1f}s")
                    if act and act.startswith("pick:"):
                        name = act.split(":")[1]
                        if busy:
                            log("a pick is already running")
                        else:
                            sel["colour"] = name
                            proc["p"] = await run_pick(name, sel["world"],
                                                       sam is not None, log)
            finally:
                state["stop"] = True
                if proc["p"] is not None:
                    proc["p"].terminate()
                done, pending = await asyncio.wait(pumps, timeout=1.5)
                for t in pending:
                    t.cancel()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
