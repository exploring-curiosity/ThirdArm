"""Per-object 3D shape and orientation, camera view left, interactive 3D right.

Standalone: imports nothing from the pick pipeline and talks to no robot.

What is measured and what is not
--------------------------------
Read this before trusting a number on screen.

  MEASURED   the mask outline, the in-plane grasp angle (yaw), each object's
             pixel dimensions, and the RELATIVE depth structure inside a mask
             (which end of an object is nearer the camera).

  RELATIVE   depth comes from Depth Anything V2, a monocular estimator. Its
             output is unscaled and has no metric meaning: a surface reading
             "nearer" really is nearer, but by an unknown amount. Tilt is
             therefore indicative, not metric.

  NOT KNOWN  absolute height, absolute size, and the true 3D pose in robot
             coordinates. Those need the wrist RealSense.

So the 3D panel is a faithful picture of measured relative structure, not a
calibrated pose. It is labelled that way on screen rather than quietly
implying more precision than exists.

Why the depth map is cropped to the work region
-----------------------------------------------
Monocular depth normalises across whatever it is shown. On the full frame the
chair and foreground consumed the entire range and the table objects came back
within 1-5 levels of the paper they sit on -- less than the variation inside a
single object, i.e. no usable signal. Cropping to the work region and
upscaling before inference spends the range on the objects instead:

    full frame   objects 18-22, paper 23        (unusable)
    cropped      objects 78/83/159, paper 120   (clearly separated)

Controls
--------
    drag / arrow keys   orbit the 3D view
    scroll / +-         zoom
    click an object     select it (camera view or 3D panel)
    tab                 cycle selection
    a                   show all objects vs just the selected one
    d                   toggle the depth map underlay
    s                   save a frame
    q                   quit

    python pose3d.py
    python pose3d.py --grid 9
    python pose3d.py --snapshot
"""
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch

from auto_pose import (
    DEVICE_FILE, GRID, MAX_COVER, WORK_W, WORK_H,
    Tracker, camera_index, describe, is_shadow, load_model, load_roi,
    save_roi, segment_all,
)

OUT_PATH = Path(__file__).with_name("pose3d_out.jpg")

PANEL = 540              # 3D panel is square, same height as the camera view

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

# The crop is upscaled before depth inference. The model resamples to its own
# working resolution anyway, but feeding it a larger crop keeps more detail in
# small objects; measured to matter for ~35 px cubes.
DEPTH_UPSCALE = 3

# How far the relative depth range is allowed to lift a shape in the 3D view,
# as a fraction of the object's own width. Purely a display scale -- there is
# no metric depth to honour, and an arbitrary large value would look precise
# while meaning nothing.
# Pixels of rendered height for an object one HEIGHT_UNIT above the table.
# Display only: there is no metric depth to honour, so this is chosen to make
# differences visible, not to assert a size.
HEIGHT_RENDER_PX = 45.0

PALETTE = [(0, 255, 255), (0, 220, 0), (255, 150, 0), (255, 0, 200),
           (80, 200, 255), (200, 100, 255), (0, 130, 255), (180, 255, 60)]


def load_depth_model():
    from transformers import pipeline

    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {DEPTH_MODEL} on {dev} ...")
    t0 = time.time()
    pipe = pipeline("depth-estimation", model=DEPTH_MODEL, device=dev)
    print(f"  ready in {time.time() - t0:.0f}s")
    return pipe


def depth_for_roi(pipe, frame_bgr, roi):
    """Relative depth for the work region only, at full frame coordinates.

    Returns a float array the size of the frame, valid inside the ROI and NaN
    outside it. Cropping before inference is not an optimisation -- on the
    full frame these objects are indistinguishable from the paper (see the
    module docstring).
    """
    h, w = frame_bgr.shape[:2]
    x0, y0 = int(roi[0] * w), int(roi[1] * h)
    x1, y1 = int(roi[2] * w), int(roi[3] * h)
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    big = cv2.resize(crop, (crop.shape[1] * DEPTH_UPSCALE,
                            crop.shape[0] * DEPTH_UPSCALE),
                     interpolation=cv2.INTER_CUBIC)
    out = pipe(PIL.Image.fromarray(cv2.cvtColor(big, cv2.COLOR_BGR2RGB)))
    d = np.array(out["depth"]).astype(np.float32)
    d = cv2.resize(d, (crop.shape[1], crop.shape[0]))

    full = np.full((h, w), np.nan, np.float32)
    full[y0:y1, x0:x1] = d
    return full


def fit_table_plane(depth, roi, frame_shape):
    """Least-squares plane through the work surface, ignoring objects.

    The camera views the table obliquely, so bare paper is NOT at constant
    depth: measured across an empty area it ramped 47 -> 123, a 76-level
    gradient with nothing on it. That is larger than any object's height
    signal, so height has to be measured against this fitted plane, never
    against a single "table depth" value.

    Objects sit ABOVE the surface, so they are outliers on one side. The fit
    iterates, each time keeping the lower residuals, which walks the plane
    down onto the surface instead of splitting the difference.
    """
    if depth is None:
        return None
    h, w = frame_shape[:2]
    x0, y0 = int(roi[0] * w), int(roi[1] * h)
    x1, y1 = int(roi[2] * w), int(roi[3] * h)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    z = depth[y0:y1, x0:x1]
    ok = ~np.isnan(z)
    if ok.sum() < 200:
        return None
    X = xs[ok].ravel().astype(np.float64)
    Y = ys[ok].ravel().astype(np.float64)
    Z = z[ok].ravel().astype(np.float64)

    keep = np.ones(Z.shape, bool)
    coef = None
    for _ in range(6):
        A = np.column_stack([X[keep], Y[keep], np.ones(int(keep.sum()))])
        try:
            coef, *_ = np.linalg.lstsq(A, Z[keep], rcond=None)
        except np.linalg.LinAlgError:
            return None
        resid = Z - (coef[0] * X + coef[1] * Y + coef[2])
        keep = resid < np.percentile(resid, 75)
    return tuple(float(c) for c in coef)


def height_above_plane(depth, plane, shape):
    """Depth expressed as height above the fitted work surface.

    One shared scale for every object, which is the whole point: normalising
    each object separately (the previous approach) stretched a flat slab and a
    standing block to the same rendered height, destroying exactly the
    difference the view is meant to show.
    """
    if depth is None or plane is None:
        return None
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    return depth - (plane[0] * xs + plane[1] * ys + plane[2])


def object_height(obj, height_map):
    """Height of one object above the surface, in relative depth units.

    Returned with an explicit confidence, because on this hardware it is
    frequently wrong. Measured against three blocks of known size (one 60 mm
    standing, two 30 mm lying flat) the estimates correlated with truth at
    r = +0.22 -- the 60 mm block ranked SECOND of three. The brightest and
    most saturated object read tallest, i.e. the model keys on appearance, not
    geometry, for small textureless objects viewed top-down.

    So this is reported as an unreliable indication and never drawn as if it
    were a measurement. Real heights need the RealSense.
    """
    if height_map is None:
        return None
    vals = height_map[obj["mask"]]
    vals = vals[~np.isnan(vals)]
    if vals.size < 20:
        return None
    return {
        "top": float(np.percentile(vals, 90)),
        "median": float(np.percentile(vals, 50)),
        "spread": float(np.percentile(vals, 90) - np.percentile(vals, 10)),
    }


def object_relief(obj, height_map, scale):
    """Per-pixel height inside one mask, on the SHARED scale.

    `scale` divides every object by the same number, so relative heights
    between objects survive into the render.
    """
    if height_map is None or scale is None or scale <= 0:
        return None
    mask = obj["mask"]
    relief = np.full(mask.shape, np.nan, np.float32)
    vals = height_map[mask] / scale
    relief[mask] = np.clip(vals, -0.2, 1.5)
    if np.all(np.isnan(relief[mask])):
        return None
    return relief


def tilt_from_relief(obj, relief):
    """Direction and strength of the height gradient across an object.

    Fits a plane to the height inside the mask. Direction says which way it
    leans, magnitude how strongly -- both in relative units, never degrees,
    since degrees would imply a metric depth that does not exist.
    """
    if relief is None:
        return None
    ys, xs = np.nonzero(obj["mask"])
    z = relief[ys, xs]
    good = ~np.isnan(z)
    if good.sum() < 30:
        return None
    xs, ys, z = xs[good], ys[good], z[good]
    A = np.column_stack([xs - xs.mean(), ys - ys.mean(), np.ones(xs.size)])
    try:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b = float(coef[0]), float(coef[1])
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1)
    strength = float(np.hypot(a, b) * span)
    direction = float(np.rad2deg(np.arctan2(b, a)) % 360)
    return {"direction_deg": direction,
            "strength": min(strength, 1.0),
            "flat": strength < 0.15}


# --- 3D rendering -------------------------------------------------------


def rotation(yaw_deg, pitch_deg):
    """Camera orbit matrix. Yaw about the table normal, then pitch."""
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p), np.cos(p)]])
    return rx @ rz


def build_mesh(obj, relief, step=1):
    """A 3D surface for one object, from its mask and height map.

    step=1 by default: the previous step=3 sampled every third pixel, which on
    a 35 px object leaves ~12 points across it and renders as a scattered
    cloud rather than a recognisable shape. These masks are small enough that
    full sampling is cheap.

    Every vertex is a real mask pixel -- x/y from the measured outline, z from
    that pixel's height above the fitted table plane. Nothing is fitted or
    assumed, so a flat object renders flat.
    """
    mask = obj["mask"]
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    cx, cy = obj["centroid"]

    pts, cols = [], []
    for y in range(ys.min(), ys.max() + 1, step):
        for x in range(xs.min(), xs.max() + 1, step):
            if not mask[y, x]:
                continue
            z = 0.0
            if relief is not None and not np.isnan(relief[y, x]):
                z = float(relief[y, x]) * HEIGHT_RENDER_PX
            pts.append((x - cx, y - cy, max(z, 0.0)))
            cols.append(min(max(z / max(HEIGHT_RENDER_PX, 1e-6), 0.0), 1.0))

            # Skirt down to the table on boundary pixels, so an object reads
            # as a solid body sitting on the surface rather than a floating
            # sheet of points.
            if z > 2 and _is_edge(mask, x, y):
                for t in np.linspace(0, z, max(2, int(z / 2))):
                    pts.append((x - cx, y - cy, float(t)))
                    cols.append(min(max(t / max(HEIGHT_RENDER_PX, 1e-6),
                                        0.0), 1.0))
    if not pts:
        return None
    return np.array(pts, np.float32), np.array(cols, np.float32)


def _is_edge(mask, x, y):
    h, w = mask.shape
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < w and 0 <= ny < h) or not mask[ny, nx]:
            return True
    return False


def render_3d(objects, reliefs, selected, yaw, pitch, zoom, size=PANEL,
              show_all=True):
    """Orthographic 3D panel.

    Deliberately a simple painter's-algorithm point render rather than a real
    3D engine: the geometry here is a relief surface, not a solid, and a
    heavier renderer would dress up relative data as something more.
    """
    canvas = np.full((size, size, 3), 24, np.uint8)
    R = rotation(yaw, pitch)

    # Ground grid, so orbiting has a reference and the table plane is visible.
    half = size * 0.42 / zoom
    for i in range(-4, 5):
        t = i / 4.0 * half
        for (p0, p1) in (((-half, t, 0), (half, t, 0)),
                         ((t, -half, 0), (t, half, 0))):
            a = R @ np.array(p0)
            b = R @ np.array(p1)
            pa = (int(size / 2 + a[0] * zoom), int(size / 2 + a[1] * zoom))
            pb = (int(size / 2 + b[0] * zoom), int(size / 2 + b[1] * zoom))
            cv2.line(canvas, pa, pb, (44, 44, 44), 1)

    order = []
    for obj in objects:
        if not show_all and obj["id"] != selected:
            continue
        mesh = build_mesh(obj, reliefs.get(obj["id"]))
        if mesh is None:
            continue
        pts, cols = mesh
        cx, cy = obj["centroid"]
        # Lay objects out at their real table positions, centred on the group.
        offset = np.array([cx, cy, 0.0], np.float32)
        order.append((obj, pts, cols, offset))

    if not order:
        cv2.putText(canvas, "no objects", (16, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
        return canvas

    centre = np.mean([o[3] for o in order], axis=0)

    drawn = []
    for obj, pts, cols, offset in order:
        world = pts + (offset - centre)
        cam = world @ R.T
        for (p, c) in zip(cam, cols):
            drawn.append((p[2], p, c, obj))

    # Painter's algorithm: far points first so near ones cover them.
    drawn.sort(key=lambda d: d[0])
    for _, p, c, obj in drawn:
        x = int(size / 2 + p[0] * zoom)
        y = int(size / 2 + p[1] * zoom)
        if not (0 <= x < size and 0 <= y < size):
            continue
        base = np.array(PALETTE[obj["id"] % len(PALETTE)], np.float32)
        shade = 0.45 + 0.55 * float(c)
        colour = tuple(int(v) for v in np.clip(base * shade, 0, 255))
        r = 2 if obj["id"] == selected else 1
        cv2.circle(canvas, (x, y), r, colour, -1)

    # Grasp axis of the selected object, drawn in the table plane.
    for obj, pts, cols, offset in order:
        if obj["id"] != selected:
            continue
        ang = np.deg2rad(obj["grasp_deg"])
        half_len = obj["grasp_length"] / 2
        for sign, colour, width_px in ((1, (0, 230, 0), 2),):
            p0 = np.array([np.cos(ang) * -half_len,
                           np.sin(ang) * -half_len, 0.0])
            p1 = np.array([np.cos(ang) * half_len,
                           np.sin(ang) * half_len, 0.0])
            off = offset - centre
            a = R @ (p0 + off)
            b = R @ (p1 + off)
            cv2.line(canvas,
                     (int(size / 2 + a[0] * zoom), int(size / 2 + a[1] * zoom)),
                     (int(size / 2 + b[0] * zoom), int(size / 2 + b[1] * zoom)),
                     colour, width_px)

    return canvas


def annotate_camera(frame, objects, selected, depth=None, show_depth=False):
    view = frame.copy()
    if show_depth and depth is not None:
        valid = ~np.isnan(depth)
        if valid.any():
            norm = np.zeros(depth.shape, np.uint8)
            d = depth[valid]
            lo, hi = np.percentile(d, 2), np.percentile(d, 98)
            scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
            norm[valid] = (scaled[valid] * 255).astype(np.uint8)
            colour = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
            view[valid] = cv2.addWeighted(view, 0.35, colour, 0.65, 0)[valid]

    for obj in objects:
        colour = PALETTE[obj["id"] % len(PALETTE)]
        thickness = 3 if obj["id"] == selected else 1
        cv2.drawContours(view, [obj["contour"]], -1, colour, thickness)
        cx, cy = obj["centroid"]
        ang = np.deg2rad(obj["grasp_deg"])
        L = obj["grasp_length"] * 0.45
        cv2.line(view,
                 (int(cx - np.cos(ang) * L), int(cy - np.sin(ang) * L)),
                 (int(cx + np.cos(ang) * L), int(cy + np.sin(ang) * L)),
                 (0, 230, 0), 2)
        tag = f"#{obj['id']}"
        cv2.putText(view, tag, (int(cx) + 10, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(view, tag, (int(cx) + 10, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return view


def info_panel(obj, tilt, height, size=PANEL):
    """Text block for the selected object, stating what each number is."""
    panel = np.full((size, 320, 3), 18, np.uint8)
    if obj is None:
        cv2.putText(panel, "no selection", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return panel

    colour = PALETTE[obj["id"] % len(PALETTE)]
    lines = [
        (f"object #{obj['id']}", colour),
        (obj["name"], (220, 220, 220)),
        ("", None),
        ("MEASURED", (120, 220, 120)),
        (f"  grasp yaw    {obj['grasp_deg']:.0f} deg", (220, 220, 220)),
        (f"  jaw width    {obj['grasp_width']:.0f} px", (220, 220, 220)),
        (f"  length       {obj['grasp_length']:.0f} px", (220, 220, 220)),
        (f"  solidity     {obj['solidity']:.2f}", (220, 220, 220)),
        (f"  fits jaws    {'yes' if obj['grasp_feasible'] else 'NO'}",
         (220, 220, 220) if obj["grasp_feasible"] else (80, 80, 255)),
        ("", None),
        ("UNRELIABLE (monocular)", (90, 160, 230)),
    ]
    if height is not None:
        lines.append((f"  height ~{height['top']:.0f} (rel units)",
                      (170, 170, 170)))
    if tilt is None:
        lines.append(("  depth: not resolved", (170, 170, 170)))
    elif tilt["flat"]:
        lines.append(("  lies flat", (170, 170, 170)))
    else:
        lines.append((f"  leans toward {tilt['direction_deg']:.0f} deg",
                      (170, 170, 170)))
    # Do not let this read as a measurement: against blocks of known size the
    # estimate correlated with truth at r=+0.22 and mis-ranked the tallest.
    lines.append(("  height often WRONG:", (80, 110, 210)))
    lines.append(("  r=+0.22 vs known sizes", (80, 110, 210)))
    lines += [
        ("", None),
        ("NOT MEASURED", (110, 110, 200)),
        ("  absolute height/size", (150, 150, 150)),
        ("  metric 3D pose", (150, 150, 150)),
        ("  (needs RealSense depth)", (150, 150, 150)),
    ]

    y = 28
    for text, col in lines:
        if text:
            cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, col, 1)
        y += 21
    return panel


# --- entry points -------------------------------------------------------


def read_frame(cap):
    for _ in range(4):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        return None
    return cv2.resize(frame, (WORK_W, WORK_H))


def analyse(model, processor, depth_pipe, frame, grid, roi, tracker):
    found = segment_all(model, processor, frame, grid, MAX_COVER, roi)
    objects = [d for d in (describe(m, s) for m, s in found) if d]
    tracked = tracker.update(objects)

    depth = depth_for_roi(depth_pipe, frame, roi)
    plane = fit_table_plane(depth, roi, frame.shape)
    height_map = height_above_plane(depth, plane, frame.shape)

    # One scale shared by every object, taken from the tallest thing present.
    # Per-object normalisation is what made a flat slab and a standing block
    # render identically.
    heights = {}
    for obj in tracked:
        heights[obj["id"]] = object_height(obj, height_map)
    tops = [h["top"] for h in heights.values() if h]
    scale = max(tops) if tops else None

    reliefs, tilts = {}, {}
    for obj in tracked:
        rel = object_relief(obj, height_map, scale)
        reliefs[obj["id"]] = rel
        tilts[obj["id"]] = tilt_from_relief(obj, rel)
    return tracked, depth, reliefs, tilts, heights


def compose(frame, objects, depth, reliefs, tilts, heights, selected, yaw,
            pitch, zoom, show_all, show_depth, ms):
    cam = annotate_camera(frame, objects, selected, depth, show_depth)
    panel = render_3d(objects, reliefs, selected, yaw, pitch, zoom,
                      show_all=show_all)
    sel = next((o for o in objects if o["id"] == selected), None)
    info = info_panel(sel, tilts.get(selected) if sel else None,
                      heights.get(selected) if sel else None)

    cv2.putText(cam, f"{len(objects)} objects   {ms:.0f}ms   "
                     f"drag=orbit  tab=select  a=all  d=depth",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(panel, f"3D  yaw {yaw:.0f}  pitch {pitch:.0f}   "
                       f"relative depth",
                (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return np.hstack([cam, panel, info])


def live(index, grid, roi):
    model, processor = load_model()
    depth_pipe = load_depth_model()

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"camera index {index} would not open")
        return

    tracker = Tracker()
    shared = {"objects": [], "depth": None, "reliefs": {}, "tilts": {},
              "heights": {}, "busy": False, "ms": 0.0, "roi": roi}
    lock = threading.Lock()
    view = {"yaw": 35.0, "pitch": 62.0, "zoom": 1.5, "selected": 0,
            "all": True, "depth": False, "drag": None}

    def worker(frame):
        t0 = time.time()
        try:
            tracked, depth, reliefs, tilts, heights = analyse(
                model, processor, depth_pipe, frame, grid,
                shared["roi"], tracker)
        except Exception as exc:                      # noqa: BLE001
            print(f"  analysis error: {exc}")
            tracked, depth, reliefs, tilts, heights = [], None, {}, {}, {}
        with lock:
            shared.update(objects=tracked, depth=depth, reliefs=reliefs,
                          tilts=tilts, heights=heights,
                          ms=(time.time() - t0) * 1000, busy=False)

    def on_mouse(event, x, y, flags, _p):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < WORK_W:
                # Click in the camera view selects the object under the cursor.
                with lock:
                    for obj in shared["objects"]:
                        if obj["mask"][min(y, WORK_H - 1), min(x, WORK_W - 1)]:
                            view["selected"] = obj["id"]
                            break
            else:
                view["drag"] = (x, y, view["yaw"], view["pitch"])
        elif event == cv2.EVENT_MOUSEMOVE and view["drag"]:
            x0, y0, yaw0, pitch0 = view["drag"]
            view["yaw"] = (yaw0 + (x - x0) * 0.5) % 360
            view["pitch"] = float(np.clip(pitch0 + (y - y0) * 0.5, 5, 89))
        elif event == cv2.EVENT_LBUTTONUP:
            view["drag"] = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            view["zoom"] = float(np.clip(
                view["zoom"] * (1.1 if flags > 0 else 0.9), 0.3, 8.0))

    cv2.namedWindow("pose3d")
    cv2.setMouseCallback("pose3d", on_mouse)
    print("drag right panel = orbit, scroll = zoom, click camera = select,")
    print("tab cycle, a all/one, d depth overlay, r work region, s save, q quit")

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            if not shared["busy"]:
                shared["busy"] = True
                threading.Thread(target=worker, args=(frame.copy(),),
                                 daemon=True).start()

            with lock:
                objects = shared["objects"]
                depth, reliefs, tilts = (shared["depth"], shared["reliefs"],
                                         shared["tilts"])
                heights, ms = shared["heights"], shared["ms"]
            out = compose(frame, objects, depth, reliefs, tilts, heights,
                          view["selected"], view["yaw"], view["pitch"],
                          view["zoom"], view["all"], view["depth"], ms)
            cv2.imshow("pose3d", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 9 and objects:          # tab
                ids = sorted(o["id"] for o in objects)
                if view["selected"] in ids:
                    view["selected"] = ids[(ids.index(view["selected"]) + 1)
                                           % len(ids)]
                else:
                    view["selected"] = ids[0]
            if key == ord("a"):
                view["all"] = not view["all"]
            if key == ord("d"):
                view["depth"] = not view["depth"]
            if key in (ord("+"), ord("=")):
                view["zoom"] = min(view["zoom"] * 1.2, 8.0)
            if key == ord("-"):
                view["zoom"] = max(view["zoom"] / 1.2, 0.3)
            if key == 81:
                view["yaw"] = (view["yaw"] - 5) % 360
            if key == 83:
                view["yaw"] = (view["yaw"] + 5) % 360
            if key == 82:
                view["pitch"] = float(np.clip(view["pitch"] + 5, 5, 89))
            if key == 84:
                view["pitch"] = float(np.clip(view["pitch"] - 5, 5, 89))
            if key == ord("s"):
                cv2.imwrite(str(OUT_PATH), out)
                print(f"saved {OUT_PATH.name}")
            if key == ord("r"):
                box = cv2.selectROI("pose3d", frame, showCrosshair=True)
                if box[2] > 10 and box[3] > 10:
                    h, w = frame.shape[:2]
                    new = (box[0] / w, box[1] / h,
                           (box[0] + box[2]) / w, (box[1] + box[3]) / h)
                    shared["roi"] = new
                    save_roi(new)
                    tracker.tracks.clear()
                    print(f"  work region {tuple(round(v, 2) for v in new)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def snapshot(index, grid, roi):
    model, processor = load_model()
    depth_pipe = load_depth_model()
    cap = cv2.VideoCapture(index)
    frame = read_frame(cap)
    cap.release()
    if frame is None:
        print("no frame")
        return
    tracker = Tracker()
    t0 = time.time()
    objects, depth, reliefs, tilts, heights = analyse(
        model, processor, depth_pipe, frame, grid, roi, tracker)
    print(f"{len(objects)} object(s) in {time.time() - t0:.1f}s")
    for obj in sorted(objects, key=lambda o: -o["area"]):
        t = tilts.get(obj["id"])
        cx, cy = obj["centroid"]
        print(f"  #{obj['id']} {obj['name']:20s} at px({cx:.0f},{cy:.0f})")
        print(f"      grasp yaw {obj['grasp_deg']:5.1f} deg  "
              f"width {obj['grasp_width']:.0f} px")
        hh = heights.get(obj["id"])
        if hh:
            print(f"      height ~{hh['top']:.0f} rel units (UNRELIABLE)")
        if t is None:
            print("      depth: not resolved")
        elif t["flat"]:
            print(f"      lies flat (gradient {t['strength']:.2f})")
        else:
            print(f"      leans toward {t['direction_deg']:.0f} deg "
                  f"(strength {t['strength']:.2f}, relative)")
    out = compose(frame, objects, depth, reliefs, tilts, heights,
                  objects[0]["id"] if objects else 0,
                  35.0, 62.0, 1.5, True, False, 0.0)
    cv2.imwrite(str(OUT_PATH), out)
    print(f"\nwrote {OUT_PATH.name}")


if __name__ == "__main__":
    argv = sys.argv
    idx = camera_index(argv)
    grid = int(argv[argv.index("--grid") + 1]) if "--grid" in argv else GRID
    roi = load_roi()
    if "--snapshot" in argv:
        snapshot(idx, grid, roi)
    else:
        live(idx, grid, roi)
