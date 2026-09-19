"""Segment every object and measure its grasp orientation. No prompts, no colour.

Standalone: imports nothing from the pick pipeline and talks to no robot.

Why this exists
---------------
Two earlier approaches were tried and rejected for concrete reasons:

  colour thresholds  Failed on the first real frame: the red hook and the
                     wooden desk measure hue 10.2 vs 10.4, so the threshold
                     segmented the whole desk. Cannot survive a pile either,
                     where objects occlude each other and share colours.

  SAM 3 text prompt  "red plastic hook" stops matching the moment the hook is
                     turned so it no longer looks like a hook, and there is no
                     single phrase covering several objects of different
                     shapes, colours and sizes.

This file prompts SAM 3 with a GRID OF POINTS instead of words. Every point is
a "what is the object here?" question, so segmentation depends on nothing but
image structure: not colour, not shape, not orientation, not a description.
Rotating an object changes its mask outline and its measured angle, never
whether it is found.

How it works
------------
The image is encoded ONCE (~0.1 s), then each point prompt reuses those
features (~120 ms). Points landing on the same object return near-identical
masks, so duplicates are removed by IoU. What survives is one mask per object.

Objects are then filtered by area and by how much of the frame they cover: a
grid point on the desk or the notepad returns a huge "background" mask, which
is not a graspable object.

Orientation, per object
-----------------------
  PCA axis      direction of greatest pixel spread. For a hook or an L this
                points along the shape's diagonal, NOT where jaws fit.
  min-area box  tightest rotated rectangle; dominated by bounding extremes on
                a non-convex shape.
  GRASP AXIS    where two parallel jaws actually close, found by scanning
                angles for the narrowest width the jaws can span.

The grasp axis is what a gripper should use. On the test hook PCA and grasp
disagreed by 90 degrees, which is the entire reason all three are computed.

2D only for now. Per-instance masks and principal axes are the interfaces a
depth map later turns into 6-DoF.

    python auto_pose.py                  # live
    python auto_pose.py --snapshot       # one frame -> auto_pose_out.jpg
    python auto_pose.py --grid 8         # denser point grid (slower)
    python auto_pose.py --max-cover 0.25 # reject masks covering more than this
    python auto_pose.py --full-frame     # search the whole frame, not the ROI

Press r in the live view to drag the work region over wherever the objects
are. Without it SAM segments the whole desk, correctly but uselessly.
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

DEVICE_FILE = Path(__file__).with_name("overhead_device.json")
OUT_PATH = Path(__file__).with_name("auto_pose_out.jpg")

WORK_W, WORK_H = 960, 540

# Point grid density.
#
# This must be finer than the SMALLEST object, or that object falls between
# rows and is silently missed. Observed directly: at 7x7 over a 231x286 px
# work region the vertical spacing is 41 px while the cubes are ~35 px tall,
# and the blue cube vanished from the results — not because of shadows or
# thresholds, but because no point ever landed on it. Probed directly it
# segmented at q=0.94.
#
# Spacing is (region size / grid), so check it against your objects:
#     9x9  over 231x286 -> 26 x 32 px
#    11x11 over 231x286 -> 21 x 26 px
#
# Cost is linear in the number of points (~120 ms each), so raise this only
# until the smallest object is reliably caught.
GRID = 9

# An object must be at least this many pixels, and cover no more than this
# fraction of the WORK REGION. SAM segments everything it is asked about, so
# without an upper bound a grid point on the desk returns a perfectly valid,
# high-confidence mask of the desk. The first run found the laptop, the
# keyboard, the chair and the floor -- all correct, none graspable.
MIN_AREA_PX = 300
MAX_COVER = 0.30

# Work region, as fractions of the frame (x0, y0, x1, y1).
#
# Restricting WHERE the grid is placed is more reliable than filtering results
# afterwards: a mask is rejected by size or position only after the model has
# spent 120 ms on it, and a large background mask can still pass a size filter
# while being useless. Set this to the area the objects actually sit in --
# press r in the live view to drag a new one.
DEFAULT_ROI = (0.33, 0.35, 0.57, 0.88)
# Grid points decoded per model call. Measured on MPS at grid 13 (169
# points): 1 pt/call = 102 ms/pt, 32 = 56, 64 = 55, all-169 = 72. The win is
# flat between 32 and 64 and reverses beyond it.
PROMPT_BATCH = 64

ROI_FILE = Path(__file__).with_name("auto_pose_roi.json")


def load_roi():
    if ROI_FILE.exists():
        try:
            return tuple(json.loads(ROI_FILE.read_text())["roi"])
        except Exception:                             # noqa: BLE001
            pass
    return DEFAULT_ROI


def save_roi(roi):
    ROI_FILE.write_text(json.dumps({"roi": list(roi)}, indent=2))

# Two masks overlapping by more than this are the same object seen from two
# grid points; the higher-scoring one is kept.
DEDUP_IOU = 0.5

# Minimum mask quality the model reports. Below this the point landed on an
# ambiguous boundary.
MIN_IOU = 0.7

# --- shadow rejection -------------------------------------------------------
#
# Standing objects cast shadows, and SAM segments a shadow as readily as an
# object: it is a coherent region with a real boundary. Two distinct failures
# were observed with three cubes stood upright — a shadow returned as its own
# object, and an object's mask bleeding into its adjoining shadow.
#
# Neither is fixable by size or position. But a shadow is the SURFACE seen
# under less light, so it keeps the surface's colour and only loses
# saturation, while a coloured object keeps its own. Measured on the failing
# frame:
#
#     object faces     saturation 146-185
#     their shadows    saturation  62- 80
#     bare notepad     saturation  36
#
# So a region whose median saturation is low, yet which is darker than its
# surroundings, is a shadow rather than a thing to pick up. The margin here is
# wide; MIN_OBJECT_SAT sits between the two clusters.
#
# This is a photometric test on the MASK SAM returns, not a colour rule for
# finding objects: no hue is named, and an object of any colour passes as long
# as it is more saturated than its own shadow.
MIN_OBJECT_SAT = 100     # median saturation below this is suspect
SHADOW_DARK_RATIO = 0.9  # and darker than the surrounding surface

MAX_JAW_PX = 140
ANGLE_STEP_DEG = 3
ISOTROPY_RATIO = 1.15

IOU_MATCH = 0.3
CENTROID_MATCH_PX = 60
TRACK_STALE_FRAMES = 10


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def camera_index(argv):
    if "--index" in argv:
        return int(argv[argv.index("--index") + 1])
    if DEVICE_FILE.exists():
        try:
            return json.loads(DEVICE_FILE.read_text())["index"]
        except Exception:                             # noqa: BLE001
            pass
    return 0


def load_model():
    """Build SAM 3 with the point-promptable (SAM 1 task) head attached.

    enable_inst_interactivity=True is what provides predict_inst. Note that
    build_sam3_image_model does NOT honour its `device` argument -- every
    parameter comes back on CPU and inference then dies on a dtype mismatch
    against MPS inputs, so the explicit .to() is required.
    """
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    dev = device()
    print(f"loading SAM 3 on {dev} ...")
    t0 = time.time()
    model = build_sam3_image_model(
        device=dev, enable_inst_interactivity=True).to(dev).eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params "
          f"in {time.time() - t0:.0f}s")
    return model, Sam3Processor(model, device=dev, confidence_threshold=0.4)


def segment_all(model, processor, frame_bgr, grid=GRID, max_cover=MAX_COVER,
                roi=None, extra_points=None):
    """Every distinct object in the frame, via a grid of point prompts.

    Returns a list of (mask, score). Nothing here references colour, shape or
    any object description -- each grid point simply asks the model what
    object occupies that pixel.

    `extra_points` are additional (x, y) pixels to prompt, beyond the grid.
    A uniform grid misses any object narrower than its spacing: at grid 13 on
    a 960x540 frame the spacing is 74 px, and a 54 px block simply falls
    between the points -- which is why SAM reported the power strip and the
    adapter but not the object being picked. Raising the grid to 19 would
    guarantee a hit but costs 361 prompts instead of 169, more than doubling
    a pass. Prompting the few pixels where other detectors ALREADY say
    something is costs one prompt each and cannot miss.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # A PIL image, not a numpy array: Sam3Processor.set_image reads dimensions
    # with image.shape[-2:], which on an HWC array yields (width, channels)
    # and silently produces empty masks of the wrong shape.
    state = processor.set_image(PIL.Image.fromarray(rgb))

    h, w = frame_bgr.shape[:2]
    rx0, ry0, rx1, ry1 = roi or (0.0, 0.0, 1.0, 1.0)
    x0, y0 = int(rx0 * w), int(ry0 * h)
    x1, y1 = int(rx1 * w), int(ry1 * h)
    # Size limits are relative to the work region, not the whole frame: an
    # object filling the notepad is still a plausible object.
    region_area = float(max(1, (x1 - x0) * (y1 - y0)))
    found = []

    # All grid points up front, then decoded in batches.
    #
    # This was one predict_inst call PER POINT: 169 sequential decoder
    # invocations at grid 13, on an image the encoder had already embedded
    # once. The decoder accepts a batch of independent prompts, and feeding
    # it that way is ~2x faster end to end (measured: 102 ms/point one at a
    # time, 55 ms/point at batch 64, on MPS). Batches larger than ~64 lose
    # the gain again, so PROMPT_BATCH is a measured value, not a guess.
    pts = [(x0 + int((i + 0.5) * (x1 - x0) / grid),
            y0 + int((j + 0.5) * (y1 - y0) / grid))
           for i in range(grid) for j in range(grid)]
    if extra_points:
        h_f, w_f = frame_bgr.shape[:2]
        seen = set(pts)
        for px, py in extra_points:
            q = (int(px), int(py))
            if 0 <= q[0] < w_f and 0 <= q[1] < h_f and q not in seen:
                seen.add(q)
                pts.append(q)

    for start in range(0, len(pts), PROMPT_BATCH):
        chunk = pts[start:start + PROMPT_BATCH]
        coords = np.array([[p] for p in chunk])          # (B, 1, 2)
        labels = np.ones((len(chunk), 1), dtype=int)     # (B, 1)
        masks, ious, _ = model.predict_inst(
            state,
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
        masks = np.asarray(masks)
        ious = np.asarray(ious)
        # A batch of one still returns (1, 3, H, W); older single-prompt
        # calls returned (3, H, W). Normalise so the loop below is identical
        # either way.
        if masks.ndim == 3:
            masks = masks[None]
            ious = ious[None]

        for b in range(masks.shape[0]):
            # multimask_output returns three nested candidates (part,
            # subpart, whole). Take the one the model rates highest that is
            # also a plausible object rather than the background.
            order = np.argsort(ious[b])[::-1]
            for k in order:
                score = float(ious[b][k])
                if score < MIN_IOU:
                    continue
                mask = masks[b][k] > 0
                area = int(mask.sum())
                if area < MIN_AREA_PX or area / region_area > max_cover:
                    continue
                # Reject anything spilling well outside the work region: that
                # is the background showing through, not an object on it.
                ys_m, xs_m = np.nonzero(mask)
                inside = np.mean((xs_m >= x0) & (xs_m < x1)
                                 & (ys_m >= y0) & (ys_m < y1))
                if inside < 0.85:
                    continue
                if is_shadow(frame_bgr, mask):
                    continue
                found.append((mask, score))
                break

    return dedup(found)


def is_shadow(frame_bgr, mask):
    """True if this mask looks like a shadow rather than an object.

    Compares the masked region against a ring of surface just outside it. A
    shadow is unsaturated AND darker than that ring; a dark object (the blue
    cube reads value 99, darker than the paper) stays saturated and is kept.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    sat = float(np.median(hsv[:, :, 1][mask]))
    if sat >= MIN_OBJECT_SAT:
        return False

    # Surrounding surface: a dilated ring around the mask, excluding the mask.
    ring = cv2.dilate(mask.astype(np.uint8),
                      np.ones((21, 21), np.uint8)).astype(bool) & ~mask
    if ring.sum() < 50:
        return False
    inside_v = float(np.median(hsv[:, :, 2][mask]))
    ring_v = float(np.median(hsv[:, :, 2][ring]))
    return inside_v < ring_v * SHADOW_DARK_RATIO


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter / np.logical_or(a, b).sum())


def dedup(candidates):
    """Collapse masks that describe the same object.

    Several grid points land on one object and each returns its own mask, so
    without this a single object is reported many times. Highest score wins.
    """
    kept = []
    for mask, score in sorted(candidates, key=lambda c: -c[1]):
        if any(iou(mask, k[0]) > DEDUP_IOU for k in kept):
            continue
        kept.append((mask, score))
    return kept


# --- geometry -----------------------------------------------------------


def width_at_angle(points, angle_deg):
    """Object width perpendicular to `angle_deg`, and the span along it."""
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)],
                    [np.sin(t), np.cos(t)]])
    local = points @ rot
    return (float(local[:, 0].max() - local[:, 0].min()),
            float(local[:, 1].max() - local[:, 1].min()))


def grasp_axis(points, max_jaw_px=MAX_JAW_PX):
    """Angle where two parallel jaws close most easily.

    Returns (angle, width, feasible). `feasible` is False when no angle fits
    the jaws -- a real outcome, surfaced rather than hidden behind the
    least-bad number.
    """
    best = None
    for a in range(0, 180, ANGLE_STEP_DEG):
        w, _ = width_at_angle(points, a)
        if best is None or w < best[1]:
            best = (a, w)
    angle, width = best
    return float(angle), float(width), width <= max_jaw_px


def pca_axis(points):
    centred = points - points.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(centred.T))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    major = vecs[:, 0]
    angle = float(np.rad2deg(np.arctan2(major[1], major[0])) % 180)
    spreads = np.sqrt(np.maximum(vals, 0))
    return angle, float(spreads[0]), float(spreads[1])


def describe(mask, score):
    """Shape and grasp geometry for one instance mask."""
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < MIN_AREA_PX:
        return None

    ys, xs = np.nonzero(mask)
    cx, cy = float(xs.mean()), float(ys.mean())

    # Grasp geometry uses the FILLED mask, not the outline: jaws close on the
    # body, and for a concave shape those differ.
    pts = np.column_stack([xs, ys]).astype(np.float64)
    centred = pts - np.array([cx, cy])

    pca_deg, spread_major, spread_minor = pca_axis(pts)
    (bw, bh), box_deg = cv2.minAreaRect(contour)[1:]
    if bw < bh:
        box_deg += 90
    box_deg = float(box_deg % 180)

    g_deg, g_width, feasible = grasp_axis(centred)
    _, g_length = width_at_angle(centred, g_deg)

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0
    perimeter = cv2.arcLength(contour, True)
    circularity = (float(4 * np.pi * area / (perimeter ** 2))
                   if perimeter > 0 else 0.0)
    vertices = len(cv2.approxPolyDP(contour, 0.02 * perimeter, True))
    isotropic = (spread_minor > 0
                 and spread_major / spread_minor < ISOTROPY_RATIO)

    return {
        "mask": mask, "score": score, "contour": contour,
        "centroid": (cx, cy), "area": area,
        "pca_deg": pca_deg, "box_deg": box_deg,
        "grasp_deg": g_deg, "grasp_width": g_width,
        "grasp_length": g_length, "grasp_feasible": feasible,
        "solidity": solidity, "circularity": circularity,
        "vertices": vertices, "isotropic": isotropic,
        "name": classify(solidity, circularity, vertices, isotropic),
    }


def classify(solidity, circularity, vertices, isotropic):
    """A coarse shape label, for the human watching.

    Deliberately coarse: the grasp angle is what a gripper needs, and a
    confident-sounding name for an arbitrary blob would be worse than an
    honest vague one.
    """
    if circularity > 0.85 and isotropic:
        return "disc/circle"
    if solidity < 0.75:
        return "concave (hook/L/U)"
    if vertices == 3:
        return "triangle"
    if vertices == 4:
        return "square" if isotropic else "rectangle"
    if vertices >= 8 and circularity > 0.7:
        return "rounded"
    return f"polygon({vertices})"


# --- tracking -----------------------------------------------------------


class Tracker:
    """Keeps instance IDs stable across frames.

    Matched by mask IoU first, then centroid proximity. IoU alone loses a
    track when an object moves far in one frame -- exactly what happens when
    someone reorients it, which is the case this has to handle.
    """

    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def update(self, detections):
        unmatched = list(range(len(detections)))
        assigned = {}

        for tid, tr in list(self.tracks.items()):
            best, best_score = None, 0.0
            for di in unmatched:
                det = detections[di]
                score = iou(tr["mask"], det["mask"])
                if score < IOU_MATCH:
                    dx = det["centroid"][0] - tr["centroid"][0]
                    dy = det["centroid"][1] - tr["centroid"][1]
                    if (dx * dx + dy * dy) ** 0.5 < CENTROID_MATCH_PX:
                        score = max(score, 0.05)
                if score > best_score:
                    best, best_score = di, score
            if best is not None:
                det = detections[best]
                turned = abs(det["grasp_deg"] - tr["grasp_deg"])
                det["turned_deg"] = min(turned, 180 - turned)
                det["id"] = tid
                self.tracks[tid] = det
                assigned[tid] = det
                unmatched.remove(best)
            else:
                tr["missing"] = tr.get("missing", 0) + 1

        for di in unmatched:
            det = detections[di]
            det["id"] = self.next_id
            det["turned_deg"] = 0.0
            self.tracks[self.next_id] = det
            assigned[self.next_id] = det
            self.next_id += 1

        for tid, tr in list(self.tracks.items()):
            if tr.get("missing", 0) > TRACK_STALE_FRAMES:
                del self.tracks[tid]
            elif tid in assigned:
                tr["missing"] = 0

        return list(assigned.values())


# --- rendering ----------------------------------------------------------

PALETTE = [(0, 255, 255), (0, 220, 0), (255, 150, 0), (255, 0, 200),
           (80, 200, 255), (200, 100, 255), (0, 130, 255), (180, 255, 60)]


def draw(frame, objects, show_outline_only=False):
    if not show_outline_only:
        overlay = frame.copy()
        for obj in objects:
            overlay[obj["mask"]] = PALETTE[obj["id"] % len(PALETTE)]
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    for obj in objects:
        colour = PALETTE[obj["id"] % len(PALETTE)]
        cx, cy = obj["centroid"]
        # The outline is the thing to watch while rotating an object, so draw
        # it thick enough to read at a glance.
        cv2.drawContours(frame, [obj["contour"]], -1, colour, 2)

        def ray(angle_deg, length, col, thickness):
            t = np.deg2rad(angle_deg)
            dx, dy = np.cos(t) * length / 2, np.sin(t) * length / 2
            cv2.line(frame, (int(cx - dx), int(cy - dy)),
                     (int(cx + dx), int(cy + dy)), col, thickness)

        ray(obj["pca_deg"], obj["grasp_length"], (170, 170, 170), 1)
        # GREEN: the direction the jaws close along. This is grasp_deg itself,
        # and it is the line the gripper must line up with.
        ray(obj["grasp_deg"], obj["grasp_length"] * 0.9, (0, 230, 0), 2)
        # RED: the jaw OPENING, drawn across the object at grasp_deg + 90 with
        # length grasp_width. It shows how wide the gripper must open, NOT the
        # direction it closes in -- do not read this as the grasp axis.
        ray((obj["grasp_deg"] + 90) % 180, obj["grasp_width"],
            (0, 0, 255) if obj["grasp_feasible"] else (0, 0, 110), 3)
        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)

        lines = [f"#{obj['id']} {obj['name']}",
                 f"grasp {obj['grasp_deg']:.0f}deg w={obj['grasp_width']:.0f}"
                 + ("" if obj["grasp_feasible"] else " TOO WIDE")]
        if obj["isotropic"]:
            lines.append("no dominant axis")
        if obj.get("turned_deg", 0) > 5:
            lines.append(f"turned {obj['turned_deg']:.0f}deg")
        y = int(cy) - 6 - 15 * len(lines)
        for line in lines:
            y += 15
            cv2.putText(frame, line, (int(cx) + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(frame, line, (int(cx) + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return frame


def report(objects):
    if not objects:
        print("  nothing found")
        return
    for obj in sorted(objects, key=lambda o: -o["area"]):
        cx, cy = obj["centroid"]
        print(f"  #{obj['id']} {obj['name']:22s} q={obj['score']:.2f} "
              f"at px({cx:.0f},{cy:.0f}) area={obj['area']:.0f}")
        print(f"      grasp {obj['grasp_deg']:5.1f} deg  "
              f"width {obj['grasp_width']:5.1f} px  "
              f"{'fits jaws' if obj['grasp_feasible'] else 'TOO WIDE'}")
        print(f"      pca {obj['pca_deg']:5.1f}  box {obj['box_deg']:5.1f}  "
              f"solidity {obj['solidity']:.2f}")


# --- entry points -------------------------------------------------------


def read_frame(cap):
    for _ in range(4):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        return None
    return cv2.resize(frame, (WORK_W, WORK_H))


def snapshot(index, grid, max_cover, roi):
    model, processor = load_model()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"camera index {index} would not open")
        return
    frame = read_frame(cap)
    cap.release()
    if frame is None:
        print("no frame")
        return

    t0 = time.time()
    found = segment_all(model, processor, frame, grid, max_cover, roi)
    objects = [d for d in (describe(m, s) for m, s in found) if d]
    for i, o in enumerate(objects):
        o["id"] = i
    print(f"{grid}x{grid} grid in ROI {tuple(round(v,2) for v in roi)}: "
          f"{len(objects)} object(s) in {time.time() - t0:.1f}s")
    report(objects)
    vis = draw(frame.copy(), objects)
    h, w = frame.shape[:2]
    cv2.rectangle(vis, (int(roi[0]*w), int(roi[1]*h)),
                  (int(roi[2]*w), int(roi[3]*h)), (255, 255, 255), 1)
    cv2.imwrite(str(OUT_PATH), vis)
    print(f"\nwrote {OUT_PATH.name}")


def live(index, grid, max_cover, roi):
    model, processor = load_model()
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"camera index {index} would not open")
        return

    tracker = Tracker()
    shared = {"objects": [], "busy": False, "ms": 0.0, "roi": roi}
    lock = threading.Lock()

    def worker(frame):
        """Detection runs off the display thread.

        A full grid pass takes several seconds. Inline, the view would freeze
        for that long each time, which defeats watching an object being
        rotated.
        """
        t0 = time.time()
        try:
            found = segment_all(model, processor, frame, grid,
                                max_cover, shared["roi"])
            objects = [d for d in (describe(m, s) for m, s in found) if d]
            tracked = tracker.update(objects)
        except Exception as exc:                      # noqa: BLE001
            print(f"  detection error: {exc}")
            tracked = []
        with lock:
            shared["objects"] = tracked
            shared["ms"] = (time.time() - t0) * 1000
            shared["busy"] = False

    print("q quit, s save+report, o outline-only, +/- grid density, "
          "r set work region")
    print(f"  work region {tuple(round(v, 2) for v in shared['roi'])} "
          f"— press r to drag a new one")
    outline_only = False
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
                objects, ms = shared["objects"], shared["ms"]
            view = draw(frame.copy(), objects, outline_only)
            h, w = view.shape[:2]
            r = shared["roi"]
            cv2.rectangle(view, (int(r[0]*w), int(r[1]*h)),
                          (int(r[2]*w), int(r[3]*h)), (255, 255, 255), 1)
            cv2.putText(view,
                        f"{len(objects)} objects  {grid}x{grid} grid  "
                        f"{ms:.0f}ms  green=grasp red=jaw grey=pca",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.imshow("auto shape + orientation", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("o"):
                outline_only = not outline_only
            if key == ord("s"):
                cv2.imwrite(str(OUT_PATH), view)
                print(f"saved {OUT_PATH.name}")
                report(objects)
            if key in (ord("+"), ord("=")):
                grid = min(grid + 1, 12)
                print(f"  grid {grid}x{grid}")
            if key == ord("-"):
                grid = max(grid - 1, 3)
                print(f"  grid {grid}x{grid}")
            if key == ord("r"):
                # Blocking selector: the live loop pauses while the region is
                # dragged, which is fine and keeps the interaction obvious.
                box = cv2.selectROI("auto shape + orientation", frame,
                                    showCrosshair=True)
                cv2.destroyWindow("ROI selector")
                if box[2] > 10 and box[3] > 10:
                    h, w = frame.shape[:2]
                    new = (box[0]/w, box[1]/h,
                           (box[0]+box[2])/w, (box[1]+box[3])/h)
                    shared["roi"] = new
                    save_roi(new)
                    tracker.tracks.clear()
                    print(f"  work region {tuple(round(v,2) for v in new)} "
                          f"saved")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    argv = sys.argv
    idx = camera_index(argv)
    grid = int(argv[argv.index("--grid") + 1]) if "--grid" in argv else GRID
    cover = (float(argv[argv.index("--max-cover") + 1])
             if "--max-cover" in argv else MAX_COVER)
    roi = load_roi()
    if "--full-frame" in argv:
        roi = (0.0, 0.0, 1.0, 1.0)
    if "--snapshot" in argv:
        snapshot(idx, grid, cover, roi)
    else:
        live(idx, grid, cover, roi)
