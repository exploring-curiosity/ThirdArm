"""Learned instance segmentation and grasp orientation, from a text prompt.

Standalone: imports nothing from the pick pipeline and talks to no robot.

Why this replaces the colour-threshold version
----------------------------------------------
Colour thresholding cannot survive the actual objective, which is orientation
of objects in a PILE. In a pile objects occlude each other, share colours, and
cast shadows on each other, so "pixels near this hue" stops corresponding to
"one object". It also failed immediately on a cluttered desk: the red hook and
the wooden desk measured hue 10.2 and 10.4, so a hue threshold segmented the
whole desk.

SAM 3 (Meta, 841M params) segments from a TEXT prompt instead. It returns one
mask per instance, so two touching objects of the same colour come back as two
masks. Nothing about the object's colour is hardcoded anywhere in this file.

Orientation
-----------
Three angles are computed per instance, because they disagree and the
disagreement is the point:

  PCA axis      direction of greatest pixel spread. For a hook or an L this
                points along the shape's diagonal, which is NOT where jaws fit.
  min-area box  orientation of the tightest rotated rectangle. Good for
                rectangular parts; for a non-convex shape it is dominated by
                the bounding extremes.
  GRASP AXIS    where two parallel jaws actually close, found by scanning
                angles and taking the narrowest width the jaws can span.

The grasp axis is the one a gripper should use. A cube hides the difference;
the hook exposes it.

Tracking
--------
Instances are matched frame to frame by mask IoU and centroid distance, so an
object keeps its ID as the pile is disturbed, and a moved object is re-measured
rather than re-numbered. Detection runs on a worker thread so the display stays
smooth while the model is thinking.

2D only for now: angle is in the image plane. The interfaces (per-instance
masks, principal axes) are what a depth map later turns into 6-DoF.

    python sam_pose.py                          # live, default prompt
    python sam_pose.py --prompt "red hook"      # whatever you want to find
    python sam_pose.py --snapshot               # one frame -> sam_pose_out.jpg
    python sam_pose.py --conf 0.5               # confidence threshold
    python sam_pose.py --every 3                # detect every Nth frame
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
OUT_PATH = Path(__file__).with_name("sam_pose_out.jpg")

WORK_W, WORK_H = 960, 540

DEFAULT_PROMPT = "object"
DEFAULT_CONF = 0.4

# Jaw opening in pixels of the working frame. The grasp search rejects any
# angle whose width exceeds this. Pixels rather than mm because this file is
# deliberately robot-free; convert once a homography exists.
MAX_JAW_PX = 140

ANGLE_STEP_DEG = 3
MIN_AREA_PX = 200

# A shape whose two principal spreads are within this ratio has no meaningful
# long axis (a square, a disc). Reporting a confident angle for one of those
# would be false precision, so it is flagged instead.
ISOTROPY_RATIO = 1.15

# Track association. A detection matches an existing track if masks overlap,
# or failing that if the centroid is close - IoU alone drops a track when an
# object is nudged far in one frame.
IOU_MATCH = 0.3
CENTROID_MATCH_PX = 60
TRACK_STALE_FRAMES = 15


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


def load_model(dev=None):
    """Build SAM 3 and move it onto the accelerator.

    build_sam3_image_model takes a `device` argument but does NOT honour it --
    every parameter comes back on CPU, and inference then dies with a dtype
    mismatch against MPS inputs. The explicit .to() is the fix.
    """
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    dev = dev or device()
    print(f"loading SAM 3 on {dev} ...")
    t0 = time.time()
    model = build_sam3_image_model(device=dev).to(dev).eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params "
          f"in {time.time() - t0:.0f}s")
    return model, Sam3Processor, dev


def detect(processor, frame_bgr, prompt):
    """Per-instance masks for `prompt`. Returns list of (mask, score).

    The frame is handed over as a PIL image, not a numpy array: Sam3Processor
    reads dimensions with `image.shape[-2:]`, which on an HWC array yields
    (width, channels) and silently produces empty masks of the wrong shape.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    state = processor.set_image(PIL.Image.fromarray(rgb))
    state = processor.set_text_prompt(prompt, state)
    out = []
    for mask, score in zip(state["masks"], state["scores"]):
        m = mask.squeeze().detach().cpu().numpy().astype(bool)
        if m.sum() >= MIN_AREA_PX:
            out.append((m, float(score)))
    return out


# --- geometry -----------------------------------------------------------


def width_at_angle(points, angle_deg):
    """Object width perpendicular to `angle_deg`, and the span along it.

    This is what jaws closing along `angle_deg` must span. Rotating the points
    and taking the extent is exact, and simpler than reasoning about the
    contour directly.
    """
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)],
                    [np.sin(t), np.cos(t)]])
    local = points @ rot
    return (float(local[:, 0].max() - local[:, 0].min()),
            float(local[:, 1].max() - local[:, 1].min()))


def grasp_axis(points, max_jaw_px=MAX_JAW_PX):
    """Angle where two parallel jaws close most easily.

    Returns (angle, width, feasible). `feasible` is False when no angle fits
    the jaws -- a real outcome worth surfacing rather than returning the
    least-bad number as if it were fine.
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
    cov = np.cov(centred.T)
    vals, vecs = np.linalg.eigh(cov)
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

    # Grasp geometry is computed on the FILLED mask, not the contour outline:
    # the jaws close on the object's body, and for a concave shape the two
    # differ.
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
        "mask": mask,
        "score": score,
        "contour": contour,
        "centroid": (cx, cy),
        "area": area,
        "pca_deg": pca_deg,
        "box_deg": box_deg,
        "grasp_deg": g_deg,
        "grasp_width": g_width,
        "grasp_length": g_length,
        "grasp_feasible": feasible,
        "solidity": solidity,
        "circularity": circularity,
        "vertices": vertices,
        "isotropic": isotropic,
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


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter / np.logical_or(a, b).sum())


class Tracker:
    """Keeps instance IDs stable across frames.

    Matching is by mask IoU first, then centroid proximity. IoU alone loses a
    track when an object is moved far in one frame -- which is exactly what
    happens when someone disturbs the pile, the case this has to handle.
    """

    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.frame = 0

    def update(self, detections):
        self.frame += 1
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
                        # Weak positive, ranked below any real overlap.
                        score = max(score, 0.05)
                if score > best_score:
                    best, best_score = di, score
            if best is not None:
                det = detections[best]
                moved = ((det["centroid"][0] - tr["centroid"][0]) ** 2
                         + (det["centroid"][1] - tr["centroid"][1]) ** 2) ** 0.5
                turned = abs(det["grasp_deg"] - tr["grasp_deg"])
                turned = min(turned, 180 - turned)
                det.update({"id": tid, "moved_px": moved,
                            "turned_deg": turned, "age": tr["age"] + 1})
                self.tracks[tid] = det
                assigned[tid] = det
                unmatched.remove(best)
            else:
                tr["missing"] = tr.get("missing", 0) + 1

        for di in unmatched:
            det = detections[di]
            det.update({"id": self.next_id, "moved_px": 0.0,
                        "turned_deg": 0.0, "age": 0})
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

PALETTE = [(0, 255, 255), (0, 200, 0), (255, 140, 0), (255, 0, 200),
           (80, 200, 255), (200, 100, 255), (0, 120, 255)]


def draw(frame, objects):
    overlay = frame.copy()
    for obj in objects:
        colour = PALETTE[obj.get("id", 0) % len(PALETTE)]
        overlay[obj["mask"]] = colour
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    for obj in objects:
        colour = PALETTE[obj.get("id", 0) % len(PALETTE)]
        cx, cy = obj["centroid"]
        cv2.drawContours(frame, [obj["contour"]], -1, colour, 2)

        def ray(angle_deg, length, col, thickness):
            t = np.deg2rad(angle_deg)
            dx, dy = np.cos(t) * length / 2, np.sin(t) * length / 2
            cv2.line(frame, (int(cx - dx), int(cy - dy)),
                     (int(cx + dx), int(cy + dy)), col, thickness)

        # grey = PCA, green = grasp axis, red = jaw travel (perpendicular to
        # the grasp axis, the direction the jaws actually move).
        ray(obj["pca_deg"], obj["grasp_length"], (170, 170, 170), 2)
        ray(obj["grasp_deg"], obj["grasp_length"] * 0.9, (0, 230, 0), 2)
        jaw_colour = (0, 0, 255) if obj["grasp_feasible"] else (0, 0, 110)
        ray((obj["grasp_deg"] + 90) % 180, obj["grasp_width"], jaw_colour, 3)
        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)

        lines = [
            f"#{obj.get('id', '?')} {obj['name']} {obj['score']:.2f}",
            f"grasp {obj['grasp_deg']:.0f}deg w={obj['grasp_width']:.0f}px"
            + ("" if obj["grasp_feasible"] else " TOO WIDE"),
            f"pca {obj['pca_deg']:.0f}  solidity {obj['solidity']:.2f}",
        ]
        if obj["isotropic"]:
            lines.append("no dominant axis")
        if obj.get("turned_deg", 0) > 5:
            lines.append(f"turned {obj['turned_deg']:.0f}deg")
        y = int(cy) - 8 - 15 * len(lines)
        for line in lines:
            y += 15
            cv2.putText(frame, line, (int(cx) + 14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(frame, line, (int(cx) + 14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return frame


def report(objects):
    if not objects:
        print("  nothing matched the prompt")
        return
    for obj in sorted(objects, key=lambda o: -o["area"]):
        cx, cy = obj["centroid"]
        print(f"  #{obj.get('id','?')} {obj['name']:22s} "
              f"score {obj['score']:.2f} at px({cx:.0f},{cy:.0f}) "
              f"area={obj['area']:.0f}")
        print(f"      grasp {obj['grasp_deg']:5.1f} deg   "
              f"width {obj['grasp_width']:5.1f} px   "
              f"{'fits jaws' if obj['grasp_feasible'] else 'TOO WIDE FOR JAWS'}")
        print(f"      pca {obj['pca_deg']:5.1f}   box {obj['box_deg']:5.1f}   "
              f"(pca-vs-grasp {abs(obj['pca_deg'] - obj['grasp_deg']):.0f} deg)")
        if obj["isotropic"]:
            print("      NOTE: no dominant axis - any grasp angle is valid")


# --- entry points -------------------------------------------------------


def read_frame(cap):
    for _ in range(4):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        return None
    return cv2.resize(frame, (WORK_W, WORK_H))


def snapshot(index, prompt, conf):
    model, Processor, dev = load_model()
    processor = Processor(model, device=dev, confidence_threshold=conf)

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
    dets = detect(processor, frame, prompt)
    objects = [d for d in (describe(m, s) for m, s in dets) if d]
    for i, o in enumerate(objects):
        o["id"] = i
    print(f"prompt {prompt!r}: {len(objects)} instance(s) "
          f"in {time.time() - t0:.1f}s")
    report(objects)
    cv2.imwrite(str(OUT_PATH), draw(frame.copy(), objects))
    print(f"\nwrote {OUT_PATH.name}")


def live(index, prompt, conf, every):
    model, Processor, dev = load_model()
    processor = Processor(model, device=dev, confidence_threshold=conf)

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"camera index {index} would not open")
        return

    tracker = Tracker()
    shared = {"objects": [], "busy": False, "ms": 0.0, "prompt": prompt}
    lock = threading.Lock()

    def worker(frame):
        """Detection runs off the display thread.

        One SAM 3 pass takes ~1s. Running it inline would freeze the view for
        a second per frame and make the tracker useless for watching a pile
        being disturbed.
        """
        t0 = time.time()
        try:
            dets = detect(processor, frame, shared["prompt"])
            objects = [d for d in (describe(m, s) for m, s in dets) if d]
            tracked = tracker.update(objects)
        except Exception as exc:                      # noqa: BLE001
            print(f"  detection error: {exc}")
            tracked = []
        with lock:
            shared["objects"] = tracked
            shared["ms"] = (time.time() - t0) * 1000
            shared["busy"] = False

    print(f"prompt {prompt!r} — q quit, s save, p new prompt (in terminal)")
    frames = 0
    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            frames += 1

            if not shared["busy"] and frames % every == 0:
                shared["busy"] = True
                threading.Thread(target=worker, args=(frame.copy(),),
                                 daemon=True).start()

            with lock:
                objects = shared["objects"]
                ms = shared["ms"]
            view = draw(frame.copy(), objects)
            cv2.putText(view,
                        f"{shared['prompt']!r}  {len(objects)} obj  "
                        f"{ms:.0f}ms/detect  "
                        f"green=grasp red=jaw grey=pca",
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.imshow("SAM3 shape + orientation", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imwrite(str(OUT_PATH), view)
                print(f"saved {OUT_PATH.name}")
                report(objects)
            if key == ord("p"):
                new = input("new prompt: ").strip()
                if new:
                    shared["prompt"] = new
                    tracker.tracks.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    argv = sys.argv
    idx = camera_index(argv)
    prompt = (argv[argv.index("--prompt") + 1]
              if "--prompt" in argv else DEFAULT_PROMPT)
    conf = (float(argv[argv.index("--conf") + 1])
            if "--conf" in argv else DEFAULT_CONF)
    every = (int(argv[argv.index("--every") + 1])
             if "--every" in argv else 1)
    if "--snapshot" in argv:
        snapshot(idx, prompt, conf)
    else:
        live(idx, prompt, conf, every)
