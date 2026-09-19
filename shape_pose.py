"""Detect an object's 2D shape and grasp orientation from the overhead camera.

Standalone: imports nothing from the pick pipeline and talks to no robot, so
it cannot disturb the existing code. Camera index is read from
overhead_device.json if present, else --index.

What it computes, and why each step exists
------------------------------------------
The goal is the angle to rotate the gripper to. Three candidate answers get
computed, because they disagree in ways that matter:

  PCA axis        The direction of greatest spread of the object's pixels.
                  Stable and smooth, but for a hook or an L it points along
                  the whole shape's diagonal, which is not where the jaws fit.

  min-area box    The orientation of the tightest rotated rectangle. Good for
                  rectangular parts, but for a non-convex shape the box is
                  dominated by the bounding extremes, not the graspable part.

  GRASP AXIS      Where two parallel jaws actually close. Found by scanning
                  candidate angles and measuring the object's width across
                  each one, then taking the narrowest that the jaws can span.
                  This is the one a gripper should use.

The distinction is the point of this file. A cube hides it — every method
agrees. A hook exposes it, which is why a "weird looking shape" is the right
test object.

Grasp angle is reported modulo 180 degrees: a two-jaw gripper closing at 10
and at 190 degrees is the same grasp.

    python shape_pose.py                 # live view, all three axes drawn
    python shape_pose.py --snapshot      # one frame -> shape_pose_out.jpg
    python shape_pose.py --index 1       # force a camera
    python shape_pose.py --min-area 800  # ignore blobs smaller than this

In the live view, CLICK the object to lock onto its colour — the most reliable
mode on a cluttered desk. Press c to go back to automatic.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DEVICE_FILE = Path(__file__).with_name("overhead_device.json")
OUT_PATH = Path(__file__).with_name("shape_pose_out.jpg")

WORK_W, WORK_H = 960, 540

# Minimum blob area in the working frame. Below this it is noise or a speck.
MIN_AREA = 600

# Jaw opening, in pixels of the working frame. The grasp search rejects any
# angle whose width exceeds this, because the jaws physically cannot span it.
# In pixels rather than mm since this file is deliberately robot-free; convert
# once a homography exists.
MAX_JAW_PX = 140

# Angles to test when searching for the grasp axis. 3 degrees is finer than
# the gripper can meaningfully be commanded and keeps the scan cheap.
ANGLE_STEP_DEG = 3

# A shape whose two principal spreads are within this ratio has no meaningful
# long axis -- a square or a disc. Reporting a confident angle for one of those
# would be false precision, so it is flagged instead.
ISOTROPY_RATIO = 1.15


def camera_index(argv):
    if "--index" in argv:
        return int(argv[argv.index("--index") + 1])
    if DEVICE_FILE.exists():
        try:
            return json.loads(DEVICE_FILE.read_text())["index"]
        except Exception:                             # noqa: BLE001
            pass
    return 0


# Segmentation strategy. "auto" suits a coloured object on a plain surface;
# "click" is for anything else — you point at the object and it learns the
# colour. Measured on the live scene, the red hook and the wooden desk share
# almost the same HUE (10.2 vs 10.4), so hue alone cannot separate them; they
# differ in brightness (V 213 vs 38) and in Lab a* against the notepad
# (157.5 vs 139.9). Hence Lab, not HSV.
LAB_TOLERANCE = 18        # how far from the sampled colour still counts
MIN_CHROMA = 12           # ignore near-neutral pixels in auto mode


def segment(frame, reference=None, tolerance=LAB_TOLERANCE):
    """Binary mask of the object(s) of interest.

    Two modes:

      reference given  — pixels close to that Lab colour. This is the reliable
                         one for a cluttered desk, and is what --click sets.

      reference None   — the most chromatic connected region. Works when one
                         coloured object sits on a neutral surface.

    Lab rather than HSV because the failure that motivated this is a hue
    collision: the red hook and the wooden desk measured 10.2 and 10.4 in hue,
    so an HSV threshold segmented the entire desk. Lab separates them on a*
    and on lightness, which is what actually differs.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)

    if reference is not None:
        ref = np.array(reference, dtype=np.int16)
        # Chroma distance only (a*, b*), ignoring L, so the object stays one
        # region across its own shading and highlights.
        dist = np.sqrt(((lab[:, :, 1:] - ref[1:]) ** 2).sum(axis=2))
        mask = (dist < tolerance).astype(np.uint8) * 255
    else:
        a = lab[:, :, 1] - 128
        b = lab[:, :, 2] - 128
        chroma = np.sqrt(a * a + b * b).astype(np.uint8)
        _, mask = cv2.threshold(chroma, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask[chroma < MIN_CHROMA] = 0
        # Very dark pixels have unreliable chroma.
        mask[lab[:, :, 0] < 40] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def sample_colour(frame, x, y, half=4):
    """Mean Lab colour of a small patch, for use as a segmentation reference."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    patch = lab[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1]
    return patch.reshape(-1, 3).mean(axis=0)


def width_at_angle(points, angle_deg):
    """Object width perpendicular to `angle_deg`, and the span along it.

    This is what a pair of jaws closing along `angle_deg` would have to span.
    Rotating the points and taking the extent is exact for this, and far
    simpler than reasoning about the contour's geometry directly.
    """
    t = np.deg2rad(angle_deg)
    # Column 0 runs along the jaw-closing direction, column 1 across it.
    rot = np.array([[np.cos(t), -np.sin(t)],
                    [np.sin(t), np.cos(t)]])
    local = points @ rot
    width = local[:, 0].max() - local[:, 0].min()
    length = local[:, 1].max() - local[:, 1].min()
    return float(width), float(length)


def grasp_axis(points, max_jaw_px=MAX_JAW_PX):
    """Best angle for two parallel jaws to close at.

    Scans candidate angles and keeps the one with the smallest width, which is
    where the jaws have the most clearance and the most symmetric purchase.
    Angles wider than the jaw opening are rejected outright.

    Returns (angle_deg, width_px, feasible). `feasible` is False when no angle
    fits the jaws — a real outcome worth surfacing rather than silently
    returning the least-bad number.
    """
    best = None
    for a in range(0, 180, ANGLE_STEP_DEG):
        w, _ = width_at_angle(points, a)
        if best is None or w < best[1]:
            best = (a, w)
    angle, width = best
    return float(angle), float(width), width <= max_jaw_px


def pca_axis(points):
    """Principal axis angle and the two spreads along the eigenvectors."""
    centred = points - points.mean(axis=0)
    cov = np.cov(centred.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    major = vecs[:, 0]
    angle = float(np.rad2deg(np.arctan2(major[1], major[0])) % 180)
    spreads = np.sqrt(np.maximum(vals, 0))
    return angle, float(spreads[0]), float(spreads[1])


def analyse(mask, min_area=MIN_AREA, max_jaw_px=MAX_JAW_PX):
    """Describe every object in the mask: shape, pose and grasp angle."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        points = contour.reshape(-1, 2).astype(np.float64)
        centred = points - np.array([cx, cy])

        pca_deg, spread_major, spread_minor = pca_axis(points)
        (bw, bh), box_deg = cv2.minAreaRect(contour)[1:]
        # OpenCV's angle convention flips with the side ordering; normalise so
        # the angle always describes the LONGER side.
        if bw < bh:
            box_deg += 90
        box_deg = float(box_deg % 180)

        g_deg, g_width, feasible = grasp_axis(centred, max_jaw_px)
        _, g_length = width_at_angle(centred, g_deg)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        # How much of its own convex hull the shape fills. A hook or an L sits
        # well below 1; a rectangle or disc sits near it. This is what says
        # whether the PCA axis can be trusted as a grasp direction.
        solidity = float(area / hull_area) if hull_area > 0 else 0.0

        perimeter = cv2.arcLength(contour, True)
        circularity = (float(4 * np.pi * area / (perimeter ** 2))
                       if perimeter > 0 else 0.0)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        isotropic = (spread_minor > 0
                     and spread_major / spread_minor < ISOTROPY_RATIO)

        out.append({
            "centroid": (float(cx), float(cy)),
            "area": float(area),
            "contour": contour,
            "box": cv2.boxPoints(cv2.minAreaRect(contour)),
            "pca_deg": pca_deg,
            "box_deg": box_deg,
            "grasp_deg": g_deg,
            "grasp_width": g_width,
            "grasp_length": g_length,
            "grasp_feasible": feasible,
            "solidity": solidity,
            "circularity": circularity,
            "vertices": len(approx),
            "isotropic": isotropic,
            "name": classify(solidity, circularity, len(approx), isotropic),
        })
    out.sort(key=lambda o: -o["area"])
    return out


def classify(solidity, circularity, vertices, isotropic):
    """A coarse shape label.

    Deliberately coarse. The grasp angle is what the gripper needs; the label
    is for the human watching, and a confident-sounding name for an arbitrary
    blob would be worse than an honest vague one.
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


def draw(frame, objects):
    """Annotate the frame with each object's axes and readings."""
    for i, obj in enumerate(objects):
        cx, cy = obj["centroid"]
        cv2.drawContours(frame, [obj["contour"]], -1, (0, 255, 255), 2)
        cv2.drawContours(frame, [obj["box"].astype(int)], -1, (120, 120, 120), 1)

        def ray(angle_deg, length, colour, thickness):
            t = np.deg2rad(angle_deg)
            dx, dy = np.cos(t) * length / 2, np.sin(t) * length / 2
            cv2.line(frame, (int(cx - dx), int(cy - dy)),
                     (int(cx + dx), int(cy + dy)), colour, thickness)

        # PCA in grey, grasp axis in green, and the jaw-closing direction —
        # perpendicular to the grasp axis — in red, since that is the one the
        # gripper actually moves along.
        ray(obj["pca_deg"], obj["grasp_length"], (160, 160, 160), 2)
        ray(obj["grasp_deg"], obj["grasp_length"] * 0.9, (0, 220, 0), 2)
        jaw = (obj["grasp_deg"] + 90) % 180
        colour = (0, 0, 255) if obj["grasp_feasible"] else (0, 0, 120)
        ray(jaw, obj["grasp_width"], colour, 3)

        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 255, 255), -1)

        lines = [
            f"#{i} {obj['name']}",
            f"grasp {obj['grasp_deg']:.0f}deg w={obj['grasp_width']:.0f}px"
            + ("" if obj["grasp_feasible"] else " TOO WIDE"),
            f"pca {obj['pca_deg']:.0f}deg  solidity {obj['solidity']:.2f}",
        ]
        if obj["isotropic"]:
            lines.append("no dominant axis")
        y = int(cy) - 10 - 16 * len(lines)
        for line in lines:
            y += 16
            cv2.putText(frame, line, (int(cx) + 14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(frame, line, (int(cx) + 14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return frame


def report(objects):
    if not objects:
        print("  no objects found")
        return
    for i, obj in enumerate(objects):
        cx, cy = obj["centroid"]
        print(f"  #{i} {obj['name']:22s} at px({cx:.0f},{cy:.0f}) "
              f"area={obj['area']:.0f}")
        print(f"      grasp angle {obj['grasp_deg']:5.1f} deg   "
              f"width {obj['grasp_width']:5.1f} px   "
              f"{'fits jaws' if obj['grasp_feasible'] else 'TOO WIDE FOR JAWS'}")
        print(f"      pca {obj['pca_deg']:5.1f} deg   "
              f"min-area-box {obj['box_deg']:5.1f} deg   "
              f"(disagreement {abs(obj['pca_deg'] - obj['grasp_deg']):.0f} deg)")
        print(f"      solidity {obj['solidity']:.2f}  "
              f"circularity {obj['circularity']:.2f}  "
              f"vertices {obj['vertices']}")
        if obj["isotropic"]:
            print("      NOTE: no dominant axis — any grasp angle is "
                  "equally valid")


def open_camera(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"camera index {index} would not open")
        return None
    return cap


def read_frame(cap):
    for _ in range(4):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok:
        return None
    return cv2.resize(frame, (WORK_W, WORK_H))


def snapshot(index, min_area, max_jaw, reference=None):
    cap = open_camera(index)
    if cap is None:
        return
    frame = read_frame(cap)
    cap.release()
    if frame is None:
        print("no frame")
        return
    mask = segment(frame, reference)
    objects = analyse(mask, min_area, max_jaw)
    print(f"camera index {index}: {len(objects)} object(s)")
    report(objects)
    annotated = draw(frame.copy(), objects)
    side = np.hstack([annotated,
                      cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(str(OUT_PATH), side)
    print(f"\nwrote {OUT_PATH.name} (annotated | mask)")


def live(index, min_area, max_jaw, reference=None):
    cap = open_camera(index)
    if cap is None:
        return
    print(f"camera index {index} — q quit, m mask, s save, "
          f"click an object to lock onto its colour, c to clear")
    show_mask = False
    state = {"ref": reference, "frame": None}

    def on_click(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN or state["frame"] is None:
            return
        # The window may show the annotated frame beside the mask; a click on
        # the right-hand half still refers to the same pixel column.
        x = x % WORK_W
        state["ref"] = sample_colour(state["frame"], x, y)
        print(f"  locked onto Lab {np.round(state['ref'], 1)} at ({x},{y})")

    cv2.namedWindow("shape + orientation")
    cv2.setMouseCallback("shape + orientation", on_click)
    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                break
            state["frame"] = frame
            mask = segment(frame, state["ref"])
            objects = analyse(mask, min_area, max_jaw)
            annotated = draw(frame.copy(), objects)
            cv2.putText(annotated,
                        f"{len(objects)} object(s)   "
                        f"green=grasp axis  red=jaw travel  grey=PCA"
                        + ("   [colour locked]" if state["ref"] is not None
                           else "   [auto]"),
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            view = (np.hstack([annotated,
                               cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
                    if show_mask else annotated)
            cv2.imshow("shape + orientation", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                show_mask = not show_mask
            if key == ord("c"):
                state["ref"] = None
                print("  cleared colour lock (auto mode)")
            if key == ord("s"):
                cv2.imwrite(str(OUT_PATH), view)
                print(f"saved {OUT_PATH.name}")
                report(objects)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    idx = camera_index(sys.argv)
    area = (int(sys.argv[sys.argv.index("--min-area") + 1])
            if "--min-area" in sys.argv else MIN_AREA)
    jaw = (int(sys.argv[sys.argv.index("--jaw") + 1])
           if "--jaw" in sys.argv else MAX_JAW_PX)
    ref = None
    if "--lab" in sys.argv:
        ref = np.array([float(v) for v in
                        sys.argv[sys.argv.index("--lab") + 1].split(",")])
    if "--snapshot" in sys.argv:
        snapshot(idx, area, jaw, ref)
    else:
        live(idx, area, jaw, ref)
