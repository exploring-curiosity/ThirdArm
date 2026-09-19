"""Render a build plan as an interactive 3D structure.

Shows what the planner intends to build, as geometry rather than a list of
steps. Orbit it, step through the build order, and see each block land.

Heights
-------
The overhead survey measures footprint, width and depth. It does NOT measure
height — from directly above, a block lying flat and the same block standing
differ in footprint, not in anything the camera can see. So block height is
ESTIMATED by the VLM from the image (it can judge proportions and shadows) and
is labelled as an estimate everywhere it appears.

This matters for the demo: the rendered tower's proportions are indicative,
but the ORDER, the supports and the stability checks are real. Tomorrow the
wrist camera measures each layer's true top on the return path, and those
estimates get replaced by measurements.

    python plan_view3d.py                    # render build_plan.json
    python plan_view3d.py --plan other.json
    python plan_view3d.py --save out.jpg     # write a still, no window

Controls: drag orbit, scroll zoom, left/right arrows step the build,
          a show all, q quit.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PLAN_PATH = Path(__file__).with_name("build_plan.json")
SIZE = 720

# One relative unit (the largest block's width) maps to this many pixels in
# the render. Display only.
UNIT_PX = 150.0

# Fallback height if the plan carries no estimate, as a fraction of the
# block's own width. Used only so a missing estimate still renders something.
DEFAULT_HEIGHT = 0.45

PALETTE = {
    "red": (60, 60, 220), "orange": (50, 140, 240), "yellow": (70, 210, 230),
    "green": (90, 200, 90), "blue": (220, 140, 60), "cyan": (210, 210, 80),
    "purple": (200, 90, 190), "white": (230, 230, 230),
    "grey": (150, 150, 150), "black": (70, 70, 70),
}


def block_colour(block):
    return PALETTE.get(block.get("colour", ""), (180, 180, 180))


def rotation(yaw_deg, pitch_deg):
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(p), -np.sin(p)],
                   [0, np.sin(p), np.cos(p)]])
    return rx @ rz


def prism_faces(outline, cx, cy, cz, h, scale):
    """Vertices and faces of the block's real silhouette extruded to height h.

    The previous version drew an axis-aligned box for everything, so a
    right-angled wedge rendered as a cube. This extrudes SAM's measured
    outline instead, so the footprint shape is real. The vertical profile is
    still a straight extrusion — an overhead camera cannot see a sloped face,
    so a wedge appears as a prism with the wedge's footprint until the wrist
    camera supplies a side view.
    """
    if not outline or len(outline) < 3:
        # Fall back to a square only when there is genuinely no outline.
        outline = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
        scale = 1.0

    n = len(outline)
    bottom = [[cx + px * scale, cy + py * scale, cz] for px, py in outline]
    top = [[cx + px * scale, cy + py * scale, cz + h] for px, py in outline]
    verts = np.array(bottom + top, np.float64)

    faces = []
    # Side walls, one quad per outline edge.
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))
    # Top and bottom as fans from the first vertex — fine for the convex-ish
    # footprints here, and each triangle is filled separately anyway.
    for i in range(1, n - 1):
        faces.append((n, n + i, n + i + 1, n))
        faces.append((0, i + 1, i, 0))
    return verts, faces


def layout(plan, blocks, px_per_unit=1.0):
    """Resolve each placement into a 3D box.

    Walks the build order, stacking each block on its support and applying the
    planner's offset. Returns boxes in build order so the view can step
    through them.
    """
    by_id = {b["id"]: b for b in blocks}
    boxes = []
    placed = {}
    # Outlines are in pixels about each block's centroid; convert to the same
    # relative units the offsets use.
    out_scale = 1.0 / max(px_per_unit, 1e-6)

    for step, p in enumerate(plan.get("placements", [])):
        bid = p.get("block_id")
        block = by_id.get(bid)
        if block is None:
            continue

        w = float(block.get("width", 0.5))
        d = float(block.get("depth", 0.4))
        # Prefer a MEASURED height; fall back to the VLM estimate, and mark
        # which one was used so the panel can say so.
        measured = block.get("height_measured")
        if measured:
            h, h_source = float(measured), "measured"
        elif block.get("height_est"):
            h, h_source = float(block["height_est"]), "estimated"
        else:
            h, h_source = w * DEFAULT_HEIGHT, "assumed"

        support_id = p.get("on_top_of")
        offset = p.get("offset", [0, 0])
        try:
            ox, oy = float(offset[0]), float(offset[1])
        except (TypeError, IndexError, ValueError):
            ox = oy = 0.0

        if support_id is None or support_id not in placed:
            # Layer 0 sits on the table. Spread the bases apart so a plan with
            # several ground blocks does not draw them on top of each other.
            ground = [b for b in boxes if b["support"] is None]
            cx, cy, cz = len(ground) * 1.3, 0.0, 0.0
        else:
            sup = placed[support_id]
            cx = sup["cx"] + ox * sup["w"]
            cy = sup["cy"] + oy * sup["w"]
            cz = sup["cz"] + sup["h"]

        box = {"step": step, "id": bid, "name": block.get("name", str(bid)),
               "colour": block_colour(block), "cx": cx, "cy": cy, "cz": cz,
               "w": w, "d": d, "h": h, "support": support_id,
               "outline": block.get("outline") or [],
               "out_scale": out_scale,
               "h_source": h_source,
               "reason": p.get("reason", "")}
        boxes.append(box)
        placed[bid] = box
    return boxes


def render(boxes, yaw, pitch, zoom, upto=None, size=SIZE):
    canvas = np.full((size, size, 3), 22, np.uint8)
    R = rotation(yaw, pitch)
    scale = UNIT_PX * zoom

    if boxes:
        mid_x = np.mean([b["cx"] for b in boxes])
        mid_y = np.mean([b["cy"] for b in boxes])
    else:
        mid_x = mid_y = 0.0

    def project(p):
        q = R @ np.array([p[0] - mid_x, p[1] - mid_y, p[2]])
        return (int(size / 2 + q[0] * scale),
                int(size * 0.62 - q[2] * scale - q[1] * scale * 0.35),
                q[1])

    # Ground grid for a sense of the table plane.
    for i in range(-4, 5):
        t = i * 0.5
        for a, b in (((-2.0, t, 0), (2.0, t, 0)), ((t, -2.0, 0), (t, 2.0, 0))):
            pa, pb = project(a), project(b)
            cv2.line(canvas, pa[:2], pb[:2], (46, 46, 46), 1)

    shown = boxes if upto is None else boxes[:upto + 1]

    # Collect every face, depth sort, then fill. Painter's algorithm keeps
    # near blocks in front of far ones without a real renderer.
    polys = []
    for box in shown:
        verts, faces = prism_faces(box["outline"], box["cx"], box["cy"],
                                   box["cz"], box["h"], box["out_scale"])
        projected = [project(v) for v in verts]
        n_side = len(box["outline"]) if box["outline"] else 4
        for fi, face in enumerate(faces):
            pts = np.array([projected[i][:2] for i in face], np.int32)
            depth = np.mean([projected[i][2] for i in face])
            # Side walls shaded by facing so the prism reads as solid; the
            # top face is brightest.
            if fi < n_side:
                shade = 0.6 + 0.3 * (fi / max(n_side, 1))
            else:
                shade = 1.0
            colour = tuple(int(min(c * shade, 255)) for c in box["colour"])
            polys.append((depth, pts, colour, box))

    polys.sort(key=lambda p: -p[0])
    for _, pts, colour, box in polys:
        cv2.fillConvexPoly(canvas, pts, colour)
        cv2.polylines(canvas, [pts], True, (20, 20, 20), 1)

    # Label each block at its top centre.
    for box in shown:
        top = project((box["cx"], box["cy"], box["cz"] + box["h"]))
        label = box["name"]
        cv2.putText(canvas, label, (top[0] + 8, top[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(canvas, label, (top[0] + 8, top[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1)

    return canvas


def side_panel(data, boxes, upto, width=460, size=SIZE):
    panel = np.full((size, width, 3), 18, np.uint8)
    plan = data.get("plan") or {}
    lines = [("GOAL", (120, 200, 255))]
    lines += [(f"  {c}", (200, 200, 200))
              for c in _wrap(data.get("goal", ""), 52)]
    lines.append(("", None))
    lines.append(("CONCEPT", (120, 200, 255)))
    lines += [(f"  {c}", (215, 215, 215))
              for c in _wrap(plan.get("concept", ""), 52)]
    lines.append(("", None))
    lines.append(("BUILD ORDER", (120, 200, 255)))

    for i, box in enumerate(boxes):
        active = upto is None or i <= upto
        colour = (235, 235, 235) if active else (95, 95, 95)
        marker = ">" if (upto is not None and i == upto) else " "
        where = ("table" if box["support"] is None
                 else next((b["name"] for b in boxes
                            if b["id"] == box["support"]), "?"))
        lines.append((f" {marker}{i + 1}. {box['name']} -> {where}", colour))

    lines.append(("", None))
    if data.get("valid"):
        lines.append(("VALIDATION: PASSED", (120, 230, 120)))
    else:
        lines.append(("VALIDATION: REJECTED", (90, 90, 255)))
        for p in data.get("problems", [])[:5]:
            lines += [(f"  {c}", (150, 150, 240)) for c in _wrap(p, 50)]

    lines.append(("", None))
    sources = {b["h_source"] for b in boxes}
    if sources == {"measured"}:
        lines.append(("heights MEASURED (wrist camera)", (120, 230, 120)))
    else:
        lines.append(("heights are VLM ESTIMATES", (110, 150, 200)))
        lines.append(("overhead view cannot see height;", (110, 150, 200)))
        lines.append(("wrist side-view replaces these", (110, 150, 200)))
    # Only claim a measured footprint when outlines are actually present.
    # A plan saved before build_planner started emitting outlines renders as
    # unit squares, and captioning that "MEASURED" states the opposite of the
    # truth.
    if all(b.get("outline") for b in boxes):
        lines.append(("footprint shape is MEASURED", (120, 200, 255)))
    else:
        lines.append(("footprint shape is a PLACEHOLDER", (110, 150, 200)))
        lines.append(("square; re-run build_planner.py", (110, 150, 200)))

    y = 30
    for text, colour in lines:
        if text:
            cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, colour, 1)
        y += 19
    return panel


def _wrap(text, width):
    words, line, out = str(text).split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out or [""]


def main(argv):
    path = Path(argv[argv.index("--plan") + 1]) if "--plan" in argv \
        else PLAN_PATH
    if not path.exists():
        print(f"{path.name} not found — run build_planner.py first")
        return
    data = json.loads(path.read_text())
    plan = data.get("plan")
    blocks = data.get("blocks", [])
    if not plan:
        print("no plan in file")
        return

    boxes = layout(plan, blocks, data.get("px_per_unit", 1.0))
    if not boxes:
        print("plan has no placeable blocks")
        return
    print(f"{len(boxes)} blocks in the plan")

    # A plan written before build_planner emitted outlines/px_per_unit still
    # renders, but every block falls back to a same-sized unit square, so the
    # blocks interpenetrate and the shapes are meaningless. Say so rather than
    # letting a plausible-looking picture be believed.
    if not all(b.get("outline") for b in boxes):
        print("  WARNING: this plan has no measured outlines — blocks are "
              "drawn as equal-sized squares.")
        print("  Re-run build_planner.py to regenerate build_plan.json.")

    view = {"yaw": 38.0, "pitch": 24.0, "zoom": 1.0,
            "upto": len(boxes) - 1, "drag": None}

    if "--save" in argv:
        out = argv[argv.index("--save") + 1]
        frame = np.hstack([render(boxes, view["yaw"], view["pitch"],
                                  view["zoom"]),
                           side_panel(data, boxes, None)])
        cv2.imwrite(out, frame)
        print(f"wrote {out}")
        return

    def on_mouse(event, x, y, flags, _p):
        if event == cv2.EVENT_LBUTTONDOWN:
            view["drag"] = (x, y, view["yaw"], view["pitch"])
        elif event == cv2.EVENT_MOUSEMOVE and view["drag"]:
            x0, y0, yaw0, pitch0 = view["drag"]
            view["yaw"] = (yaw0 + (x - x0) * 0.4) % 360
            view["pitch"] = float(np.clip(pitch0 + (y - y0) * 0.3, -5, 80))
        elif event == cv2.EVENT_LBUTTONUP:
            view["drag"] = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            view["zoom"] = float(np.clip(
                view["zoom"] * (1.1 if flags > 0 else 0.9), 0.3, 4.0))

    cv2.namedWindow("build plan")
    cv2.setMouseCallback("build plan", on_mouse)
    print("drag orbit, scroll zoom, left/right step the build, a all, q quit")

    while True:
        frame = np.hstack([
            render(boxes, view["yaw"], view["pitch"], view["zoom"],
                   view["upto"]),
            side_panel(data, boxes, view["upto"]),
        ])
        cv2.imshow("build plan", frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("a"):
            view["upto"] = len(boxes) - 1
        if key == 83:                       # right arrow
            view["upto"] = min(view["upto"] + 1, len(boxes) - 1)
        if key == 81:                       # left arrow
            view["upto"] = max(view["upto"] - 1, 0)
        if key == ord("s"):
            cv2.imwrite("build_plan_3d.jpg", frame)
            print("saved build_plan_3d.jpg")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
