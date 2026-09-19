"""Plan a stable 3D structure from blocks on the table, and validate it.

Standalone: no robot, no Viam. Reads the overhead camera, segments with SAM 3,
asks a VLM what to build, then checks the proposal against geometry before any
of it could reach an arm.

The division of labour is the whole point:

    VLM         decides WHAT to build and in what order. Good at intent
                ("stable", "interesting"), bad at precise geometry.

    validator   decides whether each placement is PHYSICALLY SOUND. Centre of
                mass over support, no excessive overhang, block available,
                support already placed. Deterministic, no model involved.

Neither is trusted alone. A VLM asked to do physics will confidently stack a
wide slab on a narrow block; a geometric planner asked to be interesting will
build the same tower every time. Rejected placements go back to the VLM with
the reason, so it revises rather than the code silently "fixing" its plan.

Dimensions are RELATIVE throughout — pixel widths and areas from the overhead
view, normalised to the largest block. Nothing here is metric, and nothing
needs to be: stability is about ratios (is the top narrower than the bottom,
is the centre over the support), not millimetres. Metric enters only when the
arm executes, where the existing calibration already handles it.

    python build_planner.py                 # capture, plan, validate, render
    python build_planner.py --image X.jpg   # plan from a saved frame
    python build_planner.py --goal "..."    # a different build instruction
    python build_planner.py --no-vlm        # geometry only, skip the VLM
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import auto_pose

OUT_PATH = Path(__file__).with_name("build_plan_out.jpg")
PLAN_PATH = Path(__file__).with_name("build_plan.json")

DEFAULT_GOAL = ("Build something stable and artistically interesting from "
                "these blocks.")

VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# --- stability rules ---------------------------------------------------
#
# All ratios, so they hold whatever the real block sizes are.

# A block's centre must sit within this fraction of the supporting block's
# half-width, measured from the support's centre. 0.5 means the centre of mass
# stays within the middle half of the support -- comfortably inside the
# tipping point, which is 1.0.
MAX_CENTRE_OFFSET = 0.5

# A block may not be wider than its support by more than this factor. Stacking
# a wide slab on a narrow block is the classic VLM failure.
MAX_WIDTH_RATIO = 1.15

# Concave pieces (an arch, a hook) cannot reliably support a block on top:
# the contact patch is whatever the shape leaves, not a flat face.
MIN_SUPPORT_SOLIDITY = 0.80

# Two blocks sharing a support must not occupy the same space. Their centres
# must be at least this fraction of their combined half-widths apart.
# Caught by rendering the plan in 3D: a VLM happily placed two blocks on one
# support at offsets 0.0 and 0.3, which overlap for blocks 0.54 wide.
MIN_SEPARATION = 0.9


def capture(index=None):
    """One frame from the overhead camera."""
    import json as _json
    if index is None:
        idx_file = Path(__file__).with_name("overhead_device.json")
        index = (_json.loads(idx_file.read_text())["index"]
                 if idx_file.exists() else 0)
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    for _ in range(8):
        cap.grab()
    ok, frame = cap.retrieve()
    cap.release()
    if not ok:
        return None
    return cv2.resize(frame, (auto_pose.WORK_W, auto_pose.WORK_H))


def colour_name(bgr):
    """A human-readable colour, so the VLM and the operator can agree on
    which block is which. Hue-based, and only ever used for NAMING -- never
    for detection, which is SAM's job."""
    b, g, r = [int(v) for v in bgr]
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 60:
        return "black"
    if s < 40:
        return "white" if v > 180 else "grey"
    if h < 8 or h >= 170:
        return "red"
    if h < 20:
        return "orange"
    if h < 33:
        return "yellow"
    if h < 85:
        return "green"
    if h < 100:
        return "cyan"
    if h < 130:
        return "blue"
    return "purple"


def outline_of(mask, max_points=40):
    """The mask's outline as a simplified polygon, centred on its centroid.

    Stored with the block so the 3D view can extrude the REAL silhouette
    instead of an axis-aligned box. Without it a right-angled wedge renders
    as a cube, which is what it did.

    Simplified to keep the plan file small; 40 points is far more than enough
    for these shapes and keeps concave outlines (an L, a hook) recognisable.
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    while len(approx) > max_points:
        eps *= 1.3
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    return [[round(float(px - cx), 1), round(float(py - cy), 1)]
            for px, py in approx]


def survey(frame, grid=11):
    """Every block on the table, with relative dimensions.

    Returns a list of dicts the VLM can reason about and the validator can
    check. `width`/`depth`/`area` are normalised to the largest block, so the
    plan is scale-free.
    """
    model, processor = auto_pose.load_model()
    roi = auto_pose.load_roi()
    found = auto_pose.segment_all(model, processor, frame, grid,
                                  auto_pose.MAX_COVER, roi)

    blocks = []
    for mask, score in found:
        obj = auto_pose.describe(mask, score)
        if obj is None:
            continue
        ys, xs = np.nonzero(mask)
        bgr = frame[mask].mean(axis=0)
        blocks.append({
            "px": (int(obj["centroid"][0]), int(obj["centroid"][1])),
            "w_px": int(xs.max() - xs.min()),
            "d_px": int(ys.max() - ys.min()),
            "area_px": int(obj["area"]),
            "solidity": round(float(obj["solidity"]), 2),
            "yaw_deg": int(obj["grasp_deg"]),
            "jaw_px": int(obj["grasp_width"]),
            "shape": obj["name"],
            "colour": colour_name(bgr),
            "mask": mask,
            "outline": outline_of(mask),
        })

    blocks.sort(key=lambda b: -b["area_px"])
    if not blocks:
        return []

    # Relative dimensions: 1.0 is the largest block's footprint.
    biggest = max(max(b["w_px"], b["d_px"]) for b in blocks)
    big_area = max(b["area_px"] for b in blocks)
    for i, b in enumerate(blocks):
        b["id"] = i
        b["width"] = round(max(b["w_px"], b["d_px"]) / biggest, 2)
        b["depth"] = round(min(b["w_px"], b["d_px"]) / biggest, 2)
        b["footprint"] = round(b["area_px"] / big_area, 2)
        # Name blocks so a human can follow the plan out loud.
        b["name"] = f"{b['colour']}_{b['shape'].split()[0]}"
    return blocks


def px_per_unit(blocks):
    """Pixels per relative unit, so the renderer can scale outlines.

    One relative unit is the largest block's longest side, which is how
    `width` and `depth` are normalised.
    """
    if not blocks:
        return 1.0
    return float(max(max(b["w_px"], b["d_px"]) for b in blocks))


def describe_for_vlm(blocks):
    """The block list as text, for the prompt."""
    lines = []
    for b in blocks:
        flat = "flat top" if b["solidity"] >= MIN_SUPPORT_SOLIDITY else \
               "irregular/concave top - cannot support much"
        lines.append(
            f"  id={b['id']} {b['name']}: relative width {b['width']}, "
            f"depth {b['depth']}, footprint {b['footprint']}, "
            f"shape {b['shape']}, {flat}")
    return "\n".join(lines)


PROMPT = """You are planning a structure a robot arm will build from real blocks on a table.

Blocks available (dimensions are RELATIVE, 1.0 = largest block):
{blocks}

Goal: {goal}

Plan a structure and return ONLY a JSON object, no other text:

{{
  "concept": "one sentence describing what you are building and why it is interesting",
  "heights": {{"0": 0.4, "1": 0.3}},
  "placements": [
    {{"block_id": 0, "layer": 0, "on_top_of": null, "offset": [0.0, 0.0],
      "reason": "why this block here"}},
    {{"block_id": 2, "layer": 1, "on_top_of": 0, "offset": [0.0, 0.0],
      "reason": "..."}}
  ]
}}

"heights" is your ESTIMATE of each block's height, keyed by block id, as a
fraction of the largest block's width. The overhead camera looks straight down
and cannot measure height, so judge it from the image — shadows, visible side
faces, and how the block sits. A block lying flat is shorter than the same
block standing on end.

Rules you must follow:
- layer 0 sits on the table; on_top_of must be null for layer 0.
- every other block sits on exactly one block already placed in a lower layer.
- offset is [x, y] relative to the supporting block's centre, in units of the
  SUPPORT's width. [0,0] is centred. Keep |offset| small for stability.
- a block must not be much wider than the block it sits on.
- blocks with an irregular or concave top cannot support another block.
- use as many blocks as you can while keeping it stable; it is fine to leave
  a block out if using it would be unstable, but say so in the concept.
- build order is the order of the placements list: bottom layers first.
"""


def ask_vlm(frame, blocks, goal, model_id=VLM_MODEL, retry_note=None):
    """Ask the VLM for a structure. Returns the parsed plan, or None."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    prompt = PROMPT.format(blocks=describe_for_vlm(blocks), goal=goal)
    if retry_note:
        # Give the numbers, not just the complaint. A bare rejection made the
        # VLM nudge an offset from 0.28 to 0.38 when 0.49 was required; it was
        # guessing at the target rather than solving for it.
        prompt += (
            "\n\nYour previous plan was REJECTED by a physics check:\n"
            f"{retry_note}\n\n"
            "Fix it by satisfying the numbers quoted above exactly. Where an "
            "overlap is reported, either move the block far enough apart to "
            "meet the required separation (round UP, do not just nudge it), "
            "or stack it on a different block, or leave it out. Do not "
            "repeat the same offsets.")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="mps")

    import PIL.Image
    pil = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)

    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=900, do_sample=False)
    text = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    print(f"  VLM responded in {time.time() - t0:.0f}s")
    return parse_plan(text), text


def apply_heights(plan, blocks):
    """Copy the VLM's height estimates onto the block records.

    Kept separate from the measured fields and named `height_est` so nothing
    downstream mistakes an estimate for a measurement. Tomorrow the wrist
    camera replaces these with real numbers.
    """
    heights = (plan or {}).get("heights") or {}
    for b in blocks:
        raw = heights.get(str(b["id"]), heights.get(b["id"]))
        try:
            b["height_est"] = round(float(raw), 2)
        except (TypeError, ValueError):
            b["height_est"] = None
    return blocks


def parse_plan(text):
    """Pull the JSON object out of a VLM response."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# --- validation --------------------------------------------------------


def resolve_offsets(plan, blocks):
    """Place co-supported blocks apart automatically.

    The VLM chooses WHICH block goes on WHICH support — the structural
    decision. It is measurably bad at the arithmetic of "how far apart",
    though: asked to fix a 0.28 separation needing 0.49 it produced 0.38, then
    0.47 needing 0.69. It nudges rather than solves, even when told the exact
    target.

    So the code solves that part. Blocks sharing a support are spread along a
    line through the support's centre at exactly the required separation,
    keeping the group centred so the combined centre of mass stays over the
    support. The VLM's intent survives; only the numbers are corrected.
    """
    by_id = {b["id"]: b for b in blocks}
    groups = {}
    for p in plan.get("placements", []):
        sup = p.get("on_top_of")
        if sup is None:
            continue
        groups.setdefault(sup, []).append(p)

    for sup_id, members in groups.items():
        if len(members) < 2 or sup_id not in by_id:
            continue
        support = by_id[sup_id]
        widths = [by_id[m["block_id"]]["width"] for m in members
                  if m.get("block_id") in by_id]
        if len(widths) != len(members):
            continue

        # Required gap between each adjacent pair, in support-width units.
        gaps = [((widths[i] + widths[i + 1]) / 2 * MIN_SEPARATION
                 / max(support["width"], 1e-6))
                for i in range(len(members) - 1)]
        span = sum(gaps)
        pos, x = [], -span / 2
        for i in range(len(members)):
            pos.append(x)
            if i < len(gaps):
                x += gaps[i]

        for m, px in zip(members, pos):
            m["offset"] = [round(px, 2), 0.0]
            m["offset_auto"] = True
    return plan


def validate(plan, blocks):
    """Check every placement against geometry. Returns (ok, problems).

    Deterministic and model-free. This is what stops a plausible-sounding
    plan from toppling on the table.
    """
    problems = []
    if not plan or "placements" not in plan:
        return False, ["plan has no placements"]

    by_id = {b["id"]: b for b in blocks}
    placed = {}                      # block_id -> placement

    for n, p in enumerate(plan["placements"]):
        bid = p.get("block_id")
        tag = f"step {n + 1}"

        if bid not in by_id:
            problems.append(f"{tag}: block_id {bid} does not exist")
            continue
        if bid in placed:
            problems.append(f"{tag}: block {by_id[bid]['name']} placed twice")
            continue

        block = by_id[bid]
        support_id = p.get("on_top_of")
        layer = p.get("layer", 0)

        if layer == 0:
            if support_id is not None:
                problems.append(f"{tag}: layer 0 must sit on the table, "
                                f"not on block {support_id}")
            placed[bid] = p
            continue

        if support_id is None:
            problems.append(f"{tag}: {block['name']} is on layer {layer} "
                            f"but has no support")
            continue
        if support_id not in placed:
            problems.append(f"{tag}: {block['name']} sits on block "
                            f"{support_id}, which has not been placed yet")
            continue

        support = by_id[support_id]

        if support["solidity"] < MIN_SUPPORT_SOLIDITY:
            problems.append(
                f"{tag}: {support['name']} has an irregular top "
                f"(solidity {support['solidity']}) and cannot support "
                f"{block['name']}")

        if block["width"] > support["width"] * MAX_WIDTH_RATIO:
            problems.append(
                f"{tag}: {block['name']} (width {block['width']}) is wider "
                f"than its support {support['name']} "
                f"(width {support['width']})")

        offset = p.get("offset", [0, 0])
        try:
            ox, oy = float(offset[0]), float(offset[1])
        except (TypeError, IndexError, ValueError):
            problems.append(f"{tag}: offset {offset!r} is not [x, y]")
            ox = oy = 0.0
        reach = (ox * ox + oy * oy) ** 0.5
        if reach > MAX_CENTRE_OFFSET:
            problems.append(
                f"{tag}: {block['name']} is offset {reach:.2f} of "
                f"{support['name']}'s width from centre; max "
                f"{MAX_CENTRE_OFFSET} keeps the centre of mass over support")

        # Collision with anything already sharing this support.
        for other_id, other_p in placed.items():
            if other_p.get("on_top_of") != support_id:
                continue
            other = by_id[other_id]
            o_off = other_p.get("offset", [0, 0])
            try:
                oox, ooy = float(o_off[0]), float(o_off[1])
            except (TypeError, IndexError, ValueError):
                continue
            gap = (((ox - oox) * support["width"]) ** 2
                   + ((oy - ooy) * support["width"]) ** 2) ** 0.5
            need = (block["width"] + other["width"]) / 2 * MIN_SEPARATION
            if gap < need:
                problems.append(
                    f"{tag}: {block['name']} overlaps {other['name']} — both "
                    f"sit on {support['name']} and their centres are only "
                    f"{gap:.2f} apart, needs {need:.2f}")

        placed[bid] = p

    return len(problems) == 0, problems


def render(frame, blocks, plan, ok, problems):
    """Camera view with detections, beside the plan as text."""
    view = frame.copy()
    for b in blocks:
        colour = auto_pose.PALETTE[b["id"] % len(auto_pose.PALETTE)]
        cnts, _ = cv2.findContours(b["mask"].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, cnts, -1, colour, 2)
        x, y = b["px"]
        label = f"{b['id']} {b['name']}"
        cv2.putText(view, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3)
        cv2.putText(view, label, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1)

    panel = np.full((auto_pose.WORK_H, 560, 3), 20, np.uint8)
    lines = []
    if plan:
        lines.append(("CONCEPT", (120, 200, 255)))
        for chunk in _wrap(plan.get("concept", "(none)"), 60):
            lines.append((f"  {chunk}", (220, 220, 220)))
        lines.append(("", None))
        lines.append(("BUILD ORDER", (120, 200, 255)))
        by_id = {b["id"]: b for b in blocks}
        for n, p in enumerate(plan.get("placements", []), 1):
            b = by_id.get(p.get("block_id"))
            nm = b["name"] if b else f"?{p.get('block_id')}"
            sup = p.get("on_top_of")
            where = "on the table" if sup is None else \
                f"on {by_id[sup]['name']}" if sup in by_id else f"on {sup}"
            lines.append((f"  {n}. {nm} {where}", (220, 220, 220)))
    else:
        lines.append(("no plan", (150, 150, 150)))

    lines.append(("", None))
    if ok:
        lines.append(("VALIDATION: PASSED", (120, 230, 120)))
    else:
        lines.append(("VALIDATION: REJECTED", (90, 90, 255)))
        for pr in problems[:8]:
            for chunk in _wrap(pr, 58):
                lines.append((f"  {chunk}", (150, 150, 240)))

    y = 30
    for text, col in lines:
        if text:
            cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, col, 1)
        y += 20
    return np.hstack([view, panel])


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
    goal = (argv[argv.index("--goal") + 1] if "--goal" in argv
            else DEFAULT_GOAL)
    grid = int(argv[argv.index("--grid") + 1]) if "--grid" in argv else 11

    if "--image" in argv:
        frame = cv2.imread(argv[argv.index("--image") + 1])
        frame = cv2.resize(frame, (auto_pose.WORK_W, auto_pose.WORK_H))
    else:
        frame = capture()
    if frame is None:
        print("no frame")
        return

    print("surveying blocks ...")
    blocks = survey(frame, grid)
    if not blocks:
        print("no blocks found")
        return
    print(f"{len(blocks)} blocks:")
    for b in blocks:
        print(f"  id={b['id']} {b['name']:22s} w={b['width']:.2f} "
              f"d={b['depth']:.2f} footprint={b['footprint']:.2f} "
              f"solidity={b['solidity']:.2f} shape={b['shape']}")

    if "--no-vlm" in argv:
        print("\n--no-vlm: skipping planning")
        return

    print(f"\ngoal: {goal}")
    print("asking the VLM for a structure ...")
    plan, raw = ask_vlm(frame, blocks, goal)
    if plan is None:
        print("  VLM did not return usable JSON. Raw response:")
        print(raw[:600])
        return

    apply_heights(plan, blocks)
    resolve_offsets(plan, blocks)
    print(f"\nconcept: {plan.get('concept')}")
    ok, problems = validate(plan, blocks)
    print(f"\nvalidation: {'PASSED' if ok else 'REJECTED'}")
    for pr in problems:
        print(f"  - {pr}")

    if not ok:
        print("\nasking the VLM to revise ...")
        plan2, raw2 = ask_vlm(frame, blocks, goal,
                              retry_note="\n".join(problems))
        if plan2:
            apply_heights(plan2, blocks)
            resolve_offsets(plan2, blocks)
            ok2, problems2 = validate(plan2, blocks)
            print(f"revised concept: {plan2.get('concept')}")
            print(f"revised validation: "
                  f"{'PASSED' if ok2 else 'STILL REJECTED'}")
            for pr in problems2:
                print(f"  - {pr}")
            plan, ok, problems = plan2, ok2, problems2

    PLAN_PATH.write_text(json.dumps(
        {"goal": goal, "plan": plan, "valid": ok, "problems": problems,
         "px_per_unit": px_per_unit(blocks),
         "blocks": [{k: v for k, v in b.items() if k != "mask"}
                    for b in blocks]}, indent=2))
    cv2.imwrite(str(OUT_PATH), render(frame, blocks, plan, ok, problems))
    print(f"\nwrote {PLAN_PATH.name} and {OUT_PATH.name}")


if __name__ == "__main__":
    main(sys.argv)
