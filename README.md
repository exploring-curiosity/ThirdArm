# ThirdArm

Vision-guided pick and place on a Viam machine: a uFactory arm with a
wrist-mounted RealSense camera picks coloured objects off a table and drops them
into a green box.

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install viam-sdk python-dotenv opencv-python numpy
```

Create `.env` next to the scripts:

```
VIAM_API_KEY=...
VIAM_API_KEY_ID=...
VIAM_MACHINE_ADDRESS=your-machine.viam.cloud
```

All scripts load it from their own directory, so they work from any working
directory. Activate the venv before running anything below.

## Quick start

```
python multi_pick.py --list                 # what is on the table
python multi_pick.py --dry-run blue         # plan it, move nothing
python multi_pick.py blue                   # pick it up and drop it in the box
```

## Scripts

| script | what it does |
|---|---|
| `multi_pick.py` | **The main one.** Pick any coloured object, drop it in the green box. |
| `multi_pick_box.py` | Same as `multi_pick.py`, plus excludes objects already sitting in the green box. |
| `viewer.py` | Live camera feed with detections and world coordinates. Never moves the arm. |
| `tutorial.py` | The original walkthrough: saved poses, gripper, live feed. |
| `seg_pick.py` | Pick using the `det-to-segment` 3D segmenter. Superseded by `multi_pick.py`. |
| `pc_slow_pick.py` | Pick using raw point-cloud colour segmentation. Earlier approach. |
| `slow_pick.py` | Stepped-motion helpers. Imported by the others, also runnable alone. |

### multi_pick.py

```
python multi_pick.py [--list|--locate|--dry-run] [--slow] [--no-drop] <colour>
```

Colours: `orange`, `yellow`, `blue`. Each has its own `color-detector-*` service
on the machine, and the colour you name selects which one to query.

| flag | effect |
|---|---|
| `--list` | Show every colour's on-table object and exit. |
| `--locate` | Measure the named colour and exit. No motion. |
| `--dry-run` | Full plan with collision checks. No motion. |
| `--slow` | Stepped crawl (15 mm waypoints, 0.6 s pause) instead of smooth motion. |
| `--no-drop` | Lift and hold instead of dropping in the green box. |

The sequence: go to `top-pose`, locate the object, locate the green drop box,
then hover → descend → grab → lift → carry → release.

`--dry-run` and `--locate` look from wherever the arm currently is. A full run
sends it to `top-pose` first. If a dry run reports nothing detected, check the
arm is actually at `top-pose`.

### multi_pick_box.py

```
python multi_pick_box.py [--list|--locate|--dry-run] [--slow] [--no-drop] <colour>
```

Identical to `multi_pick.py`, with one addition: the green drop box is located
*before* the target colour, and any candidate detection that falls inside it is
excluded. Without this, a colour detector can still fire on an object already
dropped in the box (seen through the open top), and the arm would try to pick
it up again. The exclusion is skipped under `--no-drop`, since the box position
is never located in that mode.

### viewer.py

```
python viewer.py
```

Live feed with detection boxes and world coordinates. `q` quits, `a` toggles
between showing only the target label and showing everything. Read-only — useful
for checking what the detectors see before committing to a pick.

## How it works

1. **Detect** — one `color_detector` service per colour returns 2D boxes.
2. **Filter** — three stages, all necessary:
   - bbox size 25-110 px (the detectors return large background regions; blue's
     biggest was 409x719 at the image corner)
   - 3D workspace bounds (blue carpet on the floor otherwise reads as an object)
   - frame persistence over 8 frames, keeping the most consistent cluster
3. **Localise** — deproject the detection through the point cloud, take the
   nearest eighth of points as the object's top face, transform `cam` → `world`.
4. **Correct** — apply a measured horizontal offset (below).
5. **Move** — one continuous motion per leg, with the straight-line path checked
   against a collision floor first.

The camera is wrist-mounted, so a detection is only valid for the arm pose it
was captured at. Every locate transforms to world coordinates immediately.

## Tuning constants

| constant | file | value | meaning |
|---|---|---|---|
| `GRASP_OFFSET_X/Y` | `seg_pick.py` | 13.57, 16.58 | Measured horizontal correction, mm |
| `WORKING_GRASP_Z` | `seg_pick.py` | 28.36 | Hand-verified `reach-cuboid` grasp height |
| `APPROACH_CLEARANCE_MM` | `seg_pick.py` | 120 | Hover height above the object |
| `DROP_CLEARANCE_MM` | `multi_pick.py` | 90 | Release height above the box rim |
| `BOX_EXCLUSION_RADIUS_MM` | `multi_pick_box.py` | 90 | Half-width of the drop-box exclusion zone, mm (unmeasured, tune as needed) |
| `WORKSPACE` | `multi_pick.py` | x 200-700, y ±200, z 20-150 | Valid object positions, mm |
| `MIN/MAX_BOX_PX` | `multi_pick.py` | 25, 110 | Plausible object size on screen |
| `STEP_MM` / `STEP_PAUSE_S` | `slow_pick.py` | 15, 0.6 | Stepped-mode granularity |

## Known issues

**A ~21 mm horizontal offset.** Vision consistently reports objects short of
where the gripper actually grasps them. Confirmed systematic across two objects
and two independent measurement methods, so it is corrected via
`GRASP_OFFSET_X/Y`. The cause is unconfirmed — most likely camera-mount
calibration or a gripper jaw offset the config does not model. **The correction
was measured at one table position**; if the cause is camera-mount error it may
vary across the workspace. See `docs/measurements/vision-vs-grasp-calibration.md`.

**Top-face height is only good to about ±5 mm.** Three objects known to be
identical 60 mm measured 49.7, 54.0 and 57.8 mm tall. The table beneath them is
flat to 0.8 mm, so this is measurement error, not geometry — objects without a
cleanly separated flat top face pull the estimate low. The taller green box
measures far better (97.3 vs 100 mm). See `docs/measurements/equal-height-objects.md`.

**The table collision geometry is wrong.** The configured model demands the
gripper stay above z = 41.0, but the hand-recorded `reach-cuboid` grasp at
z = 28.36 physically works. Since the model rejects a grasp that demonstrably
succeeds, the scripts use the verified height as the floor instead. Fixing this
properly means correcting the `table` frame or the `claws` box in the machine
config. See `docs/measurements/table-collision-geometry.md`.

**The detectors are noisy frame to frame.** They drop real objects and invent
phantoms — over 12 frames of a stationary scene, three phantom labels appeared
in 2-5 frames each, one with a 877x1045x1006 mm bounding box. Hence the
persistence filtering. Never trust a single detection call.

**The two red objects are 4 hue units apart** (H=2 and H=6), so no colour
detector can separate them. Telling them apart needs a trained model.

## Machine resources

Components: `arm`, `cam`, `gripper`, plus `table` / `wall-front` / `wall-side` /
`ceiling` as collision geometry.

Saved poses (arm-position-saver switches): `home-pose`, `start-position`,
`top-pose`, `reach-cuboid`. Each is a 3-position switch —
**0 = idle, 1 = update config (overwrites the saved pose), 2 = go to**. The code
only ever sends 2; sending 1 would silently overwrite a pose.

Vision services: `color-detector-orange` / `-yellow` / `-blue` / `-green`,
plus `detector` and `det-to-segment`. Each colour detector handles exactly one
colour.

## Notes

- `motion.move` takes `component_name` as a **string**, not a `ResourceName`.
  Passing the object raises `TypeError: bad argument type for built-in operation`.
- `arm.get_end_position()` returns the **wrist**, 150 mm behind the gripper. For
  anything commanded in the gripper frame, use `transform_pose(gripper → world)`.
- `camera.get_images()` returns both a colour JPEG and a ~1.8 MB depth map.
  Always pass `filter_source_names=["color"]` for video — fetching both caps the
  live feed at ~7 fps instead of ~41.
- The connection to the machine drops intermittently
  (`deadline has elapsed` / `StreamTerminatedError`). Retry; it is not a code fault.

## Measurements

Every calibration result and experiment is written up in `docs/measurements/`,
with the raw numbers and what they rule in or out. `docs/memory/` holds session
state: decisions, open todos, and accumulated context.
