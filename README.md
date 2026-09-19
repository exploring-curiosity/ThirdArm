# ThirdArm

Vision-guided pick and place on a Viam machine. A uFactory arm with a
wrist-mounted RealSense picks coloured blocks off a table and sorts them into
drop boxes by colour. A fixed overhead webcam watches the whole table.

The usual way to run it is the web service: one long-lived process that keeps
the robot session, the SAM 3 model and the detectors warm, and performs picks
on demand.

```
.venv/bin/python web_gui.py          # then open http://localhost:8752
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` next to the scripts:

```
VIAM_API_KEY=...
VIAM_API_KEY_ID=...
VIAM_MACHINE_ADDRESS=your-machine.viam.cloud
```

All scripts load it from their own directory, so they work from any working
directory.

SAM 3 needs three local patches on Apple Silicon. `python patch_sam3.py
--check` exits non-zero if they are missing; `python patch_sam3.py` re-applies
them. Re-run it after any `pip install` that touches `sam3`. See
`docs/sam3-install-notes.md`.

## The web service

```
.venv/bin/python web_gui.py [--no-sam] [--port N]
```

Open `http://localhost:8752`. The page shows the wrist camera, the overhead
camera and a top-down plot, with buttons to pick a colour, sort the whole
table, stop, go home and re-measure the drop boxes.

It is a **service, not a program you re-run per pick**. At startup it opens one
robot session, loads SAM 3 once (~860M params, ~5 s), locates both drop boxes
and starts its perception loops. A pick is an action on that live state: no
model reload, no re-measuring, no second session. Two processes talking to the
same machine is also what causes `depth error: [Errno 2] No such file or
directory` — the machine serves one camera client reliably, so do not run a CLI
pick against a running service.

### Sorting the table

The **sort table** button clears the table unattended:

1. pick whichever object SAM is most confident about
2. drop it in that colour's box
3. return to `top-pose` and re-measure the decluttered scene
4. repeat until nothing is left, or after `SORT_MAX_FAILS` (3) failures

Routing is `DROP_FOR = {"orange": "green", "yellow": "blue"}`. Only those two
colours are sorted; the boxes themselves are found by their own colour
detectors. Objects already inside a box are excluded, so nothing is picked
back out.

Ordering is by SAM confidence, not colour, so it interleaves colours. An object
SAM never segmented is still pickable — just with a neutral wrist angle.

### Perception loops

| loop | does | rate |
|---|---|---|
| `depth_pump` / `pose_pump` | depth map and `cam`→`world` transform | continuous |
| `seg_loop` | `det-to-segment` point-cloud objects | ~1.3 Hz, intermittent |
| `sam_loop` | SAM 3 grasp angles | ~10 s/pass, event-driven |
| `box_loop` | locate both drop boxes, once | at startup |
| `render_loop` | the three camera panels, overlays, commands | ~15 Hz |

Object tracking runs inside the pick loop and the overhead watch rather than as
a separate pump; see below.

SAM re-runs when the scene changes, when a pick finishes, or after
`SAM_MAX_AGE_S` (3 s) — not on a fixed timer. It waits for the wrist to be
still (`STILL_S` 0.4 s), because a frame grabbed mid-move is blurred and its
angle is wrong.

### Object motion detection and tracking

Objects are tracked continuously, from two independent views.

The **wrist camera** tracks the target through the whole approach: `observe()`
finds every blob of the target colour each frame, `nearest()` matches the one
belonging to this target, and the estimate is smoothed into `Target` so the arm
re-aims as the object shifts. Blob size limits scale with camera range
(`MAX_BLOB_SCALE`), so the same object stays trackable from survey height down
to grasp distance.

The **overhead camera** tracks the whole table from a fixed mount, so its view
cannot degrade as the arm closes in. `Overhead.locate()` maps any colour blob
to world coordinates through the calibrated homography, and objects are matched
frame to frame by proximity so a second block of the same colour cannot be
mistaken for this one having jumped.

Motion is judged **overhead-to-overhead** — one sensor against its own previous
reading — which cancels the homography's own bias. Comparing across sensors
(overhead against the point cloud) made a standing calibration disagreement
read as continuous motion. Detection thresholds come from measurement: a
stationary block steps at most 1.3 mm between overhead frames, so `MOVED_MM`
(20 mm) sits well clear of the noise floor, and a displacement must persist to
a second frame before it counts.

A `Target` exposes this as `disturbed` for a short window after a move, and the
pick's matching radius widens while it is set (`SEG_MATCH_DISTURBED_MM`) so the
segmenter can re-find an object that has just been displaced.

## Which detector supplies what

This is the core design decision. No single detector is trusted for
everything; each supplies only what it measures well.

| quantity | source | why |
|---|---|---|
| **x / y** | point cloud, else the colour blob | The only measurements validated against the arm. Both deproject through the same depth map. |
| **z (grasp height)** | `det-to-segment` box top (`centre + dims.z/2`) | The depth map does not resolve object height at this range. |
| **grasp angle** | SAM 3 | Nothing else measures one. Matched to the chosen object by position within `SAM_MATCH_MM`. |
| **which object** | colour detector | `det-to-segment` labels are not colour names. |
| **drop box** | `color-detector-green` / `-blue` | Each box has its own detector. |

SAM supplies the **angle only**. It never sets a position — its
`xyz_approx` field is named that way to make misuse visible.

## Command-line tools

The service is the normal path. These remain for calibration and debugging,
and **must not run while the service is up**.

| script | what it does |
|---|---|
| `track_pick.py` | One closed-loop pick, tracking the object all the way down. `web_gui` imports its `run_pick`. |
| `multi_pick_box.py` | Single open-loop pick with drop-box exclusion. |
| `multi_pick.py` | Single pick, no exclusion. |
| `overhead.py --calibrate` | Fit the overhead camera's homography. Required once. |
| `calib_gui.py` | Side-by-side calibration capture with a live preview. |
| `auto_pose.py` | SAM segmentation and grasp-angle analysis on one frame. |
| `viewer.py` | Live detections and world coordinates. Read-only. |
| `seg_pick.py` | Pick via the 3D segmenter. Superseded. |
| `patch_sam3.py` | Re-apply the SAM 3 Apple Silicon patches. |

```
.venv/bin/python track_pick.py [--sam] [--slow] [--dry-run] [--watch] <colour>
```

## Geometry and conventions

**Grasp angle.** The jaws close along the gripper's **+y** axis, so
`theta = 90 - world_direction`. Established from a physical test after a
closed-loop test had passed at 0.04° against the same wrong assumption it was
built on — see `docs/memory/decisions.md`.

**Gripper vs wrist.** `arm.get_end_position()` returns the **wrist**, 150 mm
behind the gripper. Use `gripper_pose_in_world()` for anything in the gripper
frame.

**`motion.move` takes a string** component name (`"gripper"`), not a
`ResourceName`.

**Drop-box exclusion is rectangular**, ±90 × ±155 mm, from the measured
147 × 276 mm box. A square excluded only 1 of 5 objects actually inside it.

**Containers are identified by size, not colour** — any overhead blob with a
side ≥ `OH_CONTAINER_SIDE` (100 px). Measured: block 42×60, green box 118×199,
blue box 189×182. Fragments of a box are merged first (`OH_MERGE_PX` 25) and
hollow outlines rejected (`OH_MIN_FILL` 0.45), because a rim splits into
sliver-sized "objects" when the arm crosses it.

## Tuning constants

| constant | file | value | meaning |
|---|---|---|---|
| `GRASP_OFFSET_X/Y` | `seg_pick.py` | 13.57, 16.58 | Measured horizontal correction, mm |
| `MIN_GRASP_Z` | `track_pick.py` | 28.36 | Hand-verified grasp height |
| `APPROACH_CLEARANCE_MM` | `seg_pick.py` | 120 | Hover height above the object |
| `DROP_CLEARANCE_MM` | `multi_pick.py` | 90 | Release height above the box rim |
| `BOX_EXCLUSION_X/Y_MM` | `multi_pick_box.py` | 90, 155 | Drop-box exclusion half-extents |
| `SEG_FRESH_S` | `web_gui.py` | 2.0 | Point-cloud segments older than this are not used for grasp height |
| `SAM_MAX_AGE_S` | `web_gui.py` | 3.0 | SAM re-runs at least this often |
| `SCENE_MOVED_MM` | `web_gui.py` | 15.0 | Layout change that triggers a SAM pass |
| `GRID_WRIST` | `sam_observe.py` | 13 | SAM point-grid density |
| `SAM_MATCH_MM` | `track_pick.py` | 40 | Max distance for a SAM angle to belong to an object |
| `WORKSPACE` | `multi_pick.py` | x 150-800, y ±450, z -15-250 | Valid object positions, mm |

## Known issues

**SAM's grid can miss small objects.** A 13×13 grid steps 74 px on a 960×540
frame, and a 54 px block falls between the points — the object being picked had
no grasp axis while larger clutter did. The service seeds the grid with the
pixel of every detected colour blob (`extra_points`), which costs one prompt
per known object and cannot miss. Raising the grid to 19 would also work but
more than doubles a ~10 s pass.

**SAM is not realtime.** One grid-13 pass is ~10 s on an M5 Pro (MPS).
Profiling: image encoder 0.07 s, mask post-processing 0.21 s — the decoder
dominates. Prompts are batched 64 at a time (`PROMPT_BATCH`), worth ~1.12× with
bit-identical masks. ROI cropping gives nothing: the full grid is still
evaluated.

**`det-to-segment` is intermittent.** It often returns one object when the
table holds several, and its labels are not colour names. Hence the blob
fallback for position and the freshness window for height.

**A ~21 mm horizontal offset**, corrected by `GRASP_OFFSET_X/Y`. Systematic
across two objects and two measurement methods; cause unconfirmed, most likely
camera-mount calibration. Measured at one table position, so it may vary across
the workspace. See `docs/measurements/vision-vs-grasp-calibration.md`.

**Top-face height is good to about ±5 mm.** Three identical 60 mm objects
measured 49.7, 54.0 and 57.8. The table beneath is flat to 0.8 mm, so this is
measurement error. See `docs/measurements/equal-height-objects.md`.

**The table collision geometry is wrong.** The configured model demands z ≥ 41.0
but the hand-recorded grasp at 28.36 physically works, so the scripts use the
verified height. See `docs/measurements/table-collision-geometry.md`.

**The connection drops intermittently** (`deadline has elapsed` /
`StreamTerminatedError`). Retry. Clearing `/tmp/proxy-*.sock` after a hard kill
helps.

## Machine resources

Components: `arm`, `cam`, `gripper`, plus `table` / `wall-front` / `wall-side` /
`ceiling` as collision geometry.

Saved poses (arm-position-saver switches): `home-pose`, `start-position`,
`top-pose`, `reach-cuboid`. Each is a 3-position switch —
**0 = idle, 1 = update config (overwrites the saved pose), 2 = go to**. The code
only ever sends 2; sending 1 would silently overwrite a pose.

Vision services: `color-detector-orange` / `-yellow` / `-blue` / `-green`, plus
`detector` and `det-to-segment`.

The overhead camera is a Lenovo USB webcam at index 0, calibrated by homography
to the table plane (`overhead_calib.json`, 2.63 mm mean / 5.21 mm max residual).

## Development

Two static checks catch the failures that `py_compile` and `import` do not —
Python resolves names at runtime, so a function reading an undefined or
not-yet-assigned local passes both and fails only when that line executes:

```
.venv/bin/python tools/freevars.py web_gui.py track_pick.py   # name exists at all
.venv/bin/python tools/useorder.py web_gui.py track_pick.py   # name exists YET
```

Run both after moving code between functions. `useorder.py` caught a variable
left behind in `main()` during a refactor that would have raised mid-descent.

## Documentation

`docs/measurements/` holds every calibration result and experiment with its raw
numbers. `docs/memory/` holds session state: decisions and why they were made,
open todos, accumulated context, and dated session logs.
