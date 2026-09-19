"""SAM 3 object detection on the wrist camera, in world coordinates.

This is the bridge between `auto_pose.py` (which segments and measures grasp
geometry but knows nothing about the robot) and `track_pick.py` (which drives
the arm but finds objects with HSV colour thresholds). Until now the two
shared no code.

What SAM is and is not allowed to decide
----------------------------------------
SAM supplies the GRASP ANGLE, and identifies which object to grasp. It does
NOT supply the position the arm moves to.

Position stays on the two sources that were actually validated:
  - x/y  the 2D blob centroid deprojected through the depth map, tracked at
         16 Hz and measured stable to +/-0.3 mm
  - z    det-to-segment's point cloud (box centre + dims.z/2), which carries
         each object's own height

The `xyz_approx` this module returns is a SAM mask centroid: a third estimate of the
same quantity, useful for ranking candidates and for reporting, and never used
to aim the arm. track_pick seeds its Target from the blob path and takes only
`grasp_deg` from here. Verified: with SAM returning nothing at all, the
commanded position changes by 0.000000 mm.

That separation is the point. Any single detector can fail -- SAM has been
observed missing an object entirely because no grid point landed on it -- and
when it does, the arm must degrade to its previous behaviour rather than move
somewhere wrong.

Why a separate survey step rather than a replacement for observe()
-----------------------------------------------------------------
SAM 3 takes 2.0-2.8 s on a 1280x720 wrist frame (measured, MPS). The tracking
loop in track_pick runs at 15 Hz. SAM cannot go in that loop, and it does not
need to: the grasp ANGLE of a rigid object does not change while the camera
approaches it. So SAM runs once, up front, to decide *what to pick and at what
wrist angle*; the existing fast HSV tracker then keeps x/y locked during the
descent. Perception and tracking run at the rates each can sustain.

Why the masks index straight into depth
---------------------------------------
The colour and depth streams are both 1280x720 and pixel-aligned (verified),
so a mask selects its own depth pixels with no registration step. Depth is
read over the whole mask rather than a fixed patch around the centroid, which
is what `observe()` does -- a mask knows exactly which pixels belong to the
object, so there is no need to guess a patch size.

Caveats carried over from the measurements
------------------------------------------
- The depth sensor does NOT resolve these blocks at top-pose range; z here is
  effectively the table plane. Treat `z` as "where the table is under this
  object", not the grasp height. See docs/measurements/why-z-is-wrong.md.
- GRID must be finer than the smallest object in pixels. The wrist view is
  wider than the overhead view, so objects are smaller and the overhead's
  GRID=9 misses every table block. GRID_WRIST=13 finds them, and even then a
  block can fall between grid points.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

import auto_pose as ap

# The wrist view is wider than the overhead view auto_pose was tuned for, so
# objects subtend fewer pixels and need a finer grid. Measured on the real
# scene: GRID=9 found zero table blocks, GRID=13 found the yellow and red.
GRID_WRIST = 13

# A mask must have depth on this fraction of its pixels to be trusted. A mask
# straddling an object edge picks up background depth and gives a centroid
# that belongs to neither.
MIN_DEPTH_COVER = 0.30

# Depth percentile taken over the mask. The 25th matches observe(): it leans
# toward the nearer surface, which is the object rather than the table showing
# through at the edges.
DEPTH_PCT = 25

# How far a measured hue may sit outside a COLOURS band and still be called
# that colour. The bands have gaps between them (a real yellow block measured
# H=19.0, between orange's 0-18 and yellow's 20-38), so exact containment
# rejects genuine objects.
MAX_HUE_DISTANCE = 8.0


def load(grid=GRID_WRIST):
    """Load SAM 3 once. Costs ~5 s, so hoist it out of any loop."""
    model, processor = ap.load_model()
    return {"model": model, "processor": processor, "grid": grid}


def _mask_world(mask, depth, intr, R, T):
    """World position of a mask's surface, or None if depth is too sparse."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None, "empty mask"

    d = depth[ys, xs]
    valid = d[d > 0]
    if valid.size < MIN_DEPTH_COVER * xs.size or valid.size < 20:
        return None, "no depth on mask"

    z_mm = float(np.percentile(valid, DEPTH_PCT))

    # Deproject the mask CENTROID, not the mean of per-pixel deprojections:
    # the centroid is where the gripper is aimed, and averaging world points
    # over a mask that includes edge pixels at table depth biases the result
    # toward the table.
    cx, cy = float(xs.mean()), float(ys.mean())
    cam = np.array([
        (cx - intr.center_x_px) * z_mm / intr.focal_x_px,
        (cy - intr.center_y_px) * z_mm / intr.focal_y_px,
        z_mm,
    ])
    return R @ cam + T, None


def survey(frame_bgr, depth, intr, R, T, sam, workspace=None, drop=None,
           in_drop_box=None, roi=None, rejects=None, extra_points=None):
    """Every graspable object SAM can see, in world coordinates.

    Returns a list of dicts. Each is exactly what `auto_pose.describe()`
    produced -- mask, contour, grasp_deg, grasp_width, solidity, name and the
    rest, so auto_pose's own draw() and report() work on this unchanged --
    plus the world-frame fields the arm needs:

        xyz_approx (x, y, z) in world mm, from the mask centroid. ADVISORY:
                   for ranking and reporting. Never used to aim the arm.
        feasible   alias of grasp_feasible: can the jaws span it
        depth_mm   raw camera-frame depth at that surface

    The field that matters most is `grasp_deg`. Nothing in the arm code has
    ever had a measured grasp angle before -- every grasp is issued with a
    hardcoded theta=0.0.

    `rejects` collects (xyz, reason) for anything found and discarded, so a
    failed survey can say why instead of reporting nothing. Most rejections
    here are correct and expected: the wrist view also contains the ceiling,
    the carpet, a power strip and the parts bin.
    """
    t0 = time.time()
    cands = ap.segment_all(sam["model"], sam["processor"], frame_bgr,
                           grid=sam["grid"], max_cover=ap.MAX_COVER, roi=roi,
                           extra_points=extra_points)

    out = []
    for mask, score in cands:
        obj = ap.describe(mask, score)
        if obj is None:
            continue

        # Shadow rejection, ported straight from auto_pose: SAM segments a
        # shadow as readily as an object, and a shadow has no depth of its own.
        if ap.is_shadow(frame_bgr, mask):
            if rejects is not None:
                rejects.append((None, f"{obj['name']}: shadow"))
            continue

        world, why = _mask_world(mask, depth, intr, R, T)
        if world is None:
            if rejects is not None:
                rejects.append((None, f"{obj['name']}: {why}"))
            continue

        x, y, z = (float(world[0]), float(world[1]), float(world[2]))

        # The workspace bound is what removes the ceiling, the carpet and the
        # far wall -- everything SAM correctly segments and the arm cannot or
        # must not touch.
        if workspace is not None:
            axis = None
            for k, v in (("x", x), ("y", y), ("z", z)):
                lo, hi = workspace[k]
                if not lo < v < hi:
                    axis = k
                    break
            if axis is not None:
                if rejects is not None:
                    rejects.append(((x, y, z),
                                    f"{obj['name']}: outside workspace {axis}"))
                continue

        if drop is not None and in_drop_box is not None:
            from viam.proto.common import Pose
            if in_drop_box(Pose(x=x, y=y, z=z, o_z=1), drop):
                if rejects is not None:
                    rejects.append(((x, y, z),
                                    f"{obj['name']}: already in the drop box"))
                continue

        # Keep every field describe() produced and add the world-frame ones.
        # Copying rather than rebuilding means auto_pose.draw() and report()
        # work on a survey result unchanged; rebuilding a subset meant
        # rediscovering each required key through a KeyError.
        record = dict(obj)
        record.update({
            # Named "approx" deliberately: this is SAM's own estimate, used
            # for ranking and reporting only. The arm is aimed from the blob
            # centroid and the segmenter, never from here.
            "xyz_approx": (x, y, z),
            "feasible": bool(obj["grasp_feasible"]),
            "depth_mm": float(np.percentile(
                depth[obj["mask"]][depth[obj["mask"]] > 0], DEPTH_PCT)),
        })
        out.append(record)

    # Order by how much each looks like a block sitting on the table, NOT by
    # area: sorting by area put the power strip and the parts-bin rim ahead of
    # every real block, because scene furniture is simply bigger. The three
    # signals that separated them in practice:
    #   - height above the table (blocks sit ON it; furniture stands well
    #     above it, at z = 176-227 mm against a block's 2-4 mm)
    #   - solidity (a block is convex; a bin rim is a thin concave outline)
    #   - jaw width, preferring objects comfortably inside the jaws' span
    out.sort(key=_pick_score, reverse=True)
    return out, time.time() - t0


# A block lies within this of the table plane. Measured: real blocks read
# z = 1.6 to 4.0 mm; the power strip 226 mm and the parts-bin rim 176 mm.
TABLE_Z_TOLERANCE_MM = 60.0


def _pick_score(o):
    """How much this looks like a graspable block, high is better.

    Deliberately simple and readable: three independent signals, each in
    [0, 1], multiplied. Any one being near zero rules the object out, which is
    the behaviour wanted -- a tall object is not a block no matter how solid.
    """
    z = o["xyz_approx"][2]
    on_table = max(0.0, 1.0 - abs(z) / TABLE_Z_TOLERANCE_MM)
    solid = float(o.get("solidity", 0.0))
    # Prefer a comfortable jaw span: neither a sliver nor nearly the full gape.
    frac = float(o["grasp_width"]) / ap.MAX_JAW_PX
    span = max(0.0, 1.0 - abs(frac - 0.45) / 0.55)
    return on_table * solid * span


def bbox_of(obj):
    """The mask's bounding box as (bx, by, bw, bh).

    track_pick's tracker speaks bounding boxes, so this is what hands a
    SAM-chosen object over to the fast HSV tracker for the descent.
    """
    ys, xs = np.nonzero(obj["mask"])
    bx, by = int(xs.min()), int(ys.min())
    return bx, by, int(xs.max()) - bx + 1, int(ys.max()) - by + 1


def colour_of(obj, frame_bgr):
    """Best-matching name from live3d.COLOURS for this mask.

    The tracker follows an object by colour, so a SAM-selected object has to
    be named in those terms to be handed over. Matching is done in HSV hue on
    the mask's own pixels rather than by thresholding the whole frame.
    """
    from live3d import COLOURS

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    pixels = hsv[obj["mask"]]
    if pixels.size == 0:
        return None
    # Saturated pixels only: a block's own colour, not its shading.
    sat = pixels[pixels[:, 1] > 80]
    if sat.shape[0] == 0:
        return None
    h, s_, v = (float(np.median(sat[:, 0])),
                float(np.median(sat[:, 1])),
                float(np.median(sat[:, 2])))

    # COLOURS maps name -> (hsv_lo, hsv_hi, bgr_draw). Match to the NEAREST
    # band rather than requiring exact containment: the bands were tuned for
    # cv2.inRange thresholding and have gaps between them. A real yellow block
    # measured H=19.0 with every one of its 3716 pixels saturated -- a clean
    # reading that fell in the 1-degree gap between orange (0-18) and yellow
    # (20-38) and matched nothing. Nearest-band matching is also more robust
    # to the lighting shifts that move a hue a few degrees.
    best, best_d = None, None
    for name, (lo, hi, _bgr) in COLOURS.items():
        if s_ < lo[1] or v < lo[2]:
            continue                      # too washed out to be this colour
        if lo[0] <= h <= hi[0]:
            d = 0.0                       # inside the band
        else:
            d = min(abs(h - lo[0]), abs(h - hi[0]))
        if best_d is None or d < best_d:
            best, best_d = name, d

    # Beyond this the hue is not that colour at all, it is simply the closest
    # of four. Half the widest gap between adjacent bands.
    return best if best_d is not None and best_d <= MAX_HUE_DISTANCE else None


def image_deg_to_world_theta(grasp_deg, R):
    """Convert auto_pose's image-plane grasp angle to a world theta.

    auto_pose measures `grasp_deg` in the image: the direction, in [0, 180),
    along which the jaws close. Verified on a synthetic bar 100 px long on the
    image x axis and 20 px wide: grasp_axis returns grasp_deg = 90, which is
    the narrow direction -- the one the jaws must close along. (auto_pose.draw
    draws its red ray at grasp_deg + 90 with length grasp_width; that ray is
    the jaw OPENING shown spanning the object, not the closing direction.)

    The arm wants an angle in the world xy plane.
    The two differ by however the camera is rotated, which R already encodes,
    so the mapping is derived from R rather than hardcoded -- the wrist camera
    moves, and a constant would silently go stale the moment it did.

    Measured at top-pose: image +x maps to world +1.13 deg and image +y to
    world -88.87 deg, i.e. the image y axis is FLIPPED relative to world y.
    A naive copy of grasp_deg would therefore mirror every angle.

    Returned modulo 180: two parallel jaws at 10 and at 190 degrees are the
    same grasp, and the smaller magnitude is the shorter wrist rotation.
    """
    a = np.radians(float(grasp_deg))
    # The grasp direction as an image-plane unit vector, pushed through R into
    # world coordinates. Only the xy components matter -- the jaws close in
    # the horizontal plane with the gripper pointing straight down.
    v = R @ np.array([np.cos(a), np.sin(a), 0.0])
    world_dir = np.degrees(np.arctan2(v[1], v[0]))

    # theta is NOT the world direction of the jaws, and the jaws close along
    # the gripper's +y axis, not its +x. Measured on the arm by rotating the
    # wrist and transforming the gripper frame's +y axis into world:
    #
    #     theta =   0 deg -> jaws close along world  90.0 deg
    #     theta =  30 deg -> jaws close along world  60.0 deg
    #     theta =  60 deg -> jaws close along world  30.0 deg
    #     theta =  90 deg -> jaws close along world   0.0 deg
    #     theta = -45 deg -> jaws close along world 135.0 deg
    #
    # so theta = 90 - direction, fitting all five to 0.04 deg.
    #
    # The earlier version used the +x axis and theta = 180 - direction. That
    # is self-consistent and verifies perfectly in a closed loop, but +x is
    # the axis the jaws span rather than close along, so every grasp came out
    # rotated by exactly 90 degrees. A closed-loop test cannot catch this: it
    # validates the maths against the same wrong assumption it was built on.
    theta = 90.0 - world_dir

    # Fold into (-90, 90]: a two-jaw gripper is symmetric under 180 deg, and
    # this keeps the wrist away from its rotation limits.
    theta = (theta + 90.0) % 180.0 - 90.0
    return float(theta)
