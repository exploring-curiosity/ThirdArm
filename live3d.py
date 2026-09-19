"""Real-time 3D object tracking.

Design follows what the measurements allow, not what would be ideal:

  colour JPEG      38 KB    ~36 Hz   -> drives the tracking rate
  depth map       1.8 MB    ~6.6 Hz  -> refreshed in the background
  camera pose (arm FK)               -> 103 Hz, exact, no SLAM needed
  point cloud     14.7 MB   ~0.8 Hz  -> never used in the loop
  det-to-segment            ~1.3 Hz  -> never used in the loop

The two streams are decoupled: blob detection runs on every colour frame, while
a separate task keeps the depth map fresh. Tracking reads the most recent depth
rather than waiting for a new one. This works because x/y changes every frame as
an object slides, while z barely changes for something resting on a flat table.

Depth is also fused across frames per object, so a viewpoint where the sensor
drops out is covered by earlier ones.

    python live3d.py                 # track everything, print to terminal
    python live3d.py --view          # with a live annotated window
    python live3d.py --colour blue   # track one colour only
"""
import asyncio
import sys
import time

import cv2
import numpy as np

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.media.video import CameraMimeType
from viam.proto.common import Pose, PoseInFrame

from tutorial import connect, _decode_depth

# HSV windows, measured from the live camera. Applied locally so no detector
# RPC is needed — that call alone costs 68 ms, which would halve the rate.
COLOURS = {
    "orange": ((0, 110, 80), (18, 255, 255), (0, 140, 255)),
    "yellow": ((20, 90, 90), (38, 255, 255), (0, 220, 240)),
    # Widened 2026-09-19 to cover the blue drop box, a crumpled matte
    # container the old band missed entirely (0 of 21000 sampled pixels
    # matched). MEASURED on its face: H 114-119, S ~158, V ~64 -- past the
    # old hue ceiling of 110 and below its value floor of 70. The wider band
    # still has zero pixel overlap with orange or green in the same frame.
    "blue": ((95, 80, 40), (125, 255, 255), (255, 140, 0)),
    "green": ((40, 80, 60), (85, 255, 255), (80, 220, 80)),
}

MIN_AREA, MAX_AREA = 600, 12000
MIN_SIDE, MAX_SIDE = 20, 140

WORKSPACE = dict(x=(150.0, 750.0), y=(-250.0, 300.0), z=(-40.0, 250.0))

DEPTH_PATCH = 6          # half-width of the depth sample around a blob centre
FUSE_ALPHA = 0.35        # weight of a new z reading against the running estimate
STALE_AFTER_S = 1.0      # drop a track not seen for this long


class Track:
    """One object's fused state across frames."""

    def __init__(self, colour, pose):
        self.colour = colour
        self.x, self.y, self.z = pose
        self.last_seen = time.monotonic()
        self.hits = 1

    def update(self, pose):
        x, y, z = pose
        # x/y follow the detection directly — they are precise and change fast.
        self.x, self.y = x, y
        # z is fused: the depth sensor drops out on these surfaces, so a single
        # bad reading should not move the estimate far.
        self.z += FUSE_ALPHA * (z - self.z)
        self.last_seen = time.monotonic()
        self.hits += 1

    @property
    def stale(self):
        return time.monotonic() - self.last_seen > STALE_AFTER_S


def find_blobs(hsv, colour, max_scale=1.0):
    """Colour blobs as (cx, cy, x, y, w, h), filtered to plausible sizes.

    The upper size limits exist to reject the carpet, walls and other large
    coloured expanses at survey range. They are calibrated for the ~670 mm
    top-pose view, where a table-top object is well under 140 px across.

    `max_scale` widens ONLY those upper limits. It matters during a descent:
    apparent size goes as 1/range, so the 72x52 px object seen from top-pose
    passes 140 px at roughly 350 mm and is then thrown away as implausible —
    the tracker goes blind exactly when the grasp needs it most. Callers that
    approach an object pass a scale derived from how far they have closed in.
    The lower limits are never relaxed, so specks are still rejected.
    """
    lo, hi, _ = COLOURS[colour]
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    max_area = MAX_AREA * max_scale * max_scale
    max_side = MAX_SIDE * max_scale
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if not (MIN_AREA < area < max_area):
            continue
        if not (MIN_SIDE < w < max_side and MIN_SIDE < h < max_side):
            continue
        out.append((int(centroids[k][0]), int(centroids[k][1]), x, y, w, h))
    return out


async def depth_pump(cam, state):
    """Keep the most recent depth map available, at whatever rate it allows."""
    while not state["stop"]:
        try:
            images, _ = await cam.get_images(filter_source_names=["depth"])
            state["depth"] = _decode_depth(images[0].data)
            state["depth_n"] += 1
        except Exception as exc:
            print(f"  depth error: {exc}")
        await asyncio.sleep(0)


async def pose_pump(machine, state):
    """Keep the cam->world transform fresh, as a matrix we can apply locally.

    transform_pose is a network round trip (~9 ms). Calling it per blob per
    frame capped the loop at 3.5 Hz. Instead, recover the 3x3 rotation and
    translation once per refresh by transforming four known points, then apply
    it to every blob locally with numpy — verified exact to 0.000000 mm.

    The camera is wrist-mounted, so this must keep refreshing as the arm moves;
    it must never be cached across a motion.
    """
    async def at(x, y, z):
        pif = await machine.transform_pose(
            PoseInFrame(reference_frame="cam", pose=Pose(x=x, y=y, z=z, o_z=1)),
            "world",
        )
        return np.array([pif.pose.x, pif.pose.y, pif.pose.z])

    while not state["stop"]:
        try:
            # All four probes are independent — issue them together so the
            # refresh costs one round trip rather than four.
            origin, ex, ey, ez = await asyncio.gather(
                at(0, 0, 0), at(100, 0, 0), at(0, 100, 0), at(0, 0, 100)
            )
            state["R"] = np.stack(
                [(ex - origin) / 100.0,
                 (ey - origin) / 100.0,
                 (ez - origin) / 100.0], axis=1
            )
            state["T"] = origin
        except Exception:
            pass
        await asyncio.sleep(0.05)


def to_world(machine_pose, intr, px, py, z_mm):
    """Deproject a pixel to camera-frame mm. Caller transforms to world."""
    x = (px - intr.center_x_px) * z_mm / intr.focal_x_px
    y = (py - intr.center_y_px) * z_mm / intr.focal_y_px
    return x, y, z_mm


def in_workspace_xyz(v):
    return (WORKSPACE["x"][0] < v[0] < WORKSPACE["x"][1]
            and WORKSPACE["y"][0] < v[1] < WORKSPACE["y"][1]
            and WORKSPACE["z"][0] < v[2] < WORKSPACE["z"][1])


async def main(argv):
    show = "--view" in argv
    only = None
    if "--colour" in argv:
        only = argv[argv.index("--colour") + 1]

    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        arm = Arm.from_robot(machine, "arm")
        intr = (await cam.get_properties()).intrinsic_parameters

        state = {"depth": None, "R": None, "T": None,
                 "stop": False, "depth_n": 0}

        # Start both pumps and the first colour fetch together. Depth (~166 ms)
        # and the pose matrix (~37 ms) are independent, so running them
        # concurrently costs only the slower of the two instead of their sum.
        t_start = time.monotonic()
        pumps = [
            asyncio.create_task(depth_pump(cam, state)),
            asyncio.create_task(pose_pump(machine, state)),
        ]

        # Poll finely rather than on a 100 ms tick: the data usually arrives in
        # ~170 ms, and a coarse tick added up to 100 ms of pure waiting.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if state["depth"] is not None and state["R"] is not None:
                break
            await asyncio.sleep(0.005)

        if state["depth"] is None or state["R"] is None:
            print("no depth or pose — aborting")
            state["stop"] = True
            for t in pumps:
                t.cancel()
            return
        print(f"ready in {(time.monotonic() - t_start) * 1000:.0f} ms")

        wanted = [only] if only else list(COLOURS)
        tracks = {}
        frames = 0
        t0 = time.monotonic()
        last_print = 0.0

        print(f"tracking {', '.join(wanted)} — ctrl-c to stop")
        if show:
            print("  (press q in the window to quit)")

        try:
            while True:
                images, _ = await cam.get_images(filter_source_names=["color"])
                jpeg = next(
                    (i for i in images if i.mime_type == CameraMimeType.JPEG), None
                )
                if jpeg is None:
                    continue
                bgr = cv2.imdecode(
                    np.frombuffer(jpeg.data, np.uint8), cv2.IMREAD_COLOR
                )
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                depth = state["depth"]
                frames += 1

                for colour in wanted:
                    for cx, cy, bx, by, bw, bh in find_blobs(hsv, colour):
                        patch = depth[
                            max(0, cy - DEPTH_PATCH):cy + DEPTH_PATCH + 1,
                            max(0, cx - DEPTH_PATCH):cx + DEPTH_PATCH + 1,
                        ]
                        valid = patch[patch > 0]
                        if valid.size < 20:
                            continue
                        # Nearest quartile, not the median: the patch straddles
                        # the object's top face and the table beyond its edge.
                        z_mm = float(np.percentile(valid, 25))
                        cam_xyz = np.array(to_world(None, intr, cx, cy, z_mm))
                        world = state["R"] @ cam_xyz + state["T"]
                        if not in_workspace_xyz(world):
                            continue
                        p = (float(world[0]), float(world[1]), float(world[2]))
                        key = colour
                        if key in tracks and not tracks[key].stale:
                            tracks[key].update(p)
                        else:
                            tracks[key] = Track(colour, p)

                        if show:
                            _, _, bgr_colour = COLOURS[colour]
                            cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh),
                                          bgr_colour, 2)
                            t = tracks[key]
                            cv2.putText(
                                bgr,
                                f"{colour} {t.x:.0f},{t.y:.0f},{t.z:.0f}",
                                (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                bgr_colour, 2,
                            )

                for key in [k for k, t in tracks.items() if t.stale]:
                    del tracks[key]

                elapsed = time.monotonic() - t0
                hz = frames / elapsed

                if show:
                    cv2.putText(
                        bgr,
                        f"{hz:.1f} Hz track | {state['depth_n']/elapsed:.1f} Hz depth",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    )
                    cv2.imshow("live3d", bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                elif elapsed - last_print > 0.5:
                    last_print = elapsed
                    row = "  ".join(
                        f"{c}({t.x:6.1f},{t.y:6.1f},{t.z:5.1f})"
                        for c, t in sorted(tracks.items())
                    )
                    print(f"\r{hz:5.1f} Hz  {row}          ", end="", flush=True)

                await asyncio.sleep(0)
        except KeyboardInterrupt:
            print()
        finally:
            state["stop"] = True
            for t in pumps:
                t.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            if show:
                cv2.destroyAllWindows()
                for _ in range(5):
                    cv2.waitKey(1)
            elapsed = time.monotonic() - t0
            print(f"\n{frames} frames in {elapsed:.1f}s -> {frames/elapsed:.1f} Hz "
                  f"tracking, {state['depth_n']/elapsed:.1f} Hz depth")


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
