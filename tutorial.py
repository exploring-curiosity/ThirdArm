import asyncio
import os
import struct
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from dotenv import load_dotenv

from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient
from viam.proto.common import Pose, PoseInFrame
from viam.components.switch import Switch
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.media.video import CameraMimeType
from viam.components.gripper import Gripper
from viam.services.generic import Generic as GenericService

load_dotenv(Path(__file__).with_name('.env'))


def _require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to the .env file next to tutorial.py.")
    return value


async def connect():
    opts = RobotClient.Options.with_api_key(
        api_key=_require_env('VIAM_API_KEY'),
        api_key_id=_require_env('VIAM_API_KEY_ID')
    )

    return await RobotClient.at_address(_require_env('VIAM_MACHINE_ADDRESS'), opts)

# The arm-position-saver switch exposes three positions:
#   0 = idle, 1 = update config (overwrites the saved pose!), 2 = go to saved pose.
# Only ever send 2 — sending 1 would clobber the pose saved in the Viam app.
GO_TO = 2

WINDOW = "ThirdArm live feed"

# The camera serves two sources; "depth" is ~48x larger and unused here.
COLOR_SOURCE = "color"


async def goto_saved_pose(machine, name):
    """Drive an arm-position-saver switch to its saved pose.

    set_position(2) blocks until the arm finishes moving, and the module
    returns itself to idle (0) on completion.
    """
    switch = Switch.from_robot(machine, name)
    print(f"moving to '{name}'...")
    await switch.set_position(GO_TO)
    print(f"reached '{name}'.")


async def _frame_pump(cam, state):
    """Continuously pull JPEG frames from the camera into `state`.

    Runs as a background task. Requests only the "color" source: the camera
    also serves a ~1.8 MB depth map, and fetching both caps the feed at ~7 fps
    (134 ms/frame) versus ~58 fps (17 ms/frame) for color alone. Decoding costs
    only ~1.5 ms, so the transfer is the entire bottleneck.
    """
    fps_t0 = time.monotonic()
    frames = 0
    while not state['stop']:
        try:
            images, _ = await cam.get_images(filter_source_names=[COLOR_SOURCE])
            jpeg = next(
                (i for i in images if i.mime_type == CameraMimeType.JPEG), None
            )
            if jpeg is not None:
                buf = np.frombuffer(jpeg.data, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is not None:
                    state['frame'] = frame
                    frames += 1
                    elapsed = time.monotonic() - fps_t0
                    if elapsed >= 1.0:
                        state['fps'] = frames / elapsed
                        frames, fps_t0 = 0, time.monotonic()
        except Exception as exc:  # keep the feed alive across transient errors
            print(f"  camera error: {exc}")
        # No sleep: the network round-trip (~17 ms) already paces the loop.
        # Yield so the robot sequence and window loop still get scheduled.
        await asyncio.sleep(0)


# ---------------------------------------------------------------- perception

DEPTH_SOURCE = "depth"
DEPTH_MAGIC = b"DEPTHMAP"
DEPTH_HEADER_BYTES = 24

# Which vision-service label counts as "the orange one". The configured
# color-detector reports hue ~6 (orange-red) under the name below; change this
# if you rename the detector's output class.
TARGET_LABEL = "red"

# How far above the detected object centre to hover before descending, and how
# far below the detected centre the gripper should close (objects are detected
# at their top face, but grasped around their middle).
# motion.move takes component_name as a STRING, not a ResourceName object.
GRIPPER_NAME = "gripper"

APPROACH_CLEARANCE_MM = 120.0
GRASP_DEPTH_MM = 15.0


def _decode_depth(raw):
    """Decode Viam's image/vnd.viam.dep buffer into a uint16 (h, w) array.

    Layout: b"DEPTHMAP" + width and height as big-endian uint64 + big-endian
    uint16 depths in millimetres, row-major. Same resolution as the colour
    frame, so detection pixels index straight into it.
    """
    if not raw.startswith(DEPTH_MAGIC):
        raise ValueError("unexpected depth payload: missing DEPTHMAP header")
    width, height = struct.unpack(">QQ", raw[8:DEPTH_HEADER_BYTES])
    depth = np.frombuffer(raw[DEPTH_HEADER_BYTES:], dtype=">u2")
    return depth.reshape(height, width)


def _depth_at(depth, cx, cy, half=5):
    """Median of the valid (non-zero) depths in a small patch around (cx, cy).

    A single pixel is often a dropout on shiny or dark surfaces, so sample a
    patch and ignore zeros. Returns None when the whole patch is invalid.
    """
    h, w = depth.shape
    patch = depth[
        max(0, cy - half):min(h, cy + half + 1),
        max(0, cx - half):min(w, cx + half + 1),
    ]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


async def locate_target(machine, cam, detector):
    """Find the orange object and return its position as a world-frame Pose.

    Returns None when nothing matching TARGET_LABEL is visible or the depth
    reading at the detection is invalid.
    """
    detections = await detector.get_detections_from_camera("cam")
    matches = [d for d in detections if d.class_name == TARGET_LABEL]
    if not matches:
        others = {d.class_name for d in detections}
        print(f"  no '{TARGET_LABEL}' detected"
              + (f" (saw: {', '.join(sorted(others))})" if others else ""))
        return None

    # Most confident, then largest — guards against a small spurious blob.
    best = max(
        matches,
        key=lambda d: (d.confidence,
                       (d.x_max - d.x_min) * (d.y_max - d.y_min)),
    )
    cx = (best.x_min + best.x_max) // 2
    cy = (best.y_min + best.y_max) // 2

    images, _ = await cam.get_images(filter_source_names=[DEPTH_SOURCE])
    depth = _decode_depth(images[0].data)
    z_mm = _depth_at(depth, cx, cy)
    if z_mm is None:
        print(f"  '{TARGET_LABEL}' found at px=({cx},{cy}) but depth is invalid there")
        return None

    # Deproject the pixel into camera-frame millimetres using the camera's
    # own intrinsics, then let the frame system convert cam -> world. The
    # camera is mounted on the arm, so this must be done from the same pose
    # the image was captured at.
    intr = (await cam.get_properties()).intrinsic_parameters
    x_cam = (cx - intr.center_x_px) * z_mm / intr.focal_x_px
    y_cam = (cy - intr.center_y_px) * z_mm / intr.focal_y_px

    in_cam = PoseInFrame(
        reference_frame="cam",
        pose=Pose(x=x_cam, y=y_cam, z=z_mm, o_z=1, theta=0),
    )
    in_world = await machine.transform_pose(in_cam, "world")
    p = in_world.pose
    print(f"  '{TARGET_LABEL}' conf={best.confidence:.2f} px=({cx},{cy}) "
          f"depth={z_mm:.0f}mm -> world=({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")
    return p


async def pick_at(machine, motion, gripper, target):
    """Approach from directly above, descend, grasp, and lift clear.

    Keeps the gripper pointing straight down (o_z=-1) throughout so the
    approach is a pure vertical descent onto the object.
    """
    down = dict(o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)

    above = Pose(x=target.x, y=target.y,
                 z=target.z + APPROACH_CLEARANCE_MM, **down)
    grasp = Pose(x=target.x, y=target.y,
                 z=target.z - GRASP_DEPTH_MM, **down)

    print("  opening gripper...")
    await gripper.open()

    print(f"  moving above target (z={above.z:.1f})...")
    await motion.move(
        component_name=GRIPPER_NAME,
        destination=PoseInFrame(reference_frame="world", pose=above),
    )

    print(f"  descending to grasp (z={grasp.z:.1f})...")
    await motion.move(
        component_name=GRIPPER_NAME,
        destination=PoseInFrame(reference_frame="world", pose=grasp),
    )

    print("  closing gripper...")
    grabbed = await gripper.grab()
    print("    grabbed." if grabbed else "    WARNING: grab() reported nothing grasped.")

    print("  lifting clear...")
    await motion.move(
        component_name=GRIPPER_NAME,
        destination=PoseInFrame(reference_frame="world", pose=above),
    )
    return grabbed


# ------------------------------------------------------------------ routine

async def run_sequence(machine, state):
    """Look from top-pose, find the orange object, pick it, drop at reach-cuboid."""
    try:
        gripper = Gripper.from_robot(machine, "gripper")
        cam = Camera.from_robot(machine, "cam")
        detector = VisionClient.from_robot(machine, "color-detector")
        motion = MotionClient.from_robot(machine, "builtin")

        # Survey the table from the overhead pose. The camera rides on the
        # arm, so the arm must be settled here before the image is taken.
        await goto_saved_pose(machine, "top-pose")
        print("locating target...")
        await asyncio.sleep(0.5)  # let the frame catch up with the motion

        target = await locate_target(machine, cam, detector)
        if target is None:
            print("nothing to pick — returning to start.")
            await goto_saved_pose(machine, "start-position")
            return

        print("picking...")
        await pick_at(machine, motion, gripper, target)

        # Carry it to the drop pose and release.
        await goto_saved_pose(machine, "reach-cuboid")
        print("releasing...")
        await gripper.open()
        print("  released.")

        await goto_saved_pose(machine, "start-position")
        print("sequence complete — press q in the video window to quit.")
    except Exception:
        traceback.print_exc()
    finally:
        state['done'] = True


async def main():
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        state = {'frame': None, 'stop': False, 'done': False, 'fps': 0.0}

        pump = asyncio.create_task(_frame_pump(cam, state))

        # Wait for the first frame so the window opens with something in it.
        print("waiting for first camera frame...")
        for _ in range(100):
            if state['frame'] is not None:
                break
            await asyncio.sleep(0.1)

        sequence = asyncio.create_task(run_sequence(machine, state))

        # cv2's window loop must run on the main thread on macOS. Yielding to
        # the event loop between waitKey calls lets the arm and camera tasks run.
        try:
            while True:
                frame = state['frame']
                if frame is not None:
                    cv2.putText(
                        frame, f"{state['fps']:.0f} fps", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
                    )
                    cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("quit requested.")
                    sequence.cancel()
                    break
                if state['done'] and sequence.done():
                    break
                await asyncio.sleep(0.01)
        finally:
            state['stop'] = True
            sequence.cancel()
            pump.cancel()
            await asyncio.gather(sequence, pump, return_exceptions=True)
            cv2.destroyAllWindows()
            # macOS needs a few more waitKey cycles to actually tear the window down.
            for _ in range(5):
                cv2.waitKey(1)


if __name__ == '__main__':
    asyncio.run(main())
