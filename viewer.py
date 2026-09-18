"""Live camera view with colour detections and world-frame positions.

Read-only: this never commands the arm. Use it to check what the vision
service sees and to tune the detector before running tutorial.py.

    .venv/bin/python viewer.py

Keys:  q = quit   a = show ALL detections, not just the target label
"""
import asyncio

import cv2
import numpy as np

from viam.components.camera import Camera
from viam.media.video import CameraMimeType
from viam.proto.common import Pose, PoseInFrame
from viam.services.vision import VisionClient

from tutorial import (
    COLOR_SOURCE,
    DEPTH_SOURCE,
    TARGET_LABEL,
    _decode_depth,
    _depth_at,
    connect,
)

WINDOW = "ThirdArm detections"

# BGR. The target gets its own colour so it stands out from other labels.
TARGET_COLOR = (0, 165, 255)   # orange
OTHER_COLOR = (0, 255, 255)    # yellow
TEXT_COLOR = (255, 255, 255)


async def _detect_pump(machine, cam, detector, state):
    """Detect, deproject to world coords, and stash results for the drawer.

    Kept separate from the frame pump: detection costs far more per call than
    a colour frame, so the video stays smooth while boxes refresh slower.
    """
    intr = (await cam.get_properties()).intrinsic_parameters
    while not state['stop']:
        try:
            detections = await detector.get_detections_from_camera("cam")
            images, _ = await cam.get_images(filter_source_names=[DEPTH_SOURCE])
            depth = _decode_depth(images[0].data)

            found = []
            for d in detections:
                cx = (d.x_min + d.x_max) // 2
                cy = (d.y_min + d.y_max) // 2
                z_mm = _depth_at(depth, cx, cy)
                world = None
                if z_mm is not None:
                    x_cam = (cx - intr.center_x_px) * z_mm / intr.focal_x_px
                    y_cam = (cy - intr.center_y_px) * z_mm / intr.focal_y_px
                    pif = await machine.transform_pose(
                        PoseInFrame(
                            reference_frame="cam",
                            pose=Pose(x=x_cam, y=y_cam, z=z_mm, o_z=1, theta=0),
                        ),
                        "world",
                    )
                    world = pif.pose
                found.append({
                    'label': d.class_name,
                    'conf': d.confidence,
                    'box': (d.x_min, d.y_min, d.x_max, d.y_max),
                    'px': (cx, cy),
                    'depth': z_mm,
                    'world': world,
                })
            state['dets'] = found
        except Exception as exc:
            state['dets'] = []
            print(f"  detection error: {exc}")
        await asyncio.sleep(0.05)


async def _frame_pump(cam, state):
    """Pull colour frames only — fast path, keeps the window responsive."""
    while not state['stop']:
        try:
            images, _ = await cam.get_images(filter_source_names=[COLOR_SOURCE])
            jpeg = next(
                (i for i in images if i.mime_type == CameraMimeType.JPEG), None
            )
            if jpeg is not None:
                frame = cv2.imdecode(
                    np.frombuffer(jpeg.data, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    state['frame'] = frame
        except Exception as exc:
            print(f"  camera error: {exc}")
        await asyncio.sleep(0)


def _draw(frame, state):
    """Draw boxes, labels and world coordinates onto a copy of the frame."""
    canvas = frame.copy()
    shown = 0

    for det in state['dets']:
        is_target = det['label'] == TARGET_LABEL
        if not is_target and not state['show_all']:
            continue
        shown += 1

        x0, y0, x1, y1 = det['box']
        color = TARGET_COLOR if is_target else OTHER_COLOR
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3 if is_target else 2)
        cv2.drawMarker(canvas, det['px'], color, cv2.MARKER_CROSS, 18, 2)

        lines = [f"{det['label']} {det['conf']:.2f}"]
        if det['world'] is not None:
            w = det['world']
            lines.append(f"x={w.x:.0f} y={w.y:.0f} z={w.z:.0f} mm")
            lines.append(f"depth {det['depth']:.0f} mm")
        else:
            lines.append("no valid depth")

        # Label above the box, or below it when there is no room up top.
        ty = y0 - 8 if y0 > 60 else y1 + 22
        for i, text in enumerate(lines):
            cv2.putText(canvas, text, (x0, ty + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(canvas, text, (x0, ty + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    mode = "ALL labels" if state['show_all'] else f"target '{TARGET_LABEL}' only"
    header = f"{shown} shown | {len(state['dets'])} detected | {mode} | a=toggle q=quit"
    cv2.putText(canvas, header, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(canvas, header, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
    return canvas


async def main():
    async with await connect() as machine:
        cam = Camera.from_robot(machine, "cam")
        detector = VisionClient.from_robot(machine, "color-detector")

        state = {'frame': None, 'dets': [], 'stop': False, 'show_all': False}

        pump = asyncio.create_task(_frame_pump(cam, state))
        detect = asyncio.create_task(_detect_pump(machine, cam, detector, state))

        print("waiting for first frame...")
        for _ in range(100):
            if state['frame'] is not None:
                break
            await asyncio.sleep(0.1)

        print("viewer running — the arm will NOT move. q to quit, a to show all labels.")
        try:
            while True:
                if state['frame'] is not None:
                    cv2.imshow(WINDOW, _draw(state['frame'], state))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('a'):
                    state['show_all'] = not state['show_all']
                await asyncio.sleep(0.01)
        finally:
            state['stop'] = True
            pump.cancel()
            detect.cancel()
            await asyncio.gather(pump, detect, return_exceptions=True)
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)


if __name__ == '__main__':
    asyncio.run(main())
