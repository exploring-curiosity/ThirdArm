"""Show every connected camera side by side, labelled by index.

No detection, no guessing which is which — look at the windows and tell the
code. Once you know, pin the overhead camera with:

    python cams.py --set 0      # remember index 0 as the overhead camera

which writes overhead_device.json, the file the rest of the code reads.

    python cams.py              # all cameras side by side
    python cams.py --set N      # remember index N as the overhead camera
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DEVICE_FILE = Path(__file__).with_name("overhead_device.json")

PANEL_W, PANEL_H = 480, 270
MAX_INDEX = 4


def open_all(max_index=MAX_INDEX):
    """Every camera index that opens and returns a frame."""
    caps = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            continue
        ok, _ = cap.read()
        if ok:
            caps.append((i, cap))
        else:
            cap.release()
    return caps


def show_all():
    caps = open_all()
    if not caps:
        print("no cameras opened")
        return

    saved = None
    if DEVICE_FILE.exists():
        try:
            saved = json.loads(DEVICE_FILE.read_text()).get("index")
        except Exception:                             # noqa: BLE001
            pass

    print(f"showing {len(caps)} camera(s): "
          + ", ".join(f"index {i}" for i, _ in caps))
    if saved is not None:
        print(f"currently remembered as overhead: index {saved}")
    print("press q to quit")

    try:
        while True:
            panels = []
            for i, cap in caps:
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
                panel = cv2.resize(frame, (PANEL_W, PANEL_H))
                tag = f"index {i}"
                if i == saved:
                    tag += "  (overhead)"
                # Dark strip behind the label so it stays readable over a
                # bright table or a dark room alike.
                cv2.rectangle(panel, (0, 0), (PANEL_W, 30), (0, 0, 0), -1)
                cv2.putText(panel, tag, (10, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                panels.append(panel)

            cv2.imshow("cameras", np.hstack(panels))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for _, cap in caps:
            cap.release()
        cv2.destroyAllWindows()


def set_index(index):
    DEVICE_FILE.write_text(json.dumps({"index": index}, indent=2))
    print(f"overhead camera set to index {index} "
          f"({DEVICE_FILE.name})")


if __name__ == "__main__":
    if "--set" in sys.argv:
        set_index(int(sys.argv[sys.argv.index("--set") + 1]))
    else:
        show_all()
