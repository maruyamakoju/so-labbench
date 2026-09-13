# Lock the fixed camera's exposure before a session.
#   python lock_exposure.py                 # report what the camera is doing now
#   python lock_exposure.py --measure       # sweep exposure values: fps and brightness
#   python lock_exposure.py --set -5        # apply and verify
#
# Why this exists. During the pilot the room got darker between trial 7 and trial 8.
# With auto exposure the C270 answered by lengthening exposure, and three things moved
# at once: frame brightness halved (60 -> 32), edge density halved (7.8% -> 3.4%, which
# broke camera alignment measurement outright), and the capture rate fell from 30 to 15,
# dragging the control loop from 23.8 Hz to 14.9 Hz.
#
# That matters most for condition C5's lighting change, which the protocol requires to
# move ONE variable. Auto exposure made it move three - and worse, it partly cancelled
# the manipulation itself by brightening the image back up. A locked exposure means a
# darker room produces a darker image and nothing else.
#
# The setting persists at the driver level across processes, so lerobot inherits it
# without any patch. Choose the value under BASELINE lighting, then leave it alone.
import sys
import time
import cv2
import numpy as np

from labbench import CAMERA_INDEX, FRAME_W, FRAME_H

# DirectShow encodes these as a flag, not a boolean.
AUTO, MANUAL = 0.75, 0.25
# Exposure is nominally log2 seconds, but measure rather than trust it: on this C270,
# -4 held 29.9 fps where the arithmetic (1/16 s) says it should not, and it passes about
# six times the light of -5. Take the longest value the sweep shows holding 30 fps.
SUSTAINS_30FPS = -4


def open_camera():
    cap = cv2.VideoCapture(int(CAMERA_INDEX), cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"camera {CAMERA_INDEX} unavailable")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def sample(cap, secs=3.0):
    for _ in range(15):
        cap.read()
    t0, n, bright, edges = time.perf_counter(), 0, [], []
    while time.perf_counter() - t0 < secs:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        if n % 10 == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            bright.append(float(g.mean()))
            edges.append(float((cv2.Canny(g, 60, 160) > 0).mean() * 100))
    dt = time.perf_counter() - t0
    return n / dt, float(np.mean(bright or [np.nan])), float(np.mean(edges or [np.nan]))


def report():
    cap = open_camera()
    fps, b, e = sample(cap)
    auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    cap.release()
    print(f"capture rate {fps:.1f} fps   mean brightness {b:.1f}   edge density {e:.2f}%")
    print(f"auto_exposure flag {auto}   exposure {exp:.0f}")
    if fps < 25:
        print("\nBelow 30 fps. The control loop cannot outrun the camera, so every")
        print("frame-counted threshold shifts with the room lights. Lock the exposure.")
    return 0


def measure():
    print("Run this under the lighting you intend as baseline.\n")
    print(f"{'exposure':>9} {'fps':>6} {'brightness':>11} {'edges %':>8}")
    cap = open_camera()
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO)
    fps, b, e = sample(cap)
    print(f"{'auto':>9} {fps:6.1f} {b:11.1f} {e:8.2f}")
    cap.release()

    for v in (-4, -5, -6, -7):
        cap = open_camera()
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, MANUAL)
        cap.set(cv2.CAP_PROP_EXPOSURE, v)
        fps, b, e = sample(cap)
        cap.release()
        flag = "" if fps >= 25 else "   <- too slow"
        print(f"{v:>9} {fps:6.1f} {b:11.1f} {e:8.2f}{flag}")

    print(f"\nPick the longest exposure that still holds 30 fps (usually {SUSTAINS_30FPS}) and whose")
    print("brightness is closest to the training data. Demo frames average about 60.")
    print("Then: python lock_exposure.py --set <value>")
    return 0


def set_exposure(v):
    cap = open_camera()
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, MANUAL)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(v))
    fps, b, e = sample(cap)
    got = cap.get(cv2.CAP_PROP_EXPOSURE)
    cap.release()
    print(f"exposure set to {got:.0f}   {fps:.1f} fps   brightness {b:.1f}   edges {e:.2f}%")
    if abs(got - v) > 0.5:
        print(f"WARNING: camera reports {got:.0f}, not {v}. It may not support that value.")
    if fps < 25:
        print("WARNING: still under 30 fps. Try a shorter exposure.")
    print("\nThe setting persists at the driver level, so lerobot will inherit it.")
    print("Record the value in the run notes - it is part of the measurement setup.")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(report())
    if a[0] == "--measure":
        sys.exit(measure())
    if a[0] == "--set":
        sys.exit(set_exposure(float(a[1])))
    raise SystemExit(__doc__ or "usage: lock_exposure.py [--measure | --set <value>]")
