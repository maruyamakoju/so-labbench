# Camera alignment helper: live view blended with the reference framing.
#   python align_camera.py watch [reference.png]   -> keep writing a comparison image for the viewer
#   python align_camera.py snap  [reference.png]   -> write one comparison image and exit
# Aim the fixed camera until the live image lines up with the reference.
# Used before every session and to restore the reference view after condition C5 (camera shift).
# The lerobot env ships headless opencv, so the preview is a file the viewer window reloads.
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

from camera_pose_check import advise, decompose, estimate as pose_estimate
from camera_pose_check import set_match_size as pose_set_match_size

HERE = Path(__file__).parent
OUT_DIR = HERE / "_frames"
# Canonical baseline pose, built by make_reference_frame.py from the training data.
# camera_pose_check.py scores against the same file.
DEFAULT_REF = OUT_DIR / "camera_reference.png"
LIVE = Path(os.environ.get("ALIGN_LIVE", OUT_DIR / "camera_live.png"))
LIVE_RAW = LIVE.with_name("camera_live_raw.png")  # unannotated frame for camera_pose_check.py
STOP = LIVE.with_name("align_stop.flag")

W, H = 640, 480   # H is re-derived from the reference image's aspect ratio in main()
CAM_INDEX = int(os.environ.get("ALIGN_CAM_INDEX", 0))   # which camera to aim (front/top/wrist)


def draw_verdict(view, status):
    """status: None while measuring, else (ok, lines, inliers, samples)."""
    if status is None:
        return
    ok, lines, n_inl, n_samp = status
    colour = (80, 220, 80) if ok else (60, 210, 255)
    if n_samp < 3:
        colour = (200, 200, 200)
    head = ("ALIGNED - close this window" if ok
            else f"NOT ALIGNED  (match {n_inl}, median of {n_samp})")
    rows = [head] + ([] if ok else lines)
    pad, lh = 10, 30
    h = pad * 2 + lh * len(rows)
    panel = view[0:h, 0:520].copy()
    view[0:h, 0:520] = cv2.addWeighted(panel, 0.25, np.zeros_like(panel), 0.75, 0)
    for i, text in enumerate(rows):
        y = pad + lh * i + 22
        cv2.putText(view, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72 if i == 0 else 0.62, colour, 2)


def compose(frame, ref, ref_edges):
    blend = cv2.addWeighted(frame, 0.5, ref, 0.5, 0)
    edges = frame.copy()
    edges[ref_edges > 0] = (0, 255, 0)
    top = np.hstack([ref, frame])
    bottom = np.hstack([blend, edges])
    view = np.vstack([top, bottom])
    for text, org in [("REFERENCE", (8, 22)), ("LIVE", (W + 8, 22)),
                      ("BLEND", (8, H + 22)), ("REFERENCE EDGES ON LIVE", (W + 8, H + 22))]:
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return view


def write_atomic(path: Path, img):
    # The viewer reloads this file constantly; on Windows an open handle makes replace fail.
    # Dropping the occasional frame is fine at preview rates.
    tmp = path.with_suffix(".tmp.png")
    cv2.imwrite(str(tmp), img)
    for _ in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02)
    tmp.unlink(missing_ok=True)


def main():
    argv = sys.argv[1:]
    mode = argv[0] if argv and argv[0] in ("watch", "snap") else "watch"
    rest = [a for a in argv if a not in ("watch", "snap")]
    ref_path = Path(rest[0]) if rest else DEFAULT_REF

    global H
    try:
        H = pose_set_match_size(ref_path)     # the pose checker and this view agree on the size
    except SystemExit:
        print(f"reference image not found: {ref_path}")
        return 1
    ref = cv2.resize(cv2.imread(str(ref_path)), (W, H))
    ref_edges = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 60, 160)

    OUT_DIR.mkdir(exist_ok=True)
    STOP.unlink(missing_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"camera {CAM_INDEX} unavailable (set ALIGN_CAM_INDEX)")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    print(f"camera {CAM_INDEX} vs {ref_path.name} at {W}x{H}; live preview -> {LIVE}")
    if mode == "watch":
        print(f"stop by creating {STOP.name} in the same folder (or stopping this process)")

    frame = None
    status = None
    history = []
    i = 0
    deadline = time.monotonic() + 1800  # never run past 30 minutes unattended
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame read failed")
            break
        frame = cv2.resize(frame, (W, H))

        # Feature matching costs ~100ms, so refresh the verdict every few frames.
        # A single estimate jitters by tens of pixels at this inlier count, which is
        # unusable for aiming, so the advice comes off the median of a short history.
        if i % 4 == 0:
            M, _, n_inl = pose_estimate(ref, frame)
            if M is not None and n_inl >= 12:
                history.append((*decompose(M), n_inl))
                if len(history) > 5:
                    history.pop(0)
            elif not history:
                status = None
            if history:
                med = [float(np.median([h[j] for h in history])) for j in range(5)]
                status = (*advise(*med[:4]), int(med[4]), len(history))
        i += 1

        view = compose(frame, ref, ref_edges)
        draw_verdict(view, status)
        write_atomic(LIVE_RAW, frame)
        write_atomic(LIVE, view)
        if mode == "snap" or STOP.exists() or time.monotonic() > deadline:
            break
        time.sleep(0.15)

    cap.release()
    if frame is not None:
        cv2.imwrite(str(OUT_DIR / "camera_aligned.png"), frame)
        cv2.imwrite(str(OUT_DIR / "camera_aligned_vs_reference.png"), np.hstack([ref, frame]))
        print(f"saved -> {OUT_DIR / 'camera_aligned.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
