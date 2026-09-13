# Build a filmstrip per trial for independent human labelling.
#   python make_filmstrip.py <repo_prefix> <n_trials> [frames_per_trial]
#   e.g.  python make_filmstrip.py eval_rq1_pilot_act_ 5
# One row per trial, frames sampled evenly across the episode, so a reader can see
# the sequence a single final frame hides: did it descend, did it close, did it lift,
# did it drop it again.
#
# Deliberately carries NO scorer output - no gripper value, no stage flags, no verdict.
# The pilot asks whether a human and the scorer agree, which means nothing.
import os
import sys
import cv2
import numpy as np
from pathlib import Path

from labbench import CAMERA_NAME, episode_seconds, episode_video

HERE = Path(__file__).parent
TW, TH = 360, 270  # large enough to read whether the gripper actually closed
LABEL_W = 120


def demos(dataset: str, eps: list, k: int = 8):
    rows = []
    for e in eps:
        strip = strip_from(episode_video(dataset, e, camera=CAMERA_NAME), k)
        label = np.zeros((TH, LABEL_W, 3), np.uint8)
        if strip is None:
            strip = np.zeros((TH, TW * k, 3), np.uint8)
            cv2.putText(strip, "MISSING", (20, TH // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(label, f"ep{e}", (6, TH // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        rows.append(np.hstack([label, strip]))
        rows.append(np.full((3, LABEL_W + TW * k, 3), 60, np.uint8))
    out = HERE / "_frames" / f"filmstrip_{dataset}_demos.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows[:-1]))
    print(f"filmstrip -> {out}  ({len(eps)} episodes x {k} frames)")
    return 0


def strip_from(v: Path, k: int, episode_sec: float = None):
    """A row of k frames spanning the episode.

    The timestamp on each tile is wall clock, computed from the trial's recorded duration,
    not from the video's declared fps. The declared fps is a fiction here - the recording
    loop runs on wall clock and writes one frame per iteration, so a 45 s episode at a
    declared 30 fps holds about 1000 frames, not 1350. Labelling tiles with frame/30 would
    tell someone judging "did it hold long enough?" that a 1.5 s hold was a 1.0 s one.
    """
    if v is None or not v.exists():
        return None
    cap = cv2.VideoCapture(str(v))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    fps = (total / episode_sec) if episode_sec else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stamp = "s" if episode_sec else "s?"      # "?" marks a timestamp from the declared rate
    idx = np.linspace(0, total - 1, k).astype(int)
    tiles = []
    for j in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(j))
        ok, f = cap.read()
        if not ok:
            f = np.zeros((TH, TW, 3), np.uint8)
        f = cv2.resize(f, (TW, TH))
        t = f"{j / fps:4.1f}{stamp}"
        cv2.putText(f, t, (6, TH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(f, t, (6, TH - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(f)
    cap.release()
    return np.hstack(tiles)


def strip_for(run_id: str, k: int, episode_sec: float = None):
    try:
        video = episode_video(run_id, camera=CAMERA_NAME)
    except FileNotFoundError:
        return None
    return strip_from(video, k, episode_sec)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: make_filmstrip.py <run_prefix> <n_trials> [frames_per_row]\n"
                         "       make_filmstrip.py --demos <dataset> <episode>...")
    # --demos <dataset> <ep...> reads episodes inside one dataset instead of the
    # one-episode-per-dataset layout the eval runs use.
    if sys.argv[1] == "--demos":
        dataset = sys.argv[2]
        eps = [int(x) for x in sys.argv[3:]]
        return demos(dataset, eps)

    prefix, n = sys.argv[1], int(sys.argv[2])
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seconds = episode_seconds()

    rows = []
    for i in range(1, n + 1):
        run_id = f"{prefix}{i}"
        strip = strip_for(run_id, k, seconds.get(run_id))
        label = np.zeros((TH, LABEL_W, 3), np.uint8)
        if strip is None:
            strip = np.zeros((TH, TW * k, 3), np.uint8)
            cv2.putText(strip, "MISSING", (20, TH // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(label, f"#{i}", (10, TH // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        rows.append(np.hstack([label, strip]))
        sep = np.full((3, LABEL_W + TW * k, 3), 60, np.uint8)
        rows.append(sep)

    sheet = np.vstack(rows[:-1])
    out = HERE / f"{prefix}filmstrip.png"
    cv2.imwrite(str(out), sheet)
    print(f"filmstrip -> {out}  ({n} trials x {k} frames)")
    print("Label each row before looking at any scorer output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
