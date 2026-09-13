# Build the camera reference framing from the training data the frozen policies actually saw.
#   python make_reference_frame.py [dataset_name] [episodes...]
# Writes a montage of first frames (to eyeball setup consistency) and the canonical
# reference image that align_camera.py aims at.
import sys
import cv2
import numpy as np
from pathlib import Path

from labbench import CAMERA_NAME, FRAME_H, FRAME_W, episode_video

HERE = Path(__file__).parent
OUT_DIR = HERE / "_frames"
W, H = FRAME_W, FRAME_H

# The training data was recorded from two different camera positions - episodes 0-29 from
# one and 30-59 from another, with no feature match between the groups. The baseline is the
# 30-59 group, because that is where the grasping demonstrations and the v2 audit were
# recorded. A default that sampled across both produced a median frame matching neither
# position, and every alignment residual was then measured against that blend.
DEFAULT_EPISODES = [30, 37, 45, 52, 59]


def first_frame(video: Path):
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    return cv2.resize(frame, (W, H)) if ok else None


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "so101_pick_remote"
    eps = [int(x) for x in sys.argv[2:]] or DEFAULT_EPISODES
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for e in eps:
        v = episode_video(dataset, e, camera=CAMERA_NAME)
        if not v.exists():
            print(f"skip ep{e}: no video")
            continue
        f = first_frame(v)
        if f is None:
            print(f"skip ep{e}: unreadable")
            continue
        frames.append((e, f))

    if not frames:
        print("no frames extracted")
        return 1

    cols = 3
    labelled = []
    for e, f in frames:
        f = f.copy()
        cv2.putText(f, f"ep{e}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(f, f"ep{e}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        labelled.append(f)
    rows = [labelled[i:i + cols] for i in range(0, len(labelled), cols)]
    grid = []
    for imgs in rows:
        imgs = list(imgs)
        while len(imgs) < cols:
            imgs.append(np.zeros((H, W, 3), np.uint8))
        grid.append(np.hstack(imgs))
    montage = np.vstack(grid)

    montage_path = OUT_DIR / "training_first_frames.png"
    cv2.imwrite(str(montage_path), montage)

    # canonical reference = median of the sampled first frames, which suppresses
    # per-episode object placement and leaves the fixed scene geometry
    stack = np.stack([f for _, f in frames]).astype(np.float32)
    ref = np.median(stack, axis=0).astype(np.uint8)
    ref_path = OUT_DIR / "camera_reference.png"
    cv2.imwrite(str(ref_path), ref)

    print(f"episodes used: {[e for e, _ in frames]}")
    print(f"montage   -> {montage_path}")
    print(f"reference -> {ref_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
