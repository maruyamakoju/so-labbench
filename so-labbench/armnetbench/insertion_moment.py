# What does the last push look like, from the gripper?
#
#   python insertion_moment.py cable_clip
#
# The trajectory analysis says every policy on cable_clip reaches 58-91% of the way through
# the demonstrated sequence and then retreats. Joint angles cannot say what stopped it:
# they record neither millimetres nor contact, and pushing a DisplayPort connector into a
# holder is made of both.
#
# The wrist camera can. This finds, for each rollout, the frame where it got deepest into the
# demonstrated sequence - its own best attempt - and puts that frame beside a demonstration
# frame from the same stage of the task. Side by side, "stopped a few millimetres short" and
# "arrived at the wrong angle" and "was never near the holder" look different.
#
# Bounded on purpose: a handful of episodes per policy, and only the wrist camera, because
# the point is to look at them rather than to compute a statistic.
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATASET = "armnet/armnetbench_v01_lerobot_so101"
TASKS = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))["tasks"]
CAMERA = "videos/observation.images.wrist"
POLICIES = ["act", "diffusion", "smolvla", "pi0", "pi0.5", "grootn1.7", "molmoact2"]
PER_POLICY = 2
DEMOS = 4
TILE_W = 420


def main():
    import cv2
    from huggingface_hub import hf_hub_download

    task_key = sys.argv[1] if len(sys.argv) > 1 else "cable_clip"
    instruction = TASKS[task_key]["instruction"]
    meta = pd.read_parquet(hf_hub_download(DATASET, "meta/episodes/chunk-000/file-000.parquet",
                                           repo_type="dataset"))
    meta["task"] = meta["tasks"].apply(
        lambda x: x[0] if hasattr(x, "__len__") and not isinstance(x, str) else str(x))
    all_eps = meta[meta.task == instruction]

    # Reuse the trajectory analysis: it already knows, per rollout, how deep it got.
    anatomy = pd.read_csv(HERE / f"failure_anatomy_{task_key}.csv").set_index("episode")

    chosen = list(all_eps[all_eps.policy_type == "teleoperated"].head(DEMOS).episode_index)
    for policy in POLICIES:
        pool = all_eps[all_eps.policy_type == policy]
        # the two rollouts that got deepest, so each policy is shown at its best
        ranked = sorted((e for e in pool.episode_index if e in anatomy.index),
                        key=lambda e: -anatomy.loc[e, "reached_phase"])
        chosen += ranked[:PER_POLICY]
    sample = all_eps[all_eps.episode_index.isin(chosen)]
    print(f"{task_key}: {len(sample)} episodes, wrist camera only")

    tiles = []
    for (chunk, file_index), group in sample.groupby([f"{CAMERA}/chunk_index", f"{CAMERA}/file_index"]):
        path = f"{CAMERA}/chunk-{int(chunk):03d}/file-{int(file_index):03d}.mp4"
        local = hf_hub_download(DATASET, path, repo_type="dataset")
        cap = cv2.VideoCapture(local)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        for _, ep in group.iterrows():
            start = ep[f"{CAMERA}/from_timestamp"]
            length = int(ep.length)
            if ep.policy_type == "teleoperated":
                # the demonstration's own deepest point is the end of the task, just before
                # the arm withdraws; 85% is inside the push and clear of the retreat
                at = 0.85
                label = "DEMO (human)"
            else:
                at = float(anatomy.loc[int(ep.episode_index), "reached_phase"])
                # reached_phase is a position in the DEMONSTRATION, not in this rollout. Find
                # the rollout frame that matched it: its own deepest attempt.
                at = float(anatomy.loc[int(ep.episode_index), "deepest_at"]) \
                    if "deepest_at" in anatomy.columns else 0.5
                label = f"{ep.policy_type} ({ep.success_class})"
            frame_index = int(round((start + at * length / fps) * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                print(f"  ep{int(ep.episode_index)}: no frame at {at:.2f}")
                continue
            tile = cv2.resize(frame, (TILE_W, int(TILE_W * frame.shape[0] / frame.shape[1])))
            cv2.rectangle(tile, (0, 0), (TILE_W, 28), (0, 0, 0), -1)
            cv2.putText(tile, f"{label}  ep{int(ep.episode_index)}", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            tiles.append((ep.policy_type, tile))
        cap.release()

    if not tiles:
        raise SystemExit("no frames extracted")
    order = ["teleoperated"] + POLICIES
    tiles.sort(key=lambda t: (order.index(t[0]) if t[0] in order else 99))
    images = [t for _, t in tiles]
    h = min(i.shape[0] for i in images)
    per_row = 4
    rows = []
    for i in range(0, len(images), per_row):
        row = [im[:h] for im in images[i:i + per_row]]
        while len(row) < per_row:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    out = HERE / "frames" / f"insertion_moment_{task_key}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out}")
    print("\nThe demonstration tiles show what the last push looks like when it works.")
    print("Each policy tile is that rollout's own deepest attempt. Compare the cable's")
    print("position and angle against the holder, not the arm's pose.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
