# RQ0 exploratory metric: does each arm PREDICT a closure when shown the demonstrations?
#   python offline_closure.py [stride]          # default stride 5 frames
#
# Re-runs audit D (audit_report.md) for the three RQ0 arms. Every frame of every demo is
# fed with its true state and image, and the first gripper value of the predicted chunk is
# compared with the demonstrated one. Two numbers per (arm, episode):
#   closure_recall   - of frames where the demo gripper is closed (<40), how many the policy
#                      also closes. Audit D found this near-perfect for the 60-demo models,
#                      which is why the rollout failure is not "did not learn to close".
#   false_closure    - of frames where the demo gripper is open, how many the policy closes.
#                      On the 13 demos that never close, this is the interesting one: a policy
#                      trained only on complete pick-ups has never seen "approach and hover",
#                      so if H-data is right it should try to close where 60all hovers.
# Episodes are tagged in_train per arm (rq0_subsets.json), so held-out and seen are separable.
# Exploratory by pre-registration; it does not decide anything.
import csv
import json
import sys
import torch
import numpy as np
from pathlib import Path

from labbench import CAMERA_NAME, CLOSE_T, DATASET_NAMESPACE, policy_path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy

HERE = Path(__file__).parent
ARMS = ["60all", "23complete", "23random"]
SOURCE = "so101_pick_remote"
OUT = HERE / "rq0_offline_closure.csv"


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    subsets = json.loads((HERE / "rq0_subsets.json").read_text(encoding="ascii"))["arms"]
    verdict = {int(r["ep"]): r["verdict"] for r in csv.DictReader((HERE / "so101_pick_remote_quality.csv").open())}
    ds = LeRobotDataset(f"{DATASET_NAMESPACE}/{SOURCE}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for arm in ARMS:
        path = policy_path(arm)
        policy = ACTPolicy.from_pretrained(path).to(dev).eval()
        in_train = set(subsets[arm]["episodes"])
        for ep in range(ds.num_episodes):
            a, b = int(ds.episode_data_index["from"][ep]), int(ds.episode_data_index["to"][ep])
            pred, tgt = [], []
            with torch.no_grad():
                for i in range(a, b, stride):
                    it = ds[i]
                    batch = {"observation.state": it["observation.state"][None].to(dev),
                             f"observation.images.{CAMERA_NAME}": it[f"observation.images.{CAMERA_NAME}"][None].to(dev)}
                    chunk = policy.predict_action_chunk(batch)          # (1, chunk, 6)
                    pred.append(float(chunk[0, 0, 5]))
                    tgt.append(float(it["action"][5]))
            pred, tgt = np.array(pred), np.array(tgt)
            closed = tgt < CLOSE_T
            recall = float((pred[closed] < CLOSE_T).mean()) if closed.any() else float("nan")
            false_c = float((pred[~closed] < CLOSE_T).mean()) if (~closed).any() else float("nan")
            rows.append(dict(arm=arm, ep=ep, verdict=verdict[ep], in_train=ep in in_train,
                             frames=len(pred), target_closed_frac=round(float(closed.mean()), 3),
                             closure_recall=round(recall, 3), false_closure=round(false_c, 3),
                             pred_min=round(float(pred.min()), 1), target_min=round(float(tgt.min()), 1)))
            print(f"{arm:11} ep{ep:02d} {verdict[ep]:16} {'train' if ep in in_train else 'held ':5} "
                  f"recall={recall:5.2f} false={false_c:5.2f} pred_min={pred.min():5.1f}", flush=True)
        del policy
        torch.cuda.empty_cache()

    with OUT.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary: by arm x verdict, mean recall (where defined) and mean false closure
    print(f"\n{'arm':11} {'verdict':16} {'n':>3} {'held':>4} {'recall':>7} {'false_closure':>13} {'pred_min':>8}")
    for arm in ARMS:
        for v in ["complete_pickup", "closure_no_lift", "brief_closure", "no_closure"]:
            g = [r for r in rows if r["arm"] == arm and r["verdict"] == v]
            if not g:
                continue
            rec = [r["closure_recall"] for r in g if r["closure_recall"] == r["closure_recall"]]
            fc = [r["false_closure"] for r in g]
            held = sum(not r["in_train"] for r in g)
            print(f"{arm:11} {v:16} {len(g):3d} {held:4d} {np.mean(rec) if rec else float('nan'):7.2f} "
                  f"{np.mean(fc):13.2f} {np.mean([r['pred_min'] for r in g]):8.1f}")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
