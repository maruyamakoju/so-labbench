# SO-LabBench: stage scoring + failure taxonomy for evaluation trials
# Usage: python score_trials.py <repo_prefix> <n_trials>
#   e.g.  python score_trials.py eval_c0_ 30
# Reads  ~/.cache/huggingface/lerobot/maruo/<repo_prefix><i>/  (single-episode eval datasets)
# Writes <repo_prefix>scores.csv next to this script + prints summary.
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
CFG = json.loads((HERE / "labbench_config.json").read_text(encoding="utf-8"))
BASE = Path(CFG["paths"]["hf_cache"])
S = CFG["scoring"]

def score_episode(parquet: Path) -> dict:
    df = pd.read_parquet(parquet)
    st = np.array(df["observation.state"].tolist())   # pan, lift, elbow, wrist_f, wrist_r, grip
    act = np.array(df["action"].tolist())
    pan, lift, grip = st[:, 0], st[:, 1], st[:, 5]

    descend = bool(lift.min() < S["descend_lift_threshold"])
    closed_mask = grip < S["grip_close_threshold"]
    # stable closure: >= grip_stable_frames consecutive closed frames
    run = best = onset = 0
    cur = 0
    for i, c in enumerate(closed_mask):
        cur = cur + 1 if c else 0
        if cur > best:
            best = cur
            onset = i - cur + 1
    grasp = bool(best >= S["grip_stable_frames"])
    # transfer: pan moved by >= delta while closed
    transfer = False
    release_after_transfer = False
    if grasp:
        seg = slice(onset, onset + best)
        transfer = bool(abs(pan[seg].max() - pan[seg].min()) >= S["transfer_pan_delta_min"])
        end = onset + best
        release_after_transfer = bool(transfer and end < len(grip) - 5 and grip[end:].max() > 60)
    return dict(
        frames=len(grip), grip_min=float(grip.min()), grip_cmd_min=float(act[:, 5].min()),
        lift_min=float(lift.min()),
        descend=descend, grasp=grasp, hold_frames=int(best),
        transfer=transfer, release_after_transfer=release_after_transfer,
    )

def taxonomy(r: dict) -> str:
    if not r["descend"] and not r["grasp"]:
        return "no_attempt_or_hover"
    if r["descend"] and not r["grasp"]:
        return "grasp_fail"
    if r["grasp"] and not r["transfer"]:
        return "no_transfer_after_grasp"
    if r["transfer"] and not r["release_after_transfer"]:
        return "no_release"
    return "full_sequence"     # placement correctness still needs video/visual check

def main():
    prefix, n = sys.argv[1], int(sys.argv[2])
    rows = []
    for i in range(1, n + 1):
        p = BASE / f"{prefix}{i}" / "data" / "chunk-000" / "episode_000000.parquet"
        if not p.exists():
            rows.append(dict(trial=i, status="missing"))
            continue
        r = score_episode(p)
        r["trial"] = i
        r["status"] = "ok"
        r["taxonomy"] = taxonomy(r)
        rows.append(r)
    d = pd.DataFrame(rows)
    out = HERE / f"{prefix}scores.csv"
    d.to_csv(out, index=False)
    ok = d[d.status == "ok"]
    print(f"scored {len(ok)}/{n} trials -> {out}")
    if len(ok):
        for stage in ["descend", "grasp", "transfer", "release_after_transfer"]:
            print(f"  {stage:>24}: {int(ok[stage].sum())}/{len(ok)} ({100*ok[stage].mean():.0f}%)")
        print("  taxonomy:", dict(ok.taxonomy.value_counts()))

if __name__ == "__main__":
    main()
