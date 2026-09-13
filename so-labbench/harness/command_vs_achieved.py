# Did the policy ask for it, or did the robot fail to do it?
#   python command_vs_achieved.py <repo_prefix> <n>     # or no args for the manifest
#
# The failure taxonomy attributes everything to the policy, but a gripper that never
# closes has two very different causes: the policy never commanded a closure, or it did
# and the hardware did not follow. Those need different fixes and the scorer cannot tell
# them apart, because it only reads observation.state.
#
# The datasets carry the commanded action alongside the achieved state, so the question
# is answerable for free on every trial already recorded.
#
# RQ0 primary outcome (rq0_data_quality_protocol.md): a trial "commanded a closure" when
# the commanded gripper value stays below CLOSE_T for at least CLOSE_S seconds. Seconds,
# not frames: the control rate is measured per trial (frames / episode_sec, from the
# manifest) exactly as score_rq1.py v1.3 does. Rows also go to <out>.csv for the memo.
import sys
import csv
import numpy as np
import pandas as pd
from pathlib import Path

from labbench import (CLOSE_T, STABLE_S, episode_parquet, episode_seconds, read_manifest,
                      study_output, true_hz)

HERE = Path(__file__).parent
CLOSE_S = STABLE_S     # the same decision as the scorer's stable closure, not a second one


def longest_run(mask) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def runs(prefix=None, n=None, study=None, condition=None):
    if prefix:
        return [(f"{prefix}{i}", "", "") for i in range(1, n + 1)]
    rows = read_manifest(study=study, condition=condition)
    if not rows:
        raise SystemExit("no matching trials in the manifest, and no prefix given")
    return [(r["run_id"], r.get("model", ""), r.get("condition", "")) for r in rows]


def main():
    args = sys.argv[1:]
    study = args[args.index("--study") + 1] if "--study" in args else None
    condition = args[args.index("--condition") + 1] if "--condition" in args else None
    positional = [a for i, a in enumerate(args)
                  if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    if len(positional) >= 2:
        items = runs(positional[0], int(positional[1]))
    else:
        items = runs(study=study, condition=condition)
    out_csv = study_output("command_vs_achieved.csv", study, condition)
    ep_secs = episode_seconds()
    episode_sec = float(positional[2]) if len(positional) >= 3 else None

    print(f"{'run_id':32} {'model':9} {'cmd min':>8} {'obs min':>8} {'gap':>6} {'cmd closed':>11} {'Hz':>5} {'cmd_s':>6} {'>=0.5s':>6}  verdict")
    counts = {}
    rows = []
    for run_id, model, _ in items:
        try:
            pq = episode_parquet(run_id)
        except FileNotFoundError:
            continue
        if not pq.exists():
            continue
        df = pd.read_parquet(pq)
        if len(df) == 0:
            continue
        cmd = np.array(df["action"].tolist())[:, 5]
        obs = np.array(df["observation.state"].tolist())[:, 5]
        cmd_min, obs_min = float(cmd.min()), float(obs.min())
        cmd_closed = int((cmd < CLOSE_T).sum())
        obs_closed = int((obs < CLOSE_T).sum())
        # How far the servo sat from its target while the target was lowest. A policy that
        # never moves the gripper has no frames below its own 10th percentile, and the median
        # of nothing is nan - which is the population this tool exists to study, so the
        # degenerate case is the common one rather than the rare one.
        lowest = cmd < np.percentile(cmd, 10)
        gap = float(np.median(obs[lowest] - cmd[lowest])) if lowest.any() else 0.0

        if cmd_closed == 0:
            verdict = "policy never commanded a closure"
        elif obs_closed == 0:
            verdict = "COMMANDED BUT NOT EXECUTED - check the hardware"
        elif obs_closed < cmd_closed * 0.5:
            verdict = "commanded, only partly executed"
        else:
            verdict = "commanded and executed"
        counts[verdict] = counts.get(verdict, 0) + 1
        # seconds-based primary outcome; unknown rate when the trial is in no manifest and no length was given
        sec = ep_secs.get(run_id, episode_sec)
        hz = true_hz(len(df), sec) if sec else float("nan")
        cmd_run_s = longest_run(cmd < CLOSE_T) / hz if sec else float("nan")
        commanded = (cmd_run_s >= CLOSE_S) if sec else None
        flag = "?" if commanded is None else ("yes" if commanded else "no")
        print(f"{run_id:32} {model:9} {cmd_min:8.1f} {obs_min:8.1f} {gap:6.1f} {cmd_closed:11d} {hz:5.1f} {cmd_run_s:6.2f} {flag:>6}  {verdict}")
        rows.append(dict(run_id=run_id, model=model, cmd_min=round(cmd_min, 1), obs_min=round(obs_min, 1),
                         tracking_gap=round(gap, 1), cmd_closed_frames=cmd_closed, obs_closed_frames=obs_closed,
                         true_hz=round(hz, 1), cmd_closed_longest_s=round(cmd_run_s, 2),
                         commanded_closure=flag, verdict=verdict))

    print()
    for v, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}  {v}")
    known = [r for r in rows if r["commanded_closure"] != "?"]
    if known:
        yes = sum(r["commanded_closure"] == "yes" for r in known)
        print(f"\n  commanded closure >= {CLOSE_S}s: {yes}/{len(known)} trials")
    if any(r["commanded_closure"] == "?" for r in rows):
        print("  (? = no episode_sec known for that trial; pass it as the 3rd argument)")
    if rows:
        with out_csv.open("w", newline="", encoding="ascii") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  wrote {out_csv.name}")
    if counts.get("policy never commanded a closure"):
        print("\nWhere the policy never commanded a closure, the arm and the control path are\n"
              "not implicated at all. Nothing about the gripper, the servos or the wiring can\n"
              "explain those trials - the output simply never asked for it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
