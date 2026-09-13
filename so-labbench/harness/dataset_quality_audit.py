# What does each demonstration actually contain?
#   python dataset_quality_audit.py [dataset]
# Writes <dataset>_quality.csv, one row per episode, and prints the breakdown.
#
# Exists because "the demos are successful pickups" turned out to be false for a fifth
# of them. Any subset design for RQ1-B needs to start from what is really in each
# episode rather than from that assumption.
#
# Verdicts:
#   complete_pickup   closure long enough to count, with the arm rising after it
#   closure_no_lift   gripper closed but the arm never rose - grasp attempted, not lifted
#   brief_closure     closure too short to be a grasp
#   no_closure        gripper never left the open range: the target behaviour is absent
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from labbench import (CLOSE_T, DEMO_ASSUMED_HZ, LIFT_GAIN, closure_runs, iter_episode_parquets,
                      lift_gain, stable_frames)

# The demonstrations carry no wall-clock duration, so "long enough to be a grasp" has to be
# converted with an assumed rate rather than a measured one. That assumption is stated once,
# in labbench.py, and it is the same 0.5 s the scorer applies to the trials this audit's
# output is used to evaluate. It used to be a bare 12 here against the scorer's 15.
MIN_RUN = stable_frames(DEMO_ASSUMED_HZ)


def audit(pq: Path):
    st = np.array(pd.read_parquet(pq)["observation.state"].tolist())
    lift, grip = st[:, 1], st[:, 5]
    all_runs = closure_runs(grip)
    runs = [r for r in all_runs if r[1] >= MIN_RUN]

    def gain(o, n):
        return lift_gain(lift, o, n)

    best_gain = max((gain(*r) for r in runs), default=0.0)
    if not all_runs:
        verdict = "no_closure"
    elif not runs:
        verdict = "brief_closure"
    elif best_gain >= LIFT_GAIN:
        verdict = "complete_pickup"
    else:
        verdict = "closure_no_lift"

    return dict(
        frames=len(st),
        grip_min=round(float(grip.min()), 1),
        n_closures=len(runs),
        starts_closed=bool(grip[0] < CLOSE_T),
        longest_closure=max((n for _, n in runs), default=0),
        best_lift_gain=round(best_gain, 1),
        descent=round(float(lift[0] - lift.min()), 1),
        verdict=verdict,
    )


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "so101_pick_remote"
    episodes = iter_episode_parquets(dataset)
    if not episodes:
        raise SystemExit(f"no episodes for {dataset}")

    rows = []
    for ep, f in episodes:
        r = audit(f)
        r["ep"] = ep
        rows.append(r)
    d = pd.DataFrame(rows)[["ep", "frames", "grip_min", "n_closures", "starts_closed",
                            "longest_closure", "best_lift_gain", "descent", "verdict"]]
    out = Path(__file__).parent / f"{dataset}_quality.csv"
    d.to_csv(out, index=False)

    print(f"{dataset}: {len(d)} episodes\n")
    for v, n in d.verdict.value_counts().items():
        eps = ", ".join(f"ep{e}" for e in d[d.verdict == v].ep.tolist())
        print(f"  {v:16} {n:>3}   {eps if len(eps) < 200 else eps[:197] + '...'}")

    print(f"\n  begin already closed (previous episode's state): {int(d.starts_closed.sum())}")
    for lo, hi, name in [(0, 29, "ep0-29  first session"), (30, 59, "ep30-59 grasp-focused")]:
        h = d[(d.ep >= lo) & (d.ep <= hi)]
        ok = int((h.verdict == "complete_pickup").sum())
        print(f"  {name}: {ok}/{len(h)} complete pickups")

    usable = d[d.verdict == "complete_pickup"].ep.tolist()
    print(f"\n  episodes demonstrating the full task: {len(usable)}/{len(d)}")
    print("  " + ", ".join(str(e) for e in usable))
    print(f"\ncsv -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
