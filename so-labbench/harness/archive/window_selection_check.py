# Does the scorer look at the right closure when an episode contains several?
#   python window_selection_check.py [dataset]
#
# score_rq1.py takes the LONGEST run of closed-gripper frames as the grasp and measures
# the lift from its onset. Plotting the demos showed that assumption breaking: episodes
# frequently begin with the gripper already closed, carried over from the end of the
# previous recording, and that idle stretch is often the longest run in the episode.
# The scorer then measures a lift that never had anything to do with a grasp.
#
# Compares three ways of choosing the window on the demos, which unlike the pilot trials
# actually contain grasps:
#   longest    - what the scorer does today
#   best_lift  - among closure runs long enough to count, the one with the largest lift
#   last       - the final closure run, on the theory that the grasp ends the episode
#
# Reports only. The pilot cannot settle this - none of its ten trials closed the gripper
# at all - so this needs its own evidence before anyone edits the scorer.
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from labbench import (CLOSE_T, DEMO_ASSUMED_HZ, LIFT_GAIN, closure_runs, iter_episode_parquets,
                      lift_gain, stable_frames)

MIN_RUN = stable_frames(DEMO_ASSUMED_HZ)   # 0.5 s at the rate the demonstrations declare


def runs_of_closure(grip):
    closed = grip < CLOSE_T
    runs, start = [], None
    for i, c in enumerate(closed):
        if c and start is None:
            start = i
        elif not c and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(closed) - start))
    return [r for r in runs if r[1] >= MIN_RUN]


def gain_for(lift, onset, length):
    seg = lift[onset:min(onset + length, len(lift))]
    return float(seg.max() - lift[onset]) if len(seg) else 0.0


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "so101_pick_remote"
    files = [p for _, p in iter_episode_parquets(dataset)]
    if not files:
        raise SystemExit("no episodes")

    rows = []
    for f in files:
        ep = int(f.stem.split("_")[1])
        df = pd.read_parquet(f)
        st = np.array(df["observation.state"].tolist())
        lift, grip = st[:, 1], st[:, 5]
        runs = runs_of_closure(grip)
        if not runs:
            rows.append(dict(ep=ep, n_runs=0, longest=np.nan, best=np.nan, last=np.nan,
                             starts_closed=bool(grip[0] < CLOSE_T)))
            continue
        longest = max(runs, key=lambda r: r[1])
        best = max(runs, key=lambda r: gain_for(lift, *r))
        last = runs[-1]
        rows.append(dict(ep=ep, n_runs=len(runs),
                         longest=gain_for(lift, *longest),
                         best=gain_for(lift, *best),
                         last=gain_for(lift, *last),
                         starts_closed=bool(grip[0] < CLOSE_T)))

    d = pd.DataFrame(rows)
    got = d.dropna(subset=["longest"])
    print(f"{dataset}: {len(d)} demos, {len(got)} with a closure of at least {MIN_RUN} frames")
    print(f"  episodes that BEGIN already closed: {int(d.starts_closed.sum())}"
          "   <- carried over from the previous recording, not a grasp")
    print(f"  episodes with more than one closure: {int((got.n_runs > 1).sum())}")

    print(f"\n  demos whose lift gain clears {LIFT_GAIN:.0f}, by window choice:")
    for name in ("longest", "best", "last"):
        print(f"    {name:9} {int((got[name] >= LIFT_GAIN).sum()):>3}/{len(got)}")

    changed = got[(got["longest"] < LIFT_GAIN) & (got["best"] >= LIFT_GAIN)]
    print(f"\n  demos the current choice scores as no-lift but another window lifts: {len(changed)}")
    for _, r in changed.head(12).iterrows():
        print(f"    ep{int(r['ep']):<3} runs {int(r['n_runs'])}  longest gain {r['longest']:6.1f}"
              f"  best {r['best']:6.1f}  last {r['last']:6.1f}"
              f"{'   (starts closed)' if r['starts_closed'] else ''}")

    print("\n  lift gain of the best window, distribution over demos that closed:")
    q = [0, 10, 25, 50, 75, 90, 100]
    v = np.percentile(got["best"], q)
    print("   " + "  ".join(f"p{a}={b:.1f}" for a, b in zip(q, v)))
    print("\n  how many demos would clear each candidate LIFT_GAIN (best window):")
    for t in (2, 5, 10, 15, 20, 30):
        print(f"    >= {t:>2}   {int((got['best'] >= t).sum()):>3}/{len(got)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
