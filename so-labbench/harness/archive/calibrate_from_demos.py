# Calibrate the scorer's position thresholds against the demonstrations themselves.
#   python calibrate_from_demos.py [dataset] [close_t]
#
# The training demos are human teleoperation of the task succeeding, so they are the
# one set of trials whose label is known without anyone re-watching them. Whatever a
# real grasp looks like in this rig's signals, it looks like it here.
#
# Only the POSITION thresholds can be calibrated this way - CLOSE_T, LIFT_GAIN,
# DESCEND_D, MOVE_D are joint values or path lengths and do not depend on the control
# rate. The duration thresholds cannot: demo episodes end when the operator finished
# the task, not on a timer, so frames cannot be converted to seconds without knowing a
# rate that was never recorded. Those two numbers are a protocol decision (0.5s / 1.0s)
# rather than something to fit.
#
# Reports only. Nothing here edits the scorer.
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from labbench import DATASETS as BASE

CLOSE_T = 40.0
LIFT_GAIN = 10.0
DESCEND_D = 15.0
MOVE_D = 25.0


def episode_signals(pq: Path):
    df = pd.read_parquet(pq)
    st = np.array(df["observation.state"].tolist())
    lift, grip = st[:, 1], st[:, 5]

    closed = grip < CLOSE_T
    best = onset = cur = 0
    for i, c in enumerate(closed):
        cur = cur + 1 if c else 0
        if cur > best:
            best, onset = cur, i - cur + 1

    gain = np.nan
    if best > 0:
        seg_end = min(onset + best, len(lift))
        gain = float(lift[onset:seg_end].max() - lift[onset])

    return dict(
        frames=len(df),
        grip_min=float(grip.min()),
        closure_frames=int(best),
        closure_frac=round(best / len(df), 3),
        lift_gain=gain,
        descend=float(lift[0] - lift.min()),
        lift_min=float(lift.min()),
        moved=float(np.abs(np.diff(st[:, :5], axis=0)).sum()),
    )


def pct(v, q):
    v = [x for x in v if not np.isnan(x)]
    return np.percentile(v, q) if v else float("nan")


def describe(name, rows):
    if not rows:
        print(f"\n{name}: no episodes")
        return
    print(f"\n{name}  (n={len(rows)})")
    for key, label in [("grip_min", "gripper minimum"), ("closure_frames", "longest closure (frames)"),
                       ("lift_gain", "lift gain during closure"), ("descend", "descent depth"),
                       ("moved", "total joint travel")]:
        v = [r[key] for r in rows]
        print(f"  {label:28} min {pct(v,0):8.1f}  p10 {pct(v,10):8.1f}  median {pct(v,50):8.1f}"
              f"  p90 {pct(v,90):8.1f}  max {pct(v,100):8.1f}")

    closed_any = sum(1 for r in rows if r["closure_frames"] > 0)
    lifted = sum(1 for r in rows if not np.isnan(r["lift_gain"]) and r["lift_gain"] >= LIFT_GAIN)
    descended = sum(1 for r in rows if r["descend"] >= DESCEND_D or r["lift_min"] < -30)
    movedok = sum(1 for r in rows if r["moved"] >= MOVE_D)
    n = len(rows)
    print(f"  would pass, with the scorer's current position thresholds:")
    print(f"    any closure at CLOSE_T={CLOSE_T:.0f}      {closed_any}/{n}")
    print(f"    descent at DESCEND_D={DESCEND_D:.0f}       {descended}/{n}")
    print(f"    lift gain at LIFT_GAIN={LIFT_GAIN:.0f}      {lifted}/{n}")
    print(f"    movement at MOVE_D={MOVE_D:.0f}          {movedok}/{n}")


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "so101_pick_remote"
    ddir = BASE / dataset / "data" / "chunk-000"
    files = sorted(ddir.glob("episode_*.parquet"))
    if not files:
        raise SystemExit(f"no episodes under {ddir}")

    rows = []
    for f in files:
        ep = int(f.stem.split("_")[1])
        r = episode_signals(f)
        r["ep"] = ep
        rows.append(r)

    early = [r for r in rows if r["ep"] < 30]
    late = [r for r in rows if r["ep"] >= 30]

    print(f"{dataset}: {len(rows)} demonstrations")
    describe("ep0-29  (first session)", early)
    describe("ep30-59 (grasp-focused session)", late)

    print("\nepisodes with no closure at all:")
    none_closed = [r["ep"] for r in rows if r["closure_frames"] == 0]
    print("  " + (", ".join(f"ep{e}" for e in none_closed) if none_closed else "none"))

    print("\nlowest closure fractions (closure frames / episode frames):")
    for r in sorted(rows, key=lambda r: r["closure_frac"])[:8]:
        print(f"  ep{r['ep']:<3} {r['closure_frac']:.3f}  ({r['closure_frames']} of {r['frames']} frames,"
              f" grip_min {r['grip_min']:.0f}, lift gain {r['lift_gain']:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
