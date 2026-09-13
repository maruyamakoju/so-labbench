# Where was the object, for each demonstration?
#   python demo_position_labels.py [dataset] [k]
# Writes <dataset>_positions.csv and prints the spread.
#
# RQ1-B needs positional labels the dataset never recorded, and the protocol assumed
# they would come from watching sixty videos. They do not have to. The arm's own joint
# angles at the moment it closes on the object encode where the object was, in the
# robot's frame - which is better than vision here, because the two recording sessions
# used different camera poses and pixel coordinates are not comparable across them.
#
# Only episodes that actually closed on something get a label. The rest have no grasp
# pose to read.
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from labbench import CLOSE_T, DEMO_ASSUMED_HZ, closure_runs, iter_episode_parquets, stable_frames

MIN_RUN = stable_frames(DEMO_ASSUMED_HZ)   # 0.5 s at the rate the demonstrations declare
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex"]


def grasp_pose(pq: Path):
    st = np.array(pd.read_parquet(pq)["observation.state"].tolist())
    grip = st[:, 5]
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
    runs = [r for r in runs if r[1] >= MIN_RUN and r[0] > 0]  # onset at frame 0 = carried over
    if not runs:
        return None
    lift = st[:, 1]
    onset, _ = max(runs, key=lambda r: (lift[r[0]:min(r[0] + r[1], len(lift))].max() - lift[r[0]]))
    return st[onset, :3]


def kmeans(x, k, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None, :, :]) ** 2).sum(-1), axis=1)
        new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return lab, c


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "so101_pick_remote"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    files = [p for _, p in iter_episode_parquets(dataset)]

    rows = []
    for f in files:
        ep = int(f.stem.split("_")[1])
        p = grasp_pose(f)
        rows.append(dict(ep=ep, labelled=p is not None,
                         **{n: (round(float(p[i]), 1) if p is not None else np.nan)
                            for i, n in enumerate(JOINTS)}))
    d = pd.DataFrame(rows)
    got = d[d.labelled].copy()
    print(f"{dataset}: {len(got)}/{len(d)} episodes have a readable grasp pose")
    if len(got) < k:
        raise SystemExit("too few labelled episodes to cluster")

    x = got[JOINTS].to_numpy()
    lab, cent = kmeans(x, k)
    got["position"] = [f"P{i+1}" for i in lab]
    d = d.merge(got[["ep", "position"]], on="ep", how="left")

    print("\nspread of the grasp pose (degrees of the normalised joint range):")
    for i, n in enumerate(JOINTS):
        v = x[:, i]
        print(f"  {n:14} min {v.min():7.1f}  max {v.max():7.1f}  range {np.ptp(v):6.1f}  sd {v.std():5.1f}")

    print(f"\n{k} clusters:")
    for j in range(k):
        m = lab == j
        eps = got.ep.to_numpy()[m]
        print(f"  P{j+1}  n={m.sum():>3}  centre "
              + " ".join(f"{n}={cent[j][i]:7.1f}" for i, n in enumerate(JOINTS)))
        print(f"       eps: {', '.join(str(e) for e in eps)}")

    for lo, hi, name in [(0, 29, "ep0-29 "), (30, 59, "ep30-59")]:
        h = got[(got.ep >= lo) & (got.ep <= hi)]
        if len(h):
            hx = h[JOINTS].to_numpy()
            spread = float(np.linalg.norm(hx.std(0)))
            print(f"\n  {name} labelled {len(h):>2}  positional spread (sd norm) {spread:5.1f}"
                  f"  clusters used {h.position.nunique()}")

    out = Path(__file__).parent / f"{dataset}_positions.csv"
    d.to_csv(out, index=False)
    print(f"\ncsv -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
