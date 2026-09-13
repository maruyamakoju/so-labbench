# Plot gripper and lift traces so the shape of an episode is visible rather than inferred.
#   python plot_traces.py <dataset> <ep|trial ...>            # training demos
#   python plot_traces.py --runs eval_rq1_pilot_act_1 ...     # eval datasets
# Written because the summary statistics disagreed with what the demos are supposed to
# contain, and a disagreement like that is settled by looking at the signal.
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from labbench import DATASETS as BASE

HERE = Path(__file__).parent
CLOSE_T = 40.0


def load(pq: Path):
    df = pd.read_parquet(pq)
    st = np.array(df["observation.state"].tolist())
    return st[:, 1], st[:, 5]      # shoulder_lift, gripper


def longest_closure(grip):
    closed = grip < CLOSE_T
    best = onset = cur = 0
    for i, c in enumerate(closed):
        cur = cur + 1 if c else 0
        if cur > best:
            best, onset = cur, i - cur + 1
    return onset, best


def main():
    if sys.argv[1] == "--runs":
        items = [(r, BASE / r / "data" / "chunk-000" / "episode_000000.parquet") for r in sys.argv[2:]]
        out = HERE / "_frames" / "traces_runs.png"
    else:
        dataset, eps = sys.argv[1], [int(x) for x in sys.argv[2:]]
        items = [(f"ep{e}", BASE / dataset / "data" / "chunk-000" / f"episode_{e:06d}.parquet")
                 for e in eps]
        out = HERE / "_frames" / f"traces_{dataset}.png"

    items = [(n, p) for n, p in items if p.exists()]
    if not items:
        raise SystemExit("no episodes found")

    cols = 2
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 2.6 * rows), squeeze=False)

    for k, (name, pq) in enumerate(items):
        ax = axes[k // cols][k % cols]
        lift, grip = load(pq)
        x = np.arange(len(grip))
        ax.plot(x, grip, lw=1.0, label="gripper (100=open)")
        ax.plot(x, lift, lw=1.0, label="shoulder_lift")
        ax.axhline(CLOSE_T, ls="--", lw=0.8, color="grey")
        onset, best = longest_closure(grip)
        if best > 0:
            ax.axvspan(onset, onset + best, alpha=0.15, color="tab:green")
            ax.set_title(f"{name}   longest closure {best}f from {onset}", fontsize=9)
        else:
            ax.set_title(f"{name}   NO CLOSURE  (grip min {grip.min():.0f})", fontsize=9)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")

    for k in range(len(items), rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.tight_layout()
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"traces -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
