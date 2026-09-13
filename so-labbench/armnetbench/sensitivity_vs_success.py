# Does what a policy attends to predict whether it succeeds?
#
#   python sensitivity_vs_success.py
#
# ArmnetBench publishes a success rate for every policy on every task. The sensitivity grid
# measures, for the same cells, how far each policy's output moves when its inputs are
# disturbed. Putting the two side by side asks a question nobody has been able to ask before
# the benchmark existed: is there anything about how a policy uses its cameras that goes with
# how often it succeeds?
#
# Four readings per cell, each a ratio to that cell's own natural one-frame step, so tasks
# with different motion scales are comparable:
#
#   vision_dependence  blanking all three cameras. How much of the output is vision at all.
#   camera_dominance   the largest single-camera effect divided by the smallest. A policy
#                      that leans on one view scores high; one that spreads across three
#                      scores near 1.
#   aiming_sensitivity the largest shift/rotate/zoom effect. How much hand-aiming would cost.
#   noise_floor        the same frame drawn twice. Zero for a deterministic policy.
#
# What this cannot say. Eight tasks is eight points, and the published rates carry their own
# n=30 uncertainty, so a correlation here is suggestive and nothing more. The honest use is
# the opposite direction: a reading that does NOT move with success rules that explanation
# out, and cable_clip - where all seven published policies score exactly 0.000 - is a case
# where knowing what the policy still responds to says something about what defeats it.
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
GRID = HERE / "grid"
REFERENCE = HERE / "reference_results.csv"
TASKS = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))["tasks"]
CAMERAS = ("front", "top", "wrist")
AIMING = ("shift_x", "shift_y", "rotate", "zoom")
OUT = HERE / "sensitivity_vs_success.csv"


def summarise(cell: pd.DataFrame) -> dict:
    row = cell.set_index("perturbation")
    natural = float(row.loc["identity", "natural_shoulder_pan":"natural_gripper"].mean())
    identity = float(row.loc["identity", "delta_mean"])
    singles = {c: float(row.loc[f"{c}_blank", "delta_mean"]) for c in CAMERAS if f"{c}_blank" in row.index}
    aiming = [float(v) for name, v in row["delta_mean"].items() if any(a in name for a in AIMING)]
    out = {
        "identity": round(identity, 4),
        "natural_step": round(natural, 3),
        "noise_floor": round(float(row.loc["resample", "delta_mean"]) / natural, 3),
        "aiming_sensitivity": round(max(aiming) / natural, 3) if aiming else float("nan"),
        "frames": int(cell["frames"].iloc[0]),
    }
    if "all_blank" in row.index:
        out["vision_dependence"] = round(float(row.loc["all_blank", "delta_mean"]) / natural, 3)
    if len(singles) == 3:
        out["camera_dominance"] = round(max(singles.values()) / max(min(singles.values()), 1e-9), 3)
        out["dominant_camera"] = max(singles, key=singles.get)
        for c, v in singles.items():
            out[f"blank_{c}"] = round(v / natural, 3)
    return out


def main():
    cells = sorted(GRID.glob("*__*.csv"))
    if not cells:
        raise SystemExit(f"no cells in {GRID}; run run_sensitivity_grid.sh first")
    published = {}
    with REFERENCE.open() as f:
        for r in csv.DictReader(f):
            published[(r["task"], r["policy_type"])] = float(r["success_rate"])

    rows = []
    for path in cells:
        task, policy = path.stem.split("__")
        d = pd.read_csv(path)
        if "identity" not in set(d.perturbation):
            print(f"  {path.name}: no identity row, skipped")
            continue
        summary = summarise(d)
        if summary["identity"] > 1e-9:
            print(f"  {path.name}: identity reads {summary['identity']}, must be 0 - skipped")
            continue
        instruction = TASKS[task]["instruction"]
        rows.append(dict(task=task, policy=policy,
                         success=published.get((instruction, policy), float("nan")), **summary))

    d = pd.DataFrame(rows).sort_values(["policy", "success"], ascending=[True, False])
    d.to_csv(OUT, index=False)

    for policy in d.policy.unique():
        mine = d[d.policy == policy]
        print(f"\n{'=' * 104}\n{policy}   ({len(mine)} tasks)\n")
        print(f"{'task':22} {'published':>9} {'vision':>7} {'dominant':>9} {'dom.ratio':>9} {'aiming':>7} {'noise':>6}")
        for _, r in mine.iterrows():
            print(f"{r.task:22} {r.success:9.3f} {r.get('vision_dependence', float('nan')):7.1f} "
                  f"{str(r.get('dominant_camera', '?')):>9} {r.get('camera_dominance', float('nan')):9.2f} "
                  f"{r.aiming_sensitivity:7.2f} {r.noise_floor:6.2f}")

        usable = mine.dropna(subset=["success"])
        if len(usable) >= 4:
            print(f"\n  correlation with the published success rate over {len(usable)} tasks:")
            for column in ["vision_dependence", "camera_dominance", "aiming_sensitivity", "noise_floor"]:
                if column not in usable or usable[column].isna().all():
                    continue
                pair = usable[["success", column]].dropna()
                if len(pair) < 4 or pair[column].std() == 0:
                    continue
                r = float(np.corrcoef(pair.success, pair[column])[0, 1])
                # Fisher interval, which at n=8 is wide enough to be the point.
                z = np.arctanh(np.clip(r, -0.999, 0.999))
                lo, hi = np.tanh(z - 1.96 / np.sqrt(len(pair) - 3)), np.tanh(z + 1.96 / np.sqrt(len(pair) - 3))
                verdict = "excludes zero" if lo * hi > 0 else "spans zero"
                print(f"    {column:20} r = {r:+.2f}   95% CI {lo:+.2f} to {hi:+.2f}   {verdict}")
            print(f"    (n={len(usable)} tasks, and each published rate is itself from 30 rollouts.")
            print("     A correlation here is suggestive; the interval is the honest part.)")

    zero = d[d.success == 0.0]
    if not zero.empty:
        print(f"\n{'=' * 104}\nTasks where the published rate is exactly 0.000:\n")
        for _, r in zero.iterrows():
            print(f"  {r.task}/{r.policy}: the policy still responds to vision at "
                  f"{r.get('vision_dependence', float('nan')):.1f}x its natural step, and to aiming at "
                  f"{r.aiming_sensitivity:.2f}x.")
        print("\n  A policy that never succeeds but still reacts to what it sees is not failing")
        print("  because it cannot see the task. Whatever defeats it is downstream of perception.")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
