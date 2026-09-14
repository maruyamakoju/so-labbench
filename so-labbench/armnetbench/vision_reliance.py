# How much does a policy's commanded action depend on its cameras, in units anyone can check?
#
#   python vision_reliance.py
#
# The sensitivity grid reports each perturbation as a multiple of the policy's natural
# one-frame step. That is the right scale for asking "is this rig difference something the
# policy notices", because it compares an induced change against the change the output makes
# anyway. It is the WRONG scale for comparing two policies to each other: if one architecture
# happens to move more between consecutive frames, dividing by that flatters it, and a
# difference in the denominator would read as a difference in vision reliance.
#
# So this prints the same quantity against three denominators that fail in different ways:
#
#   raw          the commanded action's mean absolute change, in benchmark action units.
#                No denominator at all, so no denominator can be blamed - but action units
#                are a percentage of one arm's calibrated range and mean nothing physical.
#   natural      multiples of that policy's own one-frame step. Policy-relative.
#   demo sigma   percent of the standard deviation of the human demonstrations' own actions
#                on that task. The only denominator that is a property of the TASK rather
#                than of either policy, so it is the one that makes the comparison fair.
#
# A conclusion that survives all three is not an artefact of the scale. A conclusion that
# appears in only one is a fact about that denominator.
#
# What this cannot say: moving less is not the same as using vision less well. A policy could
# be reading its cameras carefully and changing its command very little, and nothing here
# distinguishes that from ignoring them. The floors below are what separate "barely responds"
# from "does not respond at all".
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
GRID = HERE / "grid"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# Perturbations that remove a view entirely, in the order a reader should meet them.
VIEWS = ["all_blank", "wrist_blank", "front_blank", "top_blank"]


def cell(path):
    d = pd.read_csv(path).set_index("perturbation")
    task, policy = os.path.basename(path)[:-4].split("__")
    any_row = d.iloc[0]
    natural = float(np.mean([any_row[f"natural_{j}"] for j in JOINTS]))
    sigma = float(np.mean([any_row[f"sigma_{j}"] for j in JOINTS]))
    out = dict(task=task, policy=policy, natural_step=round(natural, 2),
               demo_sigma=round(sigma, 1),
               # same frame, different seed: zero for a deterministic policy, and the real
               # floor for a sampling one
               noise_raw=round(float(d.delta_mean.get("resample", 0.0)), 2))
    for v in VIEWS:
        if v in d.index:
            raw = float(d.delta_mean[v])
            out[f"{v}_raw"] = round(raw, 2)
            out[f"{v}_natural"] = round(float(d.relative_mean[v]), 1)
            out[f"{v}_sigma"] = round(100 * raw / sigma, 1) if sigma else float("nan")
    return out


def main():
    cells = sorted(GRID.glob("*__*.csv"))
    if not cells:
        raise SystemExit(f"no cells in {GRID}; run run_sensitivity_grid.sh first")
    d = pd.DataFrame([cell(p) for p in cells])
    complete = d.groupby("policy").task.nunique()
    d.to_csv(HERE / "vision_reliance.csv", index=False)

    print("cells read: " + ", ".join(f"{p} {n}/8 tasks" for p, n in complete.items()))
    if (complete < 8).any():
        print("  PARTIAL - a policy with fewer than 8 tasks is not comparable to one with 8,")
        print("  because tasks differ in how much vision they need. Read the rows, not a mean.\n")

    print("blanking ALL cameras, against three denominators")
    print(f"{'task':21} {'policy':9} {'raw':>7} {'x natural':>10} {'% demo sigma':>13} "
          f"{'own noise':>10}")
    for r in d.sort_values(["task", "policy"]).itertuples():
        if not hasattr(r, "all_blank_raw"):
            continue
        print(f"{r.task:21} {r.policy:9} {r.all_blank_raw:7.2f} {r.all_blank_natural:10.1f} "
              f"{r.all_blank_sigma:12.1f}% {r.noise_raw:10.2f}")

    print("\nper policy, across the tasks it has")
    print(f"{'policy':9} {'tasks':>6}   {'raw':^15} {'x natural':^15} {'% demo sigma':^17}")
    for policy, g in d.groupby("policy"):
        if "all_blank_raw" not in g:
            continue
        raw = f"{g.all_blank_raw.min():.2f} - {g.all_blank_raw.max():.2f}"
        nat = f"{g.all_blank_natural.min():.1f} - {g.all_blank_natural.max():.1f}"
        sig = f"{g.all_blank_sigma.min():.1f}% - {g.all_blank_sigma.max():.1f}%"
        print(f"{policy:9} {len(g):6d}   {raw:^15} {nat:^15} {sig:^17}")

    # Does the policy still tell the perturbations apart? A policy whose every response sits
    # at its own noise floor is not responding weakly, it is not responding, and the two
    # deserve different words.
    print("\ndoes each policy still discriminate between views?")
    print(f"{'policy':9} {'single-view range':^19} {'own noise':>10} {'weakest / floor':>16}")
    for policy, g in d.groupby("policy"):
        singles = [c for c in ("wrist_blank_raw", "front_blank_raw", "top_blank_raw") if c in g]
        if not singles:
            continue
        lo = min(g[c].min() for c in singles)
        hi = max(g[c].max() for c in singles)
        floor = float(g.noise_raw.max())
        # "Above the floor" is a threshold, and a threshold hides how close a thing came to
        # failing it. The margin is the number that should be read.
        margin = lo / floor if floor > 1e-9 else float("inf")
        note = "deterministic, no floor" if floor <= 1e-9 else (
            f"{margin:.1f}x" + ("" if margin >= 2 else "  <- weakest view barely clears its own noise"))
        print(f"{policy:9} {f'{lo:.2f} - {hi:.2f}':^19} {floor:10.2f} {note:>16}")

    print(f"\nwrote vision_reliance.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
