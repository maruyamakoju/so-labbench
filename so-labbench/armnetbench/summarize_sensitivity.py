# Read rig_sensitivity.csv the way it should be read.
#
#   python summarize_sensitivity.py [csv]
#
# The raw table is 34 rows per policy and invites cherry-picking. This prints it in the
# order the question demands: the two floors first, so the reader calibrates before seeing
# any effect, then the perturbations that exceed both, then the ones that do not.
#
# A perturbation only deserves attention if it moves the policy's output by more than
#   * `resample` - the same frame drawn again, which for a sampling policy is movement for
#     no reason at all, and
#   * `natural_step` - what the output does between two consecutive frames anyway.
# Below either floor, the rig difference is not something this policy responds to.
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

# Which of our known rig deviations each perturbation stands for. Anything not listed is a
# probe, included to give the scale a shape rather than because we expect to see it.
MEANS = {
    "front_aspect_4_3": "our front camera is 4:3 where theirs is 16:9 (lower bound - the lenses also differ)",
    "front_blank": "front camera missing or mis-indexed",
    "top_blank": "top camera missing or mis-indexed",
    "wrist_blank": "wrist camera missing or mis-indexed",
    "front_uniform": "front camera present but seeing nothing useful",
    "top_uniform": "top camera present but seeing nothing useful",
    "wrist_uniform": "wrist camera present but seeing nothing useful",
    "all_dark_30pct": "room lit dimmer than theirs",
    "all_bright_30pct": "room lit brighter than theirs",
    "all_blur_sigma2": "softer optics than theirs",
    "all_blank": "EVERY camera blanked - the upper bound on anything visual, and the test of "
                 "whether this policy uses vision at all",
    "all_uniform": "every camera present but uninformative",
}
ALIGNMENT_PREFIXES = ("shift_x", "shift_y", "rotate", "zoom")


def describe(name: str) -> str:
    if name in MEANS:
        return MEANS[name]
    if any(p in name for p in ALIGNMENT_PREFIXES):
        return "aiming a camera by hand"
    return ""


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "rig_sensitivity.csv"
    d = pd.read_csv(path)
    for policy in d.policy.unique():
        mine = d[d.policy == policy].set_index("perturbation")
        identity = float(mine.loc["identity", "delta_mean"])
        resample = float(mine.loc["resample", "delta_mean"])
        natural = float(mine.loc["identity", "natural_shoulder_pan":"natural_gripper"].mean())
        phase = mine["phase"].iloc[0] if "phase" in mine.columns else "all"
        frames = int(mine["frames"].iloc[0])

        print(f"\n{'=' * 100}\n{policy}   ({frames} frames, phase={phase})")
        if identity > 1e-9:
            print(f"  WARNING: identity reads {identity:.4f} and must read 0. The harness is wrong; "
                  "do not read anything below.")
        print(f"  floors: re-running the policy on the same frame moves it {resample:.2f}; "
              f"one frame of ordinary motion moves it {natural:.2f}")
        if resample < 1e-6:
            print("          (this policy is deterministic, so its own noise floor is zero and the "
                  "natural step is the only floor)")
        floor = max(resample, natural)

        rows = mine.drop(index=["identity", "resample"]).sort_values("delta_mean", ascending=False)
        above = rows[rows.delta_mean > floor]
        below = rows[rows.delta_mean <= floor]

        print(f"\n  moves the policy MORE than both floors ({floor:.2f}):")
        if above.empty:
            print("    nothing - no rig difference tested changes this policy's output "
                  "more than re-running it does")
        for name, r in above.iterrows():
            print(f"    {name:24} {r.delta_mean:6.2f}  = {r.delta_mean / floor:5.1f}x floor, "
                  f"{r.relative_mean:5.2f}x natural   {describe(name)}")

        print(f"\n  below the floor, so not something this policy responds to:")
        for name, r in below.iterrows():
            print(f"    {name:24} {r.delta_mean:6.2f}   {describe(name)}")

    print(f"\n{'=' * 100}")
    print("Reading this: a large value is a necessary condition for a rig difference to cost")
    print("success on the robot, not a sufficient one - offline agreement is already known not")
    print("to predict rollout success on this hardware. The informative direction is the small")
    print("one: a difference the policy's output does not respond to cannot explain a change in")
    print("its success rate, and can be dropped from the list of things to fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
