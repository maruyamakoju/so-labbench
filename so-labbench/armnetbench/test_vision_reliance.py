# Check the three denominators against a cell whose answers are known before running.
#
#   python test_vision_reliance.py
#
# vision_reliance.py exists to stop one denominator from manufacturing a conclusion, so a bug
# in how it builds those denominators would defeat its only purpose - and would do it quietly,
# by producing three numbers that agree with each other because they are secretly the same
# number. The synthetic cell below has different values in every joint and a different scale
# for each denominator, so averaging the wrong columns, or dividing by the wrong one, cannot
# come out right by luck.
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vision_reliance import cell            # noqa: E402

FAILED, PASSED = [], []
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Deliberately unequal, and deliberately not each other's multiples.
NATURAL = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0]       # mean 4.0
SIGMA = [10.0, 20.0, 30.0, 40.0, 50.0, 100.0]  # mean 41.666...
DELTAS = {"identity": 0.0, "resample": 0.8, "all_blank": 12.0,
          "wrist_blank": 5.0, "front_blank": 3.0, "top_blank": 1.0, "shift_1px": 0.4}


def check(name, got, want):
    ok = abs(got - want) < 1e-6 if isinstance(want, float) else got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    (PASSED if ok else FAILED).append(name)


def build(path):
    fields = (["task", "policy", "perturbation", "phase", "frames"]
              + [f"delta_{j}" for j in JOINTS]
              + ["delta_mean", "relative_mean", "relative_gripper", "gripper_flip_rate"]
              + [f"natural_{j}" for j in JOINTS] + [f"sigma_{j}" for j in JOINTS])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name, delta in DELTAS.items():
            row = dict(task="ring_insert", policy="smolvla", perturbation=name,
                       phase="all", frames=100, delta_mean=delta,
                       # The grid writes relative_mean itself; give it a value that does NOT
                       # equal delta/natural, so a reader of the wrong column is caught.
                       relative_mean=delta * 7.0, relative_gripper=0.0, gripper_flip_rate=0.0)
            row.update({f"delta_{j}": delta for j in JOINTS})
            row.update({f"natural_{j}": v for j, v in zip(JOINTS, NATURAL)})
            row.update({f"sigma_{j}": v for j, v in zip(JOINTS, SIGMA)})
            w.writerow(row)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # The task and policy are read from the file NAME, not from the columns, because that
        # is how the grid addresses its cells.
        path = Path(tmp) / "ring_insert__smolvla.csv"
        build(path)
        got = cell(path)

    print("identity, read from the file name")
    check("task", got["task"], "ring_insert")
    check("policy", got["policy"], "smolvla")

    print("\ndenominators, averaged over all six joints")
    check("natural step is the mean of six unequal joints", got["natural_step"], 4.0)
    check("demo sigma is the mean of six unequal joints", got["demo_sigma"], 41.7)

    print("\nthe floor")
    check("own noise is resample, not identity", got["noise_raw"], 0.8)

    print("\nthe same perturbation against three denominators")
    check("raw is the action units themselves", got["all_blank_raw"], 12.0)
    # 12/4 would be 3.0; the grid's own relative_mean column says 84.0. The grid's column is
    # the authority, because it was computed per joint before averaging.
    check("x natural comes from the grid's own column", got["all_blank_natural"], 84.0)
    check("% demo sigma is raw over sigma", got["all_blank_sigma"], round(100 * 12.0 / 41.666666, 1))

    print("\neach view separately")
    check("wrist", got["wrist_blank_raw"], 5.0)
    check("front", got["front_blank_raw"], 3.0)
    check("top", got["top_blank_raw"], 1.0)
    check("the three views are not collapsed into one number",
          len({got["wrist_blank_raw"], got["front_blank_raw"], got["top_blank_raw"]}), 3)

    print("\nthe three denominators must not secretly be the same number")
    check("raw, natural and sigma all differ",
          len({got["all_blank_raw"], got["all_blank_natural"], got["all_blank_sigma"]}), 3)

    print("\na cell missing a view does not invent one")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cable_clip__act.csv"
        saved = DELTAS.pop("top_blank")
        build(path)
        DELTAS["top_blank"] = saved
        partial = cell(path)
    check("the absent view is absent", "top_blank_raw" in partial, False)
    check("the present views survive", partial["wrist_blank_raw"], 5.0)

    print(f"\n{len(PASSED)} checks passed, {len(FAILED)} failed")
    if FAILED:
        print("  " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
