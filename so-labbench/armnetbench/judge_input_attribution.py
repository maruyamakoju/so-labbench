# Apply the preregistered decision rules, and nothing else.
#
#   python judge_input_attribution.py
#
# so-labbench/prereg_input_attribution.md fixed four hypotheses, one validity condition and a
# single threshold before any measurement ran. This file is those rules as code. It does not
# look for patterns, choose a denominator, or soften a verdict: a hypothesis either reaches 7
# of 8 tasks or it is written up as not supported.
#
# The verdict logic is in judge(), which takes a table and returns verdicts, so it can be tested
# against tables whose right answers are known. A preregistration is worth nothing if the code
# that applies it has an off-by-one in the threshold.
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
GRID = HERE / "attribution"
NEEDED_TASKS = 8
THRESHOLD = 7          # "8課題中7以上", one-sided sign test p = 0.035
H2_NOISE_MULTIPLE = 2.0


def _value(df, policy, task, perturbation):
    row = df[(df.policy == policy) & (df.task == task) & (df.perturbation == perturbation)]
    if len(row) != 1:
        raise KeyError(f"{policy}/{task}/{perturbation}: expected 1 row, found {len(row)}")
    return float(row.delta_mean.iloc[0])


def judge(df):
    """Verdicts for H1-H4 and validity, exactly as preregistered."""
    out = {}
    tasks_by_policy = {p: sorted(df[df.policy == p].task.unique()) for p in ("act", "smolvla")}
    complete = all(len(t) == NEEDED_TASKS for t in tasks_by_policy.values())
    out["complete"] = complete
    out["tasks"] = {p: len(t) for p, t in tasks_by_policy.items()}
    if not complete:
        # A verdict on 5 of 8 tasks would need a different threshold, and choosing one after
        # seeing which tasks finished is exactly what the preregistration forbids.
        return out

    def count(policy, predicate):
        return sum(1 for t in tasks_by_policy[policy] if predicate(policy, t))

    # Validity: identity must be exactly zero everywhere.
    nonzero = [(p, t) for p in tasks_by_policy for t in tasks_by_policy[p]
               if _value(df, p, t, "identity") != 0.0]
    out["valid"] = not nonzero
    out["identity_nonzero"] = nonzero

    # Added before running (see the preregistration's addendum): ACT takes no instruction, so
    # changing it must move ACT's command by exactly zero. A second wiring check.
    act_language = [t for t in tasks_by_policy["act"]
                    if _value(df, "act", t, "task_blank") != 0.0
                    or _value(df, "act", t, "task_other") != 0.0]
    out["act_ignores_language"] = not act_language
    out["act_language_nonzero"] = act_language

    h4 = {p: count(p, lambda p, t: _value(df, p, t, "state_noise_1sd")
                                   > _value(df, p, t, "state_noise_0.5sd"))
          for p in ("act", "smolvla")}
    out["H4_counts"] = h4
    out["H4"] = all(n >= THRESHOLD for n in h4.values())

    h1 = count("smolvla", lambda p, t: _value(df, p, t, "state_mean") > _value(df, p, t, "all_blank"))
    h3 = count("act", lambda p, t: _value(df, p, t, "all_blank") > _value(df, p, t, "state_mean"))
    out["H1_count"], out["H3_count"] = h1, h3
    if out["H4"]:
        out["H1"] = h1 >= THRESHOLD
        out["H3"] = h3 >= THRESHOLD
    else:
        # "これが成り立たなければ、状態の摂動が壊れているので H1 と H3 は解釈しない"
        out["H1"] = "not interpreted"
        out["H3"] = "not interpreted"

    def language_quiet(p, t):
        floor = H2_NOISE_MULTIPLE * _value(df, p, t, "resample")
        return (_value(df, p, t, "task_blank") < floor
                and _value(df, p, t, "task_other") < floor)

    h2 = count("smolvla", language_quiet)
    out["H2_count"] = h2
    out["H2"] = h2 >= THRESHOLD
    return out


def main():
    cells = sorted(GRID.glob("*.csv"))
    if not cells:
        raise SystemExit(f"no results in {GRID}")
    df = pd.concat([pd.read_csv(p) for p in cells], ignore_index=True)
    v = judge(df)

    print(f"tasks present: act {v['tasks']['act']}/8, smolvla {v['tasks']['smolvla']}/8")
    if not v["complete"]:
        print("INCOMPLETE - no verdicts. The threshold is 7 of 8 and is not rescaled for fewer.")
        return 1

    if not v["valid"]:
        print(f"INVALID RUN - identity is not zero for: {v['identity_nonzero']}")
        print("Per the preregistration, no results are written. Fix the wiring and rerun.")
        return 1
    print("validity: identity is exactly 0 for both policies on all 8 tasks")
    print(f"wiring:   ACT ignores the instruction on all tasks: {v['act_ignores_language']}"
          + ("" if v["act_ignores_language"] else f"  <- NONZERO on {v['act_language_nonzero']}"))

    def verdict(x):
        return x if isinstance(x, str) else ("SUPPORTED" if x else "not supported")

    print(f"\nH4 (instrument) state push doubles, command moves further: "
          f"act {v['H4_counts']['act']}/8, smolvla {v['H4_counts']['smolvla']}/8 -> {verdict(v['H4'])}")
    print(f"H1 SmolVLA moves more without proprioception than without vision: "
          f"{v['H1_count']}/8 -> {verdict(v['H1'])}")
    print(f"H2 SmolVLA's instruction moves it less than 2x its own noise:     "
          f"{v['H2_count']}/8 -> {verdict(v['H2'])}")
    print(f"H3 ACT moves more without vision than without proprioception:   "
          f"{v['H3_count']}/8 -> {verdict(v['H3'])}")
    print(f"\nthreshold for every hypothesis: {THRESHOLD} of {NEEDED_TASKS} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
