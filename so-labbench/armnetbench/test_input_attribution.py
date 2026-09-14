# Check the perturbations and the preregistered verdicts against known answers.
#
#   python test_input_attribution.py
#
# Two things could quietly defeat the preregistration. A perturbation that does not do what its
# name says would measure something else under the right label. And a judge with its threshold
# off by one - 6 of 8 counted as 7, a tie counted as a win - would report a hypothesis as
# supported that the registered rule rejects. Both are checked here without a GPU.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from input_attribution import (ALL_PERTURBATIONS, GRIPPER, TASK_ORDER, TASKS,   # noqa: E402
                               other_task_instruction, perturb_state, perturb_task)
from judge_input_attribution import THRESHOLD, judge                               # noqa: E402

FAILED, PASSED = [], []


def check(name, got, want):
    if isinstance(want, np.ndarray):
        ok = isinstance(got, np.ndarray) and got.shape == want.shape and np.allclose(got, want)
    elif isinstance(want, float):
        ok = abs(got - want) < 1e-9
    else:
        ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    (PASSED if ok else FAILED).append(name)


# ------------------------------------------------------------------ perturbations
def test_perturbations():
    print("perturb_state")
    state = np.array([10.0, -20.0, 30.0, 5.0, 0.0, 40.0])
    sd = np.array([2.0, 4.0, 6.0, 1.0, 3.0, 8.0])
    mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 25.0])
    z = np.array([1.0, -1.0, 0.5, 2.0, 0.0, -0.25])

    check("identity leaves the state alone", perturb_state(state, "identity", sd, mean, z), state)
    check("state_mean replaces the state with the mean",
          perturb_state(state, "state_mean", sd, mean, z), mean)
    half = perturb_state(state, "state_noise_0.5sd", sd, mean, z)
    full = perturb_state(state, "state_noise_1sd", sd, mean, z)
    check("0.5sd noise is half a standard deviation along z", half, state + 0.5 * sd * z)
    check("1sd noise is one standard deviation along z", full, state + sd * z)
    # H4 is only a clean dose test if the two pushes point the same way.
    check("the 1sd push is exactly twice the 0.5sd push", full - state, 2 * (half - state))

    pushed = perturb_state(state, "state_gripper_1sd", sd, mean, z)
    check("the gripper push moves only the gripper", pushed[:GRIPPER], state[:GRIPPER])
    # gripper 40 is above its mean 25, so it is pushed down by one sd, toward the mean side
    check("an open gripper is pushed toward the mean", float(pushed[GRIPPER]), 32.0)
    low = state.copy(); low[GRIPPER] = 10.0
    check("a closed gripper is pushed toward the mean",
          float(perturb_state(low, "state_gripper_1sd", sd, mean, z)[GRIPPER]), 18.0)

    check("an image perturbation leaves the state alone",
          perturb_state(state, "all_blank", sd, mean, z), state)
    check("a language perturbation leaves the state alone",
          perturb_state(state, "task_blank", sd, mean, z), state)
    check("resample leaves the state alone", perturb_state(state, "resample", sd, mean, z), state)
    original = state.copy()
    perturb_state(state, "state_gripper_1sd", sd, mean, z)
    check("the caller's state is not modified in place", state, original)
    try:
        perturb_state(state, "state_typo", sd, mean, z)
        check("an unknown name is refused", "accepted", "refused")
    except ValueError:
        check("an unknown name is refused", "refused", "refused")

    print("\nperturb_task")
    check("task_blank empties the instruction", perturb_task("Put it in", "task_blank", "other"), "")
    check("task_other swaps in the other instruction",
          perturb_task("Put it in", "task_other", "Stack them"), "Stack them")
    check("a state perturbation leaves the instruction alone",
          perturb_task("Put it in", "state_mean", "Stack them"), "Put it in")

    print("\nother_task_instruction")
    first, last = TASK_ORDER[0], TASK_ORDER[-1]
    check("it is a different task's instruction",
          other_task_instruction(first) != TASKS[first]["instruction"], True)
    check("it is the next task in order", other_task_instruction(first),
          TASKS[TASK_ORDER[1]]["instruction"])
    check("the last task wraps to the first", other_task_instruction(last),
          TASKS[TASK_ORDER[0]]["instruction"])
    check("every perturbation named in the preregistration is run",
          sorted(ALL_PERTURBATIONS),
          sorted(["identity", "resample", "all_blank", "state_mean", "state_noise_0.5sd",
                  "state_noise_1sd", "state_gripper_1sd", "task_blank", "task_other"]))


# ------------------------------------------------------------------ judge
TASK_NAMES = [f"t{i}" for i in range(8)]


def table(overrides=None, tasks=TASK_NAMES):
    """A complete, valid table where every hypothesis holds on all 8 tasks, then overridden."""
    base = {
        "smolvla": dict(identity=0.0, resample=0.5, all_blank=3.0, state_mean=9.0,
                        **{"state_noise_0.5sd": 2.0, "state_noise_1sd": 4.0},
                        state_gripper_1sd=1.0, task_blank=0.3, task_other=0.4),
        "act": dict(identity=0.0, resample=0.0, all_blank=25.0, state_mean=6.0,
                    **{"state_noise_0.5sd": 1.0, "state_noise_1sd": 2.0},
                    state_gripper_1sd=0.5, task_blank=0.0, task_other=0.0),
    }
    rows = []
    for policy, values in base.items():
        for t in tasks:
            for pert, v in values.items():
                v = (overrides or {}).get((policy, t, pert), v)
                rows.append(dict(task=t, policy=policy, perturbation=pert, delta_mean=v))
    return pd.DataFrame(rows)


def flip(policy, perturbation, value, n):
    """Override one perturbation on the first n tasks."""
    return {(policy, TASK_NAMES[i], perturbation): value for i in range(n)}


def test_judge():
    print("\njudge: the all-true baseline")
    v = judge(table())
    check("complete", v["complete"], True)
    check("valid", v["valid"], True)
    check("H1 supported", v["H1"], True)
    check("H2 supported", v["H2"], True)
    check("H3 supported", v["H3"], True)
    check("H4 supported", v["H4"], True)
    check("ACT ignores language", v["act_ignores_language"], True)
    check("the threshold is seven", THRESHOLD, 7)

    print("\njudge: the threshold boundary")
    # H1 fails on exactly one task: 7 of 8, still supported
    v = judge(table(flip("smolvla", "state_mean", 1.0, 1)))
    check("7 of 8 is supported", (v["H1_count"], v["H1"]), (7, True))
    # fails on two: 6 of 8, not supported - no in-between wording
    v = judge(table(flip("smolvla", "state_mean", 1.0, 2)))
    check("6 of 8 is not supported", (v["H1_count"], v["H1"]), (6, False))

    print("\njudge: ties and strict inequalities")
    v = judge(table(flip("smolvla", "state_mean", 3.0, 8)))      # equal to all_blank
    check("a tie is not a win for H1", v["H1_count"], 0)
    v = judge(table(flip("act", "all_blank", 6.0, 8)))            # equal to state_mean
    check("a tie is not a win for H3", v["H3_count"], 0)
    v = judge(table(flip("smolvla", "task_blank", 1.0, 8)))       # exactly 2x resample (0.5)
    check("exactly twice the noise is not quiet", v["H2_count"], 0)
    v = judge(table(flip("smolvla", "state_noise_1sd", 2.0, 8)))  # equal to 0.5sd
    check("an equal dose response does not count for H4", v["H4_counts"]["smolvla"], 0)

    print("\njudge: directions")
    # H3 is the reverse of H1; a judge that used the same comparison for both would pass H1 here
    v = judge(table(flip("act", "all_blank", 1.0, 8)))
    check("ACT losing more to state than vision fails H3", v["H3"], False)
    check("... and does not touch H1", v["H1"], True)

    print("\njudge: H2 needs BOTH language perturbations quiet")
    v = judge(table(flip("smolvla", "task_other", 5.0, 8)))
    check("a loud task_other alone fails the task", (v["H2_count"], v["H2"]), (0, False))
    v = judge(table(flip("smolvla", "task_blank", 5.0, 8)))
    check("a loud task_blank alone fails the task", (v["H2_count"], v["H2"]), (0, False))

    print("\njudge: H4 gates H1 and H3")
    v = judge(table(flip("act", "state_noise_1sd", 0.5, 8)))
    check("H4 failing on ACT alone fails H4", v["H4"], False)
    check("H1 is then not interpreted", v["H1"], "not interpreted")
    check("H3 is then not interpreted", v["H3"], "not interpreted")
    check("H2 does not depend on H4", v["H2"], True)
    v = judge(table(flip("smolvla", "state_noise_1sd", 0.5, 2)))  # smolvla 6/8
    check("H4 needs both policies at 7", v["H4"], False)

    print("\njudge: validity")
    v = judge(table({("smolvla", "t3", "identity"): 1e-6}))
    check("any nonzero identity invalidates the run", v["valid"], False)
    check("... and names where", v["identity_nonzero"], [("smolvla", "t3")])
    v = judge(table({("act", "t5", "task_other"): 0.01}))
    check("ACT moving on an instruction change is flagged", v["act_ignores_language"], False)

    print("\njudge: incomplete runs give no verdict")
    v = judge(table(tasks=TASK_NAMES[:7]))
    check("seven tasks is incomplete", v["complete"], False)
    check("... and carries no H1 verdict", "H1" in v, False)
    check("... and reports how many tasks it has", v["tasks"], {"act": 7, "smolvla": 7})


def main():
    test_perturbations()
    test_judge()
    print(f"\n{len(PASSED)} checks passed, {len(FAILED)} failed")
    if FAILED:
        print("  " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
