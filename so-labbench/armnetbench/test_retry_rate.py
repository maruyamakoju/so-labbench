# Check the two functions the retry result rests on, against cases with known answers.
#
#   python test_retry_rate.py
#
# count_closures and otsu_threshold decide the whole analysis. A bug in either would produce
# a number that looks like a result and is not, so they are checked on traces built by hand
# where the right answer is known before running, including the boundary cases a run-length
# encoder gets wrong: a trace that starts closed, ends closed, is closed throughout, or holds
# a grasp for exactly the minimum duration.
import sys

import numpy as np
import pandas as pd

from retry_rate import (FPS, MIN_PER_SIDE, count_closures, fisher_ci, otsu_threshold,
                        recompute, within_cell_sign_test)

FAILED = []
PASSED = []


def check(name, got, want):
    ok = got == want if not isinstance(want, float) else abs(got - want) < 1e-6
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    (PASSED if ok else FAILED).append(name)


def trace(*values):
    return np.array(values, dtype=np.float32)


def main():
    print("count_closures, threshold 10, minimum 3 frames")
    T, M = 10.0, 3
    check("no closure at all", count_closures(trace(50, 50, 50, 50), T, M), 0)
    check("one closure in the middle",
          count_closures(trace(50, 5, 5, 5, 50), T, M), 1)
    check("two separated closures",
          count_closures(trace(50, 5, 5, 5, 50, 50, 5, 5, 5, 50), T, M), 2)
    # the run-length boundaries: a trace can begin or end mid-grasp
    check("closure at the very start", count_closures(trace(5, 5, 5, 50, 50), T, M), 1)
    check("closure at the very end", count_closures(trace(50, 50, 5, 5, 5), T, M), 1)
    check("closed for the whole trace", count_closures(trace(5, 5, 5, 5), T, M), 1)
    check("both ends closed, open between",
          count_closures(trace(5, 5, 5, 50, 50, 5, 5, 5), T, M), 2)
    # duration filter
    check("exactly the minimum counts", count_closures(trace(50, 5, 5, 5, 50), T, M), 1)
    check("one frame short does not", count_closures(trace(50, 5, 5, 50), T, M), 0)
    check("chatter is rejected",
          count_closures(trace(50, 5, 50, 5, 50, 5, 50), T, M), 0)
    check("chatter beside a real grasp counts once",
          count_closures(trace(50, 5, 50, 5, 5, 5, 50), T, M), 1)
    # the threshold is strict: a value equal to it is open
    check("value equal to the threshold is open",
          count_closures(trace(50, 10, 10, 10, 50), T, M), 0)
    check("empty trace", count_closures(trace(), T, M), 0)

    print("\nsensitivity of the count to the minimum duration")
    long_then_short = trace(50, 5, 5, 5, 5, 5, 50, 5, 5, 50)
    check("min 2 sees both", count_closures(long_then_short, T, 2), 2)
    check("min 3 sees only the long one", count_closures(long_then_short, T, 3), 1)

    print("\notsu_threshold on a bimodal gripper trace")
    rng = np.random.default_rng(0)
    closed = rng.normal(5, 1.5, 4000)
    open_ = rng.normal(60, 4.0, 6000)
    t = otsu_threshold(np.concatenate([closed, open_]))
    print(f"  split found at {t:.1f}")
    between = 5 < t < 60
    check("split lands between the two modes", between, True)
    # it must separate the modes, not merely sit between them
    mixed = np.concatenate([closed, open_])
    purity = max((mixed[mixed < t] < 20).mean(), 0.0)
    check("almost everything below the split is the closed mode", purity > 0.98, True)

    print("\notsu_threshold degenerate inputs")
    flat = np.full(500, 42.0)
    check("a constant trace returns that constant", abs(otsu_threshold(flat) - 42.0) < 1.0, True)
    check("a single value does not crash", isinstance(otsu_threshold(np.array([7.0])), float), True)

    print("\notsu_threshold and count_closures together, on a trace with a known answer")
    # Ten grasps of 20 frames, separated by open stretches spread wide enough that the
    # distribution's mean and its mid-range both land INSIDE the open mode. A split placed at
    # either would cut the open mode in half and invent grasps that are not there, which is
    # exactly how a plausible wrong number gets produced without anything crashing.
    rng = np.random.default_rng(7)
    segments = [rng.uniform(40, 100, 50)]
    for _ in range(10):
        segments.append(np.full(20, 5.0) + rng.normal(0, 0.3, 20))
        segments.append(rng.uniform(40, 100, 50))
    built = np.concatenate(segments).astype(np.float32)
    split = otsu_threshold(built)
    check("ten built grasps are recovered", count_closures(built, split, 3), 10)
    check("the split separates the modes", bool(5.0 < split < 40.0), True)
    check("the mean would not have separated them",
          count_closures(built, float(built.mean()), 3) != 10, True)
    check("the mid-range would not have separated them",
          count_closures(built, float((built.min() + built.max()) / 2), 3) != 10, True)

    print("\nfisher_ci (exact values, not approximations)")
    lo, hi = fisher_ci(0.0, 56)
    check("zero correlation gives a symmetric interval", abs(lo + hi) < 1e-9, True)
    check("n=56, r=0 upper bound", round(hi, 6), 0.262901)
    # Small n is where n vs n-3 separates; at n=56 the two differ by less than a rounding
    # tolerance, so a wrong denominator hid here.
    check("n=7, r=0 upper bound", round(fisher_ci(0.0, 7)[1], 6), 0.753058)
    # r=0 cannot detect a wrong z transform, because z(0) is 0 either way.
    lo, hi = fisher_ci(0.5, 20)
    check("n=20, r=0.5 lower bound", round(lo, 6), 0.073811)
    check("n=20, r=0.5 upper bound", round(hi, 6), 0.771761)
    lo, hi = fisher_ci(0.9, 14)
    check("strong correlation excludes zero", lo > 0, True)
    check("n=14, r=0.9 lower bound", round(lo, 6), 0.707054)
    check("too few points gives nan", fisher_ci(0.5, 3)[0] != fisher_ci(0.5, 3)[0], True)

    print("\nrecompute: the per-episode rate table")
    # Two policy rollouts with the SAME number of grasps but different durations, plus the
    # demonstrations that everything else is measured against.
    def grasps(n, length):
        trace = np.full(length, 60.0, dtype=np.float32)
        for k in range(n):
            trace[10 + k * 20: 10 + k * 20 + 6] = 5.0
        return trace

    meta = pd.DataFrame([
        dict(episode_index=0, task_key="t", policy_type="teleoperated", success_class="successful"),
        dict(episode_index=1, task_key="t", policy_type="act", success_class="successful"),
        dict(episode_index=2, task_key="t", policy_type="act", success_class="failure"),
    ])
    traces = {0: grasps(2, 100), 1: grasps(2, 100), 2: grasps(2, 200)}
    out = recompute(meta, traces, {"t": 30.0}, 3, 1.0)
    check("the demonstrations survive into the table",
          "teleoperated" in set(out.policy), True)
    check("a 100-frame rollout with 2 grasps rates 0.4 per second",
          round(float(out[out.policy == "teleoperated"].rate.iloc[0]), 3), 0.4)
    rates = sorted(round(float(v), 3) for v in out[out.policy == "act"].rate)
    check("the same grasp count over twice the time halves the rate", rates, [0.2, 0.4])
    check("the frame rate used is the benchmark's", FPS, 20.0)

    print("\nwithin_cell_sign_test")
    def cell(task, policy, win_rate, lose_rate, n_win=3, n_lose=3):
        return ([dict(task=task, policy=policy, outcome="successful", rate=win_rate)] * n_win
                + [dict(task=task, policy=policy, outcome="failure", rate=lose_rate)] * n_lose)

    frame = pd.DataFrame(
        cell("a", "act", 0.10, 0.20)            # failures grasp more: +0.10
        + cell("b", "act", 0.20, 0.10)          # failures grasp less: -0.10
        + cell("c", "act", 0.15, 0.15)          # a tie: 0.00
        + cell("d", "act", 0.10, 0.30, n_win=2)  # too few successes to compare
        + cell("a", "teleoperated", 0.10, 0.90))  # the human is not a policy being scored
    rows, higher, n, p, median = within_cell_sign_test(frame)
    check("only comparable policy cells are used", n, 3)
    check("the demonstrations are not one of them",
          all(r["policy"] != "teleoperated" for r in rows), True)
    check("a cell short on one side is dropped",
          all(r["task"] != "d" for r in rows), True)
    check("ties count toward neither tail", higher, 1)
    check("the p value carries the whole upper tail", round(p, 6), 0.875)
    check("the median difference", round(median, 6), 0.0)
    check("successes and failures are not the same side",
          sorted(round(r["delta"], 2) for r in rows), [-0.1, 0.0, 0.1])
    check("a cell needs three per side", MIN_PER_SIDE, 3)
    empty = within_cell_sign_test(pd.DataFrame(cell("a", "act", 0.1, 0.2, n_win=1, n_lose=1)))
    check("no comparable cell returns no test", empty[1:3], (0, 0))

    # One machine-readable line, so mutation_check.py can tell a caught mutation from a
    # survivor without parsing the whole run.
    print(f"\n{len(PASSED)} checks passed, {len(FAILED)} failed")
    if FAILED:
        print("  " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
