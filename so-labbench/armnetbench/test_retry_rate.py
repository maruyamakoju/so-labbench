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

from retry_rate import count_closures, otsu_threshold, fisher_ci

FAILED = []


def check(name, got, want):
    ok = got == want if not isinstance(want, float) else abs(got - want) < 1e-6
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    if not ok:
        FAILED.append(name)


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

    print("\nfisher_ci")
    lo, hi = fisher_ci(0.0, 56)
    check("zero correlation gives a symmetric interval", abs(lo + hi) < 1e-9, True)
    check("n=56 interval is about +-0.26", abs(hi - 0.2646) < 0.01, True)
    lo, hi = fisher_ci(0.9, 14)
    check("strong correlation excludes zero", lo > 0, True)
    check("too few points gives nan", fisher_ci(0.5, 3)[0] != fisher_ci(0.5, 3)[0], True)

    print(f"\n{'ALL CHECKS PASSED' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
