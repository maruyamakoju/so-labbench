# Do the retry-rate tests actually catch anything?
#
#   python mutation_check.py            # break the code, each way must be caught
#   python mutation_check.py --list     # what it breaks, without touching the file
#
# test_retry_rate.py passing is not evidence by itself. Tests written after the code tend to
# assert what the code does rather than what it should do, and the finding that rests on this
# code is a NEGATIVE result - "grasping more than the human does not predict failure" - which
# is exactly the kind of claim a broken counter produces by accident. A counter that always
# returned the same number would make every difference vanish and every test of the pipeline
# look calm.
#
# So each mutation below is a defect that was real here or would be easy to introduce, and
# every one must make at least one check fail. A survivor is a hole in the suite at exactly
# that behaviour.
#
# The file is restored from its own bytes rather than from git, so this is safe on a dirty
# tree. It refuses to start if the suite is already failing, because a survivor could not
# then be told apart from the existing failure.
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TARGET = "retry_rate.py"

# (what it breaks, exact text, replacement)
MUTATIONS = [
    ("a grasp lasting exactly the minimum stops counting",
     "return sum(1 for a, b in zip(starts, ends) if b - a >= min_frames)",
     "return sum(1 for a, b in zip(starts, ends) if b - a > min_frames)"),
    ("the duration filter is dropped, so chatter counts as grasps",
     "return sum(1 for a, b in zip(starts, ends) if b - a >= min_frames)",
     "return len(starts)"),
    ("a trace that begins mid-grasp loses that grasp",
     "    if closed[0]:\n        starts.insert(0, 0)",
     "    if False:\n        starts.insert(0, 0)"),
    ("a trace that ends mid-grasp loses that grasp",
     "    if closed[-1]:\n        ends.append(len(closed))",
     "    if False:\n        ends.append(len(closed))"),
    ("the threshold becomes inclusive, so a value equal to it reads closed",
     "    closed = gripper < threshold",
     "    closed = gripper <= threshold"),
    ("run starts and ends are swapped",
     "    starts = list(np.where(edges == 1)[0] + 1)\n    ends = list(np.where(edges == -1)[0] + 1)",
     "    starts = list(np.where(edges == -1)[0] + 1)\n    ends = list(np.where(edges == 1)[0] + 1)"),
    ("the closed/open split minimises separation instead of maximising it",
     "    return float(centres[int(np.argmax(between))])",
     "    return float(centres[int(np.argmin(between))])"),
    ("the split becomes the mean, which is not where two modes divide",
     "    return float(centres[int(np.argmax(between))])",
     "    return float(np.mean(values))"),
    ("the split becomes the midpoint of the range",
     "    return float(centres[int(np.argmax(between))])",
     "    return float((values.min() + values.max()) / 2)"),
    ("the correlation interval uses n instead of n-3, so it reads too narrow",
     "    se = 1 / math.sqrt(n - 3)",
     "    se = 1 / math.sqrt(n)"),
    ("the interval silently becomes 90% while still being labelled 95%",
     "    crit = 1.959963985 if conf == 0.95 else 2.5758",
     "    crit = 1.644853627 if conf == 0.95 else 2.5758"),
    ("the z transform loses its half, inflating every interval",
     "    z = 0.5 * math.log((1 + r) / (1 - r))",
     "    z = math.log((1 + r) / (1 - r))"),
    # The pipeline, not just its pieces. These are the bugs that produce a plausible
    # publishable number rather than a crash, which is why they are the dangerous kind.
    ("the demonstrations are dropped from the recompute, so there is no human to compare to",
     "        if e not in traces:\n            continue\n        g = traces[e]\n        n_close = count_closures(g, thresholds[ep.task_key] * scale, min_frames)",
     "        if e not in traces or ep.policy_type == \"teleoperated\":\n            continue\n        g = traces[e]\n        n_close = count_closures(g, thresholds[ep.task_key] * scale, min_frames)"),
    ("closures stop being divided by duration, so the stopping rule leaks back in",
     "        rows.append((ep.task_key, ep.policy_type, ep.success_class, n_close / (len(g) / FPS)))",
     "        rows.append((ep.task_key, ep.policy_type, ep.success_class, float(n_close)))"),
    ("the within-cell test compares failures to failures",
     "        won, lost = g[g.outcome == \"successful\"], g[g.outcome == \"failure\"]",
     "        won, lost = g[g.outcome == \"failure\"], g[g.outcome == \"failure\"]"),
    ("a cell needs only one rollout per side, so one trial decides its sign",
     "MIN_PER_SIDE = 3 ",
     "MIN_PER_SIDE = 1 "),
    ("the sign test counts ties as evidence",
     "    higher, n = sum(1 for d in deltas if d > 0), len(deltas)",
     "    higher, n = sum(1 for d in deltas if d >= 0), len(deltas)"),
    ("the sign test loses its upper tail, so any result looks significant",
     "    p = sum(math.comb(n, k) for k in range(higher, n + 1)) / 2 ** n",
     "    p = math.comb(n, higher) / 2 ** n"),
    ("the demonstrations are scored as if they were a policy",
     "    for (task, policy), g in frame[frame.policy != \"teleoperated\"].groupby([\"task\", \"policy\"]):",
     "    for (task, policy), g in frame.groupby([\"task\", \"policy\"]):"),
]


def run_suite():
    out = subprocess.run([sys.executable, str(HERE / "test_retry_rate.py")],
                         cwd=HERE, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if "checks passed" in l), "")
    if not line:
        # A mutation that makes the suite crash outright is caught, not survived.
        return 99, (out.stderr.strip().splitlines() or ["no output"])[-1][:160]
    return int(line.split(",")[1].strip().split()[0]), line.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the mutations without applying any")
    args = ap.parse_args()

    if args.list:
        for i, (what, _, _) in enumerate(MUTATIONS, 1):
            print(f"{i:2}. {what}")
        return 0

    baseline_failed, baseline_line = run_suite()
    if baseline_failed:
        raise SystemExit(f"the suite is already failing ({baseline_line}); fix that first, "
                         "or a surviving mutation cannot be told apart from it")
    print(f"baseline: {baseline_line}\n")

    path = HERE / TARGET
    survivors = []
    for i, (what, old, new) in enumerate(MUTATIONS, 1):
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            survivors.append((what, "PATTERN NOT FOUND - this mutation no longer applies"))
            print(f"{i:2}. {'SKIP':8} {what}")
            continue
        path.write_text(original.replace(old, new), encoding="utf-8")
        try:
            failed, line = run_suite()
        finally:
            path.write_text(original, encoding="utf-8")
        if failed:
            print(f"{i:2}. {'caught':8} {what}  ({failed} failed)")
        else:
            survivors.append((what, line))
            print(f"{i:2}. {'SURVIVED':8} {what}")

    print()
    if survivors:
        print(f"{len(survivors)} of {len(MUTATIONS)} mutations were not caught:\n")
        for what, detail in survivors:
            print(f"  {what}\n    {detail}")
        print("\nEach survivor is a hole in test_retry_rate.py at exactly that behaviour.")
        return 1
    print(f"all {len(MUTATIONS)} mutations caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
