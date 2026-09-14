# Do the analysis tests actually catch anything?
#
#   python mutation_check.py            # break the code, each way must be caught
#   python mutation_check.py --list     # what it breaks, without touching any file
#
# A passing suite is not evidence by itself. Tests written after the code tend to assert what
# the code does rather than what it should do, and both files covered here are unusually good
# at hiding a defect behind a plausible number:
#
#   retry_rate.py      produces a NEGATIVE result, and a broken counter is the easiest way to
#                      get one. A counter that flattened every trace to the same value would
#                      make every difference vanish and every test read calm.
#   vision_reliance.py exists to stop one denominator from manufacturing a conclusion. A bug
#                      in how it builds those denominators would defeat its only purpose, and
#                      would do it quietly, by making three numbers agree because they are
#                      secretly the same number.
#
# So each mutation below is a defect that was real here or would be easy to introduce, and
# every one must make at least one check fail. A survivor is a hole in the suites at exactly
# that behaviour. This is not hypothetical: the retry-rate suite as first written caught 8 of
# its 19, and the survivors included removing the division by duration, which alone would have
# inverted the finding's central claim.
#
# Each file is restored from its own bytes rather than from git, so this is safe on a dirty
# tree. It refuses to start if a suite is already failing, because a survivor could not then
# be told apart from the existing failure.
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SUITES = ["test_retry_rate.py", "test_vision_reliance.py", "test_input_attribution.py"]

# (what it breaks, exact text, replacement)
RETRY_MUTATIONS = [
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


# Which analysis file each mutation edits. Both suites run for every mutation, because a
# defect in one file can be the thing another file's test was relying on.
VISION_MUTATIONS = [
    ("the natural step is taken from one joint instead of all six",
     "    natural = float(np.mean([any_row[f\"natural_{j}\"] for j in JOINTS]))",
     "    natural = float(any_row[f\"natural_{JOINTS[0]}\"])"),
    ("the task-relative denominator is secretly the policy-relative one",
     "    sigma = float(np.mean([any_row[f\"sigma_{j}\"] for j in JOINTS]))",
     "    sigma = float(np.mean([any_row[f\"natural_{j}\"] for j in JOINTS]))"),
    ("the noise floor reads identity, which is zero by construction",
     "               noise_raw=round(float(d.delta_mean.get(\"resample\", 0.0)), 2))",
     "               noise_raw=round(float(d.delta_mean.get(\"identity\", 0.0)), 2))"),
    ("the percentage of demo spread is divided by the natural step instead",
     "            out[f\"{v}_sigma\"] = round(100 * raw / sigma, 1) if sigma else float(\"nan\")",
     "            out[f\"{v}_sigma\"] = round(100 * raw / natural, 1) if natural else float(\"nan\")"),
    ("the percentage loses its hundred, so everything looks a hundred times smaller",
     "            out[f\"{v}_sigma\"] = round(100 * raw / sigma, 1) if sigma else float(\"nan\")",
     "            out[f\"{v}_sigma\"] = round(raw / sigma, 1) if sigma else float(\"nan\")"),
    ("the policy-relative figure is recomputed after averaging instead of read per joint",
     "            out[f\"{v}_natural\"] = round(float(d.relative_mean[v]), 1)",
     "            out[f\"{v}_natural\"] = round(raw / natural, 1)"),
    ("the single views collapse into the all-cameras number",
     "VIEWS = [\"all_blank\", \"wrist_blank\", \"front_blank\", \"top_blank\"]",
     "VIEWS = [\"all_blank\"]"),
    ("task and policy are read the wrong way round out of the file name",
     "    task, policy = os.path.basename(path)[:-4].split(\"__\")",
     "    policy, task = os.path.basename(path)[:-4].split(\"__\")"),
]

# A preregistration is only as good as the code that applies it. Every one of these is a way
# to report a hypothesis as supported that the registered rule rejects, or to measure
# something other than what the perturbation's name promises.
JUDGE_MUTATIONS = [
    ("the threshold drops from 7 of 8 to 6",
     "THRESHOLD = 7 ", "THRESHOLD = 6 "),
    ("a tie counts as a win for H1",
     'lambda p, t: _value(df, p, t, "state_mean") > _value(df, p, t, "all_blank"))',
     'lambda p, t: _value(df, p, t, "state_mean") >= _value(df, p, t, "all_blank"))'),
    ("H3 is judged in the same direction as H1 instead of the reverse",
     'lambda p, t: _value(df, p, t, "all_blank") > _value(df, p, t, "state_mean"))',
     'lambda p, t: _value(df, p, t, "state_mean") > _value(df, p, t, "all_blank"))'),
    ("H2 needs only one of the two language perturbations to be quiet",
     "        return (_value(df, p, t, \"task_blank\") < floor\n                and _value(df, p, t, \"task_other\") < floor)",
     "        return (_value(df, p, t, \"task_blank\") < floor\n                or _value(df, p, t, \"task_other\") < floor)"),
    ("exactly twice the noise counts as quiet",
     '        return (_value(df, p, t, "task_blank") < floor',
     '        return (_value(df, p, t, "task_blank") <= floor'),
    ("H1 and H3 are read even when the state perturbation is broken",
     '    if out["H4"]:\n        out["H1"] = h1 >= THRESHOLD',
     '    if True:\n        out["H1"] = h1 >= THRESHOLD'),
    ("H4 passes if either policy passes",
     '    out["H4"] = all(n >= THRESHOLD for n in h4.values())',
     '    out["H4"] = any(n >= THRESHOLD for n in h4.values())'),
    ("an incomplete run still gets verdicts",
     "    if not complete:\n        # A verdict",
     "    if False:\n        # A verdict"),
    ("identity only has to be small, not exactly zero",
     '               if _value(df, p, t, "identity") != 0.0]',
     '               if abs(_value(df, p, t, "identity")) > 0.01]'),
]

ATTRIBUTION_MUTATIONS = [
    ("the two noise doses use the same scale, so H4 compares a push to itself",
     "        return state + 1.0 * np.asarray(sd) * np.asarray(z)",
     "        return state + 0.5 * np.asarray(sd) * np.asarray(z)"),
    ("state_mean leaves the state alone, so nothing is removed",
     "        return np.asarray(mean, dtype=np.float64).copy()",
     "        return state.copy()"),
    ("the gripper is always pushed up, past its end stop when already open",
     "        direction = -1.0 if state[GRIPPER] > mean[GRIPPER] else 1.0",
     "        direction = 1.0"),
    ("the gripper push is applied to the whole arm",
     "        out[GRIPPER] = state[GRIPPER] + direction * sd[GRIPPER]",
     "        out = state + direction * sd"),
    ("task_blank sends the real instruction",
     '    if name == "task_blank":\n        return ""',
     '    if name == "task_blank":\n        return instruction'),
    ("task_other sends the task's own instruction, so the swap is a no-op",
     "    return TASKS[TASK_ORDER[(i + 1) % len(TASK_ORDER)]][\"instruction\"]",
     "    return TASKS[TASK_ORDER[i]][\"instruction\"]"),
    ("a perturbation is silently dropped from the run",
     'TASK_PERTURBATIONS = ["task_blank", "task_other"]',
     'TASK_PERTURBATIONS = ["task_blank"]'),
]

MUTATIONS = ([("retry_rate.py", *m) for m in RETRY_MUTATIONS]
             + [("vision_reliance.py", *m) for m in VISION_MUTATIONS]
             + [("judge_input_attribution.py", *m) for m in JUDGE_MUTATIONS]
             + [("input_attribution.py", *m) for m in ATTRIBUTION_MUTATIONS])


def run_suite():
    failed, lines = 0, []
    for suite in SUITES:
        out = subprocess.run([sys.executable, str(HERE / suite)],
                             cwd=HERE, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if "checks passed" in l), "")
        if not line:
            # A mutation that makes a suite crash outright is caught, not survived.
            failed += 99
            lines.append(f"{suite}: " + (out.stderr.strip().splitlines() or ["no output"])[-1][:100])
            continue
        failed += int(line.split(",")[1].strip().split()[0])
        lines.append(f"{suite}: {line.strip()}")
    return failed, "; ".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the mutations without applying any")
    args = ap.parse_args()

    if args.list:
        for i, (f, what, _, _) in enumerate(MUTATIONS, 1):
            print(f"{i:2}. {f:20} {what}")
        return 0

    baseline_failed, baseline_line = run_suite()
    if baseline_failed:
        raise SystemExit(f"the suite is already failing ({baseline_line}); fix that first, "
                         "or a surviving mutation cannot be told apart from it")
    print(f"baseline: {baseline_line}\n")

    survivors = []
    for i, (filename, what, old, new) in enumerate(MUTATIONS, 1):
        path = HERE / filename
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            survivors.append((filename, what, "PATTERN NOT FOUND - this mutation no longer applies"))
            print(f"{i:2}. {'SKIP':8} {filename:20} {what}")
            continue
        path.write_text(original.replace(old, new), encoding="utf-8")
        try:
            failed, line = run_suite()
        finally:
            path.write_text(original, encoding="utf-8")
        if failed:
            print(f"{i:2}. {'caught':8} {filename:20} {what}")
        else:
            survivors.append((filename, what, line))
            print(f"{i:2}. {'SURVIVED':8} {filename:20} {what}")

    print()
    if survivors:
        print(f"{len(survivors)} of {len(MUTATIONS)} mutations were not caught:\n")
        for filename, what, detail in survivors:
            print(f"  {filename}: {what}\n    {detail}")
        print("\nEach survivor is a hole in the suites at exactly that behaviour.")
        return 1
    print(f"all {len(MUTATIONS)} mutations caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
