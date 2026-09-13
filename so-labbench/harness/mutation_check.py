# Do the tests actually catch anything?
#
#   python mutation_check.py            # break the code twelve ways, each must be caught
#   python mutation_check.py --list     # what it breaks, without touching the files
#
# A test suite that passes is not evidence of anything by itself: tests written after the
# code tend to assert what the code does rather than what it should do, and a test can decay
# into one as the code changes underneath it. So each mutation below is a defect that either
# was real here or would be easy to introduce, and every one of them must make at least one
# check fail. If a mutation survives, the suite has a hole exactly there.
#
# Each file is restored from its own bytes, not from git, so this is safe to run on a dirty
# tree. It refuses to start if the suite is already failing, because then a survivor would be
# unreadable.
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# (file, what it breaks, exact text, replacement, which suites should notice)
MUTATIONS = [
    ("score_rq1.py", "the v1.4 grasp window reverts to the longest closure",
     "window = select_grasp_window(runs, lift, stable_f)",
     "window = max([(o, n) for o, n in runs if n >= stable_f], default=None, key=lambda r: r[1])",
     "scorer demos"),
    ("labbench.py", "the stable threshold is hardcoded to 15 frames again",
     "def stable_frames(hz: float) -> int:\n    return _frames_for(STABLE_S, hz)",
     "def stable_frames(hz: float) -> int:\n    return 15",
     "scorer"),
    ("labbench.py", "the closure length comparison flips from >= to >",
     "candidates = [(o, n) for o, n in runs if n >= min_frames",
     "candidates = [(o, n) for o, n in runs if n > min_frames",
     "scorer demos"),
    ("labbench.py", "a closure that runs to the last frame is one frame short",
     "runs.append((start, len(grip) - start))",
     "runs.append((start, len(grip) - start - 1))",
     "scorer"),
    ("labbench.py", "the lift is measured over the whole episode, not the grasp",
     "segment = lift[onset:onset + length]\n    return float(max(segment) - lift[onset])",
     "return float(max(lift) - lift[onset])",
     "scorer demos"),
    ("labbench.py", "frames round instead of ceil, so 0.49s can satisfy a 0.5s threshold",
     "return max(1, math.ceil(seconds * hz - 1e-9))",
     "return max(1, round(seconds * hz))",
     "scorer"),
    ("labbench.py", "the v3.0 dataset layout falls back to v2.1",
     'return DATASETS / run / "data" / "chunk-000" / f"file-{episode:03d}.parquet"',
     'return DATASETS / run / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"',
     "paths"),
    ("labbench.py", "an absent dataset is guessed to be v2.1 instead of raising",
     'raise FileNotFoundError(f"no dataset \'{run}\' under {DATASETS}")',
     'return {"codebase_version": "v2.1"}',
     "paths"),
    ("labbench.py", "the Wilson interval loses its continuity term",
     "centre = (p + z * z / (2 * n)) / d",
     "centre = p",
     "scorer"),
    ("score_rq1.py", "the lift threshold doubles",
     "lift_ok = grasp and gain >= LIFT_GAIN",
     "lift_ok = grasp and gain >= LIFT_GAIN * 2",
     "scorer demos"),
    ("score_rq1.py", "instability counts reversals instead of a rate",
     "rev_per_s = rev / episode_sec",
     "rev_per_s = rev",
     "scorer"),
    ("rqp1_results.py", "suboptimal trials count as successes",
     'k = sum(l == "successful" for l in valid)',
     'k = sum(l in ("successful", "suboptimal") for l in valid)',
     "rqp1"),
    ("rqp1_results.py", "invalid trials stay in the denominator as failures",
     'valid = [l for l in labs if l in ("successful", "suboptimal", "failure")]',
     "valid = [l for l in labs if l is not None]",
     "rqp1"),
    ("blind_review.py", "review ids follow the recording order instead of being shuffled",
     "random.Random(SEED).shuffle(order)",
     "pass",
     "blind"),
    ("blind_review.py", "the sheet names the run, so the judge can see the policy",
     'lines += [f"| {review_id} | | |" for review_id, _ in assignment]',
     'lines += [f"| {review_id} | | {row[chr(34)]} |".replace(chr(34), "run_id") for review_id, row in assignment]',
     "blind"),
    # Wiring, not arithmetic. These are the failures a unit test cannot see, because each
    # tool is individually right and they disagree with each other.
    ("labbench.py", "the run-id grammar loses its study field, so filters match nothing",
     'RUN_ID_RE = re.compile(r"^eval_(?P<study>[^_]+)_(?P<condition>[^_]+)_(?P<model>.+)_(?P<index>\\d+)$")',
     'RUN_ID_RE = re.compile(r"^eval_(?P<study>x)_(?P<condition>[^_]+)_(?P<model>.+)_(?P<index>\\d+)$")',
     "pipeline"),
    ("labbench.py", "outputs stop being named for the study, so conditions overwrite each other",
     'stem = "_".join(parts + [name]) if parts else name',
     "stem = name",
     "pipeline"),
    ("labbench.py", "the manifest's episode length is ignored, so rates are assumed",
     'return {r["run_id"]: float(r["episode_sec"]) for r in read_manifest() if r.get("episode_sec")}',
     "return {}",
     "pipeline"),
    ("labbench.py", "the manifest filter is dropped, so one condition's tools see every study",
     'if study and parsed.get("study") != study:\n            continue',
     "if False:\n            continue",
     "pipeline"),
]


def run_suite(suites: str):
    out = subprocess.run([sys.executable, str(HERE / "test_harness.py"), *suites.split()],
                         cwd=HERE, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if "checks passed" in l), "")
    failed = 0
    if line:
        failed = int(line.split(",")[1].strip().split()[0])
    return failed, line.strip() or out.stderr[-200:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the mutations without applying any")
    args = ap.parse_args()

    if args.list:
        for i, (f, what, _, _, suites) in enumerate(MUTATIONS, 1):
            print(f"{i:2}. {f:20} {what}   [{suites}]")
        return 0

    baseline_failed, baseline_line = run_suite("")
    if baseline_failed:
        raise SystemExit(f"the suite is already failing ({baseline_line}); fix that first, "
                         "or a surviving mutation cannot be told apart from it")
    print(f"baseline: {baseline_line}\n")

    survivors = []
    for i, (filename, what, old, new, suites) in enumerate(MUTATIONS, 1):
        path = HERE / filename
        original = path.read_text(encoding="utf-8")
        if original.count(old) != 1:
            survivors.append((filename, what, "PATTERN NOT FOUND - this mutation no longer applies"))
            print(f"{i:2}. {'SKIP':7} {filename:20} {what}")
            print(f"    the code it targets has changed; update or remove this mutation")
            continue
        path.write_text(original.replace(old, new), encoding="utf-8")
        try:
            failed, line = run_suite(suites)
        finally:
            path.write_text(original, encoding="utf-8")
        if failed:
            print(f"{i:2}. {'caught':7} {filename:20} {what}  ({failed} failed)")
        else:
            survivors.append((filename, what, line))
            print(f"{i:2}. {'SURVIVED':7} {filename:20} {what}")

    print()
    if survivors:
        print(f"{len(survivors)} of {len(MUTATIONS)} mutations were not caught:\n")
        for filename, what, detail in survivors:
            print(f"  {filename}: {what}\n    {detail}")
        print("\nEach survivor is a hole in test_harness.py at exactly that behaviour.")
        return 1
    print(f"all {len(MUTATIONS)} mutations caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
