# Regression tests for the measurement code. No pytest, no extra install:
#   python test_harness.py            # all suites
#   python test_harness.py scorer     # one suite by name
#
# Why this file exists: every scorer change so far was checked by running it once and
# reading the output. That catches a change that is obviously wrong and nothing else.
# The v1.4 window-selection change in particular is invisible on the pilot trials (none
# of them closed the gripper), so nothing in the repository would have noticed if it had
# been broken. Each test below states the behaviour it pins down and why it matters.
#
# Synthetic episodes are built here rather than read from disk, so the expected answer is
# known exactly instead of eyeballed from a plot.
import csv
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

FAILURES = []
PASSED = 0
_CURRENT_SUITE = None


def check(condition, message):
    global PASSED
    if condition:
        PASSED += 1
    else:
        FAILURES.append(f"{_CURRENT_SUITE}: {message}")


def check_close(got, want, tol, message):
    check(abs(got - want) <= tol, f"{message} (got {got!r}, want {want!r} +/- {tol})")


# --------------------------------------------------------------------------------------
# synthetic episodes


def make_episode(path: Path, grip, lift=None, pan=None, action_grip=None):
    """Write a one-episode parquet with the columns the scorers read.

    grip:        gripper position per frame (100 open, 0 closed) - observation.state[:, 5]
    lift:        shoulder_lift per frame                          - observation.state[:, 1]
    pan:         shoulder_pan per frame                           - observation.state[:, 0]
    action_grip: commanded gripper; defaults to the observed one (a servo that follows).
    """
    grip = np.asarray(grip, dtype=np.float32)
    n = len(grip)
    lift = np.zeros(n, dtype=np.float32) if lift is None else np.asarray(lift, dtype=np.float32)
    pan = np.zeros(n, dtype=np.float32) if pan is None else np.asarray(pan, dtype=np.float32)
    action_grip = grip if action_grip is None else np.asarray(action_grip, dtype=np.float32)
    zeros = np.zeros(n, dtype=np.float32)
    state = np.stack([pan, lift, zeros, zeros, zeros, grip], axis=1)
    action = np.stack([pan, lift, zeros, zeros, zeros, action_grip], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"observation.state": list(state), "action": list(action)}).to_parquet(path)
    return path


def constant(n, value):
    return np.full(n, float(value), dtype=np.float32)


# --------------------------------------------------------------------------------------
# suite: scorer (score_rq1.score_episode)


def suite_scorer(tmp: Path):
    import score_rq1

    # 720 frames over 24 s = 30 Hz, so stable = 15 frames and hold = 30 frames.
    N, SEC = 720, 24.0
    OPEN, CLOSED = 90.0, 20.0

    def score(grip, lift=None, pan=None, sec=SEC):
        p = make_episode(tmp / f"ep_{len(list(tmp.glob('*.parquet')))}.parquet", grip, lift, pan)
        return score_rq1.score_episode(p, sec)

    r = score(constant(N, OPEN))
    check(r["true_hz"] == 30.0, f"control rate is frames/seconds, not the dataset's fps field (got {r['true_hz']})")
    check(r["stable_f"] == 15 and r["hold_f"] == 30, f"0.5s/1.0s convert to 15/30 frames at 30 Hz (got {r['stable_f']}/{r['hold_f']})")
    check(not r["grasp"] and not r["success"], "an episode that never closes is not a grasp")
    check(r["n_closures"] == 0, "no closures counted when the gripper stays open")
    check(r["taxonomy"] == "approach_failure", f"open + motionless is approach_failure (got {r['taxonomy']})")

    # descended but never closed -> grasp_failure, which is the failure ACT and SmolVLA
    # actually produced in the pilot; if this branch breaks, every pilot trial is mislabelled.
    lift = np.concatenate([constant(200, 0), np.linspace(0, -40, 100), constant(420, -40)])
    r = score(constant(N, OPEN), lift=lift)
    check(r["approach"], "a 40-unit descent counts as an approach")
    check(r["taxonomy"] == "grasp_failure", f"descend + no closure is grasp_failure (got {r['taxonomy']})")

    # a clean pick-up
    grip = np.concatenate([constant(300, OPEN), constant(200, CLOSED), constant(220, OPEN)])
    lift = np.concatenate([constant(300, -40), np.linspace(-40, 10, 200), constant(220, 10)])
    r = score(grip, lift=lift)
    check(r["success"] and r["taxonomy"] == "success", f"closure + lift + hold is success (got {r['taxonomy']})")
    check(r["n_closures"] == 1, f"one closed stretch counted (got {r['n_closures']})")
    check_close(r["lift_gain"], 50.0, 1.0, "lift gain is measured from the grasp onset")

    # closure long enough to be stable but not to be held
    grip = np.concatenate([constant(300, OPEN), constant(20, CLOSED), constant(400, OPEN)])
    lift = np.concatenate([constant(300, -40), np.linspace(-40, 10, 20), constant(400, 10)])
    r = score(grip, lift=lift)
    check(r["grasp"] and not r["hold"], "20 frames at 30 Hz is stable (>=15) but not held (<30)")
    check(r["taxonomy"] == "hold_failure", f"stable + lifted + not held is hold_failure (got {r['taxonomy']})")

    # closed and held but never lifted
    grip = np.concatenate([constant(300, OPEN), constant(200, CLOSED), constant(220, OPEN)])
    r = score(grip, lift=constant(N, -40))
    check(r["grasp"] and r["hold"] and not r["lift"], "a held closure with no rise is not a lift")
    check(r["taxonomy"] == "lift_failure", f"held but not lifted is lift_failure (got {r['taxonomy']})")

    # THE v1.4 REGRESSION. 27 of the 60 demonstrations start with the gripper still closed
    # from the previous recording. That idle stretch is longer than the real grasp, so the
    # v1.3 rule ("longest closed run") measured the lift over a window that was never a
    # grasp and reported lift_failure on a successful pick-up.
    grip = np.concatenate([constant(300, CLOSED), constant(100, OPEN), constant(60, CLOSED), constant(260, OPEN)])
    lift = np.concatenate([constant(400, -40), np.linspace(-40, 10, 60), constant(260, 10)])
    r = score(grip, lift=lift)
    check(r["n_closures"] == 2, f"carry-over and grasp are two separate closures (got {r['n_closures']})")
    check(r["window_onset"] == 400, f"the window is the closure that lifted, not the longest one (got onset {r['window_onset']}, expected 400)")
    check(r["success"], "a real grasp after a carry-over closure still scores as success")
    check_close(r["lift_gain"], 50.0, 1.0, "the lift is measured over the grasp, not the carry-over")

    # ... and the carry-over alone must not become a success by itself
    grip = np.concatenate([constant(300, CLOSED), constant(420, OPEN)])
    r = score(grip, lift=constant(N, -40))
    check(not r["success"], "a carry-over closure with no lift is not a success")

    # thresholds follow the measured rate: the same trace judged at half the rate needs
    # half the frames. This is the v1.3 fix and the reason a slow policy is not penalised.
    grip = np.concatenate([constant(300, OPEN), constant(20, CLOSED), constant(400, OPEN)])
    lift = np.concatenate([constant(300, -40), np.linspace(-40, 10, 20), constant(400, 10)])
    fast = score(grip, lift=lift, sec=24.0)   # 30 Hz -> stable 15, hold 30
    slow = score(grip, lift=lift, sec=48.0)   # 15 Hz -> stable  8, hold 15
    check(fast["stable_f"] == 15 and slow["stable_f"] == 8, f"stable frames track the rate ({fast['stable_f']} vs {slow['stable_f']})")

    # "At least 0.5 s" must never be satisfied by less than 0.5 s. At 14.9 Hz - a rate the
    # pilot actually ran at - half a second is 7.45 frames, so 7 frames is 0.47 s and must
    # not count. Rounding to nearest would accept it, and Python rounds halves to even, so
    # the boundary would also behave differently at 29 Hz than at 31 Hz.
    import labbench
    check(labbench.stable_frames(14.9) == 8, f"0.5s at 14.9 Hz is 8 frames, not 7 (got {labbench.stable_frames(14.9)})")
    check(labbench.stable_frames(29.0) == 15, f"0.5s at 29 Hz is 15 frames, not 14 (got {labbench.stable_frames(29.0)})")
    check(labbench.stable_frames(22.2) == 12, f"0.5s at 22.2 Hz is 12 frames, not 11 (got {labbench.stable_frames(22.2)})")
    check(labbench.stable_frames(30.0) == 15, "an exact multiple is not rounded up past itself")
    check(labbench.hold_frames(14.9) == 15, f"1.0s at 14.9 Hz is 15 frames (got {labbench.hold_frames(14.9)})")
    check(labbench.stable_frames(0.5) == 1, "a threshold never drops below one frame")
    # and the same boundary through the scorer, not just the helper
    n_frames, secs = 447, 30.0                       # 14.9 Hz
    for length, want in [(7, False), (8, True)]:
        grip_b = np.concatenate([constant(200, OPEN), constant(length, CLOSED), constant(n_frames - 200 - length, OPEN)])
        r = score(grip_b, sec=secs)
        check(r["grasp"] == want, f"at 14.9 Hz a {length}-frame closure is {'a' if want else 'not a'} grasp")
    check(not fast["hold"] and slow["hold"], "the same 20-frame closure is held at 15 Hz but not at 30 Hz")

    # a closure at the very end of the episode must not index past the array
    grip = np.concatenate([constant(660, OPEN), constant(60, CLOSED)])
    lift = np.concatenate([constant(660, -40), np.linspace(-40, 10, 60)])
    r = score(grip, lift=lift)
    check(r["success"], "a grasp that runs to the final frame is scored, not dropped")

    # Boundary frames, because a closure length is compared with >= against a threshold
    # and one frame either way changes the verdict. A trailing run is measured by a
    # different line of code than an interior one, so both are pinned here.
    for length, want_grasp, want_hold, where in [(14, False, False, "interior"), (15, True, False, "interior"),
                                                 (29, True, False, "interior"), (30, True, True, "interior")]:
        grip = np.concatenate([constant(300, OPEN), constant(length, CLOSED), constant(N - 300 - length, OPEN)])
        r = score(grip)
        check(r["grasp"] == want_grasp and r["hold"] == want_hold,
              f"a {where} closure of exactly {length} frames: grasp={want_grasp}, hold={want_hold} "
              f"(got grasp={r['grasp']}, hold={r['hold']})")
    for length, want_hold in [(29, False), (30, True)]:
        grip = np.concatenate([constant(N - length, OPEN), constant(length, CLOSED)])
        r = score(grip)
        check(r["hold"] == want_hold,
              f"a closure of exactly {length} frames ending at the last frame: hold={want_hold} (got {r['hold']})")

    # a closure too short to be stable is still counted and still reported as a grasp failure
    grip = np.concatenate([constant(300, OPEN), constant(5, CLOSED), constant(415, OPEN)])
    lift = np.concatenate([constant(200, 0), np.linspace(0, -40, 100), constant(420, -40)])
    r = score(grip, lift=lift)
    check(r["n_closures"] == 1 and not r["grasp"], "a 5-frame closure is counted but is not a grasp")
    check(r["taxonomy"] == "grasp_failure", f"a too-short closure after a descent is grasp_failure (got {r['taxonomy']})")
    check(r["lift_gain"] == 0.0, "no lift is reported when there is no qualifying window")

    # instability is a rate, not a count: the same oscillation over a longer episode
    # must not become "more unstable" just because there are more samples.
    osc = np.tile([0.0, 5.0], N // 2)
    a = score(constant(N, OPEN), pan=osc, sec=24.0)
    b = score(constant(N, OPEN), pan=osc, sec=48.0)
    check(a["reversals"] == b["reversals"], "the raw reversal count is the same trace either way")
    check(a["rev_per_s"] > b["rev_per_s"], "reversals per second halve when the same trace takes twice as long")

    # The taxonomy resolves by stage before it considers instability, so a trial that
    # descended is classified by what it did there even if it also thrashed. Only a trial
    # that never descended and moved a great deal reaches the instability branch. Pinned
    # because the ordering is a choice, not an accident, and a reordering would silently
    # relabel every thrashing approach.
    osc_big = np.tile([0.0, 40.0], N // 2)                      # a lot of travel, many reversals
    r = score(constant(N, OPEN), pan=osc_big, lift=constant(N, 0.0))
    check(not r["approach"], "the oscillation fixture never descends")
    check(r["moved"] > 25, f"and it moves far more than the travel threshold (got {r['moved']})")
    check(r["taxonomy"] == "policy_instability", f"no descent + large travel + reversals is instability (got {r['taxonomy']})")
    lift_down = np.concatenate([constant(200, 0), np.linspace(0, -40, 100), constant(420, -40)])
    r = score(constant(N, OPEN), pan=osc_big, lift=lift_down)
    check(r["taxonomy"] == "grasp_failure",
          f"the same thrashing WITH a descent is a grasp failure, because the stage says more (got {r['taxonomy']})")

    # "Held" means the grasp persisted, not that the gripper was closed for that long in
    # total somewhere in the episode. Two short closures do not add up to one hold.
    grip = np.concatenate([constant(200, OPEN), constant(20, CLOSED), constant(60, OPEN),
                           constant(20, CLOSED), constant(N - 300, OPEN)])
    lift = np.concatenate([constant(200, -40), np.linspace(-40, 10, 20), constant(60, 10),
                           np.linspace(10, 60, 20), constant(N - 300, 60)])
    r = score(grip, lift=lift)
    check(r["n_closures"] == 2, "two closures of 20 frames each")
    check(r["grasp"] and not r["hold"], "each is stable (>=15) but neither is held (<30), and they do not sum")

    # Wilson interval against hand-computed values (z=1.96)
    lo, hi = score_rq1.wilson_ci(0, 5)
    check_close(lo, 0.0, 1e-9, "Wilson lower bound at 0/5")
    check_close(hi, 0.4345, 1e-3, "Wilson upper bound at 0/5")
    lo, hi = score_rq1.wilson_ci(0, 10)
    check_close(hi, 0.2775, 1e-3, "Wilson upper bound at 0/10")
    lo, hi = score_rq1.wilson_ci(5, 10)
    check_close(lo, 0.2366, 1e-3, "Wilson lower bound at 5/10")
    check_close(hi, 0.7634, 1e-3, "Wilson upper bound at 5/10")
    check(score_rq1.wilson_ci(0, 0) == (0.0, 0.0), "an empty sample has no interval rather than a crash")


# --------------------------------------------------------------------------------------
# suite: paths (labbench layout resolution)


def suite_paths(tmp: Path):
    """The RQ0/RQ1 recordings are dataset format v2.1 and the new environment writes v3.0.
    One wrong branch here silently scores nothing, or scores the wrong episode."""
    root = tmp / "datasets"
    for name, version in [("v2run", "v2.1"), ("v3run", "v3.0")]:
        (root / name / "meta").mkdir(parents=True)
        (root / name / "meta" / "info.json").write_text(json.dumps({"codebase_version": version}))
    (root / "noinfo").mkdir(parents=True)

    os.environ["LABBENCH_DATASETS"] = str(root)
    import importlib
    import labbench
    importlib.reload(labbench)

    check(labbench.dataset_version("v2run") == "v2.1", "v2.1 is read from meta/info.json")
    check(labbench.dataset_version("v3run") == "v3.0", "v3.0 is read from meta/info.json")
    check(labbench.dataset_exists("v2run") and not labbench.dataset_exists("noinfo"),
          "a dataset without meta/info.json does not exist as far as the harness is concerned")
    # It used to assume v2.1 here and hand back a path that could never exist, so a missing
    # recording and a v3.0 recording both surfaced as the same silent "missing".
    try:
        labbench.dataset_version("noinfo")
        check(False, "an absent dataset raises rather than guessing a layout")
    except FileNotFoundError as exc:
        check("noinfo" in str(exc), "and the error names the dataset it could not find")

    p2 = labbench.episode_parquet("v2run").relative_to(root).as_posix()
    p3 = labbench.episode_parquet("v3run").relative_to(root).as_posix()
    check(p2 == "v2run/data/chunk-000/episode_000000.parquet", f"v2.1 parquet path (got {p2})")
    check(p3 == "v3run/data/chunk-000/file-000.parquet", f"v3.0 parquet path (got {p3})")

    v2 = labbench.episode_video("v2run", camera="fixed").relative_to(root).as_posix()
    v3 = labbench.episode_video("v3run", camera="front").relative_to(root).as_posix()
    check(v2 == "v2run/videos/chunk-000/observation.images.fixed/episode_000000.mp4", f"v2.1 video path (got {v2})")
    check(v3 == "v3run/videos/observation.images.front/chunk-000/file-000.mp4", f"v3.0 video path (got {v3})")

    del os.environ["LABBENCH_DATASETS"]
    importlib.reload(labbench)


# --------------------------------------------------------------------------------------
# suite: commanded vs achieved


def suite_command(tmp: Path):
    """"The policy never asked" and "the robot did not obey" need opposite fixes, so the
    tool that separates them has to keep them apart."""
    import command_vs_achieved as cva

    check(cva.longest_run([False, False, False]) == 0, "no closed frames is a zero-length run")
    check(cva.longest_run([True, True, False, True]) == 2, "the longest run, not the total")
    check(cva.longest_run([True, True, True]) == 3, "a run that fills the array")
    check(cva.longest_run([]) == 0, "an empty trace does not crash")

    # A demonstration where the leader commanded closed and the follower stayed open is
    # real: episode 7 of the training set does exactly this for its whole length.
    n = 300
    commanded = constant(n, 24.0)
    observed = constant(n, 59.0)
    check(commanded.min() < cva.CLOSE_T <= observed.min(), "the fixture is a commanded-but-not-executed closure")

    # the primary outcome: a commanded closure shorter than CLOSE_S does not count
    hz = 20.0
    short = np.concatenate([constant(280, 90.0), constant(20, 24.0)])   # 20 frames = 1.0 s
    brief = np.concatenate([constant(295, 90.0), constant(5, 24.0)])    # 5 frames  = 0.25 s
    check(cva.longest_run(short < cva.CLOSE_T) / hz >= cva.CLOSE_S, "a 1.0 s commanded closure counts")
    check(cva.longest_run(brief < cva.CLOSE_T) / hz < cva.CLOSE_S, "a 0.25 s commanded closure does not count")


# --------------------------------------------------------------------------------------
# suite: RQ-P1 results


def suite_rqp1(tmp: Path):
    """The reproduction's headline number. Strict success, invalid trials excluded from the
    denominator rather than counted as failures, and the delta measured against the
    published rate for the same task."""
    import rqp1_results

    work = tmp / "rqp1"
    (work / "armnetbench").mkdir(parents=True)
    (work / "harness").mkdir(parents=True)
    for name in ["rqp1_results.py", "labbench.py", "labbench_config.json"]:
        shutil.copy(HERE / name, work / "harness" / name)
    shutil.copy(HERE.parent / "armnetbench" / "reference_results.csv", work / "armnetbench" / "reference_results.csv")

    rows = ["run_id,timestamp,model,condition,start_position,repo_id,episode_sec,video_path"]
    sheet = ["| run_id | round | position | policy | label | notes |", "|---|---|---|---|---|---|"]
    # act: 6 successful, 2 suboptimal, 1 failure, 1 invalid  -> 6/9
    # smolvla: 1 successful, 9 failure                       -> 1/10
    plan = {"act": ["successful"] * 6 + ["suboptimal"] * 2 + ["failure", "invalid"],
            "smolvla": ["successful"] + ["failure"] * 9}
    for policy, labels in plan.items():
        for i, label in enumerate(labels, 1):
            run = f"eval_rqp1_eyedrops_{policy}_{i}"
            rows.append(f"{run},t,{policy},eyedrops,P1,x,45,v")
            sheet.append(f"| {run} | {i} | P1 | {policy} | {label} | |")
    (work / "harness" / "rq1_manifest.csv").write_text("\n".join(rows) + "\n")
    (work / "harness" / "rqp1_human_labels.md").write_text("\n".join(sheet) + "\n", encoding="utf-8")

    import subprocess
    results_csv = work / "harness" / "rqp1_eyedrops_results.csv"
    out = subprocess.run([sys.executable, "rqp1_results.py", "eyedrops"], cwd=work / "harness",
                         capture_output=True, text=True)
    check(out.returncode == 0, f"the results script runs ({out.stderr[-300:]})")
    check(results_csv.exists(), "the results file is named for the study and condition, "
                                "so a second condition does not overwrite the first")
    result = pd.read_csv(results_csv).set_index("policy")

    check(result.loc["act", "n"] == 9, f"an invalid trial leaves the denominator (got n={result.loc['act', 'n']})")
    check(result.loc["act", "invalid"] == 1, "the invalid trial is still reported")
    check(result.loc["act", "successful"] == 6, "only 'successful' counts as success")
    check(result.loc["act", "suboptimal"] == 2, "suboptimal trials are reported separately, not as successes")
    check_close(result.loc["act", "rate"], 6 / 9, 1e-3, "act rate is successes over valid trials")
    check_close(result.loc["smolvla", "rate"], 0.1, 1e-3, "smolvla rate")

    # deltas against the published eye-drops numbers: act 19/30, smolvla 7/30
    check_close(result.loc["act", "ref_rate"], 19 / 30, 1e-3, "published act rate for this task")
    check_close(result.loc["smolvla", "ref_rate"], 7 / 30, 1e-3, "published smolvla rate for this task")
    check_close(result.loc["act", "delta"], 6 / 9 - 19 / 30, 1e-3, "act delta")
    check_close(result.loc["smolvla", "delta"], 0.1 - 7 / 30, 1e-3, "smolvla delta")

    check(result.loc["act", "judging"].startswith("unblinded"),
          "a per-run sheet is reported as unblinded, because it names the policy on every row")

    # an unlabelled trial is reported, not silently treated as a failure
    sheet_partial = [line for line in sheet if not line.startswith("| eval_rqp1_eyedrops_smolvla_9")]
    (work / "harness" / "rqp1_human_labels.md").write_text("\n".join(sheet_partial) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "rqp1_results.py", "eyedrops"], cwd=work / "harness", capture_output=True, text=True)
    result = pd.read_csv(results_csv).set_index("policy")
    check(result.loc["smolvla", "unlabelled"] == 1, "a missing label is counted as unlabelled")
    check(result.loc["smolvla", "n"] == 9, "an unlabelled trial is not scored as a failure")

    # The blind path must give the same answer as the unblinded one for the same judgements,
    # and must take precedence: the point of the key file is that it is the only join.
    shutil.copy(HERE / "blind_review.py", work / "harness" / "blind_review.py")
    shutil.copy(HERE / "labbench.py", work / "harness" / "labbench.py")
    shutil.copy(HERE / "labbench_config.json", work / "harness" / "labbench_config.json")
    runs = [line.split("|")[1].strip() for line in sheet if line.startswith("| eval_")]
    labels_by_run = {line.split("|")[1].strip(): line.split("|")[5].strip() for line in sheet if line.startswith("| eval_")}
    order = list(runs)[::-1]          # any order; the key is what makes it recoverable
    with (work / "harness" / "rqp1_eyedrops_review_key.csv").open("w", newline="", encoding="ascii") as f:
        w = csv.writer(f); w.writerow(["review_id", "run_id", "model", "start_position", "seed"])
        for i, run in enumerate(order, 1):
            w.writerow([f"R{i:03d}", run, run.split("_")[3], "P1", 1])
    blind_sheet = ["| review_id | label | notes |", "|---|---|---|"]
    blind_sheet += [f"| R{i:03d} | {labels_by_run[run]} | |" for i, run in enumerate(order, 1)]
    (work / "harness" / "rqp1_eyedrops_labels.md").write_text("\n".join(blind_sheet) + "\n", encoding="utf-8")
    out = subprocess.run([sys.executable, "rqp1_results.py", "eyedrops"], cwd=work / "harness", capture_output=True, text=True)
    check(out.returncode == 0, f"the results script runs against a blind review set ({out.stderr[-300:]})")
    blind = pd.read_csv(results_csv).set_index("policy")
    check(blind.loc["act", "judging"] == "blind", "the blind key takes precedence over the per-run sheet")
    check(blind.loc["act", "successful"] == 6 and blind.loc["act", "n"] == 9,
          "the same judgements through the key give the same counts")
    check_close(blind.loc["smolvla", "rate"], 0.1, 1e-3, "and the same rates")


# --------------------------------------------------------------------------------------
# suite: subsets and configs


def suite_config(tmp: Path):
    import importlib
    import rq0_subsets

    subsets = json.loads((HERE / "rq0_subsets.json").read_text(encoding="ascii"))
    arms = subsets["arms"]
    check(set(arms) == {"60all", "23complete", "23random"}, "the three RQ0 arms are present")
    check(len(arms["60all"]["episodes"]) == 60, "the full arm has 60 episodes")
    check(len(arms["23complete"]["episodes"]) == len(arms["23random"]["episodes"]) == 23,
          "the two subset arms are the same size, which is what makes the random arm a size control")
    check(set(arms["23complete"]["composition"]) == {"complete_pickup"},
          "the complete arm contains only complete pick-ups")
    check(len(set(arms["23random"]["episodes"])) == 23, "the random draw has no repeats")
    check(len(set(arms["23complete"]["episodes"]) & set(arms["23random"]["episodes"])) < 23,
          "the two arms are not the same set")

    quality = pd.read_csv(HERE / "so101_pick_remote_quality.csv")
    complete = sorted(quality.loc[quality.verdict == "complete_pickup", "ep"])
    check(arms["23complete"]["episodes"] == complete,
          "the complete arm is derived from the quality audit, not typed by hand")

    # re-deriving the subsets must reproduce the file, or the arms are not reproducible
    import random
    rng = random.Random(subsets["seed"])
    redrawn = sorted(rng.sample(list(quality.ep), 23))
    check(redrawn == arms["23random"]["episodes"], "the seeded random arm reproduces exactly")

    for name in ["labbench_config.json", "labbench_config_rqp1.json"]:
        cfg = json.loads((HERE / name).read_text(encoding="utf-8"))
        check("paths" in cfg and "cameras" in cfg and "policies" in cfg, f"{name} has the expected top-level sections")
        for cam, spec in cfg["cameras"].items():
            if cam.startswith("_"):
                continue
            missing = {"type", "index_or_path", "width", "height", "fps"} - set(spec)
            check(not missing, f"{name}: camera {cam} is missing {missing}")
        for policy, path in cfg["policies"].items():
            if policy.startswith("_"):
                continue
            is_hub_id = "/" in path and "\\" not in path and ":" not in path
            check(is_hub_id or Path(path).exists(), f"{name}: policy {policy} resolves ({path})")

    # the working camera is selected by name, so every config must name the one its tools use
    rqp1 = json.loads((HERE / "labbench_config_rqp1.json").read_text(encoding="utf-8"))
    os.environ["LABBENCH_CONFIG"] = str(HERE / "labbench_config_rqp1.json")
    os.environ["LABBENCH_CAMERA"] = "wrist"
    import labbench
    importlib.reload(labbench)
    # Read the expected values from the config rather than writing them here: the camera
    # assignment changes when hardware does, and a test that hardcodes it fails for the
    # wrong reason.
    wrist = rqp1["cameras"]["wrist"]
    check(labbench.CAMERA_INDEX == wrist["index_or_path"] and labbench.FRAME_W == wrist["width"]
          and labbench.FRAME_H == wrist["height"],
          f"LABBENCH_CAMERA resolves to that camera's entry in the active config "
          f"(got index {labbench.CAMERA_INDEX} at {labbench.FRAME_W}x{labbench.FRAME_H}, "
          f"config says {wrist['index_or_path']} at {wrist['width']}x{wrist['height']})")
    check(labbench.CAMERA_NAMES == [n for n in rqp1["cameras"] if not n.startswith("_")],
          f"and every camera in the config is listed (got {labbench.CAMERA_NAMES})")
    del os.environ["LABBENCH_CONFIG"], os.environ["LABBENCH_CAMERA"]
    importlib.reload(labbench)
    check(labbench.CAMERA_NAME == "fixed", "the default camera is the RQ0/RQ1 one")


# --------------------------------------------------------------------------------------

def suite_demos(tmp: Path):
    """The strongest available regression test: score all 60 recorded demonstrations and
    require the verdicts to match the independent quality audit exactly. This is the
    property that justified scorer v1.4 (60/60 agreement); locking it in means a future
    change to the window rule cannot quietly undo it. Skipped when the dataset is absent,
    so a third party without the recordings can still run every other suite."""
    import labbench
    import score_rq1

    dataset = labbench.DATASETS / "so101_pick_remote" / "data" / "chunk-000"
    quality_csv = HERE / "so101_pick_remote_quality.csv"
    if not dataset.exists() or not quality_csv.exists():
        print("             (skipped: so101_pick_remote not on this machine)")
        return

    quality = pd.read_csv(quality_csv).set_index("ep")
    # 722 frames per demonstration recorded at a nominal 30 Hz -> 24.07 s of wall clock.
    episode_sec = 722 / 30.0
    disagreements = []
    for ep in quality.index:
        parquet = dataset / f"episode_{ep:06d}.parquet"
        if not parquet.exists():
            continue
        r = score_rq1.score_episode(parquet, episode_sec)
        got_complete = bool(r["grasp"] and r["lift"])
        want_complete = quality.loc[ep, "verdict"] == "complete_pickup"
        if got_complete != want_complete:
            disagreements.append((ep, quality.loc[ep, "verdict"], r["taxonomy"], r["n_closures"], r["lift_gain"]))
    check(not disagreements,
          f"the scorer and the quality audit agree on all 60 demonstrations (disagree on {disagreements})")

    # and the count itself, so a change that flips two episodes in opposite directions is caught
    complete = sum(1 for ep in quality.index
                   if (dataset / f"episode_{ep:06d}.parquet").exists()
                   and (lambda r: r["grasp"] and r["lift"])(score_rq1.score_episode(dataset / f"episode_{ep:06d}.parquet", episode_sec)))
    check(complete == 23, f"23 of the 60 demonstrations are complete pick-ups (got {complete})")


def suite_blind(tmp: Path):
    """The label sheet decides the headline number. If it names the policy on each row,
    the person judging knows which one they are looking at, and their prior leaks into the
    result. These checks pin the property that makes the sheet blind."""
    import subprocess

    work = tmp / "blind"
    (work / "harness").mkdir(parents=True)
    for name in ["blind_review.py", "labbench.py", "labbench_config.json"]:
        shutil.copy(HERE / name, work / "harness" / name)
    datasets = work / "datasets"
    runs = [f"eval_rqp1_eyedrops_{p}_{i}" for i in (1, 2, 3) for p in ("act", "smolvla")]
    header = "run_id,timestamp,model,condition,start_position,repo_id,episode_sec,video_path"
    lines = [header]
    for run in runs:
        policy = run.split("_")[3]
        (datasets / run / "meta").mkdir(parents=True)
        (datasets / run / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.1"}))
        video = datasets / run / "videos" / "chunk-000" / "observation.images.front" / "episode_000000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"not a real video, but a real file")
        lines.append(f"{run},t,{policy},eyedrops,P1,x,45,v")
    (work / "harness" / "rq1_manifest.csv").write_text("\n".join(lines) + "\n")

    env = dict(os.environ, LABBENCH_DATASETS=str(datasets), LABBENCH_CAMERA="front")
    out = subprocess.run([sys.executable, "blind_review.py", "prepare", "rqp1", "eyedrops"],
                         cwd=work / "harness", capture_output=True, text=True, env=env)
    check(out.returncode == 0, f"prepare runs ({out.stderr[-300:]})")

    key = work / "harness" / "rqp1_eyedrops_review_key.csv"
    sheet = work / "harness" / "rqp1_eyedrops_labels.md"
    check(key.exists() and sheet.exists(), "prepare writes a key and a sheet")
    with key.open() as f:
        mapping = list(csv.DictReader(f))
    check(len(mapping) == len(runs), f"every trial gets a review id (got {len(mapping)})")
    check(len({m['review_id'] for m in mapping}) == len(runs), "review ids are unique")
    check(sorted(m["run_id"] for m in mapping) == sorted(runs), "every run is in the key exactly once")

    text = sheet.read_text(encoding="utf-8")
    for run in runs:
        check(run not in text, f"the sheet does not name the run ({run})")
    check("| act " not in text and "| smolvla " not in text, "the sheet has no policy column")
    check(text.count("| R") == len(runs), f"the sheet has one row per trial (got {text.count('| R')})")
    copied = list((work / "harness" / "review" / "rqp1_eyedrops").glob("R*.mp4"))
    check(len(copied) == len(runs), f"each trial's video is copied under its review id (got {len(copied)})")

    # The shuffle must actually shuffle. The comparison has to be against the order the
    # trials come out of the manifest, which is sorted by run id and therefore groups the
    # policies together - if review ids followed that, R001-R003 would all be one policy.
    review_order = [m["model"] for m in mapping]
    unshuffled = [r["run_id"].split("_")[3] for r in
                  sorted(({"run_id": r} for r in runs), key=lambda r: r["run_id"])]
    check(review_order != unshuffled,
          f"the review order is not the order the trials are read in (got {review_order})")
    check(len(set(review_order[:3])) > 1,
          f"the first few reviews are not all the same policy (got {review_order[:3]})")

    # and it must not be redrawable once labelling has begun
    again = subprocess.run([sys.executable, "blind_review.py", "prepare", "rqp1", "eyedrops"],
                           cwd=work / "harness", capture_output=True, text=True, env=env)
    check(again.returncode != 0, "prepare refuses to reassign review ids over an existing key")
    check("already exists" in again.stdout + again.stderr, "and says why")

    # check reports progress rather than a number, until every trial is judged
    partial = ["| R001 | successful | |" if line.startswith("| R001 |") else line
               for line in text.splitlines()]
    sheet.write_text("\n".join(partial) + "\n", encoding="utf-8")
    prog = subprocess.run([sys.executable, "blind_review.py", "check", "rqp1", "eyedrops"],
                          cwd=work / "harness", capture_output=True, text=True, env=env)
    check("1/6 judged" in prog.stdout, f"check counts what is done (got {prog.stdout.splitlines()[:1]})")
    check(prog.returncode != 0, "check exits nonzero while labels are missing")


def suite_pipeline(tmp: Path):
    """The whole reproduction, end to end, on invented trials.

    The unit suites protect each tool. Nothing protects the wiring between them, and the
    wiring is where a one-shot 70-minute robot session goes wrong: a manifest column the next
    tool does not read, a run-id spelling that two tools disagree about, an output filename
    that moves. This builds a complete 20-trial study - v3.0 recordings, videos, manifest -
    and runs it through scoring, timing, the commanded-closure outcome, the blind review and
    the results table, checking the number that comes out the far end is the one put in.
    """
    import subprocess
    try:
        import cv2
    except ImportError:
        print("             (skipped: no cv2)")
        return

    work = tmp / "pipeline"
    (work / "harness").mkdir(parents=True)
    (work / "armnetbench").mkdir(parents=True)
    for name in ["labbench.py", "score_rq1.py", "trial_timing.py", "command_vs_achieved.py",
                 "blind_review.py", "rqp1_results.py", "make_filmstrip.py", "labbench_config.json"]:
        shutil.copy(HERE / name, work / "harness" / name)
    shutil.copy(HERE.parent / "armnetbench" / "reference_results.csv", work / "armnetbench" / "reference_results.csv")
    datasets = work / "datasets"

    # 10 rounds x 2 policies, recorded at 20 Hz for 45 s like the real thing. Six of act's
    # trials grasp and lift; one of smolvla's does. Those are the numbers that must survive.
    HZ, SECS = 20.0, 45.0
    N = int(HZ * SECS)
    OPEN, CLOSED = 34.0, 1.0          # the benchmark's gripper scale, not ours
    plan = {"act": [True] * 6 + [False] * 4, "smolvla": [True] + [False] * 9}
    manifest = ["run_id,timestamp,model,condition,start_position,repo_id,episode_sec,video_path"]
    truth = {}
    for policy, outcomes in plan.items():
        for i, grasped in enumerate(outcomes, 1):
            run = f"eval_rqp1_eyedrops_{policy}_{i}"
            truth[run] = grasped
            if grasped:
                grip = np.concatenate([constant(200, OPEN), constant(60, CLOSED), constant(N - 260, OPEN)])
                lift = np.concatenate([constant(200, -40), np.linspace(-40, 20, 60), constant(N - 260, 20)])
            else:
                grip = constant(N, OPEN)
                lift = np.concatenate([constant(100, 0), np.linspace(0, -40, 100), constant(N - 200, -40)])
            root = datasets / run
            (root / "meta").mkdir(parents=True)
            (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 20}))
            make_episode(root / "data" / "chunk-000" / "file-000.parquet", grip, lift)
            video = root / "videos" / "observation.images.front" / "chunk-000" / "file-000.mp4"
            video.parent.mkdir(parents=True)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
            for f in range(40):
                writer.write(np.full((48, 64, 3), f * 6 % 256, np.uint8))
            writer.release()
            manifest.append(f"{run},t,{policy},eyedrops,P{(i - 1) % 5 + 1},x,{SECS:g},{video}")

    # A different study, fully recorded, sharing the manifest. This is not decoration: the
    # manifest holds every trial ever run, and the per-condition tools used to read all of
    # it and write one output file, so finishing the reproduction overwrote the earlier
    # study's record with the wrong trials. If the filters stop working, the counts below
    # go up.
    # The condition is deliberately the same word as the reproduction's. A different word
    # would be excluded by the condition filter alone, and then the study filter could stop
    # working without anything noticing.
    for i in range(1, 5):
        run = f"eval_rq0_eyedrops_60all_{i}"
        root = datasets / run
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.1", "fps": 30}))
        make_episode(root / "data" / "chunk-000" / "episode_000000.parquet", constant(900, 90.0))
        other = root / "videos" / "chunk-000" / "observation.images.front" / "episode_000000.mp4"
        other.parent.mkdir(parents=True)
        writer = cv2.VideoWriter(str(other), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
        for f in range(30):
            writer.write(np.zeros((48, 64, 3), np.uint8))
        writer.release()
        manifest.append(f"{run},t,60all,eyedrops,P1,x,30,{other}")
    (work / "harness" / "rq1_manifest.csv").write_text("\n".join(manifest) + "\n")

    env = dict(os.environ, LABBENCH_DATASETS=str(datasets), LABBENCH_CAMERA="front",
               LABBENCH_CLOSE_T="15", LABBENCH_FPS="20")

    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=work / "harness",
                              capture_output=True, text=True, env=env)

    # 1. scoring, with the benchmark's closure threshold rather than our arm's
    out = run("score_rq1.py", "eval_rqp1_eyedrops_act_", "10")
    check(out.returncode == 0, f"the scorer runs over the study ({out.stderr[-200:]})")
    scores = pd.read_csv(work / "harness" / "eval_rqp1_eyedrops_act_scores.csv")
    check(len(scores) == 10 and (scores.status == "ok").all(), "all ten act trials are found and scored")
    check(int(scores.success.sum()) == 6, f"the scorer recovers the six successes built in (got {int(scores.success.sum())})")
    check(set(scores.true_hz.round(0)) == {20.0}, f"and reads the rate off the manifest (got {set(scores.true_hz)})")
    check("WARNING" not in out.stdout, "no assumed episode length, because the manifest has every trial")

    # 2. the rate report
    out = run("trial_timing.py", "--study", "rqp1", "--condition", "eyedrops")
    check(out.returncode == 0 and "20.0" in out.stdout, "trial_timing scopes to the study and reports 20 Hz")
    check("act" in out.stdout and "smolvla" in out.stdout, "and covers both policies")
    check("eval_rq0" not in out.stdout, "and does not report the other study's trials")

    # 3. commanded vs achieved, scoped and named for the study
    out = run("command_vs_achieved.py", "--study", "rqp1", "--condition", "eyedrops")
    check(out.returncode == 0, f"the commanded-closure tool runs ({out.stderr[-200:]})")
    cva = work / "harness" / "rqp1_eyedrops_command_vs_achieved.csv"
    check(cva.exists(), "and writes a file named for the study and condition")
    rows = pd.read_csv(cva)
    check(len(rows) == 20, f"covering this study's 20 trials and not the other study's 4 (got {len(rows)})")
    check(not any(r.startswith("eval_rq0") for r in rows.run_id),
          "no trial from the other study leaks into this condition's output")
    check(int((rows.commanded_closure == "yes").sum()) == 7,
          f"seven trials commanded a closure of at least 0.5 s (got {int((rows.commanded_closure == 'yes').sum())})")

    # 4. blind review over the whole study
    out = run("blind_review.py", "prepare", "rqp1", "eyedrops")
    check(out.returncode == 0, f"blind review prepares ({out.stderr[-300:]})")
    key = work / "harness" / "rqp1_eyedrops_review_key.csv"
    sheet = work / "harness" / "rqp1_eyedrops_labels.md"
    with key.open() as f:
        mapping = list(csv.DictReader(f))
    check(len(mapping) == 20, f"only this study's trials go into the review set (got {len(mapping)})")
    strips = list((work / "harness" / "review" / "rqp1_eyedrops").glob("R*.jpg"))
    check(len(strips) == 20, f"and a filmstrip built from its video (got {len(strips)})")

    # 5. a judge labels blind: they cannot see the policy, only the video, so here the label
    #    comes from what the trial actually did - which is what a video would show.
    labels = {m["review_id"]: ("successful" if truth[m["run_id"]] else "failure") for m in mapping}
    text = sheet.read_text(encoding="utf-8").splitlines()
    filled = [f"| {line.split('|')[1].strip()} | {labels[line.split('|')[1].strip()]} | |"
              if line.startswith("| R") else line for line in text]
    sheet.write_text("\n".join(filled) + "\n", encoding="utf-8")
    out = run("blind_review.py", "check", "rqp1", "eyedrops")
    check(out.returncode == 0 and "20/20 judged" in out.stdout, f"check sees a complete sheet ({out.stdout[:120]})")

    # 6. the headline number
    out = run("rqp1_results.py", "eyedrops")
    check(out.returncode == 0, f"the results table builds ({out.stderr[-300:]})")
    check("judging: blind" in out.stdout, "and records that the judging was blind")
    result = pd.read_csv(work / "harness" / "rqp1_eyedrops_results.csv").set_index("policy")
    check_close(result.loc["act", "rate"], 0.6, 1e-6, "act's rate is the six successes in ten trials")
    check_close(result.loc["smolvla", "rate"], 0.1, 1e-6, "smolvla's rate is the one in ten")
    check_close(result.loc["act", "delta"], 0.6 - 19 / 30, 1e-3, "the delta is against the published eye-drops rate")
    check(result.loc["act", "unlabelled"] == 0 and result.loc["act", "invalid"] == 0, "nothing is left unaccounted for")


SUITES = {
    "scorer": suite_scorer,
    "paths": suite_paths,
    "command": suite_command,
    "rqp1": suite_rqp1,
    "config": suite_config,
    "demos": suite_demos,
    "blind": suite_blind,
    "pipeline": suite_pipeline,
}


def main():
    global _CURRENT_SUITE
    wanted = sys.argv[1:] or list(SUITES)
    unknown = [w for w in wanted if w not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; available: {', '.join(SUITES)}")
    tmp = Path(tempfile.mkdtemp(prefix="labbench_tests_"))
    try:
        for name in wanted:
            _CURRENT_SUITE = name
            before = len(FAILURES)
            try:
                SUITES[name](tmp / name)
            except Exception:
                FAILURES.append(f"{name}: raised\n{traceback.format_exc()}")
            status = "ok" if len(FAILURES) == before else f"{len(FAILURES) - before} FAILED"
            print(f"  {name:9} {status}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    for f in FAILURES:
        print(f"FAIL  {f}")
    print(f"\n{PASSED} checks passed, {len(FAILURES)} failed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
