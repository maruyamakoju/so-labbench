# Trial scorer: stage rates + failure taxonomy + Wilson interval.
#   python score_rq1.py <run_prefix> <n_trials> [episode_sec]
#   e.g.  python score_rq1.py eval_rq1_c0_act_ 20
#
# Success (pick-up) = a stable closure, a lift after it, and a hold. Every threshold lives
# in labbench.py and is stated in seconds; this file decides how they are combined, not what
# they are.
#
# Two corrections are worth carrying forward, because both were invisible until someone
# looked and both changed the answer:
#
# v1.3 (2026-08-15, found during the pilot): the closure thresholds were fixed frame counts
# annotated "@30fps". The record loop does not reach 30 Hz - it runs for a wall-clock
# duration and appends one frame per iteration, so a heavier policy yields fewer frames.
# Measured back to back: ACT 23.6 Hz, SmolVLA 18.4 Hz. Under the old constants "0.5 s" meant
# 0.63 s for ACT and 0.81 s for SmolVLA, holding the slower model to a 1.28x longer grasp -
# in a benchmark whose entire purpose was to compare the two.
#
# v1.4 (2026-09-12, before the RQ0 evaluation): the grasp window was the longest closed run.
# 27 of the 60 demonstrations begin with the gripper still closed from the previous
# recording, and that idle stretch is usually the longest, so the lift was measured over a
# window that was never a grasp. The window is now the stable closure with the largest lift.
# Scoring the demonstrations this way agrees with the independent quality audit 60/60.
import sys

import numpy as np
import pandas as pd
from pathlib import Path

from labbench import (DESCEND_D, DESCEND_FLOOR, EPISODE_SEC, HOLD_S, INSTABILITY_REV_PER_S,
                      LIFT_GAIN, MOVE_D, STABLE_S, closure_runs, episode_parquet, episode_seconds,
                      hold_frames, lift_gain, select_grasp_window, stable_frames, true_hz, wilson_ci)

HERE = Path(__file__).parent


def score_episode(parquet: Path, episode_sec: float) -> dict:
    df = pd.read_parquet(parquet)
    st = np.array(df["observation.state"].tolist())
    pan, lift, grip = st[:, 0], st[:, 1], st[:, 5]

    # The dataset's timestamp column is frame_index/fps, a synthetic value, so it cannot
    # report the real rate. Derive it from frames over the wall-clock episode length.
    hz = true_hz(len(df), episode_sec)
    stable_f, hold_f = stable_frames(hz), hold_frames(hz)

    moved = float(np.abs(np.diff(st[:, :5], axis=0)).sum())
    descend = bool((lift[0] - lift.min()) >= DESCEND_D or lift.min() < DESCEND_FLOOR)

    runs = closure_runs(grip)
    window = select_grasp_window(runs, lift, stable_f)
    grasp = window is not None
    if grasp:
        onset, best = window
        gain = lift_gain(lift, onset, best)
    else:
        # No qualifying window. Report the longest closure's position for the audit trail,
        # but nothing derived from it: the hold below is zero, not that idle stretch's length.
        onset, best = max(runs, key=lambda r: r[1]) if runs else (0, 0)
        gain = 0.0
    hold_ok = grasp and best >= hold_f
    lift_ok = grasp and gain >= LIFT_GAIN
    success = bool(grasp and lift_ok and hold_ok)

    # Oscillation index for instability: direction reversals of pan and lift. The raw count
    # grows with the number of samples, so comparing it to a fixed number carried the same
    # defect as the closure thresholds - a faster policy logs more reversals for identical
    # physical behaviour. Judge the rate.
    rev = int((np.diff(np.sign(np.diff(pan))) != 0).sum() + (np.diff(np.sign(np.diff(lift))) != 0).sum())
    rev_per_s = rev / episode_sec

    # taxonomy: success / approach_failure / grasp_failure / lift_failure / hold_failure /
    # policy_instability. invalid_trial is operator-marked and never inferred here.
    if success:
        tax = "success"
    elif grasp and lift_ok and not hold_ok:
        tax = "hold_failure"
    elif grasp and not lift_ok:
        tax = "lift_failure"
    elif descend and not grasp:
        tax = "grasp_failure"
    elif not descend and moved < MOVE_D:
        tax = "approach_failure"
    elif not descend and rev_per_s > INSTABILITY_REV_PER_S:
        tax = "policy_instability"
    else:
        tax = "approach_failure"
    return dict(success=success, approach=descend, grasp=bool(grasp), lift=lift_ok, hold=hold_ok,
                taxonomy=tax,
                hold_frames=int(best) if grasp else 0,
                hold_sec=round(best / hz, 2) if grasp else 0.0,
                true_hz=round(hz, 1), stable_f=stable_f, hold_f=hold_f,
                n_closures=len(runs), window_onset=int(onset), lift_gain=round(gain, 1),
                grip_min=float(grip.min()), lift_min=float(lift.min()),
                reversals=rev, rev_per_s=round(rev_per_s, 2), moved=round(moved, 1))


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: score_rq1.py <run_prefix> <n_trials> [episode_sec]\n"
                         "  e.g. score_rq1.py eval_rq1_c0_act_ 20")
    prefix, n = sys.argv[1], int(sys.argv[2])
    cli_sec = float(sys.argv[3]) if len(sys.argv) > 3 else None
    ep_secs = episode_seconds()
    # operator-marked invalid trials (one trial number per line): excluded, not policy failures
    invalid_file = HERE / f"{prefix}invalid.txt"
    invalid = set()
    if invalid_file.exists():
        invalid = {int(x) for x in invalid_file.read_text().split() if x.strip().isdigit()}
    rows, assumed = [], []
    for i in range(1, n + 1):
        run = f"{prefix}{i}"
        if i in invalid:
            rows.append(dict(trial=i, status="invalid_trial"))
            continue
        try:
            parquet = episode_parquet(run)
        except FileNotFoundError:
            rows.append(dict(trial=i, status="missing"))
            continue
        if not parquet.exists():
            rows.append(dict(trial=i, status="missing"))
            continue
        sec = cli_sec or ep_secs.get(run)
        if sec is None:
            sec = EPISODE_SEC
            assumed.append(i)
        r = score_episode(parquet, sec)
        r.update(trial=i, status="ok")
        rows.append(r)
    d = pd.DataFrame(rows)
    out = HERE / f"{prefix}scores.csv"
    d.to_csv(out, index=False)
    ok = d[d.status == "ok"]
    n_ok = len(ok)
    print(f"== {prefix} ==  scored {n_ok}/{n}" +
          (f"  (excluded invalid: {len(invalid)})" if invalid else ""))
    if assumed:
        # Silence here would mean scoring at an assumed rate: every seconds-based threshold
        # converted with the wrong divisor, and nothing saying so.
        print(f"  WARNING: no manifest row for trial(s) {assumed}; assumed {EPISODE_SEC:g}s episodes. "
              f"Pass the real length as the third argument if that is wrong.")
    if n_ok:
        k = int(ok.success.sum())
        lo, hi = wilson_ci(k, n_ok)
        print(f"SUCCESS: {k}/{n_ok} = {100 * k / n_ok:.0f}%  (95% CI {100 * lo:.0f}-{100 * hi:.0f}%)")
        for stage in ["approach", "grasp", "lift", "hold"]:
            print(f"  {stage:>9}: {int(ok[stage].sum())}/{n_ok} ({100 * ok[stage].mean():.0f}%)")
        print("  taxonomy:", {k: int(v) for k, v in ok.taxonomy.value_counts().items()})
        print(f"  control rate: {ok.true_hz.min():.1f}-{ok.true_hz.max():.1f} Hz"
              f"  -> {STABLE_S:g}s = {int(ok.stable_f.min())}-{int(ok.stable_f.max())} frames,"
              f" {HOLD_S:g}s = {int(ok.hold_f.min())}-{int(ok.hold_f.max())} frames")
    print(f"csv -> {out}")


if __name__ == "__main__":
    main()
