# Report the true control rate of each trial, and what the scorer's frame thresholds
# actually mean in seconds because of it.
#   python trial_timing.py [repo_prefix n] ...
#
# lerobot's record loop runs for episode_time_s of WALL CLOCK and appends one frame per
# iteration (record.py: `while timestamp < control_time_s` with `busy_wait(1/fps - dt_s)`,
# which does not wait when the iteration already overran). When inference cannot hold
# 30 Hz the episode simply contains fewer frames. The dataset's timestamp column is
# frame_index/fps, so it reports 30 Hz regardless and cannot reveal this.
#
# The scorer therefore states its thresholds in seconds and converts them per trial with
# that trial's own rate. This tool shows the conversion: how many frames 0.5 s and 1.0 s
# came to at each trial's measured rate, and how far apart two policies' rates are. When
# the thresholds were frame counts they matched the protocol only at a true 30 Hz, and a
# slower policy was silently held to a longer hold than a faster one.
import os
import sys
import json
import pandas as pd
from pathlib import Path

from labbench import (HOLD_S, STABLE_S, dataset_info, dataset_exists, episode_parquet,
                      hold_frames, read_manifest, stable_frames)

HERE = Path(__file__).parent


def trials_from_args(argv):
    # `--ep S` supplies episode_time_s for prefix mode, where the manifest is not
    # available. Rates across prefixes are only comparable if S really was the same.
    ep = None
    if "--ep" in argv:
        k = argv.index("--ep")
        ep = float(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]

    study = argv[argv.index("--study") + 1] if "--study" in argv else None
    condition = argv[argv.index("--condition") + 1] if "--condition" in argv else None
    for flag in ("--study", "--condition"):
        if flag in argv:
            k = argv.index(flag)
            argv = argv[:k] + argv[k + 2:]

    out = []
    if argv:
        if len(argv) % 2:
            raise SystemExit(f"expected pairs of <prefix> <n>, got an odd number of arguments: {argv}")
        for i in range(0, len(argv), 2):
            prefix, n = argv[i], int(argv[i + 1])
            out += [(f"{prefix}{j}", prefix.rstrip("_"), ep) for j in range(1, n + 1)]
        return out
    rows = read_manifest(study=study, condition=condition)
    if not rows:
        raise SystemExit("no matching trials in the manifest, and no prefix given")
    return [(r["run_id"], r.get("model", ""), float(r.get("episode_sec") or 0) or None) for r in rows]


def main():
    rows = []
    for run_id, model, ep_sec in trials_from_args(sys.argv[1:]):
        if not dataset_exists(run_id):
            continue
        pq = episode_parquet(run_id)
        if not pq.exists():
            continue
        n = len(pd.read_parquet(pq))
        fps = dataset_info(run_id).get("fps")
        rows.append(dict(run_id=run_id, model=model, frames=n, declared_fps=fps,
                         episode_sec=ep_sec, true_hz=(n / ep_sec) if ep_sec else None))

    if not rows:
        print("no trials found")
        return 1

    print(f"{'run_id':32} {'model':9} {'frames':>7} {'decl':>5} {'true Hz':>8}"
          f" {f'{STABLE_S:g}s =':>8} {f'{HOLD_S:g}s =':>8}")
    for r in rows:
        hz = r["true_hz"]
        stable = f"{stable_frames(hz)}f" if hz else "?"
        hold = f"{hold_frames(hz)}f" if hz else "?"
        hz_s = f"{hz:.1f}" if hz else "?"
        print(f"{r['run_id']:32} {r['model']:9} {r['frames']:>7} {r['declared_fps']:>5} {hz_s:>8} {stable:>8} {hold:>8}")

    by_model = {}
    for r in rows:
        if r["true_hz"]:
            by_model.setdefault(r["model"] or "?", []).append(r["true_hz"])
    if by_model:
        print("\nmean true rate by model:")
        for m, v in sorted(by_model.items()):
            mean = sum(v) / len(v)
            print(f"  {m:9} {mean:5.1f} Hz  ->  {STABLE_S:g}s = {stable_frames(mean)} frames, "
                  f"{HOLD_S:g}s = {hold_frames(mean)} frames")
        if len(by_model) > 1:
            means = {m: sum(v) / len(v) for m, v in by_model.items()}
            lo, hi = min(means, key=means.get), max(means, key=means.get)
            ratio = means[hi] / means[lo]
            print(f"\n  {lo} runs {ratio:.2f}x slower than {hi}. A scorer that counted frames"
                  f"\n  would ask {lo} for a {ratio:.2f}x longer hold than {hi} at the same nominal"
                  "\n  threshold, which is why these thresholds are stated in seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
