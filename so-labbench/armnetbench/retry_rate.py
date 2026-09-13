# Do policies that grasp more often than the human succeed less often?
#
#   python retry_rate.py
#
# On cable_clip, the one task all seven policies fail, the action traces showed something the
# success rate cannot: the policies are not failing to grasp. They grasp five to seventeen
# times where the human demonstration grasps three, over roughly twice the duration. They
# retry, and the retrying executes correctly; what is missing is whatever makes one attempt
# work.
#
# That was one task. Two, counting the control. Two tasks is an anecdote. The benchmark
# publishes the action trace of every rollout across eight tasks and seven policies, so the
# same question can be asked of all fifty-six cells at once, and either becomes a property of
# the benchmark or does not survive.
#
# THE TRAP, and it is the whole difficulty here. v0.1 enforced no wall-clock limit: an
# operator stopped each rollout by hand. So a failed rollout runs longer than a successful
# one almost by construction, and anything counted per rollout - closures included - inherits
# that. "Failures have more closures" would then restate "failures run longer" and predict
# nothing. Two things are done about it:
#
#   1. The primary quantity is closures per SECOND, not per rollout. Duration divides out.
#   2. The cleanest test is WITHIN a policy-task cell: successful rollouts against failed ones
#      by the same policy on the same task. Task difficulty and policy identity cancel, and
#      only the within-cell difference is read.
#
# The raw per-rollout count is reported too, precisely so the confound can be seen rather than
# hidden: if the rate result and the count result disagree, the count is the contaminated one.
#
# The closed/open threshold is not assumed. A benchmark action value is a percentage of the
# recording arm's calibrated range (see finding_calibration_portability.md), so a number that
# means "closed" here means nothing anywhere else, and hardcoding 15 or 40 would be importing
# a constant from a different arm. It is measured per task from the demonstrations' own
# bimodal gripper distribution, by the split that best separates the two modes.
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATASET = "armnet/armnetbench_v01_lerobot_so101"
TASKS = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))["tasks"]
GRIPPER = 5                    # index of the gripper in the six-joint action vector
MIN_CLOSED_FRAMES = 3          # 0.15 s at 20 fps: below this it is chatter, not a grasp
FPS = 20.0


def otsu_threshold(values):
    """The split of a bimodal distribution that best separates its two modes.

    A gripper trace in a pick-and-place demonstration is open most of the time and closed
    while carrying, so its histogram has two humps. Otsu finds the cut maximising between-class
    variance - the cut a person would draw by eye, but written down and reproducible.
    """
    hist, edges = np.histogram(values, bins=100)
    centres = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return float(np.median(values))
    weight_lo = np.cumsum(hist)
    weight_hi = total - weight_lo
    mean_lo = np.cumsum(hist * centres) / np.maximum(weight_lo, 1)
    grand = (hist * centres).sum()
    mean_hi = (grand - np.cumsum(hist * centres)) / np.maximum(weight_hi, 1)
    between = weight_lo * weight_hi * (mean_lo - mean_hi) ** 2
    valid = (weight_lo > 0) & (weight_hi > 0)
    if not valid.any():
        return float(np.median(values))
    between = np.where(valid, between, -1.0)
    return float(centres[int(np.argmax(between))])


def count_closures(gripper, threshold, min_frames):
    """Maximal runs below the threshold that last long enough to be a grasp."""
    closed = gripper < threshold
    if not closed.any():
        return 0
    edges = np.diff(closed.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if closed[0]:
        starts.insert(0, 0)
    if closed[-1]:
        ends.append(len(closed))
    return sum(1 for a, b in zip(starts, ends) if b - a >= min_frames)


def load_all():
    """Every episode's gripper trace, reading each of the 120 data files exactly once."""
    from huggingface_hub import hf_hub_download

    meta = pd.read_parquet(hf_hub_download(DATASET, "meta/episodes/chunk-000/file-000.parquet",
                                           repo_type="dataset"))
    meta["task"] = meta["tasks"].apply(
        lambda x: x[0] if hasattr(x, "__len__") and not isinstance(x, str) else str(x))
    by_instruction = {v["instruction"]: k for k, v in TASKS.items()}
    meta["task_key"] = meta["task"].map(by_instruction)
    meta = meta[meta.task_key.notna()].copy()

    traces = {}
    groups = list(meta.groupby(["data/chunk_index", "data/file_index"]))
    for n, ((chunk, file_index), group) in enumerate(groups, 1):
        path = f"data/chunk-{int(chunk):03d}/file-{int(file_index):03d}.parquet"
        table = pd.read_parquet(hf_hub_download(DATASET, path, repo_type="dataset"),
                                columns=["index", "action"])
        actions = np.array(table["action"].tolist(), dtype=np.float32)
        idx = table["index"].to_numpy()
        for _, ep in group.iterrows():
            a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            rows = (idx >= a) & (idx < b)
            if not rows.any():
                continue
            traces[int(ep.episode_index)] = actions[rows, GRIPPER]
        print(f"  [{n:3d}/{len(groups)}] {path}: {len(group)} episodes", flush=True)
    return meta, traces


def fisher_ci(r, n, conf=0.95):
    """Confidence interval for a correlation, through the z transform."""
    if n < 4 or abs(r) >= 1 or r != r:
        return float("nan"), float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1 / math.sqrt(n - 3)
    crit = 1.959963985 if conf == 0.95 else 2.5758
    return math.tanh(z - crit * se), math.tanh(z + crit * se)


def recompute(meta, traces, thresholds, min_frames, scale):
    """Every episode's closure rate at one threshold setting.

    The sensitivity check has to recompute the SAME quantities the primary result is stated
    in, or it answers a different question and reassures about nothing. An earlier version
    returned the raw rate while the headline was the rate relative to the human, and the two
    do not even share a sign.
    """
    rows = []
    for _, ep in meta.iterrows():
        e = int(ep.episode_index)
        if e not in traces:
            continue
        g = traces[e]
        n_close = count_closures(g, thresholds[ep.task_key] * scale, min_frames)
        rows.append((ep.task_key, ep.policy_type, ep.success_class, n_close / (len(g) / FPS)))
    return pd.DataFrame(rows, columns=["task", "policy", "outcome", "rate"])


def within_cell_sign_test(frame):
    """Per policy-task cell, does the failed rollout grasp more often than the successful one?

    This is the only comparison here that is not between tasks. Task difficulty, policy
    identity and the operator's stopping habits are all held fixed inside a cell, so whatever
    is left is the thing being asked about.
    """
    deltas = []
    for _, g in frame[frame.policy != "teleoperated"].groupby(["task", "policy"]):
        won, lost = g[g.outcome == "successful"], g[g.outcome == "failure"]
        if len(won) < 3 or len(lost) < 3:
            continue
        deltas.append(lost.rate.median() - won.rate.median())
    if not deltas:
        return 0, 0, float("nan"), float("nan")
    higher, n = sum(1 for d in deltas if d > 0), len(deltas)
    p = sum(math.comb(n, k) for k in range(higher, n + 1)) / 2 ** n
    return higher, n, p, float(np.median(deltas))


def main():
    meta, traces = load_all()

    # One threshold per task, from that task's demonstrations only.
    thresholds, demo_pools = {}, {}
    for key in TASKS:
        demo_eps = meta[(meta.task_key == key) & (meta.policy_type == "teleoperated")]
        pool = [traces[int(e)] for e in demo_eps.episode_index if int(e) in traces]
        pool = np.concatenate(pool) if pool else np.array([0.0], dtype=np.float32)
        demo_pools[key] = (len(demo_eps), pool)
        thresholds[key] = otsu_threshold(pool)

    print("\nclosed/open split, measured per task from its own demonstrations")
    print(f"{'task':22} {'demos':>6} {'threshold':>10}   demo gripper range")
    for key in TASKS:
        n_demo, pool = demo_pools[key]
        print(f"{key:22} {n_demo:6d} {thresholds[key]:10.1f}   "
              f"{pool.min():.1f} .. {pool.max():.1f}")

    rows = []
    for _, ep in meta.iterrows():
        e = int(ep.episode_index)
        if e not in traces:
            continue
        g = traces[e]
        t = thresholds[ep.task_key]
        seconds = len(g) / FPS
        n_close = count_closures(g, t, MIN_CLOSED_FRAMES)
        rows.append(dict(episode=e, task=ep.task_key, policy=ep.policy_type,
                         outcome=ep.success_class, frames=len(g), seconds=round(seconds, 1),
                         closures=n_close, closures_per_s=round(n_close / seconds, 4),
                         closed_frac=round(float((g < t).mean()), 3)))
    per_ep = pd.DataFrame(rows)
    per_ep.to_csv(HERE / "retry_rate_episodes.csv", index=False)

    # ---------- cell level: 8 tasks x 7 policies, against the human on the same task ----------
    ref = pd.read_csv(HERE / "reference_results.csv")
    by_instruction = {v["instruction"]: k for k, v in TASKS.items()}
    ref["task_key"] = ref["task"].map(by_instruction)
    success = {(r.task_key, r.policy_type): r.success_rate for r in ref.itertuples()}

    demos = per_ep[per_ep.policy == "teleoperated"]
    demo_rate = {k: demos[demos.task == k].closures_per_s.median() for k in TASKS}
    demo_count = {k: demos[demos.task == k].closures.median() for k in TASKS}

    cells = []
    for (task, policy), g in per_ep[per_ep.policy != "teleoperated"].groupby(["task", "policy"]):
        sr = success.get((task, policy))
        if sr is None or not demo_rate.get(task):
            continue
        cells.append(dict(task=task, policy=policy, n=len(g), success_rate=sr,
                          closures_per_s=round(g.closures_per_s.median(), 4),
                          rate_vs_human=round(g.closures_per_s.median() / demo_rate[task], 2),
                          closures=round(g.closures.median(), 1),
                          count_vs_human=round(g.closures.median() / max(demo_count[task], 0.5), 2),
                          seconds=round(g.seconds.median(), 1)))
    cell = pd.DataFrame(cells)
    cell.to_csv(HERE / "retry_rate_cells.csv", index=False)

    policies = sorted(cell.policy.unique())
    print(f"\n{len(cell)} policy-task cells")
    print(f"{'':22} " + " ".join(f"{p:>10}" for p in policies))
    for task in sorted(cell.task.unique()):
        g = cell[cell.task == task].set_index("policy")
        line = f"{task:22} "
        for p in policies:
            line += f"{g.loc[p, 'rate_vs_human']:>10.2f} " if p in g.index else f"{'-':>10} "
        print(line)
    print("  (median closures per second, as a multiple of the human's on the same task)")

    print("\nDOES IT PREDICT SUCCESS? (across the 56 cells, so BETWEEN tasks)")
    for label, column in [("closures per second, raw", "closures_per_s"),
                          ("closures per second, as a multiple of the human", "rate_vs_human"),
                          ("closures per rollout vs human (duration-confounded)", "count_vs_human"),
                          ("rollout duration in seconds", "seconds")]:
        x = cell[column].to_numpy(dtype=float)
        y = cell["success_rate"].to_numpy(dtype=float)
        r = float(np.corrcoef(x, y)[0, 1])
        lo, hi = fisher_ci(r, len(x))
        verdict = "excludes zero" if (lo > 0 or hi < 0) else "crosses zero"
        print(f"  {label:52} r={r:+.2f}  [{lo:+.2f} .. {hi:+.2f}]  {verdict}")

    # ---------- the clean test: within a cell, successes against failures ----------
    print("\nWITHIN each policy-task cell: do the rollouts that SUCCEED grasp less often?")
    print("  (same policy, same task, so difficulty and policy identity cancel)")
    paired = []
    for (task, policy), g in per_ep[per_ep.policy != "teleoperated"].groupby(["task", "policy"]):
        won, lost = g[g.outcome == "successful"], g[g.outcome == "failure"]
        if len(won) < 3 or len(lost) < 3:
            continue
        paired.append(dict(task=task, policy=policy, n_success=len(won), n_failure=len(lost),
                           win_per_s=round(won.closures_per_s.median(), 3),
                           lose_per_s=round(lost.closures_per_s.median(), 3),
                           delta=round(lost.closures_per_s.median() - won.closures_per_s.median(), 3)))
    pair = pd.DataFrame(paired)
    if pair.empty:
        print("  no cell has at least 3 successes and 3 failures")
    else:
        pair = pair.sort_values("delta", ascending=False)
        pair.to_csv(HERE / "retry_rate_within_cell.csv", index=False)
        print(f"{'task':22} {'policy':11} {'win n':>6} {'lose n':>7} "
              f"{'win /s':>8} {'lose /s':>8} {'lose - win':>11}")
        for r in pair.itertuples():
            print(f"{r.task:22} {r.policy:11} {r.n_success:6d} {r.n_failure:7d} "
                  f"{r.win_per_s:8.3f} {r.lose_per_s:8.3f} {r.delta:+11.3f}")
        higher, n = int((pair.delta > 0).sum()), len(pair)
        # sign test: with no relation between grasping rate and outcome, the sign is a coin flip
        p = sum(math.comb(n, k) for k in range(higher, n + 1)) / 2 ** n
        print(f"\n  failures grasp more often in {higher} of {n} cells "
              f"(one-sided sign test p={p:.3f})")
        print(f"  median difference: {pair.delta.median():+.3f} closures per second")

    # Why the raw rate and the human-relative rate disagree in sign. If the human grasps
    # often per second on exactly the tasks the policies solve, then dividing by the human's
    # rate divides out task easiness, and the two normalisations must point opposite ways.
    # Neither is then evidence about grasping; both are measuring which tasks are easy.
    task_sr = {k: cell[cell.task == k].success_rate.mean() for k in TASKS}
    xs = np.array([demo_rate[k] for k in TASKS], dtype=float)
    ys = np.array([task_sr[k] for k in TASKS], dtype=float)
    r_demo = float(np.corrcoef(xs, ys)[0, 1])
    lo, hi = fisher_ci(r_demo, len(xs))
    print("\nWHY THE TWO NORMALISATIONS DISAGREE")
    print(f"  the HUMAN's own grasp rate against the task's mean policy success rate: "
          f"r={r_demo:+.2f}  [{lo:+.2f} .. {hi:+.2f}]  (n=8 tasks)")
    print("  the human grasps faster on the tasks policies solve, so that rate carries task")
    print("  easiness; dividing by it removes the easiness and flips the sign. Neither")
    print("  between-task correlation is evidence about grasping.")

    print("\nSENSITIVITY: does the answer survive moving the threshold?")
    print(f"  {'setting':40} {'r(raw)':>8} {'r(vs human)':>12}   within-cell sign test")
    for scale, sname in [(0.75, "threshold x0.75"), (1.0, "as measured"), (1.33, "threshold x1.33")]:
        for min_f in (2, 3, 5):
            tag = f"{sname}, min {min_f} frames" + (" (primary)" if scale == 1.0 and min_f == 3 else "")
            frame = recompute(meta, traces, thresholds, min_f, scale)
            med = frame.groupby(["task", "policy"]).rate.median().reset_index()
            demo = {k: med[(med.task == k) & (med.policy == "teleoperated")].rate.iloc[0]
                    for k in TASKS}
            pol = med[med.policy != "teleoperated"].copy()
            pol["sr"] = [success.get((t, p)) for t, p in zip(pol.task, pol.policy)]
            pol["ratio"] = [r / demo[t] if demo[t] else float("nan")
                            for r, t in zip(pol.rate, pol.task)]
            pol = pol.dropna()
            r_raw = float(np.corrcoef(pol.rate, pol.sr)[0, 1])
            r_rel = float(np.corrcoef(pol.ratio, pol.sr)[0, 1])
            higher, n, p, med_d = within_cell_sign_test(frame)
            print(f"  {tag:40} {r_raw:+8.2f} {r_rel:+12.2f}   {higher:2d}/{n} p={p:.3f}")

    print("\nwrote retry_rate_episodes.csv, retry_rate_cells.csv, retry_rate_within_cell.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
