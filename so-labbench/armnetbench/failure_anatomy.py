# Where does a failing policy leave the demonstrated behaviour?
#
#   python failure_anatomy.py cable_clip            # the task all seven policies fail
#   python failure_anatomy.py eye_drops_to_basket   # a task most of them solve, as a control
#
# ArmnetBench records the action trace of every evaluation rollout, not only its video and
# its label. That means all seven policies can be analysed here, including the four this
# machine cannot run - openpi, GR00T and MolmoAct never have to be loaded, because what they
# did was written down.
#
# The measurement is one question asked frame by frame: how far is this rollout from anything
# a human demonstration ever did? The demonstrations define a set of arm configurations that
# are known to be on the way to success. A rollout that stays near that set is doing roughly
# the demonstrated thing; one that leaves it is somewhere the demonstrations never went, and
# the moment it leaves is the moment worth looking at.
#
# Distance is nearest-neighbour in joint space over all demonstration frames of the same
# task, scaled per joint by that joint's spread in the demonstrations, so a joint that barely
# moves does not dominate a joint that sweeps.
#
# What this cannot say: being off-manifold is not the same as failing, and a rollout can sit
# on the manifold and still miss by a millimetre in a way joint angles do not record. It
# localises the divergence in time; it does not explain the physics.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATASET = "armnet/armnetbench_v01_lerobot_so101"
import json
TASKS = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))["tasks"]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
BINS = 10          # phases of normalized episode time


def load_episodes(task_key: str):
    from huggingface_hub import hf_hub_download
    meta = pd.read_parquet(hf_hub_download(DATASET, "meta/episodes/chunk-000/file-000.parquet",
                                           repo_type="dataset"))
    meta["task"] = meta["tasks"].apply(
        lambda x: x[0] if hasattr(x, "__len__") and not isinstance(x, str) else str(x))
    wanted = meta[meta.task == TASKS[task_key]["instruction"]].copy()
    if wanted.empty:
        raise SystemExit(f"no episodes for {task_key}")

    frames = {}
    for (chunk, file_index), group in wanted.groupby(["data/chunk_index", "data/file_index"]):
        path = f"data/chunk-{int(chunk):03d}/file-{int(file_index):03d}.parquet"
        table = pd.read_parquet(hf_hub_download(DATASET, path, repo_type="dataset"))
        for _, ep in group.iterrows():
            a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            rows = table[(table["index"] >= a) & (table["index"] < b)]
            if rows.empty:
                continue
            frames[int(ep.episode_index)] = (ep.policy_type, ep.success_class,
                                             np.array(rows["action"].tolist(), dtype=np.float32))
        print(f"  read {path}: {len(group)} episodes", flush=True)
    return frames


def main():
    task_key = sys.argv[1] if len(sys.argv) > 1 else "cable_clip"
    print(f"{task_key}: {TASKS[task_key]['instruction']!r}")
    episodes = load_episodes(task_key)

    demos = [a for policy, _, a in episodes.values() if policy == "teleoperated"]
    if not demos:
        raise SystemExit("no teleoperated demonstrations for this task")
    manifold = np.concatenate(demos, axis=0)
    # Which part of the task each demonstration frame belongs to, 0 at the start and 1 at the
    # end. Staying on the manifold says the arm is in a pose the demonstrations visited; this
    # says WHICH pose, so a rollout that cycles through the first half forever can be told
    # apart from one genuinely working through the task.
    manifold_phase = np.concatenate([np.linspace(0, 1, len(d), dtype=np.float32) for d in demos])
    scale = manifold.std(axis=0)
    scale[scale < 1e-6] = 1.0
    print(f"\n{len(demos)} demonstrations, {len(manifold)} frames define the manifold")
    print(f"  per-joint spread: {dict(zip(JOINTS, manifold.std(axis=0).round(1)))}")

    def offmanifold(trace):
        """Nearest demonstration frame, per frame, in units of demonstration spread."""
        a = trace / scale
        b = manifold / scale
        out = np.empty(len(a), dtype=np.float32)
        where = np.empty(len(a), dtype=np.float32)
        for i in range(0, len(a), 256):                  # blocked, so the cross product fits
            chunk = a[i:i + 256]
            d = np.sqrt(((chunk[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
            nearest = d.argmin(axis=1)
            out[i:i + 256] = d[np.arange(len(chunk)), nearest]
            where[i:i + 256] = manifold_phase[nearest]
        return out, where

    # A demonstration measured against the others is the floor: how far apart two humans
    # doing the same task are. Anything a policy does below that is not a divergence.
    floor = []
    for i, demo in enumerate(demos[:10]):
        others = np.concatenate([d for j, d in enumerate(demos) if j != i], axis=0) / scale
        a = demo / scale
        d = np.sqrt(((a[::4, None, :] - others[None, ::3, :]) ** 2).sum(axis=2)).min(axis=1)
        floor.append(d)
    floor = np.concatenate(floor)
    print(f"  a held-out demonstration sits {np.median(floor):.2f} from the others "
          f"(p90 {np.percentile(floor, 90):.2f}) - that is the floor")

    rows = []
    for ep, (policy, outcome, trace) in sorted(episodes.items()):
        if policy == "teleoperated":
            continue
        d, matched = offmanifold(trace)
        phase = np.linspace(0, 1, len(d))
        reached = float(np.percentile(matched, 95))
        # Where in THIS rollout that deepest point happened, so a camera frame can be pulled
        # from it. reached_phase is a position in the demonstration; deepest_at is a position
        # in the rollout, and the two are not the same clock.
        deepest_at = float(phase[int(np.argmax(matched))])
        tail = matched[int(0.75 * len(matched)):]
        ended_at = float(np.median(tail)) if len(tail) else float("nan")
        # the first moment it goes and stays clearly off the demonstrated set
        threshold = np.percentile(floor, 90) * 2
        beyond = np.where(d > threshold)[0]
        left_at = float(phase[beyond[0]]) if len(beyond) else float("nan")
        rows.append(dict(episode=ep, policy=policy, outcome=outcome, frames=len(d),
                         median_off=round(float(np.median(d)), 2),
                         max_off=round(float(d.max()), 2),
                         left_manifold_at=round(left_at, 3) if left_at == left_at else "",
                         frac_off=round(float((d > threshold).mean()), 3),
                         reached_phase=round(reached, 3), ended_at_phase=round(ended_at, 3),
                         deepest_at=round(deepest_at, 3),
                         **{f"phase_{k}": round(float(np.median(d[(phase >= k / BINS) & (phase < (k + 1) / BINS)])), 2)
                            for k in range(BINS) if ((phase >= k / BINS) & (phase < (k + 1) / BINS)).any()}))

    d = pd.DataFrame(rows)
    out = HERE / f"failure_anatomy_{task_key}.csv"
    d.to_csv(out, index=False)

    print(f"\n{'policy':12} {'n':>3} {'success':>7} {'median off':>11} {'max off':>8} "
          f"{'% off':>6} {'leaves at':>10}")
    for policy in sorted(d.policy.unique()):
        g = d[d.policy == policy]
        left = pd.to_numeric(g.left_manifold_at, errors="coerce").dropna()
        print(f"{policy:12} {len(g):3d} {int((g.outcome == 'successful').sum()):7d} "
              f"{g.median_off.median():11.2f} {g.max_off.median():8.2f} "
              f"{g.frac_off.median() * 100:5.0f}% "
              f"{(f'{left.median():.0%}' if len(left) else 'never'):>10}")

    print("\nhow far through the demonstrated task each policy actually got")
    print("  (the nearest demonstration frame's position within its own episode; 1.0 is the end)")
    print(f"{'policy':12} {'reached (p95)':>14} {'last quarter spent at':>23}")
    for policy in sorted(d.policy.unique()):
        g = d[d.policy == policy]
        print(f"{policy:12} {g.reached_phase.median():13.0%} {g.ended_at_phase.median():22.0%}")
    print(f"{'(a demo)':12} {'100%':>14} {'100%':>23}")

    print(f"\nmedian distance from the demonstrated set, by tenth of the episode:")
    print(f"{'policy':12} " + " ".join(f"{k * 10:>5}%" for k in range(BINS)))
    for policy in sorted(d.policy.unique()):
        g = d[d.policy == policy]
        cells = [g[f"phase_{k}"].median() if f"phase_{k}" in g else float("nan") for k in range(BINS)]
        print(f"{policy:12} " + " ".join(f"{c:6.2f}" for c in cells))
    print(f"{'(floor)':12} " + " ".join(f"{np.median(floor):6.2f}" for _ in range(BINS)))
    print(f"\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
