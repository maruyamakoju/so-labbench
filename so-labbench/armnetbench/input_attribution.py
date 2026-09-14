# Take away one kind of input at a time, and see how far the policy's command moves.
#
#   python input_attribution.py --task cable_clip --policies act smolvla --out attribution/cable_clip.csv
#
# Preregistered in so-labbench/prereg_input_attribution.md, committed before this ran.
#
# SmolVLA's command barely changes when every camera goes black - 9 to 17% of the human
# demonstrations' spread, against ACT's 82 to 132%. A policy has only three kinds of input:
# its cameras, its joint state, and (SmolVLA only) an instruction. This removes each one in
# turn on the same frames, with the same seed, and compares.
#
# The pure parts - how a state or an instruction is perturbed - live at the top with no torch
# or LeRobot import, so they can be tested in any environment. The GPU work is in main().
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TASKS = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))["tasks"]
TASK_ORDER = list(TASKS)
CAMERAS = ("front", "top", "wrist")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER = 5

# Order matters only for the report. Every name here is in the preregistration.
STATE_PERTURBATIONS = ["state_mean", "state_noise_0.5sd", "state_noise_1sd", "state_gripper_1sd"]
TASK_PERTURBATIONS = ["task_blank", "task_other"]
IMAGE_PERTURBATIONS = ["all_blank"]
FLOORS = ["identity", "resample"]
ALL_PERTURBATIONS = FLOORS + IMAGE_PERTURBATIONS + STATE_PERTURBATIONS + TASK_PERTURBATIONS


def perturb_state(state, name, sd, mean, z):
    """The joint state with one kind of information taken away or disturbed.

    `sd` and `mean` are per joint, measured from this task's sampled frames. `z` is one draw of
    six standard normals for this frame. Both noise perturbations use the SAME draw, so the
    1sd one is exactly twice the 0.5sd one in the same direction - which makes H4 a pure test
    of whether a larger push moves the command further, not a comparison of two random pushes.
    """
    state = np.asarray(state, dtype=np.float64)
    if name == "identity" or name in IMAGE_PERTURBATIONS or name in TASK_PERTURBATIONS:
        return state.copy()
    if name == "state_mean":
        return np.asarray(mean, dtype=np.float64).copy()
    if name == "state_noise_0.5sd":
        return state + 0.5 * np.asarray(sd) * np.asarray(z)
    if name == "state_noise_1sd":
        return state + 1.0 * np.asarray(sd) * np.asarray(z)
    if name == "state_gripper_1sd":
        # One standard deviation toward the far side of the mean, so the pushed value stays
        # inside the range the gripper actually visits rather than past its end stop.
        out = state.copy()
        direction = -1.0 if state[GRIPPER] > mean[GRIPPER] else 1.0
        out[GRIPPER] = state[GRIPPER] + direction * sd[GRIPPER]
        return out
    if name == "resample":
        return state.copy()
    raise ValueError(f"unknown perturbation {name!r}")


def other_task_instruction(task_key):
    """A real instruction from a different task: in-distribution text, wrong task.

    The next task in tasks.json order, wrapping. Deterministic, so a rerun asks the same
    question.
    """
    i = TASK_ORDER.index(task_key)
    return TASKS[TASK_ORDER[(i + 1) % len(TASK_ORDER)]]["instruction"]


def perturb_task(instruction, name, other):
    if name == "task_blank":
        return ""
    if name == "task_other":
        return other
    return instruction


def main():
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    sys.path.insert(0, str(HERE))
    from rig_sensitivity import blank, predict

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--policies", nargs="+", default=["act", "smolvla"])
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=None, help="first N episodes only (smoke runs)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    spec = TASKS[args.task]
    checkpoints = {p: spec["policies"][p] for p in args.policies if p in spec["policies"]}
    if not checkpoints:
        raise SystemExit(f"no runnable policy for {args.task}")
    other = other_task_instruction(args.task)

    ds = LeRobotDataset(spec["demos"])
    n_episodes = args.episodes or ds.num_episodes
    frames = []
    for ep in range(n_episodes):
        a = int(ds.meta.episodes["dataset_from_index"][ep])
        b = int(ds.meta.episodes["dataset_to_index"][ep])
        frames += list(range(a, b - 1, args.stride))       # -1 so frame+1 exists
    print(f"{args.task}: {len(frames)} frames from {n_episodes} episodes; "
          f"task_other = {other!r}", flush=True)

    # The state's spread, from the same frames the perturbations are applied to.
    states = np.stack([ds[i]["observation.state"].numpy() for i in frames]).astype(np.float64)
    state_sd = states.std(axis=0)
    state_mean = states.mean(axis=0)
    print(f"  state sd per joint: {state_sd.round(2)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for policy_name, repo in checkpoints.items():
        policy = get_policy_class(policy_name).from_pretrained(repo).to(device).eval()
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=repo)
        deltas = {name: [] for name in ALL_PERTURBATIONS}
        natural, demonstrated = [], []
        started = time.time()

        for n, index in enumerate(frames):
            item = ds[index]
            instruction = item["task"]
            images = {c: item[f"observation.images.{c}"].to(device) for c in CAMERAS}
            state = item["observation.state"].numpy().astype(np.float64)
            z = np.random.default_rng(index).standard_normal(len(JOINTS))

            def observation(imgs, st):
                out = {f"observation.images.{c}": imgs[c][None] for c in CAMERAS}
                out["observation.state"] = torch.tensor(st, dtype=torch.float32, device=device)[None]
                return out

            clean = predict(policy, pre, post, observation(images, state), instruction, seed=0)
            demonstrated.append(item["action"].numpy())

            nxt = ds[index + 1]
            nxt_images = {c: nxt[f"observation.images.{c}"].to(device) for c in CAMERAS}
            nxt_state = nxt["observation.state"].numpy().astype(np.float64)
            natural.append(np.abs(predict(policy, pre, post, observation(nxt_images, nxt_state),
                                          instruction, seed=0) - clean))

            for name in ALL_PERTURBATIONS:
                seed = 1 if name == "resample" else 0
                imgs = {c: blank(images[c]) for c in CAMERAS} if name == "all_blank" else images
                st = perturb_state(state, name, state_sd, state_mean, z)
                text = perturb_task(instruction, name, other)
                got = predict(policy, pre, post, observation(imgs, st), text, seed=seed)
                deltas[name].append(np.abs(got - clean))

            if (n + 1) % 25 == 0:
                rate = (time.time() - started) / (n + 1)
                print(f"  {policy_name}: {n + 1}/{len(frames)} frames, {rate:.2f} s/frame, "
                      f"eta {rate * (len(frames) - n - 1) / 60:.0f} min", flush=True)

        natural = np.mean(natural, axis=0)
        sigma = np.std(demonstrated, axis=0)
        for name in ALL_PERTURBATIONS:
            mean = np.mean(deltas[name], axis=0)
            rows.append(dict(
                task=args.task, policy=policy_name, perturbation=name, frames=len(frames),
                **{f"delta_{j}": round(float(m), 4) for j, m in zip(JOINTS, mean)},
                delta_mean=round(float(mean.mean()), 4),
                relative_mean=round(float((mean / np.maximum(natural, 1e-6)).mean()), 4),
                **{f"natural_{j}": round(float(v), 4) for j, v in zip(JOINTS, natural)},
                **{f"sigma_{j}": round(float(v), 3) for j, v in zip(JOINTS, sigma)},
                **{f"state_sd_{j}": round(float(v), 3) for j, v in zip(JOINTS, state_sd)}))
        # Written after each policy, so a crash in the second loses only the second.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="ascii") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n{policy_name}:", flush=True)
        for r in (r for r in rows if r["policy"] == policy_name):
            print(f"  {r['perturbation']:20} {r['delta_mean']:8.3f}", flush=True)
        del policy
        torch.cuda.empty_cache()

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    code = main()
    # Same reason as rig_sensitivity.py: torch and the hub leave threads that can keep a
    # finished process alive, and a finished cell must not block the cells behind it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
