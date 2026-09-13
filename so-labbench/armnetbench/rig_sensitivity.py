# Which of the differences between our rig and ArmnetBench's actually change what the
# policy does? Run in the new environment (~/.venvs/lerobot2). GPU only, no robot.
#
#   python rig_sensitivity.py --stride 30                 # both policies, all perturbations
#   python rig_sensitivity.py --stride 120 --episodes 2   # smoke run
#   python rig_sensitivity.py --policies act              # one policy
#
# The reproduction cannot match the benchmark's rig exactly: our front camera is 4:3 where
# theirs is 16:9, the cameras will be aimed by hand, the room is lit differently, and the
# objects are look-alikes rather than the same items. Rather than guess which of those
# matter, measure them. Every perturbation below is applied to the benchmark's own
# demonstration frames, and we ask how far the policy's output moves.
#
# What this can and cannot say. It is a shift-detection instrument, not a success
# predictor: we already know that offline agreement fails to predict rollout success on
# this hardware (SmolVLA fits the demonstrations more tightly than ACT and succeeds a
# third as often). So a large reading here means "the policy sees a different world and
# acts differently", which is a necessary condition for the difference to matter on the
# robot, not a sufficient one. A near-zero reading is the informative direction: a rig
# difference the policy's output does not respond to cannot explain a change in success.
#
# Scale. A raw action delta is meaningless without something to compare it to, so every
# delta is also reported relative to `natural_step`, the change in the policy's own output
# between two consecutive frames of an untouched demonstration. A perturbation that moves
# the output less than one frame of ordinary motion is below the noise the policy already
# lives with. `identity` (must read 0) and `shift_1px` (the smallest real perturbation)
# bound the floor.
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

HERE = Path(__file__).parent
TASKS = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))["tasks"]
DEFAULT_TASK = "eye_drops_to_basket"
CAMERAS = ("front", "top", "wrist")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# The benchmark's gripper scale is not ours: its demonstrations sit at 1.6 when grasping and
# reach 34.4 when open, so our arm's 40 would call 95% of their frames closed. Derived from
# their own data by a two-means split (midpoint 18.0), same as the reproduction's config.
GRIPPER_CLOSED_BELOW = 15.0
CLOSE_T = GRIPPER_CLOSED_BELOW
OUT = HERE / "rig_sensitivity.csv"


# ---------------------------------------------------------------------------- perturbations
# Each takes and returns a CHW float tensor in [0, 1], the same thing the dataset hands the
# policy, so applying one here is equivalent to the camera having seen it that way.


def shift(img, dx=0, dy=0):
    """Translate by whole pixels, replicating the edge rather than wrapping: a camera that
    is aimed slightly off sees new scene at one edge, not the opposite edge's content."""
    out = img
    if dx:
        out = torch.roll(out, shifts=dx, dims=2)
        if dx > 0:
            out[:, :, :dx] = out[:, :, dx:dx + 1]
        else:
            out[:, :, dx:] = out[:, :, dx - 1:dx]
    if dy:
        out = torch.roll(out, shifts=dy, dims=1)
        if dy > 0:
            out[:, :dy, :] = out[:, dy:dy + 1, :]
        else:
            out[:, dy:, :] = out[:, dy - 1:dy, :]
    return out


def resize(img, size):
    return torch.nn.functional.interpolate(img[None], size=size, mode="bilinear", align_corners=False)[0]


def zoom(img, factor):
    """Camera closer (>1) or further (<1) than the reference, framing the same centre."""
    c, h, w = img.shape
    if factor > 1:
        ch, cw = int(round(h / factor)), int(round(w / factor))
        top, left = (h - ch) // 2, (w - cw) // 2
        return resize(img[:, top:top + ch, left:left + cw], (h, w))
    sh, sw = int(round(h * factor)), int(round(w * factor))
    small = resize(img, (sh, sw))
    out = img.new_zeros((c, h, w))
    out[:] = small.mean(dim=(1, 2), keepdim=True)
    top, left = (h - sh) // 2, (w - sw) // 2
    out[:, top:top + sh, left:left + sw] = small
    return out


def rotate(img, degrees):
    theta = np.deg2rad(degrees)
    m = torch.tensor([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0]],
                     dtype=img.dtype, device=img.device)[None]
    grid = torch.nn.functional.affine_grid(m, (1, *img.shape), align_corners=False)
    return torch.nn.functional.grid_sample(img[None], grid, mode="bilinear",
                                           padding_mode="border", align_corners=False)[0]


def aspect_4_3(img):
    """What a 4:3 camera sees from the 16:9 reference position: the same vertical extent,
    a narrower horizontal one. This is our front camera's known deviation - the C270 holds
    20 fps only at 640x480, where the benchmark used a 16:9 Pi Camera at 1024x576.

    Approximate, and in a knowable direction. It models the same lens behind a narrower
    sensor. The real cameras also differ in field of view (a C270 is around 60 degrees
    diagonal, a Pi Camera Module 3 around 75), so the actual front view will differ by more
    than this simulates, not less. Read the number as a lower bound on the aspect effect.
    """
    c, h, w = img.shape
    keep = int(round(h * 4 / 3))
    if keep >= w:
        return img
    left = (w - keep) // 2
    return resize(img[:, :, left:left + keep], (h, w))


def brightness(img, factor):
    return (img * factor).clamp(0, 1)


def blur(img, sigma=2.0):
    radius = int(3 * sigma)
    xs = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
    kernel = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    out = img[None]
    out = torch.nn.functional.conv2d(out, kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1),
                                     padding=(0, radius), groups=3)
    out = torch.nn.functional.conv2d(out, kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1),
                                     padding=(radius, 0), groups=3)
    return out[0]


def blank(img, grey=False):
    """A camera that is unplugged, mis-indexed, or pointed at nothing. `grey` fills with the
    frame's own mean colour instead of black, which is the fairer "no information" control:
    black is itself an unusual input."""
    return img.new_full(img.shape, 0.0) if not grey else img.new_ones(img.shape) * img.mean(dim=(1, 2), keepdim=True)


def build_perturbations():
    """name -> {camera: fn}. A camera missing from the dict is passed through untouched.

    Two rows are not image perturbations and are filled in by the measurement loop:
    `identity` (must read exactly zero, so anything else means the harness is wrong) and
    `resample` (the same frame drawn again, which is this policy's own noise floor).
    """
    p = {"identity": {}, "resample": {}}
    p["shift_1px"] = {c: (lambda i: shift(i, dx=1)) for c in CAMERAS}
    p["front_aspect_4_3"] = {"front": aspect_4_3}
    for cam in CAMERAS:
        for px in (20, 40, 80):
            p[f"{cam}_shift_x_{px}"] = {cam: (lambda i, px=px: shift(i, dx=px))}
        p[f"{cam}_shift_y_40"] = {cam: (lambda i: shift(i, dy=40))}
        for factor, label in ((0.9, "out"), (1.1, "in")):
            p[f"{cam}_zoom_{label}_10pct"] = {cam: (lambda i, f=factor: zoom(i, f))}
        p[f"{cam}_rotate_5deg"] = {cam: (lambda i: rotate(i, 5.0))}
        p[f"{cam}_blank"] = {cam: (lambda i: blank(i))}
        p[f"{cam}_uniform"] = {cam: (lambda i: blank(i, grey=True))}
    # How much of this policy's output is vision at all? If blanking every camera at once
    # barely moves the action, the policy is largely replaying a trajectory from
    # proprioception, and no amount of matching the benchmark's cameras will change what it
    # does - which would be the single most useful thing to know before spending a session
    # aiming them. This is the upper bound on every camera perturbation above.
    p["all_blank"] = {c: (lambda i: blank(i)) for c in CAMERAS}
    p["all_uniform"] = {c: (lambda i: blank(i, grey=True)) for c in CAMERAS}
    for factor, label in ((0.7, "dark"), (1.3, "bright")):
        p[f"all_{label}_30pct"] = {c: (lambda i, f=factor: brightness(i, f)) for c in CAMERAS}
    p["all_blur_sigma2"] = {c: (lambda i: blur(i)) for c in CAMERAS}
    return p


# ---------------------------------------------------------------------------- measurement


def predict(policy, pre, post, observation, task, seed=0):
    """One action from a fresh chunk.

    The seed matters: SmolVLA samples its action by flow matching, so two calls on the
    same frame differ by up to 3.2 units - more than a joint moves between consecutive
    frames. Without a fixed seed every perturbation reading here would be that noise.
    ACT is deterministic and ignores the seed. The noise is not swept under the rug: it
    is measured as the `resample` row, which is the honest floor for this policy.
    """
    batch = {k: v for k, v in observation.items()}
    batch["task"] = task
    torch.manual_seed(seed)
    policy.reset()
    with torch.no_grad():
        action = post(policy.select_action(pre(batch)))
    return action.squeeze().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=30, help="sample every Nth frame of every episode")
    ap.add_argument("--episodes", type=int, default=None, help="use only the first N episodes (smoke runs)")
    ap.add_argument("--task", default=DEFAULT_TASK, choices=sorted(TASKS),
                    help="which ArmnetBench task to measure; each has its own demonstrations and checkpoints")
    ap.add_argument("--policies", nargs="+", default=["act", "smolvla"])
    ap.add_argument("--phase", choices=["all", "critical"], default="all",
                    help="'critical' samples only the moments around a gripper transition, where "
                         "the task is decided; averaging over a whole episode buries them among "
                         "frames where the arm is parked and nothing could go wrong")
    ap.add_argument("--window", type=float, default=0.5,
                    help="seconds either side of a gripper transition to count as critical")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    task = TASKS[args.task]
    checkpoints = {p: task["policies"][p] for p in args.policies if p in task["policies"]}
    missing = [p for p in args.policies if p not in task["policies"]]
    if missing:
        print(f"no checkpoint for {missing} on {args.task}; skipping those")
    if not checkpoints:
        raise SystemExit(f"no runnable policy for {args.task}")
    perturbations = build_perturbations()
    print(f"task {args.task}: {len(perturbations)} perturbations x {len(checkpoints)} policies", flush=True)

    ds = LeRobotDataset(task["demos"])
    n_episodes = args.episodes or ds.num_episodes
    frames = []
    for ep in range(n_episodes):
        a = int(ds.meta.episodes["dataset_from_index"][ep])
        b = int(ds.meta.episodes["dataset_to_index"][ep])
        if args.phase == "all":
            frames += list(range(a, b - 1, args.stride))     # -1 so frame+1 exists for the natural step
            continue
        # The task is decided in the second around the gripper changing state: closing on the
        # object, or opening over the basket. A perturbation that moves the output only while
        # the arm is parked cannot cost a trial; one that moves it here can. Averaging over a
        # whole episode mixes the two and reports something in between.
        grip = np.array([ds[i]["action"][5].item() for i in range(a, b)])
        closed = grip < GRIPPER_CLOSED_BELOW
        transitions = [a + i for i in range(1, len(closed)) if closed[i] != closed[i - 1]]
        half = int(round(args.window * ds.fps))
        critical = set()
        for t in transitions:
            critical.update(range(max(a, t - half), min(b - 1, t + half + 1)))
        frames += sorted(critical)[::max(1, args.stride // 6)]
    label = "frames" if args.phase == "all" else "frames around a gripper transition"
    print(f"{len(frames)} sampled {label} from {n_episodes} episodes", flush=True)
    if not frames:
        raise SystemExit("no frames selected")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for policy_name in checkpoints:
        repo = checkpoints[policy_name]
        policy = get_policy_class(policy_name).from_pretrained(repo).to(device).eval()
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=repo)

        deltas = {name: [] for name in perturbations}
        flips = {name: 0 for name in perturbations}
        natural, demonstrated = [], []
        started = time.time()
        for n, index in enumerate(frames):
            item = ds[index]
            task = item["task"]
            images = {c: item[f"observation.images.{c}"].to(device) for c in CAMERAS}
            state = item["observation.state"].to(device)

            def observation(imgs):
                out = {f"observation.images.{c}": imgs[c][None] for c in CAMERAS}
                out["observation.state"] = state[None]
                return out

            clean = predict(policy, pre, post, observation(images), task, seed=0)

            # The policy's own noise floor: the identical frame, a different draw. For a
            # deterministic policy this is zero; for a sampling one it is the amount by
            # which its output moves for no reason at all, and no rig difference smaller
            # than this can be said to change the policy's behaviour.
            resampled = predict(policy, pre, post, observation(images), task, seed=1)
            deltas["resample"].append(np.abs(resampled - clean))
            flips["resample"] += int((clean[5] < CLOSE_T) != (resampled[5] < CLOSE_T))

            # the scale everything is measured against: one frame of ordinary motion
            nxt = ds[index + 1]
            next_images = {c: nxt[f"observation.images.{c}"].to(device) for c in CAMERAS}
            next_state = nxt["observation.state"].to(device)
            next_obs = {f"observation.images.{c}": next_images[c][None] for c in CAMERAS}
            next_obs["observation.state"] = next_state[None]
            natural.append(np.abs(predict(policy, pre, post, next_obs, task) - clean))
            demonstrated.append(item["action"].numpy())

            for name, spec in perturbations.items():
                if name == "resample":
                    continue          # measured above, against a different draw of the same frame
                if not spec:
                    deltas[name].append(np.zeros(6))
                    continue
                perturbed = {c: (spec[c](images[c]) if c in spec else images[c]) for c in CAMERAS}
                got = predict(policy, pre, post, observation(perturbed), task)
                deltas[name].append(np.abs(got - clean))
                flips[name] += int((clean[5] < CLOSE_T) != (got[5] < CLOSE_T))

            if (n + 1) % 25 == 0:
                rate = (time.time() - started) / (n + 1)
                print(f"  {policy_name}: {n + 1}/{len(frames)} frames, {rate:.2f} s/frame, "
                      f"eta {rate * (len(frames) - n - 1) / 60:.0f} min", flush=True)

        natural = np.mean(natural, axis=0)
        sigma = np.std(demonstrated, axis=0)
        print(f"\n{policy_name}: natural one-frame step per joint = {natural.round(2)}", flush=True)
        for name in perturbations:
            mean = np.mean(deltas[name], axis=0)
            relative = mean / np.maximum(natural, 1e-6)
            rows.append(dict(task=args.task, policy=policy_name, perturbation=name, phase=args.phase, frames=len(frames),
                             **{f"delta_{j}": round(float(m), 3) for j, m in zip(JOINTS, mean)},
                             delta_mean=round(float(mean.mean()), 3),
                             relative_mean=round(float(relative.mean()), 3),
                             relative_gripper=round(float(relative[5]), 3),
                             gripper_flip_rate=round(flips[name] / len(frames), 4),
                             **{f"natural_{j}": round(float(v), 3) for j, v in zip(JOINTS, natural)},
                             **{f"sigma_{j}": round(float(v), 2) for j, v in zip(JOINTS, sigma)}))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", newline="", encoding="ascii") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        del policy
        torch.cuda.empty_cache()

    header = f"\n{'policy':9} {'perturbation':24} {'delta':>7} {'x natural':>10} {'x resample':>11} {'grip flip':>10}"
    print(header)
    for policy_name in checkpoints:
        mine = [r for r in rows if r["policy"] == policy_name]
        floor = next((r["delta_mean"] for r in mine if r["perturbation"] == "resample"), 0.0)
        for r in sorted(mine, key=lambda r: -r["relative_mean"]):
            shown = f"{r['delta_mean'] / floor:11.2f}" if floor > 1e-9 else f"{'n/a':>11}"
            print(f"{r['policy']:9} {r['perturbation']:24} {r['delta_mean']:7.2f} "
                  f"{r['relative_mean']:10.2f} {shown} {r['gripper_flip_rate']:10.3f}")
        if floor > 1e-9:
            print(f"{'':9} this policy moves {floor:.2f} on its own, so anything near that is not a rig effect")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    code = main()
    # Leave without waiting for the interpreter to tear itself down. On the first cell of the
    # SmolVLA grid this process wrote its CSV and its last line of output and then sat there,
    # at roughly zero CPU, never exiting - torch and the hub leave threads behind that can
    # outlive the work. The shell loop was blocked on it, so seven tasks that were queued
    # behind a finished one would have waited all night for nothing.
    #
    # Everything that matters is already on disk by this point: the CSV is written and closed
    # above, and the streams are flushed here. There is no state left to lose by not unwinding
    # politely, and a batch cell that has produced its result should not be able to hold up
    # the cells behind it.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
