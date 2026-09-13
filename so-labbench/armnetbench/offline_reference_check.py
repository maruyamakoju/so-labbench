# RQ-P1 exploratory: how well do the public checkpoints predict the reference demos they
# were trained on? Run in the NEW environment (~/.venvs/lerobot2), GPU only, no robot.
#   python offline_reference_check.py [stride]
# Teacher-forced: every stride-th frame of every reference episode is fed with its true
# state and three images; the first action of the predicted chunk is compared with the
# demonstrated action. Reports per-joint mean absolute error and, for the gripper, the
# closure agreement (both < 40 or both >= 40). This is the sanity floor: if a checkpoint
# cannot reproduce its own demos offline, a rollout failure on our rig says nothing.
import sys, csv, time
import numpy as np, torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors, get_policy_class

HERE = Path(__file__).parent
DATASET = "pravsels/object_top_shelf_reset_remote"
CHECKPOINTS = {"act": "pravsels/act_eyedrops_basket_20k", "smolvla": "pravsels/smolvla_eyedrops_basket"}
CLOSE_T = 40.0
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ds = LeRobotDataset(DATASET)
    ep_from = ds.meta.episodes["dataset_from_index"] if hasattr(ds.meta, "episodes") else None
    rows = []
    for ptype, repo in CHECKPOINTS.items():
        policy = get_policy_class(ptype).from_pretrained(repo).to("cuda").eval()
        pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=repo)
        t0 = time.time()
        for ep in range(ds.num_episodes):
            a, b = int(ds.meta.episodes["dataset_from_index"][ep]), int(ds.meta.episodes["dataset_to_index"][ep])
            pred, tgt = [], []
            with torch.no_grad():
                for i in range(a, b, stride):
                    it = ds[i]
                    obs = {k: v for k, v in it.items() if k.startswith("observation.")}
                    obs["task"] = it["task"]
                    # Seed every call: SmolVLA samples its action by flow matching, and two
                    # calls on one frame differ by up to 3.2 units - more than a joint moves
                    # between consecutive frames. Without this the error below is that noise
                    # as much as it is the fit. ACT is deterministic and ignores the seed.
                    torch.manual_seed(0)
                    policy.reset()
                    act = post(policy.select_action(pre(obs)))
                    pred.append(act.squeeze().float().cpu().numpy()); tgt.append(it["action"].numpy())
            pred, tgt = np.array(pred), np.array(tgt)
            mae = np.abs(pred - tgt).mean(axis=0)
            agree = float(((pred[:, 5] < CLOSE_T) == (tgt[:, 5] < CLOSE_T)).mean())
            rows.append(dict(policy=ptype, ep=ep, frames=len(pred), **{f"mae_{j}": round(float(m), 2) for j, m in zip(JOINTS, mae)},
                             gripper_closure_agreement=round(agree, 3), target_closed_frac=round(float((tgt[:, 5] < CLOSE_T).mean()), 3)))
            print(f"{ptype:8} ep{ep:02d} mae={mae.round(1)} closure_agree={agree:.2f}", flush=True)
        print(f"{ptype}: {time.time()-t0:.0f}s for {ds.num_episodes} episodes", flush=True)
        del policy; torch.cuda.empty_cache()
    out = HERE / "offline_reference_check.csv"
    with out.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\nsummary (mean over episodes):")
    for ptype in CHECKPOINTS:
        g = [r for r in rows if r["policy"] == ptype]
        print(f"  {ptype:8} " + " ".join(f"{j}={np.mean([r[f'mae_{j}'] for r in g]):.1f}" for j in JOINTS)
              + f"  closure_agree={np.mean([r['gripper_closure_agreement'] for r in g]):.3f}")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
