# Do published SO-101 datasets share a scale? Read their own metadata and see.
#
#   python public_gripper_ranges.py
#
# A LeRobot action value is not an angle and not a distance. It is a percentage of the
# calibrated range of the arm it was recorded on, and that range is whatever the operator
# swept when they ran lerobot-calibrate. So two datasets agree about what "40" means only if
# their two operators happened to calibrate the same way - and nothing in the format records
# whether they did.
#
# This asks the cheapest possible version of that question. Every LeRobot dataset publishes
# meta/stats.json, which carries the min and max of every action dimension over the whole
# dataset. If the community shared a scale, the gripper's commanded maximum would land in
# roughly the same place everywhere, because every operator opens the gripper fully at some
# point in fifty demonstrations. It does not.
#
# The script also looks for a calibration file in each repository, because that is the thing
# that would make the numbers recoverable. Finding none is the second half of the result.
#
# What this cannot say: nothing here shows a policy fails on someone else's arm. It shows the
# published information is not sufficient to determine what its action values mean physically.
# That is a claim about the metadata, not about the policies.
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "public_gripper_ranges.csv"

# SO-101 single-arm datasets on the Hub, picked to span the community rather than one lab:
# the benchmark's own task recordings, LeRobot's reference dataset, and independent uploads.
DATASETS = [
    "pravsels/object_top_shelf_reset_remote",
    "villekuosmanen/armnetbench_block_stack",
    "villekuosmanen/armnetbench_ring_insert",
    "villekuosmanen/armnetbench_tool_insert",
    "pravsels/cable_clip_remote_v2",
    "lerobot/svla_so101_pickplace",
    "jackvial/so101_pickplace_success_120_v2",
    "hbseong/record-pick-and-place-pos5-so101",
    "5hadytru/so101_bench_real_1_v2.1",
    "szk1ck/so101-ycb-pickplace",
]

# Also checked for a shipped calibration, including one checkpoint repository.
CALIBRATION_REPOS = [
    ("pravsels/object_top_shelf_reset_remote", "dataset"),
    ("villekuosmanen/armnetbench_block_stack", "dataset"),
    ("armnet/armnetbench_v01_lerobot_so101", "dataset"),
    ("pravsels/act_eyedrops_basket_20k", "model"),
]

# The SO-101 follower's six joints, in the order LeRobot writes them.
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def action_stats(repo: str):
    """min/max of each action dimension, from the dataset's own published stats."""
    from huggingface_hub import hf_hub_download

    # v3.0 keeps stats in meta/stats.json; v2.1 sometimes only has meta/episodes_stats.jsonl,
    # where the dataset-wide range is the envelope of the per-episode ones.
    try:
        path = hf_hub_download(repo, "meta/stats.json", repo_type="dataset")
        stats = json.loads(Path(path).read_text(encoding="utf-8"))
        action = stats["action"]
        return [float(v) for v in action["min"]], [float(v) for v in action["max"]]
    except Exception:
        path = hf_hub_download(repo, "meta/episodes_stats.jsonl", repo_type="dataset")
        lo = hi = None
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            action = json.loads(line)["stats"]["action"]
            a = [float(v) for v in action["min"]]
            b = [float(v) for v in action["max"]]
            lo = a if lo is None else [min(x, y) for x, y in zip(lo, a)]
            hi = b if hi is None else [max(x, y) for x, y in zip(hi, b)]
        return lo, hi


def has_calibration(repo: str, kind: str):
    """Any file in the repository that could let a third party reinterpret the numbers."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo, repo_type=kind)
    hits = [f for f in files if "calibration" in f.lower() or f.endswith("calibration.json")]
    return len(files), hits


def main():
    rows = []
    for repo in DATASETS:
        try:
            lo, hi = action_stats(repo)
        except Exception as exc:                    # a repo can be renamed or gated
            print(f"  {repo}: unreadable ({type(exc).__name__})")
            continue
        if len(lo) < len(JOINTS):
            print(f"  {repo}: {len(lo)} action dims, not a six-joint single arm - skipped")
            continue
        g, p = JOINTS.index("gripper"), JOINTS.index("shoulder_pan")
        rows.append(dict(dataset=repo,
                         gripper_min=round(lo[g], 1), gripper_max=round(hi[g], 1),
                         gripper_span=round(hi[g] - lo[g], 1),
                         pan_min=round(lo[p], 1), pan_max=round(hi[p], 1)))
        print(f"  {repo}: gripper {lo[g]:.1f} .. {hi[g]:.1f}")

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dataset", "gripper_min", "gripper_max",
                                                "gripper_span", "pan_min", "pan_max"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT.name} ({len(rows)} datasets)")

    spans = sorted(r["gripper_span"] for r in rows)
    # One dataset is on a radian scale rather than a percentage - all six joints inside +-1.57.
    percent = [s for s in spans if s > 5]
    if percent:
        print(f"gripper travel actually commanded: {min(percent):.1f} to {max(percent):.1f} "
              f"({max(percent) / min(percent):.1f}x), over {len(percent)} datasets on a "
              f"percentage scale")
    off_scale = [r["dataset"] for r in rows if r["gripper_span"] <= 5]
    for repo in off_scale:
        print(f"  {repo} is not on the same scale at all (every joint within +-1.57: radians)")

    print("\ndoes anyone ship the calibration that would make these numbers recoverable?")
    for repo, kind in CALIBRATION_REPOS:
        try:
            count, hits = has_calibration(repo, kind)
        except Exception as exc:
            print(f"  {repo}: unreadable ({type(exc).__name__})")
            continue
        print(f"  {repo:52} {count:4d} files  "
              f"{'calibration: ' + ', '.join(hits) if hits else 'no calibration file'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
