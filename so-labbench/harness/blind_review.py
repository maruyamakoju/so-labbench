# Build a blind review set, so the person judging the trials cannot see which policy
# produced each one.
#
#   python blind_review.py prepare rqp1 eyedrops      # after the trials are recorded
#   python blind_review.py check   rqp1 eyedrops      # progress + the key, once judging is done
#
# Why: the label sheet decides the headline number, and the person reading it is the same
# person who chose the policies and has a prior about which should win. ArmnetBench's own
# operators judged their own runs with the policy in view; this is the one place where
# doing better than the reference is cheap. "prepare" assigns a shuffled review id to every
# trial, copies each trial's video and a filmstrip under that id, and writes a label sheet
# that names nothing else. The key that maps ids back to policies is written once and never
# rewritten, so the assignment cannot be redrawn after the labels start coming in.
#
# The judge opens review/<condition>/ and writes one word per row in the label sheet.
# Nothing here reads or writes the scorer's output: the human judgement stays independent.
import csv
import os
import random
import shutil
import sys
from pathlib import Path

from labbench import DATASETS, episode_video

HERE = Path(__file__).parent
MANIFEST = HERE / "rq1_manifest.csv"
SEED = 20260913          # fixed so the shuffle is reproducible from the repository alone
LABELS = ("successful", "suboptimal", "failure", "invalid")
CAMERA = os.environ.get("LABBENCH_CAMERA", "front")


def review_dir(study: str, condition: str) -> Path:
    return HERE / "review" / f"{study}_{condition}"


def key_path(study: str, condition: str) -> Path:
    return HERE / f"{study}_{condition}_review_key.csv"


def sheet_path(study: str, condition: str) -> Path:
    return HERE / f"{study}_{condition}_labels.md"


def trials(study: str, condition: str):
    if not MANIFEST.exists():
        raise SystemExit(f"no {MANIFEST.name}; run the trials first")
    prefix = f"eval_{study}_{condition}_"
    with MANIFEST.open(encoding="ascii") as f:
        rows = [r for r in csv.DictReader(f) if r["run_id"].startswith(prefix)]
    if not rows:
        raise SystemExit(f"no trials matching {prefix} in {MANIFEST.name}")
    # one row per run_id, last write wins (a retried trial appears twice)
    unique = {r["run_id"]: r for r in rows}
    return [unique[k] for k in sorted(unique)]


def make_filmstrip(video: Path, out: Path, frames: int = 12) -> bool:
    """A contact sheet of the trial, for judging at a glance before watching the video."""
    try:
        import cv2
    except ImportError:
        return False
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return False
    picks = [int(round(i * (total - 1) / (frames - 1))) for i in range(frames)]
    tiles = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        tile = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
        # the frame number is safe to show; it says nothing about which policy this is
        cv2.putText(tile, f"{idx}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(tile)
    cap.release()
    if not tiles:
        return False
    per_row = 6
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def prepare(study: str, condition: str):
    key = key_path(study, condition)
    if key.exists():
        raise SystemExit(f"{key.name} already exists. The review ids are assigned once, on purpose:\n"
                         f"redrawing them after labelling has begun would let the assignment be\n"
                         f"chosen with knowledge of the labels. Delete it by hand only if no label\n"
                         f"has been written yet.")
    rows = trials(study, condition)
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    assignment = [(f"R{position + 1:03d}", rows[trial]) for position, trial in enumerate(order)]

    out = review_dir(study, condition)
    out.mkdir(parents=True, exist_ok=True)
    copied = stripped = 0
    for review_id, row in assignment:
        video = episode_video(row["run_id"], camera=CAMERA)
        if not video.exists():
            print(f"  {review_id}: no {CAMERA} video for {row['run_id']}")
            continue
        shutil.copy(video, out / f"{review_id}.mp4")
        copied += 1
        if make_filmstrip(out / f"{review_id}.mp4", out / f"{review_id}.jpg"):
            stripped += 1

    with key.open("w", newline="", encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "run_id", "model", "start_position", "seed"])
        for review_id, row in assignment:
            w.writerow([review_id, row["run_id"], row["model"], row.get("start_position", ""), SEED])

    sheet = sheet_path(study, condition)
    lines = [
        f"# {study.upper()} 判定シート（{condition}） — 盲検",
        "",
        f"`review/{study}_{condition}/` の動画（`R001.mp4` …）を見て、`label` 欄に1語だけ書く。",
        "画像 `R001.jpg` は一覧用のコマ送り。**どの行がどの方策かは分からないようにしてある**",
        f"（対応表は `{key.name}`。判定が全部終わるまで開かない）。",
        "",
        "| 語 | 意味 |",
        "|---|---|",
        "| `successful` | 課題の目標を達成して終わった（ArmnetBench の strict success と同じ） |",
        "| `suboptimal` | 目標には達したが乱暴・落として拾い直した・容器を動かした等 |",
        "| `failure` | 達成しなかった |",
        "| `invalid` | 試行として無効（USB切断・録画失敗・物体の置き間違い・人が手を出した） |",
        "",
        "| review_id | label | notes |",
        "|---|---|---|",
    ]
    lines += [f"| {review_id} | | |" for review_id, _ in assignment]
    sheet.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{len(assignment)} trials -> {out}  ({copied} videos, {stripped} filmstrips)")
    print(f"label sheet : {sheet.name}")
    print(f"key (do not open until judging is done): {key.name}")


def check(study: str, condition: str):
    key = key_path(study, condition)
    sheet = sheet_path(study, condition)
    if not key.exists():
        raise SystemExit(f"no {key.name}; run 'prepare' first")
    with key.open(encoding="ascii") as f:
        mapping = {r["review_id"]: r for r in csv.DictReader(f)}
    labels = read_labels(sheet)
    missing = [r for r in mapping if r not in labels]
    bad = {r: l for r, l in labels.items() if l not in LABELS}
    print(f"{len(labels)}/{len(mapping)} judged" + (f", {len(missing)} remaining: {', '.join(missing[:12])}" if missing else ""))
    if bad:
        print(f"unrecognised labels: {bad}")
    if missing or bad:
        return 1
    by_policy = {}
    for review_id, label in labels.items():
        by_policy.setdefault(mapping[review_id]["model"], []).append(label)
    print("\nunblinded:")
    for policy, got in sorted(by_policy.items()):
        counts = {l: got.count(l) for l in LABELS if got.count(l)}
        print(f"  {policy:9} n={len(got):3d}  {counts}")
    print(f"\nrun: python {study}_results.py {condition}" if (HERE / f"{study}_results.py").exists() else "")
    return 0


def read_labels(sheet: Path) -> dict:
    """review_id -> label, from the markdown table."""
    out = {}
    if not sheet.exists():
        return out
    for line in sheet.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0].startswith("R") and cells[0][1:].isdigit() and cells[1]:
            out[cells[0]] = cells[1].lower()
    return out


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "check"):
        raise SystemExit(__doc__ or "usage: blind_review.py prepare|check <study> <condition>")
    study = sys.argv[2] if len(sys.argv) > 2 else "rqp1"
    condition = sys.argv[3] if len(sys.argv) > 3 else "eyedrops"
    return prepare(study, condition) if sys.argv[1] == "prepare" else check(study, condition)


if __name__ == "__main__":
    sys.exit(main() or 0)
