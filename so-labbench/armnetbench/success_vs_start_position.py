# Does the published success depend on where the object started?
#
#   python success_vs_start_position.py            # act and smolvla on eye_drops_to_basket
#
# A policy that learned to manipulate should find the object wherever it is, within the
# range it was taught. A policy that learned to replay a stereotyped trajectory succeeds
# when the object happens to be where the average demonstration put it, and fails as it
# moves away. The benchmark publishes both halves of that test and nobody has run it: the
# teleoperated demonstrations that define "where the object usually is", and thirty labelled
# rollouts per policy whose first frame shows where it actually was.
#
# So: detect the eye-drops carton in the first frame of every rollout, from the same top
# camera at the same resolution as the demonstrations, and ask whether the successes sit
# closer to the demonstrations' centre than the failures do.
#
# This is an analysis of someone else's published data, not of our rig. No robot, no GPU,
# and it needs four video files rather than the benchmark's full 60 GB.
#
# What it cannot say: a position effect does not prove a policy is replaying, and its
# absence does not prove it is not. Thirty rollouts is thirty rollouts. It is one piece of
# evidence about how much of the reported success is manipulation and how much is the
# object being in the usual place.
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATASET = "armnet/armnetbench_v01_lerobot_so101"
TASK = "Put the eye drops into the basket"
CAMERA = "videos/observation.images.top"
POLICIES = ("act", "smolvla")
OUT = HERE / "success_vs_start_position.csv"

# The carton is a saturated blue against a navy mat and brown cardboard. Same thresholds as
# the start-position derivation, so the two sets of coordinates mean the same thing.
BLUE_LO, BLUE_HI = (95, 120, 120), (130, 255, 255)
MIN_AREA = 120


def detect_box(frame):
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LO, BLUE_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    if not contours:
        return None
    (cx, cy), (w, h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return float(cx), float(cy), float(max(w, h) * min(w, h))


def main():
    import cv2
    from huggingface_hub import hf_hub_download

    episodes = pd.read_parquet(hf_hub_download(DATASET, f"meta/episodes/chunk-000/file-000.parquet",
                                               repo_type="dataset"))
    episodes["task"] = episodes["tasks"].apply(
        lambda x: x[0] if hasattr(x, "__len__") and not isinstance(x, str) else str(x))
    wanted = episodes[(episodes.task == TASK) & (episodes.policy_type.isin(POLICIES))]
    print(f"{len(wanted)} rollouts to read, across {wanted.policy_type.nunique()} policies")

    rows = []
    for (chunk, file_index), group in wanted.groupby([f"{CAMERA}/chunk_index", f"{CAMERA}/file_index"]):
        path = f"{CAMERA}/chunk-{int(chunk):03d}/file-{int(file_index):03d}.mp4"
        print(f"  {path}: {len(group)} rollouts", flush=True)
        local = hf_hub_download(DATASET, path, repo_type="dataset")
        cap = cv2.VideoCapture(local)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        for _, ep in group.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(ep[f"{CAMERA}/from_timestamp"] * fps)))
            ok, frame = cap.read()
            if not ok:
                continue
            found = detect_box(frame)
            rows.append(dict(episode=int(ep.episode_index), policy=ep.policy_type,
                             success=ep.success_class == "successful", outcome=ep.success_class,
                             cx=round(found[0], 1) if found else "", cy=round(found[1], 1) if found else "",
                             area=int(found[2]) if found else "", found=bool(found)))
        cap.release()

    d = pd.DataFrame(rows)
    # The misses are written as empty cells for the CSV's sake, which makes the column an
    # object dtype and silently breaks any arithmetic on it.
    for column in ("cx", "cy", "area"):
        d[column] = pd.to_numeric(d[column], errors="coerce")
    with OUT.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    seen = d[d.found]
    print(f"\nbox located in {len(seen)}/{len(d)} first frames")

    # Where the demonstrations put it, from the same camera and the same detector.
    demos = HERE / "start_positions_top.csv"
    centre = None
    if demos.exists():
        dm = pd.read_csv(demos)
        dm = dm[dm.found == 1]
        centre = (dm.cx.mean(), dm.cy.mean())
        print(f"demonstrations put it at ({centre[0]:.0f}, {centre[1]:.0f}) on average, "
              f"spread {dm.cx.std():.0f} x {dm.cy.std():.0f} px")

    print(f"\n{'policy':9} {'outcome':12} {'n':>3} {'centre (px)':>16} {'spread':>12}"
          + ("   distance from the demos' centre" if centre else ""))
    for policy in sorted(seen.policy.unique()):
        for label, subset in [("successful", seen[(seen.policy == policy) & seen.success]),
                              ("not successful", seen[(seen.policy == policy) & ~seen.success])]:
            if subset.empty:
                continue
            line = (f"{policy:9} {label:12} {len(subset):3d} "
                    f"({subset.cx.mean():6.0f},{subset.cy.mean():5.0f}) "
                    f"{subset.cx.std():5.0f}x{subset.cy.std():<5.0f}")
            if centre is not None:
                dist = np.hypot(subset.cx - centre[0], subset.cy - centre[1])
                line += f"   {dist.mean():6.0f} px  (median {dist.median():.0f})"
            print(line)

    if centre is not None:
        print("\nIf the successes sit systematically closer to the demonstrations' centre than")
        print("the failures do, the reported success rate is partly a statement about where the")
        print("object was put, not only about what the policy learned. If the two distances are")
        print("the same, that reading is not supported.")
    print(f"\nwrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
