# Group a dataset's episodes by camera pose.
#   python dataset_pose_audit.py [dataset] [step]
# Walks the episodes in order, measuring each against the current group's anchor and
# starting a new group when feature matching can no longer bridge the two views.
# Answers "was this recorded from one camera position?" - which the RQ1 baseline and
# the C5 shift condition both depend on, and which nothing else in the harness checks.
import sys
import cv2
from pathlib import Path

from camera_pose_check import measure, MIN_INLIERS
import camera_pose_check as cpc
from labbench import CAMERA_NAME, episode_indices, episode_video


def first_frame(dataset: str, e: int):
    v = episode_video(dataset, e, camera=CAMERA_NAME)
    if not v.exists():
        return None
    cap = cv2.VideoCapture(str(v))
    ok, f = cap.read()
    cap.release()
    # Match at the size the pose checker works in, not a fixed 640x480: a 16:9 dataset
    # squashed into a 4:3 grid compares the wrong shapes.
    return cv2.resize(f, (cpc.W, cpc.H)) if ok else None


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "so101_pick_remote"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    eps = episode_indices(dataset)[::step]
    if not eps:
        raise SystemExit(f"no episodes in {dataset}")

    # Find the first episode whose video actually reads. Anchoring on a missing one used to
    # hand None to the matcher and crash inside OpenCV.
    anchor_ep = anchor = None
    for e in eps:
        anchor = first_frame(dataset, e)
        if anchor is not None:
            anchor_ep = e
            break
    if anchor is None:
        raise SystemExit(f"no readable {CAMERA_NAME} video in {dataset}")

    groups = []          # each: dict(anchor=ep, members=[(ep, shift)])
    cur = dict(anchor=anchor_ep, members=[(anchor_ep, 0.0)])
    for e in [x for x in eps if x > anchor_ep]:
        f = first_frame(dataset, e)
        if f is None:
            continue
        r = measure(anchor, f)
        if r["method"] == "features":          # measure() only says "features" above MIN_INLIERS
            cur["members"].append((e, r["shift_px"]))
        else:
            groups.append(cur)
            anchor_ep, anchor = e, f
            cur = dict(anchor=e, members=[(e, 0.0)])
    groups.append(cur)

    print(f"{dataset}: {len(eps)} episodes sampled, {len(groups)} camera pose group(s)\n")
    for i, g in enumerate(groups, 1):
        eps_in = [e for e, _ in g["members"]]
        shifts = [s for _, s in g["members"]]
        print(f"  group {i}: ep{eps_in[0]}-{eps_in[-1]}  ({len(eps_in)} episodes, anchor ep{g['anchor']})")
        print(f"           within-group shift: max {max(shifts):.0f}px, median {sorted(shifts)[len(shifts) // 2]:.0f}px")
    if len(groups) > 1:
        print("\nMore than one pose: the policy was trained across camera positions, so a\n"
              "baseline 'same as training' is a choice of group, not a single fact. Say which.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
