# Record how far the camera sat from the reference pose on every recorded trial.
#   python alignment_log.py                      # every run in rq1_manifest.csv
#   python alignment_log.py eval_rq1_pilot_act_ 5   # or an explicit prefix + count
#
# Reads the first frame of each episode's fixed-camera video, so it measures the view
# the policy actually consumed, needs no access to the camera, and can run long after
# the session. This is why camera alignment never has to block a run: a drifted
# session stays interpretable because the drift is a recorded number.
#
# Writes rq1_alignment.csv, keyed by run_id. Deliberately a separate file from the
# preregistered trial log, whose schema is frozen.
import os
import sys
import csv
import cv2
from pathlib import Path

import camera_pose_check as cpc
from camera_pose_check import load, measure, C5_SHIFT_PX
from labbench import CAMERA_NAME, episode_video, read_manifest, reference_frame, study_output

HERE = Path(__file__).parent

FIELDS = ["run_id", "condition", "model", "method", "confidence", "aligned",
          "dx", "dy", "scale", "rot", "shift_px", "shift_vs_c5",
          "brightness", "edge_density"]


def first_frame(run_id: str):
    try:
        v = episode_video(run_id, camera=CAMERA_NAME)
    except FileNotFoundError:
        return None
    if not v.exists():
        return None
    cap = cv2.VideoCapture(str(v))
    ok, frame = cap.read()
    cap.release()
    # Match at the size the reference was loaded at. Forcing 640x480 here silently squashed
    # a 16:9 reference into a 4:3 grid, and the coarse fallback then multiplied a 640x360
    # window against a 640x480 array and raised - but only on trials too dark to match,
    # which are exactly the ones the fallback exists for.
    return cv2.resize(frame, (cpc.W, cpc.H)) if ok else None


def runs_from_args(argv):
    """A prefix and a count, or - with no arguments - every trial of one study/condition.

    The study and condition matter: with no filter this walks the whole manifest, so
    finishing one condition would rewrite the alignment record of every other."""
    if len(argv) >= 2 and not argv[0].startswith("-"):
        prefix, n = argv[0], int(argv[1])
        return [(f"{prefix}{i}", "", "") for i in range(1, n + 1)], None, None
    study = argv[argv.index("--study") + 1] if "--study" in argv else None
    condition = argv[argv.index("--condition") + 1] if "--condition" in argv else None
    rows = read_manifest(study=study, condition=condition)
    if not rows:
        raise SystemExit("no matching trials in the manifest, and no prefix given")
    return [(r["run_id"], r.get("condition", ""), r.get("model", "")) for r in rows], study, condition


def main():
    runs, study, condition = runs_from_args(sys.argv[1:])
    ref = load(reference_frame(), reference=True)
    out = study_output("alignment.csv", study, condition)
    rows, missing = [], []
    for run_id, cond, model in runs:
        frame = first_frame(run_id)
        if frame is None:
            missing.append(run_id)
            continue
        r = measure(ref, frame)
        r.pop("advice", None)
        # Room lighting belongs in the record too. It moved during the pilot and took
        # the capture rate and the alignment measurement down with it, so a low edge
        # density is the first thing to check when `method` reads coarse.
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r["brightness"] = round(float(g.mean()), 1)
        r["edge_density"] = round(float((cv2.Canny(g, 60, 160) > 0).mean() * 100), 2)
        rows.append(dict(run_id=run_id, condition=cond, model=model, **r))

    if not rows:
        print("no episodes found" + (f" ({len(missing)} missing)" if missing else ""))
        return 1

    with out.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    shifts = sorted(r["shift_px"] for r in rows)
    worst = max(rows, key=lambda r: r["shift_px"])
    n_ok = sum(1 for r in rows if r["aligned"])
    print(f"measured {len(rows)} trials" + (f", {len(missing)} missing" if missing else ""))
    print(f"  within advisory tolerance : {n_ok}/{len(rows)}")
    print(f"  median shift              : {shifts[len(shifts) // 2]:.0f}px "
          f"({shifts[len(shifts) // 2] / C5_SHIFT_PX:.2f} x the C5 move)")
    print(f"  worst                     : {worst['shift_px']:.0f}px on {worst['run_id']}")
    print(f"  spread (min-max)          : {shifts[0]:.0f}-{shifts[-1]:.0f}px")
    print(f"csv -> {out}")
    coarse = [r for r in rows if r["method"] == "coarse"]
    if coarse:
        print(f"\n{len(coarse)} trials fell back to the coarse estimate, whose numbers are"
              "\nnot trustworthy. Their brightness and edge density:")
        for r in coarse:
            print(f"  {r['run_id']:32} brightness {r['brightness']:5.1f}  edges {r['edge_density']:5.2f}%")
        lit = [r for r in rows if r["method"] == "features"]
        if lit:
            print(f"  against {sum(r['brightness'] for r in lit)/len(lit):.1f} and "
                  f"{sum(r['edge_density'] for r in lit)/len(lit):.2f}% on the trials that matched.")
        print("  Too dark to measure is a lighting fact, not a camera-position fact.")

    firm = [r["shift_px"] for r in rows if r["method"] == "features"]
    if firm and (max(firm) - min(firm)) > C5_SHIFT_PX / 2:
        print("\nNOTE: spread across reliably measured trials is a large fraction of the C5\n"
              "perturbation. Treat C5 as confounded with drift unless the camera is fixed down.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
