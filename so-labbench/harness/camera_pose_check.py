# Measure how far the fixed camera sits from the reference pose.
#   python camera_pose_check.py [live.png] [reference.png] [--json]
# Reports the rigid offset as camera motion, so alignment (and the C5 restore step)
# is a number rather than an eyeball.
#
# The verdict is ADVISORY. Nothing in the benchmark blocks on it: alignment_log.py
# records the residual of every recorded trial, so a session that drifts stays
# analysable instead of being thrown away. Chase the numbers when it is cheap,
# record them when it is not.
import os
import sys
import json
import math
import cv2
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
OUT_DIR = HERE / "_frames"
DEFAULT_LIVE = OUT_DIR / "camera_live_raw.png"
DEFAULT_REF = OUT_DIR / "camera_reference.png"

W, H = 640, 480

# C5 moves the camera 5 cm sideways. At this rig's working distance that lands near
# 60 px of frame, so residuals are reported as a fraction of it: a setup sitting at
# 0.2 cannot be confused with the perturbation the benchmark is trying to measure.
# Every threshold below is overridable per session via the environment.
C5_SHIFT_PX = float(os.environ.get("ALIGN_C5_SHIFT_PX", 60.0))
TOL_PX = float(os.environ.get("ALIGN_TOL_PX", 20.0))
TOL_SCALE = float(os.environ.get("ALIGN_TOL_SCALE", 0.05))
TOL_ROT_DEG = float(os.environ.get("ALIGN_TOL_ROT", 2.5))
MIN_INLIERS = int(os.environ.get("ALIGN_MIN_INLIERS", 12))


def set_match_size(reference: Path):
    """Fix the working height from the reference's aspect ratio.

    Everything is matched at 640 px wide. The height has to follow the reference - 480 for
    the 4:3 C270 recordings, 360 for the 16:9 ArmnetBench frames - or a 16:9 reference gets
    squashed into a 4:3 grid and the phase-correlation fallback multiplies arrays of
    different shapes. It used to be decided by sniffing for "reference" in the filename,
    which meant a reference not named that way silently kept the previous size.
    """
    global H
    img = cv2.imread(str(reference))
    if img is None:
        raise SystemExit(f"cannot read {reference}")
    H = int(round(W * img.shape[0] / img.shape[1]))
    return H


def load(p: Path, reference: bool = False):
    """Read an image at the working size. Pass reference=True for the image that DEFINES
    that size; it must be loaded before anything compared against it."""
    if reference:
        set_match_size(p)
    img = cv2.imread(str(p))
    if img is None:
        raise SystemExit(f"cannot read {p}")
    return cv2.resize(img, (W, H))


def _detector():
    # SIFT roughly doubles the surviving matches on this scene versus ORB, which is
    # what makes the estimate steady enough to aim a camera by. ORB stays as a fallback
    # for builds without it.
    try:
        return cv2.SIFT_create(nfeatures=4000), cv2.NORM_L2
    except AttributeError:
        return cv2.ORB_create(nfeatures=4000), cv2.NORM_HAMMING


def estimate(ref, live):
    g_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g_live = cv2.cvtColor(live, cv2.COLOR_BGR2GRAY)
    det, norm = _detector()
    k1, d1 = det.detectAndCompute(g_live, None)
    k2, d2 = det.detectAndCompute(g_ref, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None, 0, 0
    bf = cv2.BFMatcher(norm)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return None, len(good), 0
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0, maxIters=5000)
    n_inl = int(inl.sum()) if inl is not None else 0
    return M, len(good), n_inl


def decompose(M):
    """M maps live coordinates onto reference coordinates."""
    return (float(M[0, 2]), float(M[1, 2]),
            float(math.hypot(M[0, 0], M[1, 0])),
            float(math.degrees(math.atan2(M[1, 0], M[0, 0]))))


def advise(dx, dy, scale, rot):
    """Corrections phrased as camera motion. Signs verified against synthetic transforms:
    dx<0 means live content sits right of the reference, dy>0 means it sits above it,
    scale<1 means live is the larger view, rot<0 means live is rolled clockwise."""
    ok = (abs(dx) <= TOL_PX and abs(dy) <= TOL_PX
          and abs(scale - 1) <= TOL_SCALE and abs(rot) <= TOL_ROT_DEG)
    lines = []
    if abs(dx) > TOL_PX:
        lines.append(f"pan {'RIGHT' if dx < 0 else 'LEFT'}  ({abs(dx):.0f}px)")
    if abs(dy) > TOL_PX:
        lines.append(f"tilt {'UP' if dy > 0 else 'DOWN'}  ({abs(dy):.0f}px)")
    if abs(scale - 1) > TOL_SCALE:
        lines.append(f"move {'BACK' if scale < 1 else 'FORWARD'}  ({abs(scale - 1) * 100:.0f}%)")
    if abs(rot) > TOL_ROT_DEG:
        lines.append(f"roll {'COUNTER-CLOCKWISE' if rot > 0 else 'CLOCKWISE'}  ({abs(rot):.1f}deg)")
    return ok, lines


def edge_map(img):
    # Gradient magnitude survives the lighting and screen-content differences between
    # a training frame and today's desk far better than raw intensity does.
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                      cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    return cv2.normalize(m, None, 0.0, 1.0, cv2.NORM_MINMAX)


def coarse_estimate(ref, live):
    # Fallback for when the views are too far apart for feature matching: sweep scale
    # and take the translation from phase correlation at each step, keep the best peak.
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    R = edge_map(ref) * win
    L = edge_map(live)
    best = None
    for s in np.arange(0.60, 1.81, 0.02):
        A = cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1.0 / float(s))
        Lw = cv2.warpAffine(L, A, (W, H)) * win
        (dx, dy), resp = cv2.phaseCorrelate(R, Lw)
        if best is None or resp > best[3]:
            best = (float(s), float(dx), float(dy), float(resp))
    return best


def measure(ref, live, return_transform: bool = False):
    """Residual of live against ref. Always returns a dict; `method` says how much
    to trust it. Coarse results carry the opposite sign convention for translation,
    so they are negated here to keep one meaning across the whole harness.

    return_transform hands back the fitted transform as well, so a caller that wants to
    draw the overlay does not run the whole SIFT and RANSAC fit a second time."""
    M, n_good, n_inl = estimate(ref, live)
    if M is not None and n_inl >= MIN_INLIERS:
        dx, dy, scale, rot = decompose(M)
        method, conf = "features", float(n_inl)
    else:
        s, cdx, cdy, resp = coarse_estimate(ref, live)
        dx, dy, scale, rot = -cdx, -cdy, 1.0 / s, 0.0
        method, conf = "coarse", float(resp)
    ok, lines = advise(dx, dy, scale, rot)
    result = dict(method=method, confidence=round(conf, 3), aligned=bool(ok),
                  dx=round(dx, 1), dy=round(dy, 1), scale=round(scale, 4), rot=round(rot, 2),
                  shift_px=round(math.hypot(dx, dy), 1),
                  shift_vs_c5=round(math.hypot(dx, dy) / C5_SHIFT_PX, 2),
                  advice=lines)
    return (result, M, n_inl) if return_transform else result


def main():
    argv = [a for a in sys.argv[1:] if a != "--json"]
    as_json = "--json" in sys.argv[1:]
    live_p = Path(argv[0]) if argv else DEFAULT_LIVE
    ref_p = Path(argv[1]) if len(argv) > 1 else DEFAULT_REF
    ref = load(ref_p, reference=True)     # sets the working size; must come first
    live = load(live_p)

    r, M, n_inl = measure(ref, live, return_transform=True)
    if as_json:
        print(json.dumps(r))
        return 0

    print(f"method {r['method']} (confidence {r['confidence']})")
    print(f"  translation : dx={r['dx']:+.1f}px  dy={r['dy']:+.1f}px"
          f"   |shift|={r['shift_px']:.0f}px = {r['shift_vs_c5']:.2f} x the C5 move")
    print(f"  scale       : {r['scale']:.3f}  ({(r['scale'] - 1) * 100:+.1f}%)")
    print(f"  rotation    : {r['rot']:+.2f} deg")
    print(f"\n{'ALIGNED' if r['aligned'] else 'NOT ALIGNED'} "
          f"(advisory tolerance {TOL_PX:.0f}px / {TOL_SCALE * 100:.0f}% / {TOL_ROT_DEG} deg)")
    for s in r["advice"]:
        print(f"  - {s}")
    if not r["aligned"]:
        print("\nThis does not block a run. alignment_log.py records the residual of\n"
              "every trial, so drift stays measurable after the fact.")

    if M is not None and n_inl >= MIN_INLIERS:
        warped = cv2.warpAffine(live, M, (W, H))
        cv2.imwrite(str(OUT_DIR / "pose_check_overlay.png"),
                    np.hstack([ref, cv2.addWeighted(ref, 0.5, warped, 0.5, 0)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
