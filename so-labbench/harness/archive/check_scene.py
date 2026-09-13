# SO-LabBench: scene verification primitives for the C3 closed-loop condition.
#   - detect_red_caps(img): HSV red-blob detection -> list of (cx, cy, area)
#   - zone_occupied(img, baseline, rect): frame-difference occupancy for a zone
#   - capture(): grab one frame from the fixed camera
#   self-test:  python check_scene.py selftest   (synthetic images, no camera needed)
import sys
import numpy as np
import cv2

def capture(index=0):
    cap = cv2.VideoCapture(index)
    ok, img = cap.read()
    cap.release()
    return img if ok else None

def detect_red_caps(img, min_area=150):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 110, 70), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 110, 70), (180, 255, 255))
    mask = cv2.morphologyEx(m1 | m2, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= min_area:
            m = cv2.moments(c)
            out.append((m["m10"] / m["m00"], m["m01"] / m["m00"], a))
    return sorted(out, key=lambda t: -t[2])

def zone_occupied(img, baseline, rect, thresh=25, frac=0.06):
    """rect=(x,y,w,h). True if the zone differs from the empty-scene baseline."""
    x, y, w, h = rect
    a = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(baseline[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0))
    return float((diff > thresh).mean()) > frac

def selftest():
    # synthetic: gray desk, red disk at (200,150), zone at right
    base = np.full((480, 640, 3), 120, np.uint8)
    img = base.copy()
    cv2.circle(img, (200, 150), 22, (40, 40, 210), -1)          # red cap (BGR)
    caps = detect_red_caps(img)
    assert len(caps) == 1 and abs(caps[0][0] - 200) < 3 and abs(caps[0][1] - 150) < 3, caps
    # zone occupancy
    img2 = base.copy()
    cv2.rectangle(img2, (500, 300), (560, 400), (200, 200, 200), -1)  # object appears in zone
    assert zone_occupied(img2, base, (480, 280, 140, 140)) is True
    assert zone_occupied(base, base, (480, 280, 140, 140)) is False
    # red detection must ignore non-red object
    assert len(detect_red_caps(img2)) == 0
    print("check_scene selftest: ALL PASS")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        img = capture()
        print("caps:", detect_red_caps(img) if img is not None else "no camera")
