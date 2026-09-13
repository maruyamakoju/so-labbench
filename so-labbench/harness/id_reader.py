# SO-LabBench: sample/rack identification via ArUco markers (+ QR fallback)
#
#   generate mode:  python id_reader.py generate
#       -> writes aruco_print_sheet.png (A4 landscape, markers id 0-9, ~30mm each)
#          Print at 100% scale, cut out, tape to racks / tube caps.
#   detect mode:    python id_reader.py detect [image_path]
#       -> detects markers in the image (or grabs one frame from camera 0)
#          prints: id, center_x, center_y  (one line per marker)
import sys
import numpy as np
from pathlib import Path

import cv2

HERE = Path(__file__).parent
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

def generate(path=None, ids=range(10), px=240, pad=60):
    # Beside this file, not in whatever directory the shell happened to be standing in.
    path = HERE / "aruco_print_sheet.png" if path is None else Path(path)
    cols, rows = 5, 2
    ids = list(ids)[:cols * rows]          # the grid holds ten; silently writing past it raised
    W = cols * (px + pad) + pad
    H = rows * (px + pad + 50) + pad
    sheet = np.full((H, W), 255, np.uint8)
    for k, mid in enumerate(ids):
        r, c = divmod(k, cols)
        m = cv2.aruco.generateImageMarker(DICT, mid, px)
        y = pad + r * (px + pad + 50)
        x = pad + c * (px + pad)
        sheet[y:y + px, x:x + px] = m
        cv2.putText(sheet, f"id {mid}", (x + px // 3, y + px + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.imwrite(path, sheet)
    print(f"saved {path} ({W}x{H}) - print at 100% scale, markers ~30mm")

def detect(image_path=None):
    if image_path:
        img = cv2.imread(image_path)
    else:
        cap = cv2.VideoCapture(0)
        ok, img = cap.read()
        cap.release()
        if not ok:
            print("camera read failed")
            return
    detector = cv2.aruco.ArucoDetector(DICT)
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        print("no aruco markers found")
    else:
        for c, mid in zip(corners, ids.flatten()):
            cx, cy = c[0].mean(axis=0)
            print(f"aruco id={mid} center=({cx:.0f},{cy:.0f})")
    # QR fallback
    qr = cv2.QRCodeDetector()
    data, pts, _ = qr.detectAndDecode(img)
    if data:
        print(f"qr text='{data}'")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if mode == "generate":
        generate()
    else:
        detect(sys.argv[2] if len(sys.argv) > 2 else None)
