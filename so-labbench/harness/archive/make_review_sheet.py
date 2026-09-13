# SO-LabBench: build a review sheet of final frames for human placement judgment.
#   python make_review_sheet.py <repo_prefix> <n_trials> [out.png]
#   e.g.  python make_review_sheet.py eval_c0_ 30
# Pulls the last frame of each trial's fixed-camera video into a labeled grid.
import sys
import json
import subprocess
import math
from pathlib import Path
from PIL import Image, ImageDraw

from labbench import DATASETS as BASE, FFMPEG

HERE = Path(__file__).parent

def last_frame(video: Path, out: Path):
    subprocess.run([FFMPEG, "-y", "-sseof", "-1", "-i", str(video),
                    "-update", "1", "-frames:v", "1", str(out)],
                   capture_output=True)
    return out.exists()

def main():
    prefix, n = sys.argv[1], int(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / f"{prefix}review.png"
    tmp = HERE / "_frames"
    tmp.mkdir(exist_ok=True)
    thumbs = []
    for i in range(1, n + 1):
        v = BASE / f"{prefix}{i}" / "videos" / "chunk-000" / "observation.images.fixed" / "episode_000000.mp4"
        f = tmp / f"{prefix}{i}.png"
        thumbs.append((i, f if v.exists() and last_frame(v, f) else None))

    tw, th, label_h = 320, 240, 24
    cols = min(5, n)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + label_h)), "black")
    draw = ImageDraw.Draw(sheet)
    for k, (i, f) in enumerate(thumbs):
        r, c = divmod(k, cols)
        x, y = c * tw, r * (th + label_h)
        if f:
            sheet.paste(Image.open(f).resize((tw, th)), (x, y + label_h))
            draw.text((x + 6, y + 4), f"trial {i}", fill="white")
        else:
            draw.text((x + 6, y + 4), f"trial {i}: MISSING", fill="red")
    sheet.save(out_path)
    print(f"review sheet -> {out_path}  ({n} trials, judge placement per tile)")

if __name__ == "__main__":
    main()
