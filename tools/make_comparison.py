"""A side-by-side visual comparison, on a face that can actually be published.

Every quantitative result here is measured on the Chicago Face Database, whose
licence forbids redistributing images — so none of those renders can go in the
repo or a slide. This uses one of the developer's own lab test recordings
instead, which has no such restriction.

Numbers say the dose is more equal. They do not say whether it *looks* right,
and that is the question a 2D warp can fail. So: sham, current and normalized,
same frame, cropped to the mouth, at both presets.

    .venv/Scripts/python.exe tools/make_comparison.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import facegeom as fg  # noqa: E402
from build_corpus import FRAME_H, FRAME_W, TARGET_EYE_Y, TARGET_IPD, reframe  # noqa: E402

HERE = Path(__file__).parent.parent
SELF_DIR = Path(r"C:\lab-corpus") / "self"
# Contains a face, so it goes in the subdirectory the .gitignore keeps out
# of the repo, even though it is the developer's own.
OUT = HERE / "artifacts" / "figures" / "faces"
PORT = 8905

LAB = Path(r"C:\Users\amuel\OneDrive\Desktop\wisc-psychology-lab")
DEFAULT_CLIP = LAB / "session_20260812-115425.mkv"


def extract_frame(clip: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {index} of {clip}")
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    ap.add_argument("--frame", type=int, default=12)
    ap.add_argument("--alphas", default="1.35,1.9")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]

    SELF_DIR.mkdir(parents=True, exist_ok=True)
    img = extract_frame(args.clip, args.frame)

    lm = fg.Landmarker()
    det = lm.detect(img)
    lm.close()
    if det is None:
        raise SystemExit("no face in that frame; try a different --frame")
    framed = reframe(img, det[0], 1.0, TARGET_IPD, TARGET_EYE_Y)
    src_path = SELF_DIR / "SELF-001_N.jpg"
    cv2.imwrite(str(src_path), framed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"wrote {src_path} ({FRAME_W}x{FRAME_H})")

    server = subprocess.Popen(
        [sys.executable, str(HERE / "tools" / "serve.py"), "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.2)

    from playwright.sync_api import sync_playwright
    renders: dict[str, np.ndarray] = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=False,
                args=["--window-position=-2400,-2400", "--window-size=400,300"])
            page = browser.new_page(viewport={"width": 400, "height": 300})
            page.goto(f"http://127.0.0.1:{PORT}/harness.html")
            page.evaluate("o => window.__harness.init(o)",
                          {"unit": "population", "law": "additive", "calibrate": False})
            rows = page.evaluate("s => window.__harness.run(s)", {
                "frames": [{"id": "SELF-001", "expr": "N", "group": "SELF",
                            "url": "/corpus/self/SELF-001_N.jpg"}],
                "alphas": alphas,
                "views": [{"name": "base", "params": {}}],
                "impls": ["current", "normalized"],
                "exportPngs": True,
            })
            browser.close()
    finally:
        server.terminate()

    for r in rows:
        key = f"{r['impl']}_a{r['alpha']}"
        renders[key] = decode(r["png"])
        if "sham" not in renders:
            renders["sham"] = decode(r["shamPng"])

    # Crop to the mouth, in the reframed geometry, so the difference is visible
    # at all — at full frame the morph is a few dozen pixels.
    lm = fg.Landmarker()
    d = lm.detect(framed)
    lm.close()
    pts = d[0][:, :2]
    cx, cy = pts[fg.OUTER_LIP].mean(0)
    half_w, half_h = int(TARGET_IPD * 1.5), int(TARGET_IPD * 1.05)
    x0, y0 = int(cx - half_w), int(cy - half_h)
    x1, y1 = int(cx + half_w), int(cy + half_h)

    def crop(a: np.ndarray) -> np.ndarray:
        return a[max(0, y0):y1, max(0, x0):x1]

    cols = ["sham"] + [f"{impl}_a{a}" for a in alphas for impl in ("current", "normalized")]
    labels = ["unmorphed"] + [f"{impl} a={a}" for a in alphas
                              for impl in ("current", "normalized")]

    tiles = [crop(renders[c]) for c in cols if c in renders]
    if not tiles:
        raise SystemExit("nothing rendered")
    h = min(t.shape[0] for t in tiles)
    tiles = [t[:h] for t in tiles]

    pad = 8
    strip = np.full((h + 34, sum(t.shape[1] for t in tiles) + pad * (len(tiles) + 1), 3),
                    245, np.uint8)
    x = pad
    for t, lab in zip(tiles, labels):
        strip[30:30 + h, x:x + t.shape[1]] = t
        cv2.putText(strip, lab, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (40, 40, 40), 1, cv2.LINE_AA)
        x += t.shape[1] + pad

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "5-visual-comparison.png"
    cv2.imwrite(str(dest), strip)
    print(f"wrote {dest}")
    print("  (developer's own face — your own face, so it is yours to share; kept out of git by default)")
    return 0


def decode(b64: str) -> np.ndarray:
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


if __name__ == "__main__":
    _ = os, json
    raise SystemExit(main())
