"""Does the appearance morph work through the real code path?

The offline prototype showed that morphing a neutral photograph toward the same
person's smiling photograph produces teeth, folds and shading. This runs the
same idea through the shipped pipeline instead: seed the processor with the
person's smile so it lands in the bank exactly as it would mid-session, then
render their neutral frame at a range of intensities and lay the results next to
what the geometric warp produces from the same input.

Output goes to artifacts/figures/faces/, which the .gitignore keeps out of the
repo -- these are CFD faces.

    .venv/Scripts/python.exe tools/test_appearance.py --ids BF-001,BM-002,WM-004
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent.parent
CORPUS = Path(r"C:\lab-corpus")
OUT = HERE / "artifacts" / "figures" / "faces"
PORT = 8910
ALPHAS = [1.35, 1.5, 1.9]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="BF-001,BM-002,WM-004")
    ap.add_argument("--seed-expr", default="HO", choices=["HC", "HO"],
                    help="which smile to bank: HO is toothy, HC closed-mouth")
    args = ap.parse_args()
    ids = [i for i in args.ids.split(",") if i]

    amps = json.loads((HERE / "artifacts" / "amplitudes.json").read_text(encoding="utf-8"))
    rests = json.loads((HERE / "artifacts" / "rests.json").read_text(encoding="utf-8"))

    frames = []
    for pid in ids:
        n = CORPUS / "frames" / f"{pid}_N.jpg"
        s = CORPUS / "frames" / f"{pid}_{args.seed_expr}.jpg"
        if not n.exists() or not s.exists():
            print(f"  skip {pid}: missing a frame")
            continue
        frames.append({"id": pid, "expr": "N", "group": pid.split("-")[0],
                       "url": f"/corpus/frames/{pid}_N.jpg",
                       "seedUrl": f"/corpus/frames/{pid}_{args.seed_expr}.jpg"})
    if not frames:
        raise SystemExit("no usable identities")

    server = subprocess.Popen(
        [sys.executable, str(HERE / "tools" / "serve.py"), "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.2)

    from playwright.sync_api import sync_playwright
    rows = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False, args=[
                "--window-position=-2400,-2400", "--window-size=400,300",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows"])
            page = browser.new_page(viewport={"width": 400, "height": 300})
            page.on("pageerror", lambda e: print(f"  [page error] {e}", file=sys.stderr))
            page.goto(f"http://127.0.0.1:{PORT}/harness.html")
            page.evaluate("o => window.__harness.init(o)",
                          {"unit": "self", "law": "additive",
                           "calibrate": False, "mode": "appearance"})
            rows = page.evaluate("s => window.__harness.run(s)", {
                "frames": frames,
                "alphas": ALPHAS,
                "views": [{"name": "base", "params": {}}],
                "impls": ["current", "normalized"],
                "amplitudes": {k: amps[k] for k in ids if k in amps},
                "rests": {k: rests[k] for k in ids if k in rests},
                "exportPngs": True,
            })
            browser.close()
    finally:
        server.terminate()

    def decode(b64):
        return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                            cv2.IMREAD_COLOR)

    got = {}
    for r in rows:
        if r.get("png"):
            got[(r["id"], r["impl"], r["alpha"])] = decode(r["png"])
        if r.get("shamPng") and (r["id"], "sham") not in got:
            got[(r["id"], "sham")] = decode(r["shamPng"])
        if r["impl"] == "normalized":
            print(f"  {r['id']} a={r['alpha']}: mode={r.get('usedMode')} "
                  f"bank={r.get('bankCount')}  "
                  f"detect {r['msDetect']:.1f}ms  morph {r['msWarp']:.1f}ms")

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["sham"] + [("current", a) for a in ALPHAS] + [("normalized", a) for a in ALPHAS]
    labels = ["unmorphed"] + [f"warp {a}" for a in ALPHAS] + [f"own-smile {a}" for a in ALPHAS]

    panels = []
    for pid in ids:
        row = []
        for cdef in cols:
            key = (pid, "sham") if cdef == "sham" else (pid, cdef[0], cdef[1])
            img = got.get(key)
            if img is None:
                row = []
                break
            # Whole face. The corpus framing puts the eyes at y=274 with an
            # interpupillary distance of 95 px, so this is centred on it.
            row.append(img[110:600, 420:860])
        if row:
            panels.append(row)

    if not panels:
        print("nothing rendered")
        return 1

    ph, pw = panels[0][0].shape[:2]
    pad = 6
    sheet = np.full((len(panels) * (ph + pad) + pad + 26,
                     len(cols) * (pw + pad) + pad, 3), 245, np.uint8)
    for r, row in enumerate(panels):
        for c, tile in enumerate(row):
            y = 26 + pad + r * (ph + pad)
            x = pad + c * (pw + pad)
            sheet[y:y + ph, x:x + pw] = tile
    for c, lab in enumerate(labels):
        cv2.putText(sheet, lab, (pad + c * (pw + pad), 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    dest = OUT / "7-appearance-vs-warp.png"
    cv2.imwrite(str(dest), sheet)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
