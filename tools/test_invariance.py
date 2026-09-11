"""Regression gate: the same face, sitting differently, must get the same dose.

This is the claim the whole geometry layer exists to make, so it is asserted
rather than eyeballed. A few faces through a handful of synthetic camera views;
the commanded displacement must be invariant to within a few percent, and the
delivered displacement to within the detector's own noise.

Any residual above the thresholds here is a bug, not a limitation — the whole
point of working in a canonical frame is that scale, roll and position cannot
reach the arithmetic.

    .venv/Scripts/python.exe tools/test_invariance.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.parent
CORPUS = Path(r"C:\lab-corpus")
PORT = 8904

# Camera changes that must not alter the dose at all: they are a similarity
# transform of the image, and the algorithm works in a frame fitted to the face.
EXACT_VIEWS = [
    {"name": "base", "params": {}},
    {"name": "roll-18", "params": {"rollDeg": -18}},
    {"name": "roll+18", "params": {"rollDeg": 18}},
    {"name": "near", "params": {"scale": 1.3}},
    {"name": "far", "params": {"scale": 0.75}},
    {"name": "low", "params": {"dyFrac": 0.15}},
]

# Commanded dose is defined on the face, so it must be flat across all of these.
MAX_CV_COMMANDED = 0.05
# Delivered dose is re-measured through a detector that has its own ~12%
# frame-to-frame spread, so it gets a looser bound.
MAX_CV_DELIVERED = 0.20


def main() -> int:
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    frames = [f for f in manifest["frames"] if f["expr"] == "N"][:3]
    if not frames:
        raise SystemExit("no corpus frames; run tools/build_corpus.py")

    amps = json.loads((HERE / "artifacts" / "amplitudes.json").read_text(encoding="utf-8"))
    rests = json.loads((HERE / "artifacts" / "rests.json").read_text(encoding="utf-8"))

    server = subprocess.Popen(
        [sys.executable, str(HERE / "tools" / "serve.py"), "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.2)

    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=False, args=["--window-position=-2400,-2400",
                                      "--window-size=400,300"])
            page = browser.new_page(viewport={"width": 400, "height": 300})
            page.goto(f"http://127.0.0.1:{PORT}/harness.html")
            page.evaluate("o => window.__harness.init(o)",
                          {"unit": "self", "law": "additive", "calibrate": False})

            st_ = page.evaluate("() => window.__harness.selfTest()")
            assert st_["warpDirectionOk"], f"warp direction self-test failed: {st_}"
            print(f"warp direction self-test: ok ({st_['observedShiftPx']:+.0f} px)")

            ids = [f["id"] for f in frames]
            rows = page.evaluate("s => window.__harness.run(s)", {
                "frames": [{"id": f["id"], "expr": f["expr"], "group": f["group"],
                            "url": f"/corpus/frames/{f['file']}"} for f in frames],
                "alphas": [1.9],
                "views": EXACT_VIEWS,
                "impls": ["current", "normalized"],
                "amplitudes": {k: amps[k] for k in ids if k in amps},
                "rests": {k: rests[k] for k in ids if k in rests},
            })
            browser.close()
    finally:
        server.terminate()

    cmd: dict[str, list[float]] = defaultdict(list)
    dlv: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if not r["d"]["faceFound"]:
            continue
        dlv[(r["impl"], r["id"])].append(r["measuredTravel"])
        if r["impl"] == "normalized" and r.get("travelPerIpd") is not None:
            cmd[r["id"]].append(r["travelPerIpd"])

    def cv(v: list[float]) -> float:
        v = [x for x in v if x == x and math.isfinite(x)]
        if len(v) < 2 or st.fmean(v) == 0:
            return float("nan")
        return st.stdev(v) / st.fmean(v)

    print(f"\nacross {len(EXACT_VIEWS)} camera views, {len(frames)} faces:\n")
    failures = []

    print("  commanded displacement (normalized implementation)")
    for fid, vals in cmd.items():
        c = cv(vals)
        ok = c <= MAX_CV_COMMANDED
        print(f"    {fid:<10} CV={c:.4f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"commanded CV {c:.4f} > {MAX_CV_COMMANDED} for {fid}")

    print("\n  delivered displacement, re-measured from the rendered pixels")
    for impl in ("current", "normalized"):
        cvs = [cv(v) for (im, _f), v in dlv.items() if im == impl]
        cvs = [c for c in cvs if c == c]
        if not cvs:
            continue
        mean_cv = st.fmean(cvs)
        print(f"    {impl:<12} mean within-face CV = {mean_cv:.4f}"
              + (f"   (bound {MAX_CV_DELIVERED})" if impl == "normalized" else ""))
        if impl == "normalized" and mean_cv > MAX_CV_DELIVERED:
            failures.append(f"delivered CV {mean_cv:.4f} > {MAX_CV_DELIVERED}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
