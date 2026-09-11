"""Smoke-test the demo page without a human in front of the camera.

Loads the page headless (so getUserMedia fails and it falls back to a corpus
face), lets it settle, then sweeps head roll and records what each
implementation actually delivered at each angle. That is the demo's central
claim measured rather than asserted: the current morph's dose should swing with
roll, the normalized one should not.

Screenshots go to C:\\lab-corpus\\renders — they contain CFD faces, which must
not enter the repo.

    .venv/Scripts/python.exe tools/demo_smoke.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.parent
SHOTS = Path(r"C:\lab-corpus") / "renders" / "demo"
PORT = 8903


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.9)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--rolls", type=str, default="-20,-10,0,10,20")
    args = ap.parse_args()
    rolls = [float(v) for v in args.rolls.split(",")]

    SHOTS.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, str(HERE / "tools" / "serve.py"), "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.2)

    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            launch = {"headless": not args.headed}
            if args.headed:
                launch["args"] = ["--window-position=-2400,-2400"]
            browser = pw.chromium.launch(**launch)
            page = browser.new_page(viewport={"width": 1500, "height": 1400})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            def on_console(m):
                if m.type == "error" and "XNNPACK" not in m.text                         and not m.text.startswith("INFO:"):
                    errors.append(f"console.error: {m.text}")
            page.on("console", on_console)

            page.goto(f"http://127.0.0.1:{PORT}/demo.html")
            page.wait_for_function(
                "() => document.getElementById('engine').textContent.includes('MediaPipe')"
                " || document.getElementById('engine').textContent.includes('no camera')",
                timeout=60000)
            time.sleep(3)
            engine = page.text_content("#engine")
            print(f"engine: {engine}")

            # The page falls back to a corpus face when there is no camera.
            face = page.text_content("#faceName")
            print(f"face:   {face}")

            set_slider = ("([sel, v]) => { const el = document.querySelector(sel);"
                          " el.value = String(v);"
                          " el.dispatchEvent(new Event('input')); }")
            def settle(frames: int = 90, timeout_s: float = 60.0):
                """Wait for N *rendered* frames, not N seconds.

                Headless renders far slower than real time, and both the alpha
                tween and the landmark smoothing advance per frame — so a wall
                clock sleep can look settled while the pipeline is still moving.
                """
                start = page.evaluate("() => window.__demoFrames || 0")
                page.wait_for_function(
                    "n => (window.__demoFrames || 0) >= n",
                    arg=start + frames, timeout=timeout_s * 1000)

            page.evaluate(set_slider, ["#alpha", args.alpha])
            settle(120)

            rows = []
            for r in rolls:
                page.evaluate(set_slider, ["#roll", r])
                settle(90)
                a = page.text_content("#sA") or ""
                b = page.text_content("#sB") or ""
                va = parse_delivered(a)
                vb = parse_delivered(b)
                rows.append({"roll": r, "current": va, "normalized": vb})
                print(f"  roll {r:+5.0f}deg   current {fmt(va)}   normalized {fmt(vb)}")
                page.screenshot(path=str(SHOTS / f"demo_roll{int(r):+03d}.png"))

            ok = [r for r in rows if r["current"] == r["current"]
                  and r["normalized"] == r["normalized"]]
            if len(ok) >= 3:
                ca = [r["current"] for r in ok]
                cb = [r["normalized"] for r in ok]
                cv_a = statistics.pstdev(ca) / statistics.fmean(ca)
                cv_b = statistics.pstdev(cb) / statistics.fmean(cb)
                print(f"\n  delivered dose across roll: "
                      f"current CV {cv_a:.3f}   normalized CV {cv_b:.3f}")
            else:
                print("\n  not enough valid readings to compute a CV")

            (SHOTS / "roll_sweep.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8")

            page.screenshot(path=str(SHOTS / "demo_full.png"), full_page=True)
            browser.close()

            if errors:
                print("\npage errors:")
                for e in errors[:10]:
                    print("  !", e)
                return 1
    finally:
        server.terminate()

    print(f"\nscreenshots in {SHOTS}")
    return 0


def parse_delivered(text: str) -> float:
    # "delivered 0.0628 · target 0.0908"
    try:
        return float(text.split("delivered", 1)[1].split("·", 1)[0].strip())
    except (IndexError, ValueError):
        return float("nan")


def fmt(v: float) -> str:
    return f"{v:.4f}" if v == v else "  —   "


if __name__ == "__main__":
    raise SystemExit(main())
