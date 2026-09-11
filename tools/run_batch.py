"""Drive the batch harness in Chromium and collect measurements.



Starts the static server, opens the harness page, feeds it the corpus in chunks,

and writes one JSON row per (face, view, implementation, alpha) to

artifacts/runs/<runId>/rows.jsonl.



Timings collected here are NOT quotable as performance numbers. Headless

Chromium falls back to a software GPU, which makes canvas work roughly an order

of magnitude slower than the real thing. Performance figures come from the demo

page on real hardware, and finally from the Electron app.



    .venv/Scripts/python.exe tools/run_batch.py --n 120

"""



from __future__ import annotations



import argparse

import base64

import json

import subprocess

import sys

import time

from datetime import datetime

from pathlib import Path



HERE = Path(__file__).parent.parent

CORPUS = Path(r"C:\lab-corpus")

RUNS = HERE / "artifacts" / "runs"

# Rendered frames are derived from CFD images, so they stay outside the repo.

RENDERS = CORPUS / "renders"

PORT = 8901



# Production presets, plus the sham which the harness always renders anyway.

DEFAULT_ALPHAS = [1.35, 1.9]



VIEWS_BASE = [{"name": "base", "params": {}}]

VIEWS_POSE = [

    {"name": "base", "params": {}},

    {"name": "roll-15", "params": {"rollDeg": -15}},

    {"name": "roll+15", "params": {"rollDeg": 15}},

    {"name": "roll+25", "params": {"rollDeg": 25}},

    {"name": "pitch-15", "params": {"pitchDeg": -15}},

    {"name": "pitch+15", "params": {"pitchDeg": 15}},

    {"name": "yaw+20", "params": {"yawDeg": 20}},

    {"name": "near", "params": {"scale": 1.35}},

    {"name": "far", "params": {"scale": 0.72}},

    {"name": "low-in-frame", "params": {"dyFrac": 0.18}},

]





def load_manifest() -> list[dict]:

    mf = CORPUS / "manifest.json"

    if not mf.exists():

        raise SystemExit(f"missing {mf}; run tools/build_corpus.py first")

    return json.loads(mf.read_text(encoding="utf-8"))["frames"]





def load_amplitudes() -> dict[str, float]:

    """Per-identity smile amplitude, measured offline from their own smile

    photographs. Supplying it simulates a converged calibration, which is the

    ceiling any live calibration scheme could reach."""

    p = HERE / "artifacts" / "amplitudes.json"

    if not p.exists():

        return {}

    return json.loads(p.read_text(encoding="utf-8"))





def main() -> int:

    ap = argparse.ArgumentParser()

    ap.add_argument("--n", type=int, default=120, help="identities")

    ap.add_argument("--expr", default="N", help="which photograph to morph")

    ap.add_argument("--pose", action="store_true", help="include the pose ablation")

    ap.add_argument("--unit", default="self", choices=["self", "population"])

    ap.add_argument("--law", default="additive",

                    choices=["additive", "multiplicative", "endpoint"])

    ap.add_argument("--calibrated", action="store_true",

                    help="feed each identity's measured amplitude")

    ap.add_argument("--chunk", type=int, default=10)

    ap.add_argument("--headed", action="store_true",

                    help="real GPU, window parked off-screen; much faster")

    ap.add_argument("--label", default="")

    ap.add_argument("--export-pngs", action="store_true",

                    help="save rendered frames for offline scoring by OpenFace")

    args = ap.parse_args()



    frames_all = [f for f in load_manifest() if f["expr"] == args.expr]

    ids = sorted({f["id"] for f in frames_all})[: args.n]

    frames = [

        {"id": f["id"], "expr": f["expr"], "group": f["group"],

         "url": f"/corpus/frames/{f['file']}"}

        for f in frames_all if f["id"] in set(ids)

    ]

    if not frames:

        raise SystemExit("no frames selected")



    amps = load_amplitudes() if args.calibrated else {}
    rests = {}
    if args.calibrated:
        rp = HERE / "artifacts" / "rests.json"
        if rp.exists():
            rests = json.loads(rp.read_text(encoding="utf-8"))

    views = VIEWS_POSE if args.pose else VIEWS_BASE



    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args.label:

        run_id += f"-{args.label}"

    out_dir = RUNS / run_id

    out_dir.mkdir(parents=True, exist_ok=True)



    server = subprocess.Popen(

        [sys.executable, str(HERE / "tools" / "serve.py"), "--port", str(PORT)],

        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,

    )

    time.sleep(1.2)



    from playwright.sync_api import sync_playwright



    rows_path = out_dir / "rows.jsonl"

    written = 0

    t0 = time.time()



    try:

        with sync_playwright() as pw:

            launch = {"headless": not args.headed}

            if args.headed:

                # Off-screen rather than hidden: keeps the real GPU, keeps the

                # desktop usable during a long run.

                launch["args"] = ["--window-position=-2400,-2400",

                                  "--window-size=400,300"]

            browser = pw.chromium.launch(**launch)

            page = browser.new_page(viewport={"width": 400, "height": 300})

            page.on("pageerror", lambda e: print(f"  [page error] {e}", file=sys.stderr))

            page.goto(f"http://127.0.0.1:{PORT}/harness.html")



            opts = {"unit": args.unit, "law": args.law,

                    "calibrate": False}   # stills contain no smile events

            info = page.evaluate(

                "o => window.__harness.init(o)", opts)

            print(f"harness ready: {info}")



            st = page.evaluate("() => window.__harness.selfTest()")

            print(f"warp self-test: {st}")

            if not st["warpDirectionOk"]:

                raise SystemExit(

                    "warp direction self-test FAILED -- the backward map is "

                    "inverted, every measurement below would be meaningless")



            with rows_path.open("w", encoding="utf-8") as fh:

                for i in range(0, len(frames), args.chunk):

                    chunk = frames[i:i + args.chunk]

                    spec = {

                        "frames": chunk,

                        "alphas": DEFAULT_ALPHAS,

                        "views": views,

                        "impls": ["current", "normalized"],

                        "amplitudes": amps,
                        "rests": {k: rests[k] for k in
                                  {f["id"] for f in chunk} & set(rests)},

                        "exportPngs": bool(args.export_pngs),

                    }

                    rows = page.evaluate(

                        "s => window.__harness.run(s)", spec)

                    for r in rows:

                        png = r.pop("png", None)

                        sham = r.pop("shamPng", None)

                        if png:

                            d = RENDERS / run_id

                            d.mkdir(parents=True, exist_ok=True)

                            stem = f"{r['id']}_{r['impl']}_{r['view']}"

                            (d / f"{stem}_a{r['alpha']}.png").write_bytes(

                                base64.b64decode(png))

                            sp = d / f"{stem}_sham.png"

                            if sham and not sp.exists():

                                sp.write_bytes(base64.b64decode(sham))

                        fh.write(json.dumps(r) + "\n")
                    written += len(rows)

                    el = time.time() - t0

                    pct = (i + len(chunk)) / len(frames)

                    print(f"  {i+len(chunk)}/{len(frames)} faces  "

                          f"{written} rows  {el:.0f}s elapsed  "

                          f"~{el/max(pct,1e-6)-el:.0f}s left")



            browser.close()

    finally:

        server.terminate()



    meta = {

        "runId": run_id, "frames": len(frames), "views": [v["name"] for v in views],

        "alphas": DEFAULT_ALPHAS, "unit": args.unit, "law": args.law,

        "calibrated": bool(amps), "expr": args.expr, "rows": written,

        "headed": args.headed, "seconds": round(time.time() - t0, 1),

        "note": "timings in rows are software-rendered and not quotable",

    }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nwrote {written} rows to {rows_path}")

    return 0





if __name__ == "__main__":

    raise SystemExit(main())

