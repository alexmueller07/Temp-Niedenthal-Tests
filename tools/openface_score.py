"""Score the rendered frames with OpenFace — an instrument that shares nothing
with the one steering the morph.

Everything measured inside the page goes through MediaPipe, which is also what
the normalized implementation uses to decide where to move things. That makes
in-page numbers useful for tuning and unsuitable for deciding which
implementation wins. OpenFace 2.2 uses a different face model (CE-CLM) and a
separately trained action-unit regressor, and AU12 — lip corner puller — is the
action a smile is made of.

Two things to keep honest about it:

  It is an instrument, not ground truth. Automated AU intensity correlates with
  facial EMG but is less sensitive and less accurate than it, and this lab's PI
  does not accept off-the-shelf expression models as ground truth for perceived
  expression. The only ground truth for "did this read as a bigger smile" is
  human ratings.

  Because each morphed render is differenced against a sham render of the very
  same frame, the instrument's per-identity bias cancels to first order. What it
  does not cancel is per-identity differences in the instrument's *slope*, or
  its known degradation on darker skin tones, beards and non-frontal faces.

    .venv/Scripts/python.exe tools/openface_score.py --run <runId>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.parent
RENDERS = Path(r"C:\lab-corpus") / "renders"
OPENFACE = Path(r"C:\lab-corpus") / "tools" / "OpenFace"

NAME_RE = re.compile(r"^(?P<id>[A-Z]{2}-\d+)_(?P<impl>current|normalized)"
                     r"_(?P<view>[^_]+)_(?P<cond>sham|a[\d.]+)$")


def find_exe() -> Path:
    hits = list(OPENFACE.rglob("FeatureExtraction.exe"))
    if not hits:
        raise SystemExit(
            f"FeatureExtraction.exe not found under {OPENFACE}.\n"
            "Download OpenFace_2.2.0_win_x64.zip from the OpenFace releases page "
            "and unzip it there.")
    return hits[0]


def run_openface(img_dir: Path, out_dir: Path) -> Path:
    exe = find_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), "-fdir", str(img_dir), "-out_dir", str(out_dir),
           "-aus", "-q"]
    print(f"running {exe.name} over {len(list(img_dir.glob('*.png')))} images…")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        raise SystemExit(f"OpenFace failed (exit {res.returncode})")
    return out_dir


def read_aus(out_dir: Path, img_dir: Path) -> dict[str, dict[str, float]]:
    """{image stem: {AU name: intensity}}.

    With -fdir, OpenFace writes ONE csv for the whole directory and identifies
    rows by frame number rather than by filename — so the mapping back to images
    is positional, over the directory listing in sorted order. Worth stating
    because a silent off-by-one here would swap conditions and invert every
    conclusion; the sanity check below catches that.
    """
    images = sorted(img_dir.glob("*.png"))
    vals: dict[str, dict[str, float]] = {}
    for csv_path in out_dir.glob("*.csv"):
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != len(images):
            raise SystemExit(
                f"{csv_path.name} has {len(rows)} rows for {len(images)} images; "
                "the positional mapping would be wrong")
        for img, row in zip(images, rows):
            clean = {k.strip(): (v.strip() if isinstance(v, str) else v)
                     for k, v in row.items() if k}
            if clean.get("success", "1") != "1":
                continue
            aus = {k: float(v) for k, v in clean.items()
                   if k.startswith("AU") and k.endswith("_r")}
            if aus:
                vals[img.stem] = aus
    return vals


def stats(v: list[float]) -> dict:
    v = [x for x in v if x == x and math.isfinite(x)]
    if len(v) < 2:
        return {"n": len(v)}
    m = st.fmean(v)
    sd = st.stdev(v)
    s = sorted(v)
    p10 = s[max(0, int(0.10 * (len(s) - 1)))]
    p90 = s[min(len(s) - 1, int(0.90 * (len(s) - 1)))]
    return {"n": len(v), "mean": m, "sd": sd,
            "cv": sd / m if m else float("nan"),
            "p10": p10, "p90": p90,
            "ratio": p90 / p10 if p10 > 0 else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--alpha", default="a1.9")
    ap.add_argument("--au", default="AU12_r",
                    help="AU12 is the lip corner puller -- the smile action")
    args = ap.parse_args()

    img_dir = RENDERS / args.run
    if not img_dir.exists():
        raise SystemExit(f"no renders at {img_dir}; run with --export-pngs")

    out_dir = img_dir / "openface"
    if not any(out_dir.glob("*.csv")):
        run_openface(img_dir, out_dir)
    aus = read_aus(out_dir, img_dir)
    print(f"OpenFace produced {len(aus)} successful detections "
          f"of {len(list(img_dir.glob('*.png')))} images")

    # Sanity check on the positional mapping: a morphed smile render must score
    # higher on AU12 than the sham render of the same frame, on average. If this
    # comes out negative the frame-to-file mapping is off and nothing below means
    # anything.
    ups = [aus[k].get(args.au, 0) for k in aus if not k.endswith("_sham")]
    shams = [aus[k].get(args.au, 0) for k in aus if k.endswith("_sham")]
    if ups and shams and st.fmean(ups) <= st.fmean(shams):
        raise SystemExit(
            f"sanity check failed: mean {args.au} on morphed renders "
            f"({st.fmean(ups):.3f}) is not above sham ({st.fmean(shams):.3f}). "
            "The frame-to-file mapping is probably wrong.")
    if not aus:
        raise SystemExit("no successful detections")

    # stem -> (id, impl, view, condition)
    by_key: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for stem, vals in aus.items():
        m = NAME_RE.match(stem)
        if not m:
            continue
        key = (m["id"], m["impl"], m["view"])
        by_key[key][m["cond"]] = vals.get(args.au, float("nan"))

    amps = json.loads((HERE / "artifacts" / "amplitudes.json").read_text(encoding="utf-8"))

    print(f"\n{'='*72}\n{args.au} change vs the sham render of the same frame\n{'='*72}")
    out: dict = {}
    for impl in ("current", "normalized"):
        deltas, selfref = [], []
        for (fid, im, _view), conds in by_key.items():
            if im != impl or "sham" not in conds or args.alpha not in conds:
                continue
            d = conds[args.alpha] - conds["sham"]
            deltas.append(d)
            if fid in amps and amps[fid] > 0:
                selfref.append(d / amps[fid])
        sd, ss = stats(deltas), stats(selfref)
        print(f"\n  [{impl}]")
        if sd.get("n", 0) >= 2:
            print(f"    delta {args.au:<8} n={sd['n']:<4} mean={sd['mean']:+.4f}  "
                  f"CV={sd['cv']:.3f}  P90/P10={sd['ratio']:.2f}")
        if ss.get("n", 0) >= 2:
            print(f"    per own smile size     n={ss['n']:<4} mean={ss['mean']:+.4f}  "
                  f"CV={ss['cv']:.3f}  P90/P10={ss['ratio']:.2f}")
        out[impl] = {"delta": sd, "self_referenced": ss}

    print("\n  Lower CV = the manipulation lands more equally across faces.")
    print("  This is an independent instrument, but still an instrument:\n"
          "  the claim it supports is about geometry arriving consistently,\n"
          "  not about what a person would perceive.")

    dest = HERE / "artifacts" / f"openface_{args.run}.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
