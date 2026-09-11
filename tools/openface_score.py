"""Score the rendered frames with OpenFace — an instrument that shares nothing
with the one steering the morph.

Everything measured inside the page goes through MediaPipe, which is also what
the normalized implementation uses to decide where to move things. That makes
in-page numbers useful for tuning and unsuitable for deciding which
implementation wins. OpenFace 2.2 uses a different face model (CE-CLM) and a
separately trained action-unit regressor, and AU12 — lip corner puller — is the
action a smile is made of.

**Use FaceLandmarkImg, not FeatureExtraction.** Not cosmetic, and it produced a
wrong answer before the control caught it. FeatureExtraction treats a directory
as a *video of one person* and applies OpenFace's dynamic, person-normalized
action-unit models; over a folder of sixty different faces it normalizes each
against the mixture, and real smiles came out scoring *lower* than the neutral
photographs of the same people. FaceLandmarkImg scores each image independently
with the static models and writes one csv per image, so the mapping back to
filenames is by name rather than by position.

Two things to stay honest about even when it is set up correctly:

  It is an instrument, not ground truth. Automated action-unit intensity
  correlates with facial EMG but is less sensitive and less accurate than it,
  and this lab's PI does not accept off-the-shelf expression models as ground
  truth for perceived expression. The only ground truth for "did this read as a
  bigger smile" is human ratings.

  Each morphed render is differenced against a sham render of the very same
  frame, so the instrument's per-identity bias cancels to first order. What it
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
CORPUS = Path(r"C:\lab-corpus")
RENDERS = CORPUS / "renders"
OPENFACE = CORPUS / "tools" / "OpenFace"
CONTROL = CORPUS / "of_control"

NAME_RE = re.compile(r"^(?P<id>[A-Z]{2}-\d+)_(?P<impl>[a-z]+)"
                     r"_(?P<view>[^_]+)_(?P<cond>sham|a[\d.]+)$")


def find_exe() -> Path:
    hits = list(OPENFACE.rglob("FaceLandmarkImg.exe"))
    if not hits:
        raise SystemExit(
            "FaceLandmarkImg.exe not found under " + str(OPENFACE) + ". "
            "Download OpenFace_2.2.0_win_x64.zip from the OpenFace releases "
            "page, unzip it there, and fetch the CEN patch experts with its "
            "download_models script.")
    return hits[0]


def run_openface(img_dir: Path, out_dir: Path) -> None:
    exe = find_exe()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(img_dir.glob("*.png")))
    print("running " + exe.name + " over " + str(n) + " images...")
    res = subprocess.run([str(exe), "-fdir", str(img_dir),
                          "-out_dir", str(out_dir), "-aus"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        raise SystemExit("OpenFace failed with exit " + str(res.returncode))


def read_aus(out_dir: Path) -> dict[str, dict[str, float]]:
    """{image stem: {AU name: intensity}}, keyed by filename, never by order."""
    vals: dict[str, dict[str, float]] = {}
    for csv_path in sorted(out_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        clean = {k.strip(): (v.strip() if isinstance(v, str) else v)
                 for k, v in rows[0].items() if k}
        if clean.get("success", "1") != "1":
            continue
        aus = {k: float(v) for k, v in clean.items()
               if k.startswith("AU") and k.endswith("_r")}
        if aus:
            vals[csv_path.stem] = aus
    return vals


def stats(v: list[float]) -> dict:
    v = [x for x in v if x == x and math.isfinite(x)]
    if len(v) < 2:
        return {"n": len(v)}
    m = st.fmean(v)
    sd = st.stdev(v)
    return {"n": len(v), "mean": m, "sd": sd,
            "cv": sd / m if m else float("nan"),
            "median": st.median(v),
            "positive": sum(1 for x in v if x > 0)}


def score_control(au: str) -> dict:
    """Does this instrument register a REAL smile on these same faces?

    Without this the main result is uninterpretable: "no detected change" could
    mean the morph is subtle or that the measurement is broken. It also catches
    the dynamic-model mistake immediately, because that makes real smiles score
    negative.
    """
    if not CONTROL.exists() or not list(CONTROL.glob("*.png")):
        print("  no control set at " + str(CONTROL) + "; skipping")
        return {}
    out = CONTROL / "openface_img"
    if not any(out.glob("*.csv")):
        run_openface(CONTROL, out)

    rows: dict[tuple[str, str], float] = {}
    for stem, v in read_aus(out).items():
        parts = stem.split("_")
        if len(parts) >= 4:
            rows[(parts[0], parts[-1])] = v.get(au, float("nan"))
    ids = sorted({k[0] for k in rows})
    d = [rows[(i, "a9.9")] - rows[(i, "sham")]
         for i in ids if (i, "a9.9") in rows and (i, "sham") in rows]
    s = stats(d)
    if s.get("n", 0) < 2:
        return {}
    print("  CONTROL, real neutral to real closed-mouth smile, same faces:")
    print("    n=" + str(s["n"])
          + "  mean d" + au + " " + format(s["mean"], "+.3f")
          + "  median " + format(s["median"], "+.3f")
          + "  positive in " + str(s["positive"]) + "/" + str(s["n"]))
    if s["mean"] <= 0:
        raise SystemExit(
            "the control is not positive, so the instrument is misconfigured "
            "(most likely dynamic AU models over a mixed directory). Nothing "
            "below would mean anything.")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="render run id under C:/lab-corpus/renders")
    ap.add_argument("--alpha", default="a1.9")
    ap.add_argument("--au", default="AU12_r",
                    help="AU12 is the lip corner puller, the smile action")
    ap.add_argument("--control", action="store_true", help="control only")
    args = ap.parse_args()

    out: dict = {}
    print("=" * 72)
    print(args.au + " measured by OpenFace 2.2, static models, one image at a time")
    print("=" * 72)
    out["control"] = score_control(args.au)
    if args.control or not args.run:
        return 0

    img_dir = RENDERS / args.run
    if not img_dir.exists():
        raise SystemExit("no renders at " + str(img_dir) + "; run with --export-pngs")
    of_dir = img_dir / "openface"
    if not any(of_dir.glob("*.csv")):
        run_openface(img_dir, of_dir)
    aus = read_aus(of_dir)
    print("\n  " + str(len(aus)) + " scored of "
          + str(len(list(img_dir.glob("*.png")))) + " renders")

    by_key: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for stem, vals in aus.items():
        m = NAME_RE.match(stem)
        if m:
            by_key[(m["id"], m["impl"], m["view"])][m["cond"]] = \
                vals.get(args.au, float("nan"))

    amps = json.loads((HERE / "artifacts" / "amplitudes.json").read_text(encoding="utf-8"))

    print("\n  THE MORPH at " + args.alpha
          + ", against the sham render of the same frame:")
    for impl in ("current", "normalized"):
        deltas, selfref = [], []
        for (fid, im, _v), conds in by_key.items():
            if im != impl or "sham" not in conds or args.alpha not in conds:
                continue
            d = conds[args.alpha] - conds["sham"]
            deltas.append(d)
            if fid in amps and amps[fid] > 0:
                selfref.append(d / amps[fid])
        sd, ss = stats(deltas), stats(selfref)
        print("\n    [" + impl + "]")
        if sd.get("n", 0) >= 2:
            print("      delta " + args.au
                  + "  n=" + str(sd["n"])
                  + "  mean " + format(sd["mean"], "+.4f")
                  + "  median " + format(sd["median"], "+.4f")
                  + "  sd " + format(sd["sd"], ".4f")
                  + "  positive in " + str(sd["positive"]) + "/" + str(sd["n"]))
        if ss.get("n", 0) >= 2:
            print("      per own smile size  n=" + str(ss["n"])
                  + "  mean " + format(ss["mean"], "+.4f")
                  + "  CV " + format(ss["cv"], ".3f"))
        out[impl] = {"delta": sd, "self_referenced": ss}

    ctrl = out.get("control", {}).get("mean")
    if ctrl:
        print()
        for impl in ("current", "normalized"):
            m = out.get(impl, {}).get("delta", {}).get("mean")
            if m is not None:
                print("  " + impl + ": " + format(m / ctrl, ".1%")
                      + " of the change a real smile of this size produces")

    dest = HERE / "artifacts" / ("openface_" + args.run + ".json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote " + str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
