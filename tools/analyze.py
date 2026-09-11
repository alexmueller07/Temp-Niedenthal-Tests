"""Turn a batch run into the numbers that go in the report.

Primary outcome is the dispersion of the *delivered* manipulation — what the
warp actually did to the pixels, re-measured by a detector that is not the one
steering it — expressed two ways:

  geometric        how far the mouth corners moved, in interpupillary units.
                   Equal here means everyone's face was deformed by the same
                   physical amount.
  self-referenced  that displacement as a share of the person's own full smile.
                   Equal here means everyone received the same manipulation
                   relative to their own expressive range.

Those are different constructs and they cannot both be equalized at once; the
whole point of reporting both is that the lab has to choose.

    .venv/Scripts/python.exe tools/analyze.py --run <runId> [--run <runId> ...]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent.parent
RUNS = HERE / "artifacts" / "runs"


def load(run_id: str) -> tuple[list[dict], dict]:
    d = RUNS / run_id
    rows = [json.loads(x) for x in (d / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x]
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
    return rows, meta


def stats(vals: list[float]) -> dict:
    v = sorted(x for x in vals if x == x and math.isfinite(x))
    if len(v) < 2:
        return {"n": len(v)}
    n = len(v)
    mean = sum(v) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1))

    def pct(p: float) -> float:
        k = (n - 1) * p
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        return v[lo] if lo == hi else v[lo] + (k - lo) * (v[hi] - v[lo])

    p10, p90 = pct(0.10), pct(0.90)
    return {
        "n": n, "mean": mean, "sd": sd,
        "cv": sd / mean if mean else float("nan"),
        "p10": p10, "p50": pct(0.5), "p90": p90,
        "ratio": p90 / p10 if p10 > 0 else float("nan"),
        "min": v[0], "max": v[-1],
    }


def line(label: str, s: dict) -> str:
    if s.get("n", 0) < 2:
        return f"  {label:<34} (n={s.get('n', 0)})"
    return (f"  {label:<34} n={s['n']:<4} mean={s['mean']:.4f}  "
            f"CV={s['cv']:.3f}  P90/P10={s['ratio']:.2f}  "
            f"[{s['min']:.4f}, {s['max']:.4f}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--alpha", type=float, default=1.9)
    args = ap.parse_args()

    amps = json.loads((HERE / "artifacts" / "amplitudes.json").read_text(encoding="utf-8"))
    out: dict = {}

    for run_id in args.run:
        rows, meta = load(run_id)
        rows = [r for r in rows if r["d"]["faceFound"]]
        print(f"\n{'='*78}\n{run_id}\n{'='*78}")
        if meta:
            print(f"  {meta.get('frames')} faces, views {meta.get('views')}, "
                  f"unit={meta.get('unit')} law={meta.get('law')} "
                  f"calibrated={meta.get('calibrated')}")
        res: dict = {"meta": meta}

        # ---------------- cross-face dispersion, base view ----------------
        base = [r for r in rows if r["view"] == "base" and r["alpha"] == args.alpha]
        if base:
            print(f"\n--- cross-face dispersion at alpha {args.alpha} "
                  f"(base view) ---")
            for impl in ("current", "normalized"):
                sub = [r for r in base if r["impl"] == impl]
                if not sub:
                    continue
                geo = [r["measuredTravel"] for r in sub]
                selfref = [r["measuredTravel"] / amps[r["id"]]
                           for r in sub if r["id"] in amps and amps[r["id"]] > 0]
                blend = [r["d"]["blendSmile"] for r in sub]
                sg, ss, sb = stats(geo), stats(selfref), stats(blend)
                print(f"\n  [{impl}]")
                print(line("delivered travel / IPD", sg))
                print(line("delivered / own smile size", ss))
                print(line("blendshape smile change", sb))
                res.setdefault("crossface", {})[impl] = {
                    "geometric": sg, "self_referenced": ss, "blendshape": sb,
                }

        # ---------------- within-face pose invariance ---------------------
        views = sorted({r["view"] for r in rows})
        if len(views) > 1:
            print(f"\n--- within-face dispersion across {len(views)} camera views "
                  f"at alpha {args.alpha} ---")
            for impl in ("current", "normalized"):
                per_face: dict[str, list[float]] = defaultdict(list)
                asym: dict[str, list[float]] = defaultdict(list)
                for r in rows:
                    if r["impl"] != impl or r["alpha"] != args.alpha:
                        continue
                    per_face[r["id"]].append(r["measuredTravel"])
                    asym[r["id"]].append(r["d"]["cornerAsymmetry"])
                cvs, arange = [], []
                for fid, vals in per_face.items():
                    v = [x for x in vals if x == x]
                    if len(v) >= 3 and sum(v) / len(v) > 1e-6:
                        m = sum(v) / len(v)
                        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
                        cvs.append(sd / m)
                    a = [x for x in asym[fid] if x == x]
                    if len(a) >= 3:
                        arange.append(max(a) - min(a))
                sc, sa = stats(cvs), stats(arange)
                print(f"\n  [{impl}]")
                print(line("within-face CV of delivered dose", sc))
                print(line("induced L/R asymmetry, range", sa))
                res.setdefault("pose", {})[impl] = {
                    "within_face_cv": sc, "asymmetry_range": sa}

            # Per-view means show WHERE it breaks, which a single CV hides.
            print("\n  delivered dose by view (mean travel / IPD):")
            print(f"    {'view':<16}{'current':>12}{'normalized':>13}")
            for v in views:
                cur = stats([r["measuredTravel"] for r in rows
                             if r["view"] == v and r["impl"] == "current"
                             and r["alpha"] == args.alpha])
                nor = stats([r["measuredTravel"] for r in rows
                             if r["view"] == v and r["impl"] == "normalized"
                             and r["alpha"] == args.alpha])
                print(f"    {v:<16}{cur.get('mean', float('nan')):>12.4f}"
                      f"{nor.get('mean', float('nan')):>13.4f}")
                res.setdefault("by_view", {})[v] = {
                    "current": cur.get("mean"), "normalized": nor.get("mean")}

        # ---------------- warp quality -----------------------------------
        q = [r for r in rows if r["impl"] == "normalized" and r.get("maxExpansion")]
        if q:
            print("\n--- warp quality (normalized implementation) ---")
            print(line("worst local stretch", stats([r["maxExpansion"] for r in q])))
            print(line("worst local compression", stats([r["minExpansion"] for r in q])))
            folded = sum(1 for r in q if r.get("folded"))
            print(f"  {'frames with a folded cell':<34} {folded}/{len(q)}")
            res["quality"] = {
                "max_expansion": stats([r["maxExpansion"] for r in q]),
                "min_expansion": stats([r["minExpansion"] for r in q]),
                "folded": folded, "n": len(q),
            }

        out[run_id] = res

    dest = HERE / "artifacts" / "analysis.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
