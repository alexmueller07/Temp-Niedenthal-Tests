"""Figures for the write-up.

Deliberately contains no faces — only aggregate geometry — so these can live in
the repo and go into a slide without worrying about the Chicago Face Database
licence.

    .venv/Scripts/python.exe tools/figures.py --e1 <runId> [--e2 <runId>]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent.parent
RUNS = HERE / "artifacts" / "runs"
FIGS = HERE / "artifacts" / "figures"

INK = "#1d2330"
DIM = "#7a869d"
CUR = "#e08a2e"      # current implementation
NEW = "#2e9e70"      # normalized
GRID = "#e4e8ef"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 9, "axes.edgecolor": DIM, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": DIM, "ytick.color": DIM,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def load(run_id: str) -> list[dict]:
    p = RUNS / run_id / "rows.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


def cv(v: list[float]) -> float:
    v = [x for x in v if x == x and math.isfinite(x)]
    return st.stdev(v) / st.fmean(v) if len(v) > 1 and st.fmean(v) else float("nan")


def fig_amplitude_distribution(amps: dict[str, float], out: Path) -> None:
    """What the 'strong' preset actually delivers, person by person."""

    src = (HERE / "src" / "algo" / "faceModel.gen.ts").read_text(encoding="utf-8")
    full = float(src.split("CORNER_TRAVEL_AT_FULL_SMILE = ")[1].split("\n")[0])
    cur = float(src.split("CURRENT_TRAVEL_AT_ALPHA_1_9 = ")[1].split("\n")[0])

    a = sorted(amps.values())
    # Each person's own corner travel at a full smile scales with their overall
    # amplitude; the median person's is `full`.
    med = st.median(a)
    own_travel = [x / med * full for x in a]
    share = [cur / t for t in own_travel]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))

    ax1.hist(a, bins=28, color=NEW, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax1.axvline(med, color=INK, lw=1.2, ls="--")
    ax1.set_xlabel("own smile amplitude (interpupillary units)")
    ax1.set_ylabel("people")
    ax1.set_title(f"How big people's own smiles are\n"
                  f"n={len(a)}, P90/P10 = "
                  f"{a[int(.9*(len(a)-1))]/a[int(.1*(len(a)-1))]:.1f}x",
                  loc="left", fontsize=9.5)

    ax2.hist([min(s, 1.6) for s in share], bins=28, color=CUR, alpha=0.8,
             edgecolor="white", linewidth=0.5)
    ax2.axvline(1.0, color="#c0392b", lw=1.2)
    ax2.set_xlabel("share of that person's own full smile")
    ax2.set_ylabel("people")
    ax2.set_title("What the current 'strong' preset adds\n"
                  "red line = as much as their whole smile",
                  loc="left", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_crossface(rows: list[dict], amps: dict[str, float], alpha: float,
                  out: Path) -> None:
    """Delivered dose per face, in self-referenced units."""
    series = {}
    for impl in ("current", "normalized"):
        vals = [(r["id"], r["measuredTravel"] / amps[r["id"]])
                for r in rows
                if r["impl"] == impl and r["alpha"] == alpha
                and r["view"] == "base" and r["d"]["faceFound"]
                and r["id"] in amps and amps[r["id"]] > 0]
        vals.sort(key=lambda t: t[1])
        series[impl] = [v for _, v in vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                   gridspec_kw={"width_ratios": [2, 1]})

    for impl, color in (("current", CUR), ("normalized", NEW)):
        v = series[impl]
        ax1.plot(range(len(v)), v, "o", ms=3.2, color=color, alpha=0.8,
                 label=f"{impl}  (CV {cv(v):.3f})")
        ax1.axhline(st.fmean(v), color=color, lw=1, ls="--", alpha=0.6)
    ax1.set_xlabel("faces, sorted within each implementation")
    ax1.set_ylabel("delivered dose / own smile size")
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax1.set_title("Same command, every face: how much of their own\n"
                  "smile actually arrived", loc="left", fontsize=9.5)

    labels = ["current", "normalized"]
    cvs = [cv(series["current"]), cv(series["normalized"])]
    ax2.bar(labels, cvs, color=[CUR, NEW], width=0.55)
    for i, c in enumerate(cvs):
        ax2.text(i, c + 0.006, f"{c:.3f}", ha="center", fontsize=9, color=INK)
    ax2.set_ylabel("dispersion across faces (CV)")
    ax2.set_ylim(0, max(cvs) * 1.35)
    red = 1 - cvs[1] / cvs[0] if cvs[0] else float("nan")
    ax2.set_title(f"{red:.0%} less spread", loc="left", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_views(rows: list[dict], alpha: float, out: Path) -> None:
    """Dose by camera view — the same face, sitting differently."""
    views = sorted({r["view"] for r in rows})
    if len(views) < 2:
        return
    order = [v for v in ["base", "near", "far", "low-in-frame", "roll-15",
                         "roll+15", "roll+25", "pitch-15", "pitch+15", "yaw+20"]
             if v in views] + [v for v in views if v not in {
                 "base", "near", "far", "low-in-frame", "roll-15", "roll+15",
                 "roll+25", "pitch-15", "pitch+15", "yaw+20"}]

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    x = range(len(order))
    for impl, color in (("current", CUR), ("normalized", NEW)):
        means, errs = [], []
        for v in order:
            vals = [r["measuredTravel"] for r in rows
                    if r["view"] == v and r["impl"] == impl
                    and r["alpha"] == alpha and r["d"]["faceFound"]
                    and r["measuredTravel"] == r["measuredTravel"]]
            means.append(st.fmean(vals) if vals else float("nan"))
            errs.append(st.stdev(vals) / math.sqrt(len(vals))
                        if len(vals) > 1 else 0)
        base = means[0] if means and means[0] == means[0] else 1
        ax.errorbar(x, [m / base for m in means], yerr=[e / base for e in errs],
                    fmt="o-", ms=4, lw=1.4, capsize=3, color=color, label=impl)

    ax.axhline(1.0, color=DIM, lw=1, ls="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("delivered dose, relative to frontal")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Same face, different camera: the dose should not move",
                 loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_calibration(out: Path) -> None:
    """How many observed smiles calibration needs, from the measured reliability."""
    s0 = json.loads((HERE / "artifacts" / "s0_amplitude.json").read_text(encoding="utf-8"))
    c = s0["C_calibration"]
    resid = c["residual_cv"]
    ns = [k for k in resid if k != "none"]
    xs = [int(k.split("=")[1]) for k in ns]
    ys = [resid[k] for k in ns]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.axhline(resid["none"], color=CUR, lw=1.5, ls="--",
               label=f"no calibration ({resid['none']:.3f})")
    ax.plot(xs, ys, "o-", color=NEW, ms=5, lw=1.6, label="calibrated")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("smiles observed before the manipulation starts")
    ax.set_ylabel("residual spread in delivered dose (CV)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title(f"One smile is barely better than none\n"
                 f"single-observation reliability r = {c['reliability']:.2f}",
                 loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", help="cross-face run id (calibrated)")
    ap.add_argument("--e2", help="camera-ablation run id")
    ap.add_argument("--alpha", type=float, default=1.9)
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    amps = json.loads((HERE / "artifacts" / "amplitudes.json").read_text(encoding="utf-8"))

    fig_amplitude_distribution(amps, FIGS / "1-amplitude-spread.png")
    print("wrote 1-amplitude-spread.png")

    fig_calibration(FIGS / "4-calibration-need.png")
    print("wrote 4-calibration-need.png")

    if args.e1:
        fig_crossface(load(args.e1), amps, args.alpha, FIGS / "2-crossface.png")
        print("wrote 2-crossface.png")
    if args.e2:
        fig_views(load(args.e2), args.alpha, FIGS / "3-camera-views.png")
        print("wrote 3-camera-views.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
