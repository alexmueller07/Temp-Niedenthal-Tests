"""Export the data-driven parts of the algorithm from CFD into TypeScript.

Two things cross from the offline analysis into the runtime:

  1. The canonical face template -- the rigid-anchor configuration the live
     frame is fitted against. Built as the Procrustes mean of 823 faces, so
     "canonical units" are the average of a diverse corpus rather than one
     arbitrary person's proportions.

  2. The population smile axis -- the mean shape change from neutral to a
     genuine smile, measured on 153 people, with jaw-opening and lip-parting
     projected out. This replaces the hand-tuned displacement formula in the
     production morph, whose field peaks 1.28x harder on the cheek than on the
     lip corner it is supposed to be moving.

Emitted as a keyed map, not a bare array, so the two languages cannot silently
disagree about landmark ordering.

    .venv/Scripts/python.exe tools/export_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import facegeom as fg  # noqa: E402

HERE = Path(__file__).parent.parent
CACHE = HERE / "artifacts" / "cache" / "cfd_landmarks3d.npz"
OUT_TS = HERE / "src" / "algo" / "faceModel.gen.ts"

PROJECT = ("jaw_open", "lip_part")


def main() -> int:
    z = np.load(CACHE, allow_pickle=False)
    ids, exprs, pts = z["ids"], z["exprs"], z["pts"].astype(float)

    neutral = np.where(exprs == "N")[0]
    template = fg.scale_template_to_ipd(
        fg.build_template([pts[i][fg.RIGID_IDX, :2] for i in neutral]))
    frames = {i: fg.fit_frame(pts[i], template) for i in range(len(ids))}

    by_id: dict[str, dict[str, int]] = {}
    for i, (pid, e) in enumerate(zip(ids, exprs)):
        by_id.setdefault(str(pid), {})[str(e)] = i

    # --- population smile axis ---------------------------------------------
    # Each person's neutral->smile shape change, nuisance-projected and scaled
    # to unit norm, then averaged. Unit-normalising *before* averaging keeps a
    # few very expressive models from dominating the shape of the axis; their
    # amplitude is a separate quantity, handled at runtime.
    axes, amps = [], []
    for pid, m in by_id.items():
        if "N" not in m:
            continue
        for expr in ("HC", "HO"):
            if expr not in m:
                continue
            cn = frames[m["N"]].to_canonical(pts[m["N"]])[fg.DRIVEN]
            cs = frames[m[expr]].to_canonical(pts[m[expr]])[fg.DRIVEN]
            y = (cs - cn).reshape(-1)
            basis = fg.nuisance_basis(cn, fg.DRIVEN)
            y = fg.project_out(y, [basis[k] for k in PROJECT])
            a = float(np.linalg.norm(y))
            if a > 1e-6:
                axes.append(y / a)
                amps.append(a)

    axis = np.mean(axes, axis=0)
    axis /= np.linalg.norm(axis)
    amps = np.array(amps)

    # Rest shape: the mean canonical mouth over neutral faces. The runtime uses
    # it only as a fallback before the live person's own rest shape is known.
    rest = np.mean([frames[i].to_canonical(pts[i])[fg.DRIVEN] for i in neutral], axis=0)

    consistency = float(np.mean([float(a @ axis) for a in axes]))

    # Reliability of a single amplitude observation: correlate each person's
    # closed-mouth against their toothy smile amplitude.
    pair_a, pair_b = [], []
    for pid, m in by_id.items():
        if not {"N", "HC", "HO"} <= set(m):
            continue
        vals = []
        for e in ("HC", "HO"):
            cn = frames[m["N"]].to_canonical(pts[m["N"]])[fg.DRIVEN]
            cs = frames[m[e]].to_canonical(pts[m[e]])[fg.DRIVEN]
            y = (cs - cn).reshape(-1)
            basis = fg.nuisance_basis(cn, fg.DRIVEN)
            vals.append(float(np.linalg.norm(
                fg.project_out(y, [basis[k] for k in PROJECT]))))
        pair_a.append(vals[0])
        pair_b.append(vals[1])
    reliability = float(np.corrcoef(pair_a, pair_b)[0, 1])

    # Two physical scales, so the new implementation can be dialled to the same
    # intensity as the current one. Without this the A/B comparison would change
    # two things at once -- how equal the dose is, AND how big it is.
    ci = {idx: k for k, idx in enumerate(fg.DRIVEN)}
    corner_travel = []
    for pid, m in by_id.items():
        if "N" not in m:
            continue
        for e in ("HC", "HO"):
            if e not in m:
                continue
            cn = frames[m["N"]].to_canonical(pts[m["N"]])[fg.DRIVEN]
            cs = frames[m[e]].to_canonical(pts[m[e]])[fg.DRIVEN]
            y = (cs - cn).reshape(-1)
            basis = fg.nuisance_basis(cn, fg.DRIVEN)
            d = fg.project_out(y, [basis[k] for k in PROJECT]).reshape(-1, 2)
            corner_travel.append(float(
                (np.linalg.norm(d[ci[fg.LEFT_CORNER]])
                 + np.linalg.norm(d[ci[fg.RIGHT_CORNER]])) / 2))
    corner_full = float(np.median(corner_travel))

    cur = []
    for i in neutral:
        w_, h_ = int(z["dims"][i][0]), int(z["dims"][i][1])
        mm = fg.current_morph_delivery(pts[i], 1.9, w_, h_)
        ipd = fg.ipd_px(pts[i])
        if ipd > 1e-6:
            cur.append(mm.d_px / ipd)
    cur_19 = float(np.median(cur))
    alpha_scale = (cur_19 / corner_full) / 0.9

    def fmt(v: float) -> str:
        return f"{v:.6f}"

    tmpl_lines = [
        f"  [{int(idx)}, [{fmt(template[k, 0])}, {fmt(template[k, 1])}]],"
        for k, idx in enumerate(fg.RIGID_IDX)
    ]
    axis2 = axis.reshape(-1, 2)
    axis_lines = [
        f"  [{int(idx)}, [{fmt(axis2[k, 0])}, {fmt(axis2[k, 1])}]],"
        for k, idx in enumerate(fg.DRIVEN)
    ]
    rest_lines = [
        f"  [{int(idx)}, [{fmt(rest[k, 0])}, {fmt(rest[k, 1])}]],"
        for k, idx in enumerate(fg.DRIVEN)
    ]

    src = f'''// GENERATED by tools/export_model.py -- do not edit by hand.
//
// Fitted on the Chicago Face Database: {len(neutral)} neutral faces for the
// canonical template, {len(axes)} neutral->smile pairs from {len(amps)//2 or len(amps)}
// identities for the smile axis. No image data is reproduced here, only
// aggregate geometry.

/** Canonical rigid-anchor positions, in interpupillary units. Keyed by
 *  MediaPipe landmark index so ordering cannot drift between languages. */
export const CANONICAL_TEMPLATE: ReadonlyArray<readonly [number, readonly [number, number]]> = [
{chr(10).join(tmpl_lines)}
]

/** The mean shape change from neutral to smile, unit-norm over the whole driven
 *  set, with jaw-opening and lip-parting projected out. Multiply by a person's
 *  amplitude and by the commanded fraction to get a canonical displacement.
 *
 *  Mean cosine of an individual smile with this axis: {consistency:.3f} --
 *  i.e. real smiles point almost the same way in almost everyone, which is why
 *  the runtime learns each person's smile *size* and not its direction. */
export const SMILE_AXIS: ReadonlyArray<readonly [number, readonly [number, number]]> = [
{chr(10).join(axis_lines)}
]

/** Mean canonical mouth shape over the neutral corpus. Fallback rest shape,
 *  used only until the live person's own resting mouth has been observed. */
export const POPULATION_REST: ReadonlyArray<readonly [number, readonly [number, number]]> = [
{chr(10).join(rest_lines)}
]

/** Distribution of own-smile amplitude across the corpus, in interpupillary
 *  units of total shape change. The population median is the cold-start gain;
 *  the spread is the size of the problem this project exists to fix. */
export const AMPLITUDE_STATS = {{
  mean: {fmt(float(amps.mean()))},
  sd: {fmt(float(amps.std(ddof=1)))},
  p10: {fmt(float(np.percentile(amps, 10)))},
  p50: {fmt(float(np.percentile(amps, 50)))},
  p90: {fmt(float(np.percentile(amps, 90)))},
  n: {len(amps)},
}} as const

/** Mean cosine between an individual's smile axis and the population axis. */
export const AXIS_CONSISTENCY = {consistency:.4f}

/** Test-retest reliability of a single observed smile amplitude, from the
 *  correlation between each person's closed-mouth and toothy smile. It sets how
 *  fast a live estimate should overrule the population prior: the Bayes-optimal
 *  weight on N observations is N / (N + (1 - r) / r), so this constant is
 *  measured rather than tuned. Treat it as a lower bound -- CFD's two smiles are
 *  separate posed acts, so some of their disagreement is not measurement error. */
export const AMPLITUDE_RELIABILITY = {reliability:.4f}

/** Median mouth-corner travel at a full smile, interpupillary units. The
 *  denominator that turns a displacement into "a fraction of a whole smile". */
export const CORNER_TRAVEL_AT_FULL_SMILE = {fmt(corner_full)}

/** What the CURRENT production morph delivers at its strongest preset
 *  (alpha 1.9), median over the corpus, same units. */
export const CURRENT_TRAVEL_AT_ALPHA_1_9 = {fmt(cur_19)}

/** Smile-units per unit of alpha, chosen so alpha 1.9 lands at the same
 *  physical magnitude the current morph already delivers on a median face.
 *  Keeping intensity matched is what makes the A/B honest: the comparison is
 *  about how EQUAL the dose is, not how big. */
export const ALPHA_TO_SMILE_UNITS = {fmt(alpha_scale)}
'''

    OUT_TS.write_text(src, encoding="utf-8")
    print(f"wrote {OUT_TS}")
    print(f"  template anchors : {len(fg.RIGID_IDX)}")
    print(f"  driven landmarks : {len(fg.DRIVEN)}")
    print(f"  smile pairs      : {len(axes)}")
    print(f"  axis consistency : {consistency:.4f}")
    print(f"  amplitude reliability (1 obs) : {reliability:.4f}" f"  from {len(pair_a)} paired identities")
    print(f"  amplitude p10/p50/p90 : {np.percentile(amps,10):.3f} / "
          f"{np.percentile(amps,50):.3f} / {np.percentile(amps,90):.3f}")
    print(f"  corner travel at full smile (median) : {corner_full:.4f} IPD")
    print(f"  current morph at alpha 1.9  (median) : {cur_19:.4f} IPD"
          f"  = {cur_19/corner_full:.1%} of a full smile")
    print(f"  alpha -> smile units scale           : {alpha_scale:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
