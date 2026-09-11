"""Stage 0b: the amplitude question, which Q1-Q6 showed is the real problem.

Q1 found the current morph already delivers a near-constant displacement in its
own unit (CV 0.4%), varying only 7% across faces once put in interpupillary
units -- and all of that 7% is explained by anatomical mouth width. So the warp
is not what makes the manipulation unequal.

What does is that people's own smiles differ by roughly 2x, so the same added
displacement is a very different share of each person's expressive range. Three
questions follow, and they decide the design:

  A  Is own-smile amplitude a stable property of the person, or just how
     enthusiastically they happened to pose? If it is noise, nothing can
     normalize it and the honest answer to Randy is "you cannot".
  B  Can it be predicted from anatomy and appearance? If yes, a static gain
     model needs no calibration and the participant never does anything.
  C  If not, how many observed smiles does calibration actually need? That sets
     whether a silent waiting room is enough or the session needs a moment that
     reliably elicits smiling.

Method note. Two measurements per person -- a closed-mouth smile and a toothy
one -- are treated as parallel measures of one latent trait, so classical test
theory applies: the correlation between them IS the reliability of a single
measurement, and everything in C follows from it. That framing matters, because
comparing the two measurements directly (a ratio of two noisy numbers) inflates
dispersion rather than revealing it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import facegeom as fg  # noqa: E402
import norming  # noqa: E402

HERE = Path(__file__).parent.parent
CACHE = HERE / "artifacts" / "cache" / "cfd_landmarks3d.npz"
OUT = HERE / "artifacts" / "s0_amplitude.json"

PROJECT = ("jaw_open", "lip_part")


def ridge_loo_r2(x: np.ndarray, y: np.ndarray, lam: float) -> float:
    xs = (x - x.mean(0)) / (x.std(0) + 1e-12)
    xs = np.column_stack([xs, np.ones(len(xs))])
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        a = xs[m].T @ xs[m] + lam * np.eye(xs.shape[1])
        preds[i] = xs[i] @ np.linalg.solve(a, xs[m].T @ y[m])
    return 1.0 - float(((y - preds) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def cvs(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / x.mean())


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

    def shape_delta(i_n: int, i_e: int) -> tuple[np.ndarray, np.ndarray]:
        cn = frames[i_n].to_canonical(pts[i_n])[fg.DRIVEN]
        ce = frames[i_e].to_canonical(pts[i_e])[fg.DRIVEN]
        y = (ce - cn).reshape(-1)
        basis = fg.nuisance_basis(cn, fg.DRIVEN)
        return fg.project_out(y, [basis[k] for k in PROJECT]), cn

    # --- the population smile axis, needed to define a projected amplitude ---
    axes = []
    for pid, m in by_id.items():
        if "N" not in m:
            continue
        for e in ("HC", "HO"):
            if e in m:
                y, _ = shape_delta(m["N"], m[e])
                n = np.linalg.norm(y)
                if n > 1e-6:
                    axes.append(y / n)
    pop_axis = np.mean(axes, axis=0)
    pop_axis /= np.linalg.norm(pop_axis)

    ci = {idx: k for k, idx in enumerate(fg.DRIVEN)}
    lc_k, rc_k = ci[fg.LEFT_CORNER], ci[fg.RIGHT_CORNER]

    def measures(i_n: int, i_e: int) -> dict[str, float]:
        """Three candidate amplitude estimators, to be chosen by reliability."""
        y, _ = shape_delta(i_n, i_e)
        d = y.reshape(-1, 2)
        return {
            "whole_shape_norm": float(np.linalg.norm(y)),
            "projection_on_axis": float(y @ pop_axis),
            "corner_travel": float((np.linalg.norm(d[lc_k]) + np.linalg.norm(d[rc_k])) / 2),
        }

    results: dict = {}
    print(f"\n{'='*78}\nSTAGE 0b -- the amplitude question\n{'='*78}")

    # ---------------------------------------------------------------- A ----
    print(f"\n{'-'*78}\nA  Is own-smile amplitude a stable property of the person?\n{'-'*78}")
    pid_ab, rows_hc, rows_ho, rows_ang, rows_fear = [], [], [], [], []
    for pid, m in by_id.items():
        if not {"N", "HC", "HO"} <= set(m):
            continue
        pid_ab.append(pid)
        rows_hc.append(measures(m["N"], m["HC"]))
        rows_ho.append(measures(m["N"], m["HO"]))
        rows_ang.append(measures(m["N"], m["A"]) if "A" in m else None)
        rows_fear.append(measures(m["N"], m["F"]) if "F" in m else None)

    print(f"  n = {len(pid_ab)} identities with neutral + closed + toothy smile\n")
    print(f"  {'amplitude estimator':<24} {'CV(closed)':>11} {'CV(toothy)':>11}"
          f" {'r(closed,toothy)':>18} {'r(smile,angry)':>16}")
    rel: dict[str, dict[str, float]] = {}
    for key in ("whole_shape_norm", "projection_on_axis", "corner_travel"):
        a = np.array([r[key] for r in rows_hc])
        b = np.array([r[key] for r in rows_ho])
        ok = np.array([r is not None for r in rows_ang])
        g = np.array([r[key] if r else np.nan for r in rows_ang])
        r_ab = float(np.corrcoef(a, b)[0, 1])
        r_ag = float(np.corrcoef(a[ok], g[ok])[0, 1])
        rel[key] = {"cv_hc": cvs(a), "cv_ho": cvs(b), "r_hc_ho": r_ab,
                    "r_hc_angry": r_ag,
                    "spearman_brown_2": 2 * r_ab / (1 + r_ab)}
        print(f"  {key:<24} {cvs(a):>11.3f} {cvs(b):>11.3f} {r_ab:>+18.3f} {r_ag:>+16.3f}")

    chosen = max(rel, key=lambda k: rel[k]["r_hc_ho"])
    r1 = rel[chosen]["r_hc_ho"]
    print(f"\n  chosen estimator: {chosen}  (highest test-retest reliability)")
    print(f"  single-measurement reliability r = {r1:.3f};"
          f" two-measure average = {rel[chosen]['spearman_brown_2']:.3f}")
    print("\n  Reading: r(closed, toothy) high and r(smile, angry) ~ 0 means smile\n"
          "  amplitude is a smile-specific property of the person, not a general\n"
          "  tendency to pose hard -- so it is a real thing to calibrate to.")
    results["A_trait"] = {"n": len(pid_ab), "estimators": rel, "chosen": chosen}

    amp_hc = np.array([r[chosen] for r in rows_hc])
    amp_ho = np.array([r[chosen] for r in rows_ho])
    cv_obs = float(np.mean([cvs(amp_hc), cvs(amp_ho)]))

    # ---------------------------------------------------------------- B ----
    print(f"\n{'-'*78}\nB  Can anatomy or appearance predict how big someone's smile is?\n{'-'*78}")
    norm = norming.load()
    target = (amp_hc + amp_ho) / 2.0   # the more reliable two-measure average

    geo = np.array([[
        float(np.linalg.norm(frames[by_id[p]['N']].to_canonical(pts[by_id[p]['N']])[fg.RIGHT_CORNER]
                             - frames[by_id[p]['N']].to_canonical(pts[by_id[p]['N']])[fg.LEFT_CORNER])),
        float(frames[by_id[p]['N']].to_canonical(pts[by_id[p]['N']])[fg.OUTER_LIP][:, 1].ptp()),
        float(np.median(frames[by_id[p]['N']].to_canonical(pts[by_id[p]['N']])[fg.OUTER_LIP][:, 1])
              - (frames[by_id[p]['N']].to_canonical(pts[by_id[p]['N']])[fg.LEFT_CORNER][1]
                 + frames[by_id[p]['N']].to_canonical(pts[by_id[p]['N']])[fg.RIGHT_CORNER][1]) / 2),
        float(frames[by_id[p]['N']].to_canonical(pts[by_id[p]['N']])[454][0]
              - frames[by_id[p]['N']].to_canonical(pts[by_id[p]['N']])[234][0]),
        float(frames[by_id[p]['N']].to_canonical(pts[by_id[p]['N']])[152][1]
              - frames[by_id[p]['N']].to_canonical(pts[by_id[p]['N']])[168][1]),
    ] for p in pid_ab])

    def block(names: list[str]) -> np.ndarray | None:
        """Columns with >=80% coverage; missing cells filled with the column mean.
        Requiring every variable on every model wipes the sample out."""
        usable = [n for n in names
                  if sum(1 for p in pid_ab if p in norm and n in norm[p]) >= 0.8 * len(pid_ab)]
        if not usable:
            return None
        cols = []
        for n in usable:
            v = np.array([norm[p][n] if (p in norm and n in norm[p]) else np.nan
                          for p in pid_ab], dtype=float)
            v[~np.isfinite(v)] = np.nanmean(v)
            cols.append(v)
        return np.column_stack(cols)

    blocks: dict[str, np.ndarray | None] = {
        "landmark geometry only": geo,
        "CFD physical measurements": block(norming.NUMERIC_COVARIATES),
        "CFD perceptual ratings": block(norming.RATING_COVARIATES),
    }
    phys, rate = blocks["CFD physical measurements"], blocks["CFD perceptual ratings"]
    if phys is not None and rate is not None:
        blocks["everything"] = np.column_stack([geo, phys, rate])

    res_b = {}
    for label, x in blocks.items():
        if x is None:
            print(f"  {label:<30} no usable columns")
            continue
        r2, lam = max((ridge_loo_r2(x, target, l), l)
                      for l in (0.3, 1, 3, 10, 30, 100, 300, 1000))
        print(f"  {label:<30} predictors={x.shape[1]:<3} leave-one-out R^2 = {r2:+.3f}")
        res_b[label] = {"predictors": int(x.shape[1]), "loo_r2": r2, "lambda": lam}
    results["B_predictability"] = res_b
    print(f"\n  For scale, the reliability ceiling is R^2 = {rel[chosen]['spearman_brown_2']:.3f}:"
          " no predictor\n  can beat that, because the target itself is measured with error.")
    print("  R^2 near zero means a face's measurements do not tell you how expressive\n"
          "  its owner is, so a static per-face gain model cannot work and the person\n"
          "  has to be observed.")

    # ---------------------------------------------------------------- C ----
    print(f"\n{'-'*78}\nC  How many observed smiles does calibration need?\n{'-'*78}")
    # Classical test theory. Observed amplitude A = T + e, with
    # r = var(T)/var(A) estimated by the closed-vs-toothy correlation.
    # Setting the gain from the mean of N observations leaves error var(e)/N,
    # so residual dispersion relative to the person's true scale is
    #     CV_resid(N) = CV_obs * sqrt((1 - r) / N).
    # With no calibration at all the residual is the full trait spread,
    #     CV_resid(pop) = CV_obs * sqrt(r).
    cv_pop = cv_obs * np.sqrt(r1)
    print(f"  observed amplitude spread            CV = {cv_obs:.3f}")
    print(f"  of which trait (reliable) share       r = {r1:.3f}\n")
    print(f"  {'calibration':<34} {'residual CV':>12} {'vs no calibration':>20}")
    print(f"  {'none (population mean gain)':<34} {cv_pop:>12.3f} {'--':>20}")
    rows_c = {"none": cv_pop}
    for n in (1, 2, 3, 5, 10, 20):
        cvn = cv_obs * np.sqrt((1 - r1) / n)
        print(f"  {f'mean of {n} observed smile(s)':<34} {cvn:>12.3f}"
              f" {1 - cvn / cv_pop:>19.0%}")
        rows_c[f"n={n}"] = cvn
    print("\n  Reading: one smile is barely better than no calibration, because one\n"
          "  posed smile is itself a noisy sample of the person. The gain has to be\n"
          "  averaged over several smiles -- which a few minutes of conversation\n"
          "  supplies and a silent waiting room does not.")
    print("\n  Caveat to state to Randy: CFD smiles are POSED, so part of the\n"
          "  closed-vs-toothy disagreement is the two shots being different acts\n"
          "  rather than measurement noise. That makes r a LOWER bound and these\n"
          "  sample counts an upper bound. Re-check on video before committing.")
    results["C_calibration"] = {"cv_observed": cv_obs, "reliability": r1,
                                "residual_cv": rows_c}

    # ------------------------------------------------------- sanity check --
    print(f"\n{'-'*78}\nSanity check: does resting corner rise track rated 'Happy'?\n{'-'*78}")
    rr, hh = [], []
    for pid, m in by_id.items():
        if "N" not in m or pid not in norm or "Happy" not in norm[pid]:
            continue
        cn = frames[m["N"]].to_canonical(pts[m["N"]])
        lip = cn[fg.OUTER_LIP]
        rr.append(float(np.median(lip[:, 1])
                        - (cn[fg.LEFT_CORNER][1] + cn[fg.RIGHT_CORNER][1]) / 2))
        hh.append(norm[pid]["Happy"])
    r_h = float(np.corrcoef(rr, hh)[0, 1])
    print(f"  r(resting corner rise, rated Happy of the neutral photo) = {r_h:+.3f}  n={len(rr)}")
    print("  Weak but reliable at this n: resting mouth curvature carries a little of\n"
          "  what raters call a happy-looking face, and it is not the whole story.")
    results["sanity_resting_vs_rated_happy"] = {"r": r_h, "n": len(rr)}

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
