"""Stage 0: answer the design questions offline, before rendering anything.

Six questions, all from cached landmarks -- no image rendering, no browser:

  Q1  How unequal is the CURRENT morph across identities? (the baseline number)
  Q2  Where does the field actually peak -- the lip corner, or the cheek?
  Q3  Is cross-person variation in a person's own smile about DIRECTION or
      AMPLITUDE? This decides whether the online axis learner gets built at all.
  Q4  Do a toothy smile and a closed-mouth smile give the same axis once the
      jaw-open and lip-part components are projected out? (Randy's question.)
  Q5  Can cheap observables predict a person's own smile amplitude? (If yes, a
      static covariate gain model replaces online calibration entirely.)
  Q6  How much does the current morph's dose move with camera framing alone --
      same face, different distance, height in frame, and roll?

    .venv/Scripts/python.exe tools/cfd_analysis.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import facegeom as fg  # noqa: E402

HERE = Path(__file__).parent.parent
CACHE = HERE / "artifacts" / "cache" / "cfd_landmarks3d.npz"
OUT = HERE / "artifacts" / "s0_results.json"
ALPHAS = (1.35, 1.9)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def cv(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / abs(x.mean())) if len(x) > 1 and x.mean() != 0 else float("nan")


def p90_p10(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    lo = np.percentile(x, 10)
    return float(np.percentile(x, 90) / lo) if lo > 0 else float("nan")


def summarise(name: str, x: np.ndarray, unit: str = "") -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    d = {
        "n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
        "cv": cv(x), "p10": float(np.percentile(x, 10)),
        "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
        "p90_over_p10": p90_p10(x), "min": float(x.min()), "max": float(x.max()),
    }
    print(f"  {name:<38} n={d['n']:<5} mean={d['mean']:.4f}{unit:<6} "
          f"CV={d['cv']:.3f}  P90/P10={d['p90_over_p10']:.2f}  "
          f"range [{d['min']:.4f}, {d['max']:.4f}]")
    return d


def ridge_r2(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> tuple[float, np.ndarray]:
    """Leave-one-out R^2 for a ridge fit. LOO, not in-sample, because with a
    dozen covariates and 150 rows an in-sample R^2 is mostly self-congratulation.
    """
    xs = (x - x.mean(0)) / (x.std(0) + 1e-12)
    xs = np.column_stack([xs, np.ones(len(xs))])
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        a = xs[m].T @ xs[m] + lam * np.eye(xs.shape[1])
        b = xs[m].T @ y[m]
        preds[i] = xs[i] @ np.linalg.solve(a, b)
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    a = xs.T @ xs + lam * np.eye(xs.shape[1])
    beta = np.linalg.solve(a, xs.T @ y)
    return 1.0 - ss_res / ss_tot, beta


# ---------------------------------------------------------------------------

def main() -> int:
    if not CACHE.exists():
        print(f"missing {CACHE}; run tools/cfd_extract.py first", file=sys.stderr)
        return 1

    z = np.load(CACHE, allow_pickle=False)
    ids, exprs, groups = z["ids"], z["exprs"], z["groups"]
    pts, dims = z["pts"].astype(float), z["dims"]
    results: dict = {"n_faces": int(len(ids))}

    print(f"\n{'='*78}\nSTAGE 0 -- CFD offline analysis  ({len(ids)} faces)\n{'='*78}")

    # --- canonical template from the neutral faces -------------------------
    neutral = np.where(exprs == "N")[0]
    template = fg.scale_template_to_ipd(
        fg.build_template([pts[i][fg.RIGID_IDX, :2] for i in neutral])
    )
    frames = {i: fg.fit_frame(pts[i], template) for i in range(len(ids))}
    resid = np.array([frames[i].residual for i in range(len(ids))])
    print(f"\ncanonical frame fit residual (IPD units): "
          f"median {np.median(resid):.4f}  p95 {np.percentile(resid, 95):.4f}")
    results["frame_residual_median"] = float(np.median(resid))

    # =======================================================================
    print(f"\n{'-'*78}\nQ1  Current morph: how unequal is the delivered dose?\n{'-'*78}")
    q1: dict = {}
    for alpha in ALPHAS:
        d_ipd, d_mw, mw_ipd, cw, wins, clamp = [], [], [], [], [], []
        for i in neutral:
            w, h = int(dims[i][0]), int(dims[i][1])
            m = fg.current_morph_delivery(pts[i], alpha, w, h)
            ipd = fg.ipd_px(pts[i])
            if ipd < 1e-6:
                continue
            d_ipd.append(m.d_px / ipd)
            d_mw.append(m.d_px / m.mouth_width)
            mw_ipd.append(m.mouth_width / ipd)
            cw.append(m.corner_w)
            wins.append(m.win)
            clamp.append(m.roi_clamped)

        print(f"\nalpha = {alpha}")
        s_ipd = summarise("delivered corner travel / IPD", np.array(d_ipd))
        s_mw = summarise("delivered corner travel / mouthWidth", np.array(d_mw))
        s_geo = summarise("mouthWidth / IPD (anatomy)", np.array(mw_ipd))
        summarise("field weight at the corner", np.array(cw))
        q1[str(alpha)] = {"per_ipd": s_ipd, "per_mouthwidth": s_mw,
                          "mouthwidth_per_ipd": s_geo,
                          "roi_clamped_frac": float(np.mean(clamp))}
    results["q1_current_dispersion"] = q1
    print("\n  Reading: CV of 'delivered / IPD' is how unequal the manipulation is\n"
          "  across faces in a body-normalized unit, holding framing constant.")

    # =======================================================================
    print(f"\n{'-'*78}\nQ2  Where does the displacement field peak?\n{'-'*78}")
    peaks = []
    for i in neutral:
        w, h = int(dims[i][0]), int(dims[i][1])
        m = fg.current_morph_delivery(pts[i], 1.9, w, h)
        if np.isfinite(m.peak_ratio):
            peaks.append(m.peak_ratio)
    s_peak = summarise("field peak / field at lip corner", np.array(peaks), " x")
    results["q2_peak_over_corner"] = s_peak
    print("\n  Reading: >1 means the warp pulls skin lateral to the mouth (cheek)\n"
          "  harder than it pulls the lip corner itself.")

    # =======================================================================
    print(f"\n{'-'*78}\nQ3/Q4  Each person's OWN smile: direction vs amplitude\n{'-'*78}")

    by_id: dict[str, dict[str, int]] = {}
    for i, (pid, e) in enumerate(zip(ids, exprs)):
        by_id.setdefault(str(pid), {})[str(e)] = i

    def smile_vector(i_neutral: int, i_smile: int, project: tuple[str, ...]):
        cn = frames[i_neutral].to_canonical(pts[i_neutral])[fg.DRIVEN]
        cs = frames[i_smile].to_canonical(pts[i_smile])[fg.DRIVEN]
        y = (cs - cn).reshape(-1)
        if project:
            basis = fg.nuisance_basis(cn, fg.DRIVEN)
            y = fg.project_out(y, [basis[k] for k in project])
        return y

    PROJECTIONS = {
        "none": (),
        "jaw_open": ("jaw_open",),
        "jaw_open+lip_part": ("jaw_open", "lip_part"),
    }

    q34: dict = {}
    for label, proj in PROJECTIONS.items():
        amp_hc, dir_hc, ids_hc = [], [], []
        cos_hc_ho = []
        for pid, m in by_id.items():
            if "N" not in m:
                continue
            if "HC" in m:
                y = smile_vector(m["N"], m["HC"], proj)
                a = float(np.linalg.norm(y))
                if a > 1e-6:
                    amp_hc.append(a)
                    dir_hc.append(y / a)
                    ids_hc.append(pid)
            if "HC" in m and "HO" in m:
                y1 = smile_vector(m["N"], m["HC"], proj)
                y2 = smile_vector(m["N"], m["HO"], proj)
                n1, n2 = np.linalg.norm(y1), np.linalg.norm(y2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_hc_ho.append(float(y1 @ y2 / (n1 * n2)))

        dir_hc_arr = np.array(dir_hc)
        gram = dir_hc_arr @ dir_hc_arr.T
        iu = np.triu_indices(len(dir_hc_arr), k=1)
        pair_cos = gram[iu]

        print(f"\nprojection = {label}   ({len(amp_hc)} identities with N+HC)")
        s_amp = summarise("own smile amplitude (IPD units)", np.array(amp_hc))
        print(f"  {'pairwise cos between people’s axes':<38} "
              f"median={np.median(pair_cos):.3f}  p10={np.percentile(pair_cos,10):.3f}")
        if cos_hc_ho:
            print(f"  {'cos(closed-mouth, toothy) WITHIN person':<38} "
                  f"median={np.median(cos_hc_ho):.3f}  "
                  f"p10={np.percentile(cos_hc_ho,10):.3f}  n={len(cos_hc_ho)}")

        q34[label] = {
            "amplitude": s_amp,
            "pairwise_direction_cos_median": float(np.median(pair_cos)),
            "pairwise_direction_cos_p10": float(np.percentile(pair_cos, 10)),
            "hc_vs_ho_cos_median": float(np.median(cos_hc_ho)) if cos_hc_ho else None,
            "hc_vs_ho_cos_p10": float(np.percentile(cos_hc_ho, 10)) if cos_hc_ho else None,
            "ids": ids_hc,
            "amps": [float(a) for a in amp_hc],
        }
    results["q34_own_smile"] = {k: {kk: vv for kk, vv in v.items()
                                    if kk not in ("ids", "amps")}
                                for k, v in q34.items()}
    print("\n  Reading: high pairwise cos => everyone's smile points the same way,\n"
          "  so learning a per-person DIRECTION buys nothing and amplitude is the\n"
          "  whole game. High cos(closed, toothy) => the projection works and the\n"
          "  kind of smile someone gives during calibration does not matter.")

    # =======================================================================
    print(f"\n{'-'*78}\nQ5  Is the current dose aligned with what each face needs?\n{'-'*78}")
    best = q34["jaw_open+lip_part"]
    amp_by_id = dict(zip(best["ids"], best["amps"]))
    rows_dose, rows_amp, rows_cov, cov_names = [], [], [], None

    for pid, amp in amp_by_id.items():
        i = by_id[pid]["N"]
        w, h = int(dims[i][0]), int(dims[i][1])
        m = fg.current_morph_delivery(pts[i], 1.9, w, h)
        ipd = fg.ipd_px(pts[i])
        if ipd < 1e-6:
            continue
        cn = frames[i].to_canonical(pts[i])
        lc, rc = cn[fg.LEFT_CORNER], cn[fg.RIGHT_CORNER]
        lip = cn[fg.OUTER_LIP]
        corner_y = (lc[1] + rc[1]) / 2.0
        cov = {
            "mouthwidth_per_ipd": m.mouth_width / ipd,
            "lip_height_per_ipd": float(lip[:, 1].max() - lip[:, 1].min()),
            "resting_corner_rise": float(np.median(lip[:, 1]) - corner_y),
            "lower_face_height": float(cn[152][1] - cn[168][1]) if len(cn) > 152 else 0.0,
            "face_width": float(cn[454][0] - cn[234][0]),
            "corner_sep_per_facewidth": float(np.linalg.norm(rc - lc) /
                                              max(1e-6, cn[454][0] - cn[234][0])),
        }
        cov_names = list(cov)
        rows_dose.append(m.d_px / ipd)
        rows_amp.append(amp)
        rows_cov.append([cov[k] for k in cov_names])

    dose = np.array(rows_dose)
    amp = np.array(rows_amp)
    cov_x = np.array(rows_cov)

    # In self-referenced units: what fraction of their own smile does the
    # current morph deliver? If that ratio were constant, the manipulation
    # would already be equal in the sense "same share of each person's smile".
    ratio = dose / amp
    s_ratio = summarise("delivered dose / own smile amplitude", ratio)
    r = float(np.corrcoef(dose, amp)[0, 1])
    print(f"  {'corr(delivered dose, own amplitude)':<38} r={r:+.3f}")
    results["q5_self_referenced"] = {"ratio": s_ratio, "corr_dose_amplitude": r}
    print("\n  Reading: CV of that ratio is the dispersion of the manipulation in\n"
          "  self-referenced units. r near 0 means the current morph is blind to\n"
          "  how expressive the face is; r<0 would mean it is actively backwards.")

    # covariate model for amplitude
    r2, beta = ridge_r2(cov_x, amp)
    print(f"\n  covariate model for own smile amplitude:  leave-one-out R^2 = {r2:.3f}")
    for n_, b_ in sorted(zip(cov_names, beta[:-1]), key=lambda t: -abs(t[1])):
        print(f"      {n_:<30} beta={b_:+.4f}")
    results["q5_covariate_model"] = {
        "loo_r2": r2, "covariates": cov_names,
        "beta": [float(b) for b in beta],
    }

    # =======================================================================
    print(f"\n{'-'*78}\nQ6  Same face, different camera: how much does the dose move?\n{'-'*78}")
    # Pure landmark arithmetic: place each face into a 1280x720 webcam frame at a
    # given interpupillary size, vertical position and head orientation, then
    # recompute what the current algorithm would deliver. Rotations go through a
    # real 3D rotate-and-reproject with a plausible webcam focal length, so
    # foreshortening is modelled rather than assumed away. Nothing is rendered.
    W, H = 1280, 720
    FOCAL = 0.9 * W
    sample = [by_id[p]["N"] for p in list(amp_by_id)[:80]]

    def reframe(p3, ipd_target, cy_frac, yaw, pitch, roll):
        if yaw or pitch or roll:
            q = fg.rotate_3d_project(p3, yaw, pitch, roll, FOCAL)
        else:
            q = np.asarray(p3, float)[:, :2].copy()
        ipd0 = math.hypot(*(q[fg.RIGHT_IRIS] - q[fg.LEFT_IRIS]))
        q = (q - q.mean(0)) * (ipd_target / max(ipd0, 1e-6))
        eyes = (q[fg.LEFT_IRIS] + q[fg.RIGHT_IRIS]) / 2.0
        return q - eyes + np.array([W / 2.0, H * cy_frac])

    conditions = {
        "distance (IPD 45..200 px)":
            [(v, 0.42, 0, 0, 0) for v in (45, 60, 80, 100, 130, 165, 200)],
        "height in frame (eyes 0.12..0.78 H)":
            [(110, v, 0, 0, 0) for v in (0.12, 0.25, 0.38, 0.50, 0.62, 0.70, 0.78)],
        "head roll (-25..+25 deg)":
            [(95, 0.42, 0, 0, v) for v in (-25, -15, -7, 0, 7, 15, 25)],
        "camera height / pitch (-20..+20 deg)":
            [(95, 0.42, 0, v, 0) for v in (-20, -12, -6, 0, 6, 12, 20)],
        "head yaw (-30..+30 deg)":
            [(95, 0.42, v, 0, 0) for v in (-30, -20, -10, 0, 10, 20, 30)],
    }

    q6: dict = {}
    for label, combos in conditions.items():
        within, clamped, gated = [], 0, 0
        for i in sample:
            vals = []
            for ipd_t, cy, yaw, pitch, roll in combos:
                p = reframe(pts[i], ipd_t, cy, yaw, pitch, roll)
                m = fg.current_morph_delivery(p, 1.9, W, H)
                vals.append(m.d_px / ipd_t)
                clamped += int(m.roi_clamped)
                gated += int(m.yaw_scale < 0.999)
            v = np.array(vals)
            within.append(v.std(ddof=1) / v.mean() if v.mean() > 1e-9 else np.nan)
        cells = len(sample) * len(combos)
        s_ = summarise(f"within-face CV, {label}", np.array(within))
        print(f"  {'    ROI clipped by the frame edge in':<40} {clamped}/{cells} cells")
        print(f"  {'    yaw gate attenuating the morph in':<40} {gated}/{cells} cells")
        q6[label] = {"within_face_cv": s_, "roi_clamped": clamped,
                     "yaw_gated": gated, "cells": cells}
    results["q6_framing"] = q6
    print("\n  Reading: a face identical in every way except how it sits in front of\n"
          "  the camera should get an identical dose, so within-face CV should be ~0.\n"
          "  The yaw-gate count matters separately: that gate fades the morph toward\n"
          "  zero, and head turns track who is speaking -- so every cell it touches is\n"
          "  a dose that silently depends on conversational role.")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
