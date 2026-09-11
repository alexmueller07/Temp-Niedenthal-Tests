"""How much does each morph stretch skin, and is that the warp's fault?

The normalized implementation reports local area stretch of about 2.4x at the
strongest preset, which sounds alarming. Before blaming the warp it is worth
asking whether the *commanded deformation* already demands that much — because
if it does, no warp can avoid it and the fix is a smaller preset, not a better
algorithm.

So this compares the two implementations on the deformation itself rather than
on the rendered pixels: evaluate each one's displacement at the twenty outer-lip
landmarks of a real face, and measure the strain between neighbouring landmarks,
|d_i - d_j| / |p_i - p_j|. That is the fraction by which the skin between two
points has to stretch or compress. It is a property of what was asked for, and
it is directly comparable between the two because neither warp is involved.

    .venv/Scripts/python.exe tools/stretch_compare.py
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import facegeom as fg  # noqa: E402

HERE = Path(__file__).parent.parent
CACHE = HERE / "artifacts" / "cache" / "cfd_landmarks3d.npz"
ALPHAS = (1.35, 1.9)


def parse_gen(block: str) -> dict[int, tuple[float, float]]:
    src = (HERE / "src" / "algo" / "faceModel.gen.ts").read_text(encoding="utf-8")
    m = re.search(block + r".*?= \[(.*?)\n\]", src, re.S)
    return {int(a): (float(x), float(y))
            for a, x, y in re.findall(r"\[(\d+), \[([-\d.]+), ([-\d.]+)\]\]", m.group(1))}


def scalar_const(name: str) -> float:
    src = (HERE / "src" / "algo" / "faceModel.gen.ts").read_text(encoding="utf-8")
    return float(src.split(f"{name} = ")[1].split("\n")[0])


def current_field(pts: np.ndarray, alpha: float, w: int, h: int) -> np.ndarray:
    """The production displacement, evaluated at each outer-lip landmark.

    Reimplements renderer/lib/faceMorph.ts:400-428 pointwise. The production
    code evaluates this on a 12x8 grid and interpolates; evaluating it at the
    landmarks instead gives the deformation it is asking for, sampled where the
    anatomy is.
    """
    p = np.asarray(pts, float)[:, :2]
    lc, rc = p[fg.LEFT_CORNER], p[fg.RIGHT_CORNER]
    centre = (lc + rc) / 2
    mw = float(np.hypot(*(rc - lc)))

    lip = p[fg.OUTER_LIP]
    min_x, min_y = lip.min(0)
    max_x, max_y = lip.max(0)
    pad_x, pad_y = mw * 0.55, mw * 0.7
    roi_x = max(0.0, min_x - pad_x)
    roi_y = max(0.0, min_y - pad_y)
    roi_w = min(float(w), max_x + pad_x) - roi_x
    roi_h = min(float(h), max_y + pad_y) - roi_y

    nose, le, re_ = p[fg.NOSE_TIP], p[fg.LEFT_FACE_EDGE], p[fg.RIGHT_FACE_EDGE]
    dl, dr = abs(nose[0] - le[0]), abs(re_[0] - nose[0])
    sym = min(dl, dr) / max(1e-3, max(dl, dr))
    yaw_scale = float(np.clip((sym - fg.YAW_FADE_END)
                              / (fg.YAW_FADE_START - fg.YAW_FADE_END), 0, 1))
    strength = (alpha - 1.0) * yaw_scale
    mag = abs(strength) * mw
    sigma_y = mw * 0.6

    out = np.zeros((len(fg.OUTER_LIP), 2))
    for k, idx in enumerate(fg.OUTER_LIP):
        sx, sy = p[idx]
        u = (sx - roi_x) / max(roi_w, 1e-9)
        v = (sy - roi_y) / max(roi_h, 1e-9)
        xn = (sx - centre[0]) / (mw / 2)
        vy = math.exp(-((sy - centre[1]) ** 2) / (2 * sigma_y * sigma_y))
        win = (math.sin(math.pi * float(np.clip(u, 0, 1)))
               * math.sin(math.pi * float(np.clip(v, 0, 1))))
        cw = min(1.6, xn * xn) * vy * win
        d = mag * fg.SMILE_GAIN * cw
        out[k] = (math.copysign(math.cos(fg.SMILE_ANGLE_RAD) * d, xn),
                  -math.sin(fg.SMILE_ANGLE_RAD) * d)
    return out


def neighbour_strain(positions: np.ndarray, disp: np.ndarray) -> float:
    """Worst strain between any landmark and its nearest neighbour."""
    worst = 0.0
    n = len(positions)
    for i in range(n):
        best_d, best_j = None, -1
        for j in range(n):
            if i == j:
                continue
            d = float(np.hypot(*(positions[i] - positions[j])))
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        gap = max(best_d or 0.0, 1e-9)
        dd = float(np.hypot(*(disp[i] - disp[best_j])))
        worst = max(worst, dd / gap)
    return worst


def main() -> int:
    z = np.load(CACHE, allow_pickle=False)
    ids, exprs, pts, dims = z["ids"], z["exprs"], z["pts"].astype(float), z["dims"]
    neutral = np.where(exprs == "N")[0]

    template = fg.scale_template_to_ipd(
        fg.build_template([pts[i][fg.RIGID_IDX, :2] for i in neutral]))
    axis_map = parse_gen("SMILE_AXIS")
    alpha_scale = scalar_const("ALPHA_TO_SMILE_UNITS")
    amps = json.loads((HERE / "artifacts" / "amplitudes.json").read_text(encoding="utf-8"))

    axis = np.array([axis_map[i] for i in fg.OUTER_LIP])

    sample = [i for i in neutral if str(ids[i]) in amps][:120]
    print(f"\n{'='*72}\nCommanded deformation: strain between neighbouring lip "
          f"landmarks\n{'='*72}")
    print(f"  {len(sample)} faces. Strain 1.0 means the skin between two "
          f"neighbouring\n  landmarks has to stretch to double its length.\n")
    print(f"  {'alpha':<8}{'current':>22}{'normalized':>22}")

    out: dict = {}
    for alpha in ALPHAS:
        cur_s, new_s = [], []
        for i in sample:
            w, h = int(dims[i][0]), int(dims[i][1])
            frame = fg.fit_frame(pts[i], template)
            pos_px = np.asarray(pts[i], float)[fg.OUTER_LIP, :2]

            cur_s.append(neighbour_strain(pos_px, current_field(pts[i], alpha, w, h)))

            # Normalized: canonical displacement, mapped to pixels the same way
            # the runtime does.
            delta = (alpha - 1.0) * alpha_scale * amps[str(ids[i])]
            d_canon = axis * delta
            c, s_ = math.cos(frame.theta), math.sin(frame.theta)
            r = np.array([[c, -s_], [s_, c]])
            new_s.append(neighbour_strain(pos_px, (d_canon @ r.T) * frame.scale))

        print(f"  {alpha:<8}{st.median(cur_s):>14.2f} (med){st.median(new_s):>16.2f} (med)")
        print(f"  {'':<8}{st.fmean(cur_s):>14.2f} (mean){st.fmean(new_s):>15.2f} (mean)")
        out[str(alpha)] = {
            "current_median": st.median(cur_s), "current_mean": st.fmean(cur_s),
            "normalized_median": st.median(new_s), "normalized_mean": st.fmean(new_s),
        }

    # The reference that decides whether either number is alarming: how much a
    # REAL smile strains the same skin, measured from each person's own pair of
    # photographs. Without it, "strain 1.08" is a number with no scale.
    by: dict[str, dict[str, int]] = {}
    for i, (pid, e) in enumerate(zip(ids, exprs)):
        by.setdefault(str(pid), {})[str(e)] = i

    real: dict[str, list[float]] = {"HC": [], "HO": []}
    for pid, m in by.items():
        if "N" not in m:
            continue
        fN = fg.fit_frame(pts[m["N"]], template)
        posN = np.asarray(pts[m["N"]], float)[fg.OUTER_LIP, :2]
        cN = fN.to_canonical(pts[m["N"]])[fg.OUTER_LIP]
        c, s_ = math.cos(fN.theta), math.sin(fN.theta)
        r = np.array([[c, -s_], [s_, c]])
        for e in ("HC", "HO"):
            if e not in m:
                continue
            cE = fg.fit_frame(pts[m[e]], template).to_canonical(pts[m[e]])[fg.OUTER_LIP]
            real[e].append(neighbour_strain(posN, ((cE - cN) @ r.T) * fN.scale))

    print()
    print("  reference, the same measure on REAL smiles:")
    for e, label in (("HC", "real closed-mouth smile"), ("HO", "real toothy smile")):
        v = real[e]
        print(f"  {label:<26}{st.median(v):>8.2f} (med, n={len(v)})")
        out[f"real_{e}"] = {"median": st.median(v), "mean": st.fmean(v), "n": len(v)}

    print()
    print("  Reading: the number to judge against is what a real smile does, not")
    print("  zero. A deformation that strains skin less than a real smile of the")
    print("  same size is not gentler, it is less like a smile.")
    print("  Caveat: real skin is elastic and self-shadows; an image warp only")
    print("  moves texture, so a faithful magnitude does not guarantee it LOOKS")
    print("  right. That is a question for eyes on the demo, not this script.")

    dest = HERE / "artifacts" / "stretch_compare.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
