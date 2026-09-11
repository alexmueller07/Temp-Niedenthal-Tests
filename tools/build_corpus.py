"""Turn CFD studio portraits into webcam-framed test frames.

CFD photographs are 2444x1718, studio-lit, with the face filling much of the
frame. Lab conditions are a 1280x720 webcam with the face a good deal smaller.
Measuring on the raw portraits would test the algorithm in conditions it will
never see, so every face is re-framed to a realistic webcam geometry first:
1280x720, interpupillary distance about 95 px, eyes a little above centre.

That reframing is also the knob for the framing ablation — the same code can
place a face high or low in the frame, near or far.

Output goes to C:\\lab-corpus (outside the repo and outside OneDrive; CFD's
licence forbids redistributing images and a few thousand JPEGs inside a synced
folder causes sync storms).

    .venv/Scripts/python.exe tools/build_corpus.py --n 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import facegeom as fg  # noqa: E402

HERE = Path(__file__).parent.parent
CACHE = HERE / "artifacts" / "cache" / "cfd_landmarks3d.npz"
CFD_ROOT = Path(r"C:\Users\amuel\Downloads\cfd\CFD Version 3.0\Images")
OUT_ROOT = Path(r"C:\lab-corpus")

FRAME_W, FRAME_H = 1280, 720
TARGET_IPD = 95.0        # a person sitting a normal distance from a laptop
TARGET_EYE_Y = 0.38      # eyes a little above centre, as people actually sit


def find_source(root: Path, model_id: str, expr: str) -> Path | None:
    grp = model_id.split("-")[0]
    for sub in ("CFD", "CFD-MR", "CFD-INDIA"):
        d = root / sub
        if not d.exists():
            continue
        # CFD keeps a folder per target; the extension sets keep files flat.
        for pat in (f"{model_id}/CFD-{model_id}-*-{expr}.jpg",
                    f"CFD-{model_id}-*-{expr}.jpg"):
            hits = sorted(d.glob(pat))
            if hits:
                return hits[0]
    _ = grp
    return None


def reframe(img: np.ndarray, pts: np.ndarray, scale_from: float,
            ipd_target: float, eye_y_frac: float,
            out_w: int = FRAME_W, out_h: int = FRAME_H) -> np.ndarray:
    """Place a detected face into a webcam-shaped frame at a chosen size and
    height, via a similarity warp. Areas outside the original image are filled
    with a mid grey rather than black, so the edge does not read as a hard
    vignette to the landmark detector."""
    p = np.asarray(pts, float)[:, :2] / scale_from
    left, right = p[fg.LEFT_IRIS], p[fg.RIGHT_IRIS]
    ipd0 = float(np.hypot(*(right - left)))
    if ipd0 < 1e-6:
        raise ValueError("degenerate interpupillary distance")

    s = ipd_target / ipd0
    theta = -math.atan2(right[1] - left[1], right[0] - left[0])
    c, sn = math.cos(theta) * s, math.sin(theta) * s
    eyes = (left + right) / 2.0
    tx = out_w / 2.0 - (c * eyes[0] - sn * eyes[1])
    ty = out_h * eye_y_frac - (sn * eyes[0] + c * eyes[1])
    m = np.array([[c, -sn, tx], [sn, c, ty]], dtype=np.float32)
    return cv2.warpAffine(img, m, (out_w, out_h), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(110, 112, 118))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="identities to include")
    ap.add_argument("--out", type=Path, default=OUT_ROOT / "frames")
    ap.add_argument("--paired-only", action="store_true",
                    help="only identities that also have both smile photographs")
    args = ap.parse_args()

    z = np.load(CACHE, allow_pickle=False)
    ids, exprs, pts, dims = z["ids"], z["exprs"], z["pts"].astype(float), z["dims"]

    by_id: dict[str, dict[str, int]] = {}
    for i, (pid, e) in enumerate(zip(ids, exprs)):
        by_id.setdefault(str(pid), {})[str(e)] = i

    chosen = [p for p, m in by_id.items()
              if "N" in m and (not args.paired_only or {"HC", "HO"} <= set(m))]
    # Spread across the CFD sub-sets so the sample is not all one group.
    chosen.sort(key=lambda p: (p.split("-")[0], p))
    step = max(1, len(chosen) // args.n)
    chosen = chosen[::step][: args.n]

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    failed = []

    for k, pid in enumerate(chosen):
        for expr in ("N", "HC", "HO"):
            if expr not in by_id[pid]:
                continue
            src = find_source(CFD_ROOT, pid, expr)
            if src is None:
                failed.append(f"{pid}/{expr}: source not found")
                continue
            img = cv2.imread(str(src))
            if img is None:
                failed.append(f"{pid}/{expr}: unreadable")
                continue

            i = by_id[pid][expr]
            # Landmarks were detected on a copy scaled so max(dim) == 1024.
            scale_from = int(dims[i][0]) / img.shape[1]
            try:
                out = reframe(img, pts[i], scale_from, TARGET_IPD, TARGET_EYE_Y)
            except ValueError as e:
                failed.append(f"{pid}/{expr}: {e}")
                continue

            name = f"{pid}_{expr}.jpg"
            cv2.imwrite(str(args.out / name), out,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            manifest.append({"id": pid, "expr": expr, "file": name,
                             "group": pid.split("-")[0]})

        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(chosen)} identities")

    (args.out.parent / "manifest.json").write_text(
        json.dumps({"frameW": FRAME_W, "frameH": FRAME_H,
                    "targetIpd": TARGET_IPD, "eyeYFrac": TARGET_EYE_Y,
                    "frames": manifest}, indent=2), encoding="utf-8")

    print(f"\nwrote {len(manifest)} frames to {args.out}")
    print(f"manifest: {args.out.parent / 'manifest.json'}")
    for line in failed[:15]:
        print("  !", line)
    if len(failed) > 15:
        print(f"  ... and {len(failed)-15} more failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
