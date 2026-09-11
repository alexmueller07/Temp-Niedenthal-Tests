"""Does morphing toward a person's OWN smile look like a smile?

The geometric warp moves landmarks and nothing else: no teeth, no nasolabial
fold, no cheek shading, no change in how light falls. Randy's reaction to it was
"it just stretches left to right and looks like a messy warp", which is the same
thing the numbers said — 58% of a full smile's displacement but only 21% of the
action-unit change a real smile produces.

This tests the alternative before any of it is built into the live pipeline:
take a person's neutral photograph and their own smiling photograph, warp BOTH
to an intermediate geometry, and cross-dissolve. That is the textbook image
morph, and it is the same construction psychology already uses to build
expression continua — so an intermediate frame is "this person, t of the way to
their own smile" rather than "this person with stretched pixels".

If the intermediate frames read as smiles, the live version is engineering: the
neutral frame becomes the live camera frame and the smile frame becomes a
keyframe captured earlier in the session.

    .venv/Scripts/python.exe tools/proto_morph.py --ids BF-001,WM-004
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import facegeom as fg  # noqa: E402

HERE = Path(__file__).parent.parent
FRAMES = Path(r"C:\lab-corpus") / "frames"
OUT = HERE / "artifacts" / "figures" / "faces"

# Fractions of the way to the person's own smile.
STEPS = (0.0, 0.25, 0.5, 0.75, 1.0)


def delaunay_triangles(points: np.ndarray, w: int, h: int) -> list[tuple[int, int, int]]:
    """Triangulate once, on the average shape, so both images use the same mesh.

    Triangulating each shape separately would give different topologies and the
    two warps would not correspond.
    """
    sub = cv2.Subdiv2D((0, 0, w, h))
    for p in points:
        sub.insert((float(np.clip(p[0], 0, w - 1)), float(np.clip(p[1], 0, h - 1))))
    index = {}
    for i, p in enumerate(points):
        index[(round(float(p[0])), round(float(p[1])))] = i

    tris = []
    for t in sub.getTriangleList():
        pts = [(round(t[0]), round(t[1])), (round(t[2]), round(t[3])),
               (round(t[4]), round(t[5]))]
        ids = []
        for q in pts:
            best, bestd = None, 1e9
            for k, i in index.items():
                d = (k[0] - q[0]) ** 2 + (k[1] - q[1]) ** 2
                if d < bestd:
                    best, bestd = i, d
            if bestd > 4:
                ids = []
                break
            ids.append(best)
        if len(ids) == 3 and len(set(ids)) == 3:
            tris.append(tuple(ids))
    return tris


def warp_to(img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
            tris: list[tuple[int, int, int]]) -> np.ndarray:
    """Piecewise-affine warp of `img` so src_pts land on dst_pts."""
    out = np.zeros_like(img)
    for a, b, c in tris:
        s = np.float32([src_pts[a], src_pts[b], src_pts[c]])
        d = np.float32([dst_pts[a], dst_pts[b], dst_pts[c]])
        r_s = cv2.boundingRect(s)
        r_d = cv2.boundingRect(d)
        if r_s[2] <= 0 or r_s[3] <= 0 or r_d[2] <= 0 or r_d[3] <= 0:
            continue
        s_off = s - np.float32([r_s[0], r_s[1]])
        d_off = d - np.float32([r_d[0], r_d[1]])
        patch = img[r_s[1]:r_s[1] + r_s[3], r_s[0]:r_s[0] + r_s[2]]
        if patch.size == 0:
            continue
        m = cv2.getAffineTransform(s_off, d_off)
        warped = cv2.warpAffine(patch, m, (r_d[2], r_d[3]), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((r_d[3], r_d[2], 3), np.float32)
        cv2.fillConvexPoly(mask, np.int32(d_off), (1.0, 1.0, 1.0), cv2.LINE_AA)
        region = out[r_d[1]:r_d[1] + r_d[3], r_d[0]:r_d[0] + r_d[2]]
        if region.shape[:2] != mask.shape[:2]:
            continue
        region[:] = region * (1 - mask) + warped * mask
    return out


def control_points(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """Face landmarks plus a frame border, so the warp covers the whole image
    and the background stays put."""
    border = np.float32([[0, 0], [w // 2, 0], [w - 1, 0], [0, h // 2], [w - 1, h // 2],
                         [0, h - 1], [w // 2, h - 1], [w - 1, h - 1]])
    return np.vstack([pts[:, :2].astype(np.float32), border])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--expr", default="HC", choices=["HC", "HO"],
                    help="HC = closed-mouth smile, HO = toothy")
    args = ap.parse_args()

    ids = [i for i in args.ids.split(",") if i] or \
        sorted({p.name.split("_")[0] for p in FRAMES.glob(f"*_{args.expr}.jpg")})[:3]

    lm = fg.Landmarker()
    OUT.mkdir(parents=True, exist_ok=True)
    panels = []

    for pid in ids:
        n_path = FRAMES / f"{pid}_N.jpg"
        s_path = FRAMES / f"{pid}_{args.expr}.jpg"
        if not n_path.exists() or not s_path.exists():
            print(f"  skip {pid}: missing a frame")
            continue
        img_n, img_s = cv2.imread(str(n_path)), cv2.imread(str(s_path))
        det_n, det_s = lm.detect(img_n), lm.detect(img_s)
        if det_n is None or det_s is None:
            print(f"  skip {pid}: no face")
            continue
        h, w = img_n.shape[:2]
        p_n = control_points(det_n[0], w, h)
        p_s = control_points(det_s[0], w, h)

        mid = (p_n + p_s) / 2
        tris = delaunay_triangles(mid, w, h)

        row = []
        for t in STEPS:
            target = (1 - t) * p_n + t * p_s
            a = warp_to(img_n.astype(np.float32), p_n, target, tris)
            b = warp_to(img_s.astype(np.float32), p_s, target, tris)
            blend = np.clip((1 - t) * a + t * b, 0, 255).astype(np.uint8)
            row.append(blend)

        # Crop to the lower face, where the difference lives.
        cx, cy = det_n[0][fg.OUTER_LIP][:, :2].mean(0)
        half = int(fg.ipd_px(det_n[0]) * 1.7)
        y0, y1 = max(0, int(cy - half)), int(cy + half * 0.8)
        x0, x1 = max(0, int(cx - half)), int(cx + half)
        panels.append([r[y0:y1, x0:x1] for r in row])
        print(f"  morphed {pid}")

    lm.close()
    if not panels:
        print("nothing to show")
        return 1

    ph = min(p[0].shape[0] for p in panels)
    pw = min(p[0].shape[1] for p in panels)
    pad = 6
    sheet = np.full((len(panels) * (ph + pad) + pad + 26,
                     len(STEPS) * (pw + pad) + pad, 3), 245, np.uint8)
    for r, row in enumerate(panels):
        for c, tile in enumerate(row):
            y = 26 + pad + r * (ph + pad)
            x = pad + c * (pw + pad)
            sheet[y:y + ph, x:x + pw] = tile[:ph, :pw]
    for c, t in enumerate(STEPS):
        label = "neutral" if t == 0 else ("their own smile" if t == 1 else f"{int(t*100)}%")
        cv2.putText(sheet, label, (pad + c * (pw + pad), 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)

    dest = OUT / f"6-own-smile-morph-{args.expr}.png"
    cv2.imwrite(str(dest), sheet)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
