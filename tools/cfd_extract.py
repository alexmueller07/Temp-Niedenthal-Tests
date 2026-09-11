"""Extract face landmarks from the Chicago Face Database into a local cache.

Run once; everything in the S0 analysis reads the cache. Images never leave the
machine and the cache is gitignored -- CFD's licence forbids redistribution and
this repo is public.

    .venv/Scripts/python.exe tools/cfd_extract.py
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from facegeom import Landmarker  # noqa: E402

CFD_ROOT = Path(r"C:\Users\amuel\Downloads\cfd\CFD Version 3.0\Images")
CACHE = Path(__file__).parent.parent / "artifacts" / "cache" / "cfd_landmarks3d.npz"

# CFD-BF-001-021-HC.jpg -> set BF, target BF-001, photo 021, expression HC
NAME_RE = re.compile(
    r"^CFD-(?P<grp>[A-Z]{2})-(?P<num>\d+)-(?P<photo>\d+)-(?P<expr>[A-Z]+)\.jpg$",
    re.IGNORECASE,
)

MAX_DIM = 1024  # detection resolution; every downstream measure is scale-free


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=CFD_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", type=Path, default=CACHE)
    args = ap.parse_args()

    files = sorted(args.root.rglob("*.jpg"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no images under {args.root}", file=sys.stderr)
        return 1

    print(f"{len(files)} images under {args.root}")
    lm = Landmarker()

    ids, exprs, groups = [], [], []
    pts_all, dims, bs_all, pose_all = [], [], [], []
    bs_names: list[str] | None = None
    failed: list[str] = []
    t0 = time.time()

    for k, f in enumerate(files):
        m = NAME_RE.match(f.name)
        if not m:
            failed.append(f"{f.name}: unparsed name")
            continue

        img = cv2.imread(str(f))
        if img is None:
            failed.append(f"{f.name}: unreadable")
            continue

        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            s = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        det = lm.detect(img)
        if det is None:
            failed.append(f"{f.name}: no face")
            continue
        pts, bs, pose = det

        if bs_names is None:
            bs_names = sorted(bs)
        ids.append(f"{m['grp']}-{m['num']}")
        groups.append(m["grp"])
        exprs.append(m["expr"].upper())
        pts_all.append(pts.astype(np.float32))
        dims.append((w, h))
        bs_all.append([bs.get(n, 0.0) for n in bs_names])
        pose_all.append(pose if pose is not None else np.full((4, 4), np.nan))

        if (k + 1) % 100 == 0:
            rate = (k + 1) / (time.time() - t0)
            print(f"  {k+1}/{len(files)}  {rate:.1f} img/s  {len(failed)} failed")

    lm.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        ids=np.array(ids), groups=np.array(groups), exprs=np.array(exprs),
        pts=np.array(pts_all, dtype=np.float32),
        dims=np.array(dims, dtype=np.int32),
        blendshapes=np.array(bs_all, dtype=np.float32),
        bs_names=np.array(bs_names if bs_names else []),
        poses=np.array(pose_all, dtype=np.float32),
    )

    print(f"\nwrote {args.out}  ({len(ids)} faces, {len(failed)} failures, "
          f"{time.time()-t0:.0f}s)")
    for line in failed[:20]:
        print("  !", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
