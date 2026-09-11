"""Per-identity resting mouth shape, from each person's neutral photograph.

Paired with tools/export_amplitudes.py: together they let a stills batch run
measure the algorithm rather than the behaviour of its cold-start fallbacks.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import facegeom as fg

HERE = Path(__file__).parent.parent
z = np.load(HERE / "artifacts" / "cache" / "cfd_landmarks3d.npz", allow_pickle=False)
ids, exprs, pts = z["ids"], z["exprs"], z["pts"].astype(float)
neutral = np.where(exprs == "N")[0]
template = fg.scale_template_to_ipd(
    fg.build_template([pts[i][fg.RIGID_IDX, :2] for i in neutral]))

out: dict[str, list[float]] = {}
for i in neutral:
    f = fg.fit_frame(pts[i], template)
    out[str(ids[i])] = [round(float(v), 6)
                        for v in f.to_canonical(pts[i])[fg.DRIVEN].reshape(-1)]

p = HERE / "artifacts" / "rests.json"
p.write_text(json.dumps(out), encoding="utf-8")
print(f"wrote {p}: {len(out)} identities x {len(next(iter(out.values())))} values")
