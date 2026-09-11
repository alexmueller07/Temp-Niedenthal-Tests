"""Per-identity smile amplitude, measured from each person's own smile photos.

Feeding these to the harness simulates a calibration that has fully converged,
which answers "how equal would the dose be if we knew each person's expressive
range" -- the ceiling any live scheme could reach. Averaging the closed-mouth
and toothy measurements uses the more reliable of the two estimates available
(Spearman-Brown 0.72 vs 0.56 for a single one).
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
frames = {i: fg.fit_frame(pts[i], template) for i in range(len(ids))}

by_id: dict[str, dict[str, int]] = {}
for i, (pid, e) in enumerate(zip(ids, exprs)):
    by_id.setdefault(str(pid), {})[str(e)] = i

out: dict[str, float] = {}
for pid, m in by_id.items():
    if "N" not in m:
        continue
    vals = []
    for e in ("HC", "HO"):
        if e not in m:
            continue
        cn = frames[m["N"]].to_canonical(pts[m["N"]])[fg.DRIVEN]
        cs = frames[m[e]].to_canonical(pts[m[e]])[fg.DRIVEN]
        y = (cs - cn).reshape(-1)
        b = fg.nuisance_basis(cn, fg.DRIVEN)
        vals.append(float(np.linalg.norm(
            fg.project_out(y, [b[k] for k in ("jaw_open", "lip_part")]))))
    if vals:
        out[pid] = sum(vals) / len(vals)

p = HERE / "artifacts" / "amplitudes.json"
p.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
a = np.array(list(out.values()))
print(f"wrote {p}: {len(out)} identities, "
      f"p10/p50/p90 = {np.percentile(a,10):.3f}/{np.percentile(a,50):.3f}/{np.percentile(a,90):.3f}")
