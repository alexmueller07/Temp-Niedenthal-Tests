"""Shared face geometry for the offline (S0) analysis.

Everything here mirrors what the TypeScript algorithm will do, so the offline
findings transfer directly. Three pieces:

  1. Landmark extraction via the *same* MediaPipe model the app ships
     (renderer/public/mediapipe/face_landmarker.task), so indices and model
     behaviour match the production pipeline exactly.
  2. A canonical face frame: weighted Umeyama similarity fit on rigid landmarks
     only. Rigid means bone -- eye corners, nose bridge, nasion. Never the
     mouth (it is the thing being measured), never the jaw (it moves when
     talking), never the face-oval silhouette points 234/454 that the current
     app uses for its yaw estimate (those are the silhouette, not the skull:
     they slide across the face as the head turns).
  3. An analytic model of the *current* morph's delivered corner displacement,
     so the baseline dispersion can be computed without rendering anything.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe FaceMesh, 478-point refined model)
# ---------------------------------------------------------------------------

# Outer lip ring -- identical list and order to LIP_INDICES in faceMorph.ts:44.
OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
             291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Inner lip ring, same winding: left corner, along the lower lip, right corner,
# back along the upper lip.
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
             308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# The driven set: what the morph is allowed to move.
DRIVEN = OUTER_LIP + INNER_LIP

LEFT_CORNER = 61
RIGHT_CORNER = 291

# Iris centres, present only in the 478-point refined model. These give a true
# interpupillary distance, which is the most stable scale reference on a face:
# rigid, expression-invariant, and unaffected by facial hair.
LEFT_IRIS = 468
RIGHT_IRIS = 473

# Rigid anchors for the canonical fit, with weights. Eye corners are the most
# reliable; the nose bridge adds vertical extent so the fit is not degenerate
# along y; nasion ties the top. Nose tip is deliberately down-weighted because
# the nostrils flare during AU9 and, to a lesser extent, AU12.
RIGID_ANCHORS: dict[int, float] = {
    33: 1.0, 133: 1.0, 362: 1.0, 263: 1.0,   # outer/inner eye corners
    468: 1.2, 473: 1.2,                      # iris centres
    168: 0.9, 6: 0.9, 197: 0.7, 195: 0.6,    # nose bridge, descending
    8: 0.6,                                  # nasion
    1: 0.3,                                  # nose tip, down-weighted
}
RIGID_IDX = np.array(sorted(RIGID_ANCHORS), dtype=int)
RIGID_W = np.array([RIGID_ANCHORS[i] for i in RIGID_IDX], dtype=float)

# Mandible membership for the driven set: 1.0 = moves with the jaw, 0.0 = fixed
# to the maxilla, 0.5 = the mouth corners, which sit on the hinge line between
# them. Needed to build the jaw-open nuisance direction analytically.
_MANDIBLE_FULL = {146, 91, 181, 84, 17, 314, 405, 321, 375,
                  95, 88, 178, 87, 14, 317, 402, 318, 324}
_MANDIBLE_HALF = {61, 291, 78, 308}


def mandible_weights(indices: list[int]) -> np.ndarray:
    w = np.zeros(len(indices))
    for k, i in enumerate(indices):
        if i in _MANDIBLE_FULL:
            w[k] = 1.0
        elif i in _MANDIBLE_HALF:
            w[k] = 0.5
    return w


def inner_upper_mask(indices: list[int]) -> np.ndarray:
    """Inner-lip landmarks on the upper lip (they rise when the lips part)."""
    upper = {415, 310, 311, 312, 13, 82, 81, 80, 191}
    return np.array([1.0 if i in upper else 0.0 for i in indices])


def inner_lower_mask(indices: list[int]) -> np.ndarray:
    lower = {95, 88, 178, 87, 14, 317, 402, 318, 324}
    return np.array([1.0 if i in lower else 0.0 for i in indices])


# ---------------------------------------------------------------------------
# Landmark extraction
# ---------------------------------------------------------------------------

APP_MODEL = Path(
    r"C:\Users\amuel\OneDrive\Desktop\wisc-psychology-lab"
    r"\niedenthal-ducksoup-research-video-conferencing"
    r"\renderer\public\mediapipe\face_landmarker.task"
)


class Landmarker:
    """Thin wrapper over the MediaPipe FaceLandmarker task, IMAGE mode.

    IMAGE mode is stateless. That matters: VIDEO mode reuses the previous
    frame's region of interest, so running a folder of unrelated photos through
    it would let photo N inherit photo N-1's ROI.
    """

    def __init__(self, model_path: Path = APP_MODEL, blendshapes: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not model_path.exists():
            raise FileNotFoundError(f"face_landmarker.task not found at {model_path}")

        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=blendshapes,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, bgr: np.ndarray):
        """Return (landmarks_px [N,3], blendshapes dict, pose 4x4) or None."""
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect(img)
        if not res.face_landmarks:
            return None

        h, w = bgr.shape[:2]
        # z is in roughly the same units as x (image-width-normalised depth).
        # Stored for pose simulation only -- it is NOT used to drive the morph,
        # because MediaPipe's per-landmark depth is a regressed estimate with
        # expression-correlated bias.
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]], dtype=float)

        bs = {}
        if res.face_blendshapes:
            bs = {c.category_name: c.score for c in res.face_blendshapes[0]}

        pose = None
        if res.facial_transformation_matrixes:
            pose = np.array(res.facial_transformation_matrixes[0], dtype=float).reshape(4, 4)

        return pts, bs, pose

    def close(self):
        self._lm.close()


# ---------------------------------------------------------------------------
# Canonical frame
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Frame:
    """A similarity transform from image pixels into canonical face units."""
    theta: float          # head roll, radians
    scale: float          # pixels per canonical unit
    tx: float
    ty: float
    residual: float       # weighted RMS fit residual, canonical units

    def to_canonical(self, pts_px: np.ndarray) -> np.ndarray:
        """Image pixels -> canonical units. Accepts Nx2 or Nx3 (z is dropped:
        the morph is driven by the in-plane projection only)."""
        xy = np.asarray(pts_px, dtype=float)[:, :2]
        c, s = math.cos(-self.theta), math.sin(-self.theta)
        r = np.array([[c, -s], [s, c]])
        return (xy - np.array([self.tx, self.ty])) @ r.T / self.scale

    def to_image_delta(self, delta_canon: np.ndarray) -> np.ndarray:
        """Map a canonical-space displacement back into pixels (rotation+scale,
        no translation -- it is a delta)."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        r = np.array([[c, -s], [s, c]])
        return (delta_canon @ r.T) * self.scale


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, w: np.ndarray):
    """Weighted similarity fit mapping `dst` (template) onto `src` (image).

    Closed form. Returns (theta, scale, tx, ty) such that
    src ~= scale * R(theta) @ dst + t.
    """
    w = w / w.sum()
    src_mu = (w[:, None] * src).sum(0)
    dst_mu = (w[:, None] * dst).sum(0)
    p = src - src_mu
    q = dst - dst_mu

    sxx = (w * (q[:, 0] * p[:, 0] + q[:, 1] * p[:, 1])).sum()
    sxy = (w * (q[:, 0] * p[:, 1] - q[:, 1] * p[:, 0])).sum()
    sqq = (w * (q ** 2).sum(1)).sum()

    theta = math.atan2(sxy, sxx)
    scale = math.hypot(sxx, sxy) / max(sqq, 1e-12)

    c, s = math.cos(theta), math.sin(theta)
    r = np.array([[c, -s], [s, c]])
    t = src_mu - scale * (dst_mu @ r.T)
    return theta, scale, t[0], t[1]


def fit_frame(pts_px: np.ndarray, template: np.ndarray, huber: int = 2) -> Frame:
    """Fit the canonical frame with a couple of IRLS passes.

    The reweighting keeps one bad landmark -- a half-closed eye, a detector
    glitch -- from tilting the whole frame, which would otherwise show up as a
    spurious roll and rotate the morph.
    """
    src = np.asarray(pts_px, dtype=float)[RIGID_IDX, :2]
    w = RIGID_W.copy()
    theta = scale = tx = ty = 0.0
    resid = np.zeros(len(src))

    for _ in range(max(1, huber)):
        theta, scale, tx, ty = umeyama_similarity(src, template, w)
        c, s = math.cos(theta), math.sin(theta)
        r = np.array([[c, -s], [s, c]])
        pred = scale * (template @ r.T) + np.array([tx, ty])
        resid = np.linalg.norm(src - pred, axis=1) / max(scale, 1e-9)
        k = 1.345 * max(np.median(resid), 1e-6)
        w = RIGID_W * np.minimum(1.0, k / np.maximum(resid, 1e-9))

    rms = math.sqrt(float((RIGID_W * resid ** 2).sum() / RIGID_W.sum()))
    return Frame(theta, scale, tx, ty, rms)


def build_template(all_rigid_px: list[np.ndarray], iters: int = 3) -> np.ndarray:
    """Generalized Procrustes mean of the rigid anchor configurations.

    Gives a canonical face that is the average of the corpus rather than one
    arbitrary person's -- so 'canonical units' are not secretly indexed to
    whoever happened to be first.
    """
    template = all_rigid_px[0].copy()
    template -= template.mean(0)
    template /= np.linalg.norm(template) / math.sqrt(len(template))

    for _ in range(iters):
        aligned = []
        for src in all_rigid_px:
            theta, scale, tx, ty = umeyama_similarity(src, template, RIGID_W)
            c, s = math.cos(theta), math.sin(theta)
            r = np.array([[c, -s], [s, c]])
            back = ((src - np.array([tx, ty])) @ r) / max(scale, 1e-12)
            aligned.append(back)
        template = np.mean(aligned, axis=0)
        template -= template.mean(0)
        template /= np.linalg.norm(template) / math.sqrt(len(template))

    return template


def scale_template_to_ipd(template: np.ndarray) -> np.ndarray:
    """Rescale so one canonical unit == one interpupillary distance.

    Makes every number downstream read as 'fraction of IPD', which is the unit
    the report should speak in.
    """
    order = {int(i): k for k, i in enumerate(RIGID_IDX)}
    li, ri = order[LEFT_IRIS], order[RIGHT_IRIS]
    ipd = float(np.linalg.norm(template[ri] - template[li]))
    return template / ipd


# ---------------------------------------------------------------------------
# Nuisance directions, derived from anatomy rather than estimated from data
# ---------------------------------------------------------------------------

def nuisance_basis(canon: np.ndarray, indices: list[int]) -> dict[str, np.ndarray]:
    """Displacement fields for the expression components that are NOT a smile.

    Each is a flat 2*len(indices) vector in canonical space.

    jaw_open  -- the mandible rotates about a horizontal axis through the two
                 condyles. Projected frontally that is a downward displacement
                 proportional to how far below the hinge a point sits, applied
                 only to mandibular landmarks. No free parameters.
    lip_part  -- the inner lip contours separate vertically while the outer
                 contour stays put.

    Projecting these out is what makes a toothy smile and a closed-mouth smile
    yield the same smile axis. Lip pucker is deliberately NOT included: it is
    radially inward, which is anti-parallel to the smile's radial-outward
    component, so removing it would remove real smile. Whether that is the right
    call is tested empirically in the analysis, not assumed.
    """
    mand = mandible_weights(indices)
    hinge_y = float(canon[:, 1].min()) - 0.55   # condyles sit above the mouth

    jaw = np.zeros_like(canon)
    jaw[:, 1] = mand * (canon[:, 1] - hinge_y)

    part = np.zeros_like(canon)
    part[:, 1] = -inner_upper_mask(indices) + inner_lower_mask(indices)

    out = {}
    for name, field in (("jaw_open", jaw), ("lip_part", part)):
        v = field.reshape(-1)
        n = np.linalg.norm(v)
        out[name] = v / n if n > 1e-9 else v
    return out


def project_out(y: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Remove the span of `basis` from `y` (Gram-Schmidt, numerically safe)."""
    out = y.copy()
    for b in basis:
        bb = b.copy()
        for prev in basis:
            if prev is b:
                break
            bb = bb - float(bb @ prev) * prev
        n = np.linalg.norm(bb)
        if n < 1e-9:
            continue
        bb = bb / n
        out = out - float(out @ bb) * bb
    return out


# ---------------------------------------------------------------------------
# The CURRENT algorithm, modelled analytically
# ---------------------------------------------------------------------------

SMILE_ANGLE_RAD = math.radians(25.0)
SMILE_GAIN = 0.17
YAW_FADE_START = 0.65
YAW_FADE_END = 0.35
NOSE_TIP, LEFT_FACE_EDGE, RIGHT_FACE_EDGE = 1, 234, 454


@dataclasses.dataclass
class CurrentMorph:
    """What faceMorph.ts actually delivers to a mouth corner, in pixels.

    Reimplements renderer/lib/faceMorph.ts:226-272 and :400-428 exactly, so the
    delivered displacement can be computed for thousands of faces without
    rendering a single frame.
    """
    d_px: float           # total corner travel, pixels
    dx_px: float
    dy_px: float
    mouth_width: float
    corner_w: float       # the field weight actually landing on the corner
    win: float
    u_corner: float
    v_corner: float
    yaw_scale: float
    roi_clamped: bool     # did the ROI hit a frame edge?
    peak_ratio: float     # field peak (on the cheek) / field at the lip corner


def current_morph_delivery(pts: np.ndarray, alpha: float,
                           width: int, height: int) -> CurrentMorph:
    pts = np.asarray(pts, dtype=float)[:, :2]
    lc, rc = pts[LEFT_CORNER], pts[RIGHT_CORNER]
    center = (lc + rc) / 2.0
    mouth_width = float(np.hypot(*(rc - lc)))

    nose, le, re = pts[NOSE_TIP], pts[LEFT_FACE_EDGE], pts[RIGHT_FACE_EDGE]
    dl, dr = abs(nose[0] - le[0]), abs(re[0] - nose[0])
    symmetry = min(dl, dr) / max(1e-3, max(dl, dr))
    yaw_scale = float(np.clip((symmetry - YAW_FADE_END) /
                              (YAW_FADE_START - YAW_FADE_END), 0.0, 1.0))

    lip = pts[OUTER_LIP]
    min_x, min_y = lip.min(0)
    max_x, max_y = lip.max(0)
    pad_x, pad_y = mouth_width * 0.55, mouth_width * 0.7

    roi_x = max(0.0, min_x - pad_x)
    roi_y = max(0.0, min_y - pad_y)
    roi_w = min(float(width), max_x + pad_x) - roi_x
    roi_h = min(float(height), max_y + pad_y) - roi_y
    clamped = (min_x - pad_x < 0 or min_y - pad_y < 0 or
               max_x + pad_x > width or max_y + pad_y > height)

    # Evaluate the displacement field at the right-hand mouth corner.
    sx, sy = float(rc[0]), float(rc[1])
    u = (sx - roi_x) / max(roi_w, 1e-9)
    v = (sy - roi_y) / max(roi_h, 1e-9)
    xn = (sx - center[0]) / (mouth_width / 2.0)
    sigma_y = mouth_width * 0.6
    vy = math.exp(-((sy - center[1]) ** 2) / (2 * sigma_y * sigma_y))
    win = math.sin(math.pi * np.clip(u, 0, 1)) * math.sin(math.pi * np.clip(v, 0, 1))
    corner_w = min(1.6, xn * xn) * vy * win

    strength = (alpha - 1.0) * yaw_scale
    mag = abs(strength) * mouth_width
    d = mag * SMILE_GAIN * corner_w

    # Where does the field actually peak? min(1.6, xn^2)*sin(pi*u) is maximised
    # lateral to the corner, out on the cheek -- so the corner never receives
    # the nominal gain.
    us = np.linspace(u, min(1.0, u + 0.45), 400)
    xs = roi_x + us * roi_w
    xns = (xs - center[0]) / (mouth_width / 2.0)
    field = np.minimum(1.6, xns ** 2) * np.sin(np.pi * np.clip(us, 0, 1)) * math.sin(math.pi * np.clip(v, 0, 1))
    peak_ratio = float(field.max() / corner_w) if corner_w > 1e-9 else float("nan")

    return CurrentMorph(
        d_px=d,
        dx_px=math.copysign(math.cos(SMILE_ANGLE_RAD) * d, xn),
        dy_px=-math.sin(SMILE_ANGLE_RAD) * d,
        mouth_width=mouth_width,
        corner_w=corner_w,
        win=win,
        u_corner=float(u),
        v_corner=float(v),
        yaw_scale=yaw_scale,
        roi_clamped=bool(clamped),
        peak_ratio=peak_ratio,
    )


def ipd_px(pts: np.ndarray) -> float:
    p = np.asarray(pts, dtype=float)
    return float(np.linalg.norm(p[RIGHT_IRIS, :2] - p[LEFT_IRIS, :2]))


def rotate_3d_project(pts3: np.ndarray, yaw_deg: float, pitch_deg: float,
                      roll_deg: float, focal_px: float) -> np.ndarray:
    """Rotate a landmark cloud about its own centroid and re-project.

    Used to simulate a different camera angle without needing photographs from
    that angle. For a pure camera rotation this is exact; for head rotation it
    ignores self-occlusion and shading, so claims are kept inside +/-25 degrees.
    Perspective divide uses a plausible webcam focal length so foreshortening is
    modelled rather than assumed away.
    """
    p = np.asarray(pts3, dtype=float).copy()
    centre = p.mean(0)
    q = p - centre

    ay, ax, az = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0],
                   [-math.sin(ay), 0, math.cos(ay)]])
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)],
                   [0, math.sin(ax), math.cos(ax)]])
    rz = np.array([[math.cos(az), -math.sin(az), 0],
                   [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    q = q @ (rz @ rx @ ry).T

    # Place the head at a depth consistent with the requested focal length, so
    # the projected size is unchanged at zero rotation.
    z0 = focal_px
    denom = np.maximum(z0 + q[:, 2], 1e-3)
    out = np.empty((len(q), 2))
    out[:, 0] = centre[0] + focal_px * q[:, 0] / denom
    out[:, 1] = centre[1] + focal_px * q[:, 1] / denom
    return out
