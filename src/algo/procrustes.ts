// The canonical face frame.
//
// Everything the normalized morph does happens in canonical units, where one
// unit is one interpupillary distance and the face is upright and centred.
// A displacement defined there is, by construction, independent of how far the
// person sits from the camera, how their head is tilted, and where in the frame
// they happen to be. The production morph works in screen pixels scaled by the
// live mouth width, which is why none of those hold for it today.

import {
  LEFT_IRIS, RIGHT_IRIS, RIGID_IDX, RIGID_W,
} from './constants'

export interface Pt { x: number; y: number }

/** A similarity transform from image pixels into canonical face units. */
export interface Frame {
  /** Head roll, radians. */
  theta: number
  /** Pixels per canonical unit (one interpupillary distance). */
  scale: number
  tx: number
  ty: number
  /** Weighted RMS fit residual in canonical units. High means the landmarks
   *  disagree with a rigid face — occlusion, a tracking glitch, or a face that
   *  is too far off-frontal to trust. */
  residual: number
}

/**
 * Weighted similarity fit: find theta, s, t minimising
 *   sum_i w_i * || src_i - (s * R(theta) * dst_i + t) ||^2
 * Closed form (Umeyama), no iteration, ~100 flops.
 */
export function umeyamaSimilarity(
  src: Float64Array, dst: Float64Array, w: Float64Array, n: number,
): { theta: number; scale: number; tx: number; ty: number } {
  let wsum = 0
  for (let i = 0; i < n; i++) wsum += w[i]
  if (wsum < 1e-12) return { theta: 0, scale: 1, tx: 0, ty: 0 }

  let sxm = 0, sym = 0, dxm = 0, dym = 0
  for (let i = 0; i < n; i++) {
    const wi = w[i] / wsum
    sxm += wi * src[2 * i]; sym += wi * src[2 * i + 1]
    dxm += wi * dst[2 * i]; dym += wi * dst[2 * i + 1]
  }

  let sxx = 0, sxy = 0, sqq = 0
  for (let i = 0; i < n; i++) {
    const wi = w[i] / wsum
    const px = src[2 * i] - sxm, py = src[2 * i + 1] - sym
    const qx = dst[2 * i] - dxm, qy = dst[2 * i + 1] - dym
    sxx += wi * (qx * px + qy * py)
    sxy += wi * (qx * py - qy * px)
    sqq += wi * (qx * qx + qy * qy)
  }

  const theta = Math.atan2(sxy, sxx)
  const scale = Math.hypot(sxx, sxy) / Math.max(sqq, 1e-12)
  const c = Math.cos(theta), s = Math.sin(theta)
  return {
    theta,
    scale,
    tx: sxm - scale * (c * dxm - s * dym),
    ty: sym - scale * (s * dxm + c * dym),
  }
}

/**
 * Fit the canonical frame to a detected face, with two IRLS reweighting passes.
 *
 * The reweighting matters more than it looks: without it one bad landmark — a
 * half-closed eye, a detector glitch on a frame with motion blur — tilts the
 * whole frame, which shows up downstream as a spurious head roll and rotates
 * the morph. Huber weights bound any single point's influence.
 */
export function fitFrame(
  landmarks: ArrayLike<Pt>, template: Float64Array, passes = 2,
): Frame {
  const n = RIGID_IDX.length
  const src = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    const p = landmarks[RIGID_IDX[i]]
    src[2 * i] = p.x
    src[2 * i + 1] = p.y
  }

  const w = Float64Array.from(RIGID_W)
  const resid = new Float64Array(n)
  let fit = { theta: 0, scale: 1, tx: 0, ty: 0 }

  for (let pass = 0; pass < Math.max(1, passes); pass++) {
    fit = umeyamaSimilarity(src, template, w, n)
    const c = Math.cos(fit.theta), s = Math.sin(fit.theta)
    for (let i = 0; i < n; i++) {
      const px = fit.scale * (c * template[2 * i] - s * template[2 * i + 1]) + fit.tx
      const py = fit.scale * (s * template[2 * i] + c * template[2 * i + 1]) + fit.ty
      resid[i] = Math.hypot(src[2 * i] - px, src[2 * i + 1] - py) / Math.max(fit.scale, 1e-9)
    }
    const k = 1.345 * Math.max(median(resid), 1e-6)
    for (let i = 0; i < n; i++) {
      w[i] = RIGID_W[i] * Math.min(1, k / Math.max(resid[i], 1e-9))
    }
  }

  let num = 0, den = 0
  for (let i = 0; i < n; i++) {
    num += RIGID_W[i] * resid[i] * resid[i]
    den += RIGID_W[i]
  }
  return { ...fit, residual: Math.sqrt(num / Math.max(den, 1e-12)) }
}

/** Image pixels -> canonical units. */
export function toCanonical(frame: Frame, p: Pt): Pt {
  const c = Math.cos(-frame.theta), s = Math.sin(-frame.theta)
  const dx = (p.x - frame.tx) / frame.scale
  const dy = (p.y - frame.ty) / frame.scale
  return { x: c * dx - s * dy, y: s * dx + c * dy }
}

/** A canonical-space *displacement* back to pixels: rotation and scale only,
 *  no translation, because a delta has no origin. This is the step that makes
 *  the morph travel along the face's own axes instead of the screen's — the
 *  production code applies its 25-degree smile vector in image coordinates, so
 *  a tilted head gets one corner sliding along the lip line and the other
 *  lifting across it. */
export function toImageDelta(frame: Frame, d: Pt): Pt {
  const c = Math.cos(frame.theta), s = Math.sin(frame.theta)
  return {
    x: (c * d.x - s * d.y) * frame.scale,
    y: (s * d.x + c * d.y) * frame.scale,
  }
}

/** Interpupillary distance in pixels, straight from the iris centres. */
export function ipdPx(landmarks: ArrayLike<Pt>): number {
  const l = landmarks[LEFT_IRIS], r = landmarks[RIGHT_IRIS]
  return Math.hypot(r.x - l.x, r.y - l.y)
}

// ---------------------------------------------------------------------------
// Head pose from MediaPipe's facial transformation matrix
// ---------------------------------------------------------------------------

export interface HeadPose {
  yawDeg: number
  pitchDeg: number
  rollDeg: number
  /** Upper-left 3x3 of the canonical->detected matrix, normalised to a pure
   *  rotation. Used to project canonical displacements through the real head
   *  orientation. */
  r3: Float64Array
}

/**
 * Decode the 4x4 `facialTransformationMatrixes` entry.
 *
 * This is the piece the production code does not enable at all — it estimates
 * yaw from the ratio of two screen-x distances, which roll contaminates, and
 * then uses that ratio to fade the morph to *zero*. Head turns correlate with
 * turn-taking in a conversation, so that fade silently confounds the dose with
 * who is speaking.
 */
export function decodePose(matrix: number[] | Float32Array | null): HeadPose | null {
  if (!matrix || matrix.length < 16) return null

  // Row-major 4x4; strip the per-axis scale so what remains is a rotation.
  const m = [
    [matrix[0], matrix[1], matrix[2]],
    [matrix[4], matrix[5], matrix[6]],
    [matrix[8], matrix[9], matrix[10]],
  ]
  const col = (j: number) => Math.hypot(m[0][j], m[1][j], m[2][j]) || 1
  const s0 = col(0), s1 = col(1), s2 = col(2)
  const r3 = new Float64Array([
    m[0][0] / s0, m[0][1] / s1, m[0][2] / s2,
    m[1][0] / s0, m[1][1] / s1, m[1][2] / s2,
    m[2][0] / s0, m[2][1] / s1, m[2][2] / s2,
  ])

  const sy = Math.hypot(r3[0], r3[3])
  const deg = 180 / Math.PI
  if (sy > 1e-6) {
    return {
      pitchDeg: Math.atan2(r3[7], r3[8]) * deg,
      yawDeg: Math.atan2(-r3[6], sy) * deg,
      rollDeg: Math.atan2(r3[3], r3[0]) * deg,
      r3,
    }
  }
  return {
    pitchDeg: Math.atan2(-r3[5], r3[4]) * deg,
    yawDeg: Math.atan2(-r3[6], sy) * deg,
    rollDeg: 0,
    r3,
  }
}

/**
 * How much the rigid anchor set itself shrinks under a given head rotation.
 *
 * This exists to stop a double count. The frame's `scale` comes from fitting an
 * isotropic similarity to the anchors, and those anchors foreshorten when the
 * head turns — so `scale` already carries some of the pose. Applying an explicit
 * cos(yaw) foreshortening to the displacement on top of that would shrink the
 * morph twice.
 *
 * So: measure what the fit would report for a face at this pose showing no
 * expression at all, and divide it out. What is left is a pose-free scale, and
 * the explicit foreshortening applied to the displacement is then the only place
 * pose enters.
 *
 * The anchors are treated as planar. They are not exactly — the nose bridge has
 * depth — which is why the nose tip carries the lowest weight in the rigid set.
 */
export function poseScaleFactor(template: Float64Array, pose: HeadPose | null): number {
  if (!pose) return 1
  const r = pose.r3
  const n = template.length >> 1

  let mx = 0, my = 0
  for (let i = 0; i < n; i++) { mx += template[2 * i]; my += template[2 * i + 1] }
  mx /= n; my /= n

  // Rotate the (planar) template and re-project, then read off the isotropic
  // scale a Procrustes fit would recover.
  let sxx = 0, sxy = 0, sqq = 0
  let rmx = 0, rmy = 0
  const rot = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    const x = template[2 * i] - mx, y = template[2 * i + 1] - my
    const px = r[0] * x + r[1] * y
    const py = r[3] * x + r[4] * y
    rot[2 * i] = px; rot[2 * i + 1] = py
    rmx += px; rmy += py
  }
  rmx /= n; rmy /= n
  for (let i = 0; i < n; i++) {
    const qx = template[2 * i] - mx, qy = template[2 * i + 1] - my
    const px = rot[2 * i] - rmx, py = rot[2 * i + 1] - rmy
    sxx += qx * px + qy * py
    sxy += qx * py - qy * px
    sqq += qx * qx + qy * qy
  }
  if (sqq < 1e-12) return 1
  const f = Math.hypot(sxx, sxy) / sqq
  // Guard: past a steep angle the planar approximation stops meaning anything,
  // and an unbounded division would blow the morph up.
  return Math.min(1, Math.max(0.6, f))
}

/**
 * Project a canonical on-face displacement out through the head rotation.
 *
 * A smile happens on the surface of the face, so under yaw or pitch its image
 * projection is foreshortened — and by different amounts on the near and far
 * side. Applying the same screen-space vector regardless, as the production
 * code does, over-delivers on one side and under-delivers on the other, which
 * manufactures exactly the left/right asymmetry the app's own classifier reads
 * as a dominance smile.
 *
 * Note this uses only the head *rotation*. MediaPipe's per-landmark z is a
 * regressed depth with expression-correlated bias; using it would inject noise
 * into the very quantity we are trying to hold steady.
 */
export function projectThroughPose(d: Pt, pose: HeadPose | null): Pt {
  if (!pose) return d
  const r = pose.r3
  // Canonical tangent plane is the face's own xy; take the x and y rows of R3.
  return {
    x: r[0] * d.x + r[1] * d.y,
    y: r[3] * d.x + r[4] * d.y,
  }
}

// ---------------------------------------------------------------------------

function median(a: Float64Array): number {
  const b = Array.from(a).sort((x, y) => x - y)
  const h = b.length >> 1
  return b.length % 2 ? b[h] : (b[h - 1] + b[h]) / 2
}
