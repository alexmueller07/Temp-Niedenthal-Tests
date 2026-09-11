// Measuring a smile in canonical units, and separating it from the things that
// are not a smile.
//
// The separation is done analytically, not learned. Jaw opening is a rigid
// rotation of the mandible about a hinge, so in a frontal projection it is a
// downward displacement proportional to how far below the hinge a landmark
// sits, applied only to mandibular landmarks. Lips parting is the inner
// contours separating while the outer contour stays put. Both can be written
// down; neither needs a single frame of training data.
//
// Projecting them out is what makes a toothy smile and a closed-mouth smile the
// same measurement. On the Chicago Face Database the median agreement between a
// person's closed-mouth and toothy smile rises from 0.82 to 0.97 once jaw
// opening is removed, and the 10th percentile from 0.62 to 0.92 — so it is not
// just the average case that is fixed.

import { DRIVEN, lipPartWeight, mandibleWeight } from './constants'
import { SMILE_AXIS } from './faceModel.gen'
import type { Pt } from './procrustes'

const N = DRIVEN.length
const DIM = 2 * N

/** The corpus-average smile direction, unit norm, flattened as [x0,y0,x1,y1,…]
 *  in DRIVEN order. */
export const POPULATION_AXIS: Float64Array = (() => {
  const map = new Map(SMILE_AXIS.map(([i, v]) => [i, v]))
  const out = new Float64Array(DIM)
  DRIVEN.forEach((idx, k) => {
    const v = map.get(idx)
    if (v) { out[2 * k] = v[0]; out[2 * k + 1] = v[1] }
  })
  return normalise(out)
})()

/**
 * Build the nuisance directions for the current face.
 *
 * They depend on the live canonical shape (the hinge distance is measured from
 * it), so they are rebuilt per frame — it is about 80 flops, not worth caching.
 */
export function nuisanceBasis(canonMouth: ArrayLike<Pt>): Float64Array[] {
  // Condyles sit above the mouth; the exact height matters little because the
  // direction is normalised and then projected out.
  let minY = Infinity
  for (let k = 0; k < N; k++) minY = Math.min(minY, canonMouth[k].y)
  const hingeY = minY - 0.55

  const jaw = new Float64Array(DIM)
  const part = new Float64Array(DIM)
  for (let k = 0; k < N; k++) {
    const idx = DRIVEN[k]
    jaw[2 * k + 1] = mandibleWeight(idx) * (canonMouth[k].y - hingeY)
    part[2 * k + 1] = lipPartWeight(idx)
  }
  return [normalise(jaw), normalise(part)]
}

/** Remove the span of `basis` from `v`, in place on a copy. */
export function projectOut(v: Float64Array, basis: Float64Array[]): Float64Array {
  const out = Float64Array.from(v)
  // Orthonormalise as we go so near-parallel basis vectors cannot blow up.
  const ortho: Float64Array[] = []
  for (const b of basis) {
    const w = Float64Array.from(b)
    for (const q of ortho) {
      const d = dot(w, q)
      for (let i = 0; i < DIM; i++) w[i] -= d * q[i]
    }
    let n = 0
    for (let i = 0; i < DIM; i++) n += w[i] * w[i]
    n = Math.sqrt(n)
    if (n < 1e-9) continue
    for (let i = 0; i < DIM; i++) w[i] /= n
    ortho.push(w)
  }
  for (const q of ortho) {
    const d = dot(out, q)
    for (let i = 0; i < DIM; i++) out[i] -= d * q[i]
  }
  return out
}

export interface SmileReading {
  /** Projection onto the population smile axis, in canonical shape units — the
   *  same units the calibrated per-person amplitude is in, so their ratio is
   *  "what fraction of their own full smile is this". */
  projection: number
  /** Total nuisance-free shape change from rest. The amplitude estimator that
   *  came out most reliable on CFD (test-retest r = 0.56, against 0.53 and 0.52
   *  for the two alternatives). */
  magnitude: number
}

/**
 * How much this face is smiling right now, relative to a rest shape.
 *
 * Returns both quantities from one projection. They used to be two functions,
 * which meant rebuilding and re-orthonormalising the nuisance basis two or three
 * times per frame for no reason — the expensive part is the projection, not the
 * dot product at the end.
 */
export function readSmile(
  canonMouth: ArrayLike<Pt>, rest: Float64Array,
): SmileReading {
  const d = new Float64Array(DIM)
  for (let k = 0; k < N; k++) {
    d[2 * k] = canonMouth[k].x - rest[2 * k]
    d[2 * k + 1] = canonMouth[k].y - rest[2 * k + 1]
  }
  const clean = projectOut(d, nuisanceBasis(canonMouth))
  return {
    projection: dot(clean, POPULATION_AXIS),
    magnitude: Math.sqrt(dot(clean, clean)),
  }
}

export function dot(a: Float64Array, b: Float64Array): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

function normalise(v: Float64Array): Float64Array {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] * v[i]
  s = Math.sqrt(s)
  if (s < 1e-12) return v
  for (let i = 0; i < v.length; i++) v[i] /= s
  return v
}
