// MediaPipe FaceMesh landmark indices (478-point refined model, which is what
// face_landmarker.task emits — the extra 10 points are the two irises).
//
// Kept in one place because the choice of which landmarks count as "rigid" is
// the single most load-bearing decision in the whole normalization, and it is
// easy to get subtly wrong.

/** Outer lip ring. Same list and winding as LIP_INDICES in the production
 *  faceMorph.ts, so the frozen baseline and the new implementation measure the
 *  same mouth. */
export const OUTER_LIP = [
  61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
] as const

/** Inner lip ring, same winding: left corner, along the lower lip, right
 *  corner, back along the upper lip. Included in the driven set so the oral
 *  aperture translates as a unit instead of being stretched — warping an open
 *  mouth without these smears the teeth and the dark interior. */
export const INNER_LIP = [
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
  308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
] as const

/** Everything the morph MEASURES: rest shape, smile projection, amplitude.
 *  Includes the inner ring because the aperture carries real information about
 *  what the mouth is doing. */
export const DRIVEN: readonly number[] = [...OUTER_LIP, ...INNER_LIP]

/**
 * What the warp is allowed to DRIVE, which is a smaller set.
 *
 * The inner-lip contours nearly coincide when the mouth is closed — opposing
 * upper and lower landmarks come within about 0.002 interpupillary units, a
 * fifth of a pixel at normal webcam scale. Giving two coincident points
 * different displacements asks the warp to pull them apart, which showed up as
 * local stretch of up to 23x and visible smearing across the vermilion border.
 * Driving from the outer ring alone leaves the interior to the warp's own
 * smoothness, which is well behaved. Mozza drives from the outer ring only for
 * the same reason.
 */
export const WARP_DRIVEN: readonly number[] = [...OUTER_LIP]

export const LEFT_CORNER = 61
export const RIGHT_CORNER = 291

/** Iris centres. Present only in the refined model; they give a true
 *  interpupillary distance, which is the most stable scale reference available
 *  on a face — rigid, expression-invariant, and unaffected by facial hair. */
export const LEFT_IRIS = 468
export const RIGHT_IRIS = 473

/**
 * Rigid anchors for the canonical frame fit, with weights.
 *
 * "Rigid" means bone. Eye corners and iris centres are the most reliable; the
 * nose bridge adds vertical extent so the fit is not degenerate along y; the
 * nose tip is down-weighted because the nostrils flare during AU9 and, a little,
 * during a strong AU12.
 *
 * Deliberately absent:
 *   - every mouth landmark — that is the thing being measured;
 *   - the jaw and chin — the mandible moves whenever the person talks;
 *   - the face-oval points 234 / 454 that the production code uses for its yaw
 *     estimate. Those are the *silhouette*, not the skull: they slide across the
 *     face as the head turns, so anchoring to them makes the frame rotate with
 *     the very motion it is supposed to cancel.
 */
export const RIGID_ANCHORS: ReadonlyArray<readonly [index: number, weight: number]> = [
  [33, 1.0], [133, 1.0], [362, 1.0], [263, 1.0],
  [468, 1.2], [473, 1.2],
  [168, 0.9], [6, 0.9], [197, 0.7], [195, 0.6],
  [8, 0.6],
  [1, 0.3],
]

export const RIGID_IDX: readonly number[] = RIGID_ANCHORS.map(([i]) => i)
export const RIGID_W: readonly number[] = RIGID_ANCHORS.map(([, w]) => w)

/** Mandible membership for the driven set: 1 moves with the jaw, 0 is fixed to
 *  the maxilla, 0.5 for the mouth corners which sit on the boundary. Used to
 *  build the jaw-open nuisance direction analytically rather than learning it. */
const MANDIBLE_FULL = new Set([
  146, 91, 181, 84, 17, 314, 405, 321, 375,
  95, 88, 178, 87, 14, 317, 402, 318, 324,
])
const MANDIBLE_HALF = new Set([61, 291, 78, 308])

export function mandibleWeight(landmarkIndex: number): number {
  if (MANDIBLE_FULL.has(landmarkIndex)) return 1
  if (MANDIBLE_HALF.has(landmarkIndex)) return 0.5
  return 0
}

const INNER_UPPER = new Set([415, 310, 311, 312, 13, 82, 81, 80, 191])
const INNER_LOWER = new Set([95, 88, 178, 87, 14, 317, 402, 318, 324])

/** +1 for inner-lower, -1 for inner-upper, 0 elsewhere: the direction the lips
 *  separate when they part, independent of any smile. */
export function lipPartWeight(landmarkIndex: number): number {
  if (INNER_UPPER.has(landmarkIndex)) return -1
  if (INNER_LOWER.has(landmarkIndex)) return 1
  return 0
}
