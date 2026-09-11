// A store of this person's own smiles, captured from the live camera.
//
// The geometric warp moves landmarks and nothing else, which is why it reads as
// a stretch rather than an expression: no teeth, no nasolabial fold, no cheek
// shading. The fix is not a better warp, it is a better *source* — if we have a
// frame of this person actually smiling, we can morph toward it and every one of
// those cues comes along, because it is their real face.
//
// This is what Face2Face (Thies et al. 2016) does with its mouth database, and
// it is the same construction psychology already uses for expression continua:
// an intermediate frame is "this person, t of the way to their own smile".
//
// Frames are stored canonically aligned — warped into a fixed square where the
// face is upright, centred and at a known scale. That does two things: it makes
// keyframes from different moments directly comparable, and it means the only
// transform left at render time is canonical-to-live-pose, so pose differences
// between capture and playback are handled once, here, rather than per frame.

import { MESH_POINTS } from './morphMesh.gen'
import type { Frame, Pt } from './procrustes'

/** Side of the canonical crop, in pixels. Big enough that the mouth is not
 *  resampled up at playback; small enough that a handful fit in memory. */
export const CROP_PX = 384
/** Canonical half-extent the crop covers, in interpupillary units. About 1.5
 *  covers the whole face with margin. */
export const CROP_UNITS = 1.5

const K = CROP_PX / (2 * CROP_UNITS)

export interface Keyframe {
  /** Canonically aligned image of the face at capture time. */
  image: HTMLCanvasElement
  /** Mesh-point positions in canonical units when it was captured. */
  canonPoints: Float64Array
  /** How big this smile was, in the person's own units. */
  level: number
  /** Jaw opening at capture. A keyframe with the mouth open is a poor source
   *  for a live frame with it closed, and vice versa. */
  jawOpen: number
  capturedAt: number
}

/** Canonical coordinates -> pixel coordinates inside the crop. */
export function canonToCrop(x: number, y: number): Pt {
  return { x: (x + CROP_UNITS) * K, y: (y + CROP_UNITS) * K }
}

/**
 * Draw a live frame into canonical alignment.
 *
 * The transform is the inverse of the fitted frame, composed with the crop
 * scaling — so whatever the head was doing, the stored image comes out upright,
 * centred and at a fixed size.
 */
export function captureAligned(
  source: CanvasImageSource, frame: Frame, width: number, height: number,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = CROP_PX
  canvas.height = CROP_PX
  const ctx = canvas.getContext('2d')!
  const s = K / frame.scale
  const c = Math.cos(-frame.theta), sn = Math.sin(-frame.theta)
  // crop = s * R(-theta) * (p - t) + K * CROP_UNITS
  const a = s * c, b = s * sn, cc = -s * sn, d = s * c
  const e = -(a * frame.tx + cc * frame.ty) + K * CROP_UNITS
  const f = -(b * frame.tx + d * frame.ty) + K * CROP_UNITS
  ctx.setTransform(a, b, cc, d, e, f)
  ctx.drawImage(source, 0, 0, width, height)
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  return canvas
}

export interface BankStatus {
  count: number
  /** Smile levels held, smallest first. */
  levels: number[]
  /** The biggest smile captured so far, in the person's own units. */
  best: number
}

/**
 * Keeps a small spread of this person's smiles.
 *
 * Spread rather than "the biggest", because morphing all the way to a broad
 * open smile and then dissolving only 20% of the way there is not the same
 * image as a real 20% smile — the teeth would be faintly there at every
 * intensity. Holding several levels lets the renderer pick a source close to
 * what it was asked for and dissolve most of the way to it.
 */
export class SmileBank {
  private frames: Keyframe[] = []

  constructor(
    /** How many to keep. Each is a CROP_PX square canvas. */
    private capacity = 5,
    /** Two keyframes closer than this in level are near-duplicates; keep one. */
    private minLevelGap = 0.12,
  ) {}

  get size(): number { return this.frames.length }

  get status(): BankStatus {
    const levels = this.frames.map((f) => f.level).sort((a, b) => a - b)
    return { count: this.frames.length, levels, best: levels[levels.length - 1] ?? 0 }
  }

  clear(): void { this.frames = [] }

  /**
   * Offer a frame. Kept only if it adds something the bank does not have.
   *
   * The caller is responsible for offering only frames worth keeping — near
   * frontal, well tracked, at the peak of a genuine smile.
   */
  offer(kf: Keyframe): boolean {
    const near = this.frames.find((f) => Math.abs(f.level - kf.level) < this.minLevelGap)
    if (near) {
      // Same level already held: keep whichever is the cleaner source, which
      // here means the one nearer a closed mouth for its level, since an open
      // mouth constrains what it can be morphed to.
      if (kf.jawOpen < near.jawOpen) {
        this.frames[this.frames.indexOf(near)] = kf
        return true
      }
      return false
    }
    this.frames.push(kf)
    if (this.frames.length > this.capacity) {
      // Drop whichever is most redundant: the one closest in level to another.
      let worst = 0, worstGap = Infinity
      for (let i = 0; i < this.frames.length; i++) {
        let gap = Infinity
        for (let j = 0; j < this.frames.length; j++) {
          if (i !== j) gap = Math.min(gap, Math.abs(this.frames[i].level - this.frames[j].level))
        }
        if (gap < worstGap) { worstGap = gap; worst = i }
      }
      this.frames.splice(worst, 1)
    }
    return true
  }

  /**
   * The best source for a requested smile level.
   *
   * Prefers the smallest keyframe that still reaches the target, so the
   * dissolve runs most of the way to it rather than a little of the way to a
   * much bigger smile. Ties break toward a mouth opening close to the live one.
   */
  pick(target: number, liveJawOpen: number): Keyframe | null {
    if (!this.frames.length) return null
    const reaching = this.frames.filter((f) => f.level >= target * 0.9)
    const pool = reaching.length ? reaching : this.frames
    let best = pool[0]
    let bestCost = Infinity
    for (const f of pool) {
      const levelCost = Math.abs(f.level - target)
      const jawCost = Math.abs(f.jawOpen - liveJawOpen) * 0.5
      const cost = levelCost + jawCost
      if (cost < bestCost) { bestCost = cost; best = f }
    }
    return best
  }
}

/** Mesh-point positions in canonical units, from a canonicalised landmark set. */
export function meshCanonPoints(canonAll: (i: number) => Pt): Float64Array {
  const out = new Float64Array(2 * MESH_POINTS.length)
  MESH_POINTS.forEach(([idx], k) => {
    const p = canonAll(idx)
    out[2 * k] = p.x
    out[2 * k + 1] = p.y
  })
  return out
}
