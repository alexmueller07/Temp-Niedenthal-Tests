// 1-Euro filter (Casiez, Roussel & Vogel 2012) over a landmark array.
//
// The production morph has no landmark filtering at all, and it scales every
// displacement by a *live* measurement (mouth width) — so detector jitter is
// multiplied straight into the morph amplitude and shows up as a shimmer at
// noise bandwidth. A shimmer is much more likely to give the manipulation away
// than a slightly wrong displacement, because it reads as a video artifact
// rather than as a facial expression.
//
// 1-Euro is the right filter here rather than a plain low-pass: its cutoff
// rises with speed, so it suppresses jitter while the face is still and gets
// out of the way during fast motion instead of adding lag.

export interface Pt2 { x: number; y: number }

export class OneEuroLandmarks {
  private prev: Float64Array | null = null
  private prevDx: Float64Array | null = null
  private lastTs = 0

  constructor(
    /** Cutoff at zero speed, Hz. Lower = steadier at rest, more lag. */
    private minCutoff = 1.0,
    /** How fast the cutoff opens up with speed. */
    private beta = 0.007,
    /** Cutoff for the speed estimate itself, Hz. */
    private dCutoff = 1.0,
  ) {}

  reset() {
    this.prev = null
    this.prevDx = null
  }

  /** Filter in place-ish: returns a stable Float64Array of [x,y,...]. */
  filter(flat: Float64Array, tsMs: number): Float64Array {
    const n = flat.length
    if (!this.prev || this.prev.length !== n) {
      this.prev = Float64Array.from(flat)
      this.prevDx = new Float64Array(n)
      this.lastTs = tsMs
      return this.prev
    }

    const dt = Math.max(1e-3, (tsMs - this.lastTs) / 1000)
    this.lastTs = tsMs
    const aD = smoothingFactor(dt, this.dCutoff)
    const prev = this.prev
    const prevDx = this.prevDx!

    for (let i = 0; i < n; i++) {
      const dx = (flat[i] - prev[i]) / dt
      const dxHat = prevDx[i] + aD * (dx - prevDx[i])
      prevDx[i] = dxHat
      const cutoff = this.minCutoff + this.beta * Math.abs(dxHat)
      const a = smoothingFactor(dt, cutoff)
      prev[i] = prev[i] + a * (flat[i] - prev[i])
    }
    return prev
  }
}

function smoothingFactor(dt: number, cutoff: number): number {
  const r = 2 * Math.PI * cutoff * dt
  return r / (r + 1)
}
