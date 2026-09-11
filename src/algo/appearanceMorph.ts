// Morphing the live frame toward a stored frame of the person's own smile.
//
// Both images are warped onto one intermediate geometry and cross-dissolved.
// That is the textbook image morph, and the reason it beats deforming the live
// frame alone is that the dissolve is what carries appearance: teeth, the
// nasolabial fold, the raised cheek, the way light falls differently on a
// smiling face. None of that can be produced by moving pixels that are already
// there, which is why the geometric warp reads as a stretch.
//
// Everything happens inside a region around the lower face, and the blend is
// feathered to nothing at its edge, so the rest of the frame is untouched.

import { MESH_MOVING_COUNT, MESH_POINTS, MESH_TRIANGLES } from './morphMesh.gen'
import { CROP_PX, canonToCrop, type Keyframe } from './smileBank'
import type { Frame, Pt } from './procrustes'

export interface MorphQuality {
  /** Triangles that came out degenerate and were skipped. */
  skipped: number
  /** Luminance correction applied to the keyframe, as a multiplier. */
  exposure: number
  /** How far the source keyframe's head was from the live one, canonical units.
   *  Large values mean the transplant is being asked to cross a pose gap. */
  poseGap: number
  /** Where the time went, milliseconds. */
  msWarpLive: number
  msWarpSmile: number
  msExposure: number
  msComposite: number
}

interface Scratch {
  canvas: HTMLCanvasElement
  ctx: CanvasRenderingContext2D
}

function scratch(): Scratch {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d', { willReadFrequently: false })
  if (!ctx) throw new Error('2D context unavailable')
  return { canvas, ctx }
}

export class AppearanceMorph {
  private live = scratch()
  private smile = scratch()
  private mask = scratch()
  private roi = { x: 0, y: 0, w: 0, h: 0 }
  /** Exposure is re-measured rarely and reused between times: reading pixels
   *  back forces the GPU to finish and hand over, which costs more than every
   *  triangle in the warp put together. Lighting does not change in a tenth of
   *  a second, so measuring at that rate is plenty. */
  private cachedGain = 1
  private lastExposureTs = 0
  private exposureIntervalMs = 400

  /**
   * Render the morph into `dstCtx`, which must already hold the unmorphed frame.
   *
   * `t` is how far toward the keyframe to go, 0 to 1. `liveMesh` and the
   * keyframe's stored points are both in canonical units; `frame` maps canonical
   * back to this frame's pixels.
   */
  render(
    dstCtx: CanvasRenderingContext2D,
    source: CanvasImageSource,
    width: number,
    height: number,
    frame: Frame,
    liveMesh: Float64Array,
    kf: Keyframe,
    t: number,
    ipdPx: number,
  ): MorphQuality {
    const n = MESH_POINTS.length
    const c = Math.cos(frame.theta), s = Math.sin(frame.theta)
    const toImage = (cx: number, cy: number): Pt => ({
      x: (c * cx - s * cy) * frame.scale + frame.tx,
      y: (s * cx + c * cy) * frame.scale + frame.ty,
    })

    // Target geometry: moving points travel t of the way from where this face
    // is now to where it is in the keyframe; anchors stay put, so the head does
    // not move, only the expression.
    const liveImg = new Float64Array(2 * n)
    const target = new Float64Array(2 * n)
    const smileSrc = new Float64Array(2 * n)
    let poseGap = 0
    for (let k = 0; k < n; k++) {
      const lx = liveMesh[2 * k], ly = liveMesh[2 * k + 1]
      const kx = kf.canonPoints[2 * k], ky = kf.canonPoints[2 * k + 1]
      const moving = k < MESH_MOVING_COUNT
      const tx = moving ? lx + t * (kx - lx) : lx
      const ty = moving ? ly + t * (ky - ly) : ly

      const li = toImage(lx, ly)
      liveImg[2 * k] = li.x; liveImg[2 * k + 1] = li.y
      const ti = toImage(tx, ty)
      target[2 * k] = ti.x; target[2 * k + 1] = ti.y
      const sc = canonToCrop(kx, ky)
      smileSrc[2 * k] = sc.x; smileSrc[2 * k + 1] = sc.y

      if (!moving) poseGap += Math.hypot(kx - lx, ky - ly)
    }
    poseGap /= Math.max(1, n - MESH_MOVING_COUNT)

    // Region of interest: everything the mesh touches, plus a margin for the
    // feather. Sized from the interpupillary distance so it does not depend on
    // any expression-sensitive measurement.
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (let k = 0; k < n; k++) {
      minX = Math.min(minX, liveImg[2 * k], target[2 * k])
      maxX = Math.max(maxX, liveImg[2 * k], target[2 * k])
      minY = Math.min(minY, liveImg[2 * k + 1], target[2 * k + 1])
      maxY = Math.max(maxY, liveImg[2 * k + 1], target[2 * k + 1])
    }
    const pad = ipdPx * 0.2
    const roi = {
      x: Math.max(0, Math.floor(minX - pad)),
      y: Math.max(0, Math.floor(minY - pad)),
      w: 0, h: 0,
    }
    roi.w = Math.min(width, Math.ceil(maxX + pad)) - roi.x
    roi.h = Math.min(height, Math.ceil(maxY + pad)) - roi.y
    if (roi.w < 16 || roi.h < 16) {
      return { skipped: 0, exposure: 1, poseGap, msWarpLive: 0, msWarpSmile: 0,
               msExposure: 0, msComposite: 0 }
    }
    this.roi = roi
    this.size(roi.w, roi.h)

    // Warp each image onto the shared target geometry. Same triangle list for
    // both, which is the whole reason the mesh is a generated constant: if the
    // topologies differed the two halves of the dissolve would not correspond.
    const tA = performance.now()
    const skippedA = this.warp(this.live.ctx, source, width, height, liveImg, target, roi)
    const tB = performance.now()
    const skippedB = this.warp(this.smile.ctx, kf.image, CROP_PX, CROP_PX,
                               smileSrc, target, roi)
    const tC = performance.now()

    const exposure = this.matchExposure(t, tC)
    const tD = performance.now()
    this.buildMask(target, roi, ipdPx)

    // Feather the smile layer, then lay it over the warped live layer.
    this.smile.ctx.save()
    this.smile.ctx.globalCompositeOperation = 'destination-in'
    this.smile.ctx.setTransform(1, 0, 0, 1, 0, 0)
    this.smile.ctx.drawImage(this.mask.canvas, 0, 0)
    this.smile.ctx.restore()

    dstCtx.save()
    dstCtx.setTransform(1, 0, 0, 1, 0, 0)
    dstCtx.drawImage(this.live.canvas, roi.x, roi.y)
    dstCtx.globalAlpha = Math.max(0, Math.min(1, t))
    dstCtx.drawImage(this.smile.canvas, roi.x, roi.y)
    dstCtx.restore()

    return {
      skipped: skippedA + skippedB, exposure, poseGap,
      msWarpLive: tB - tA, msWarpSmile: tC - tB,
      msExposure: tD - tC, msComposite: performance.now() - tD,
    }
  }

  private size(w: number, h: number): void {
    for (const s of [this.live, this.smile, this.mask]) {
      if (s.canvas.width !== w || s.canvas.height !== h) {
        s.canvas.width = w
        s.canvas.height = h
      }
      s.ctx.setTransform(1, 0, 0, 1, 0, 0)
      s.ctx.clearRect(0, 0, w, h)
    }
  }

  /** Piecewise-affine warp of one image onto the target mesh, into a scratch
   *  canvas whose origin is the region's top-left. */
  private warp(
    ctx: CanvasRenderingContext2D, img: CanvasImageSource,
    imgW: number, imgH: number,
    src: Float64Array, dst: Float64Array,
    roi: { x: number; y: number; w: number; h: number },
  ): number {
    let skipped = 0
    for (const [ia, ib, ic] of MESH_TRIANGLES) {
      const s0x = src[2 * ia], s0y = src[2 * ia + 1]
      const s1x = src[2 * ib], s1y = src[2 * ib + 1]
      const s2x = src[2 * ic], s2y = src[2 * ic + 1]
      const d0x = dst[2 * ia] - roi.x, d0y = dst[2 * ia + 1] - roi.y
      const d1x = dst[2 * ib] - roi.x, d1y = dst[2 * ib + 1] - roi.y
      const d2x = dst[2 * ic] - roi.x, d2y = dst[2 * ic + 1] - roi.y

      const denom = s0x * (s2y - s1y) - s1x * s2y + s2x * s1y + (s1x - s2x) * s0y
      if (Math.abs(denom) < 1e-6) { skipped++; continue }

      ctx.save()
      // Grow the destination triangle slightly so neighbours overlap; without
      // it, antialiasing leaves a hairline seam along every shared edge.
      const cx = (d0x + d1x + d2x) / 3, cy = (d0y + d1y + d2y) / 3
      const g = 0.6
      ctx.beginPath()
      ctx.moveTo(d0x + Math.sign(d0x - cx) * g, d0y + Math.sign(d0y - cy) * g)
      ctx.lineTo(d1x + Math.sign(d1x - cx) * g, d1y + Math.sign(d1y - cy) * g)
      ctx.lineTo(d2x + Math.sign(d2x - cx) * g, d2y + Math.sign(d2y - cy) * g)
      ctx.closePath()
      ctx.clip()

      const m11 = -(s0y * (d2x - d1x) - s1y * d2x + s2y * d1x + (s1y - s2y) * d0x) / denom
      const m12 = (s1y * d2y + s0y * (d1y - d2y) - s2y * d1y + (s2y - s1y) * d0y) / denom
      const m21 = (s0x * (d2x - d1x) - s1x * d2x + s2x * d1x + (s1x - s2x) * d0x) / denom
      const m22 = -(s1x * d2y + s0x * (d1y - d2y) - s2x * d1y + (s2x - s1x) * d0y) / denom
      const dx = (s0x * (s2y * d1x - s1y * d2x) + s0y * (s1x * d2x - s2x * d1x)
        + (s2x * s1y - s1x * s2y) * d0x) / denom
      const dy = (s0x * (s2y * d1y - s1y * d2y) + s0y * (s1x * d2y - s2x * d1y)
        + (s2x * s1y - s1x * s2y) * d0y) / denom

      ctx.setTransform(m11, m12, m21, m22, dx, dy)
      ctx.drawImage(img, 0, 0, imgW, imgH)
      ctx.restore()
    }
    return skipped
  }

  /**
   * Nudge the keyframe's brightness toward the live frame's.
   *
   * A keyframe captured minutes ago under different light will otherwise show
   * up as a patch of the wrong exposure, which is far more noticeable than a
   * slightly wrong smile. Only a gain is applied, not a full colour transform:
   * the two images are the same face under the same lamp, so the difference is
   * mostly level, and a heavier correction would start inventing colour.
   */
  private matchExposure(t: number, nowMs: number): number {
    if (t < 0.02) return 1
    if (nowMs - this.lastExposureTs < this.exposureIntervalMs) {
      return this.applyGain(this.cachedGain)
    }
    this.lastExposureTs = nowMs
    const w = Math.max(8, Math.floor(this.roi.w / 8))
    const h = Math.max(8, Math.floor(this.roi.h / 8))
    let a = 0, b = 0
    try {
      const la = this.live.ctx.getImageData(0, 0, w, h).data
      const lb = this.smile.ctx.getImageData(0, 0, w, h).data
      let na = 0, nb = 0
      for (let i = 0; i < la.length; i += 4) {
        if (la[i + 3] > 8) { a += la[i] + la[i + 1] + la[i + 2]; na++ }
        if (lb[i + 3] > 8) { b += lb[i] + lb[i + 1] + lb[i + 2]; nb++ }
      }
      if (na < 16 || nb < 16) return 1
      a /= na; b /= nb
    } catch {
      return 1
    }
    if (b < 1) return 1
    this.cachedGain = Math.max(0.85, Math.min(1.18, a / b))
    return this.applyGain(this.cachedGain)
  }

  private applyGain(gain: number): number {
    if (Math.abs(gain - 1) < 0.01) return 1
    this.smile.ctx.save()
    this.smile.ctx.globalCompositeOperation = gain > 1 ? 'lighter' : 'multiply'
    this.smile.ctx.globalAlpha = Math.min(1, Math.abs(gain - 1) * 3)
    this.smile.ctx.fillStyle = gain > 1 ? '#202020' : '#d8d8d8'
    this.smile.ctx.fillRect(0, 0, this.roi.w, this.roi.h)
    this.smile.ctx.restore()
    return gain
  }

  /**
   * Where the transplant is allowed to show, feathered to nothing at the edge.
   *
   * An ellipse over the lower face rather than the mesh's own outline: a hard
   * edge following the triangles would trace a visible seam, and the mesh
   * boundary is not where a smile stops mattering anyway.
   */
  private buildMask(
    target: Float64Array,
    roi: { x: number; y: number; w: number; h: number },
    ipdPx: number,
  ): void {
    const ctx = this.mask.ctx
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, roi.w, roi.h)

    let cx = 0, cy = 0
    for (let k = 0; k < MESH_MOVING_COUNT; k++) {
      cx += target[2 * k]; cy += target[2 * k + 1]
    }
    cx = cx / MESH_MOVING_COUNT - roi.x
    cy = cy / MESH_MOVING_COUNT - roi.y

    const rx = ipdPx * 1.05
    const ry = ipdPx * 0.95
    ctx.save()
    ctx.translate(cx, cy)
    ctx.scale(1, ry / rx)
    const grad = ctx.createRadialGradient(0, 0, rx * 0.55, 0, 0, rx)
    grad.addColorStop(0, 'rgba(255,255,255,1)')
    grad.addColorStop(0.72, 'rgba(255,255,255,0.92)')
    grad.addColorStop(1, 'rgba(255,255,255,0)')
    ctx.fillStyle = grad
    ctx.beginPath()
    ctx.arc(0, 0, rx, 0, Math.PI * 2)
    ctx.fill()
    ctx.restore()
  }
}
