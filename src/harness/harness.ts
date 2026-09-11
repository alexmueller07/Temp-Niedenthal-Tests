// Batch measurement harness.
//
// Runs both morph implementations over a corpus and reports what each one
// actually did to the rendered pixels. Driven from Python via Playwright, but
// everything numeric happens here so the algorithm under test is the algorithm
// that ships — no second implementation to drift out of sync.
//
// Two design points that matter for the numbers being trustworthy:
//
//   Frames are injected as decoded ImageBitmaps on a synthetic clock, not
//   played through a <video> element. That makes a run frame-exact and
//   bit-reproducible; with real playback the timestamps, and therefore
//   MediaPipe's frame-to-frame tracking, differ on every run and a 5% change in
//   a dispersion statistic could be the scheduler rather than the algorithm.
//
//   The probe that scores the output is a SEPARATE landmarker in IMAGE mode.
//   The morph processors use VIDEO mode, which carries a region of interest
//   between calls; scoring morphed frames through that same instance would both
//   corrupt the tracker and contaminate the expression read, which the contract
//   says must come from the raw frame only.

import { FaceMorphCurrent } from '../algo/FaceMorphCurrent'
import { FaceMorphNormalized, type NormalizedOptions } from '../algo/FaceMorphNormalized'
import type { MorphSource } from '../algo/types'
import { DRIVEN, LEFT_CORNER, OUTER_LIP, RIGHT_CORNER, RIGID_IDX } from '../algo/constants'
import { CANONICAL_TEMPLATE } from '../algo/faceModel.gen'
import { createLandmarker } from '../algo/landmarkerHost'
import { fitFrame, ipdPx, toCanonical, type Pt } from '../algo/procrustes'
import { POPULATION_AXIS, projectOut, nuisanceBasis } from '../algo/smileMetric'
import { ViewTransform, IDENTITY_VIEW, isIdentity, type ViewParams } from '../algo/viewTransform'
import { mlsRigid } from '../algo/warp'

const FRAME_W = 1280
const FRAME_H = 720
/** Frames fed per cell so the alpha tween (tau = 350 ms) reaches steady state.
 *  The tween is part of the system under test, so both implementations get it. */
const SETTLE_FRAMES = 18
const SETTLE_DT_MS = 100

export interface Measures {
  /** Mouth-corner positions in canonical units. Differenced against the sham
   *  render these give the corner displacement the warp actually achieved, as
   *  re-measured from the pixels — which is the number to compare against what
   *  the implementation says it commanded. */
  lcX: number
  lcY: number
  rcX: number
  rcY: number
  /** Mouth-corner height relative to the lip-line, in interpupillary units.
   *  Model-free given the landmarks: no learned expression model involved. */
  cornerLift: number
  /** Projection onto the corpus smile axis. Reported for completeness, and
   *  flagged: the normalized implementation is steered by this quantity, so it
   *  must NOT be used to judge which implementation wins. */
  axisProj: number
  /** MediaPipe's own smile blendshape — a different read of the same pixels,
   *  sensitive to appearance rather than only to landmark positions. */
  blendSmile: number
  /** Left/right asymmetry of the corner lift. Head roll makes the production
   *  morph manufacture this. */
  cornerAsymmetry: number
  /** Head pose as re-measured on the output. */
  rollDeg: number
  yawDeg: number
  ipdPx: number
  faceFound: boolean
}

export interface Row {
  id: string
  expr: string
  group: string
  impl: 'current' | 'normalized'
  alpha: number
  view: string
  /** Measures on the morphed output minus the same measures on the sham render
   *  of the identical frame. Frame-aligned differencing cancels the probe's
   *  per-identity bias to first order. */
  d: Measures
  /** Raw measures on the morphed output, for diagnostics. */
  raw: Measures
  msDetect: number
  msWarp: number
  /** Mean mouth-corner displacement between the sham and morphed renders,
   *  re-measured from the pixels, in interpupillary units. This is the primary
   *  outcome: it is what the warp actually achieved, not what it intended. */
  measuredTravel: number
  /** Which path produced the frame, and what the bank held. */
  usedMode?: string
  bankCount?: number
  /** Warp quality, normalized implementation only. */
  maxExpansion?: number
  minExpansion?: number
  folded?: boolean
  /** base64 PNG of the rendered output, when exportPngs is set. */
  png?: string
  shamPng?: string
  /** What the implementation says it applied. */
  realizedUnits?: number
  travelPerIpd?: number
  poseGain?: number
  amplitudeUsed?: number
}

export interface RunSpec {
  /** `seedUrl`, when given, is fed to the processor before the test frame so
   *  the appearance morph has one of this person's own smiles banked. In a
   *  session that happens by itself; over still photographs it has to be
   *  arranged, because a single neutral portrait never contains a smile. */
  frames: Array<{ id: string; expr: string; group: string; url: string;
                  seedUrl?: string }>
  alphas: number[]
  views: Array<{ name: string; params: Partial<ViewParams> }>
  impls: Array<'current' | 'normalized'>
  options?: Partial<NormalizedOptions>
  /** Per-identity smile amplitude, to simulate a converged calibration. */
  amplitudes?: Record<string, number>
  /** Per-identity resting mouth shape, flattened canonical [x,y,...] over the
   *  measurement set, for the same reason. */
  rests?: Record<string, number[]>
  /** Return the rendered frames as base64 PNG so they can be scored offline by
   *  an instrument that shares nothing with the one steering the morph. Heavy,
   *  so it is normally off and used on a subset. */
  exportPngs?: boolean
}

class Harness {
  private probe!: Awaited<ReturnType<typeof createLandmarker>>
  private current!: FaceMorphCurrent
  private normalized!: FaceMorphNormalized
  private view!: ViewTransform
  private out!: HTMLCanvasElement
  private outCtx!: CanvasRenderingContext2D
  private template!: Float64Array
  private clock = 1000
  info: Record<string, unknown> = {}

  async init(options: Partial<NormalizedOptions> = {}): Promise<void> {
    this.out = document.createElement('canvas')
    this.out.width = FRAME_W
    this.out.height = FRAME_H
    const ctx = this.out.getContext('2d', { willReadFrequently: false })
    if (!ctx) throw new Error('2D context unavailable')
    this.outCtx = ctx

    this.view = new ViewTransform()
    this.current = new FaceMorphCurrent()
    this.normalized = new FaceMorphNormalized(options)
    await this.current.init()
    await this.normalized.init()
    this.probe = await createLandmarker('IMAGE', true)

    const map = new Map(CANONICAL_TEMPLATE.map(([i, v]) => [i, v]))
    this.template = new Float64Array(2 * RIGID_IDX.length)
    RIGID_IDX.forEach((idx, k) => {
      const v = map.get(idx)!
      this.template[2 * k] = v[0]
      this.template[2 * k + 1] = v[1]
    })

    this.info = { delegate: this.probe.delegate, assets: this.probe.assets }
  }

  /**
   * Check the warp moves image content the way it was asked to.
   *
   * Worth its own test because MLS gives a *forward* map and rendering needs the
   * *backward* one, so swapping the two arguments produces a warp that moves the
   * face the opposite way — subtle enough to survive a casual look at a face,
   * and fatal to every number downstream.
   */
  selfTest(): { warpDirectionOk: boolean; observedShiftPx: number } {
    const n = 64
    const p = new Float64Array([32, 32])
    const q = new Float64Array([42, 32])       // ask content at x=32 to appear at x=42
    // Backward map: domain = destination, range = source.
    const nodes = new Float64Array([42, 32])
    mlsRigid(nodes, q, p, 1)
    void n
    const shift = nodes[0] - 42
    return { warpDirectionOk: Math.abs(shift + 10) < 1e-6, observedShiftPx: shift }
  }

  /** The current output canvas as a base64 PNG (no data: prefix). */
  private snapshot(): string {
    return this.out.toDataURL('image/png').split(',', 2)[1]
  }

  private measure(canvas: HTMLCanvasElement): Measures {
    const empty: Measures = {
      lcX: NaN, lcY: NaN, rcX: NaN, rcY: NaN,
      cornerLift: NaN, axisProj: NaN, blendSmile: NaN, cornerAsymmetry: NaN,
      rollDeg: NaN, yawDeg: NaN, ipdPx: NaN, faceFound: false,
    }
    const res = this.probe.landmarker.detect(canvas)
    if (!res.faceLandmarks?.length) return empty

    const raw = res.faceLandmarks[0]
    const lm: Pt[] = raw.map((p) => ({ x: p.x * FRAME_W, y: p.y * FRAME_H }))
    const frame = fitFrame(lm, this.template)
    const canonMouth = DRIVEN.map((i) => toCanonical(frame, lm[i]))

    // Model-free geometric smile: how high the mouth corners sit relative to the
    // rest of the lip line, in the face's own frame.
    const outer = OUTER_LIP.map((i) => toCanonical(frame, lm[i]))
    let medY = 0
    const ys = outer.map((p) => p.y).sort((a, b) => a - b)
    medY = ys[ys.length >> 1]
    const lcP = toCanonical(frame, lm[LEFT_CORNER])
    const rcP = toCanonical(frame, lm[RIGHT_CORNER])
    const lcY = lcP.y
    const rcY = rcP.y
    const cornerLift = medY - (lcY + rcY) / 2

    // Axis projection, nuisance-free. Uses the CURRENT shape as its own
    // reference, so it is a shape descriptor rather than a change measure; the
    // differencing against sham turns it into a change.
    const flat = new Float64Array(2 * DRIVEN.length)
    for (let k = 0; k < DRIVEN.length; k++) {
      flat[2 * k] = canonMouth[k].x
      flat[2 * k + 1] = canonMouth[k].y
    }
    const clean = projectOut(flat, nuisanceBasis(canonMouth))
    let axisProj = 0
    for (let i = 0; i < flat.length; i++) axisProj += clean[i] * POPULATION_AXIS[i]

    const bs = res.faceBlendshapes?.[0]?.categories ?? []
    const get = (n: string) => bs.find((c) => c.categoryName === n)?.score ?? 0
    const blendSmile = (get('mouthSmileLeft') + get('mouthSmileRight')) / 2

    const m = res.facialTransformationMatrixes?.[0]?.data ?? null
    let yawDeg = NaN
    if (m && m.length >= 16) {
      const sy = Math.hypot(m[0], m[4])
      yawDeg = Math.atan2(-m[8], sy) * 180 / Math.PI
    }

    return {
      lcX: lcP.x, lcY: lcP.y, rcX: rcP.x, rcY: rcP.y,
      cornerLift,
      axisProj,
      blendSmile,
      cornerAsymmetry: lcY - rcY,
      rollDeg: frame.theta * 180 / Math.PI,
      yawDeg,
      ipdPx: ipdPx(lm),
      faceFound: true,
    }
  }

  /** Feed a smiling frame through the processor so it lands in the bank.
   *  Rendered to a scratch canvas: nothing about the seed should reach the
   *  output being measured. */
  private seedBank(seed: ImageBitmap, view: ViewParams): void {
    const src: CanvasImageSource = isIdentity(view)
      ? seed : this.view.apply(seed, FRAME_W, FRAME_H, view)
    const scratch = document.createElement('canvas')
    scratch.width = FRAME_W
    scratch.height = FRAME_H
    const sctx = scratch.getContext('2d')!
    this.normalized.setAlpha(1)
    for (let i = 0; i < 4; i++) {
      this.clock += 600
      this.normalized.render(src as MorphSource, sctx, FRAME_W, FRAME_H, this.clock)
    }
    // Forced, not automatic. Automatic capture waits for a smile *event*, which
    // needs the person's resting mouth to be known — and a run over still
    // photographs never establishes one, because there is no stretch of frames
    // where they are not smiling.
    if (!this.normalized.captureNow()) {
      console.warn('seed frame was not usable as a smile source')
    }
  }

  private renderCell(
    impl: 'current' | 'normalized', src: CanvasImageSource, alpha: number,
  ): { measures: Measures; msDetect: number; msWarp: number; extra: Partial<Row> } {
    const proc = impl === 'current' ? this.current : this.normalized
    proc.setAlpha(alpha)

    let msDetect = 0
    let msWarp = 0
    for (let i = 0; i < SETTLE_FRAMES; i++) {
      this.clock += SETTLE_DT_MS
      const t0 = performance.now()
      proc.render(src as HTMLVideoElement, this.outCtx, FRAME_W, FRAME_H, this.clock)
      const dt = performance.now() - t0
      if (i === SETTLE_FRAMES - 1) {
        if (impl === 'normalized') {
          msDetect = this.normalized.debug.msDetect
          msWarp = this.normalized.debug.msWarp
        } else {
          msDetect = dt
        }
      }
    }

    const extra: Partial<Row> = {}
    if (impl === 'normalized') {
      const d = this.normalized.debug
      extra.maxExpansion = d.quality?.maxExpansion
      extra.minExpansion = d.quality?.minExpansion
      extra.folded = d.quality?.folded
      extra.realizedUnits = d.dose?.realized
      extra.travelPerIpd = d.dose?.travelPerIpd
      extra.poseGain = d.dose?.poseGain
      extra.amplitudeUsed = d.calibration.amplitude
      extra.usedMode = d.usedMode
      extra.bankCount = d.bank.count
    }
    return { measures: this.measure(this.out), msDetect, msWarp, extra }
  }

  async run(spec: RunSpec, onProgress?: (done: number, total: number) => void): Promise<Row[]> {
    const rows: Row[] = []
    const views = spec.views.length ? spec.views : [{ name: 'base', params: {} }]
    const total = spec.frames.length * views.length * spec.impls.length
      * spec.alphas.length
    let done = 0

    for (const f of spec.frames) {
      const bmp = await loadBitmap(f.url)
      const seed = f.seedUrl ? await loadBitmap(f.seedUrl) : null
      for (const v of views) {
        const params: ViewParams = { ...IDENTITY_VIEW, ...v.params }
        const src: CanvasImageSource = isIdentity(params)
          ? bmp
          : this.view.apply(bmp, FRAME_W, FRAME_H, params)

        for (const impl of spec.impls) {
          if (impl === 'normalized') {
            this.normalized.reset()
            const a = spec.amplitudes?.[f.id]
            this.normalized.calibration.setOverride(a ?? null)
            const rest = spec.rests?.[f.id]
            if (rest) this.normalized.calibration.setRestOverride(rest)
            if (seed) this.seedBank(seed, params)
          }
          // Sham on the identical frame, for within-frame differencing.
          const sham = this.renderCell(impl, src, 1.0)
          const shamPng = spec.exportPngs ? this.snapshot() : undefined

          for (const alpha of spec.alphas) {
            const cell = this.renderCell(impl, src, alpha)
            const png = spec.exportPngs ? this.snapshot() : undefined
            const d = diff(cell.measures, sham.measures)
            rows.push({
              id: f.id, expr: f.expr, group: f.group, impl, alpha, view: v.name,
              measuredTravel:
                (Math.hypot(d.lcX, d.lcY) + Math.hypot(d.rcX, d.rcY)) / 2,
              d,
              raw: cell.measures,
              msDetect: cell.msDetect,
              msWarp: cell.msWarp,
              ...(png ? { png, shamPng } : {}),
              ...cell.extra,
            })
            done++
            onProgress?.(done, total)
          }
        }
      }
      bmp.close()
      seed?.close()
    }
    return rows
  }
}

function diff(a: Measures, b: Measures): Measures {
  return {
    lcX: a.lcX - b.lcX, lcY: a.lcY - b.lcY,
    rcX: a.rcX - b.rcX, rcY: a.rcY - b.rcY,
    cornerLift: a.cornerLift - b.cornerLift,
    axisProj: a.axisProj - b.axisProj,
    blendSmile: a.blendSmile - b.blendSmile,
    cornerAsymmetry: a.cornerAsymmetry - b.cornerAsymmetry,
    rollDeg: a.rollDeg - b.rollDeg,
    yawDeg: a.yawDeg - b.yawDeg,
    ipdPx: a.ipdPx - b.ipdPx,
    faceFound: a.faceFound && b.faceFound,
  }
}

async function loadBitmap(url: string): Promise<ImageBitmap> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`)
  return createImageBitmap(await res.blob())
}

declare global {
  interface Window {
    __harness: {
      init(options?: Partial<NormalizedOptions>): Promise<Record<string, unknown>>
      selfTest(): { warpDirectionOk: boolean; observedShiftPx: number }
      run(spec: RunSpec): Promise<Row[]>
      progress: { done: number; total: number }
    }
  }
}

const instance = new Harness()
window.__harness = {
  async init(options = {}) {
    await instance.init(options)
    return instance.info
  },
  selfTest: () => instance.selfTest(),
  run: (spec: RunSpec) => instance.run(spec, (done, total) => {
    window.__harness.progress = { done, total }
  }),
  progress: { done: 0, total: 0 },
}
