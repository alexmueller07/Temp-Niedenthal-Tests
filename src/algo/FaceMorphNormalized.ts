// The normalized smile morph.
//
// Same public surface as the production FaceMorphProcessor, so it drops back
// into the app by swapping one import. What changes is the unit the
// manipulation is expressed in.
//
// The production morph commands a displacement in mouth-widths. Measured over
// 823 faces that is extremely consistent — CV 0.4% — so the warp is not what
// makes the manipulation unequal. What makes it unequal is that people's own
// smiles differ about twofold, so the same displacement is a very different
// share of each person's expressive range: at the "strong" preset it delivers
// roughly 58% of a median person's full smile, but around 90% of a
// 10th-percentile person's and 40% of a 90th-percentile person's.
//
// So this implementation commands a *fraction of a smile* instead, and converts
// that to pixels using (a) the corpus-average smile shape, (b) this person's own
// smile size once it has been observed, and (c) the live head pose.
//
// See FINDINGS.md for the measurements behind every constant here.

import {
  ALPHA_TO_SMILE_UNITS, AMPLITUDE_STATS, CANONICAL_TEMPLATE,
} from './faceModel.gen'
import {
  DRIVEN, LEFT_CORNER, RIGHT_CORNER, RIGID_IDX, WARP_DRIVEN,
} from './constants'
import { Calibration, type CalibrationStatus } from './calibration'
import { ExpressionDetector } from './expressionDetector'
import { createLandmarker } from './landmarkerHost'
import { OneEuroLandmarks } from './oneEuro'
import { POPULATION_AXIS, readSmile } from './smileMetric'
import {
  decodePose, fitFrame, ipdPx, poseScaleFactor, toCanonical, toImageDelta,
  type Frame, type HeadPose, type Pt,
} from './procrustes'
import { warpRegion, type WarpQuality, type Vec2 } from './warp'
import { AppearanceMorph, type MorphQuality } from './appearanceMorph'
import {
  SmileBank, captureAligned, meshCanonPoints, type BankStatus,
} from './smileBank'
import type {
  ExpressionState, FaceMorphAPI, MorphSource, RealizedDose,
} from './types'

/** What "one unit of smile" means. */
export type SmileUnit =
  /** A fraction of THIS person's own full smile. Preserves individual
   *  expressive style; a reserved person gets a physically smaller morph. */
  | 'self'
  /** A fraction of the corpus-median person's smile. Objectively equal
   *  displacement for everyone regardless of how expressive they are. */
  | 'population'

export type ControlLaw =
  /** displayed = actual + delta. Preserves their own dynamics, shifted. */
  | 'additive'
  /** displayed = gain * actual. Preserves dynamics AND the zero point, so a
   *  cross-correlation analysis can see the manipulation — but the realized
   *  dose then depends on the participant's own behaviour. */
  | 'multiplicative'
  /** displayed = a fixed level. Destroys their dynamics; included so it can be
   *  measured and argued about rather than assumed away. */
  | 'endpoint'

/** How the smile is produced. */
export type MorphMode =
  /** Deform the live frame's own pixels. Moves landmarks and nothing else --
   *  no teeth, no nasolabial fold, no shading. Always available. */
  | 'geometric'
  /** Morph toward a frame of this person actually smiling, captured earlier in
   *  the session. Brings the appearance with it. Falls back to geometric until
   *  a keyframe has been captured. */
  | 'appearance'

export interface NormalizedOptions {
  mode: MorphMode
  unit: SmileUnit
  law: ControlLaw
  /** Smile units per unit of alpha. The default is set so alpha 1.9 lands at
   *  the same physical magnitude the current morph already delivers on a median
   *  face, so an A/B changes how equal the dose is and not how big. */
  alphaScale: number
  /** Hard ceiling on the displayed smile, in units of the person's own
   *  maximum. Above about 1 the face is being pushed past anything it does
   *  naturally, which is where the RAs' "uncanny" note came from. */
  maxSmileUnits: number
  /** Warp grid spacing in pixels. Smaller is smoother and slower. */
  gridPx: number
  /** Learn this person's own smile size from the raw frames. */
  calibrate: boolean
  /**
   * Run the warp even at zero displacement.
   *
   * The production code returns early when alpha is near 1, so sham frames skip
   * ~192 affine resamples that active frames go through, and the two conditions
   * differ in local image softness for reasons unrelated to the manipulation.
   * In a deception study that is a leak. Leaving this on costs the CPU but
   * makes sham and active identical apart from the intended displacement.
   */
  alwaysWarp: boolean
  /** Above this head yaw the far mouth corner is genuinely self-occluded and
   *  the warp has nothing meaningful to move. Degrees. */
  yawCutoffDeg: number
  yawFeatherDeg: number
  /** Live smile level, in the person's own units, above which a frame is worth
   *  keeping as a morph source. */
  captureThreshold: number
  /** Don't bank frames faster than this. */
  captureIntervalMs: number
}

export const DEFAULT_OPTIONS: NormalizedOptions = {
  mode: 'appearance',
  unit: 'self',
  law: 'additive',
  alphaScale: ALPHA_TO_SMILE_UNITS,
  maxSmileUnits: 1.15,
  gridPx: 6,
  calibrate: true,
  alwaysWarp: true,
  yawCutoffDeg: 42,
  yawFeatherDeg: 8,
  captureThreshold: 0.22,
  captureIntervalMs: 500,
}

const ALPHA_TWEEN_TAU_MS = 350
/** Reject frames this far off-frontal from the calibration estimate. */
const CALIB_MAX_YAW = 22
const CALIB_MAX_PITCH = 20
const CALIB_MAX_RESIDUAL = 0.035

export interface NormalizedDebug {
  frame: Frame | null
  pose: HeadPose | null
  /** The person's live smile, in units of their own full smile. */
  liveSmileUnits: number
  /** What the partner is being shown, same units. */
  displayedSmileUnits: number
  calibration: CalibrationStatus
  quality: WarpQuality | null
  /** Set when the appearance morph ran; null when it fell back to geometry. */
  morph: MorphQuality | null
  /** What the smile bank holds. */
  bank: BankStatus
  /** Which path actually produced this frame. */
  usedMode: MorphMode
  dose: RealizedDose | null
  /** Per-stage timings, milliseconds. */
  msDetect: number
  msWarp: number
}

export class FaceMorphNormalized implements FaceMorphAPI {
  private landmarker: Awaited<ReturnType<typeof createLandmarker>> | null = null
  private src: HTMLCanvasElement
  private srcCtx: CanvasRenderingContext2D

  private alphaTarget = 1
  private alphaCurrent = 1
  private lastTweenTs: number | null = null

  private template: Float64Array
  private filter = new OneEuroLandmarks()
  private calib = new Calibration()
  private detector = new ExpressionDetector()
  private bank = new SmileBank()
  private appearance = new AppearanceMorph()
  private lastCaptureTs = 0
  /** Enough of the last accepted frame to bank it on demand. */
  private lastGood: {
    frame: Frame; mesh: Float64Array; level: number; jawOpen: number
    width: number; height: number
  } | null = null

  private lastFaceFound = false
  private lastFaceTs = 0
  private debugState: NormalizedDebug

  opts: NormalizedOptions

  constructor(opts: Partial<NormalizedOptions> = {}) {
    this.opts = { ...DEFAULT_OPTIONS, ...opts }

    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d', { willReadFrequently: false })
    if (!ctx) throw new Error('2D context unavailable')
    this.src = canvas
    this.srcCtx = ctx

    const map = new Map(CANONICAL_TEMPLATE.map(([i, v]) => [i, v]))
    this.template = new Float64Array(2 * RIGID_IDX.length)
    RIGID_IDX.forEach((idx, k) => {
      const v = map.get(idx)
      if (!v) throw new Error(`canonical template is missing landmark ${idx}`)
      this.template[2 * k] = v[0]
      this.template[2 * k + 1] = v[1]
    })

    this.debugState = {
      frame: null, pose: null, liveSmileUnits: 0, displayedSmileUnits: 0,
      calibration: this.calib.status, quality: null, morph: null,
      bank: this.bank.status, usedMode: 'geometric', dose: null,
      msDetect: 0, msWarp: 0,
    }
  }

  async init(): Promise<void> {
    this.landmarker = await createLandmarker('VIDEO', true)
  }

  get ready(): boolean { return this.landmarker !== null }
  get faceFound(): boolean { return this.lastFaceFound }
  get expression(): ExpressionState | null { return this.detector.expression }
  get debug(): NormalizedDebug { return this.debugState }
  get calibration(): Calibration { return this.calib }
  get smileBank(): SmileBank { return this.bank }

  /**
   * Bank the most recent frame as a smile source, whatever its level.
   *
   * Automatic capture waits for a genuine smile, which is right for a session
   * but unreliable for a demo — you cannot ask someone to smile on cue and also
   * watch the screen. This takes whatever is on camera now, provided the face
   * is square enough and well enough tracked to be transplantable later.
   * Returns false when it is not.
   */
  captureNow(): boolean {
    const g = this.lastGood
    if (!g || g.level < 0.05) return false
    this.bank.offer({
      image: captureAligned(this.src, g.frame, g.width, g.height),
      canonPoints: g.mesh,
      level: g.level,
      jawOpen: g.jawOpen,
      capturedAt: performance.now(),
    })
    this.debugState.bank = this.bank.status
    return true
  }

  setAlpha(alpha: number): void {
    const wasNeutral = Math.abs(this.alphaTarget - 1) < 1e-6
    this.alphaTarget = alpha
    const isNeutral = Math.abs(alpha - 1) < 1e-6
    // Hold the calibrated gain still while the manipulation runs, so the dose
    // cannot drift mid-trial; adopt whatever was learned when it returns to
    // neutral.
    if (isNeutral) this.calib.thaw()
    else if (wasNeutral) this.calib.freeze()
  }

  /**
   * Forget the landmark smoothing history without forgetting the person.
   *
   * The 1-Euro filter carries state across frames, so after a discontinuity —
   * a tracking loss, a camera switch, or the demo's virtual-camera jump — it
   * spends a fraction of a second dragging the landmarks from where the face
   * used to be. Calibration and the learned amplitude survive; only the
   * smoothing is dropped.
   */
  resetTracking(): void {
    this.filter.reset()
  }

  /** Drop everything learned about the current person. Used between faces in a
   *  batch run so one identity cannot leak into the next. */
  reset(): void {
    this.filter.reset()
    this.calib.reset()
    this.bank.clear()
    this.lastCaptureTs = 0
    this.lastGood = null
    this.detector = new ExpressionDetector()
    this.alphaCurrent = this.alphaTarget
    this.lastTweenTs = null
  }

  close(): void {
    this.landmarker?.landmarker.close()
    this.landmarker = null
  }

  render(
    source: MorphSource,
    dstCtx: CanvasRenderingContext2D,
    width: number,
    height: number,
    tsMs: number,
  ): boolean {
    if (this.src.width !== width || this.src.height !== height) {
      this.src.width = width
      this.src.height = height
    }
    this.srcCtx.drawImage(source as CanvasImageSource, 0, 0, width, height)
    dstCtx.setTransform(1, 0, 0, 1, 0, 0)
    dstCtx.drawImage(this.src, 0, 0, width, height)

    const dt = this.lastTweenTs === null ? 16 : Math.min(100, tsMs - this.lastTweenTs)
    this.lastTweenTs = tsMs
    const k = 1 - Math.exp(-dt / ALPHA_TWEEN_TAU_MS)
    this.alphaCurrent += (this.alphaTarget - this.alphaCurrent) * k
    if (Math.abs(this.alphaCurrent - this.alphaTarget) < 0.004) {
      this.alphaCurrent = this.alphaTarget
    }

    if (!this.landmarker) {
      this.lastFaceFound = false
      return false
    }

    const t0 = performance.now()
    let result
    try {
      result = this.landmarker.landmarker.detectForVideo(
        source as HTMLVideoElement, tsMs)
    } catch {
      return false
    }
    this.debugState.msDetect = performance.now() - t0

    const faces = result?.faceLandmarks
    if (!faces || faces.length === 0) {
      this.lastFaceFound = false
      this.filter.reset()
      this.calib.interrupt()
      if (tsMs - this.lastFaceTs > 1000) this.detector.update(null, tsMs)
      this.debugState.dose = null
      return false
    }
    this.lastFaceFound = true
    this.lastFaceTs = tsMs

    // Expression is always read from the RAW frame. The rules engine downstream
    // depends on it being the participant's genuine expression.
    const shapes = result.faceBlendshapes?.[0]?.categories ?? null
    this.detector.update(shapes, tsMs)
    const jawOpen = shapes?.find((c) => c.categoryName === 'jawOpen')?.score ?? 0

    // --- geometry -------------------------------------------------------
    const raw = faces[0]
    const flatPx = new Float64Array(2 * raw.length)
    for (let i = 0; i < raw.length; i++) {
      flatPx[2 * i] = raw[i].x * width
      flatPx[2 * i + 1] = raw[i].y * height
    }
    const smoothed = this.filter.filter(flatPx, tsMs)
    const lm: Pt[] = new Array(raw.length)
    for (let i = 0; i < raw.length; i++) {
      lm[i] = { x: smoothed[2 * i], y: smoothed[2 * i + 1] }
    }

    const frame = fitFrame(lm, this.template)
    const pose = decodePose(result.facialTransformationMatrixes?.[0]?.data ?? null)
    const ipd = ipdPx(lm)
    this.debugState.frame = frame
    this.debugState.pose = pose

    const canonMouth: Pt[] = DRIVEN.map((i) => toCanonical(frame, lm[i]))

    // --- calibration ----------------------------------------------------
    const poseOk = !pose
      || (Math.abs(pose.yawDeg) < CALIB_MAX_YAW
        && Math.abs(pose.pitchDeg) < CALIB_MAX_PITCH)
    if (this.opts.calibrate && poseOk && frame.residual < CALIB_MAX_RESIDUAL) {
      this.calib.update(canonMouth, tsMs)
    } else {
      this.calib.interrupt()
    }
    const amplitude = Math.max(1e-3, this.calib.amplitude)
    this.debugState.calibration = this.calib.status

    // --- control --------------------------------------------------------
    // Live smile, as a fraction of this person's own full smile.
    const liveUnits = readSmile(canonMouth, this.calib.restShape).projection / amplitude
    const commanded = (this.alphaCurrent - 1) * this.opts.alphaScale

    let targetUnits: number
    switch (this.opts.law) {
      case 'multiplicative':
        targetUnits = (1 + commanded) * Math.max(0, liveUnits)
        break
      case 'endpoint':
        targetUnits = commanded
        break
      default:
        targetUnits = liveUnits + commanded
    }
    const lo = -this.opts.maxSmileUnits
    const hi = this.opts.maxSmileUnits
    const clampedTarget = Math.min(hi, Math.max(lo, targetUnits))
    const clamped = clampedTarget !== targetUnits

    // Pose gating: a smile happens on the surface of the face, so its image
    // projection SHOULD foreshorten as the head turns. That is handled below by
    // the projection. The gate here is only about the far corner becoming
    // genuinely invisible, which no projection can fix.
    const yaw = pose ? Math.abs(pose.yawDeg) : 0
    const poseGain = yaw <= this.opts.yawCutoffDeg - this.opts.yawFeatherDeg
      ? 1
      : yaw >= this.opts.yawCutoffDeg
        ? 0
        : 0.5 * (1 + Math.cos(Math.PI
          * (yaw - (this.opts.yawCutoffDeg - this.opts.yawFeatherDeg))
          / this.opts.yawFeatherDeg))

    const deltaUnits = (clampedTarget - liveUnits) * poseGain
    const scaleAmplitude = this.opts.unit === 'population'
      ? AMPLITUDE_STATS.p50
      : amplitude

    this.debugState.liveSmileUnits = liveUnits
    this.debugState.displayedSmileUnits = liveUnits + deltaUnits

    // --- displacement ----------------------------------------------------
    // A smile happens on the surface of the face, so its image projection
    // should foreshorten as the head turns — that is what a real smile does.
    // But the fitted frame scale has already absorbed part of that, because the
    // rigid anchors foreshorten too; dividing by poseScaleFactor removes the
    // double count so pose enters exactly once.
    const poseScale = poseScaleFactor(this.template, pose)
    const cosYaw = (pose ? Math.cos(pose.yawDeg * Math.PI / 180) : 1) / poseScale
    const cosPitch = (pose ? Math.cos(pose.pitchDeg * Math.PI / 180) : 1) / poseScale

    // Index of each warp-driven landmark inside the measurement set, so the
    // axis is read from the right slot.
    const srcPts: Vec2[] = new Array(WARP_DRIVEN.length)
    const dstPts: Vec2[] = new Array(WARP_DRIVEN.length)
    let cornerTravelPx = 0
    for (let w = 0; w < WARP_DRIVEN.length; w++) {
      const idx = WARP_DRIVEN[w]
      const k = DRIVEN.indexOf(idx)
      const p = lm[idx]
      // Canonical displacement along the corpus smile axis...
      const dCanon = {
        x: deltaUnits * scaleAmplitude * POPULATION_AXIS[2 * k],
        y: deltaUnits * scaleAmplitude * POPULATION_AXIS[2 * k + 1],
      }
      // ...foreshortened by the head's own orientation, so a turned face gets
      // a correctly-projected smile rather than a screen-space one...
      const dFace = { x: dCanon.x * cosYaw, y: dCanon.y * cosPitch }
      // ...then rotated and scaled into image pixels. Because this goes through
      // the canonical frame, head roll rotates the displacement with the face
      // instead of shearing it.
      const dImg = toImageDelta(frame, dFace)
      srcPts[w] = { x: p.x, y: p.y }
      dstPts[w] = { x: p.x + dImg.x, y: p.y + dImg.y }
      if (idx === LEFT_CORNER || idx === RIGHT_CORNER) {
        cornerTravelPx += Math.hypot(dImg.x, dImg.y) / 2
      }
    }

    // --- bank this person's own smile, when they produce one -------------
    // Only the raw frame is ever banked, never the morphed output, and only
    // when the head is square enough that the stored expression can be
    // transplanted onto a different pose later without a visible pose jump.
    const canonMesh = meshCanonPoints((i) => toCanonical(frame, lm[i]))
    this.lastGood = (poseOk && frame.residual < CALIB_MAX_RESIDUAL)
      ? { frame, mesh: canonMesh, level: liveUnits, jawOpen, width, height }
      : null
    if (
      this.opts.mode === 'appearance'
      && poseOk && frame.residual < CALIB_MAX_RESIDUAL
      && liveUnits >= this.opts.captureThreshold
      && this.calib.atSmilePeak(canonMouth)
      && tsMs - this.lastCaptureTs >= this.opts.captureIntervalMs
    ) {
      this.lastCaptureTs = tsMs
      this.bank.offer({
        image: captureAligned(this.src, frame, width, height),
        canonPoints: canonMesh,
        level: liveUnits,
        jawOpen,
        capturedAt: tsMs,
      })
    }
    this.debugState.bank = this.bank.status

    // --- morph toward that smile, if there is one to morph toward ---------
    // The fraction is worked out in expression units: to display `target` when
    // the face is at `liveUnits` and the source frame is at the keyframe's
    // level, go that proportion of the way. If the person is already smiling
    // harder than anything banked there is nothing to borrow, so it falls
    // through to the geometric path.
    if (this.opts.mode === 'appearance' && deltaUnits > 0) {
      const kf = this.bank.pick(clampedTarget, jawOpen)
      const span = kf ? kf.level - liveUnits : 0
      if (kf && span > 0.05) {
        const tMorph = Math.min(1, deltaUnits / span)
        const t1 = performance.now()
        const mq = this.appearance.render(
          dstCtx, this.src, width, height, frame, canonMesh, kf, tMorph, ipd)
        this.debugState.msWarp = performance.now() - t1
        this.debugState.morph = mq
        this.debugState.usedMode = 'appearance'
        this.debugState.quality = null
        this.debugState.dose = this.makeDose(
          commanded, deltaUnits, cornerTravelPx / Math.max(ipd, 1e-6),
          poseGain, clamped, pose)
        return true
      }
    }
    this.debugState.morph = null
    this.debugState.usedMode = 'geometric'

    const nearlyZero = Math.abs(deltaUnits) * scaleAmplitude < 1e-4
    if (nearlyZero && !this.opts.alwaysWarp) {
      this.debugState.dose = this.makeDose(commanded, deltaUnits, 0, poseGain,
        clamped, pose)
      return false
    }

    // --- warp -------------------------------------------------------------
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (let k = 0; k < srcPts.length; k++) {
      minX = Math.min(minX, srcPts[k].x, dstPts[k].x)
      minY = Math.min(minY, srcPts[k].y, dstPts[k].y)
      maxX = Math.max(maxX, srcPts[k].x, dstPts[k].x)
      maxY = Math.max(maxY, srcPts[k].y, dstPts[k].y)
    }
    // Pad in interpupillary units, not in mouth-widths: IPD is rigid and does
    // not change when the person talks, so the support region is stationary.
    const pad = ipd * 0.55
    const roi = {
      x: Math.max(0, minX - pad),
      y: Math.max(0, minY - pad),
      w: 0,
      h: 0,
    }
    roi.w = Math.min(width, maxX + pad) - roi.x
    roi.h = Math.min(height, maxY + pad) - roi.y
    if (roi.w < 8 || roi.h < 8) return false

    const t1 = performance.now()
    const quality = warpRegion(dstCtx, this.src, { src: srcPts, dst: dstPts },
      roi, this.opts.gridPx, width, height)
    this.debugState.msWarp = performance.now() - t1
    this.debugState.quality = quality
    this.debugState.dose = this.makeDose(commanded, deltaUnits,
      cornerTravelPx / Math.max(ipd, 1e-6), poseGain, clamped, pose)

    return !nearlyZero
  }

  private makeDose(
    commanded: number, realized: number, travelPerIpd: number,
    poseGain: number, clamped: boolean, pose: HeadPose | null,
  ): RealizedDose {
    return {
      commanded,
      realized,
      travelPerIpd,
      poseGain,
      clamped,
      yawDeg: pose?.yawDeg ?? 0,
      pitchDeg: pose?.pitchDeg ?? 0,
      rollDeg: pose?.rollDeg ?? 0,
      faceFound: this.lastFaceFound,
    }
  }
}

/** Compile-time proof that the new class still satisfies the production
 *  contract. Type-level only -- instantiating here would touch the DOM at
 *  import time. If this stops compiling, the drop-in port has broken. */
type _SatisfiesApi = FaceMorphNormalized extends FaceMorphAPI ? true : never
const _apiCheck: _SatisfiesApi = true
void _apiCheck
