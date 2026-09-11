// The live demo.
//
// Both implementations run on byte-identical input and the same commanded
// intensity, so the only thing that differs is the algorithm. A separate
// detector scores each rendered output against a sham render of the same frame,
// which is what the strip chart shows — the manipulation as delivered, not as
// requested.
//
// The virtual camera controls are the point of the page. Sweeping head roll
// makes the current implementation's delivered dose oscillate by about 12%
// while the normalized one holds flat, and nobody has to move a chair.

import { FaceMorphCurrent } from '../algo/FaceMorphCurrent'
import {
  FaceMorphNormalized, type ControlLaw, type SmileUnit,
} from '../algo/FaceMorphNormalized'
import {
  ALPHA_TO_SMILE_UNITS, AMPLITUDE_STATS, CANONICAL_TEMPLATE,
  CORNER_TRAVEL_AT_FULL_SMILE,
} from '../algo/faceModel.gen'
import { LEFT_CORNER, RIGHT_CORNER, RIGID_IDX } from '../algo/constants'
import { createLandmarker } from '../algo/landmarkerHost'
import { fitFrame, toCanonical, type Pt } from '../algo/procrustes'
import {
  ViewTransform, IDENTITY_VIEW, isIdentity, type ViewParams,
} from '../algo/viewTransform'
import { StripChart } from './plot'

const W = 1280
const H = 720
/** The published track is 30 fps, so running the pipeline at 60 would do twice
 *  the work and throw half away. */
const TARGET_FPS = 30
/** The scoring probe is a full extra detection, so it runs rarely. */
const PROBE_HZ = 3

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T

type SourceKind = 'webcam' | 'corpus' | 'file'

class Demo {
  private current = new FaceMorphCurrent()
  private normalized = new FaceMorphNormalized()
  private view = new ViewTransform()
  private probe!: Awaited<ReturnType<typeof createLandmarker>>
  private template!: Float64Array

  private video = document.createElement('video')
  private still: ImageBitmap | null = null
  private sourceKind: SourceKind = 'webcam'

  private ctxRaw = $<HTMLCanvasElement>('cRaw').getContext('2d')!
  private ctxA = $<HTMLCanvasElement>('cA').getContext('2d')!
  private ctxB = $<HTMLCanvasElement>('cB').getContext('2d')!
  private chart = new StripChart($<HTMLCanvasElement>('chart'))
  private seriesCmd: Array<{ t: number; v: number }> = []
  private seriesA: Array<{ t: number; v: number }> = []
  private seriesB: Array<{ t: number; v: number }> = []

  private viewParams: ViewParams = { ...IDENTITY_VIEW }
  private alpha = 1
  private blind = false
  private blindSwap = false
  private lastProbe = 0
  private lastFrame = 0
  private clock = 1000
  private frameTimes: number[] = []
  private frameCount = 0
  private sweep: { key: keyof ViewParams; from: number; to: number; t0: number } | null = null

  private corpus: Array<{ id: string; file: string }> = []
  private corpusIdx = 0
  private recording: Array<Record<string, number>> | null = null

  async init() {
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

    $('engine').textContent =
      `MediaPipe ${this.probe.delegate} · assets ${this.probe.assets}`
    $('calPrior').textContent = AMPLITUDE_STATS.p50.toFixed(3)

    this.wireUi()
    await this.loadCorpusList()
    await this.startWebcam()
    requestAnimationFrame(() => this.loop())
  }

  // ---- sources ---------------------------------------------------------

  private async startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: W, height: H }, audio: false,
      })
      this.video.srcObject = stream
      this.video.muted = true
      this.video.playsInline = true
      await this.video.play()
      this.sourceKind = 'webcam'
    } catch (err) {
      $('engine').textContent = `no camera (${String(err).slice(0, 60)}) — use a test face`
      this.sourceKind = 'corpus'
      await this.loadCorpusFace(0)
    }
  }

  private async loadCorpusList() {
    try {
      const res = await fetch('/corpus/manifest.json')
      if (!res.ok) return
      const mf = await res.json()
      const seen = new Set<string>()
      for (const f of mf.frames as Array<{ id: string; expr: string; file: string }>) {
        if (f.expr === 'N' && !seen.has(f.id)) {
          seen.add(f.id)
          this.corpus.push({ id: f.id, file: f.file })
        }
      }
    } catch { /* corpus is optional */ }
  }

  private async loadCorpusFace(i: number) {
    if (!this.corpus.length) return
    this.corpusIdx = (i + this.corpus.length) % this.corpus.length
    const f = this.corpus[this.corpusIdx]
    const res = await fetch(`/corpus/frames/${f.file}`)
    this.still?.close()
    this.still = await createImageBitmap(await res.blob())
    this.sourceKind = 'corpus'
    this.normalized.reset()
    $('faceName').textContent = `${f.id}  (${this.corpusIdx + 1}/${this.corpus.length})`
  }

  private currentSource(): CanvasImageSource | null {
    if (this.sourceKind === 'webcam') {
      return this.video.readyState >= 2 ? this.video : null
    }
    return this.still
  }

  // ---- the loop --------------------------------------------------------

  private loop() {
    requestAnimationFrame(() => this.loop())
    const now = performance.now()
    if (now - this.lastFrame < 1000 / TARGET_FPS - 1) return
    this.lastFrame = now
    this.clock += 1000 / TARGET_FPS

    this.stepSweep(now)

    const src = this.currentSource()
    if (!src) return

    // One view transform, applied once, shared by both implementations — so
    // any difference between the panels is the algorithm and nothing else.
    const framed: CanvasImageSource = isIdentity(this.viewParams)
      ? src
      : this.view.apply(src, W, H, this.viewParams)

    this.ctxRaw.setTransform(1, 0, 0, 1, 0, 0)
    this.ctxRaw.drawImage(framed, 0, 0, W, H)

    this.current.setAlpha(this.alpha)
    this.normalized.setAlpha(this.alpha)
    const tA0 = performance.now()
    this.current.render(framed as HTMLVideoElement, this.ctxA, W, H, this.clock)
    const tA = performance.now() - tA0
    const tB0 = performance.now()
    this.normalized.render(framed as HTMLVideoElement, this.ctxB, W, H, this.clock)
    const tB = performance.now() - tB0

    this.frameCount++
    ;(window as unknown as Record<string, unknown>).__demoFrames = this.frameCount
    this.frameTimes.push(now)
    while (this.frameTimes.length && this.frameTimes[0] < now - 1000) this.frameTimes.shift()

    $('aMs').textContent = `${tA.toFixed(1)} ms`
    $('bMs').textContent = `${tB.toFixed(1)} ms`
    $('rawMs').textContent = `${this.frameTimes.length} fps`

    if (now - this.lastProbe > 1000 / PROBE_HZ) {
      this.lastProbe = now
      this.score(now)
    }
    this.updateReadouts()
    this.chart.draw([
      { label: 'commanded', color: '#9aa3b8', dashed: true, data: this.seriesCmd },
      { label: 'current', color: '#ffb454', data: this.seriesA },
      { label: 'normalized', color: '#5fd39a', data: this.seriesB },
    ], now)
  }

  /**
   * Score both outputs against the unmorphed frame.
   *
   * The reference is the raw panel rather than a second "sham" render, because
   * alpha is tweened rather than set: asking a processor for alpha 1 and
   * rendering one frame does not give a neutral frame, it gives a frame a
   * quarter of the way back towards neutral. Using the raw frame sidesteps that
   * entirely. It costs a little accuracy — a sham render also carries the
   * warp's resampling softness, and this does not — so the batch harness, where
   * the tween can be allowed to settle, does the stricter sham differencing.
   *
   * Differencing at all is what makes this fair across faces: the detector has
   * a per-face bias, and subtracting a reference measured on the same face
   * cancels it to first order.
   */
  private score(now: number) {
    const ref = this.corners($<HTMLCanvasElement>('cRaw'))
    const mA = this.travelFrom(ref, $<HTMLCanvasElement>('cA'))
    const mB = this.travelFrom(ref, $<HTMLCanvasElement>('cB'))
    const cmd = (this.alpha - 1) * ALPHA_TO_SMILE_UNITS

    this.seriesCmd.push({ t: now, v: Math.abs(cmd) * CORNER_TRAVEL_AT_FULL_SMILE })
    this.seriesA.push({ t: now, v: mA })
    this.seriesB.push({ t: now, v: mB })
    for (const s of [this.seriesCmd, this.seriesA, this.seriesB]) {
      while (s.length > 400) s.shift()
    }

    const dbg = this.normalized.debug
    const target = Math.abs(cmd) * CORNER_TRAVEL_AT_FULL_SMILE
    $('sA').innerHTML = `delivered <b>${fmt(mA)}</b> · target ${fmt(target)}`
    $('sB').innerHTML = `delivered <b>${fmt(mB)}</b> · target ${fmt(target)}`
      + (dbg.quality ? ` · stretch ${dbg.quality.maxExpansion.toFixed(2)}x` : '')
    $('sRaw').innerHTML = dbg.pose
      ? `roll <b>${dbg.pose.rollDeg.toFixed(0)}°</b> · turn <b>${dbg.pose.yawDeg.toFixed(0)}°</b>`
        + ` · height <b>${dbg.pose.pitchDeg.toFixed(0)}°</b>`
        + ` · own smile <b>${dbg.liveSmileUnits.toFixed(2)}</b>`
      : 'no face'

    if (this.recording) {
      this.recording.push({
        t: Math.round(now), alpha: this.alpha, target,
        current: mA, normalized: mB,
        roll: dbg.pose?.rollDeg ?? NaN, yaw: dbg.pose?.yawDeg ?? NaN,
        pitch: dbg.pose?.pitchDeg ?? NaN,
        amplitude: dbg.calibration.amplitude,
      })
    }
  }

  private travelFrom(ref: { l: Pt; r: Pt } | null, morphed: HTMLCanvasElement): number {
    if (!ref) return NaN
    const b = this.corners(morphed)
    if (!b) return NaN
    return (Math.hypot(b.l.x - ref.l.x, b.l.y - ref.l.y)
      + Math.hypot(b.r.x - ref.r.x, b.r.y - ref.r.y)) / 2
  }

  private corners(canvas: HTMLCanvasElement): { l: Pt; r: Pt } | null {
    const res = this.probe.landmarker.detect(canvas)
    if (!res.faceLandmarks?.length) return null
    const lm: Pt[] = res.faceLandmarks[0].map((p) => ({ x: p.x * W, y: p.y * H }))
    const frame = fitFrame(lm, this.template)
    return {
      l: toCanonical(frame, lm[LEFT_CORNER]),
      r: toCanonical(frame, lm[RIGHT_CORNER]),
    }
  }

  private updateReadouts() {
    const c = this.normalized.debug.calibration
    const pill = $('calState')
    pill.textContent = c.frozen ? `${c.state} (held)` : c.state
    pill.className = `pill ${c.state}`
    $('calEvents').textContent = String(c.events)
    $('calWeight').textContent = `${(c.personWeight * 100).toFixed(0)}%`
    $('calAmp').textContent = c.amplitude.toFixed(3)
    $('calResid').textContent = `${(c.expectedResidualCv * 100).toFixed(0)}%`
  }

  // ---- sweeps ----------------------------------------------------------

  private stepSweep(now: number) {
    if (!this.sweep) return
    const period = 5000
    const phase = ((now - this.sweep.t0) % period) / period
    const v = this.sweep.from
      + (this.sweep.to - this.sweep.from) * (0.5 - 0.5 * Math.cos(2 * Math.PI * phase))
    ;(this.viewParams as unknown as Record<string, number>)[this.sweep.key as string] = v
    syncSliders(this.viewParams)
  }

  // ---- ui --------------------------------------------------------------

  private wireUi() {
    const alphaIn = $<HTMLInputElement>('alpha')
    const setAlpha = (a: number) => {
      this.alpha = a
      alphaIn.value = String(a)
      $('alphaOut').textContent = a.toFixed(2)
      $('commandedUnits').textContent =
        `= ${((a - 1) * ALPHA_TO_SMILE_UNITS).toFixed(2)} smile units`
    }
    alphaIn.addEventListener('input', () => setAlpha(parseFloat(alphaIn.value)))
    document.querySelectorAll<HTMLButtonElement>('.preset').forEach((b) => {
      b.addEventListener('click', () => setAlpha(parseFloat(b.dataset.alpha!)))
    })
    setAlpha(1)

    const bindView = (id: string, key: keyof ViewParams, fmtOut: (v: number) => string) => {
      const el = $<HTMLInputElement>(id)
      el.addEventListener('input', () => {
        this.sweep = null
        ;(this.viewParams as unknown as Record<string, number>)[key as string] =
          parseFloat(el.value)
        $(`${id}Out`).textContent = fmtOut(parseFloat(el.value))
        // A slider is a teleport, not a movement: drop the smoothing history so
        // the readouts are not measuring the filter catching up.
        this.normalized.resetTracking()
      })
    }
    bindView('pitch', 'pitchDeg', (v) => `${v}°`)
    bindView('roll', 'rollDeg', (v) => `${v}°`)
    bindView('yaw', 'yawDeg', (v) => `${v}°`)
    bindView('scale', 'scale', (v) => `${v.toFixed(2)}×`)

    const sweepBtn = (id: string, key: keyof ViewParams, from: number, to: number) => {
      $(id).addEventListener('click', () => {
        const btn = $(id)
        if (this.sweep && this.sweep.key === key) {
          this.sweep = null
          btn.classList.remove('on')
          return
        }
        document.querySelectorAll('.on').forEach((e) => e.classList.remove('on'))
        this.sweep = { key, from, to, t0: performance.now() }
        btn.classList.add('on')
      })
    }
    sweepBtn('sweepRoll', 'rollDeg', -22, 22)
    sweepBtn('sweepPitch', 'pitchDeg', -18, 18)
    sweepBtn('sweepScale', 'scale', 0.7, 1.4)

    $('resetView').addEventListener('click', () => {
      this.sweep = null
      document.querySelectorAll('.on').forEach((e) => e.classList.remove('on'))
      this.viewParams = { ...IDENTITY_VIEW }
      syncSliders(this.viewParams)
    })

    $('blind').addEventListener('click', () => {
      this.blind = !this.blind
      this.blindSwap = Math.random() < 0.5
      $('blind').classList.toggle('on', this.blind)
      $('labA').textContent = this.blind ? (this.blindSwap ? 'B' : 'A') : 'Current (shipped)'
      $('labB').textContent = this.blind ? (this.blindSwap ? 'A' : 'B') : 'Normalized (new)'
    })

    $('record').addEventListener('click', () => {
      if (this.recording) {
        const blob = new Blob([JSON.stringify(this.recording, null, 1)],
          { type: 'application/json' })
        const a = document.createElement('a')
        a.href = URL.createObjectURL(blob)
        a.download = `demo-session-${Date.now()}.json`
        a.click()
        this.recording = null
        $('record').classList.remove('on')
        $('record').textContent = '⬇ record 60 s'
        return
      }
      this.recording = []
      $('record').classList.add('on')
      $('record').textContent = '■ stop & save'
      setTimeout(() => {
        if (this.recording) ($('record') as HTMLButtonElement).click()
      }, 60000)
    })

    const unit = $<HTMLSelectElement>('unit')
    unit.addEventListener('change', () => {
      this.normalized.opts.unit = unit.value as SmileUnit
    })
    const law = $<HTMLSelectElement>('law')
    law.addEventListener('change', () => {
      this.normalized.opts.law = law.value as ControlLaw
    })

    const source = $<HTMLSelectElement>('source')
    source.addEventListener('change', async () => {
      const v = source.value as SourceKind
      $('file').classList.toggle('hidden', v !== 'file')
      if (v === 'webcam') await this.startWebcam()
      else if (v === 'corpus') await this.loadCorpusFace(this.corpusIdx)
      else $<HTMLInputElement>('file').click()
    })

    $<HTMLInputElement>('file').addEventListener('change', async (e) => {
      const f = (e.target as HTMLInputElement).files?.[0]
      if (!f) return
      this.still?.close()
      this.still = await createImageBitmap(f)
      this.sourceKind = 'file'
      this.normalized.reset()
      $('faceName').textContent = f.name
    })

    $('prevFace').addEventListener('click', () => this.loadCorpusFace(this.corpusIdx - 1))
    $('nextFace').addEventListener('click', () => this.loadCorpusFace(this.corpusIdx + 1))
  }
}

function syncSliders(v: ViewParams) {
  const set = (id: string, val: number, txt: string) => {
    const el = document.getElementById(id) as HTMLInputElement | null
    if (el) el.value = String(val)
    const out = document.getElementById(`${id}Out`)
    if (out) out.textContent = txt
  }
  set('pitch', v.pitchDeg, `${v.pitchDeg.toFixed(0)}°`)
  set('roll', v.rollDeg, `${v.rollDeg.toFixed(0)}°`)
  set('yaw', v.yawDeg, `${v.yawDeg.toFixed(0)}°`)
  set('scale', v.scale, `${v.scale.toFixed(2)}×`)
}

function fmt(v: number): string {
  return Number.isFinite(v) ? v.toFixed(4) : '—'
}

new Demo().init().catch((err) => {
  document.getElementById('engine')!.textContent = `failed: ${String(err)}`
  console.error(err)
})
