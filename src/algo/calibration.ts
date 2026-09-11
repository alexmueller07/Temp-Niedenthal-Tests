// Learning how big this particular person's smile is.
//
// Why this exists at all: the offline analysis (FINDINGS.md) showed the warp
// already delivers an equal displacement to every face — the inequality is that
// people's own smiles differ about twofold, so the same displacement is a very
// different share of each person's range. It also showed that nothing about the
// face predicts that range: 37 anatomical and perceptual predictors, including
// the Chicago Face Database's own measurements, give a leave-one-out R^2 of
// about zero. So it has to be observed.
//
// Two things are estimated, and only these two:
//   rest      — the person's resting mouth shape, so "no smile" means their
//               neutral rather than the corpus average. Some people rest with
//               upturned corners; that should not read as a smile.
//   amplitude — the size of their full smile, in canonical shape units.
//
// Direction is NOT estimated. Real smiles point almost the same way in almost
// everyone (mean cosine with the corpus axis: 0.96), so there is nothing there
// to learn and learning it would only add variance.
//
// KNOWN SCALE MISMATCH, unresolved. The population prior below comes from the
// Chicago Face Database, whose smiles are *posed for a camera*. What this class
// observes live are *conversational* smiles, which are typically smaller. So the
// prior and the observations are not on the same scale, and shrinking one toward
// the other biases the estimate downward early in a session, fading as evidence
// accumulates. It does not affect the offline evaluation (which supplies
// amplitudes directly) but it does affect the live path, and the prior should be
// re-fitted on conversational video before per-person calibration is switched on
// in a study. Flagged in FINDINGS.md and PORTING.md.

import { AMPLITUDE_RELIABILITY, AMPLITUDE_STATS, POPULATION_REST } from './faceModel.gen'
import { DRIVEN } from './constants'
import { readSmile } from './smileMetric'
import type { Pt } from './procrustes'

const N = DRIVEN.length
const DIM = 2 * N

/** Bayes-optimal shrinkage: with reliability r, N observations deserve weight
 *  N / (N + (1-r)/r) against the population prior. Measured, not tuned. */
const PRIOR_OBSERVATIONS = (1 - AMPLITUDE_RELIABILITY) / AMPLITUDE_RELIABILITY

/** Rolling window for the rest-shape estimate: 20 s at 30 Hz. */
const REST_WINDOW = 600
/** Rest is the mean of the least-smiling quarter of that window. */
const REST_QUANTILE = 0.25
const REST_RECOMPUTE_EVERY = 15

/** A smile excursion has to clear this fraction of the prior amplitude to count,
 *  and fall back below the lower one to close. Hysteresis, so one noisy frame
 *  cannot open or close an event. */
const EVENT_ON = 0.30
const EVENT_OFF = 0.18
const EVENT_MIN_MS = 300

export type CalibrationState = 'cold' | 'collecting' | 'ready'

export interface CalibrationStatus {
  state: CalibrationState
  /** Number of complete smile events observed. */
  events: number
  /** Weight currently given to the observed estimate vs the population prior. */
  personWeight: number
  /** Amplitude in use, canonical shape units. */
  amplitude: number
  /** Population prior, for comparison in the UI. */
  prior: number
  /** True while the estimate is held still because the morph is running. */
  frozen: boolean
  /** Expected residual dispersion at this evidence level, from the offline
   *  reliability model — i.e. how well calibrated we actually are. */
  expectedResidualCv: number
}

export class Calibration {
  private restBuf: Float64Array[] = []
  private restQ: number[] = []
  private restIdx = 0
  private sinceRecompute = 0
  private rest: Float64Array
  private haveOwnRest = false

  private eventOpen = false
  private eventPeak = 0
  private eventStart = 0
  private observations: number[] = []

  private frozenAmplitude: number | null = null
  private override: number | null = null

  constructor() {
    this.rest = new Float64Array(DIM)
    const map = new Map(POPULATION_REST.map(([i, v]) => [i, v]))
    DRIVEN.forEach((idx, k) => {
      const v = map.get(idx)
      if (v) { this.rest[2 * k] = v[0]; this.rest[2 * k + 1] = v[1] }
    })
  }

  /** The rest shape currently in use (population until the person's own is known). */
  get restShape(): Float64Array { return this.rest }

  get amplitude(): number {
    if (this.override !== null) return this.override
    if (this.frozenAmplitude !== null) return this.frozenAmplitude
    return this.liveAmplitude()
  }

  /**
   * Force the amplitude instead of learning it.
   *
   * Only for evaluation: still photographs contain no smile events, so a batch
   * run over a stills corpus can never calibrate. Supplying each identity's
   * amplitude — measured independently from their own smile photographs —
   * answers "how equal would the dose be if calibration had converged", which
   * is the achievable ceiling. Passing null restores live estimation.
   */
  setOverride(amplitude: number | null) { this.override = amplitude }

  /**
   * Force the resting mouth shape instead of learning it.
   *
   * Same reason as the amplitude override: a single photograph gives the
   * estimator nothing to work with, so it falls back to the corpus-average
   * rest. For someone whose mouth rests more upturned than average that reads
   * as "already smiling", and the safety clamp then fires when it should not —
   * measuring the fallback rather than the algorithm.
   */
  setRestOverride(rest: ArrayLike<number> | null) {
    if (!rest) return
    const r = new Float64Array(DIM)
    for (let i = 0; i < DIM && i < rest.length; i++) r[i] = rest[i]
    this.rest = r
    this.haveOwnRest = true
  }

  /** Clear all per-person state, keeping nothing from the previous face. */
  reset() {
    this.restBuf = []
    this.restQ = []
    this.restIdx = 0
    this.sinceRecompute = 0
    this.haveOwnRest = false
    this.eventOpen = false
    this.observations = []
    this.frozenAmplitude = null
    const map = new Map(POPULATION_REST.map(([i, v]) => [i, v]))
    this.rest = new Float64Array(DIM)
    DRIVEN.forEach((idx, k) => {
      const v = map.get(idx)
      if (v) { this.rest[2 * k] = v[0]; this.rest[2 * k + 1] = v[1] }
    })
  }

  private liveAmplitude(): number {
    const n = this.observations.length
    const prior = AMPLITUDE_STATS.p50
    if (n === 0) return prior
    const mean = this.observations.reduce((a, b) => a + b, 0) / n
    const w = n / (n + PRIOR_OBSERVATIONS)
    return (1 - w) * prior + w * mean
  }

  /**
   * Hold the current estimate still.
   *
   * Called when the morph turns on. Updating the gain while the manipulation is
   * running would make the dose drift mid-trial for reasons the participant
   * caused — which is both a visible artifact and an analysis problem. Evidence
   * keeps accumulating in the background; it is adopted at the next neutral.
   */
  freeze() {
    if (this.frozenAmplitude === null) this.frozenAmplitude = this.liveAmplitude()
  }

  /** Release the hold and adopt everything learned while frozen. */
  thaw() { this.frozenAmplitude = null }

  get status(): CalibrationStatus {
    const n = this.observations.length
    const w = n / (n + PRIOR_OBSERVATIONS)
    const cvObs = AMPLITUDE_STATS.sd / AMPLITUDE_STATS.mean
    const r = AMPLITUDE_RELIABILITY
    const residual = n === 0
      ? cvObs * Math.sqrt(r)
      : cvObs * Math.sqrt((1 - r) / n)
    return {
      state: n === 0 ? (this.haveOwnRest ? 'collecting' : 'cold')
        : n < 3 ? 'collecting' : 'ready',
      events: n,
      personWeight: w,
      amplitude: this.amplitude,
      prior: AMPLITUDE_STATS.p50,
      frozen: this.frozenAmplitude !== null,
      expectedResidualCv: residual,
    }
  }

  /**
   * Feed one accepted frame. `canonMouth` must come from the RAW camera frame,
   * never the morphed output — otherwise the estimate learns from its own
   * manipulation.
   */
  update(canonMouth: ArrayLike<Pt>, tsMs: number): void {
    // --- rest shape: mean of the least-smiling quarter of a rolling window ---
    const flat = new Float64Array(DIM)
    for (let k = 0; k < N; k++) {
      flat[2 * k] = canonMouth[k].x
      flat[2 * k + 1] = canonMouth[k].y
    }
    const { projection: q, magnitude: mag } = readSmile(canonMouth, this.rest)

    if (this.restBuf.length < REST_WINDOW) {
      this.restBuf.push(flat)
      this.restQ.push(q)
    } else {
      this.restBuf[this.restIdx] = flat
      this.restQ[this.restIdx] = q
      this.restIdx = (this.restIdx + 1) % REST_WINDOW
    }

    if (++this.sinceRecompute >= REST_RECOMPUTE_EVERY && this.restBuf.length >= 60) {
      this.sinceRecompute = 0
      this.recomputeRest()
    }

    // --- smile events: peak nuisance-free shape change per excursion ---------
    const prior = AMPLITUDE_STATS.p50

    if (!this.eventOpen) {
      if (q > EVENT_ON * prior) {
        this.eventOpen = true
        this.eventStart = tsMs
        this.eventPeak = mag
      }
    } else {
      this.eventPeak = Math.max(this.eventPeak, mag)
      if (q < EVENT_OFF * prior) {
        this.eventOpen = false
        const dur = tsMs - this.eventStart
        // Reject flickers and anything implausibly large — a tracking glitch
        // should not become a permanent gain.
        if (dur >= EVENT_MIN_MS && this.eventPeak > 0.1 && this.eventPeak < 3 * prior) {
          this.observations.push(this.eventPeak)
          if (this.observations.length > 40) this.observations.shift()
        }
      }
    }
  }

  /**
   * True when a smile is in progress and the face is at or near its peak.
   *
   * This is the signal to bank a frame. A bare level threshold is not enough:
   * before the person's own resting mouth has been observed, "how much are they
   * smiling" is measured against the corpus average, and someone whose mouth
   * rests upturned reads as smiling when they are not — so a neutral frame gets
   * banked as a smile source and morphing toward it does nothing.
   */
  atSmilePeak(canonMouth: ArrayLike<Pt>): boolean {
    if (!this.eventOpen || !this.haveOwnRest) return false
    const mag = readSmile(canonMouth, this.rest).magnitude
    return mag >= this.eventPeak * 0.92
  }

  /** Whether this person's own resting mouth has been observed yet. */
  get restKnown(): boolean { return this.haveOwnRest }

  /** Face lost, or a long gap: close any open event without recording it. */
  interrupt() {
    this.eventOpen = false
  }

  private recomputeRest() {
    const n = this.restBuf.length
    const order = Array.from({ length: n }, (_, i) => i)
      .sort((a, b) => this.restQ[a] - this.restQ[b])
    const take = Math.max(10, Math.floor(n * REST_QUANTILE))
    const out = new Float64Array(DIM)
    for (let j = 0; j < take; j++) {
      const src = this.restBuf[order[j]]
      for (let i = 0; i < DIM; i++) out[i] += src[i]
    }
    for (let i = 0; i < DIM; i++) out[i] /= take
    this.rest = out
    this.haveOwnRest = true
  }
}
