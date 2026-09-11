// The participant's genuine expression, read from blendshapes on the RAW camera
// frame — never the morphed output.
//
// This is a copy of the production classifier's logic rather than a shared
// module, because the production copy lives in the frozen baseline and must not
// be touched. Two fixes applied here:
//
//   1. The production version hard-codes `dt = 33` in its exponential average,
//      but the render loop runs at ~60 Hz against a 30 fps camera, so the
//      effective time constant is half the documented 220 ms and the thresholds
//      were calibrated against whatever frame rate happened to be running. Here
//      dt is measured.
//   2. Duplicate frames are not fed twice (the caller handles that), so the
//      average is not biased by the camera's frame rate.
//
// The thresholds themselves are unchanged, including their limitations: they
// were fitted to five example photographs, the sub-type mapping is a heuristic,
// and it should not be presented as a validated instrument.

import type { ExpressionLabel, ExpressionState, SmileType } from './types'

export const DETECTION_TUNING = {
  smileOn: 0.6,
  smileOff: 0.45,
  frownOn: 0.08,
  frownOff: 0.04,
  frownSmileGate: 0.15,
  rewardOpenness: 0.2,
  dominanceRelAsymmetry: 0.12,
  emaTauMs: 220,
  debounceMs: 350,
}

export class ExpressionDetector {
  private ema: Record<string, number> = {}
  private publishedLabel: ExpressionLabel = 'neutral'
  private publishedType: SmileType | null = null
  private candidateLabel: ExpressionLabel = 'neutral'
  private candidateType: SmileType | null = null
  private candidateSince = 0
  private last: ExpressionState | null = null
  private lastTs: number | null = null

  get expression(): ExpressionState | null { return this.last }

  update(
    categories: Array<{ categoryName: string; score: number }> | null,
    tsMs: number,
  ): void {
    const raw: Record<string, number> = {}
    if (categories) for (const c of categories) raw[c.categoryName] = c.score
    const g = (n: string) => raw[n] ?? 0

    const dt = this.lastTs === null ? 33 : Math.min(250, Math.max(1, tsMs - this.lastTs))
    this.lastTs = tsMs
    const k = 1 - Math.exp(-dt / DETECTION_TUNING.emaTauMs)
    const ema = (key: string, v: number) => {
      const prev = this.ema[key] ?? v
      const next = prev + (v - prev) * k
      this.ema[key] = next
      return next
    }

    const smileL = ema('smileL', g('mouthSmileLeft'))
    const smileR = ema('smileR', g('mouthSmileRight'))
    const smile = (smileL + smileR) / 2
    const frown = ema('frown', (g('mouthFrownLeft') + g('mouthFrownRight')) / 2)
    const pressL = ema('pressL', g('mouthPressLeft'))
    const pressR = ema('pressR', g('mouthPressRight'))
    const lipPress = (pressL + pressR) / 2
    const openness = ema('open',
      (g('mouthUpperUpLeft') + g('mouthUpperUpRight')) / 2
      + g('jawOpen') * 0.8
      + ((g('mouthLowerDownLeft') + g('mouthLowerDownRight')) / 2) * 0.8)
    const asymmetry = Math.abs(smileL - smileR) + Math.abs(pressL - pressR)
    const relAsymmetry = asymmetry / Math.max(0.3, Math.max(smileL, smileR))
    const eyeConstriction = ema('eye',
      (g('eyeSquintLeft') + g('eyeSquintRight')
        + g('cheekSquintLeft') + g('cheekSquintRight')) / 4)

    const T = DETECTION_TUNING
    const frowning = (on: boolean) =>
      frown >= (on ? T.frownOn : T.frownOff) && smile < T.frownSmileGate

    let label: ExpressionLabel
    if (this.publishedLabel === 'smiling') {
      label = smile >= T.smileOff ? 'smiling' : frowning(true) ? 'frowning' : 'neutral'
    } else if (this.publishedLabel === 'frowning') {
      label = frowning(false) ? 'frowning' : smile >= T.smileOn ? 'smiling' : 'neutral'
    } else {
      label = smile >= T.smileOn ? 'smiling' : frowning(true) ? 'frowning' : 'neutral'
    }

    let smileType: SmileType | null = null
    if (label === 'smiling') {
      if (openness >= T.rewardOpenness) smileType = 'reward'
      else if (relAsymmetry >= T.dominanceRelAsymmetry) smileType = 'dominance'
      else smileType = 'affiliative'
    }

    if (label !== this.candidateLabel || smileType !== this.candidateType) {
      this.candidateLabel = label
      this.candidateType = smileType
      this.candidateSince = tsMs
    } else if (
      (label !== this.publishedLabel || smileType !== this.publishedType)
      && tsMs - this.candidateSince >= T.debounceMs
    ) {
      this.publishedLabel = label
      this.publishedType = smileType
    }

    const r2 = (v: number) => Math.round(v * 100) / 100
    this.last = {
      label: this.publishedLabel,
      smileType: this.publishedLabel === 'smiling' ? this.publishedType : null,
      smile: r2(smile),
      frown: r2(frown),
      asymmetry: r2(relAsymmetry),
      eyeConstriction: r2(eyeConstriction),
      lipPress: r2(lipPress),
      openness: r2(openness),
    }
  }
}
