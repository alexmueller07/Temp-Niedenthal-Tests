// Mirrors the production wire types in main/protocol.ts:57-72 so this directory
// can be copied into the app unchanged apart from swapping this import for
// '../protocol'. Do not add fields here that the app does not have; anything new
// belongs in the PoC-only types at the bottom.

export type SmileType = 'reward' | 'affiliative' | 'dominance'
export type ExpressionLabel = 'neutral' | 'smiling' | 'frowning'

export interface ExpressionState {
  label: ExpressionLabel
  smileType: SmileType | null
  smile: number
  frown: number
  asymmetry: number
  eyeConstriction: number
  lipPress: number
  openness: number
}

/**
 * What the production app calls. Both the frozen baseline and the normalized
 * implementation satisfy this, which is checked at compile time in apiCompat.ts.
 *
 * One deliberate widening: `render` takes any canvas image source rather than
 * only HTMLVideoElement. HTMLVideoElement is still assignable, so every
 * production call site compiles unchanged — but the batch harness can feed
 * decoded frames directly, which is what makes offline runs frame-exact and
 * reproducible.
 */
export type MorphSource = HTMLVideoElement | HTMLCanvasElement | ImageBitmap

export interface FaceMorphAPI {
  init(): Promise<void>
  setAlpha(alpha: number): void
  render(
    src: MorphSource,
    dstCtx: CanvasRenderingContext2D,
    width: number,
    height: number,
    tsMs: number,
  ): boolean
  readonly ready: boolean
  readonly faceFound: boolean
  readonly expression: ExpressionState | null
  close(): void
}

// ---- PoC-only -------------------------------------------------------------

/** Per-frame record of what was *actually* applied, not what was commanded.
 *
 * The production app logs the commanded alpha, which is not the same thing: the
 * tween, the pose gate and the output clamp all sit between the command and the
 * pixels. Under a multiplicative control law the realized dose also depends on
 * what the participant is doing, so the nominal preset is not the independent
 * variable — this is. */
export interface RealizedDose {
  /** Commanded intensity, in normalized smile units. */
  commanded: number
  /** Intensity actually applied after tween, pose gate and clamp. */
  realized: number
  /** Corner displacement actually applied, in interpupillary units. */
  travelPerIpd: number
  /** Pose attenuation actually in force, 1 = none, 0 = morph suppressed. */
  poseGain: number
  /** True when the output clamp bound the displacement. */
  clamped: boolean
  /** Head pose at the time, degrees. */
  yawDeg: number
  pitchDeg: number
  rollDeg: number
  /** Whether a face was tracked at all. */
  faceFound: boolean
}
