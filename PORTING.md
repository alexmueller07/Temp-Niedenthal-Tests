# Porting this back into the app

Nothing here has been merged. This is what it would take, in the order it should
happen, smallest and safest first.

`src/algo/` is self-contained by construction — `npm run typecheck` fails if
anything in it imports outside itself or outside `@mediapipe/tasks-vision`, the
package the app already depends on. So the code move is a directory copy.

---

## Stage 1 — the validity fixes, independent of everything else

These are worth doing whether or not the normalization ships, and they are small.
They affect studies that are already running.

**1. Make the sham condition take the same code path.**
`renderer/lib/faceMorph.ts:226`

```ts
if (Math.abs(this.alphaCurrent - 1) < 0.02) return false
```

Active frames pass through ~192 affine resamples; sham frames skip them
entirely, so the two conditions differ in local image softness for reasons that
have nothing to do with the manipulation. Run the warp with zero displacement
instead of returning early. Costs the CPU, removes the leak.

**2. Log the dose that was actually applied.**
`renderer/lib/effects.ts:211` reports `this.alpha`, the *commanded* target. The
tween, the yaw gate and the face-found state all sit between that and the pixels.
Expose `alphaCurrent` and the pose gate, and write them to `effect_state.csv`.
(The app's own documentation claims the applied alpha is already recorded; it is
not.)

**3. Decide what the yaw gate should do.**
`faceMorph.ts:241-248` fades the morph to zero between symmetry 0.65 and 0.35.
Head turns track who is speaking, so the dose is currently correlated with
conversational role — invisibly. Whatever is decided, log every attenuated frame
so it becomes a covariate rather than a confound.

**4. Put smile and frown on one scale.**
`main/presets.ts` uses gain 0.17 for smile and 0.13 for frown, with a different
field shape. "Strong smile" and "strong frown" differ by 31% in magnitude and
are not two ends of one axis, so dose-response across the presets is not
interpretable. Either match them or stop describing them as a ladder.

None of this requires the new algorithm.

---

## Stage 2 — the geometry, with no calibration

Buys pose and roll invariance and a better-behaved warp. No new state, no
calibration window, nothing to explain to a participant.

1. Copy `src/algo/*.ts` to `renderer/lib/faceMorph/`.
2. In `types.ts`, replace the local `ExpressionState` / `SmileType` /
   `ExpressionLabel` declarations with `export * from '../../../main/protocol'`.
   They are already byte-identical — that is checked by the type assertion at
   the bottom of `FaceMorphNormalized.ts`.
3. Point the asset paths in `landmarkerHost.ts` back at `/mediapipe/...`.
4. Construct with `calibrate: false, unit: 'population'`. That is the geometry
   layer alone: same physical dose for everyone, now independent of how they sit.
5. Update the two call sites:
   - `renderer/lib/effects.ts:14`
   - `renderer/lib/capture.ts:11`

   Both do `new FaceMorphProcessor()` and call `render(video, ctx, w, h, ts)`.
   The new class satisfies the same interface; `render` accepts a wider source
   type, which `HTMLVideoElement` still satisfies.

Nothing else in the app changes. `alpha` keeps its meaning and its range, and
the preset values still work — `ALPHA_TO_SMILE_UNITS` is set so alpha 1.9 lands
at the same physical magnitude the current morph already delivers on a median
face, so switching does not silently change how strong the manipulation is.

**Check before merging:** run the app, watch the partner view while tilting your
head. The current build's morph shears under roll and manufactures a left/right
asymmetry that the app's own classifier reads as a dominance smile. The new one
should not.

---

## Stage 3 — per-person calibration

Only worth doing if the lab wants the manipulation scaled to each person's own
expressive range. That is a scientific decision, not a technical one — see the
end of FINDINGS.md.

Set `calibrate: true, unit: 'self'`. Then three things need attention:

**Where the smiles come from.** Calibration needs several observed smiles, not
one — one is barely better than none. The waiting room is a silent "please wait
for the researcher" screen and there is no microphone check anywhere in the
codebase, so it will see almost nothing and every session will quietly run on
the population default. The realistic source is the opening minutes of the
conversation itself, which means the manipulation should not start immediately.
That is a protocol change and needs Randy.

**Surfacing the calibration state.** `Calibration.status` reports how many
smiles have been seen and how much weight the estimate carries. Put it in
`Telemetry` and in the session manifest: the researcher needs to know whether a
participant ran calibrated or on the default, and so does the analysis.

**The freeze rule.** The estimate is held still while the morph is on and only
adopted when it returns to neutral, so the dose cannot drift mid-trial. This is
already implemented; do not remove it.

---

## Stage 4 — things deliberately not built

**The closed-loop trim.** Measuring the warped output with MediaPipe and
adjusting the gain is circular: the warp is constructed to move exactly the
landmarks MediaPipe tracks, so the loop converges to a fixed point that means
nothing about perception. Worse, on a face where the detector under-reports —
which is the bearded, low-contrast case this was supposed to fix — it would
drive the warp to grotesque and still under-report. The same computation is
useful as an *artifact monitor* rather than a controller.

**Neural reenactment** (LivePortrait and similar). Solves the cross-identity
retargeting problem properly, and needs a 4090-class GPU. There is none on this
machine or on the lab's. Documented rather than built.

**An appearance/salience layer.** A beard does not change geometry; it degrades
landmark accuracy and makes a warped texture look worse. Geometric normalization
does not fix that, and claiming otherwise would set up a disappointment. Worth
measuring before building.
