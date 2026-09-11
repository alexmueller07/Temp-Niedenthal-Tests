# What actually makes the smile morph land unequally

Measured on the Chicago Face Database: 1,433 photographs, 823 identities, of
which 153 have a neutral photograph plus both a closed-mouth and a toothy smile.
Everything below is computed from face landmarks — no rendering, no human
ratings yet. Reproduce with `tools/cfd_extract.py` then `tools/cfd_analysis.py`
and `tools/s0_amplitude.py`.

**Headline: the warp is not the problem. The warp is almost perfectly consistent.
What varies is how big each person's own smile is, and the current morph is
blind to it.**

---

## 1. The warp already delivers an equal displacement

At `alpha = 1.9` ("Smile + strong"), across 823 faces at identical framing:

| measure | CV | P90/P10 |
|---|---|---|
| corner travel, in mouth-widths | **0.004** | 1.01 |
| corner travel, in interpupillary distances | 0.072 | 1.21 |
| mouth width itself, in interpupillary distances | 0.073 | 1.21 |

The first row is the morph doing exactly what it was designed to do. The second
row's 7% is not a flaw in the warp — it is anatomy: the CV of delivered travel
per IPD (0.072) is identical to the CV of mouth width per IPD (0.073), so every
bit of it is just people having different-sized mouths.

I expected lip thickness and mouth opening to move the gain through the
`sin(pi*u)sin(pi*v)` window term. They barely do: the field weight landing on
the corner has CV 0.004 across all 823 faces. That hypothesis was wrong.

## 2. What does vary: people's own smiles differ about twofold

Each person's own neutral-to-smile shape change, in canonical units, with
jaw-opening and lip-parting removed:

| | mean | CV | P90/P10 | full range |
|---|---|---|---|---|
| closed-mouth smile | 0.506 | 0.276 | 1.9 | 0.09 – 1.41 |

So the same added displacement is a very different share of each person's
expressive range. Expressed that way — what fraction of this person's own smile
are we adding? — the current manipulation looks like this:

| | value |
|---|---|
| delivered dose / own smile amplitude, CV | **0.559** |
| P90 / P10 | **2.07** |
| full range across 153 people | 0.085 – 1.14 (**13x**) |
| correlation between delivered dose and what the face needs | **−0.06** |

That last number is the point. The morph is not merely imprecise about
expressive range; it does not know about it at all.

**This is ~8x larger than the geometric dispersion, and it is the thing worth
fixing.**

## 3. Smile size is a real property of the person, not posing noise

Two smiles per person (closed-mouth and toothy) act as parallel measures:

| correlation | r |
|---|---|
| closed-mouth vs toothy smile amplitude, same person | **+0.56** |
| closed-mouth smile vs **angry** amplitude, same person | +0.02 |
| closed-mouth smile vs **fearful** amplitude, same person | +0.09 |

It is specific to smiling — it is not a general tendency to pose hard. So there
is a genuine per-person quantity to calibrate against. Reliability of a single
observation is 0.56; of a two-observation average, 0.72.

## 4. Nothing about the face predicts it

Leave-one-out R² predicting a person's own smile amplitude:

| predictors | count | LOO R² |
|---|---|---|
| landmark geometry (mouth width, lip height, resting corner rise, face width, lower-face height) | 5 | −0.07 |
| CFD physical measurements (lip thickness, lip fullness, face widths, cheekbone prominence, fWHR, luminance, skin colour, …) | 23 | −0.10 |
| CFD human ratings of the neutral face (happy, attractive, dominant, warm, babyfaced, masculine/feminine, …) | 9 | **+0.01** |
| all of the above | 37 | −0.10 |

The reliability ceiling here is R² = 0.72, so there was plenty of room to find
something. Nothing did. **A static per-face gain model cannot work.** How
expressive someone is is not written on their face — it has to be observed.

This kills the cheapest possible fix, which is worth knowing before building it.

## 5. How much observation is needed

From classical test theory, with single-measurement reliability 0.56:

| calibration | residual dispersion (CV) | improvement |
|---|---|---|
| none — population mean gain | 0.168 | — |
| 1 observed smile | 0.150 | 11% |
| 2 | 0.106 | 37% |
| 3 | 0.086 | 49% |
| 5 | 0.067 | 60% |
| 10 | 0.047 | 72% |
| 20 | 0.033 | 80% |

**One smile is barely better than no calibration**, because a single smile is
itself a noisy sample of the person. The gain has to be averaged over several.

That has a direct design consequence: a brief "look at the camera" calibration
moment would not work even if it did reliably produce a smile. A few minutes of
actual conversation would. The current app has neither — the waiting room is a
silent "please wait for the researcher" screen, and there is no microphone check
anywhere in the codebase.

## 6. Toothy vs closed-mouth smiles: solved, and it needs no data

Randy's worry — *what if they give us a toothy smile, does it matter, the morph
is closed-mouth* — turns out to be fixable analytically. Jaw opening is a rigid
rotation of the mandible about a fixed hinge, so its displacement field can be
written down rather than learned; same for the lips parting. Projecting both out
of the measured shape change:

| | median cos(closed, toothy) | 10th percentile |
|---|---|---|
| raw shape change | 0.820 | 0.620 |
| jaw-opening removed | **0.972** | **0.923** |
| jaw-opening + lip-parting removed | 0.968 | 0.914 |

After the projection the two kinds of smile are essentially the same
measurement. **It does not matter which kind of smile we happen to observe.**

## 7. Smile direction is the same in nearly everyone

Median cosine between any two people's smile axes: **0.94**. Mean cosine of an
individual smile with the corpus average: **0.96**.

So there is nothing to gain from learning each person's smile *direction* — only
its *size*. That removes a large chunk of proposed machinery.

A useful by-product: the corpus-average smile axis, computed from 307 real
neutral-to-smile pairs, is a better description of a smile than the current
hand-tuned formula (see §9), and it ships as `src/algo/faceModel.gen.ts`.

## 8. Camera and pose: it is head-turn, not height

Same face, varied only in how it sits in front of the camera. Within-face CV of
the delivered dose — this should be zero:

| condition | within-face CV | notes |
|---|---|---|
| distance (IPD 45–200 px) | **0.000** | already exactly invariant |
| camera height / pitch (±20°) | 0.043 | small |
| height in frame (eyes at 0.12–0.78 of frame height) | 0.070 | ROI clipped by the frame edge in 18% of cells |
| head roll (±25°) | 0.123 | and the yaw gate fires spuriously in 3% of cells |
| **head yaw (±30°)** | **1.331** | **the yaw gate attenuates the morph in 86% of cells** |

Two things here.

The one I predicted — that "how tall they are" works through the mouth region
being clipped at the frame edge — is real but minor: 7%, and only at extreme
framing. Distance, which I also expected to matter, is already perfect.

**How head turn was simulated, and what that does and does not license.** The
table above comes from rotating each face's 3D landmark cloud and re-projecting
it through a plausible webcam focal length. That is a fair simulation of the
*geometry* — where the landmarks end up — and the yaw-gate result rests on it.
It says nothing about *appearance*: a turned head also hides part of itself and
shades differently, which no landmark rotation reproduces.

Re-projecting the rendered image instead does not help. A camera-rotation
homography applied to an already-flat photograph keystones it; checked by
re-detecting the pose on the transformed frame, a commanded 20° of yaw comes back
as about 7° of yaw plus 10° of spurious roll. (Roll and distance *are* exact by
the same check: 15° in, 15.6° out; scale exact to the commanded factor.)

So the rendered behaviour under a real head turn is untested. The cheap fix is
footage: the available dev clips are 17-20 frames with a yaw range of 0-4°, so
someone needs to sit in front of the demo and turn their head while it records.
The demo has a recorder button for exactly that.

The mechanism that dominates is **head turn**. The morph's yaw gate fades the
manipulation to *zero* between symmetry 0.65 and 0.35, and across a ±30° turn
that swings the dose by more than 100% of its mean. Head turns track who is
speaking. **So the dose is silently confounded with conversational role** — a
participant receives more manipulation while listening face-on than while
turning to speak. Head roll leaks into that gate too, because it is estimated
from screen-x distances.

## 9. Four validity problems, independent of any of this

Found while reading `renderer/lib/faceMorph.ts`. These matter for studies already
running.

1. **The sham condition takes a different code path.** `if (|alphaCurrent − 1| <
   0.02) return false` skips the warp entirely, so active frames pass through
   ~192 affine resamples and sham frames pass through none. The two conditions
   differ in local image softness for reasons unrelated to the manipulation.
   Fix: always run the warp, with zero displacement, in sham.
2. **Dose is confounded with speaking role**, via the yaw gate (§8).
3. **`alpha` is not a common scale.** Smile uses gain 0.17, frown 0.13, and a
   different field shape (the frown adds a lower-lip pout term). "Strong smile"
   and "strong frown" differ by 31% in magnitude and are not two ends of one
   axis, so dose-response across the presets is not interpretable as it stands.
4. **The realized dose is never logged.** Telemetry records the commanded alpha;
   the tween, the pose gate and the face-found state all sit between that and
   the pixels. Under a multiplicative control law the realized dose would also
   depend on what the participant is doing, which means the nominal preset is
   not the independent variable.

A fifth, cosmetic rather than a validity issue: the displacement field peaks at
about 1.26 half-mouth-widths from the mouth centre — out on the cheek — where it
is **1.28x stronger than at the lip corner it is meant to be moving**. This is
consistent across faces (CV 0.001), so it does not cause inequality, but it does
mean the warp drags cheek skin harder than lip.

---

## 10. Does the replacement actually fix it?

Everything above is about the current morph. This section is about the one built
to replace it. Both implementations were run over the same 100 faces, re-framed
to webcam geometry (1280x720, interpupillary distance 95 px), and the rendered
output was measured by a *separate* detector and differenced against a sham
render of the identical frame — so what is reported is what the warp achieved,
not what it intended.

**The dose, as a share of each person's own smile** (α = 1.9):

| | CV | P90/P10 | range |
|---|---|---|---|
| current | 0.232 | 1.79 | 0.081 – 0.222 (2.7x) |
| normalized | **0.111** | **1.29** | 0.084 – 0.163 (1.9x) |

A 52% reduction, and 0.111 is at the floor of what can be measured: re-detecting
the same warp on the same face recovers it with about 12% frame-to-frame spread,
so the residual is instrument noise rather than remaining inequality.

Two things worth being precise about:

- **The geometric dose becomes *more* variable** (CV 0.098 → 0.196), and that is
  correct. Scaling to each person's expressive range means an expressive person
  gets physically more displacement. The two cannot both be flat; that is the
  choice in §"What follows".
- **The appearance change also became more consistent** — MediaPipe's smile
  blendshape change went from CV 0.517 (P90/P10 5.9) to 0.396 (P90/P10 3.3) —
  and that improvement comes from the *warp shape*, not the amplitude
  normalization: the uncalibrated run shows it too. Driving the deformation
  along a smile axis fitted to 307 real neutral-to-smile pairs produces a more
  uniform appearance change than the hand-tuned field does.

**Same face, different camera.** Ten faces, each rendered through seven camera
conditions that are exact image operations — distance 0.72x to 1.35x, framing
shifted down 18% of frame height, roll -15 to +25 degrees. A face that differs in
nothing but how it sits should get an identical dose:

| | within-face CV of delivered dose | induced L/R asymmetry, range |
|---|---|---|
| current | 0.158 | 0.061 |
| normalized | **0.113** | **0.021** |

The asymmetry number is the more interesting one. The production morph applies
its displacement along *image* axes, so a tilted head gets one corner sliding
along the lip line while the other lifts across it — and the app's own classifier
reads left/right asymmetry as a **dominance** smile. The current implementation
manufactures about three times as much of that artifact across these conditions
as the normalized one does.

Neither is flat under strong roll: at +25 degrees the current morph delivers 69%
of its frontal dose and the normalized one 85%. Some of that residual is the
detector itself getting less accurate on a rolled image — it affects both — so
85% is closer to a floor than to a failure.

## 11. The "strong" preset is past what an image warp can render

Local area stretch in the rendered output, measured per frame:

| preset | median stretch | frames over 2x | over 4x | worst |
|---|---|---|---|---|
| α = 1.35 (subtle) | 1.45x | 0/100 | 0/100 | 1.85x |
| α = 1.9 (strong) | 3.08x | 93/100 | 24/100 | 9.3x |

That is not the new warp being clumsy. Measured on the *commanded deformation*
rather than the pixels — strain between neighbouring lip landmarks, which is
warp-independent:

| | strain |
|---|---|
| current morph, α = 1.9 | 0.29 |
| normalized, α = 1.9 | 1.08 |
| **a real closed-mouth smile** | **0.65** |
| **a real toothy smile** | **1.30** |

So the normalized morph at the strong preset asks the skin to deform by roughly
what a real toothy smile does — on a closed mouth, in 2D, with no new pixels to
work with. The current morph asks for 0.29, which is *less* than any real smile:
it under-deforms locally, which is consistent with it reading as a smear rather
than an expression.

**Recommendation:** the strong preset should be about **α = 1.5**, not 1.9. That
is where the commanded deformation matches a real closed-mouth smile (strain
0.65) and where rendered stretch stays near 2x. At 1.9 the manipulation is
larger than most people's own closed-mouth smile, which is both why the RAs
called it uncanny and why it stretches texture visibly.

A caveat on all of this: real skin is elastic and self-shadows, an image warp
only moves texture. Matching the strain of a real smile does not guarantee it
*looks* like one. That is a question for eyes on the demo, and ultimately for
human ratings.

---

## What follows

- Equalizing the *geometry* is worth doing — head turn is a genuine confound and
  roll costs 12% — but it is not what Randy noticed. On its own it left the
  cross-face dispersion unchanged (§10). What it did buy is a more consistent
  *appearance* change (blendshape P90/P10 5.9 → 3.3) and a warp whose shape comes
  from 307 real smiles rather than a hand-tuned formula.
- Equalizing the *dose relative to each person's expressive range* is the real
  job. Measured, it halves the spread (CV 0.232 → 0.111, §10), which is as far
  as the measurement can resolve. It **requires observing the person smile
  several times**; there is no way around that, because the face itself does not
  tell you.
- So the design question for Randy and Paula is not really an algorithm
  question. It is: **do we want the manipulation scaled to each person's own
  expressive range, and if so, is the study willing to spend a few minutes of
  conversation calibrating before the manipulation starts?** Both answers are
  defensible; they lead to different studies.

## Is there newer technology that would just solve this?

Asked, researched, and the answer is no — not on this hardware, and not for this
problem.

**Neural portrait reenactment** (LivePortrait and its successors) does solve the
cross-identity retargeting problem properly: it has explicit modules that map a
desired scalar — how open should these eyes be — into a latent change for *this*
face, which is exactly the shape of what we want. It runs at about 13 ms, on an
RTX 4090. There is no NVIDIA GPU on this machine or on the lab's, and identity
preservation in a covert manipulation is a risk the lab would have to argue about
with the IRB rather than a free win. Documented, not built.

**3D morphable models** (FLAME and relatives) give identity-normalized
expression coefficients by construction, which is the property we want. The
obstacle is not the fit, it is the render: putting a photorealistic face back on
screen with consistent skin, teeth and lighting is the hard half, and a bad
render is far more detectable than a mild warp.

**What did help, and is not new at all:** the deformation is now defined by 307
real neutral-to-smile pairs instead of a formula, and the warp is Moving Least
Squares (Schaefer et al. 2006) — the same method the original DuckSoup's own
plugin uses. Mozza encodes its deformation as barycentric coordinates in the
live face's own landmark triangles, which gets scale and rotation invariance for
free; that idea is good and is why this implementation works in a canonical
frame. What Mozza does not have, and what this adds, is any notion of how big the
*particular person's* smile is — its template is one person's smile applied to
everyone.

## The validation this is missing, and how to run it

Every number here is geometry or detector output. The claim the lab actually
cares about — that the manipulation *reads* as equally strong on everyone — has
not been tested, and cannot be by any of this.

It is a small study. Per identity, render three clips from the same source
footage: unmorphed, current at the chosen preset, normalized at the same preset.
Show each rater one condition per identity, between-subjects within identity so
nobody can contrast conditions on the same face. Ask "how much is this person
smiling", 0-100. The outcome is the per-identity treated-minus-sham difference,
and the quantity to compare is the **spread of that difference across
identities** — the same statistic as §10, but with people as the instrument.

Rough power: rating SD is typically around 15, so ~15 raters per
identity-condition puts the standard error of one identity's difference near
5.5. With 24 identities that estimates the spread of true differences to about
±20% relative, which is enough to tell CV 0.11 from CV 0.23 but not enough to
resolve small differences between two good versions.

Worth running the detectability question in the same session: same person, same
segment, sham versus treated, counterbalanced, "which of these is digitally
altered?" That is the number a deception study actually needs, and it should be
preregistered for the strong preset rather than assumed to be zero.

## Limitations — read before quoting any of this

- **CFD smiles are posed stills, not conversation.** Part of the closed-vs-toothy
  disagreement is two different deliberate acts rather than measurement noise,
  which makes the reliability of 0.56 a *lower* bound and the sample counts in §5
  an *upper* bound. Re-check on video before committing to a calibration design.
- **No perceptual validation yet.** Everything here is geometry. Geometric
  amplitude is not the same thing as perceived smile intensity, and the
  literature is clear that equal physical change is not equal perceived change
  across identities. Human ratings are the only ground truth for that claim, and
  none have been collected.
- **The pose sweep rotates a landmark cloud and re-projects it.** For a pure
  camera rotation that is exact; for head rotation it ignores self-occlusion and
  shading, so nothing here is claimed beyond about ±30°.
- **The population prior is fitted on posed smiles.** The live calibrator would
  observe conversational ones, which are smaller, so the prior and the
  observations are on different scales and shrinkage biases the early estimate.
  This does not touch any result above (amplitudes were supplied directly) but
  it has to be fixed before per-person calibration runs in a study.
- **CFD is posed, studio-lit, frontal, high-resolution.** Lab webcam conditions
  are none of those. The geometric findings should transfer; the landmark noise
  will not.
- Amplitude is measured as total canonical shape change over 40 lip landmarks.
  Two alternatives (projection on the population axis, mouth-corner travel alone)
  give reliabilities within 0.04 of each other, so the conclusions do not hinge
  on that choice.
