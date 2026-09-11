# Smile-morph normalization — proof of concept

Making the video-call app's smile manipulation land equally on every
participant. Standalone: the production app
(`niedenthal-ducksoup-research-video-conferencing`) is not touched.

Start with **[FINDINGS.md](FINDINGS.md)** — it has the measurements and the
recommendation. This file is how to run things.

---

## The short version

Randy noticed the morph reads as a different strength on different people. It
does, but not for the reason it looks like. Measured over 823 faces, the warp
already delivers an almost perfectly constant displacement — CV 0.4% in its own
unit. What varies is that **people's own smiles differ about twofold**, so the
same displacement is a very different share of each person's expressive range.
At the "strong" preset the morph adds roughly 58% of a median person's full
smile, about 90% of a reserved person's and 40% of an expressive person's.

Nothing about a face predicts how expressive its owner is — 37 anatomical and
perceptual predictors give a leave-one-out R² of about zero — so it has to be
observed. This PoC drives the morph in *fractions of a smile* rather than
pixels, and learns each person's smile size from the raw camera frames.

It also fixes four validity problems found while reading the current morph
(sham taking a different code path, dose confounded with head turn, smile and
frown not on a common scale, realized dose never logged). Those matter for
studies already running and are independent of everything else here.

---

## Running it

```bash
npm install
npm run assets          # vendor MediaPipe from node_modules + the app's model
npm run build
.venv/Scripts/python.exe tools/serve.py     # then open http://127.0.0.1:8900/demo.html
```

The demo needs a camera and a person, so it is the one step to run by hand.
Everything else is scripted.

### Python environment

A local venv, deliberately: MediaPipe pins `numpy<2` and the global Python here
runs other projects that need numpy 2.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install mediapipe==0.10.21 opencv-python numpy \
    openpyxl pandas scipy matplotlib playwright
```

### The offline analysis (stage 0)

Answers the design questions without rendering anything. Needs the Chicago Face
Database at `C:\Users\amuel\Downloads\cfd`.

```bash
.venv/Scripts/python.exe tools/cfd_extract.py      # landmarks -> local cache (~4 min)
.venv/Scripts/python.exe tools/cfd_analysis.py     # the current morph, quantified
.venv/Scripts/python.exe tools/s0_amplitude.py     # is smile size a trait? predictable?
.venv/Scripts/python.exe tools/export_model.py     # corpus geometry -> src/algo/faceModel.gen.ts
```

### The batch experiments

```bash
.venv/Scripts/python.exe tools/build_corpus.py --n 160 --paired-only
.venv/Scripts/python.exe tools/export_amplitudes.py
.venv/Scripts/python.exe tools/export_rests.py
bash scratchpad/run_all.sh                          # ~90 min
.venv/Scripts/python.exe tools/analyze.py --run <runId>
```

`--headed` parks a real browser window off-screen and is 2.7× faster than
headless, because headless falls back to a software GPU.

---

## Layout

```
src/algo/     the algorithm — the only copy, and the thing that ships
src/demo/     the live comparison page
src/harness/  the batch measurement page
tools/        analysis, corpus building, the Playwright runner
artifacts/    results that are worth keeping
```

`src/algo/` may import only from itself and `@mediapipe/tasks-vision`, enforced
by `npm run typecheck`. That is what makes the port back a directory copy —
see [PORTING.md](PORTING.md).

---

## Things that will bite you

**Timings from the batch harness are meaningless.** Headless Chromium uses a
software GPU and the canvas work runs about an order of magnitude slow. Every
performance number comes from the demo page on real hardware, and the final one
from the Electron app. There is a note to this effect in the runner; please
leave it there.

**The corpus lives outside the repo**, at `C:\lab-corpus`. Two reasons: the
Chicago Face Database licence forbids redistributing images and this repo is
public, and the project root is OneDrive-synced, where a few thousand JPEGs
cause sync storms. Nothing under `artifacts/cache/`, `artifacts/runs/` or any
image format is committed.

**MediaPipe VIDEO mode is stateful.** It reuses the previous frame's region of
interest rather than re-detecting. Feeding a folder of unrelated photographs
through it lets image N inherit image N−1's region. Anything scoring stills, or
probing a morphed output, uses a separate IMAGE-mode instance.

**`setAlpha` sets a tween target, not a value.** Rendering one frame after
asking for alpha 1 does not give a neutral frame, it gives one a quarter of the
way back. The batch harness settles 18 frames per cell for this reason; the demo
differences against the raw frame instead.

**The Playwright-bundled ffmpeg cannot decode anything in this project** — it is
built `--disable-everything` with no mp4 demuxer and no H.264 decoder, and its
only rotation is 90° steps. Use `cv2` for anything offline and do view geometry
in the page.

---

## What this does not establish

Everything measured here is geometry and detector output. Geometric amplitude is
not perceived smile intensity, and equal physical change is known not to be
equal perceived change across identities. The only ground truth for that is human
ratings, and none have been collected. The FINDINGS limitations section is not
boilerplate — read it before quoting a number.
