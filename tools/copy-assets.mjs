// Vendor the MediaPipe runtime into public/ so the demo and harness work
// offline and at the exact version the production app ships (0.10.18 — the
// WASM and the JS must match, so this is pinned, not a caret range).
//
// The .task model is copied from the app's own vendored copy rather than
// downloaded, so both projects are provably running the same weights.

import { mkdirSync, copyFileSync, existsSync, readdirSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, '..')

const WASM_SRC = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm')
const WASM_DST = join(root, 'public', 'vendor', 'wasm')
const MODEL_SRC = resolve(
  root, '..', 'niedenthal-ducksoup-research-video-conferencing',
  'renderer', 'public', 'mediapipe', 'face_landmarker.task',
)
const MODEL_DST = join(root, 'public', 'models', 'face_landmarker.task')

function copyDir(src, dst) {
  mkdirSync(dst, { recursive: true })
  let n = 0
  for (const name of readdirSync(src)) {
    const s = join(src, name)
    if (statSync(s).isDirectory()) { n += copyDir(s, join(dst, name)); continue }
    copyFileSync(s, join(dst, name))
    n++
  }
  return n
}

if (!existsSync(WASM_SRC)) {
  console.error(`missing ${WASM_SRC}; run npm install first`)
  process.exit(1)
}
const n = copyDir(WASM_SRC, WASM_DST)
console.log(`wasm: ${n} files -> public/vendor/wasm`)

if (existsSync(MODEL_SRC)) {
  mkdirSync(dirname(MODEL_DST), { recursive: true })
  copyFileSync(MODEL_SRC, MODEL_DST)
  const mb = (statSync(MODEL_DST).size / 1e6).toFixed(2)
  console.log(`model: face_landmarker.task (${mb} MB) -> public/models`)
} else {
  console.warn(`model not found at ${MODEL_SRC}`)
  console.warn('the page will fall back to the MediaPipe CDN, which needs a network')
}
