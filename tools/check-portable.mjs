// src/algo/ has to stay copy-pasteable into the production app.
//
// That only holds if it imports nothing outside itself except the MediaPipe
// package the app already depends on. It is easy to break by accident — one
// `import { something } from '../demo/util'` and the port stops being a
// directory copy. So it is enforced rather than remembered.

import { readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const ALGO = resolve(here, '..', 'src', 'algo')
const ALLOWED_EXTERNAL = new Set(['@mediapipe/tasks-vision'])

const IMPORT_RE = /(?:^|\n)\s*(?:import|export)[\s\S]*?from\s+['"]([^'"]+)['"]/g

function walk(dir) {
  const out = []
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) out.push(...walk(p))
    else if (p.endsWith('.ts')) out.push(p)
  }
  return out
}

let bad = 0
for (const file of walk(ALGO)) {
  const src = readFileSync(file, 'utf8')
  for (const m of src.matchAll(IMPORT_RE)) {
    const spec = m[1]
    if (spec.startsWith('./') || spec.startsWith('../')) {
      const resolved = resolve(dirname(file), spec)
      if (!resolved.startsWith(ALGO)) {
        console.error(`${file}: imports outside src/algo -> ${spec}`)
        bad++
      }
      continue
    }
    if (!ALLOWED_EXTERNAL.has(spec)) {
      console.error(`${file}: imports a package the app may not have -> ${spec}`)
      bad++
    }
  }
}

if (bad) {
  console.error(`\n${bad} portability violation(s): src/algo must be self-contained.`)
  process.exit(1)
}
console.log('portable: src/algo depends only on itself and @mediapipe/tasks-vision')
