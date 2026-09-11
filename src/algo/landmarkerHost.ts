// Creating and configuring the MediaPipe face landmarker.
//
// Two differences from the production setup worth knowing about:
//
//   1. `outputFacialTransformationMatrixes` is enabled. Production does not use
//      it; it estimates yaw instead from the ratio of two screen-x distances
//      between the nose tip and the face-oval edges — which head roll
//      contaminates, because those are screen distances.
//
//   2. There is a CPU fallback. Production requests the GPU delegate with no
//      alternative, so on a machine where GPU delegate creation fails the whole
//      morph is disabled rather than running slower. The asset loader already
//      falls back local -> CDN; the delegate does not.

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

const LOCAL_WASM_BASE = '/vendor/wasm'
const LOCAL_MODEL_URL = '/models/face_landmarker.task'
const CDN_WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
const CDN_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

export type RunMode = 'VIDEO' | 'IMAGE'

export interface LandmarkerInfo {
  landmarker: FaceLandmarker
  delegate: 'GPU' | 'CPU'
  assets: 'local' | 'cdn'
}

/**
 * Create a landmarker, trying local assets then the CDN, and the GPU delegate
 * then the CPU one.
 *
 * `mode` matters more than it looks. VIDEO mode carries state between calls: it
 * reuses the previous frame's region of interest instead of re-detecting. That
 * is what makes it fast on a live stream, and exactly what makes it wrong for a
 * folder of unrelated photographs, where image N would inherit image N-1's ROI.
 * Anything that scores still images — or that probes the morphed output — needs
 * its own IMAGE-mode instance.
 */
export async function createLandmarker(
  mode: RunMode = 'VIDEO',
  blendshapes = true,
): Promise<LandmarkerInfo> {
  const attempts: Array<{ wasm: string; model: string; assets: 'local' | 'cdn' }> = [
    { wasm: LOCAL_WASM_BASE, model: LOCAL_MODEL_URL, assets: 'local' },
    { wasm: CDN_WASM_BASE, model: CDN_MODEL_URL, assets: 'cdn' },
  ]

  let lastErr: unknown = null
  for (const a of attempts) {
    for (const delegate of ['GPU', 'CPU'] as const) {
      try {
        const fileset = await FilesetResolver.forVisionTasks(a.wasm)
        const landmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: a.model, delegate },
          runningMode: mode,
          numFaces: 1,
          outputFaceBlendshapes: blendshapes,
          outputFacialTransformationMatrixes: true,
        })
        return { landmarker, delegate, assets: a.assets }
      } catch (err) {
        lastErr = err
      }
    }
  }
  throw new Error(`could not create a face landmarker: ${String(lastErr)}`)
}
