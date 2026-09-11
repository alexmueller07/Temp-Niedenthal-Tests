// Simulating a different camera, in the page.
//
// Serves two jobs with one implementation: the demo's "what if they sat lower /
// closer / tilted their head" controls, and the batch harness's pose ablation.
// Applying it to the input frame *before* either morph implementation sees it
// means both get byte-identical input, so the A/B stays clean.
//
// For a pure camera rotation about its optical centre the mapping is an exact
// homography, H = K R K^-1 — exact for any scene at any depth, no planarity
// assumption. That is worth stating when showing results: rolling and pitching
// the virtual camera is a real simulation, not an approximation.
//
// What it does NOT simulate is the head rotating relative to the camera, which
// additionally changes self-occlusion and shading. So claims are kept inside
// about +/-25 degrees, where the difference is small.

export interface ViewParams {
  /** Camera rotation, degrees. Pitch is the "how tall are they / where is the
   *  laptop" axis. */
  yawDeg: number
  pitchDeg: number
  rollDeg: number
  /** 1 = unchanged. Larger = sitting closer. */
  scale: number
  /** Translation as a fraction of frame size. */
  dxFrac: number
  dyFrac: number
}

export const IDENTITY_VIEW: ViewParams = {
  yawDeg: 0, pitchDeg: 0, rollDeg: 0, scale: 1, dxFrac: 0, dyFrac: 0,
}

export function isIdentity(v: ViewParams): boolean {
  return v.yawDeg === 0 && v.pitchDeg === 0 && v.rollDeg === 0
    && v.scale === 1 && v.dxFrac === 0 && v.dyFrac === 0
}

const VERT = `#version 300 es
in vec2 aPos;
in vec3 aTex;          // (u*w, v*w, w)
out vec3 vTex;
void main() {
  vTex = aTex;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`

const FRAG = `#version 300 es
precision highp float;
in vec3 vTex;
uniform sampler2D uTex;
out vec4 outColor;
void main() {
  vec2 uv = vTex.xy / vTex.z;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    outColor = vec4(0.0, 0.0, 0.0, 1.0);
  } else {
    outColor = texture(uTex, uv);
  }
}`

export class ViewTransform {
  private canvas: HTMLCanvasElement
  private gl: WebGL2RenderingContext
  private program: WebGLProgram
  private tex: WebGLTexture
  private posBuf: WebGLBuffer
  private texBuf: WebGLBuffer

  constructor() {
    this.canvas = document.createElement('canvas')
    const gl = this.canvas.getContext('webgl2', { preserveDrawingBuffer: true })
    if (!gl) throw new Error('WebGL2 unavailable — needed for the view simulation')
    this.gl = gl

    const vs = compile(gl, gl.VERTEX_SHADER, VERT)
    const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG)
    const p = gl.createProgram()!
    gl.attachShader(p, vs)
    gl.attachShader(p, fs)
    gl.linkProgram(p)
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(`link failed: ${gl.getProgramInfoLog(p)}`)
    }
    this.program = p

    this.tex = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, this.tex)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)

    this.posBuf = gl.createBuffer()!
    this.texBuf = gl.createBuffer()!
  }

  /** Render `source` through the virtual camera. Returns a canvas usable as an
   *  image source for the morph pipeline. */
  apply(source: CanvasImageSource, w: number, h: number, v: ViewParams): HTMLCanvasElement {
    const gl = this.gl
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w
      this.canvas.height = h
    }
    gl.viewport(0, 0, w, h)

    gl.bindTexture(gl.TEXTURE_2D, this.tex)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
      source as TexImageSource)

    // Inverse homography: for each output corner, where does it read from?
    const hInv = inverse3(homography(w, h, v))
    const corners: Array<[number, number]> = [[0, 0], [w, 0], [0, h], [w, h]]
    const pos: number[] = []
    const tex: number[] = []
    for (const [x, y] of corners) {
      // Clip space, y flipped because texture rows run top-down.
      pos.push((x / w) * 2 - 1, 1 - (y / h) * 2)
      const sx = hInv[0] * x + hInv[1] * y + hInv[2]
      const sy = hInv[3] * x + hInv[4] * y + hInv[5]
      const sw = hInv[6] * x + hInv[7] * y + hInv[8]
      // Pass (u*w, v*w, w) so the fragment shader can do a perspective-correct
      // divide; interpolating u,v directly would bend straight lines.
      tex.push((sx / w), (sy / h), sw)
    }
    // Normalise: the varyings must carry u*w and v*w, not u and v.
    for (let i = 0; i < 4; i++) {
      tex[3 * i] *= tex[3 * i + 2]
      tex[3 * i + 1] *= tex[3 * i + 2]
    }

    gl.useProgram(this.program)
    bindAttrib(gl, this.program, 'aPos', this.posBuf, new Float32Array(pos), 2)
    bindAttrib(gl, this.program, 'aTex', this.texBuf, new Float32Array(tex), 3)
    gl.uniform1i(gl.getUniformLocation(this.program, 'uTex'), 0)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.tex)

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
    return this.canvas
  }

  dispose() {
    const gl = this.gl
    gl.deleteTexture(this.tex)
    gl.deleteBuffer(this.posBuf)
    gl.deleteBuffer(this.texBuf)
    gl.deleteProgram(this.program)
  }
}

/** H = T * S * (K R K^-1), row-major 3x3. */
function homography(w: number, h: number, v: ViewParams): number[] {
  const f = 0.9 * w          // a plausible webcam focal length in pixels
  const cx = w / 2, cy = h / 2
  const k = [f, 0, cx, 0, f, cy, 0, 0, 1]
  const kInv = [1 / f, 0, -cx / f, 0, 1 / f, -cy / f, 0, 0, 1]

  const r = mul3(mul3(rotZ(v.rollDeg), rotX(v.pitchDeg)), rotY(v.yawDeg))
  let hm = mul3(mul3(k, r), kInv)

  // Scale and translate about the frame centre.
  const s = v.scale
  const st = [s, 0, cx * (1 - s) + v.dxFrac * w,
              0, s, cy * (1 - s) + v.dyFrac * h,
              0, 0, 1]
  hm = mul3(st, hm)
  return hm
}

const d2r = (d: number) => (d * Math.PI) / 180
const rotX = (d: number) => {
  const c = Math.cos(d2r(d)), s = Math.sin(d2r(d))
  return [1, 0, 0, 0, c, -s, 0, s, c]
}
const rotY = (d: number) => {
  const c = Math.cos(d2r(d)), s = Math.sin(d2r(d))
  return [c, 0, s, 0, 1, 0, -s, 0, c]
}
const rotZ = (d: number) => {
  const c = Math.cos(d2r(d)), s = Math.sin(d2r(d))
  return [c, -s, 0, s, c, 0, 0, 0, 1]
}

function mul3(a: number[], b: number[]): number[] {
  const o = new Array(9).fill(0)
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0
      for (let k = 0; k < 3; k++) s += a[3 * i + k] * b[3 * k + j]
      o[3 * i + j] = s
    }
  }
  return o
}

function inverse3(m: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = m
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
  if (Math.abs(det) < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1]
  return [
    (e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det,
    (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det,
    (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det,
  ]
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!
  gl.shaderSource(s, src)
  gl.compileShader(s)
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(`shader compile failed: ${gl.getShaderInfoLog(s)}`)
  }
  return s
}

function bindAttrib(
  gl: WebGL2RenderingContext, prog: WebGLProgram, name: string,
  buf: WebGLBuffer, data: Float32Array, size: number,
): void {
  const loc = gl.getAttribLocation(prog, name)
  gl.bindBuffer(gl.ARRAY_BUFFER, buf)
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW)
  gl.enableVertexAttribArray(loc)
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0)
}
