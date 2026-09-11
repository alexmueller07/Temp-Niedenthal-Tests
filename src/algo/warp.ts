// Landmark-driven image warp: Moving Least Squares, rigid variant
// (Schaefer, McPhail & Warren, SIGGRAPH 2006). Ported from the reference C++ in
// ducksouplab/mozza (lib/imgwarp/imgwarp_mls_rigid.cpp), which is what the
// original DuckSoup uses.
//
// Why not keep the production approach? That one evaluates an analytic
// displacement formula on a 12x8 grid laid over an axis-aligned box around the
// lips. Three consequences:
//   - the field's shape depends on the box, and the box depends on lip height
//     and on whether the box hits the edge of the frame;
//   - the field peaks about 1.26 half-mouth-widths out from the mouth centre —
//     on the cheek — where it is 1.28x stronger than at the lip corner it is
//     meant to be moving;
//   - the displacement direction is fixed in *image* axes, so a tilted head
//     gets one corner sliding along the lip line and the other lifting across
//     it, manufacturing left/right asymmetry.
// Here the deformation is specified by where the landmarks should end up, and
// the warp is whatever smoothly satisfies that. The shape of the field is a
// consequence of anatomy rather than of a bounding box.
//
// MLS-rigid specifically (rather than affine or similarity) because it is the
// variant that locally forbids shear and non-uniform scale, which is what keeps
// skin texture from smearing.

export interface Vec2 { x: number; y: number }

export interface WarpControl {
  /** Where a point is in the source image. */
  src: Vec2[]
  /** Where it should appear in the output. */
  dst: Vec2[]
}

export interface WarpQuality {
  /** Largest control-point displacement, pixels. */
  maxTravelPx: number
  /** Worst local area change over the grid; 1 is undistorted. */
  maxExpansion: number
  minExpansion: number
  /** True if any cell folded over itself. */
  folded: boolean
}

const ALPHA = 1.0        // MLS weight exponent; 1 gives w = 1/d^2
const EPS = 1e-8

/**
 * Evaluate the MLS-rigid map at `nodes`, in place.
 *
 * Maps from the space of `p` into the space of `q`. For rendering we want the
 * *backward* map — for each output pixel, where to read from — so callers pass
 * p = destination landmark positions and q = source landmark positions. Getting
 * this the wrong way round produces a warp that moves the face the opposite way,
 * which is subtle enough to survive a casual look, so there is a unit test.
 */
export function mlsRigid(
  nodes: Float64Array,      // [x0,y0,x1,y1,...] evaluated in place
  p: Float64Array,          // control points, domain
  q: Float64Array,          // control points, range
  count: number,
): void {
  const nNodes = nodes.length >> 1
  const w = new Float64Array(count)

  for (let n = 0; n < nNodes; n++) {
    const vx = nodes[2 * n], vy = nodes[2 * n + 1]

    // Weights, with the exact-hit case handled separately: at a control point
    // the weight is infinite and the map is simply that point's image.
    let sw = 0, exact = -1
    for (let i = 0; i < count; i++) {
      const dx = vx - p[2 * i], dy = vy - p[2 * i + 1]
      const d2 = dx * dx + dy * dy
      if (d2 < EPS) { exact = i; break }
      const wi = ALPHA === 1 ? 1 / d2 : Math.pow(d2, -ALPHA)
      w[i] = wi
      sw += wi
    }
    if (exact >= 0) {
      nodes[2 * n] = q[2 * exact]
      nodes[2 * n + 1] = q[2 * exact + 1]
      continue
    }
    if (sw < EPS) continue

    let psx = 0, psy = 0, qsx = 0, qsy = 0
    for (let i = 0; i < count; i++) {
      psx += w[i] * p[2 * i]; psy += w[i] * p[2 * i + 1]
      qsx += w[i] * q[2 * i]; qsy += w[i] * q[2 * i + 1]
    }
    psx /= sw; psy /= sw; qsx /= sw; qsy /= sw

    // mu_r: the rigid normalisation. Without it the map would be a similarity
    // (free to scale locally); dividing by it removes the scale and leaves a
    // rotation, which is what keeps texture from stretching.
    let s1 = 0, s2 = 0
    for (let i = 0; i < count; i++) {
      const pax = p[2 * i] - psx, pay = p[2 * i + 1] - psy
      const qax = q[2 * i] - qsx, qay = q[2 * i + 1] - qsy
      s1 += w[i] * (qax * pax + qay * pay)
      s2 += w[i] * (qax * -pay + qay * pax)
    }
    const muR = Math.hypot(s1, s2)
    if (muR < EPS) {
      nodes[2 * n] = qsx
      nodes[2 * n + 1] = qsy
      continue
    }

    const cvx = vx - psx, cvy = vy - psy
    const cjx = -cvy, cjy = cvx

    let ox = 0, oy = 0
    for (let i = 0; i < count; i++) {
      const pax = p[2 * i] - psx, pay = p[2 * i + 1] - psy
      const pjx = -pay, pjy = pax
      const qax = q[2 * i] - qsx, qay = q[2 * i + 1] - qsy
      const a = pax * cvx + pay * cvy
      const b = pjx * cvx + pjy * cvy
      const c = pax * cjx + pay * cjy
      const d = pjx * cjx + pjy * cjy
      const k = w[i] / muR
      ox += k * (a * qax - b * qay)
      oy += k * (-c * qax + d * qay)
    }
    nodes[2 * n] = ox + qsx
    nodes[2 * n + 1] = oy + qsy
  }
}

export interface WarpRoi { x: number; y: number; w: number; h: number }

/**
 * Warp a region of `srcImage` into `ctx`, driven by landmark correspondences.
 *
 * Control points are the caller's landmarks plus a ring of pinned points around
 * the region boundary. The ring is what makes the warp blend seamlessly into the
 * untouched frame: it holds the boundary fixed instead of relying on a window
 * function whose shape depends on the box.
 */
export function warpRegion(
  ctx: CanvasRenderingContext2D,
  srcImage: CanvasImageSource,
  control: WarpControl,
  roi: WarpRoi,
  gridPx: number,
  frameW: number,
  frameH: number,
): WarpQuality {
  // Drop control points that sit on top of an earlier one. Two coincident
  // points with different targets ask the warp for infinite local strain, and
  // landmark sets have degenerate pairs more often than one expects.
  const keep: number[] = []
  const minSep = Math.max(1.5, gridPx * 0.5)
  for (let i = 0; i < control.src.length; i++) {
    let ok = true
    for (const j of keep) {
      if (Math.hypot(control.src[i].x - control.src[j].x,
                     control.src[i].y - control.src[j].y) < minSep) { ok = false; break }
    }
    if (ok) keep.push(i)
  }
  const srcPts = keep.map((i) => control.src[i])
  const dstPts = keep.map((i) => control.dst[i])

  const m = srcPts.length
  const ringStep = Math.max(gridPx * 2, 24)
  const ring: Vec2[] = []
  for (let x = roi.x; x <= roi.x + roi.w + 1; x += ringStep) {
    ring.push({ x, y: roi.y }, { x, y: roi.y + roi.h })
  }
  for (let y = roi.y + ringStep; y < roi.y + roi.h - ringStep; y += ringStep) {
    ring.push({ x: roi.x, y }, { x: roi.x + roi.w, y })
  }

  const count = m + ring.length
  // Backward map: domain = where things end up, range = where to read from.
  const p = new Float64Array(2 * count)
  const q = new Float64Array(2 * count)
  let maxTravel = 0
  for (let i = 0; i < m; i++) {
    p[2 * i] = dstPts[i].x; p[2 * i + 1] = dstPts[i].y
    q[2 * i] = srcPts[i].x; q[2 * i + 1] = srcPts[i].y
    maxTravel = Math.max(maxTravel,
      Math.hypot(dstPts[i].x - srcPts[i].x, dstPts[i].y - srcPts[i].y))
  }
  for (let j = 0; j < ring.length; j++) {
    const i = m + j
    p[2 * i] = ring[j].x; p[2 * i + 1] = ring[j].y
    q[2 * i] = ring[j].x; q[2 * i + 1] = ring[j].y
  }

  const cols = Math.max(2, Math.ceil(roi.w / gridPx))
  const rows = Math.max(2, Math.ceil(roi.h / gridPx))
  const nNodes = (cols + 1) * (rows + 1)

  const dstNodes = new Float64Array(2 * nNodes)
  for (let r = 0; r <= rows; r++) {
    for (let c = 0; c <= cols; c++) {
      const i = r * (cols + 1) + c
      dstNodes[2 * i] = roi.x + (c / cols) * roi.w
      dstNodes[2 * i + 1] = roi.y + (r / rows) * roi.h
    }
  }
  const srcNodes = Float64Array.from(dstNodes)
  mlsRigid(srcNodes, p, q, count)

  // Quality: how much each cell's area changed between source and destination.
  // A cell that expanded a lot is stretched texture; a negative area means the
  // warp folded over itself. This is checked every frame and reported, rather
  // than trusted.
  let maxExp = 0, minExp = Infinity, folded = false
  const cellW = roi.w / cols, cellH = roi.h / rows
  const dstArea = cellW * cellH
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const a = r * (cols + 1) + c
      const b = a + 1
      const d = a + cols + 1
      const ar = triArea(srcNodes, a, b, d) * 2
      if (ar <= 0) { folded = true; continue }
      const exp = dstArea / ar
      maxExp = Math.max(maxExp, exp)
      minExp = Math.min(minExp, exp)
    }
  }

  const idx = (r: number, c: number) => r * (cols + 1) + c
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const a = idx(r, c), b = idx(r, c + 1)
      const d = idx(r + 1, c), e = idx(r + 1, c + 1)
      drawTriangle(ctx, srcImage, srcNodes, dstNodes, a, b, d, frameW, frameH)
      drawTriangle(ctx, srcImage, srcNodes, dstNodes, b, e, d, frameW, frameH)
    }
  }

  return {
    maxTravelPx: maxTravel,
    maxExpansion: maxExp,
    minExpansion: Number.isFinite(minExp) ? minExp : 1,
    folded,
  }
}

function triArea(n: Float64Array, a: number, b: number, c: number): number {
  const ax = n[2 * a], ay = n[2 * a + 1]
  const bx = n[2 * b], by = n[2 * b + 1]
  const cx = n[2 * c], cy = n[2 * c + 1]
  return Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2
}

/** Affine-map one source triangle onto its destination triangle and blit it. */
function drawTriangle(
  ctx: CanvasRenderingContext2D,
  img: CanvasImageSource,
  srcN: Float64Array,
  dstN: Float64Array,
  i0: number, i1: number, i2: number,
  frameW: number, frameH: number,
): void {
  const s0x = srcN[2 * i0], s0y = srcN[2 * i0 + 1]
  const s1x = srcN[2 * i1], s1y = srcN[2 * i1 + 1]
  const s2x = srcN[2 * i2], s2y = srcN[2 * i2 + 1]
  const d0x = dstN[2 * i0], d0y = dstN[2 * i0 + 1]
  const d1x = dstN[2 * i1], d1y = dstN[2 * i1 + 1]
  const d2x = dstN[2 * i2], d2y = dstN[2 * i2 + 1]

  const denom = s0x * (s2y - s1y) - s1x * s2y + s2x * s1y + (s1x - s2x) * s0y
  if (Math.abs(denom) < 1e-6) return

  ctx.save()
  // Grow the clip a fraction of a pixel so neighbouring triangles overlap
  // slightly, otherwise antialiasing leaves hairline seams between them.
  const cx = (d0x + d1x + d2x) / 3, cy = (d0y + d1y + d2y) / 3
  const g = 0.5
  ctx.beginPath()
  ctx.moveTo(d0x + Math.sign(d0x - cx) * g, d0y + Math.sign(d0y - cy) * g)
  ctx.lineTo(d1x + Math.sign(d1x - cx) * g, d1y + Math.sign(d1y - cy) * g)
  ctx.lineTo(d2x + Math.sign(d2x - cx) * g, d2y + Math.sign(d2y - cy) * g)
  ctx.closePath()
  ctx.clip()

  const m11 = -(s0y * (d2x - d1x) - s1y * d2x + s2y * d1x + (s1y - s2y) * d0x) / denom
  const m12 = (s1y * d2y + s0y * (d1y - d2y) - s2y * d1y + (s2y - s1y) * d0y) / denom
  const m21 = (s0x * (d2x - d1x) - s1x * d2x + s2x * d1x + (s1x - s2x) * d0x) / denom
  const m22 = -(s1x * d2y + s0x * (d1y - d2y) - s2x * d1y + (s2x - s1x) * d0y) / denom
  const dx = (s0x * (s2y * d1x - s1y * d2x) + s0y * (s1x * d2x - s2x * d1x)
    + (s2x * s1y - s1x * s2y) * d0x) / denom
  const dy = (s0x * (s2y * d1y - s1y * d2y) + s0y * (s1x * d2y - s2x * d1y)
    + (s2x * s1y - s1x * s2y) * d0y) / denom

  ctx.setTransform(m11, m12, m21, m22, dx, dy)
  ctx.drawImage(img, 0, 0, frameW, frameH)
  ctx.restore()
}
