// A small strip chart. No chart library: three series over a fixed time window
// is about sixty lines of canvas, and a dependency that renders it would be
// larger than the algorithm being demonstrated.

export interface Series {
  label: string
  color: string
  dashed?: boolean
  data: Array<{ t: number; v: number }>
}

export class StripChart {
  private ctx: CanvasRenderingContext2D

  constructor(
    private canvas: HTMLCanvasElement,
    private windowMs = 20000,
  ) {
    const c = canvas.getContext('2d')
    if (!c) throw new Error('2D context unavailable')
    this.ctx = c
  }

  draw(series: Series[], now: number): void {
    const { ctx } = this
    const w = this.canvas.width
    const h = this.canvas.height
    const padL = 52, padR = 8, padT = 10, padB = 20

    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, w, h)

    // Symmetric range around zero, with a floor so an idle chart is not noise
    // blown up to full scale.
    let peak = 0.12
    for (const s of series) {
      for (const p of s.data) {
        if (now - p.t <= this.windowMs) peak = Math.max(peak, Math.abs(p.v))
      }
    }
    peak = Math.ceil(peak * 10) / 10
    const y = (v: number) => padT + (1 - (v + peak) / (2 * peak)) * (h - padT - padB)
    const x = (t: number) => padL + (1 - (now - t) / this.windowMs) * (w - padL - padR)

    ctx.strokeStyle = '#2c3140'
    ctx.fillStyle = '#9aa3b8'
    ctx.font = '18px ui-monospace, Menlo, Consolas, monospace'
    ctx.textAlign = 'right'
    ctx.textBaseline = 'middle'
    ctx.lineWidth = 1
    for (const v of [-peak, -peak / 2, 0, peak / 2, peak]) {
      const yy = Math.round(y(v)) + 0.5
      ctx.beginPath()
      ctx.moveTo(padL, yy)
      ctx.lineTo(w - padR, yy)
      ctx.strokeStyle = v === 0 ? '#3d4456' : '#232836'
      ctx.stroke()
      ctx.fillText(v.toFixed(2), padL - 8, yy)
    }

    for (const s of series) {
      const pts = s.data.filter((p) => now - p.t <= this.windowMs)
      if (pts.length < 2) continue
      ctx.beginPath()
      ctx.strokeStyle = s.color
      ctx.lineWidth = s.dashed ? 2 : 3
      ctx.setLineDash(s.dashed ? [6, 6] : [])
      let started = false
      for (const p of pts) {
        const px = x(p.t), py = y(p.v)
        if (!Number.isFinite(py)) { started = false; continue }
        if (!started) { ctx.moveTo(px, py); started = true } else ctx.lineTo(px, py)
      }
      ctx.stroke()
    }
    ctx.setLineDash([])
  }
}
