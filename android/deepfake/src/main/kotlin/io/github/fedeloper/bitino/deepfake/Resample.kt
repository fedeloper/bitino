package io.github.fedeloper.bitino.deepfake

import java.nio.ByteBuffer

/**
 * The detectors' preprocessing on any rectangle of an RGBA frame, as the authors' test loader does it with torchvision
 * on a PIL image: Resize(252) (shorter side to 252, longer side int(252·long/short), PIL bilinear with antialiasing,
 * each pass rounded to u8), CenterCrop(224) (offsets rounded half to even, like Python's round), then u8 / 255.
 * Only the 224×224 output window is computed.
 */
internal object Resample {
    private const val RESIZE = 252
    private const val CROP = 224
    private const val INV255 = 1f / 255f

    /** PIL's coefficient tables for [count] outputs starting at output index [first]: bounds (xmin, n) and weights. */
    internal class Coeffs(val xmin: IntArray, val n: IntArray, val w: Array<FloatArray>)

    internal fun coeffs(inSize: Int, outSize: Int, first: Int, count: Int): Coeffs {
        val scale = inSize.toDouble() / outSize
        val filterScale = maxOf(scale, 1.0)
        val support = filterScale            // bilinear (triangle) filter, support 1
        val xmin = IntArray(count)
        val nn = IntArray(count)
        val ws = Array(count) { FloatArray(0) }
        for (k in 0 until count) {
            val i = first + k
            val center = (i + 0.5) * scale
            var lo = (center - support + 0.5).toInt()
            if (lo < 0) lo = 0
            var hi = (center + support + 0.5).toInt()
            if (hi > inSize) hi = inSize
            val m = hi - lo
            val w = DoubleArray(m)
            var sum = 0.0
            for (x in 0 until m) {
                val t = kotlin.math.abs((x + lo - center + 0.5) / filterScale)
                w[x] = if (t < 1.0) 1.0 - t else 0.0
                sum += w[x]
            }
            xmin[k] = lo
            nn[k] = m
            ws[k] = FloatArray(m) { if (sum != 0.0) (w[it] / sum).toFloat() else 0f }
        }
        return Coeffs(xmin, nn, ws)
    }

    /** Python's round(v / 2.0) for a non-negative integer v (half to even). */
    internal fun halfEven(v: Int): Int {
        val f = v / 2
        return if (v % 2 == 0) f else if (f % 2 == 0) f else f + 1
    }

    /** Resized size (width, height) of a w×h image by torchvision's Resize(252). */
    internal fun resized(w: Int, h: Int): IntArray =
        if (w <= h) intArrayOf(RESIZE, (RESIZE.toLong() * h / w).toInt()) else intArrayOf((RESIZE.toLong() * w / h).toInt(), RESIZE)

    private var tmp = FloatArray(0)
    private var row = ByteArray(0)

    /** Writes the model input (224×224×3, NHWC) for the rectangle [x0, x1) × [y0, y1) of the frame. Not thread-safe. */
    fun toInput(buf: ByteBuffer, rowStride: Int, ps: Int, x0: Int, y0: Int, x1: Int, y1: Int, out: FloatArray) {
        val w = x1 - x0
        val h = y1 - y0
        require(w > 0 && h > 0) { "empty rectangle" }
        val (ow, oh) = resized(w, h).let { it[0] to it[1] }
        val left = halfEven(ow - CROP)
        val top = halfEven(oh - CROP)
        val cx = coeffs(w, ow, left, CROP)
        val cy = coeffs(h, oh, top, CROP)
        val ry0 = cy.xmin[0]
        val ry1 = cy.xmin[CROP - 1] + cy.n[CROP - 1]
        val rows = ry1 - ry0
        if (tmp.size < rows * CROP * 3) tmp = FloatArray(rows * CROP * 3)
        val need = (w - 1) * ps + 4
        if (row.size < need) row = ByteArray(need)
        val src = buf.duplicate()
        // horizontal pass over the source rows the vertical pass needs, rounded to u8 like PIL
        for (yy in 0 until rows) {
            src.position((y0 + ry0 + yy) * rowStride + x0 * ps)
            src.get(row, 0, need)
            val base = yy * CROP * 3
            for (k in 0 until CROP) {
                val lo = cx.xmin[k]
                val wk = cx.w[k]
                var sr = 0f; var sg = 0f; var sb = 0f
                var p = lo * ps
                for (t in wk.indices) {
                    val c = wk[t]
                    sr += c * (row[p].toInt() and 0xff)
                    sg += c * (row[p + 1].toInt() and 0xff)
                    sb += c * (row[p + 2].toInt() and 0xff)
                    p += ps
                }
                tmp[base + 3 * k] = clamp8(sr)
                tmp[base + 3 * k + 1] = clamp8(sg)
                tmp[base + 3 * k + 2] = clamp8(sb)
            }
        }
        // vertical pass
        for (k in 0 until CROP) {
            val lo = cy.xmin[k] - ry0
            val wk = cy.w[k]
            val ob = k * CROP * 3
            for (i in 0 until CROP * 3) {
                var s = 0f
                var p = lo * CROP * 3 + i
                for (t in wk.indices) {
                    s += wk[t] * tmp[p]
                    p += CROP * 3
                }
                out[ob + i] = clamp8(s) * INV255
            }
        }
    }

    private fun clamp8(v: Float): Float {
        val r = (v + 0.5f).toInt()
        return (if (r < 0) 0 else if (r > 255) 255 else r).toFloat()
    }
}
