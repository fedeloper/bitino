package io.github.fedeloper.bitino.deepfake

import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Where the analysed square lies and how the screen is captured.
 *
 * The detectors expect the authors' test preprocessing: Resize(252, antialiased) -> CenterCrop(224) -> u8 / 255,
 * NHWC RGB. The app analyses the largest centred square of the screen (side [side] screen pixels). The virtual
 * display that mirrors the screen is scaled so that this square is exactly 252·[k] pixels. A k×k box filter (the
 * antialiasing) then brings it to 252, and the centre 224×224 is the model input. The 14·k-pixel margin dropped by
 * the crop is where the overlay draws its outline of the square, so the outline never reaches the model.
 */
internal class Geometry(val screenW: Int, val screenH: Int, densityDpi: Int) {
    /** Side of the analysed square in screen pixels. */
    val side: Int = min(screenW, screenH)
    /** Box-filter factor: the captured square is RESIZE·k pixels. */
    val k: Int = max(1, side / RESIZE)
    private val scale: Double = RESIZE * k / side.toDouble()
    /** Virtual display size (the whole screen, scaled). */
    val vw: Int = max(RESIZE * k, (screenW * scale).roundToInt())
    val vh: Int = max(RESIZE * k, (screenH * scale).roundToInt())
    val vdpi: Int = max(1, (densityDpi * scale).roundToInt())
    /** Top-left of the 224·k crop in the virtual display. */
    val cropX: Int = (vw - RESIZE * k) / 2 + MARGIN * k
    val cropY: Int = (vh - RESIZE * k) / 2 + MARGIN * k

    fun sameAs(o: Geometry?): Boolean = o != null && o.vw == vw && o.vh == vh && o.k == k && o.vdpi == vdpi

    companion object {
        const val RESIZE = 252
        const val CROP = 224
        const val MARGIN = (RESIZE - CROP) / 2
    }
}

internal object ScreenCrop {
    private const val INV255 = 1f / 255f

    /**
     * Writes the model input (1×224×224×3 floats, RGB, u8·(1/255) like torchvision's ToDtype) for an RGBA_8888
     * frame of the virtual display: the k×k box average of the crop, rounded to u8. [row] is scratch space of at
     * least [rowBytes] bytes.
     */
    fun toInput(buf: ByteBuffer, rowStride: Int, pixelStride: Int, g: Geometry, out: FloatArray, row: ByteArray) {
        val n = Geometry.CROP
        val k = g.k
        val kk = k * k
        val half = kk / 2
        val need = rowBytes(g, pixelStride)
        require(row.size >= need) { "scratch row too small" }
        require(out.size == n * n * 3) { "output must hold 224*224*3 floats" }
        val acc = IntArray(n * 3)
        val src = buf.duplicate()
        for (oy in 0 until n) {
            java.util.Arrays.fill(acc, 0)
            for (dy in 0 until k) {
                src.position((g.cropY + oy * k + dy) * rowStride + g.cropX * pixelStride)
                src.get(row, 0, need)
                var p = 0
                var a = 0
                for (ox in 0 until n) {
                    var r = 0
                    var gr = 0
                    var b = 0
                    for (dx in 0 until k) {
                        r += row[p].toInt() and 0xff
                        gr += row[p + 1].toInt() and 0xff
                        b += row[p + 2].toInt() and 0xff
                        p += pixelStride
                    }
                    acc[a] += r
                    acc[a + 1] += gr
                    acc[a + 2] += b
                    a += 3
                }
            }
            val base = oy * n * 3
            for (i in 0 until n * 3) out[base + i] = ((acc[i] + half) / kk).toFloat() * INV255
        }
    }

    /** Bytes read per source row: 224·k pixels. */
    fun rowBytes(g: Geometry, pixelStride: Int): Int = (Geometry.CROP * g.k - 1) * pixelStride + 4
}
