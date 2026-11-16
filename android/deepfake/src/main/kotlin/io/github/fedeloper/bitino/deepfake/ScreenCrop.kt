package io.github.fedeloper.bitino.deepfake

import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Where the analysed squares lie and how the screen is captured.
 *
 * The detectors expect the authors' test preprocessing: Resize(252, antialiased) -> CenterCrop(224) -> u8 / 255,
 * NHWC RGB. The virtual display that mirrors the screen is scaled so that the largest centred square of the screen
 * (side [side] screen pixels) is 3·252 = 756 pixels; a 3×3 box filter (the antialiasing) brings it to 252, and its
 * centre 224×224 is the "whole view" model input. The images found on screen ([RectFinder]) are cut from the same
 * capture and preprocessed by [Resample].
 */
internal class Geometry(val screenW: Int, val screenH: Int, densityDpi: Int) {
    /** Side of the largest centred square in screen pixels. */
    val side: Int = min(screenW, screenH)
    private val scale: Double = RESIZE * KS[0] / side.toDouble()
    /** Virtual display size (the whole screen, scaled). */
    val vw: Int = max(RESIZE * KS[0], (screenW * scale).roundToInt())
    val vh: Int = max(RESIZE * KS[0], (screenH * scale).roundToInt())
    val vdpi: Int = max(1, (densityDpi * scale).roundToInt())

    /** Box-filter factor of square [i] (0 = largest). */
    fun k(i: Int): Int = KS[i]
    /** Top-left of square [i]'s 224·k crop in the virtual display. */
    fun cropX(i: Int): Int = (vw - RESIZE * KS[i]) / 2 + MARGIN * KS[i]
    fun cropY(i: Int): Int = (vh - RESIZE * KS[i]) / 2 + MARGIN * KS[i]

    /** Capture pixels -> screen pixels. */
    val toScreen: Float = side / (RESIZE * KS[0]).toFloat()

    fun sameAs(o: Geometry?): Boolean = o != null && o.vw == vw && o.vh == vh && o.vdpi == vdpi

    companion object {
        const val RESIZE = 252
        const val CROP = 224
        const val MARGIN = (RESIZE - CROP) / 2
        /** Box-filter factor of the centre square (index 0; the class keeps the index for more squares). */
        val KS = intArrayOf(3)
    }
}

internal object ScreenCrop {
    private const val INV255 = 1f / 255f

    /**
     * Writes the model input (1×224×224×3 floats, RGB, u8·(1/255) like torchvision's ToDtype) of square [scale] for
     * an RGBA_8888 frame of the virtual display: the k×k box average of its crop, rounded to u8. [row] is scratch
     * space of at least [rowBytes] bytes.
     */
    fun toInput(buf: ByteBuffer, rowStride: Int, pixelStride: Int, g: Geometry, scale: Int, out: FloatArray, row: ByteArray) {
        val n = Geometry.CROP
        val k = g.k(scale)
        val cropX = g.cropX(scale)
        val cropY = g.cropY(scale)
        val kk = k * k
        val half = kk / 2
        val need = rowBytes(k, pixelStride)
        require(row.size >= need) { "scratch row too small" }
        require(out.size == n * n * 3) { "output must hold 224*224*3 floats" }
        val acc = IntArray(n * 3)
        val src = buf.duplicate()
        for (oy in 0 until n) {
            java.util.Arrays.fill(acc, 0)
            for (dy in 0 until k) {
                src.position((cropY + oy * k + dy) * rowStride + cropX * pixelStride)
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
    fun rowBytes(k: Int, pixelStride: Int): Int = (Geometry.CROP * k - 1) * pixelStride + 4
}
