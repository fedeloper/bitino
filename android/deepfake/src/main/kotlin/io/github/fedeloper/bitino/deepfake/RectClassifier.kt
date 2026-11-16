package io.github.fedeloper.bitino.deepfake

import io.github.fedeloper.bitino.BitinoModel
import java.nio.ByteBuffer
import kotlin.math.exp

/**
 * Classification mode: classifies a rectangle of an RGBA frame with an ImageNet classifier, through the classifier's
 * own preprocessing as recorded in its manifest (e.g. Resize 256 / CenterCrop 224 / ImageNet mean and std, reproduced
 * by the library's Preprocessing). Not thread-safe.
 */
internal class RectClassifier(private val model: BitinoModel, private val labels: List<String>) {
    private val pre = requireNotNull(model.preprocessing) { "the classifier's preprocessing is not one the library reproduces" }
    private val input = FloatArray(model.inputSize)
    private val logits = FloatArray(model.outputSize)
    private var argb = IntArray(0)
    private var row = ByteArray(0)

    /** Top-1 label and its softmax probability for [x0, x1) × [y0, y1). */
    fun classify(buf: ByteBuffer, rowStride: Int, ps: Int, x0: Int, y0: Int, x1: Int, y1: Int): Pair<String, Float> {
        val w = x1 - x0
        val h = y1 - y0
        require(w > 0 && h > 0) { "empty rectangle" }
        if (argb.size < w * h) argb = IntArray(w * h)
        val need = (w - 1) * ps + 4
        if (row.size < need) row = ByteArray(need)
        val src = buf.duplicate()
        for (y in 0 until h) {
            src.position((y0 + y) * rowStride + x0 * ps)
            src.get(row, 0, need)
            var p = 0
            val o = y * w
            for (x in 0 until w) {
                argb[o + x] = (0xff shl 24) or ((row[p].toInt() and 0xff) shl 16) or ((row[p + 1].toInt() and 0xff) shl 8) or
                    (row[p + 2].toInt() and 0xff)
                p += ps
            }
        }
        pre.apply(argb, w, h, input)
        model.run(input, logits)
        var best = 0
        for (i in logits.indices) if (logits[i] > logits[best]) best = i
        var sum = 0.0
        for (v in logits) sum += exp((v - logits[best]).toDouble())
        return labels.getOrElse(best) { "class $best" } to (1.0 / sum).toFloat()
    }
}
