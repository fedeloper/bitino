package io.github.fedeloper.bitino

import android.content.Context
import java.io.File

/**
 * Entry point of the Bitino Android library.
 *
 * ```
 * val model = Bitino.load(context, "ab-bnn-imagenet-reactnet18.bitino", threads = 1)   // asset name or file path
 * val scores: FloatArray = model.run(bitmap)   // preprocessing from the model's manifest
 * val top5 = Bitino.topK(scores, 5)
 * model.close()
 * ```
 */
public object Bitino {
    /** Runtime version (e.g. "0.1.0"). */
    @JvmStatic
    public val version: String get() = NativeLib.version()

    /** CPU features the runtime detected and uses to pick kernels (e.g. "aarch64: neon dotprod i8mm ..."). */
    @JvmStatic
    public val cpuFeatures: String get() = NativeLib.cpuFeatures()

    /**
     * Loads a model from the app's assets (a relative name such as "models/net.bitino") or, if [model] is an
     * absolute path, from that file. [threads]: CPU threads per inference (1 is often the most efficient choice
     * for a camera loop; 2-4 lower the latency on big cores). [verifyHash]: check the SHA-256 of the weights.
     */
    @JvmStatic
    @JvmOverloads
    public fun load(context: Context, model: String, threads: Int = 1, verifyHash: Boolean = true): BitinoModel {
        if (model.startsWith("/")) return load(File(model), threads, verifyHash)
        val bytes = context.assets.open(model).use { it.readBytes() }
        return load(bytes, threads, verifyHash)
    }

    /** Loads a model file (memory-mapped). */
    @JvmStatic
    @JvmOverloads
    public fun load(file: File, threads: Int = 1, verifyHash: Boolean = true): BitinoModel =
        BitinoModel.create(NativeLib.loadFile(file.absolutePath, verifyHash), threads)

    /** Loads a model from memory (copied; e.g. bytes read from a content URI). */
    @JvmStatic
    @JvmOverloads
    public fun load(bytes: ByteArray, threads: Int = 1, verifyHash: Boolean = true): BitinoModel =
        BitinoModel.create(NativeLib.loadBytes(bytes, verifyHash), threads)

    /** Indices and values of the [k] largest scores, best first. */
    @JvmStatic
    public fun topK(scores: FloatArray, k: Int): List<Pair<Int, Float>> {
        val n = minOf(k, scores.size)
        val idx = IntArray(n) { -1 }
        for (i in scores.indices) {
            val v = scores[i]
            var j = n - 1
            if (idx[j] >= 0 && v <= scores[idx[j]]) continue
            while (j > 0 && (idx[j - 1] < 0 || v > scores[idx[j - 1]])) {
                idx[j] = idx[j - 1]
                j--
            }
            idx[j] = i
        }
        return idx.filter { it >= 0 }.map { it to scores[it] }
    }

    /** Softmax of [logits] (float64 accumulation); returns a new array. */
    @JvmStatic
    public fun softmax(logits: FloatArray): FloatArray {
        if (logits.isEmpty()) return FloatArray(0)
        val m = logits.max()
        val e = DoubleArray(logits.size) { kotlin.math.exp((logits[it] - m).toDouble()) }
        val s = e.sum()
        return FloatArray(logits.size) { (e[it] / s).toFloat() }
    }
}
