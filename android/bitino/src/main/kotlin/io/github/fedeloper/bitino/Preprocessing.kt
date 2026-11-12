package io.github.fedeloper.bitino

import java.util.concurrent.ConcurrentHashMap
import kotlin.math.ceil


/**
 * Image preprocessing of a model, as recorded in its manifest (`manifest.input.preprocessing`), reproduced
 * bit for bit:
 *
 *  1. resize so that the short side is [resizeShortSide] (long side `int(short * long / short_in)`), with Pillow's
 *     resampling ([Filter.BILINEAR] / [Filter.BICUBIC]: Pillow's antialiased separable convolution with 22-bit
 *     fixed-point coefficients and an 8-bit intermediate, horizontal pass first), as torchvision's `Resize` does on
 *     PIL images;
 *  2. center crop [cropSize] x [cropSize] with offsets `int(round((H - crop) / 2.0))` (Python's round half to even);
 *  3. per channel `((v / divisor) - mean[c]) / std[c]` in float32 (torchvision `ToTensor` + `Normalize`: divisor
 *     255; raw 0..255 inputs: divisor 1, mean 0, std 1), channels in RGB or BGR order, NHWC.
 *
 * For a decoded RGB image this equals the Python pipeline that produced the model's evaluation tensors (PIL +
 * torchvision, tools/build_eval_set.py; OMZ Accuracy Checker, tools/eval_sets/omz_bgr.py); the unit tests check it on
 * synthetic images against fixtures generated with PIL / torchvision. Camera frames are of course not JPEG files:
 * only the steps after decoding are reproduced.
 *
 * [resizeShortSide] = 0 means the manifest specifies no resize (e.g. CIFAR models, which take 32x32 images as is):
 * an image of a different size is then resized (short side to [cropSize], bilinear) and center-cropped; that
 * framing is the app's choice, not the model's.
 */
public class Preprocessing(
    public val resizeShortSide: Int,
    public val cropSize: Int,
    public val filter: Filter = Filter.BILINEAR,
    public val divisor: Float = 255f,
    public val mean: FloatArray = floatArrayOf(0f, 0f, 0f),
    public val std: FloatArray = floatArrayOf(1f, 1f, 1f),
    public val bgr: Boolean = false,
    /** The manifest text this was parsed from (empty when built by hand). */
    public val definition: String = "",
) {
    public enum class Filter(internal val support: Double) { BILINEAR(1.0), BICUBIC(2.0) }

    init {
        require(cropSize > 0) { "cropSize must be > 0" }
        require(resizeShortSide == 0 || resizeShortSide >= cropSize) { "resizeShortSide < cropSize" }
        require(mean.size == 3 && std.size == 3) { "mean / std need 3 values" }
    }

    /** Number of floats written by [apply]: cropSize * cropSize * 3. */
    public val outputSize: Int get() = cropSize * cropSize * 3

    // value lookup per output channel: ((v / divisor) - mean[c]) / std[c] for v in 0..255, computed in float32
    private val lut: Array<FloatArray> = Array(3) { c -> FloatArray(256) { v -> (v.toFloat() / divisor - mean[c]) / std[c] } }
    private val coeffCache = ConcurrentHashMap<Long, Coeffs>()
    // horizontal-pass rows, reused per thread (ThreadLocal.withInitial needs API 26)
    private val scratch = object : ThreadLocal<IntArray>() { override fun initialValue(): IntArray = IntArray(0) }

    /**
     * Preprocesses an image given as ARGB pixels (Bitmap.getPixels layout: row-major, `0xAARRGGBB`; alpha ignored)
     * into [out] (NHWC, [outputSize] floats from [offset]).
     */
    @JvmOverloads
    public fun apply(argb: IntArray, width: Int, height: Int, out: FloatArray, offset: Int = 0) {
        require(width > 0 && height > 0 && argb.size >= width * height) { "bad image ${width}x$height" }
        require(out.size - offset >= outputSize) { "output buffer too small" }
        val short = if (resizeShortSide > 0) resizeShortSide else cropSize
        val rw: Int
        val rh: Int
        if (resizeShortSide == 0 && width == cropSize && height == cropSize) {
            rw = width; rh = height
        } else {
            val s = resizedSize(width, height, short)
            rw = s[0]; rh = s[1]
        }
        val crop = cropSize
        val top = Math.rint((rh - crop) / 2.0).toInt()
        val left = Math.rint((rw - crop) / 2.0).toInt()
        val hc = if (rw != width) coeffs(width, rw) else null
        val vc = if (rh != height) coeffs(height, rh) else null

        // source rows read by the vertical pass for output rows top until top + crop
        val r0: Int
        val r1: Int
        if (vc == null) {
            r0 = top; r1 = top + crop
        } else {
            r0 = vc.bounds[2 * top]
            r1 = vc.bounds[2 * (top + crop - 1)] + vc.bounds[2 * (top + crop - 1) + 1]
        }
        val rows = r1 - r0
        // horizontal pass (only the cropped columns: each output column depends on its own coefficients only)
        val need = rows * crop * 3
        var tmp = scratch.get()!!
        if (tmp.size < need) { tmp = IntArray(need); scratch.set(tmp) }
        val half = 1 shl (PRECISION_BITS - 1)
        if (hc == null) {
            for (r in 0 until rows) {
                val srcRow = (r0 + r) * width
                var t = r * crop * 3
                for (xx in left until left + crop) {
                    val p = argb[srcRow + xx]
                    tmp[t] = (p ushr 16) and 0xFF; tmp[t + 1] = (p ushr 8) and 0xFF; tmp[t + 2] = p and 0xFF
                    t += 3
                }
            }
        } else {
            val hb = hc.bounds
            val hk = hc.k
            val hks = hc.ksize
            for (r in 0 until rows) {
                val srcRow = (r0 + r) * width
                var t = r * crop * 3
                for (xx in left until left + crop) {
                    val base = srcRow + hb[2 * xx]
                    val n = hb[2 * xx + 1]
                    val kb = xx * hks
                    var s0 = half; var s1 = half; var s2 = half
                    for (x in 0 until n) {
                        val p = argb[base + x]
                        val k = hk[kb + x]
                        s0 += ((p ushr 16) and 0xFF) * k
                        s1 += ((p ushr 8) and 0xFF) * k
                        s2 += (p and 0xFF) * k
                    }
                    tmp[t] = clip8(s0); tmp[t + 1] = clip8(s1); tmp[t + 2] = clip8(s2)
                    t += 3
                }
            }
        }
        // vertical pass + normalization; output channel 0 is R (RGB) or B (BGR), mean/std/lut are per output channel
        val first = if (bgr) 2 else 0   // tmp channel written to output channel 0
        val last = 2 - first            // tmp channel written to output channel 2
        val lut0 = lut[0]; val lut1 = lut[1]; val lut2 = lut[2]
        var o = offset
        for (yy in top until top + crop) {
            if (vc == null) {
                var t = (yy - r0) * crop * 3
                for (xx in 0 until crop) {
                    out[o] = lut0[tmp[t + first]]
                    out[o + 1] = lut1[tmp[t + 1]]
                    out[o + 2] = lut2[tmp[t + last]]
                    t += 3; o += 3
                }
            } else {
                val ymin = vc.bounds[2 * yy] - r0
                val n = vc.bounds[2 * yy + 1]
                val kb = yy * vc.ksize
                val vk = vc.k
                val stride = crop * 3
                for (xx in 0 until crop) {
                    var s0 = half; var s1 = half; var s2 = half
                    var t = (ymin * crop + xx) * 3
                    for (y in 0 until n) {
                        val k = vk[kb + y]
                        s0 += tmp[t] * k
                        s1 += tmp[t + 1] * k
                        s2 += tmp[t + 2] * k
                        t += stride
                    }
                    val r = clip8(s0); val g = clip8(s1); val b = clip8(s2)
                    out[o] = lut0[if (bgr) b else r]
                    out[o + 1] = lut1[g]
                    out[o + 2] = lut2[if (bgr) r else b]
                    o += 3
                }
            }
        }
    }

    override fun toString(): String =
        "Preprocessing(resizeShortSide=$resizeShortSide, cropSize=$cropSize, filter=$filter, divisor=$divisor, " +
            "mean=${mean.contentToString()}, std=${std.contentToString()}, bgr=$bgr)"

    private class Coeffs(val ksize: Int, val bounds: IntArray, val k: IntArray)

    /** Pillow's precompute_coeffs + normalize_coeffs_8bpc (libImaging/Resample.c) for a full-image resize. */
    private fun coeffs(inSize: Int, outSize: Int): Coeffs = coeffCache.getOrPut((inSize.toLong() shl 32) or outSize.toLong()) {
        val scale = inSize.toDouble() / outSize
        val filterscale = if (scale < 1.0) 1.0 else scale
        val support = filter.support * filterscale
        val ksize = ceil(support).toInt() * 2 + 1
        val bounds = IntArray(outSize * 2)
        val kk = IntArray(outSize * ksize)
        val w = DoubleArray(ksize)
        for (xx in 0 until outSize) {
            val center = (xx + 0.5) * scale
            val ss = 1.0 / filterscale
            var xmin = (center - support + 0.5).toInt()
            if (xmin < 0) xmin = 0
            var xmax = (center + support + 0.5).toInt()
            if (xmax > inSize) xmax = inSize
            xmax -= xmin
            var ww = 0.0
            for (x in 0 until xmax) {
                val v = kernel((x + xmin - center + 0.5) * ss)
                w[x] = v
                ww += v
            }
            for (x in 0 until xmax) {
                val v = if (ww != 0.0) w[x] / ww else w[x]
                kk[xx * ksize + x] = if (v < 0) (-0.5 + v * (1 shl PRECISION_BITS)).toInt() else (0.5 + v * (1 shl PRECISION_BITS)).toInt()
            }
            bounds[2 * xx] = xmin
            bounds[2 * xx + 1] = xmax
        }
        Coeffs(ksize, bounds, kk)
    }

    private fun kernel(x0: Double): Double {
        val x = if (x0 < 0.0) -x0 else x0
        return when (filter) {
            Filter.BILINEAR -> if (x < 1.0) 1.0 - x else 0.0
            Filter.BICUBIC -> {
                val a = -0.5
                if (x < 1.0) ((a + 2.0) * x - (a + 3.0)) * x * x + 1 else if (x < 2.0) (((x - 5) * x + 8) * x - 4) * a else 0.0
            }
        }
    }

    public companion object {
        private const val PRECISION_BITS = 32 - 8 - 2

        private fun clip8(v: Int): Int {
            val s = v shr PRECISION_BITS
            return if (s < 0) 0 else if (s > 255) 255 else s
        }

        /** torchvision Resize(int) / OMZ scale_greater output size: [width, height]. */
        @JvmStatic
        public fun resizedSize(width: Int, height: Int, shortSide: Int): IntArray =
            if (width <= height) intArrayOf(shortSide, ((shortSide.toLong() * height).toDouble() / width).toInt())
            else intArrayOf(((shortSide.toLong() * width).toDouble() / height).toInt(), shortSide)

        /** The standard torchvision ImageNet evaluation transform (Resize 256, CenterCrop 224, ImageNet mean/std). */
        @JvmField
        public val TORCHVISION_IMAGENET: Preprocessing = Preprocessing(
            256, 224, Filter.BILINEAR, 255f, floats("0.485, 0.456, 0.406"), floats("0.229, 0.224, 0.225"), false,
            "torchvision Resize(256) -> CenterCrop(224) -> ToTensor() -> Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))",
        )

        private val RESIZE = Regex("""Resize\(\s*(?:size=)?(\d+)""")
        private val CENTER_CROP = Regex("""CenterCrop\(\s*(?:size=)?\(?(\d+)""")
        private val NORMALIZE = Regex("""Normalize\(\s*(?:mean=)?[(\[]([^)\]]*)[)\]]\s*,\s*(?:std=)?[(\[]([^)\]]*)[)\]]""")
        private val OMZ_RESIZE = Regex("""short side to (\d+)""")
        private val OMZ_CROP = Regex("""center crop (\d+)""")

        private fun floats(csv: String): FloatArray {
            val v = csv.split(',').map { it.trim() }.filter { it.isNotEmpty() }.map { it.toDouble().toFloat() }
            require(v.size == 3) { "expected 3 values in '$csv'" }
            return v.toFloatArray()
        }

        /**
         * Parses a manifest preprocessing definition (`manifest.input.preprocessing`). Recognized: the torchvision
         * evaluation transforms (`Resize` / `CenterCrop` / `ToTensor` / `Normalize`, bilinear or bicubic) and the
         * OpenVINO Accuracy Checker one (OMZ models: scale_greater resize, center crop, BGR 0..255). Returns null for
         * anything else (e.g. TensorFlow pipelines): use an explicit [Preprocessing] or run on float tensors.
         * [inputSize]: the model's input height/width, used when the definition has no crop (CIFAR-style).
         */
        @JvmStatic
        @JvmOverloads
        public fun parse(definition: String, inputSize: Int = 0): Preprocessing? {
            if (definition.contains("scale_greater") && definition.contains("RGB->BGR")) {
                val r = OMZ_RESIZE.find(definition)?.groupValues?.get(1)?.toInt() ?: return null
                val c = OMZ_CROP.find(definition)?.groupValues?.get(1)?.toInt() ?: return null
                if (!definition.contains("0..255")) return null
                return Preprocessing(r, c, Filter.BILINEAR, 1f, bgr = true, definition = definition)
            }
            if (!definition.contains("ToTensor")) return null
            if (Regex("""(?i)\bBGR\b""").containsMatchIn(definition)) return null
            val resize = RESIZE.find(definition)?.groupValues?.get(1)?.toInt() ?: 0
            val crop = CENTER_CROP.find(definition)?.groupValues?.get(1)?.toInt() ?: inputSize
            if (crop <= 0) return null
            if (resize == 0 && definition.contains("Resize")) return null   // a Resize form not understood
            val filter = if (Regex("(?i)bicubic").containsMatchIn(definition)) Filter.BICUBIC else Filter.BILINEAR
            val norm = NORMALIZE.find(definition)
            val mean = norm?.let { floats(it.groupValues[1]) } ?: floatArrayOf(0f, 0f, 0f)
            val std = norm?.let { floats(it.groupValues[2]) } ?: floatArrayOf(1f, 1f, 1f)
            if (norm == null && definition.contains("Normalize")) return null
            return Preprocessing(resize, crop, filter, 255f, mean, std, false, definition)
        }
    }
}
