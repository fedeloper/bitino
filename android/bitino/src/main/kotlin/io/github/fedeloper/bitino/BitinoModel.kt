package io.github.fedeloper.bitino

import android.graphics.Bitmap
import android.os.SystemClock
import org.json.JSONObject
import java.io.Closeable

/**
 * A loaded model with one inference session (created by [Bitino.load]). Calls are serialized: one model object
 * runs one inference at a time; use one object per thread for parallel inferences. Close it to free the native
 * memory.
 */
public class BitinoModel private constructor(
    private var model: Long,
    private var session: Long,
    /** CPU threads used per inference. */
    public val threads: Int,
) : Closeable {
    /** Shape of input 0, e.g. [1, 224, 224, 3] (NHWC) for an image classifier. */
    public val inputShape: IntArray = NativeLib.inputShape(session).map { it.toInt() }.toIntArray()

    /** Number of floats of input 0. */
    public val inputSize: Int = inputShape.fold(1) { a, b -> a * b }

    /** Number of model outputs; [run] returns output 0 (e.g. the class scores). */
    public val numOutputs: Int = NativeLib.numOutputs(session)

    /** Number of floats of output 0. */
    public val outputSize: Int = NativeLib.outputSize(session, 0)

    /** The model's manifest (provenance, licence notes, preprocessing) as JSON text. */
    public val manifestJson: String = NativeLib.manifest(model)

    /** The preprocessing definition recorded in the manifest (free text), or null. */
    public val preprocessingDefinition: String? = runCatching {
        val input = JSONObject(manifestJson).optJSONObject("input")
        input?.opt("preprocessing")?.takeIf { it is String } as String?
    }.getOrNull()

    /**
     * Image preprocessing used by [run] (Bitmap): parsed from the manifest ([Preprocessing.parse]), null if the
     * manifest's pipeline is not one the library reproduces. May be replaced, e.g. for a model without a manifest.
     */
    public var preprocessing: Preprocessing? =
        preprocessingDefinition?.let { Preprocessing.parse(it, if (inputShape.size == 4) inputShape[1] else 0) }

    /** Wall time of the last [run] steps, in milliseconds. */
    public var lastPreprocessMs: Double = 0.0
        private set
    public var lastInferenceMs: Double = 0.0
        private set

    private val input = FloatArray(inputSize)
    private var pixels = IntArray(0)
    private var rotated = IntArray(0)

    /**
     * Classifies / runs a bitmap: preprocesses it per [preprocessing] (after rotating it clockwise by
     * [rotationDegrees], e.g. CameraX's ImageInfo.rotationDegrees) and returns a new array with output 0.
     */
    @JvmOverloads
    @Synchronized
    public fun run(bitmap: Bitmap, rotationDegrees: Int = 0): FloatArray {
        val out = FloatArray(outputSize)
        run(bitmap, out, rotationDegrees)
        return out
    }

    /** As [run] (Bitmap), writing output 0 into [output] (size [outputSize]). */
    @JvmOverloads
    @Synchronized
    public fun run(bitmap: Bitmap, output: FloatArray, rotationDegrees: Int = 0) {
        val t0 = SystemClock.elapsedRealtimeNanos()
        preprocess(bitmap, input, rotationDegrees)
        val t1 = SystemClock.elapsedRealtimeNanos()
        runInternal(input, output)
        val t2 = SystemClock.elapsedRealtimeNanos()
        lastPreprocessMs = (t1 - t0) / 1e6
        lastInferenceMs = (t2 - t1) / 1e6
    }

    /** Runs a preprocessed input ([inputSize] floats, layout as [inputShape]) and returns output 0. */
    @Synchronized
    public fun run(input: FloatArray): FloatArray {
        val out = FloatArray(outputSize)
        run(input, out)
        return out
    }

    /** As [run] (FloatArray), writing output 0 into [output]. */
    @Synchronized
    public fun run(input: FloatArray, output: FloatArray) {
        val t1 = SystemClock.elapsedRealtimeNanos()
        runInternal(input, output)
        lastPreprocessMs = 0.0
        lastInferenceMs = (SystemClock.elapsedRealtimeNanos() - t1) / 1e6
    }

    /** Output [index] of the last run (for multi-output models). */
    @Synchronized
    public fun output(index: Int): FloatArray {
        check(session != 0L) { "model is closed" }
        require(index in 0 until numOutputs) { "output index $index out of range (0..${numOutputs - 1})" }
        return FloatArray(NativeLib.outputSize(session, index)).also { NativeLib.getOutput(session, index, it) }
    }

    /**
     * The model input for [bitmap] (rotated clockwise by [rotationDegrees], a multiple of 90), written into [out]
     * (default: a new array of [inputSize] floats).
     */
    @JvmOverloads
    @Synchronized
    public fun preprocess(bitmap: Bitmap, out: FloatArray = FloatArray(inputSize), rotationDegrees: Int = 0): FloatArray {
        val p = preprocessing ?: throw IllegalStateException(
            "the model's preprocessing is not one Bitino reproduces (${preprocessingDefinition ?: "no manifest entry"}); " +
                "set BitinoModel.preprocessing or pass a float tensor to run()",
        )
        check(inputShape.size == 4 && inputShape[1] == p.cropSize && inputShape[2] == p.cropSize && inputShape[3] == 3) {
            "model input ${inputShape.contentToString()} does not match preprocessing crop ${p.cropSize}x${p.cropSize}x3"
        }
        val w = bitmap.width
        val h = bitmap.height
        if (pixels.size < w * h) pixels = IntArray(w * h)
        val conf = bitmap.config
        val bmp = if (conf == Bitmap.Config.ARGB_8888 || conf == Bitmap.Config.RGB_565) bitmap
        else bitmap.copy(Bitmap.Config.ARGB_8888, false)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val rot = ((rotationDegrees % 360) + 360) % 360
        if (rot == 0) {
            p.apply(pixels, w, h, out)
        } else {
            if (rotated.size < w * h) rotated = IntArray(w * h)
            Pixels.rotate(pixels, w, h, rot, rotated)
            p.apply(rotated, if (rot == 180) w else h, if (rot == 180) h else w, out)
        }
        return out
    }

    private fun runInternal(input: FloatArray, output: FloatArray) {
        check(session != 0L) { "model is closed" }
        require(input.size == inputSize) { "input has ${input.size} floats, the model expects $inputSize" }
        require(output.size == outputSize) { "output has ${output.size} floats, output 0 has $outputSize" }
        NativeLib.run(session, input, output)
    }

    @Synchronized
    override fun close() {
        if (session != 0L) NativeLib.freeSession(session)
        if (model != 0L) NativeLib.freeModel(model)
        session = 0L
        model = 0L
    }

    internal companion object {
        fun create(model: Long, threads: Int): BitinoModel {
            require(threads >= 1) { "threads must be >= 1" }
            val session = try {
                NativeLib.createSession(model, threads)
            } catch (e: Throwable) {
                NativeLib.freeModel(model)
                throw e
            }
            return BitinoModel(model, session, threads)
        }
    }
}
