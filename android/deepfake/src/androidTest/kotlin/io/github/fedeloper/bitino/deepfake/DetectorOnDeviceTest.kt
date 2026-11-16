package io.github.fedeloper.bitino.deepfake

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.github.fedeloper.bitino.Bitino
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.exp

/**
 * The app's detector path on the phone, without the screen: each bundled detector (a local build with
 * -Pbitino.deepfakeModels) is loaded through the app's library and run on the first images of the CIFAKE eval subset
 * (android/tools/make_deepfake_testdata.py). Its outputs are compared with the reference phone's stage-optimized
 * outputs. Run with `am instrument -w io.github.fedeloper.bitino.deepfake.test/androidx.test.runner.AndroidJUnitRunner`.
 */
@RunWith(AndroidJUnit4::class)
class DetectorOnDeviceTest {
    private val testAssets = InstrumentationRegistry.getInstrumentation().context.assets
    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    private fun floats(name: String): FloatArray {
        val b = testAssets.open(name).use { it.readBytes() }
        val fb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    @Test
    fun bundledDetectorsMatchTheReferencePhone() {
        val bundled = app.assets.list("models")?.filter { it.endsWith(".bitino") }.orEmpty()
        assumeTrue("no bundled detector (build with -Pbitino.deepfakeModels)", bundled.isNotEmpty())
        val input = floats("input.f32")
        val per = 224 * 224 * 3
        val n = input.size / per
        var compared = 0
        for (file in bundled) {
            val name = file.removeSuffix(".bitino")
            val expected = runCatching { floats("expected_$name.f32") }.getOrNull() ?: continue
            Bitino.load(app, "models/$file", threads = 1).use { m ->
                assertEquals(2, m.outputSize)
                val out = FloatArray(2)
                var worst = 0.0
                var ms = 0.0
                for (i in 0 until n) {
                    val t0 = SystemClock.elapsedRealtimeNanos()
                    m.run(input.copyOfRange(i * per, (i + 1) * per), out)
                    ms += (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                    val ref = expected[i * 2 + 1]
                    val d = abs(out[1] - ref).toDouble()
                    worst = maxOf(worst, d)
                    assertEquals("$name image $i: constant logit", 0f, out[0], 0f)
                    assertTrue("$name image $i: logit ${out[1]} vs reference $ref", d <= 2e-3 + 1e-3 * abs(ref))
                    val pFake = 1.0 / (1.0 + exp(out[1].toDouble()))
                    val pRef = 1.0 / (1.0 + exp(ref.toDouble()))
                    assertEquals("$name image $i: verdict", pRef > 0.5, pFake > 0.5)
                }
                Log.i("BitinoDeepfakeTest", "$name: $n images, max |logit - reference| = $worst, mean ${"%.1f".format(ms / n)} ms (1 thread, unpinned)")
            }
            compared++
        }
        assertTrue("no detector had reference outputs", compared > 0)
    }

    @Test
    fun threeSquaresPerFrameTiming() {
        val bundled = app.assets.list("models")?.filter { it.endsWith(".bitino") }.orEmpty()
        assumeTrue("no bundled detector", bundled.isNotEmpty())
        val g = Geometry(1116, 2484, 420)                     // OnePlus Open cover screen, portrait
        val rowStride = g.vw * 4
        val frame = ByteBuffer.allocateDirect(rowStride * g.vh)
        val rnd = java.util.Random(1)
        for (i in 0 until rowStride * g.vh) frame.put(i, rnd.nextInt(256).toByte())
        val input = FloatArray(224 * 224 * 3)
        val row = ByteArray(ScreenCrop.rowBytes(3, 4))
        val out = FloatArray(2)
        Bitino.load(app, "models/${bundled.first()}", threads = 1).use { m ->
            var crop = 0.0
            var total = 0.0
            val reps = 10
            for (r in 0 until reps + 2) {
                var c = 0L
                val t0 = SystemClock.elapsedRealtimeNanos()
                for (sc in 0 until Geometry.SCALES) {
                    val a = SystemClock.elapsedRealtimeNanos()
                    ScreenCrop.toInput(frame, rowStride, 4, g, sc, input, row)
                    c += SystemClock.elapsedRealtimeNanos() - a
                    m.run(input, out)
                    assertTrue(out[1].isFinite())
                }
                if (r >= 2) {
                    total += (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                    crop += c / 1e6
                }
            }
            Log.i("BitinoDeepfakeTest", "3 squares per frame: ${"%.1f".format(total / reps)} ms (crops ${"%.1f".format(crop / reps)} ms), 1 thread, unpinned")
        }
    }
}
