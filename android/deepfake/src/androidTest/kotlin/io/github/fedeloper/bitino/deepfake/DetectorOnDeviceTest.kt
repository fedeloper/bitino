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
    fun perFrameTimingWithImagesOnScreen() {
        val bundled = app.assets.list("models")?.filter { it.endsWith(".bitino") }.orEmpty()
        assumeTrue("no bundled detector", bundled.isNotEmpty())
        val g = Geometry(1116, 2484, 420)                     // OnePlus Open cover screen, portrait
        val rowStride = g.vw * 4
        val frame = ByteBuffer.allocateDirect(rowStride * g.vh)
        for (i in 0 until rowStride * g.vh) frame.put(i, (-1).toByte())   // white UI background
        val rnd = java.util.Random(1)
        // three "photos" (noise) on the white page: full width, inset, thumbnail
        val photos = listOf(intArrayOf(0, 100, g.vw, 520), intArrayOf(40, 600, 700, 1000), intArrayOf(40, 1100, 300, 1360))
        for (p in photos) for (y in p[1] until p[3]) for (x in p[0] until p[2]) for (c in 0 until 3) {
            frame.put(y * rowStride + x * 4 + c, rnd.nextInt(256).toByte())
        }
        val input = FloatArray(224 * 224 * 3)
        val row = ByteArray(ScreenCrop.rowBytes(3, 4))
        val out = FloatArray(2)
        val finder = RectFinder(maxRects = 4)
        val detector = bundled.firstOrNull { it.startsWith("ftl-deepfake") }
        assumeTrue("no bundled deepfake detector", detector != null)
        Bitino.load(app, "models/$detector", threads = 1).use { m ->
            var find = 0.0
            var prep = 0.0
            var total = 0.0
            var found = 0
            val reps = 10
            for (r in 0 until reps + 2) {
                val t0 = SystemClock.elapsedRealtimeNanos()
                val rects = finder.find(frame, rowStride, 4, g.vw, g.vh)
                val t1 = SystemClock.elapsedRealtimeNanos()
                var p = 0L
                var a = SystemClock.elapsedRealtimeNanos()
                ScreenCrop.toInput(frame, rowStride, 4, g, 0, input, row)
                p += SystemClock.elapsedRealtimeNanos() - a
                m.run(input, out)
                for (rc in rects) {
                    a = SystemClock.elapsedRealtimeNanos()
                    Resample.toInput(frame, rowStride, 4, rc[0], rc[1], rc[2], rc[3], input)
                    p += SystemClock.elapsedRealtimeNanos() - a
                    m.run(input, out)
                    assertTrue(out[1].isFinite())
                }
                if (r >= 2) {
                    total += (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                    find += (t1 - t0) / 1e6
                    prep += p / 1e6
                }
                found = rects.size
            }
            assertEquals(3, found)
            Log.i("BitinoDeepfakeTest", "per frame with 3 images on screen: ${"%.1f".format(total / reps)} ms " +
                "(find ${"%.1f".format(find / reps)}, crops ${"%.1f".format(prep / reps)}, 4 detector runs), 1 thread, unpinned")
        }
    }

    @Test
    fun classificationModeOnASyntheticScreen() {
        val clf = app.assets.list("models")?.firstOrNull { it.startsWith("ab-bnn-imagenet-reactnet18") }
        assumeTrue("no bundled ImageNet classifier", clf != null)
        val imgs = testAssets.open("screen_images.u8").use { it.readBytes() }
        val truth = testAssets.open("screen_labels.txt").bufferedReader().readLines().map { it.trim().toInt() }
        val labels = app.assets.open("imagenet_classes.txt").bufferedReader().readLines()
        val w = 756
        val h = 1683
        val frame = ByteBuffer.allocateDirect(w * h * 4)
        for (i in 0 until w * h * 4) frame.put(i, (-1).toByte())        // white page
        // image 0 shown twice as large (448 px, nearest), image 1 at 224 px, as on a feed
        val placed = listOf(intArrayOf(40, 200, 488, 648), intArrayOf(300, 900, 524, 1124))
        for ((k, r) in placed.withIndex()) {
            val s = (r[2] - r[0]) / 224
            for (y in r[1] until r[3]) for (x in r[0] until r[2]) {
                val sx = (x - r[0]) / s
                val sy = (y - r[1]) / s
                for (c in 0 until 3) frame.put((y * w + x) * 4 + c, imgs[k * 224 * 224 * 3 + (sy * 224 + sx) * 3 + c])
            }
        }
        val found = RectFinder().find(frame, w * 4, 4, w, h)
        Log.i("BitinoDeepfakeTest", "classification screen: found ${found.map { it.toList() }}")
        Bitino.load(app, "models/$clf", threads = 1).use { m ->
            val rc = RectClassifier(m, labels)
            for ((k, r) in placed.withIndex()) {
                val f = found.firstOrNull { abs(it[0] - r[0]) <= 4 && abs(it[1] - r[1]) <= 4 && abs(it[2] - r[2]) <= 4 && abs(it[3] - r[3]) <= 4 }
                assertTrue("image $k at ${r.toList()} not found in ${found.map { it.toList() }}", f != null)
                val t0 = SystemClock.elapsedRealtimeNanos()
                val (label, p) = rc.classify(frame, w * 4, 4, f!![0], f[1], f[2], f[3])
                val ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
                Log.i("BitinoDeepfakeTest", "classification image $k: '$label' ${"%.2f".format(p)} (truth '${labels[truth[k]]}'), ${"%.1f".format(ms)} ms incl. preprocessing, 1 thread")
                assertTrue(p in 0f..1f)
            }
        }
    }
}
