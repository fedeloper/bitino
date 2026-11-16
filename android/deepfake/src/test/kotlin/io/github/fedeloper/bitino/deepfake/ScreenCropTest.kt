package io.github.fedeloper.bitino.deepfake

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import kotlin.math.abs

class ScreenCropTest {
    private fun frame(w: Int, h: Int, rowStride: Int, f: (Int, Int, Int) -> Int): ByteBuffer {
        val b = ByteBuffer.allocate(rowStride * h)
        for (y in 0 until h) for (x in 0 until w) for (c in 0 until 4) {
            b.put(y * rowStride + x * 4 + c, (if (c == 3) 255 else f(x, y, c)).toByte())
        }
        return b
    }

    @Test
    fun geometryPortraitAndLandscape() {
        val p = Geometry(1116, 2484, 420)        // OnePlus Open cover screen, portrait
        assertEquals(1116, p.side)
        assertEquals(4, p.k0)                     // centre square = 4 x 252 = 1008 (largest multiple of 252 that fits)
        assertEquals(1008, p.vw)
        assertEquals(2244, p.vh)
        assertEquals(56, p.cropX(0))
        assertEquals((2244 - 1008) / 2 + 56, p.cropY(0))
        assertEquals(1116f / 1008f, p.toScreen, 1e-6f)
        val inner = Geometry(2268, 2440, 520)    // inner screen: 2268 = 9 x 252, captured 1:1
        assertEquals(9, inner.k0)
        assertEquals(2268, inner.vw)
        assertEquals(2440, inner.vh)
        assertEquals(1f, inner.toScreen, 0f)
        val l = Geometry(2484, 1116, 420)
        assertEquals(2244, l.vw)
        assertEquals(1008, l.vh)
        for (g in listOf(p, l, inner, Geometry(720, 1600, 320), Geometry(200, 400, 160))) {
            assertTrue(g.cropX(0) >= 0 && g.cropX(0) + 224 * g.k0 <= g.vw)
            assertTrue(g.cropY(0) >= 0 && g.cropY(0) + 224 * g.k0 <= g.vh)
        }
    }

    @Test
    fun centreSquareIsABoxAverage() {
        val g = Geometry(756, 1683, 420)          // 756 = 3 x 252: a 3x3 box
        val rowStride = g.vw * 4 + 32               // padded rows, as ImageReader may deliver
        val buf = frame(g.vw, g.vh, rowStride) { x, y, c -> (x * 5 + y * 11 + c * 31) % 241 }
        val out = FloatArray(224 * 224 * 3)
        ScreenCrop.toInput(buf, rowStride, 4, g, 0, out, ByteArray(ScreenCrop.rowBytes(3, 4)))
        for (y in listOf(0, 100, 223)) for (x in listOf(0, 111, 223)) for (c in 0 until 3) {
            var s = 0
            for (dy in 0 until 3) for (dx in 0 until 3) s += ((g.cropX(0) + 3 * x + dx) * 5 + (g.cropY(0) + 3 * y + dy) * 11 + c * 31) % 241
            assertEquals(((s + 4) / 9).toFloat() * (1f / 255f), out[(y * 224 + x) * 3 + c], 0f)
        }
    }

    @Test
    fun resampleMatchesPil() {
        // android/tools/make_resample_fixture.py: PIL crop -> resize(BILINEAR) -> centre crop, sampled
        val lines = javaClass.classLoader!!.getResourceAsStream("resample_fixture.txt")!!.bufferedReader().readLines()
            .filter { !it.startsWith("#") }
        val w = 900
        val h = 1200
        val buf = frame(w, h, w * 4) { x, y, c -> ((x.toLong() * x / 7 + y * 3 + c * 60 + (x.toLong() * y) % 17) % 256).toInt() }
        val out = FloatArray(224 * 224 * 3)
        var cases = 0
        for (i in lines.indices step 2) {
            val head = lines[i].split(" ")
            val rect = head.subList(1, 5).map { it.toInt() }
            assertEquals(listOf(head[6].toInt(), head[7].toInt()), Resample.resized(rect[2] - rect[0], rect[3] - rect[1]).toList())
            val ref = lines[i + 1].split(" ").map { it.toInt() }
            Resample.toInput(buf, w * 4, 4, rect[0], rect[1], rect[2], rect[3], out)
            var k = 0
            var exact = 0
            for (y in 0 until 224 step 9) for (x in 0 until 224 step 9) for (c in 0 until 3) {
                val v = Math.round(out[(y * 224 + x) * 3 + c] * 255f)
                assertTrue("rect $rect ($x, $y, $c): $v vs PIL ${ref[k]}", abs(v - ref[k]) <= 1)
                if (v == ref[k]) exact++
                k++
            }
            assertEquals(ref.size, k)
            // PIL works in fixed point (22 fractional bits) with its own rounding: within one level, nearly always exact
            assertTrue("rect $rect: only $exact of $k exact", exact >= 0.95 * k)
            cases++
        }
        assertEquals(5, cases)
    }

    @Test
    fun halfEvenIsPythonRound() {
        assertEquals(0, Resample.halfEven(1))   // round(0.5) = 0
        assertEquals(2, Resample.halfEven(3))   // round(1.5) = 2
        assertEquals(2, Resample.halfEven(5))   // round(2.5) = 2
        assertEquals(98, Resample.halfEven(196))
    }

    @Test
    fun alertNeedsThreeFakeFramesThenBlinksUntilARealOne() {
        val a = AlertState(3)
        assertFalse(a.update(0.9f, 0.7f))
        assertFalse(a.update(0.9f, 0.7f))
        assertTrue(a.update(0.9f, 0.7f))
        assertTrue(a.update(0.8f, 0.7f))          // still fake after the alert: again at once
        assertFalse(a.update(0.2f, 0.7f))         // a real frame ends it
        assertFalse(a.update(0.9f, 0.7f))
    }
}
