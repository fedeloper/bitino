package io.github.fedeloper.bitino.deepfake

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer

class ScreenCropTest {
    @Test
    fun geometryPortraitAndLandscape() {
        val p = Geometry(1116, 2484, 420)        // OnePlus Open cover screen, portrait
        assertEquals(1116, p.side)
        assertEquals(756, p.vw)                   // largest square = 3 x 252
        assertEquals(1683, p.vh)
        assertEquals(intArrayOf(3, 2, 1).toList(), (0 until 3).map { p.k(it) })
        assertEquals(0 + 14 * 3, p.cropX(0))
        assertEquals((756 - 504) / 2 + 14 * 2, p.cropX(1))
        assertEquals((756 - 252) / 2 + 14, p.cropX(2))
        assertEquals((1683 - 756) / 2 + 42, p.cropY(0))
        assertEquals((1683 - 252) / 2 + 14, p.cropY(2))
        val l = Geometry(2484, 1116, 420)
        assertEquals(1683, l.vw)
        assertEquals(756, l.vh)
        assertEquals((1683 - 756) / 2 + 42, l.cropX(0))
        // every crop stays inside the virtual display
        for (g in listOf(p, l, Geometry(720, 1600, 320), Geometry(2268, 2440, 400))) for (i in 0 until 3) {
            assertTrue(g.cropX(i) >= 0 && g.cropX(i) + 224 * g.k(i) <= g.vw)
            assertTrue(g.cropY(i) >= 0 && g.cropY(i) + 224 * g.k(i) <= g.vh)
        }
    }

    private fun frame(w: Int, h: Int, rowStride: Int, f: (Int, Int, Int) -> Int): ByteBuffer {
        val b = ByteBuffer.allocate(rowStride * h)
        for (y in 0 until h) for (x in 0 until w) for (c in 0 until 4) {
            b.put(y * rowStride + x * 4 + c, (if (c == 3) 255 else f(x, y, c)).toByte())
        }
        return b
    }

    @Test
    fun smallestSquareIsAnExactCropScaledBy255() {
        val g = Geometry(1116, 2484, 420)
        val sc = 2
        assertEquals(1, g.k(sc))
        val rowStride = g.vw * 4 + 32               // padded rows, as ImageReader may deliver
        val buf = frame(g.vw, g.vh, rowStride) { x, y, c -> (x * 7 + y * 3 + c * 50) and 0xff }
        val out = FloatArray(224 * 224 * 3)
        ScreenCrop.toInput(buf, rowStride, 4, g, sc, out, ByteArray(ScreenCrop.rowBytes(3, 4)))
        for (y in listOf(0, 1, 100, 223)) for (x in listOf(0, 5, 223)) for (c in 0 until 3) {
            val v = ((g.cropX(sc) + x) * 7 + (g.cropY(sc) + y) * 3 + c * 50) and 0xff
            assertEquals(v.toFloat() * (1f / 255f), out[(y * 224 + x) * 3 + c], 0f)
        }
    }

    @Test
    fun middleSquareIsARoundedBoxAverage() {
        val g = Geometry(2484, 1116, 420)
        val sc = 1
        assertEquals(2, g.k(sc))
        val rowStride = g.vw * 4
        val buf = frame(g.vw, g.vh, rowStride) { x, y, c -> (x * 13 + y * 29 + c * 71) % 251 }
        val out = FloatArray(224 * 224 * 3)
        ScreenCrop.toInput(buf, rowStride, 4, g, sc, out, ByteArray(ScreenCrop.rowBytes(3, 4)))
        for (y in listOf(0, 77, 223)) for (x in listOf(0, 150, 223)) for (c in 0 until 3) {
            var s = 0
            for (dy in 0 until 2) for (dx in 0 until 2) {
                s += ((g.cropX(sc) + 2 * x + dx) * 13 + (g.cropY(sc) + 2 * y + dy) * 29 + c * 71) % 251
            }
            assertEquals(((s + 2) / 4).toFloat() * (1f / 255f), out[(y * 224 + x) * 3 + c], 0f)
        }
    }

    @Test
    fun largestSquareIsA3x3Average() {
        val g = Geometry(1116, 2484, 420)
        assertEquals(3, g.k(0))
        val rowStride = g.vw * 4
        val buf = frame(g.vw, g.vh, rowStride) { x, y, c -> (x * 5 + y * 11 + c * 31) % 241 }
        val out = FloatArray(224 * 224 * 3)
        ScreenCrop.toInput(buf, rowStride, 4, g, 0, out, ByteArray(ScreenCrop.rowBytes(3, 4)))
        for (y in listOf(0, 223)) for (x in listOf(0, 111, 223)) for (c in 0 until 3) {
            var s = 0
            for (dy in 0 until 3) for (dx in 0 until 3) s += ((g.cropX(0) + 3 * x + dx) * 5 + (g.cropY(0) + 3 * y + dy) * 11 + c * 31) % 241
            assertEquals(((s + 4) / 9).toFloat() * (1f / 255f), out[(y * 224 + x) * 3 + c], 0f)
        }
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
