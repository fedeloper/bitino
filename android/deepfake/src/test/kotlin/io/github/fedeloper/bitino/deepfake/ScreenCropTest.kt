package io.github.fedeloper.bitino.deepfake

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer

class ScreenCropTest {
    @Test
    fun geometryPortraitAndLandscape() {
        val p = Geometry(1080, 2400, 420)        // OnePlus-like portrait
        assertEquals(1080, p.side)
        assertEquals(4, p.k)
        assertEquals(1008, p.vw)
        assertEquals(2240, p.vh)
        assertEquals((1008 - 1008) / 2 + 14 * 4, p.cropX)
        assertEquals((2240 - 1008) / 2 + 14 * 4, p.cropY)
        val l = Geometry(2400, 1080, 420)
        assertEquals(2240, l.vw)
        assertEquals(1008, l.vh)
        assertEquals((2240 - 1008) / 2 + 56, l.cropX)
        val small = Geometry(240, 320, 120)       // smaller than 252: upscaled, k = 1
        assertEquals(1, small.k)
        assertEquals(252, small.vw)
        assertEquals(336, small.vh)
    }

    private fun frame(w: Int, h: Int, rowStride: Int, f: (Int, Int, Int) -> Int): ByteBuffer {
        val b = ByteBuffer.allocate(rowStride * h)
        for (y in 0 until h) for (x in 0 until w) for (c in 0 until 4) {
            b.put(y * rowStride + x * 4 + c, (if (c == 3) 255 else f(x, y, c)).toByte())
        }
        return b
    }

    @Test
    fun k1IsAnExactCropScaledBy255() {
        val g = Geometry(252, 300, 160)
        assertEquals(1, g.k)
        val rowStride = g.vw * 4 + 32               // padded rows, as ImageReader may deliver
        val buf = frame(g.vw, g.vh, rowStride) { x, y, c -> (x * 7 + y * 3 + c * 50) and 0xff }
        val out = FloatArray(224 * 224 * 3)
        ScreenCrop.toInput(buf, rowStride, 4, g, out, ByteArray(ScreenCrop.rowBytes(g, 4)))
        for (y in listOf(0, 1, 100, 223)) for (x in listOf(0, 5, 223)) for (c in 0 until 3) {
            val v = ((g.cropX + x) * 7 + (g.cropY + y) * 3 + c * 50) and 0xff
            assertEquals(v.toFloat() * (1f / 255f), out[(y * 224 + x) * 3 + c], 0f)
        }
    }

    @Test
    fun k2IsARoundedBoxAverage() {
        val g = Geometry(504, 600, 320)
        assertEquals(2, g.k)
        val rowStride = g.vw * 4
        val buf = frame(g.vw, g.vh, rowStride) { x, y, c -> (x * 13 + y * 29 + c * 71) % 251 }
        val out = FloatArray(224 * 224 * 3)
        ScreenCrop.toInput(buf, rowStride, 4, g, out, ByteArray(ScreenCrop.rowBytes(g, 4)))
        for (y in listOf(0, 77, 223)) for (x in listOf(0, 150, 223)) for (c in 0 until 3) {
            var s = 0
            for (dy in 0 until 2) for (dx in 0 until 2) {
                s += ((g.cropX + 2 * x + dx) * 13 + (g.cropY + 2 * y + dy) * 29 + c * 71) % 251
            }
            assertEquals(((s + 2) / 4).toFloat() * (1f / 255f), out[(y * 224 + x) * 3 + c], 0f)
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
