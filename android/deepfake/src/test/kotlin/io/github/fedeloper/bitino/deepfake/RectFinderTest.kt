package io.github.fedeloper.bitino.deepfake

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import kotlin.math.sin

/** Synthetic phone screens (756 x 1683 capture, light and dark): photos must be found where they are, UI must not. */
class RectFinderTest {
    private val w = 756
    private val h = 1683

    private class Screen(val buf: ByteBuffer, val photos: List<IntArray>)

    private fun hash(x: Int, y: Int, s: Int): Int {
        var v = x * 374761393 + y * 668265263 + s * 2246822519.toInt()
        v = (v xor (v ushr 13)) * 1274126177
        return (v xor (v ushr 16)) and 0xff
    }

    /** A "photo": smooth colour field plus fine texture (thousands of colours, no flat rows). */
    private fun photo(x: Int, y: Int, c: Int, seed: Int): Int {
        val v = 128 + 70 * sin(x * (0.013 + 0.004 * c) + seed) + 40 * sin(y * (0.021 - 0.003 * c) + 2 * seed) +
            (hash(x, y, seed + c) - 128) * 0.15
        return v.toInt().coerceIn(0, 255)
    }

    private fun screen(dark: Boolean, seed: Int): Screen {
        val bg = if (dark) intArrayOf(18, 18, 18) else intArrayOf(255, 255, 255)
        val ink = if (dark) intArrayOf(230, 230, 230) else intArrayOf(20, 20, 20)
        val card = if (dark) intArrayOf(40, 40, 44) else intArrayOf(242, 242, 246)
        val px = Array(h) { Array(w) { bg } }
        fun fill(x0: Int, y0: Int, x1: Int, y1: Int, col: IntArray) {
            for (y in maxOf(0, y0) until minOf(h, y1)) for (x in maxOf(0, x0) until minOf(w, x1)) px[y][x] = col
        }
        val photos = ArrayList<IntArray>()
        fun putPhoto(x0: Int, y0: Int, x1: Int, y1: Int, s: Int) {
            for (y in maxOf(0, y0) until minOf(h, y1)) for (x in x0 until x1) {
                px[y][x] = intArrayOf(photo(x, y, 0, s), photo(x, y, 1, s), photo(x, y, 2, s))
            }
            photos += intArrayOf(x0, maxOf(0, y0), x1, minOf(h, y1))
        }
        fun text(x0: Int, y0: Int, lines: Int, xMax: Int = w - 24) {
            for (l in 0 until lines) {
                var x = x0
                val y = y0 + l * 34
                while (true) {
                    val gw = 8 + hash(x, y, seed) % 10
                    if (x + gw > xMax) break
                    fill(x, y, x + gw, y + 18, ink)   // a "glyph"
                    x += gw + 3 + (if (hash(x, y, 7) % 5 == 0) 12 else 0)
                }
            }
        }
        // a full-width photo scrolled partly off the top, text, an inset photo in a card, buttons, three thumbnails
        putPhoto(0, -120, w, 420, seed)
        text(24, 450, 4)
        fill(28, 610, 728, 1080, card)
        putPhoto(48, 630, 708, 980, seed + 1)
        text(48, 1000, 2, xMax = 708)
        for (i in 0 until 3) fill(24 + i * 240, 1110, 224 + i * 240, 1170, intArrayOf(66, 133, 244))
        for (i in 0 until 3) putPhoto(24 + i * 236, 1200, 24 + i * 236 + 216, 1416, seed + 2 + i)
        text(24, 1440, 5)
        val buf = ByteBuffer.allocate(w * h * 4)
        for (y in 0 until h) for (x in 0 until w) {
            val p = px[y][x]
            val o = (y * w + x) * 4
            buf.put(o, p[0].toByte()); buf.put(o + 1, p[1].toByte()); buf.put(o + 2, p[2].toByte()); buf.put(o + 3, (-1).toByte())
        }
        return Screen(buf, photos)
    }

    private fun iou(a: IntArray, b: IntArray): Double {
        val ix = maxOf(0, minOf(a[2], b[2]) - maxOf(a[0], b[0])).toDouble() * maxOf(0, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
        return ix / ((a[2] - a[0]).toDouble() * (a[3] - a[1]) + (b[2] - b[0]).toDouble() * (b[3] - b[1]) - ix)
    }

    @Test
    fun findsEveryPhotoAndNothingElse() {
        for (dark in listOf(false, true)) for (seed in 0 until 3) {
            val s = screen(dark, seed)
            val found = RectFinder(maxRects = 10).find(s.buf, w * 4, 4, w, h)
            for (p in s.photos) {
                val best = found.maxOf { iou(it, p) }
                assertTrue("dark=$dark seed=$seed photo ${p.toList()}: best IoU $best in ${found.map { it.toList() }}", best >= 0.97)
            }
            assertEquals("dark=$dark seed=$seed: ${found.map { it.toList() }}", s.photos.size, found.size)
        }
    }

    @Test
    fun keepsTheLargestFirst() {
        val s = screen(false, 0)
        val found = RectFinder(maxRects = 2).find(s.buf, w * 4, 4, w, h)
        assertEquals(2, found.size)
        assertTrue(iou(found[0], s.photos[0]) >= 0.97)   // full-width photo, 756 x 420 visible
        assertTrue(iou(found[1], s.photos[1]) >= 0.97)   // inset photo 660 x 350
    }

    @Test
    fun drawnOutlinesDoNotMoveTheRectangles() {
        // the overlay's dashed two-colour outline, 3 capture pixels wide just inside each found image, as captured
        for (dark in listOf(false, true)) {
            val s = screen(dark, 1)
            val finder = RectFinder(maxRects = 10)
            val before = finder.find(s.buf, w * 4, 4, w, h).map { it.toList() }
            for (r in before) {
                val inset = 1
                for (t in 0 until 3) {
                    val x0 = r[0] + inset + t; val y0 = r[1] + inset + t; val x1 = r[2] - 1 - inset - t; val y1 = r[3] - 1 - inset - t
                    fun put(x: Int, y: Int, along: Int) {
                        val green = (along / 10) % 2 == 0
                        val o = (y * w + x) * 4
                        s.buf.put(o, (if (green) 0 else 255).toByte()); s.buf.put(o + 1, (if (green) 230 else 255).toByte())
                        s.buf.put(o + 2, (if (green) 90 else 255).toByte())
                    }
                    for (x in x0..x1) { put(x, y0, x); put(x, y1, x) }
                    for (y in y0..y1) { put(x0, y, y); put(x1, y, y) }
                }
            }
            val after = finder.find(s.buf, w * 4, 4, w, h).map { it.toList() }
            assertEquals("dark=$dark", before, after)

        }
    }

    @Test
    fun aThinLineAcrossTheGapsDoesNotGlueAGrid() {
        // a masonry grid of photos pressed against a toolbar shadow (1 grey line across the screen), as in a browser
        val buf = ByteBuffer.allocate(w * h * 4)
        for (i in 0 until w * h) { buf.put(i * 4, -1); buf.put(i * 4 + 1, -1); buf.put(i * 4 + 2, -1); buf.put(i * 4 + 3, -1) }
        fun set(x: Int, y: Int, c: IntArray) { val o = (y * w + x) * 4; for (k in 0 until 3) buf.put(o + k, c[k].toByte()) }
        for (x in 0 until w) for (y in 200..201) set(x, y, intArrayOf(200, 200, 205))
        val photos = listOf(intArrayOf(4, 202, 248, 700), intArrayOf(256, 202, 500, 520), intArrayOf(508, 202, 752, 600),
            intArrayOf(256, 560, 500, 900), intArrayOf(4, 740, 248, 1000))
        for ((i, p) in photos.withIndex()) for (y in p[1] until p[3]) for (x in p[0] until p[2]) {
            set(x, y, intArrayOf(photo(x, y, 0, i), photo(x, y, 1, i), photo(x, y, 2, i)))
        }
        val found = RectFinder(maxRects = 10).find(buf, w * 4, 4, w, h)
        for (p in photos) {
            val best = found.maxOf { iou(it, p) }
            assertTrue("photo ${p.toList()}: best IoU $best in ${found.map { it.toList() }}", best >= 0.97)
        }
        assertEquals(found.map { it.toList() }.toString(), photos.size, found.size)
    }

    /** Draws what the overlay draws for a found image, as captured: dashed outline just inside it, optional red tint. */
    private fun drawOverlay(buf: ByteBuffer, sw: Int, r: IntArray, tint: Boolean) {
        fun put(x: Int, y: Int, c: IntArray) { val o = (y * sw + x) * 4; for (k in 0 until 3) buf.put(o + k, c[k].toByte()) }
        fun get(x: Int, y: Int, k: Int) = buf.get((y * sw + x) * 4 + k).toInt() and 0xff
        if (tint) for (y in r[1] + 2 until r[3] - 2) for (x in r[0] + 2 until r[2] - 2) {
            put(x, y, intArrayOf((get(x, y, 0) * 135 + 255 * 120) / 255, get(x, y, 1) * 135 / 255, get(x, y, 2) * 135 / 255))
        }
        // the P(fake) pill in the top-left corner: a dark rounded box with white "text"
        if (r[2] - r[0] > 60 && r[3] - r[1] > 60) for (y in r[1] + 5 until r[1] + 17) for (x in r[0] + 5 until r[0] + 40) {
            val corner = (y == r[1] + 5 || y == r[1] + 16) && (x == r[0] + 5 || x == r[0] + 39)
            if (!corner) put(x, y, if (y in r[1] + 8..r[1] + 13 && (x - r[0]) % 4 < 2) intArrayOf(255, 255, 255) else intArrayOf(0, 140, 60))
        }
        for (t in 2 until 4) {
            val x0 = r[0] + t; val y0 = r[1] + t; val x1 = r[2] - 1 - t; val y1 = r[3] - 1 - t
            fun dash(x: Int, y: Int, along: Int) =
                put(x, y, if ((along / 10) % 2 == 1) intArrayOf(255, 255, 255) else if (tint) intArrayOf(255, 30, 30) else intArrayOf(0, 230, 90))
            for (x in x0..x1) { dash(x, y0, x); dash(x, y1, x) }
            for (y in y0..y1) { dash(x0, y, y); dash(x1, y, y) }
        }
    }

    @Test
    fun sameInputSameOutputAndNoStateCarriesOver() {
        val a = screen(false, 0)
        val b = screen(true, 2)
        val reused = RectFinder(maxRects = 24)
        val first = reused.find(a.buf, w * 4, 4, w, h).map { it.toList() }
        assertEquals(first, reused.find(a.buf, w * 4, 4, w, h).map { it.toList() })
        reused.find(b.buf, w * 4, 4, w, h)
        assertEquals(first, reused.find(a.buf, w * 4, 4, w, h).map { it.toList() })
        assertEquals(first, RectFinder(maxRects = 24).find(a.buf, w * 4, 4, w, h).map { it.toList() })
    }

    @Test
    fun theOverlayRedrawnFrameAfterFrameKeepsTheResult() {
        // frame after frame the app outlines what it found (one image tinted as fake): the result must stay the clean one
        for (dark in listOf(false, true)) {
            val s = screen(dark, 0)
            val finder = RectFinder(maxRects = 24)
            val clean = finder.find(s.buf, w * 4, 4, w, h).map { it.toList() }
            val original = s.buf.array().copyOf()
            var cur = clean
            repeat(4) { frame ->
                // each captured frame = the screen + the overlay drawn for the previous result (one layer, not cumulative)
                val buf = ByteBuffer.wrap(original.copyOf())
                for ((i, r) in cur.withIndex()) drawOverlay(buf, w, r.toIntArray(), tint = i == 1 && frame > 0)
                cur = finder.find(buf, w * 4, 4, w, h).map { it.toList() }
                assertEquals("dark=$dark frame $frame", clean, cur)
            }
        }
    }

    @Test
    fun aWrongScreenSizedOutlineDoesNotGlueTheScreen() {
        val s = screen(false, 2)
        val finder = RectFinder(maxRects = 24)
        val clean = finder.find(s.buf, w * 4, 4, w, h).map { it.toList() }
        drawOverlay(s.buf, w, intArrayOf(0, 0, w, h), tint = false)
        assertEquals(clean, finder.find(s.buf, w * 4, 4, w, h).map { it.toList() })
    }

    /** Local check on a real screenshot (never committed): the clean result, and the result with the overlay drawn in. */
    @Test
    fun realScreenshotIfGiven() {
        val path = System.getenv("BITINO_SCREENSHOT") ?: return
        val sw = System.getenv("BITINO_SCREENSHOT_W").toInt()
        val sh = System.getenv("BITINO_SCREENSHOT_H").toInt()
        val rgb = java.io.File(path).readBytes()
        val buf = ByteBuffer.allocate(sw * sh * 4)
        for (i in 0 until sw * sh) { for (k in 0 until 3) buf.put(i * 4 + k, rgb[i * 3 + k]); buf.put(i * 4 + 3, -1) }
        val finder = RectFinder(maxRects = 24)
        val found = finder.find(buf, sw * 4, 4, sw, sh)
        println("REAL_SCREENSHOT ${found.size} ${found.map { it.toList() }}")
        var cur = found.map { it.toList() }
        val original = buf.array().copyOf()
        repeat(3) { frame ->
            val b2 = ByteBuffer.wrap(original.copyOf())
            for ((i, r) in cur.withIndex()) drawOverlay(b2, sw, r.toIntArray(), tint = i == 0 && frame > 0)
            cur = finder.find(b2, sw * 4, 4, sw, sh).map { it.toList() }
            println("REAL_SCREENSHOT frame $frame with overlay: same as clean = ${cur == found.map { it.toList() }}")
            assertEquals(found.map { it.toList() }, cur)
        }
    }
}
