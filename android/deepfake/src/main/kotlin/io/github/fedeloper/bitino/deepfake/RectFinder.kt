package io.github.fedeloper.bitino.deepfake

import java.nio.ByteBuffer

/**
 * Finds the images on a screen capture with classical layout analysis (no learned model): a recursive X-Y cut.
 *
 * UI backgrounds are flat, so whole rows or columns of one colour separate the elements of a screen (margins, the gaps
 * between posts, the space between a thumbnail and its text). The capture is halved (2×2 average). Then each block,
 * starting from the whole screen, is split along its uniform rows; a block without uniform rows is split along its
 * uniform columns. Splitting trims the uniform margins too, so a card shrinks to its own background and is then split
 * inside. A block with neither is a leaf: a photo, a text line, an icon, a button.
 *
 * Leaves of at least [MIN_SIDE] half-resolution pixels per side that look like photographs are kept:
 * - at least [MIN_COLOURS] distinct colours (4 bits per channel, each on at least 0.1 % of the pixels);
 * - no luminance band (8 levels wide) on more than [MAX_MODE] of the pixels.
 * Text, buttons and flat graphics have few colours and are dominated by their background.
 *
 * Tuned on synthetic feed screens (text, cards, buttons, full-width / inset / thumbnail photos, light and dark): 92.5 % of
 * the photos found with IoU >= 0.8 (mean IoU 0.988); misses are mostly photos whose own border is a flat band (letterbox
 * bars), found slightly smaller.
 */
internal class RectFinder(private val maxRects: Int = 4) {
    companion object {
        const val UNIFORM = 10       // a row / column is uniform if each channel spans at most this inside the block
        const val MIN_SIDE = 32      // half-resolution pixels (64 capture pixels)
        const val MIN_COLOURS = 24
        const val MAX_MODE = 0.75f
        private const val MAX_DEPTH = 48
    }

    private var hw = 0
    private var hh = 0
    private var r = IntArray(0)
    private var g = IntArray(0)
    private var b = IntArray(0)
    private var row0 = ByteArray(0)
    private var row1 = ByteArray(0)
    private val leaves = ArrayList<IntArray>()
    private val colourCount = IntArray(4096)
    private val lumHist = IntArray(32)

    /**
     * Image rectangles of the RGBA frame ([w]×[h] pixels), largest first, at most [maxRects]: {x0, y0, x1, y1} in frame
     * pixels (end exclusive).
     */
    fun find(buf: ByteBuffer, rowStride: Int, pixelStride: Int, w: Int, h: Int): List<IntArray> {
        halve(buf, rowStride, pixelStride, w, h)
        leaves.clear()
        cut(0, 0, hw, hh, 0)
        val kept = leaves.filter { (it[2] - it[0]) >= MIN_SIDE && (it[3] - it[1]) >= MIN_SIDE && photoLike(it) }
            .sortedByDescending { (it[2] - it[0]).toLong() * (it[3] - it[1]) }
            .take(maxRects)
        return kept.map { intArrayOf(it[0] * 2, it[1] * 2, it[2] * 2, it[3] * 2) }
    }

    /** 2×2 average, rounded (integer RGB planes). */
    private fun halve(buf: ByteBuffer, rowStride: Int, ps: Int, w: Int, h: Int) {
        hw = w / 2
        hh = h / 2
        val n = hw * hh
        if (r.size < n) { r = IntArray(n); g = IntArray(n); b = IntArray(n) }
        val need = (w - 1) * ps + 4
        if (row0.size < need) { row0 = ByteArray(need); row1 = ByteArray(need) }
        val src = buf.duplicate()
        for (y in 0 until hh) {
            src.position(2 * y * rowStride); src.get(row0, 0, need)
            src.position((2 * y + 1) * rowStride); src.get(row1, 0, need)
            var o = y * hw
            var p = 0
            for (x in 0 until hw) {
                val q = p + ps
                r[o] = ((row0[p].toInt() and 0xff) + (row0[q].toInt() and 0xff) + (row1[p].toInt() and 0xff) + (row1[q].toInt() and 0xff) + 2) shr 2
                g[o] = ((row0[p + 1].toInt() and 0xff) + (row0[q + 1].toInt() and 0xff) + (row1[p + 1].toInt() and 0xff) + (row1[q + 1].toInt() and 0xff) + 2) shr 2
                b[o] = ((row0[p + 2].toInt() and 0xff) + (row0[q + 2].toInt() and 0xff) + (row1[p + 2].toInt() and 0xff) + (row1[q + 2].toInt() and 0xff) + 2) shr 2
                o++
                p += 2 * ps
            }
        }
    }

    private fun rowUniform(y: Int, x0: Int, x1: Int): Boolean {
        var o = y * hw + x0
        var rl = 255; var rh = 0; var gl = 255; var gh = 0; var bl = 255; var bh = 0
        for (x in x0 until x1) {
            val rv = r[o]; val gv = g[o]; val bv = b[o]
            if (rv < rl) rl = rv; if (rv > rh) rh = rv
            if (gv < gl) gl = gv; if (gv > gh) gh = gv
            if (bv < bl) bl = bv; if (bv > bh) bh = bv
            if (rh - rl > UNIFORM || gh - gl > UNIFORM || bh - bl > UNIFORM) return false
            o++
        }
        return true
    }

    private fun colUniform(x: Int, y0: Int, y1: Int): Boolean {
        var o = y0 * hw + x
        var rl = 255; var rh = 0; var gl = 255; var gh = 0; var bl = 255; var bh = 0
        for (y in y0 until y1) {
            val rv = r[o]; val gv = g[o]; val bv = b[o]
            if (rv < rl) rl = rv; if (rv > rh) rh = rv
            if (gv < gl) gl = gv; if (gv > gh) gh = gv
            if (bv < bl) bl = bv; if (bv > bh) bh = bv
            if (rh - rl > UNIFORM || gh - gl > UNIFORM || bh - bl > UNIFORM) return false
            o += hw
        }
        return true
    }

    /** Non-uniform runs [a, b) of 0 until n, given a uniformity predicate. */
    private inline fun runs(n: Int, uniform: (Int) -> Boolean): List<IntArray> {
        val out = ArrayList<IntArray>()
        var i = 0
        while (i < n) {
            if (uniform(i)) { i++; continue }
            var j = i
            while (j < n && !uniform(j)) j++
            out += intArrayOf(i, j)
            i = j
        }
        return out
    }

    private fun cut(x0: Int, y0: Int, x1: Int, y1: Int, depth: Int) {
        if (x1 - x0 < 4 || y1 - y0 < 4 || depth > MAX_DEPTH) return
        val rs = runs(y1 - y0) { rowUniform(y0 + it, x0, x1) }
        if (rs.isEmpty()) return
        if (rs.size > 1 || rs[0][0] != 0 || rs[0][1] != y1 - y0) {
            for (s in rs) cut(x0, y0 + s[0], x1, y0 + s[1], depth + 1)
            return
        }
        val cs = runs(x1 - x0) { colUniform(x0 + it, y0, y1) }
        if (cs.isEmpty()) return
        if (cs.size > 1 || cs[0][0] != 0 || cs[0][1] != x1 - x0) {
            for (s in cs) cut(x0 + s[0], y0, x0 + s[1], y1, depth + 1)
            return
        }
        leaves += intArrayOf(x0, y0, x1, y1)
    }

    private fun photoLike(c: IntArray): Boolean {
        java.util.Arrays.fill(colourCount, 0)
        java.util.Arrays.fill(lumHist, 0)
        var n = 0
        for (y in c[1] until c[3]) {
            var o = y * hw + c[0]
            for (x in c[0] until c[2]) {
                val rv = r[o]; val gv = g[o]; val bv = b[o]
                colourCount[((rv shr 4) shl 8) or ((gv shr 4) shl 4) or (bv shr 4)]++
                lumHist[((rv * 77 + gv * 150 + bv * 29) shr 8) shr 3]++
                n++
                o++
            }
        }
        if (lumHist.max() > MAX_MODE * n) return false
        val minCount = maxOf(2, n / 1000)
        var colours = 0
        for (v in colourCount) if (v >= minCount) colours++
        return colours >= MIN_COLOURS
    }
}
