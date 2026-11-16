package io.github.fedeloper.bitino.deepfake

import kotlin.math.abs

/**
 * Follows the images found on screen from frame to frame (capture pixels).
 *
 * - An image whose four edges all moved by at most [tol] pixels is the same image: it keeps its score, its place in the
 *   rotation and, if it was flagged as fake, its flag (and its box stays where it was flagged).
 * - An image that moved further (but still overlaps its old place by IoU >= 0.5) is a different view of it, for
 *   instance after a scroll: its state is reset and it is analysed again.
 * - An image not found for more than [maxMissed] frames in a row is gone.
 */
internal class Tracker(private val tol: Int, private val maxMissed: Int = 2) {
    class Track(var rect: IntArray) {
        val alert = AlertState()
        /** Last P(fake); < 0 = not analysed yet. */
        var pFake = -1f
        /** Rotation stamp of its last analysis (-1 = never). */
        var lastRun = -1L
        /** Frames in a row in the same place. */
        var still = 0
        var missed = 0
        /** Flagged as fake: stays until the image moves beyond the tolerance or disappears. */
        var fake = false
        /** Found in the current frame. */
        var seen = true
    }

    val tracks = ArrayList<Track>()

    fun same(a: IntArray, b: IntArray): Boolean = (0..3).all { abs(a[it] - b[it]) <= tol }

    /** Matches the rectangles of a new frame; returns true if the images on screen changed (new, moved or gone). */
    fun update(rects: List<IntArray>): Boolean {
        var changed = false
        for (t in tracks) t.seen = false
        for (rc in rects) {
            val t = tracks.filter { !it.seen }.maxByOrNull { iou(it.rect, rc) }?.takeIf { same(it.rect, rc) || iou(it.rect, rc) >= 0.5f }
            if (t == null) {
                tracks += Track(rc)
                changed = true
                continue
            }
            t.seen = true
            t.missed = 0
            if (same(t.rect, rc)) {
                t.still++
                if (!t.fake) t.rect = rc
            } else {
                t.rect = rc
                t.still = 0
                t.pFake = -1f
                t.lastRun = -1
                t.alert.reset()
                t.fake = false
                changed = true
            }
        }
        val it = tracks.iterator()
        while (it.hasNext()) {
            val t = it.next()
            if (!t.seen && ++t.missed > maxMissed) {
                it.remove()
                changed = true
            }
        }
        return changed
    }

    companion object {
        fun iou(a: IntArray, b: IntArray): Float {
            val ix = maxOf(0, minOf(a[2], b[2]) - maxOf(a[0], b[0])).toLong() * maxOf(0, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
            val u = (a[2] - a[0]).toLong() * (a[3] - a[1]) + (b[2] - b[0]).toLong() * (b[3] - b[1]) - ix
            return if (u > 0) ix.toFloat() / u else 0f
        }
    }
}
