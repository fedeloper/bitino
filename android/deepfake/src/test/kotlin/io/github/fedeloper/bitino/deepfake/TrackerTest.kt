package io.github.fedeloper.bitino.deepfake

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class TrackerTest {
    private fun r(x0: Int, y0: Int, x1: Int, y1: Int) = intArrayOf(x0, y0, x1, y1)

    @Test
    fun aFewPixelsIsTheSameImage() {
        val t = Tracker(tol = 8)
        assertTrue(t.update(listOf(r(10, 100, 300, 400))))          // new image: changed
        val a = t.tracks[0]
        a.pFake = 0.3f; a.lastRun = 5
        assertFalse(t.update(listOf(r(14, 96, 305, 403))))          // moved by up to 5 px: same image, nothing changed
        assertSame(a, t.tracks[0])
        assertEquals(0.3f, a.pFake, 0f)
        assertEquals(5L, a.lastRun)
        assertEquals(1, a.still)
    }

    @Test
    fun aScrollIsANewViewOfTheImage() {
        val t = Tracker(tol = 8)
        t.update(listOf(r(10, 100, 300, 400)))
        val a = t.tracks[0]
        a.pFake = 0.9f; a.lastRun = 3; a.fake = true
        assertTrue(t.update(listOf(r(10, 60, 300, 360))))           // scrolled 40 px: overlaps, but moved
        assertSame(a, t.tracks[0])
        assertFalse(a.fake)
        assertEquals(-1f, a.pFake, 0f)
        assertEquals(-1L, a.lastRun)
    }

    @Test
    fun aFlagStaysWhileTheImageStaysAndItsBoxDoesNotDrift() {
        val t = Tracker(tol = 8)
        t.update(listOf(r(10, 100, 300, 400)))
        val a = t.tracks[0]
        a.fake = true
        repeat(5) { assertFalse(t.update(listOf(r(12 + it % 2, 101, 302, 399)))) }
        assertTrue(a.fake)
        assertEquals(listOf(10, 100, 300, 400), a.rect.toList())    // the flagged box stays where it was flagged
    }

    @Test
    fun anImageIsGoneOnlyAfterAFewMissedFrames() {
        val t = Tracker(tol = 8, maxMissed = 2)
        t.update(listOf(r(10, 100, 300, 400), r(10, 500, 300, 700)))
        val b = t.tracks[1]
        b.fake = true
        assertFalse(t.update(listOf(r(10, 100, 300, 400))))         // missed once
        assertFalse(t.update(listOf(r(10, 100, 300, 400))))         // twice
        assertTrue(t.tracks.contains(b) && b.fake)
        assertTrue(t.update(listOf(r(10, 100, 300, 400))))          // third time: gone
        assertFalse(t.tracks.contains(b))
    }

    @Test
    fun aNewImageIsAChange() {
        val t = Tracker(tol = 8)
        t.update(listOf(r(10, 100, 300, 400)))
        assertTrue(t.update(listOf(r(10, 100, 300, 400), r(400, 100, 700, 400))))
        assertEquals(2, t.tracks.size)
    }
}
