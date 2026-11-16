package io.github.fedeloper.bitino.deepfake

/**
 * When to flash the screen red: [needed] consecutive analysed frames with P(fake) >= threshold. While the alert is
 * on, the overlay covers the analysed square, so those frames are skipped; when it goes off, one more fake frame
 * brings it back (the screen keeps blinking red while the fake image stays), and one frame below the threshold ends it.
 */
internal class AlertState(private val needed: Int = 3) {
    var streak: Int = 0
        private set

    /** Feeds one analysed frame; returns true if the red alert should be shown now. */
    fun update(pFake: Float, threshold: Float): Boolean {
        streak = if (pFake >= threshold) streak + 1 else 0
        if (streak >= needed) {
            streak = needed - 1
            return true
        }
        return false
    }

    fun reset() {
        streak = 0
    }
}
