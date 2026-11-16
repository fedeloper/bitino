package io.github.fedeloper.bitino.deepfake

import kotlin.math.ceil

/**
 * Where the overlay draws inside a found image, in screen pixels, and what the analysis must keep clear of it, in
 * capture pixels. One place for both so they cannot drift apart.
 *
 * Everything is drawn at least [INSET_CAPTURE] capture pixels inside the image: after the screen is scaled into the
 * capture and halved by [RectFinder], a drawing closer to the edge would bleed into a thin gap next to the image (the
 * 3-pixel separators of an image grid) and glue the grid into one block.
 */
internal class OverlayGeometry(private val density: Float, private val toScreen: Float) {
    companion object {
        const val INSET_CAPTURE = 6
        const val STROKE_DP = 2f
        const val PILL_GAP_DP = 2f
        const val PILL_H_DP = 14f
    }

    /** Screen pixels from the image edge to the outline's outer side. */
    val insetPx: Float = INSET_CAPTURE * toScreen
    val strokePx: Float = STROKE_DP * density
    /** Screen pixels from the image edge to the P(fake) pill (inside the outline). */
    val pillOffsetPx: Float = insetPx + strokePx + PILL_GAP_DP * density
    val pillHeightPx: Float = PILL_H_DP * density
    /** Screen pixels of the strip along the top edge that may hold the pill. */
    val stripPx: Float = pillOffsetPx + pillHeightPx + PILL_GAP_DP * density

    /** The pill is drawn only on images taller than this (screen pixels). */
    fun pillFits(heightPx: Float): Boolean = heightPx > 3 * stripPx

    /** Capture pixels the detector leaves out on every side (outline band + 2). */
    val cropMarginCapture: Int = INSET_CAPTURE + ceil(strokePx / toScreen).toInt() + 2
    /** Capture pixels the detector leaves out along the top of an image that may carry the pill (+ 2). */
    val cropTopCapture: Int = ceil(stripPx / toScreen).toInt() + 2
}
