package io.github.fedeloper.bitino.deepfake

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.RectF
import android.graphics.Typeface
import android.os.Build
import android.view.Gravity
import android.view.View
import android.view.WindowManager

/**
 * The full-screen, touch-through overlay. Whatever it draws is captured too, so everything is drawn inside the image
 * it belongs to (the background around images stays uniform for [RectFinder]) and no line is ever a single colour (a
 * single-colour line inside an image would read as a gap and the image would shrink a little on every frame):
 * - a dashed outline in two contrasting colours (never a uniform line) for each found image: grey = not analysed yet,
 *   green = looks real, orange = P(fake) at or above the threshold;
 * - a red tint, a dashed red and white border and the P(fake) for an image flagged as fake; it stays until the image moves or
 *   disappears (the service does not analyse a flagged image, so the tint never reaches the detector);
 * - a red banner at the top of the screen when the whole view was flagged.
 */
internal class AlertOverlay(context: Context) {
    /** A found image (screen pixels), its state and its last P(fake). */
    data class Box(val rect: RectF, val state: Int, val pFake: Float) {
        companion object { const val PENDING = 0; const val OK = 1; const val SUSPICIOUS = 2; const val FAKE = 3 }
    }

    private val wm = context.getSystemService(WindowManager::class.java)
    private val view = OverlayView(context)
    private var attached = false

    fun attach() {
        if (attached) return
        @Suppress("DEPRECATION")
        val type = if (Build.VERSION.SDK_INT >= 26) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        else WindowManager.LayoutParams.TYPE_PHONE
        val lp = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT, WindowManager.LayoutParams.MATCH_PARENT, type,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT,
        )
        lp.gravity = Gravity.TOP or Gravity.START
        if (Build.VERSION.SDK_INT >= 28) {
            lp.layoutInDisplayCutoutMode = WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }
        wm.addView(view, lp)
        attached = true
    }

    fun detach() {
        if (attached) runCatching { wm.removeViewImmediate(view) }
        attached = false
    }

    /** Main thread only. [wholeFake]: P(fake) of a flagged whole view, or null. Redraws only on a change. */
    fun update(boxes: List<Box>, wholeFake: Float?) {
        if (boxes == view.boxes && wholeFake == view.wholeFake) return
        view.boxes = boxes
        view.wholeFake = wholeFake
        view.invalidate()
    }

    private class OverlayView(context: Context) : View(context) {
        var boxes: List<Box> = emptyList()
        var wholeFake: Float? = null
        private val dp = context.resources.displayMetrics.density
        private val dash = 6f * dp
        private val stroke = 2f * dp
        private val loc = IntArray(2)
        private fun dashed(c: Int, phase: Float) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = stroke
            color = c
            pathEffect = android.graphics.DashPathEffect(floatArrayOf(dash, dash), phase)
        }
        private val pend = dashed(Color.rgb(150, 150, 150), 0f) to dashed(Color.WHITE, dash)
        private val ok = dashed(Color.rgb(0, 230, 90), 0f) to dashed(Color.WHITE, dash)
        private val sus = dashed(Color.rgb(255, 150, 0), 0f) to dashed(Color.BLACK, dash)
        private val tint = Paint().apply { color = Color.argb(120, 255, 0, 0) }
        private val fake = dashed(Color.rgb(255, 30, 30), 0f) to dashed(Color.WHITE, dash)
        private val banner = Paint().apply { color = Color.rgb(220, 0, 0) }
        private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textAlign = Paint.Align.CENTER
            typeface = Typeface.DEFAULT_BOLD
            setShadowLayer(4f * dp, 0f, 0f, Color.BLACK)
        }

        override fun onDraw(canvas: Canvas) {
            getLocationOnScreen(loc)
            for (b in boxes) {
                val r = RectF(b.rect)
                r.offset(-loc[0].toFloat(), -loc[1].toFloat())
                r.inset(stroke, stroke)   // inside the image's edge (the capture rounding needs the margin)
                if (b.state == Box.FAKE) {
                    canvas.drawRect(r, tint)
                    canvas.drawRect(r, fake.first)
                    canvas.drawRect(r, fake.second)
                    val size = (minOf(r.width(), r.height()) / 6f).coerceIn(12f * dp, 44f * dp)
                    label.textSize = size
                    canvas.drawText("FAKE", r.centerX(), r.centerY(), label)
                    label.textSize = size * 0.55f
                    canvas.drawText("P(fake) %.2f".format(b.pFake), r.centerX(), r.centerY() + size, label)
                    continue
                }
                val (pa, pb) = when (b.state) { Box.SUSPICIOUS -> sus; Box.OK -> ok; else -> pend }
                canvas.drawRect(r, pa)
                canvas.drawRect(r, pb)
            }
            wholeFake?.let { p ->
                val h = 36f * dp
                canvas.drawRect(0f, 0f, width.toFloat(), h, banner)
                label.textSize = 16f * dp
                canvas.drawText("FAKE in the whole view · P(fake) %.2f".format(p), width / 2f, h * 0.68f, label)
            }
        }
    }
}
