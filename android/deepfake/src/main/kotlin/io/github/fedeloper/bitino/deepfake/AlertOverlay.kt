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
 * The full-screen, touch-through overlay. Whatever it draws is captured too, so outside alerts it draws only what
 * leaves the layout analysis ([RectFinder]) unchanged: a dashed outline of each found image, in two contrasting
 * colours (never a uniform line, so never trimmed as a border) and inside the image's own edge (so the background
 * around it stays uniform). The detector's centre crop drops those edge pixels. During an alert (when frames are not
 * analysed) it tints each flagged image red with its P(fake), and frames the whole view in red when the centre square
 * fired without a single image doing so.
 */
internal class AlertOverlay(context: Context) {
    /** A flagged region in screen pixels; [whole] = the centre-square ("whole view") input. */
    class Flag(val rect: RectF, val pFake: Float, val whole: Boolean)
    /** A found image (screen pixels) and its state: not analysed yet, below the threshold, at or above it. */
    data class Box(val rect: RectF, val state: Int) {
        companion object { const val PENDING = 0; const val OK = 1; const val SUSPICIOUS = 2 }
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

    /** Main thread only. An empty list clears the alert. */
    fun show(flags: List<Flag>) {
        view.flags = flags
        view.invalidate()
    }

    /** Main thread only: the outlines of the found images (redrawn only when they change). */
    fun showBoxes(boxes: List<Box>) {
        if (boxes == view.boxes) return
        view.boxes = boxes
        view.invalidate()
    }

    private class OverlayView(context: Context) : View(context) {
        var flags: List<Flag> = emptyList()
        var boxes: List<Box> = emptyList()
        private val dp = context.resources.displayMetrics.density
        private val dash = 6f * dp
        private val stroke = 2f * dp
        private fun dashed(c: Int, phase: Float) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = stroke
            color = c
            pathEffect = android.graphics.DashPathEffect(floatArrayOf(dash, dash), phase)
        }
        private val okA = dashed(Color.rgb(0, 230, 90), 0f)
        private val okB = dashed(Color.WHITE, dash)
        private val susA = dashed(Color.rgb(255, 150, 0), 0f)
        private val susB = dashed(Color.BLACK, dash)
        private val pendA = dashed(Color.rgb(150, 150, 150), 0f)
        private val pendB = dashed(Color.WHITE, dash)
        private val loc = IntArray(2)
        private val tint = Paint().apply { color = Color.argb(140, 255, 0, 0) }
        private val border = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 6f * dp
            color = Color.rgb(255, 30, 30)
        }
        private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textAlign = Paint.Align.CENTER
            typeface = Typeface.DEFAULT_BOLD
            setShadowLayer(4f * dp, 0f, 0f, Color.BLACK)
        }

        override fun onDraw(canvas: Canvas) {
            getLocationOnScreen(loc)
            if (flags.isEmpty()) {
                for (b in boxes) {
                    val r = RectF(b.rect)
                    r.offset(-loc[0].toFloat(), -loc[1].toFloat())
                    r.inset(stroke, stroke)   // inside the image's edge (the capture rounding needs the margin)
                    val (pa, pb) = when (b.state) { Box.SUSPICIOUS -> susA to susB; Box.OK -> okA to okB; else -> pendA to pendB }
                    canvas.drawRect(r, pa)
                    canvas.drawRect(r, pb)
                }
                return
            }
            for (f in flags) {
                val r = RectF(f.rect)
                r.offset(-loc[0].toFloat(), -loc[1].toFloat())
                if (!f.whole) canvas.drawRect(r, tint)
                canvas.drawRect(r, border)
                val size = (minOf(r.width(), r.height()) / 6f).coerceIn(14f * dp, 48f * dp)
                label.textSize = size
                val text = if (f.whole) "FAKE (whole view)" else "FAKE"
                canvas.drawText(text, r.centerX(), r.centerY(), label)
                label.textSize = size * 0.55f
                canvas.drawText("P(fake) %.2f".format(f.pFake), r.centerX(), r.centerY() + size, label)
            }
        }
    }
}
