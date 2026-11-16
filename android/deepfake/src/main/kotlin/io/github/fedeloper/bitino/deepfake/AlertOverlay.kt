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
import kotlin.math.min

/**
 * The full-screen, touch-through overlay: an outline of the analysed square (drawn on its border, which the model's
 * centre crop drops), a small score badge in the top-left corner (outside the square), and the red alert.
 */
internal class AlertOverlay(context: Context) {
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

    /** Main thread only. */
    fun setBadge(text: String) {
        if (text != view.badge) {
            view.badge = text
            view.invalidate()
        }
    }

    /** Main thread only. */
    fun setAlert(on: Boolean, pFake: Float = 0f, scale: Int = 0) {
        if (on != view.alert || (on && (pFake != view.alertP || scale != view.alertScale))) {
            view.alert = on
            view.alertP = pFake
            view.alertScale = scale
            view.invalidate()
        }
    }

    private class OverlayView(context: Context) : View(context) {
        var badge = ""
        var alert = false
        var alertP = 0f
        var alertScale = 0
        private val dp = context.resources.displayMetrics.density
        private val loc = IntArray(2)
        private val outline = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2f * dp
            color = Color.argb(200, 255, 255, 255)
        }
        private val badgeBg = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(170, 0, 0, 0) }
        private val badgeText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 12f * dp
            typeface = Typeface.MONOSPACE
        }
        private val tick = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 2f * dp
            color = Color.argb(200, 255, 255, 255)
        }
        private val hitOutline = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f * dp
            color = Color.WHITE
        }
        private val red = Paint().apply { color = Color.argb(150, 255, 0, 0) }
        private val alertText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textAlign = Paint.Align.CENTER
            typeface = Typeface.DEFAULT_BOLD
        }

        override fun onDraw(canvas: Canvas) {
            // the analysed square: the largest centred square of the screen, in this view's coordinates
            getLocationOnScreen(loc)
            val dm = resources.displayMetrics
            val sw: Int
            val sh: Int
            if (Build.VERSION.SDK_INT >= 30) {
                val b = context.getSystemService(WindowManager::class.java).currentWindowMetrics.bounds
                sw = b.width(); sh = b.height()
            } else {
                sw = dm.widthPixels; sh = dm.heightPixels
            }
            val side = min(sw, sh).toFloat()
            val left = (sw - side) / 2f - loc[0]
            val top = (sh - side) / 2f - loc[1]
            val sq = RectF(left, top, left + side, top + side)
            if (alert) {
                // frames are not analysed while the alert is up: it may cover the squares
                canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), red)
                val f = Geometry.FRACTIONS[alertScale]
                val h = side * f / 2f
                canvas.drawRect(sq.centerX() - h, sq.centerY() - h, sq.centerX() + h, sq.centerY() + h, hitOutline)
                alertText.textSize = side / 6f
                canvas.drawText("FAKE", sq.centerX(), sq.centerY(), alertText)
                alertText.textSize = side / 16f
                canvas.drawText("P(fake) %.2f · %d%% square".format(alertP, (f * 100).toInt()), sq.centerX(), sq.centerY() + side / 8f, alertText)
            }
            canvas.drawRect(sq, outline)
            // the smaller squares: ticks outside the largest one (outside every analysed square)
            val len = 10f * dp
            for (i in 1 until Geometry.SCALES) {
                val h = side * Geometry.FRACTIONS[i] / 2f
                for (x in floatArrayOf(sq.centerX() - h, sq.centerX() + h)) {
                    canvas.drawLine(x, sq.top - len, x, sq.top - 1f * dp, tick)
                    canvas.drawLine(x, sq.bottom + 1f * dp, x, sq.bottom + len, tick)
                }
                for (y in floatArrayOf(sq.centerY() - h, sq.centerY() + h)) {
                    canvas.drawLine(sq.left - len, y, sq.left - 1f * dp, y, tick)
                    canvas.drawLine(sq.right + 1f * dp, y, sq.right + len, y, tick)
                }
            }
            if (badge.isNotEmpty()) {
                val lines = badge.split('\n')
                val pad = 6f * dp
                val lh = badgeText.textSize * 1.25f
                val w = lines.maxOf { badgeText.measureText(it) } + 2 * pad
                val h = lines.size * lh + 2 * pad
                val x = 8f * dp - loc[0].coerceAtMost(0)
                val y = 40f * dp - loc[1].coerceAtMost(0)
                canvas.drawRoundRect(RectF(x, y, x + w, y + h), 6f * dp, 6f * dp, badgeBg)
                lines.forEachIndexed { i, s -> canvas.drawText(s, x + pad, y + pad + (i + 0.8f) * lh, badgeText) }
            }
        }
    }
}
