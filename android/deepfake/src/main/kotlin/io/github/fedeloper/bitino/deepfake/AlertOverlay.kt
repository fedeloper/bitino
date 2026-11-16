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
 * - the image's P(fake) in a small pill in its top-left corner (inside the strip the detector never sees);
 * - a red banner at the top of the screen when the whole view was flagged;
 * - classification mode (an ImageNet classifier instead of a detector): a blue outline and a pill with the top-1 label
 *   and its probability on each image, and a blue banner with the whole view's label when no image is found.
 */
internal class AlertOverlay(context: Context) {

    /** A found image (screen pixels), its state and its last P(fake) (rounded to 0.01; < 0 = not analysed yet). */
    data class Box(val rect: RectF, val state: Int, val pFake: Float, val label: String? = null) {
        companion object { const val PENDING = 0; const val OK = 1; const val SUSPICIOUS = 2; const val FAKE = 3; const val CLASS = 4 }
    }
    /** A banner at the top of the screen: red for an alert, blue for information (classification mode). */
    data class Banner(val text: String, val alert: Boolean)

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

    /**
     * Main thread only. [banner]: the whole-view banner, or null. [geo]: where to draw inside each image.
     * Redraws only on a change.
     */
    fun update(boxes: List<Box>, banner: Banner?, geo: OverlayGeometry) {
        if (boxes == view.boxes && banner == view.banner && geo === view.geo) return
        view.boxes = boxes
        view.banner = banner
        view.geo = geo
        view.invalidate()
    }

    private class OverlayView(context: Context) : View(context) {
        var boxes: List<Box> = emptyList()
        var banner: Banner? = null
        var geo: OverlayGeometry? = null
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
        private val cls = dashed(Color.rgb(40, 120, 255), 0f) to dashed(Color.WHITE, dash)
        private val tint = Paint().apply { color = Color.argb(120, 255, 0, 0) }
        private val fake = dashed(Color.rgb(255, 30, 30), 0f) to dashed(Color.WHITE, dash)
        private val bannerBg = Paint()
        private val pillText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 11f * dp
            typeface = Typeface.DEFAULT_BOLD
        }
        private val pillBg = Paint(Paint.ANTI_ALIAS_FLAG)
        private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textAlign = Paint.Align.CENTER
            typeface = Typeface.DEFAULT_BOLD
            setShadowLayer(4f * dp, 0f, 0f, Color.BLACK)
        }

        /** [s] shortened with "…" (keeping the score at the end) to fit [room] pixels. */
        private fun fit(s: String, room: Float): String {
            if (pillText.measureText(s) <= room) return s
            val sp = s.lastIndexOf(' ')
            val tail = if (sp > 0) s.substring(sp) else ""
            var head = if (sp > 0) s.substring(0, sp) else s
            while (head.isNotEmpty() && pillText.measureText("$head…$tail") > room) head = head.dropLast(1)
            return "$head…$tail"
        }

        override fun onDraw(canvas: Canvas) {
            getLocationOnScreen(loc)
            val geo = geo ?: return
            for (b in boxes) {
                val r = RectF(b.rect)
                r.offset(-loc[0].toFloat(), -loc[1].toFloat())
                r.inset(geo.insetPx + stroke / 2, geo.insetPx + stroke / 2)   // the stroke's outer side at insetPx
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
                val (pa, pb) = when (b.state) { Box.SUSPICIOUS -> sus; Box.OK -> ok; Box.CLASS -> cls; else -> pend }
                canvas.drawRect(r, pa)
                canvas.drawRect(r, pb)
                // P(fake) pill inside the outline, in the top strip the detector never sees, smaller than the image
                val room = b.rect.width() - 2 * geo.pillOffsetPx - 8f * dp
                val text = fit(if (b.pFake < 0) "…" else if (b.label != null) "${b.label} %.2f".format(b.pFake) else "%.2f".format(b.pFake), room)
                val pw = pillText.measureText(text) + 8f * dp
                val top = b.rect.top - loc[1] + geo.pillOffsetPx
                val left = b.rect.left - loc[0] + geo.pillOffsetPx
                if (pw <= b.rect.width() - 2 * geo.pillOffsetPx && geo.pillFits(b.rect.height())) {
                    pillBg.color = when (b.state) {
                        Box.SUSPICIOUS -> Color.rgb(230, 120, 0)
                        Box.OK -> Color.rgb(0, 140, 60)
                        Box.CLASS -> Color.rgb(30, 90, 200)
                        else -> Color.rgb(90, 90, 90)
                    }
                    canvas.drawRoundRect(RectF(left, top, left + pw, top + geo.pillHeightPx), 4f * dp, 4f * dp, pillBg)
                    canvas.drawText(text, left + 4f * dp, top + geo.pillHeightPx * 0.78f, pillText)
                }
            }
            banner?.let { bn ->
                val h = 36f * dp
                bannerBg.color = if (bn.alert) Color.rgb(220, 0, 0) else Color.rgb(30, 90, 200)
                canvas.drawRect(0f, 0f, width.toFloat(), h, bannerBg)
                label.textSize = 16f * dp
                canvas.drawText(bn.text, width / 2f, h * 0.68f, label)
            }
        }
    }
}
