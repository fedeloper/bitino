package io.github.fedeloper.bitino.deepfake

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.os.Process
import android.os.SystemClock
import android.util.DisplayMetrics
import android.util.Log
import android.view.Display
import android.view.WindowManager
import io.github.fedeloper.bitino.Bitino
import io.github.fedeloper.bitino.BitinoModel
import java.io.File
import java.util.Locale
import kotlin.math.exp

/**
 * Foreground service: mirrors the screen into a small virtual display (MediaProjection), runs the detector on the
 * centre square of every new frame and drives the overlay. Started by [MainActivity] with the user's screen-capture
 * consent; stopped from its notification, the app, or when the system ends the projection.
 */
class ScreenDetectorService : Service() {
    companion object {
        const val ACTION_START = "io.github.fedeloper.bitino.deepfake.START"
        const val ACTION_STOP = "io.github.fedeloper.bitino.deepfake.STOP"
        const val EXTRA_RESULT_CODE = "resultCode"
        const val EXTRA_RESULT_DATA = "resultData"
        const val EXTRA_MODEL = "model"            // "asset:models/<name>.bitino" or "file:<absolute path>"
        const val EXTRA_THRESHOLD = "threshold"    // P(fake) threshold in [0, 1]
        const val EXTRA_THREADS = "threads"
        const val EXTRA_SHOW_BOXES = "showBoxes"    // outline the found images
        private const val TAG = "BitinoDeepfake"
        private const val CHANNEL = "detector"
        private const val NOTIFICATION_ID = 1
        private const val ALERT_MS = 900L          // red alert shown this long, then the square is analysed again
        private const val SETTLE_MS = 120L         // frames composed before this point may still contain the alert
        private const val PER_FRAME = 4            // images analysed per frame besides the whole view, in rotation
        private const val MAX_IMAGES = 24          // images followed on screen

        /** Shown by the activity. */
        @Volatile var running = false
            private set
        @Volatile var status = ""
            private set
    }

    private val main = Handler(Looper.getMainLooper())
    private lateinit var worker: HandlerThread
    private lateinit var work: Handler
    private var projection: MediaProjection? = null
    private var display: VirtualDisplay? = null
    private var reader: ImageReader? = null
    private var geometry: Geometry? = null
    private var model: BitinoModel? = null
    private var overlay: AlertOverlay? = null
    private var displayListener: DisplayManager.DisplayListener? = null

    // worker-thread state
    private val input = FloatArray(Geometry.CROP * Geometry.CROP * 3)
    private val output = FloatArray(2)
    private var row = ByteArray(0)
    private val finder = RectFinder(maxRects = MAX_IMAGES)
    private val wholeAlert = AlertState()
    /** Images followed from frame to frame (capture pixels), each with its own run of fake frames. */
    private class Track(var rect: IntArray, val alert: AlertState = AlertState(), var pFake: Float = -1f, var seen: Boolean = true,
                        var still: Int = 0, var lastRun: Long = -1)   // pFake < 0: not analysed yet; lastRun: frame number
    private var frameNo = 0L
    private var tracks = ArrayList<Track>()
    private var pWhole = 0f
    private var threshold = 0.7f
    private var showBoxes = true
    /** Rectangles (capture pixels) whose outlines are on screen: the finder ignores those pixels. */
    @Volatile private var drawn: List<IntArray> = emptyList()
    private var modelName = ""
    @Volatile private var resumeAtNs = 0L
    private var frames = 0
    private var windowStart = 0L
    private var lastMs = 0.0
    private var fps = 0.0
    private var lastFindMs = 0.0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (intent?.action != ACTION_START || projection != null) return START_NOT_STICKY
        startInForeground()
        val code = intent.getIntExtra(EXTRA_RESULT_CODE, 0)
        val data: Intent? = if (Build.VERSION.SDK_INT >= 33) intent.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
        else @Suppress("DEPRECATION") intent.getParcelableExtra(EXTRA_RESULT_DATA)
        threshold = intent.getFloatExtra(EXTRA_THRESHOLD, 0.7f)
        val threads = intent.getIntExtra(EXTRA_THREADS, 1)
        showBoxes = intent.getBooleanExtra(EXTRA_SHOW_BOXES, true)
        val source = intent.getStringExtra(EXTRA_MODEL) ?: ""
        try {
            model = loadModel(source, threads)
            modelName = source.substringAfterLast('/').substringBeforeLast('.')
            val mpm = getSystemService(MediaProjectionManager::class.java)
            val p = mpm.getMediaProjection(code, data ?: error("no screen-capture consent"))
                ?: error("screen capture was not granted")
            p.registerCallback(object : MediaProjection.Callback() {
                override fun onStop() { main.post { stopSelf() } }
            }, main)
            projection = p
        } catch (e: Throwable) {
            Log.e(TAG, "start failed", e)
            status = "Could not start: ${e.message}"
            stopSelf()
            return START_NOT_STICKY
        }
        worker = HandlerThread("bitino-detector", Process.THREAD_PRIORITY_DISPLAY).also { it.start() }
        work = Handler(worker.looper)
        overlay = AlertOverlay(overlayContext()).also { it.attach() }
        val dm = getSystemService(DisplayManager::class.java)
        configureCapture()
        displayListener = object : DisplayManager.DisplayListener {
            override fun onDisplayAdded(id: Int) {}
            override fun onDisplayRemoved(id: Int) {}
            override fun onDisplayChanged(id: Int) { if (id == Display.DEFAULT_DISPLAY) configureCapture() }
        }.also { dm.registerDisplayListener(it, main) }
        running = true
        status = "Looking for images on screen with $modelName"
        return START_NOT_STICKY
    }

    private fun loadModel(source: String, threads: Int): BitinoModel {
        val m = when {
            source.startsWith("asset:") -> Bitino.load(this, source.removePrefix("asset:"), threads)
            source.startsWith("file:") -> Bitino.load(File(source.removePrefix("file:")), threads)
            else -> error("no model selected")
        }
        val shape = m.inputShape
        if (!(shape.contentEquals(intArrayOf(1, Geometry.CROP, Geometry.CROP, 3)) && m.outputSize == 2)) {
            m.close()
            error("not a Faster Than Lies detector (input ${shape.contentToString()}, ${m.outputSize} outputs; " +
                "expected [1, 224, 224, 3] and the two-logit head of tools/convert_ftl_deepfake.py)")
        }
        return m
    }

    private fun overlayContext(): Context {
        if (Build.VERSION.SDK_INT < 30) return this
        val d = getSystemService(DisplayManager::class.java).getDisplay(Display.DEFAULT_DISPLAY)
        return createDisplayContext(d).createWindowContext(WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY, null)
    }

    /** (Re)creates the capture surface for the current screen size and rotation. Main thread. */
    private fun configureCapture() {
        val p = projection ?: return
        val d = getSystemService(DisplayManager::class.java).getDisplay(Display.DEFAULT_DISPLAY) ?: return
        val dm = DisplayMetrics()
        @Suppress("DEPRECATION") d.getRealMetrics(dm)
        val g = Geometry(dm.widthPixels, dm.heightPixels, dm.densityDpi)
        if (g.sameAs(geometry)) return
        val r = ImageReader.newInstance(g.vw, g.vh, PixelFormat.RGBA_8888, 3)
        r.setOnImageAvailableListener({ onFrame(it) }, work)
        val old = reader
        work.post {
            work.removeCallbacks(repass)
            held?.close()
            held = null
            geometry = g
            wholeAlert.reset()
            tracks.clear()
        }
        val vd = display
        if (vd == null) {
            display = p.createVirtualDisplay("bitino-deepfake", g.vw, g.vh, g.vdpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, r.surface, null, null)
        } else {
            vd.resize(g.vw, g.vh, g.vdpi)
            vd.surface = r.surface
        }
        reader = r
        old?.let { o -> work.post { o.close() } }
        Log.i(TAG, "capture ${g.vw}x${g.vh} for screen ${g.screenW}x${g.screenH}: centre square ${g.side} px")
    }

    /** The latest analysed frame, kept so that a still screen (no new frames) still gets every image analysed. */
    private var held: android.media.Image? = null
    private var heldPasses = 0
    private val repass = Runnable { held?.let { if (it.timestamp >= resumeAtNs) process(it) } }

    /** Worker thread. */
    private fun onFrame(r: ImageReader) {
        val img = try { r.acquireLatestImage() } catch (e: IllegalStateException) { null } ?: return
        val g = geometry
        if (g == null || model == null || img.width != g.vw || img.height != g.vh ||
            img.timestamp < resumeAtNs) {            // composed while the red alert was (still) on screen
            img.close()
            return
        }
        work.removeCallbacks(repass)
        held?.close()
        held = img
        heldPasses = 0
        process(img)
    }

    /**
     * One pass over [img]: find the images, run the whole view and the next [PER_FRAME] images in the rotation. While
     * no new frame arrives, passes repeat on the same picture until every image has had 3 turns (an alert needs 3
     * fake results in a row); then the service waits for the screen to change. Worker thread.
     */
    private fun process(img: android.media.Image) {
        val g = geometry ?: return
        val m = model ?: return
        run {
            val pl = img.planes[0]
            val need = ScreenCrop.rowBytes(Geometry.KS[0], pl.pixelStride)
            if (row.size < need) row = ByteArray(need)
            val t0 = SystemClock.elapsedRealtimeNanos()
            // 1. the images on screen (classical layout analysis)
            val rects = finder.find(pl.buffer, pl.rowStride, pl.pixelStride, g.vw, g.vh, drawn, outlineBand(g))
            val tf = SystemClock.elapsedRealtimeNanos()
            // 2. the whole view: the largest centred square
            ScreenCrop.toInput(pl.buffer, pl.rowStride, pl.pixelStride, g, 0, input, row)
            pWhole = pFake(m)
            // 3. every image found, matched to the image tracked at about the same place in the previous frame
            frameNo++
            tracks.forEach { it.seen = false }
            val now = ArrayList<Track>()
            for (rc in rects) {
                val t = tracks.filter { !it.seen }.maxByOrNull { iou(it.rect, rc) }?.takeIf { iou(it.rect, rc) >= 0.5f } ?: Track(rc)
                // "still": same place (within 2 half-resolution pixels per edge) as in the previous frame
                t.still = if ((0..3).all { kotlin.math.abs(t.rect[it] - rc[it]) <= 4 }) t.still + 1 else 0
                t.rect = rc
                t.seen = true
                now += t
            }
            tracks = now
            // 4. the detector on PER_FRAME of them, in rotation: never-analysed images first, then the longest waiting
            //    (largest first among equals); once every image has had its turn the rotation starts again
            val turn = tracks.sortedWith(compareBy<Track> { it.lastRun }.thenByDescending { (it.rect[2] - it.rect[0]).toLong() * (it.rect[3] - it.rect[1]) })
                .take(PER_FRAME)
            val flags = ArrayList<AlertOverlay.Flag>()
            for (t in turn) {
                val rc = t.rect
                Resample.toInput(pl.buffer, pl.rowStride, pl.pixelStride, rc[0], rc[1], rc[2], rc[3], input)
                t.pFake = pFake(m)
                t.lastRun = frameNo
                if (t.alert.update(t.pFake, threshold)) flags += AlertOverlay.Flag(screenRect(g, t.rect), t.pFake, whole = false)
            }
            // images analysed earlier and still fake keep blinking with the others
            if (flags.isNotEmpty()) for (t in tracks) {
                if (t !in turn && t.alert.streak >= 2 && t.pFake >= threshold) flags += AlertOverlay.Flag(screenRect(g, t.rect), t.pFake, whole = false)
            }
            if (wholeAlert.update(pWhole, threshold) && flags.isEmpty()) {
                val side = g.side.toFloat()
                val l = (g.screenW - side) / 2f
                val tp = (g.screenH - side) / 2f
                flags += AlertOverlay.Flag(android.graphics.RectF(l, tp, l + side, tp + side), pWhole, whole = true)
            }
            val t1 = SystemClock.elapsedRealtimeNanos()
            lastMs = (t1 - t0) / 1e6
            lastFindMs = (tf - t0) / 1e6
            frames++
            if (windowStart == 0L) windowStart = t0
            if (t1 - windowStart >= 1_000_000_000L) {
                fps = frames * 1e9 / (t1 - windowStart)
                frames = 0
                windowStart = t1
                val pImages = tracks.joinToString(" ") { if (it.pFake < 0) "?" else String.format(Locale.US, "%.2f", it.pFake) }
                val text = String.format(Locale.US, "%d image%s found (%d analysed per frame), P(fake) %s; whole view %.2f · %.0f ms/frame (find %.0f) · %.1f fps",
                    tracks.size, if (tracks.size == 1) "" else "s", PER_FRAME, pImages.ifEmpty { "-" }, pWhole, lastMs, lastFindMs, fps)
                Log.i(TAG, text)
                status = text
                getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification(text))
            }
            if (showBoxes) {
                // outlines only of images that held still for 2 frames: no outline is left behind on scrolling content
                val still = tracks.filter { it.still >= 2 }
                val boxes = still.map {
                    AlertOverlay.Box(screenRect(g, it.rect), when { it.pFake < 0 -> AlertOverlay.Box.PENDING; it.pFake >= threshold -> AlertOverlay.Box.SUSPICIOUS; else -> AlertOverlay.Box.OK })
                }
                val rects = still.map { it.rect.copyOf() }
                main.post {
                    overlay?.showBoxes(boxes)
                    drawn = rects
                }
            }
            if (flags.isNotEmpty()) {
                resumeAtNs = System.nanoTime() + (ALERT_MS + SETTLE_MS) * 1_000_000L
                main.post {
                    overlay?.show(flags)
                    main.postDelayed({ overlay?.show(emptyList()) }, ALERT_MS)
                }
            }
        }
        heldPasses++
        val passes = 3 * maxOf(1, (tracks.size + PER_FRAME - 1) / PER_FRAME)
        if (heldPasses < passes && img.timestamp >= resumeAtNs) work.post(repass)
    }

    /** Runs the detector on [input]; output 0 = [constant 0, logit of "real"]: P(fake) = 1 / (1 + exp(logit)). */
    private fun pFake(m: BitinoModel): Float {
        m.run(input, output)
        return (1.0 / (1.0 + exp(output[1].toDouble()))).toFloat()
    }

    private fun iou(a: IntArray, b: IntArray): Float {
        val ix = maxOf(0, minOf(a[2], b[2]) - maxOf(a[0], b[0])).toLong() * maxOf(0, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
        val u = (a[2] - a[0]).toLong() * (a[3] - a[1]) + (b[2] - b[0]).toLong() * (b[3] - b[1]) - ix
        return if (u > 0) ix.toFloat() / u else 0f
    }

    /** Capture pixels covered by an outline from the image edge inwards: inset 2 dp + 2 dp stroke, with a margin. */
    private fun outlineBand(g: Geometry): Int =
        kotlin.math.ceil(4f * resources.displayMetrics.density / g.toScreen).toInt() + 2

    private fun screenRect(g: Geometry, r: IntArray) =
        android.graphics.RectF(r[0] * g.toScreen, r[1] * g.toScreen, r[2] * g.toScreen, r[3] * g.toScreen)

    private fun startInForeground() {
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Screen analysis", NotificationManager.IMPORTANCE_LOW))
        }
        val n = buildNotification("Looking for images on screen; nothing leaves the device")
        if (Build.VERSION.SDK_INT >= 29) startForeground(NOTIFICATION_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        else startForeground(NOTIFICATION_ID, n)
    }

    private fun buildNotification(text: String): Notification {
        val stop = PendingIntent.getService(this, 0, Intent(this, ScreenDetectorService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE)
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val b = if (Build.VERSION.SDK_INT >= 26) Notification.Builder(this, CHANNEL)
        else @Suppress("DEPRECATION") Notification.Builder(this)
        return b.setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("Deepfake detector: ${modelName.removePrefix("ftl-deepfake-")}")
            .setContentText(text)
            .setStyle(Notification.BigTextStyle().bigText(text))
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .build()
    }

    override fun onDestroy() {
        running = false
        if (!status.startsWith("Could not")) status = "Stopped"
        displayListener?.let { getSystemService(DisplayManager::class.java).unregisterDisplayListener(it) }
        main.removeCallbacksAndMessages(null)
        overlay?.detach()
        overlay = null
        display?.release()
        display = null
        projection?.stop()
        projection = null
        if (::worker.isInitialized) {
            val r = reader
            val m = model
            work.post {
                work.removeCallbacks(repass)
                held?.close()
                held = null
                r?.close()
                m?.close()
            }
            worker.quitSafely()
        } else {
            model?.close()
        }
        reader = null
        model = null
        super.onDestroy()
    }
}
