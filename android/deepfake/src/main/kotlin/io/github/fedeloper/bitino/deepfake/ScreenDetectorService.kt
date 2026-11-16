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
        private const val SCREEN_CHANGE = 6        // mean luminance change (0..255) of the centre that counts as a new screen
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
    /** Whole view flagged as fake: the banner stays until the screen changes ([wholeSig] = the centre when flagged). */
    private var wholeFake = false
    private var wholeSig = IntArray(64)
    private var tracker = Tracker(tol = 8)
    private var frameNo = 0L
    private var pWhole = -1f
    private var threshold = 0.7f
    private var showBoxes = true
    private var modelName = ""
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
            wholeFake = false
            tracker = Tracker(tol = maxOf(8, g.vw / 100))
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
    private val repass = Runnable { held?.let { process(it) } }

    /** Worker thread. */
    private fun onFrame(r: ImageReader) {
        val img = try { r.acquireLatestImage() } catch (e: IllegalStateException) { null } ?: return
        val g = geometry
        if (g == null || model == null || img.width != g.vw || img.height != g.vh) {
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
            val rects = finder.find(pl.buffer, pl.rowStride, pl.pixelStride, g.vw, g.vh)
            val tf = SystemClock.elapsedRealtimeNanos()
            // 2. match them to the images followed so far (a few pixels of movement = the same image)
            frameNo++
            val changed = tracker.update(rects)
            val tracks = tracker.tracks
            if (changed) Log.i(TAG, "frame $frameNo: ${rects.size} rectangles " + rects.joinToString(" ") { it.joinToString(",", "[", "]") })
            // 3. the whole view: the largest centred square, the fallback for screens without images found. A flagged
            //    whole view stays flagged until the screen changes. It is analysed only when no image is on screen, so
            //    that nothing the overlay draws (outlines, tints) is ever part of its input.
            val sig = ScreenCrop.signature(pl.buffer, pl.rowStride, pl.pixelStride, g)
            if (wholeFake && (changed || ScreenCrop.signatureDiff(sig, wholeSig) > SCREEN_CHANGE)) {
                wholeFake = false
                wholeAlert.reset()
            }
            if (!wholeFake && tracks.none { it.seen || it.fake }) {
                ScreenCrop.toInput(pl.buffer, pl.rowStride, pl.pixelStride, g, 0, input, row)
                pWhole = pFake(m)
                if (wholeAlert.update(pWhole, threshold)) {
                    wholeFake = true
                    wholeSig = sig
                }
            }
            // 4. the detector on PER_FRAME of the images in view, in rotation: never-analysed images first, then the
            //    longest waiting (largest first among equals); once every image has had its turn the rotation starts
            //    again. Images flagged as fake are not analysed again: their flag stays until they move or disappear.
            val turn = tracks.filter { it.seen && !it.fake }
                .sortedWith(compareBy<Tracker.Track> { it.lastRun }.thenByDescending { (it.rect[2] - it.rect[0]).toLong() * (it.rect[3] - it.rect[1]) })
                .take(PER_FRAME)
            val geo = overlayGeometry(g)
            val m0 = geo.cropMarginCapture
            for (t in turn) {
                val rc = t.rect
                // the detector sees the image without a fixed margin wider than the outline band and, on images tall
                // enough to carry the P(fake) pill, without the strip along the top that may hold it, whether they are
                // drawn or not: what the overlay draws never reaches its input
                val mm = if (minOf(rc[2] - rc[0], rc[3] - rc[1]) > 4 * m0) m0 else 0
                val top = if (mm > 0 && geo.pillFits((rc[3] - rc[1]) * g.toScreen)) geo.cropTopCapture else mm
                Resample.toInput(pl.buffer, pl.rowStride, pl.pixelStride, rc[0] + mm, rc[1] + top, rc[2] - mm, rc[3] - mm, input)
                t.pFake = pFake(m)
                t.lastRun = frameNo
                if (t.alert.update(t.pFake, threshold)) t.fake = true
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
                val pImages = tracks.joinToString(" ") { if (it.fake) "FAKE" else if (it.pFake < 0) "?" else String.format(Locale.US, "%.2f", it.pFake) }
                val text = String.format(Locale.US, "%d image%s found (%d analysed per frame), P(fake) %s; whole view %s · %.0f ms/frame (find %.0f) · %.1f fps",
                    tracks.size, if (tracks.size == 1) "" else "s", PER_FRAME, pImages.ifEmpty { "-" },
                    if (wholeFake) "FAKE" else if (pWhole < 0) "?" else String.format(Locale.US, "%.2f", pWhole), lastMs, lastFindMs, fps)
                Log.i(TAG, text)
                status = text
                getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, buildNotification(text))
            }
            // overlay: flagged images always; outlines of the others (setting) once they held still for 2 frames, so
            // that nothing is left behind on scrolling content
            val shown = tracks.filter { it.seen && (it.fake || (showBoxes && it.still >= 2)) }
            val boxes = shown.map {
                val st = when {
                    it.fake -> AlertOverlay.Box.FAKE
                    it.pFake < 0 -> AlertOverlay.Box.PENDING
                    it.pFake >= threshold -> AlertOverlay.Box.SUSPICIOUS
                    else -> AlertOverlay.Box.OK
                }
                // the score is rounded as shown, so the overlay is redrawn (a new captured frame) only when what it shows changes
                AlertOverlay.Box(screenRect(g, it.rect), st, if (it.pFake < 0) -1f else Math.round(it.pFake * 100) / 100f)
            }
            val banner = if (wholeFake) pWhole else null
            main.post {
                overlay?.update(boxes, banner, geo)
            }
        }
        heldPasses++
        val open = tracker.tracks.count { it.seen && !it.fake }
        val passes = 3 * maxOf(1, (open + PER_FRAME - 1) / PER_FRAME)
        if (heldPasses < passes) work.post(repass)
    }

    /** Runs the detector on [input]; output 0 = [constant 0, logit of "real"]: P(fake) = 1 / (1 + exp(logit)). */
    private fun pFake(m: BitinoModel): Float {
        m.run(input, output)
        return (1.0 / (1.0 + exp(output[1].toDouble()))).toFloat()
    }

    private var geoCache: Pair<Geometry, OverlayGeometry>? = null
    private fun overlayGeometry(g: Geometry): OverlayGeometry =
        geoCache?.takeIf { it.first === g }?.second
            ?: OverlayGeometry(resources.displayMetrics.density, g.toScreen).also { geoCache = g to it }

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
