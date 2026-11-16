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
        private const val TAG = "BitinoDeepfake"
        private const val CHANNEL = "detector"
        private const val NOTIFICATION_ID = 1
        private const val ALERT_MS = 900L          // red alert shown this long, then the square is analysed again
        private const val SETTLE_MS = 120L         // frames composed before this point may still contain the alert

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
    private val alerts = Array(Geometry.SCALES) { AlertState() }
    private val pScale = FloatArray(Geometry.SCALES)
    private var threshold = 0.7f
    private var modelName = ""
    @Volatile private var resumeAtNs = 0L
    private var frames = 0
    private var windowStart = 0L
    private var lastMs = 0.0
    private var fps = 0.0
    private var lastBadgeNs = 0L   // badge redraws are frames too: at most 4 per second

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
        status = "Analysing the centre of the screen with $modelName"
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
            geometry = g
            alerts.forEach { it.reset() }
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
        Log.i(TAG, "capture ${g.vw}x${g.vh} for screen ${g.screenW}x${g.screenH}: squares ${g.side} px x ${Geometry.FRACTIONS.joinToString("/")}")
    }

    /** Worker thread. */
    private fun onFrame(r: ImageReader) {
        val img = try { r.acquireLatestImage() } catch (e: IllegalStateException) { null } ?: return
        try {
            val g = geometry ?: return
            val m = model ?: return
            if (img.width != g.vw || img.height != g.vh) return
            if (img.timestamp < resumeAtNs) return     // composed while the red alert was (still) on screen
            val pl = img.planes[0]
            val need = ScreenCrop.rowBytes(Geometry.KS[0], pl.pixelStride)
            if (row.size < need) row = ByteArray(need)
            val t0 = SystemClock.elapsedRealtimeNanos()
            var hit = -1
            for (i in 0 until Geometry.SCALES) {
                ScreenCrop.toInput(pl.buffer, pl.rowStride, pl.pixelStride, g, i, input, row)
                m.run(input, output)
                // output 0 = [constant 0, logit of "real"]: P(fake) = 1 / (1 + exp(logit))
                pScale[i] = (1.0 / (1.0 + exp(output[1].toDouble()))).toFloat()
            }
            // every square's streak advances on every frame; the alert names the most confident square that fired
            for (i in 0 until Geometry.SCALES) {
                if (alerts[i].update(pScale[i], threshold) && (hit < 0 || pScale[i] > pScale[hit])) hit = i
            }
            val t1 = SystemClock.elapsedRealtimeNanos()
            lastMs = (t1 - t0) / 1e6
            frames++
            if (windowStart == 0L) windowStart = t0
            if (t1 - windowStart >= 1_000_000_000L) {
                fps = frames * 1e9 / (t1 - windowStart)
                frames = 0
                windowStart = t1
                Log.i(TAG, String.format(Locale.US, "P(fake) %.3f / %.3f / %.3f (100/67/33 %%)  %.1f ms/frame (3 crops + 3 detector runs)  %.1f frames/s",
                    pScale[0], pScale[1], pScale[2], lastMs, fps))
            }
            val badge = String.format(Locale.US, "%s  thr %.2f\nP(fake) %.2f %.2f %.2f\n(squares 100/67/33%%)\n%.0f ms  %.1f fps",
                modelName.removePrefix("ftl-deepfake-"), threshold, pScale[0], pScale[1], pScale[2], lastMs, fps)
            if (hit >= 0) {
                val p = pScale[hit]
                resumeAtNs = System.nanoTime() + (ALERT_MS + SETTLE_MS) * 1_000_000L
                main.post {
                    overlay?.setBadge(badge)
                    overlay?.setAlert(true, p, hit)
                    main.postDelayed({ overlay?.setAlert(false) }, ALERT_MS)
                }
            } else if (t1 - lastBadgeNs >= 250_000_000L) {
                lastBadgeNs = t1
                main.post { overlay?.setBadge(badge) }
            }
        } finally {
            img.close()
        }
    }

    private fun startInForeground() {
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Screen analysis", NotificationManager.IMPORTANCE_LOW))
        }
        val stop = PendingIntent.getService(this, 0, Intent(this, ScreenDetectorService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE)
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val b = if (Build.VERSION.SDK_INT >= 26) Notification.Builder(this, CHANNEL)
        else @Suppress("DEPRECATION") Notification.Builder(this)
        val n = b.setSmallIcon(android.R.drawable.ic_menu_view)
            .setContentTitle("Deepfake detector running")
            .setContentText("Analysing the centre of the screen on this phone; nothing leaves the device")
            .setContentIntent(open)
            .setOngoing(true)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .build()
        if (Build.VERSION.SDK_INT >= 29) startForeground(NOTIFICATION_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        else startForeground(NOTIFICATION_ID, n)
    }

    override fun onDestroy() {
        running = false
        if (status.startsWith("Analysing")) status = "Stopped"
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
