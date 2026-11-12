package io.github.fedeloper.bitino.demo

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.graphics.Bitmap
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.Process
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.PixelCopy
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.Button
import android.widget.ArrayAdapter
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import io.github.fedeloper.bitino.Bitino
import io.github.fedeloper.bitino.BitinoModel
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.min

/**
 * Live ImageNet classification of the camera feed with Bitino: top-5, preprocessing / inference latency and fps,
 * a thread selector (1 / 2 / 4) and a model selector (bundled model, release downloads, a file).
 *
 * Scripted runs: `am start -n io.github.fedeloper.bitino.demo/.MainActivity --ei threads 4 --ei model 0`; the
 * measured numbers are also logged once per second (logcat tag BitinoDemo).
 */
class MainActivity : ComponentActivity() {
    private sealed class Source {
        data class Asset(val path: String, val fallbackDownload: Download?) : Source()
        data class Download(val fileName: String, val sha256: String) : Source()
        data class LocalFile(val file: File) : Source()
        object PickFile : Source()
    }

    private data class Choice(val title: String, val short: String, val source: Source)

    private val reactnet18 = Source.Download(
        "ab-bnn-imagenet-reactnet18.bitino", "a495cf4c138174523b6e9886b0290072b8c48c11930b89c853626898618b755c")
    private val choices = listOf(
        Choice("ReActNet-18 (A&B BNN, MIT) - bundled", "ReActNet-18",
            Source.Asset("models/ab-bnn-imagenet-reactnet18.bitino", reactnet18)),
        Choice("ReActNet-A (A&B BNN, MIT) - download 7.9 MB", "ReActNet-A",
            Source.Download("ab-bnn-imagenet-reactneta.bitino", "d633678fded225f3d93a0f033c8a516aa63286d0a3278ffdd006b34429d7eec8")),
        Choice("ResNet-18 XNOR (OpenVINO OMZ, Apache-2.0) - download 4.2 MB", "ResNet-18 XNOR",
            Source.Download("omz-resnet18-xnor-binary-onnx-0001.bitino", "6f05e3040deb208e35d733a45c887c760d7fc933e8afbfbe1c7f4606a6f521e0")),
        Choice("Load a model file...", "file", Source.PickFile),
    )

    private lateinit var previewView: PreviewView
    private lateinit var titleView: TextView
    private lateinit var statsView: TextView
    private lateinit var top5View: TextView
    private lateinit var statusView: TextView
    private lateinit var cropFrame: View
    private lateinit var modelSpinner: Spinner

    // the analysis thread runs preprocessing and (with threads = 1) the inference: display priority keeps the scheduler
    // from parking it on the little cores
    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor { r ->
        Thread({
            runCatching { Process.setThreadPriority(Process.THREAD_PRIORITY_DISPLAY) }
            r.run()
        }, "bitino-analysis")
    }
    private val loaderExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val modelLock = Any()
    private var model: BitinoModel? = null          // guarded by modelLock
    private var modelTitle = ""                      // guarded by modelLock
    @Volatile private var threads = 1
    @Volatile private var currentChoice = 0
    private var frontCamera = false
    @Volatile private var frameAspect = 3.0 / 4.0   // width / height of the upright analysis frame
    private var lastSource: Source? = null
    private lateinit var labels: List<String>

    // smoothed measurements (analysis thread only)
    private var emaPre = 0.0
    private var emaInf = 0.0
    private var emaConv = 0.0
    private val frameTimes = ArrayDeque<Long>()
    private var lastLog = 0L

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else statusView.text = "Camera permission denied"
    }

    private val pickFile = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) loadPicked(uri) else lastSource?.let { restoreSpinner() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview)
        titleView = findViewById(R.id.title)
        statsView = findViewById(R.id.stats)
        top5View = findViewById(R.id.top5)
        statusView = findViewById(R.id.status)
        cropFrame = findViewById(R.id.cropFrame)
        modelSpinner = findViewById(R.id.model)
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        val captureSeconds = intent.getIntExtra("capture_seconds", 0)
        if (captureSeconds > 0 && Build.VERSION.SDK_INT >= 26) {
            // the preview must be part of the window surface for PixelCopy (TextureView instead of SurfaceView)
            previewView.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            previewView.postDelayed({ startCapture(captureSeconds, intent.getIntExtra("capture_width", 480)) }, 1000)
        }
        previewView.addOnLayoutChangeListener { v, _, _, _, _, _, _, _, _ -> placeCropFrame(v.width, v.height) }

        labels = assets.open("imagenet_classes.txt").bufferedReader().readLines().filter { it.isNotBlank() }
        threads = intent.getIntExtra("threads", 1).takeIf { it in listOf(1, 2, 4) } ?: 1
        currentChoice = intent.getIntExtra("model", 0).coerceIn(0, choices.size - 2)
        frontCamera = intent.getStringExtra("camera") == "front"
        val cameraButton = findViewById<Button>(R.id.camera)
        cameraButton.text = if (frontCamera) "Front camera" else "Back camera"
        cameraButton.setOnClickListener {
            frontCamera = !frontCamera
            cameraButton.text = if (frontCamera) "Front camera" else "Back camera"
            startCamera()
        }

        val group = findViewById<RadioGroup>(R.id.threads)
        group.check(when (threads) { 2 -> R.id.threads2; 4 -> R.id.threads4; else -> R.id.threads1 })
        group.setOnCheckedChangeListener { _, id ->
            val t = when (id) { R.id.threads2 -> 2; R.id.threads4 -> 4; else -> 1 }
            if (t != threads) {
                threads = t
                lastSource?.let { load(it, modelTitle.ifEmpty { "model" }) }
            }
        }

        modelSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, choices.map { it.title }).also {
            it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        modelSpinner.setSelection(currentChoice, false)
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val c = choices[position]
                if (c.source is Source.PickFile) {
                    pickFile.launch(arrayOf("*/*"))
                } else if (position != currentChoice || lastSource == null) {
                    currentChoice = position
                    load(c.source, c.short)
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        load(choices[currentChoice].source, choices[currentChoice].short)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        loaderExecutor.execute { synchronized(modelLock) { model?.close(); model = null } }
        loaderExecutor.shutdown()
    }

    /**
     * Demo recording without `adb shell screenrecord` (blocked on some phones): `--ei capture_seconds N` writes the
     * app window at ~10 fps as JPEG files <external files>/capture/<ms since start>.jpg plus a `done` marker;
     * android/tools/record_demo.sh turns them into a GIF.
     */
    @RequiresApi(26)
    private fun startCapture(seconds: Int, width: Int) {
        val dir = File(getExternalFilesDir(null), "capture").apply { deleteRecursively(); mkdirs() }
        val thread = HandlerThread("capture").apply { start() }
        val handler = Handler(thread.looper)
        val t0 = SystemClock.uptimeMillis()
        var next = t0
        var n = 0
        val tick = object : Runnable {
            override fun run() {
                val now = SystemClock.uptimeMillis()
                if (now - t0 >= seconds * 1000L) {
                    File(dir, "done").writeText("$n\n")
                    Log.i(TAG, "capture done: $n frames in ${dir.absolutePath}")
                    thread.quitSafely()
                    return
                }
                val v = window.decorView
                if (v.width > 0 && v.height > 0) {
                    val full = Bitmap.createBitmap(v.width, v.height, Bitmap.Config.ARGB_8888)
                    val ms = now - t0
                    PixelCopy.request(window, full, { result ->
                        if (result == PixelCopy.SUCCESS) {
                            val small = Bitmap.createScaledBitmap(full, width, width * full.height / full.width, true)
                            File(dir, String.format(Locale.US, "%06d.jpg", ms)).outputStream().use {
                                small.compress(Bitmap.CompressFormat.JPEG, 92, it)
                            }
                            small.recycle()
                            n++
                        }
                        full.recycle()
                    }, handler)
                }
                next += 100
                handler.postAtTime(this, maxOf(next, SystemClock.uptimeMillis() + 1))
            }
        }
        handler.post(tick)
    }

    /** The model sees the center crop of 224/256 of the frame's short side (ImageNet eval framing). */
    private fun placeCropFrame(w: Int, h: Int) {
        if (w == 0 || h == 0) return
        val a = frameAspect   // FIT_CENTER: the whole frame is shown
        val shownW = min(w.toDouble(), h * a)
        val shownH = shownW / a
        val side = (min(shownW, shownH) * 224.0 / 256.0).toInt()
        val lp = cropFrame.layoutParams
        if (lp.width != side) {
            lp.width = side
            lp.height = side
            cropFrame.layoutParams = lp
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val preview = Preview.Builder()
                .setResolutionSelector(ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build())
                .build()
            preview.surfaceProvider = previewView.surfaceProvider
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor, ::analyze)
            provider.unbindAll()
            val camera = if (frontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            provider.bindToLifecycle(this, camera, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        try {
            val rotation = image.imageInfo.rotationDegrees
            val aspect = if (rotation % 180 == 0) image.width.toDouble() / image.height else image.height.toDouble() / image.width
            if (aspect != frameAspect) {
                frameAspect = aspect
                runOnUiThread { placeCropFrame(previewView.width, previewView.height) }
            }
            val t0 = SystemClock.elapsedRealtimeNanos()
            val bitmap = image.toBitmap()
            val conv = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            val scores: FloatArray
            val pre: Double
            val inf: Double
            val nThreads: Int
            val title: String
            synchronized(modelLock) {
                val m = model ?: return
                scores = m.run(bitmap, rotation)
                pre = m.lastPreprocessMs
                inf = m.lastInferenceMs
                nThreads = m.threads
                title = modelTitle
            }
            val now = SystemClock.elapsedRealtime()
            frameTimes.addLast(now)
            while (frameTimes.isNotEmpty() && now - frameTimes.first() > 2000) frameTimes.removeFirst()
            val fps = if (frameTimes.size > 1) (frameTimes.size - 1) * 1000.0 / (frameTimes.last() - frameTimes.first()) else 0.0
            val a = if (emaInf == 0.0) 1.0 else 0.1
            emaPre += a * (pre - emaPre); emaInf += a * (inf - emaInf); emaConv += a * (conv - emaConv)

            val probs = if (scores.all { it >= 0f } && abs(scores.sum() - 1f) < 1e-3f) scores else Bitino.softmax(scores)
            val top = Bitino.topK(probs, 5).joinToString("\n") { (i, p) ->
                String.format(Locale.US, "%5.1f%%  %s", p * 100, label(i, probs.size))
            }
            val stats = String.format(Locale.US, "inference %5.1f ms   pre %4.1f ms\n%5.1f fps   %d thread%s",
                emaInf, emaPre, fps, nThreads, if (nThreads == 1) "" else "s")
            if (now - lastLog > 1000) {
                lastLog = now
                Log.i(TAG, String.format(Locale.US, "model=%s threads=%d inference_ms=%.2f pre_ms=%.2f convert_ms=%.2f fps=%.1f frame=%dx%d rot=%d top1=%s",
                    title, nThreads, emaInf, emaPre, emaConv, fps, image.width, image.height, rotation,
                    Bitino.topK(probs, 1).firstOrNull()?.let { label(it.first, probs.size) }))
            }
            runOnUiThread {
                titleView.text = "Bitino  ·  $title  ·  ${image.width}x${image.height} → 224x224"
                statsView.text = stats
                top5View.text = top
            }
        } catch (e: Exception) {
            Log.e(TAG, "analysis failed", e)
            runOnUiThread { statusView.text = "Error: ${e.message}" }
        } finally {
            image.close()
        }
    }

    private fun label(i: Int, n: Int): String = when {
        n == labels.size -> labels[i]
        n == labels.size + 1 -> if (i == 0) "background" else labels[i - 1]
        else -> "class $i"
    }

    private fun load(source: Source, title: String) {
        lastSource = source
        val t = threads
        statusView.text = "Loading $title ($t thread${if (t == 1) "" else "s"})..."
        loaderExecutor.execute {
            try {
                val t0 = SystemClock.elapsedRealtime()
                val m = open(source, title, t)
                val p = m.preprocessing
                if (p == null) {
                    m.close()
                    throw IllegalStateException("model preprocessing not supported by the demo: ${m.preprocessingDefinition}")
                }
                val ms = SystemClock.elapsedRealtime() - t0
                synchronized(modelLock) {
                    model?.close()
                    model = m
                    modelTitle = title
                    emaInf = 0.0
                }
                Log.i(TAG, "loaded $title threads=$t in $ms ms; input ${m.inputShape.contentToString()}, ${m.outputSize} outputs; " +
                    "preprocessing $p; cpu ${Bitino.cpuFeatures}")
                runOnUiThread {
                    statusView.text = "$title: ${m.inputShape.joinToString("x")} → ${m.outputSize} classes, loaded in $ms ms. " +
                        "Bitino ${Bitino.version}"
                }
            } catch (e: Exception) {
                Log.e(TAG, "load failed", e)
                runOnUiThread { statusView.text = "Cannot load $title: ${e.message}" }
            }
        }
    }

    private fun open(source: Source, title: String, threads: Int): BitinoModel = when (source) {
        is Source.Asset -> {
            val bundled = runCatching { assets.open(source.path).close() }.isSuccess
            when {
                bundled -> Bitino.load(this, source.path, threads)
                source.fallbackDownload != null -> open(source.fallbackDownload, title, threads)
                else -> throw IllegalStateException("${source.path} is not bundled in this APK")
            }
        }
        is Source.Download -> Bitino.load(download(source, title), threads)
        is Source.LocalFile -> Bitino.load(source.file, threads)
        Source.PickFile -> throw IllegalStateException("no file picked")
    }

    /** Downloads a model from the release assets once (SHA-256-checked) into the app's files. */
    private fun download(d: Source.Download, title: String): File {
        val dir = File(filesDir, "models").apply { mkdirs() }
        val out = File(dir, d.fileName)
        if (out.isFile && sha256(out) == d.sha256) return out
        // side-loaded copy (adb push <file> /sdcard/Android/data/<package>/files/models/), e.g. while the release
        // assets are not public
        getExternalFilesDir("models")?.let { File(it, d.fileName) }?.takeIf { it.isFile && sha256(it) == d.sha256 }?.let {
            return it
        }
        runOnUiThread { statusView.text = "Downloading $title..." }
        val part = File(dir, d.fileName + ".part")
        val conn = URL("$RELEASE_URL/${d.fileName}").openConnection() as HttpURLConnection
        conn.instanceFollowRedirects = true
        conn.connectTimeout = 15000
        conn.readTimeout = 30000
        try {
            if (conn.responseCode != 200) {
                throw IllegalStateException("download failed (HTTP ${conn.responseCode}; the release may not be public " +
                    "yet: use 'Load a model file...')")
            }
            conn.inputStream.use { inp -> part.outputStream().use { inp.copyTo(it) } }
        } finally {
            conn.disconnect()
        }
        check(sha256(part) == d.sha256) { part.delete(); "SHA-256 mismatch for ${d.fileName}" }
        part.renameTo(out)
        return out
    }

    private fun loadPicked(uri: Uri) {
        loaderExecutor.execute {
            try {
                val dir = File(filesDir, "models").apply { mkdirs() }
                val file = File(dir, "picked.bitino")
                contentResolver.openInputStream(uri)!!.use { inp -> file.outputStream().use { inp.copyTo(it) } }
                val name = uri.lastPathSegment?.substringAfterLast('/')?.substringAfterLast(':') ?: "model file"
                runOnUiThread {
                    currentChoice = choices.size - 1
                    load(Source.LocalFile(file), name)
                }
            } catch (e: Exception) {
                runOnUiThread { statusView.text = "Cannot read the file: ${e.message}" }
            }
        }
    }

    private fun restoreSpinner() {
        if (currentChoice < choices.size - 1) modelSpinner.setSelection(currentChoice, false)
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { s ->
            val buf = ByteArray(1 shl 16)
            while (true) {
                val n = s.read(buf)
                if (n < 0) break
                md.update(buf, 0, n)
            }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    private companion object {
        const val TAG = "BitinoDemo"
        const val RELEASE_URL = "https://github.com/fedeloper/bitino/releases/download/v0.1.0"
    }
}
