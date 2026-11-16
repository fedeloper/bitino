package io.github.fedeloper.bitino.deepfake

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.OpenableColumns
import android.provider.Settings
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import java.io.File
import java.util.Locale

/**
 * Settings and start / stop of the screen detector: the model (bundled in a local build, or a converted file picked
 * on the phone), the P(fake) threshold and the threads per inference.
 */
class MainActivity : ComponentActivity() {
    private data class ModelEntry(val label: String, val source: String)

    private val prefs by lazy { getSharedPreferences("settings", MODE_PRIVATE) }
    private val ui = Handler(Looper.getMainLooper())
    private lateinit var modelSpinner: Spinner
    private lateinit var thresholdLabel: TextView
    private lateinit var thresholdBar: SeekBar
    private lateinit var threadsSpinner: Spinner
    private lateinit var startStop: Button
    private lateinit var statusView: TextView
    private var models: List<ModelEntry> = emptyList()

    private val pickFile = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) importModel(uri)
    }
    private val notificationPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        requestCapture()
    }
    private val capture = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
        val data = res.data
        if (res.resultCode != RESULT_OK || data == null) {
            statusView.text = "Screen capture was not allowed."
            return@registerForActivityResult
        }
        val i = Intent(this, ScreenDetectorService::class.java)
            .setAction(ScreenDetectorService.ACTION_START)
            .putExtra(ScreenDetectorService.EXTRA_RESULT_CODE, res.resultCode)
            .putExtra(ScreenDetectorService.EXTRA_RESULT_DATA, data)
            .putExtra(ScreenDetectorService.EXTRA_MODEL, models.getOrNull(modelSpinner.selectedItemPosition)?.source)
            .putExtra(ScreenDetectorService.EXTRA_THRESHOLD, thresholdBar.progress / 100f)
            .putExtra(ScreenDetectorService.EXTRA_THREADS, threadsSpinner.selectedItem.toString().toInt())
        if (Build.VERSION.SDK_INT >= 26) startForegroundService(i) else startService(i)
        moveTaskToBack(true)   // show what is on screen; the overlay keeps the score
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val dp = resources.displayMetrics.density
        val pad = (16 * dp).toInt()
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad * 2, pad, pad)
        }
        fun text(s: String, size: Float = 15f) = TextView(this).apply { this.text = s; textSize = size; setPadding(0, pad / 2, 0, pad / 4) }
        col.addView(text("Deepfake screen detector", 22f))
        col.addView(text(
            "Runs a Faster Than Lies binary detector (BNext backbone, CVPRW 2024) with Bitino on this phone's CPU. " +
                "On every new frame it analyses the centre of the screen at three sizes: the largest centred square " +
                "(outlined) and the squares at 2/3 and 1/3 of it (ticks), each resized to the model input. It flashes the " +
                "screen red while one of them looks generated or manipulated. Nothing leaves the device.\n\n" +
                "When Android asks what to share, choose the entire screen.", 14f))
        col.addView(text("Detector"))
        modelSpinner = Spinner(this)
        col.addView(modelSpinner)
        col.addView(Button(this).apply {
            text = "Add a converted model file…"
            setOnClickListener { pickFile.launch(arrayOf("*/*")) }
        })
        thresholdLabel = text("")
        col.addView(thresholdLabel)
        thresholdBar = SeekBar(this).apply {
            max = 99
            progress = prefs.getInt("threshold", 70)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar, v: Int, fromUser: Boolean) {
                    if (v < 50) { s.progress = 50; return }
                    thresholdLabel.text = String.format(Locale.US, "Alert when P(fake) ≥ %.2f on 3 frames in a row (any of the 3 squares)", v / 100f)
                    prefs.edit().putInt("threshold", v).apply()
                }
                override fun onStartTrackingTouch(s: SeekBar) {}
                override fun onStopTrackingTouch(s: SeekBar) {}
            })
        }
        thresholdLabel.text = String.format(Locale.US, "Alert when P(fake) ≥ %.2f on 3 frames in a row (any of the 3 squares)", thresholdBar.progress / 100f)
        col.addView(thresholdBar)
        col.addView(text("Threads per inference"))
        threadsSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, listOf("1", "2", "4"))
            setSelection(listOf(1, 2, 4).indexOf(prefs.getInt("threads", 1)).coerceAtLeast(0))
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                    prefs.edit().putInt("threads", listOf(1, 2, 4)[pos]).apply()
                }
                override fun onNothingSelected(p: AdapterView<*>?) {}
            }
        }
        col.addView(threadsSpinner)
        startStop = Button(this).apply { setOnClickListener { onStartStop() } }
        col.addView(startStop)
        statusView = text("", 14f)
        col.addView(statusView)
        setContentView(ScrollView(this).apply { addView(col) })
        refreshModels()
    }

    override fun onResume() {
        super.onResume()
        refreshUi.run()
    }

    override fun onPause() {
        super.onPause()
        ui.removeCallbacks(refreshUi)
    }

    private val refreshUi = object : Runnable {
        override fun run() {
            val on = ScreenDetectorService.running
            startStop.text = if (on) "Stop" else "Start"
            if (on || ScreenDetectorService.status.isNotEmpty()) statusView.text = ScreenDetectorService.status
            ui.postDelayed(this, 500)
        }
    }

    private fun refreshModels() {
        val list = mutableListOf<ModelEntry>()
        assets.list("models")?.sorted()?.filter { it.endsWith(".bitino") }?.forEach {
            list += ModelEntry(it.removeSuffix(".bitino") + " (bundled)", "asset:models/$it")
        }
        modelDir().listFiles()?.sortedBy { it.name }?.forEach { list += ModelEntry(it.nameWithoutExtension, "file:${it.absolutePath}") }
        models = list
        val labels = if (list.isEmpty()) listOf("No detector yet: add a converted model file") else list.map { it.label }
        modelSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)
        val saved = prefs.getString("model", null)
        val idx = list.indexOfFirst { it.source == saved }.takeIf { it >= 0 }
            ?: list.indexOfFirst { "cocofake" in it.label && "unfrozen" !in it.label }.takeIf { it >= 0 } ?: 0
        modelSpinner.setSelection(idx)
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                models.getOrNull(pos)?.let { prefs.edit().putString("model", it.source).apply() }
            }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
    }

    private fun modelDir() = File(filesDir, "models").apply { mkdirs() }

    private fun importModel(uri: Uri) {
        val name = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        } ?: "model.bitino"
        val stem = name.substringBeforeLast('.').replace(Regex("[^A-Za-z0-9._-]"), "_")
        val dst = File(modelDir(), "$stem.bitino")
        Thread {
            val msg = runCatching {
                contentResolver.openInputStream(uri)!!.use { inp -> dst.outputStream().use { inp.copyTo(it) } }
                "Added ${dst.nameWithoutExtension}"
            }.getOrElse { dst.delete(); "Could not read the file: ${it.message}" }
            ui.post {
                statusView.text = msg
                prefs.edit().putString("model", "file:${dst.absolutePath}").apply()
                refreshModels()
            }
        }.start()
    }

    private fun onStartStop() {
        if (ScreenDetectorService.running) {
            startService(Intent(this, ScreenDetectorService::class.java).setAction(ScreenDetectorService.ACTION_STOP))
            return
        }
        if (models.isEmpty()) {
            statusView.text = "Add a converted detector first (artifacts/ftl-deepfake-*/model.bitino)."
            return
        }
        if (Build.VERSION.SDK_INT >= 23 && !Settings.canDrawOverlays(this)) {
            statusView.text = "Allow \"Display over other apps\" for this app, then press Start again."
            startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")))
            return
        }
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
            return
        }
        requestCapture()
    }

    private fun requestCapture() {
        capture.launch(getSystemService(MediaProjectionManager::class.java).createScreenCaptureIntent())
    }
}
