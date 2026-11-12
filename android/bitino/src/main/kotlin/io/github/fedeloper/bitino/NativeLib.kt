package io.github.fedeloper.bitino

import java.io.File

/** JNI entry points (src/main/cpp/bitino_jni.cc). Handles are native pointers; 0 is never a valid handle. */
internal object NativeLib {
    init {
        checkCpu()
        System.loadLibrary("bitino_jni")
    }

    /**
     * The arm64-v8a build targets ARMv8.2-A with the dot-product extension (Cortex-A55/A75 and newer, 2018+); on an
     * older 64-bit core it would stop with SIGILL. Fail with a readable error instead.
     */
    private fun checkCpu() {
        if (System.getProperty("os.arch") != "aarch64") return
        val features = runCatching {
            File("/proc/cpuinfo").readLines().firstOrNull { it.startsWith("Features") }
        }.getOrNull() ?: return   // unreadable: let the library decide
        if (!features.split(Regex("\\s+")).contains("asimddp")) {
            throw UnsupportedOperationException(
                "Bitino's arm64-v8a build needs the ARMv8.2 dot-product extension (asimddp), which this CPU lacks " +
                    "(cpuinfo: $features)",
            )
        }
    }

    @JvmStatic external fun version(): String
    @JvmStatic external fun cpuFeatures(): String
    @JvmStatic external fun loadFile(path: String, verifyHash: Boolean): Long
    @JvmStatic external fun loadBytes(data: ByteArray, verifyHash: Boolean): Long
    @JvmStatic external fun manifest(model: Long): String
    @JvmStatic external fun freeModel(model: Long)
    @JvmStatic external fun createSession(model: Long, threads: Int): Long
    @JvmStatic external fun freeSession(session: Long)
    @JvmStatic external fun inputShape(session: Long): LongArray
    @JvmStatic external fun numOutputs(session: Long): Int
    @JvmStatic external fun outputSize(session: Long, index: Int): Int
    @JvmStatic external fun run(session: Long, input: FloatArray, output: FloatArray?)
    @JvmStatic external fun getOutput(session: Long, index: Int, output: FloatArray)
}
