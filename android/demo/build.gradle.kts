import java.net.URI
import java.security.MessageDigest
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "io.github.fedeloper.bitino.demo"
    compileSdk = 36
    ndkVersion = rootProject.extra["bitinoNdkVersion"] as String   // strips the native libraries
    (rootProject.extra["bitinoNdkPath"] as String?)?.let { ndkPath = it }

    defaultConfig {
        applicationId = "io.github.fedeloper.bitino.demo"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Local installs only: signed with the debug key. Use your own signingConfig to distribute.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    androidResources { noCompress += "bitino" }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    implementation(project(":bitino"))
    val camerax = "1.5.3"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")
    implementation("androidx.activity:activity-ktx:1.12.4")
}

// ---- Bundled model -------------------------------------------------------------------------------------------
// One small model with an explicit permissive weights licence is bundled in the APK (assets/models/): A&B BNN
// ReActNet-18 (ImageNet; MIT, see its .LICENSE.txt, which is bundled next to it). Taken from a local directory
// holding the release files (-Pbitino.demoModelDir=<dir> or bitino.demoModelDir in gradle.properties), else
// downloaded from the release assets. Both are SHA-256-checked. If neither is available the APK is built without it
// (the app can then download a model or load one from a file).
abstract class FetchDemoModel : DefaultTask() {
    @get:Input abstract val files: MapProperty<String, String>     // file name -> SHA-256
    @get:Input abstract val baseUrl: Property<String>
    @get:Input @get:Optional abstract val localDir: Property<String>
    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @TaskAction
    fun fetch() {
        val dst = outputDir.get().asFile.resolve("models")
        dst.mkdirs()
        val fetched = mutableListOf<File>()
        for ((name, sha) in files.get()) {
            val out = dst.resolve(name)
            if (out.isFile && sha256(out.readBytes()) == sha) { fetched += out; continue }
            val bytes = localDir.orNull?.let { File(it, name).takeIf { f -> f.isFile }?.readBytes() }
                ?: runCatching { URI("${baseUrl.get()}/$name").toURL().openStream().use { it.readBytes() } }
                    .onFailure { logger.warn("bitino demo: cannot download $name (${it.message})") }.getOrNull()
            if (bytes == null) break
            check(sha256(bytes) == sha) { "bitino demo: $name SHA-256 mismatch (expected $sha)" }
            out.writeBytes(bytes)
            fetched += out
        }
        if (fetched.size != files.get().size) {
            dst.deleteRecursively()
            logger.warn("bitino demo: no bundled model (set -Pbitino.demoModelDir=<dir with the release model files>)")
        }
    }

    private fun sha256(b: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }
}

val fetchDemoModel = tasks.register<FetchDemoModel>("fetchDemoModel") {
    files.put("ab-bnn-imagenet-reactnet18.bitino", "a495cf4c138174523b6e9886b0290072b8c48c11930b89c853626898618b755c")
    files.put("ab-bnn-imagenet-reactnet18.LICENSE.txt", "ea6e975eae37dbb7d1c867d64c0470df259e8feeedf172b482f13ea3bb6d1c4e")
    baseUrl.set("https://github.com/fedeloper/bitino/releases/download/v0.1.0")
    (findProperty("bitino.demoModelDir") as String?)?.let { localDir.set(it) }
    outputDir.set(layout.buildDirectory.dir("generated/demoModel"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(fetchDemoModel, FetchDemoModel::outputDir)
    }
}
