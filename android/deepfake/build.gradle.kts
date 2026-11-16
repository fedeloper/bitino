import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Screen deepfake detector: runs a "Faster Than Lies" binary detector (Lanzino et al., CVPRW 2024; converted with
// tools/convert_ftl_deepfake.py) on the centre of the phone screen and flashes the screen red on a fake image.
android {
    namespace = "io.github.fedeloper.bitino.deepfake"
    compileSdk = 36
    ndkVersion = rootProject.extra["bitinoNdkVersion"] as String   // strips the native libraries
    (rootProject.extra["bitinoNdkPath"] as String?)?.let { ndkPath = it }

    defaultConfig {
        applicationId = "io.github.fedeloper.bitino.deepfake"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // Instrumented-test inputs and reference outputs: written by android/tools/make_deepfake_testdata.py (build output,
    // not committed).
    sourceSets["androidTest"].assets.srcDir(layout.buildDirectory.dir("testdata"))

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
    implementation("androidx.activity:activity-ktx:1.12.4")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

// ---- Optional bundled detectors (local builds only) -------------------------------------------------------------
// The Faster Than Lies weights have no licence that allows redistribution yet, so no detector is bundled by default:
// the app loads a converted file picked on the phone. For a local build, -Pbitino.deepfakeModels=<file>[,<file>...]
// copies converted files into assets/models/ (artifacts/<model>/model.bitino becomes models/<model>.bitino). Do not
// distribute an APK built this way until the weights licence allows it.
abstract class BundleLocalModels : DefaultTask() {
    @get:InputFiles abstract val modelFiles: ConfigurableFileCollection
    @get:OutputDirectory abstract val outputDir: DirectoryProperty

    @TaskAction
    fun copy() {
        val dst = outputDir.get().asFile.resolve("models")
        dst.deleteRecursively()
        if (modelFiles.isEmpty) return
        dst.mkdirs()
        for (f in modelFiles.files) {
            check(f.isFile) { "bitino deepfake: model file $f not found" }
            val stem = f.nameWithoutExtension
            val name = if (stem.startsWith("model")) f.parentFile.name + stem.removePrefix("model") else stem
            f.copyTo(dst.resolve("$name.bitino"), overwrite = true)
            logger.lifecycle("bitino deepfake: bundled ${f.name} as models/$name.bitino (local build only)")
        }
    }
}

val bundleLocalModels = tasks.register<BundleLocalModels>("bundleLocalModels") {
    (findProperty("bitino.deepfakeModels") as String?)?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() }
        ?.let { modelFiles.from(it) }
    outputDir.set(layout.buildDirectory.dir("generated/deepfakeModels"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(bundleLocalModels, BundleLocalModels::outputDir)
    }
}
