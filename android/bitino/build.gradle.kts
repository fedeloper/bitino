import com.vanniktech.maven.publish.AndroidSingleVariantLibrary
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("com.vanniktech.maven.publish")
}

val nativeJobs = (findProperty("bitino.nativeJobs") as String?)?.toIntOrNull() ?: 0

android {
    namespace = "io.github.fedeloper.bitino"
    compileSdk = 36
    ndkVersion = rootProject.extra["bitinoNdkVersion"] as String   // NDK location: see ../build.gradle.kts
    (rootProject.extra["bitinoNdkPath"] as String?)?.let { ndkPath = it }

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a") }
        externalNativeBuild {
            cmake {
                targets += "bitino_jni"
                // The kernels are always built optimized (also in debug variants: a -O0 runtime is ~10x slower).
                arguments += listOf(
                    "-DANDROID_STL=c++_static",
                    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",   // 16 KB page devices (Android 15+)
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DBITINO_NATIVE_JOBS=$nativeJobs",
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
    explicitApi()
}

dependencies {
    testImplementation("junit:junit:4.13.2")
}

// Publishing (prepared, not run): see docs/android-deployment.md "Publishing the AAR".
// Coordinates come from gradle.properties (GROUP, POM_ARTIFACT_ID, VERSION_NAME).
mavenPublishing {
    configure(AndroidSingleVariantLibrary(variant = "release", sourcesJar = true, publishJavadocJar = true))
    publishToMavenCentral()
    // Maven Central requires signed artifacts; signing is enabled when a key is configured
    // (ORG_GRADLE_PROJECT_signingInMemoryKey, ...), so local / JitPack builds work without one.
    if (providers.gradleProperty("signingInMemoryKey").isPresent) signAllPublications()
    pom {
        name.set("Bitino for Android")
        description.set("On-device inference of binary neural networks (1-bit CNNs and transformers) on Arm CPUs: " +
            "the Bitino runtime as an Android library with a small Kotlin API.")
        url.set("https://github.com/fedeloper/bitino")
        inceptionYear.set("2026")
        licenses {
            license {
                name.set("Apache-2.0")
                url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                distribution.set("repo")
            }
        }
        developers {
            developer {
                id.set("fedeloper")
                name.set("Federico Fontana")
                url.set("https://github.com/fedeloper")
            }
        }
        scm {
            url.set("https://github.com/fedeloper/bitino")
            connection.set("scm:git:https://github.com/fedeloper/bitino.git")
            developerConnection.set("scm:git:ssh://git@github.com/fedeloper/bitino.git")
        }
    }
}
