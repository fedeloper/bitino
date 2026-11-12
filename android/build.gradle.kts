import java.util.Properties

plugins {
    id("com.android.library") version "8.13.2" apply false
    id("com.android.application") version "8.13.2" apply false
    id("org.jetbrains.kotlin.android") version "2.2.21" apply false
    id("com.vanniktech.maven.publish") version "0.34.0" apply false
}

// NDK r27c (27.2.12479018), used by both modules (the native build and the stripping of native libraries). An NDK
// outside the SDK (e.g. an unpacked android-ndk-r27c) is used when given as `bitino.ndkPath` in local.properties or
// as $ANDROID_NDK / $ANDROID_NDK_HOME; otherwise the SDK-managed ndk/27.2.12479018.
val localProps = Properties().apply {
    rootProject.file("local.properties").takeIf { it.isFile }?.inputStream()?.use { load(it) }
}
extra["bitinoNdkVersion"] = "27.2.12479018"
extra["bitinoNdkPath"] = (localProps.getProperty("bitino.ndkPath") ?: System.getenv("ANDROID_NDK") ?: System.getenv("ANDROID_NDK_HOME"))
    ?.takeIf { file(it).isDirectory }
