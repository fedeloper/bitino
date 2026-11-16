pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "bitino-android"
include(":bitino")   // the library (AAR)
include(":demo")     // CameraX demo app (APK)
include(":deepfake") // Faster Than Lies screen deepfake detector (APK)
