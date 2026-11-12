# Minimal Android integration (JNI)

For an app, use the Android library instead: `android/bitino` (AAR, Kotlin API, preprocessing from the model
manifest; see `docs/android-deployment.md`). This directory is the bare minimum over the C API, without Gradle.

`bitino_jni.cc` wraps the stable C API (`runtime/include/bitino/bitino_c.h`), and
`java/org/bitino/Bitino.java` is the Java side. The top-level CMake builds
`libbitino_jni.so` for Android (`BITINO_BUILD_JNI=ON`, the default when `ANDROID` is set):

```bash
scripts/build_android.sh          # produces build/android-arm64/libbitino_jni.so
```

To use it in an app, copy `libbitino_jni.so` to `app/src/main/jniLibs/arm64-v8a/` and
`Bitino.java` into the app sources, then:

```java
try (Bitino bf = new Bitino(context.getFilesDir() + "/model.bitino", 4)) {
  float[] logits = bf.run(preprocessedNhwcImage);
}
```

Status: the shared library is cross-compiled in CI-style builds and the C API it wraps is
exercised by `tests/cpp/test_core.cc` (host and device). This Java wrapper itself has not been run in an app; the
JNI bridge of `android/bitino` (same C API) has, in the demo app on a phone.
