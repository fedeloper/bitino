# Android deployment

Two ways to run Bitino on Android:

* **In an app: the Android library (AAR)** in `android/bitino`: the runtime for arm64-v8a and armeabi-v7a, a JNI
  bridge and a small Kotlin API that loads a model, preprocesses a `Bitmap` exactly as the model's manifest says and
  returns the scores. `android/demo` is a CameraX app built on it (live top-5, latency, fps).
* **From a shell: `bitino_bench`** pushed with adb, for benchmarks and correctness runs ([below](#command-line-runtime-adb)).

## The Android library

### Add it to an app

Until the library is on Maven Central, build it from this repository and use the AAR file (or `mavenLocal()`):

```bash
android/gradlew -p android :bitino:assembleRelease        # -> android/bitino/build/outputs/aar/bitino-release.aar
android/gradlew -p android :bitino:publishToMavenLocal    # -> ~/.m2: io.github.fedeloper:bitino-android:0.1.0
```

```kotlin
// app/build.gradle.kts
dependencies {
    implementation("io.github.fedeloper:bitino-android:0.1.0")   // mavenLocal() / Maven Central once published
    // or: implementation(files("libs/bitino-release.aar"))
}
```

### Use it

```kotlin
import io.github.fedeloper.bitino.Bitino

val model = Bitino.load(context, "ab-bnn-imagenet-reactnet18.bitino", threads = 1)  // asset, or an absolute file path
val scores: FloatArray = model.run(bitmap)                     // resize / crop / normalize per the model manifest
val top5 = Bitino.topK(Bitino.softmax(scores), 5)              // [(class index, probability), ...]
Log.i("bitino", "inference ${model.lastInferenceMs} ms, preprocessing ${model.lastPreprocessMs} ms")
model.close()
```

* `Bitino.load(context, name)` reads an asset (`src/main/assets/<name>`), `Bitino.load(file)` maps a file (e.g. a
  model downloaded at first launch), `Bitino.load(bytes)` takes the bytes of a content URI. The SHA-256 of the
  weights is checked at load (`verifyHash = false` skips it).
* `model.run(bitmap, rotationDegrees)` rotates first (CameraX: `imageProxy.imageInfo.rotationDegrees`, frames from
  `imageProxy.toBitmap()`), `model.run(floatArray)` takes an already preprocessed NHWC tensor (`model.inputShape`),
  `model.output(i)` returns further outputs of multi-output models.
* One `BitinoModel` runs one inference at a time (calls are serialized); it owns native memory: `close()` it.
* `threads`: 1 is the most energy-efficient choice for a camera loop; 2-4 lower the latency on the big cores. To change
  it, load the model again.

### Preprocessing

`model.preprocessing` is parsed from the manifest's `input.preprocessing` entry, the definition of the pipeline that
produced the model's evaluation tensors. Recognized pipelines (all the redistributable ImageNet models):

| manifest pipeline | steps |
|---|---|
| torchvision `Resize(N)` + `CenterCrop(M)` + `ToTensor()` + `Normalize(mean, std)` (bilinear or bicubic) | AB-BNN, Bi-Real / ReActNet / IR-Net families, bitorch, BHViT, Bi-ViT, daBNN |
| OpenVINO Accuracy Checker (`scale_greater` resize, center crop, RGB->BGR, 0..255) | OMZ ResNet-18 XNOR / ResNet-50 binary |
| `ToTensor()` (+ `Normalize`) without resize (32x32 CIFAR models) | the image is resized (short side) and center-cropped to the input size: the app's framing choice |

Other pipelines (TensorFlow `preprocess_image_bytes` of Larq Zoo, OpenCV-based ones, audio, text) give
`model.preprocessing == null`: set `model.preprocessing = Preprocessing(...)` or pass float tensors.

The resize is Pillow's resampling reproduced bit for bit (antialiased separable convolution, 22-bit fixed-point
coefficients, 8-bit intermediate, horizontal pass first, as `torchvision.transforms.Resize` on PIL images), the crop
offsets use Python's round-half-to-even, normalization is float32 `((v / 255) - mean) / std`. Checks
(`android/gradlew -p android :bitino:testDebugUnitTest`, JVM unit tests):

* synthetic images of 10 sizes (down- and upscaling, portrait, landscape, extreme aspect ratios) against fixtures made
  with torchvision 0.21 / Pillow 12.2 (`android/tools/make_preprocess_fixtures.py`): every float equal (SHA-256 of the
  tensor), for the torchvision bilinear and bicubic, OMZ and CIFAR pipelines;
* the 128 ImageNet validation images of the eval set (decoded with PIL, `android/tools/dump_eval_rgb.py`;
  `BITINO_EVAL_RGB_DIR=<dir>`): the Kotlin output equals `eval128/torchvision_imagenet.f32` and
  `eval128_omz_bgr/omz_bgr.f32` byte for byte, 128/128 for both (2026-11-12).

What is not reproduced is JPEG decoding (Android's decoder may differ from libjpeg-turbo in PIL by a few levels on
some pixels); camera frames are not JPEG-decoded anyway.

### ABIs, devices, size

* arm64-v8a: built for ARMv8.2-A + dot product (Cortex-A55 / A75 and newer, 2018+), the runtime's baseline; I8MM
  kernels are picked at run time where the CPU has them. On an older 64-bit core the library throws
  `UnsupportedOperationException` at first use (checked from `/proc/cpuinfo` before the native library is loaded)
  instead of crashing.
* armeabi-v7a: ARMv7-A + NEON, for 32-bit-only phones (see "32-bit ARM" below).
* `minSdk` 24. The native library is built with 16 KB page alignment (Android 15+ devices).
* Size: the AAR is 1.6 MB (libbitino_jni.so 2.3 MB arm64-v8a, 1.5 MB armeabi-v7a, stripped, static libc++); the
  demo APK is 9.0 MB, of which 4.3 MB is the bundled model.

### The demo app

`android/demo`: CameraX preview, the model's 224x224 center crop outlined, live top-5, preprocessing / inference
latency and fps, a thread selector (1 / 2 / 4), a back / front camera switch and a model selector:

* **ReActNet-18 (A&B BNN), bundled** (4.3 MB, ImageNet, weights MIT; its licence file is bundled next to it,
  `assets/models/ab-bnn-imagenet-reactnet18.LICENSE.txt`);
* **ReActNet-A (A&B BNN, MIT)** and **ResNet-18 XNOR (OpenVINO OMZ, Apache-2.0)**, downloaded from the release
  assets at first use and SHA-256-checked (or side-loaded while the release is not public: `adb push omz-resnet18-xnor-binary-onnx-0001.bitino
  /sdcard/Android/data/io.github.fedeloper.bitino.demo/files/models/`, the release asset's file name);
* **Load a model file...** (system file picker), e.g. any `.bitino` file converted with `bitino fetch`.

Only models with an explicit permissive weights licence are bundled or offered for download (`bitino list
--release`). Class names: `assets/imagenet_classes.txt`, the ImageNet-1k category names of torchvision
(`torchvision/models/_meta.py`, BSD 3-Clause, Copyright (c) Soumith Chintala 2016).

```bash
# the bundled model comes from a local copy of the release files, or is downloaded from the release at build time
android/gradlew -p android :demo:assembleRelease -Pbitino.demoModelDir=<dir with ab-bnn-imagenet-reactnet18.bitino>
adb install android/demo/build/outputs/apk/release/demo-release.apk
adb shell am start -n io.github.fedeloper.bitino.demo/.MainActivity --ei threads 4   # optional: threads, model index
adb logcat -s BitinoDemo     # one line per second: inference_ms, pre_ms, fps, top-1
android/tools/record_demo.sh # GIF of the running app (docs/img/android_demo.gif; needs ffmpeg)
```

Extras for scripted runs: `--ei threads 1|2|4`, `--ei model <index>`, `--es camera front`, `--ei capture_seconds N`.
The camera permission is asked at first launch.

The release APK is signed with the debug key (local installs); use your own signing config to distribute it.

#### On a phone (OnePlus Open, 2026-11-12)

What the demo app shows for the bundled ReActNet-18 (camera frames 640x480, `ImageProxy.toBitmap()` then
`model.run(bitmap, rotation)`; app threads are not pinned and the phone's stock power policy applies; medians of the
per-second log lines over 20 s):

| scene | threads | inference | preprocessing | fps |
|---|---|---|---|---|
| lit desk (camera at ~30 fps) | 1 | 17.6-18.9 ms | 5.2-6.8 ms | 29-30 (camera-bound) |
| dark scene (camera at 16.5 fps) | 1 | 27.0 / 28.4 ms | 12.3 / 12.8 ms | 16.5 (camera-bound) |
| dark scene | 2 | 21.9 ms | 14.6 ms | 16.5 |
| dark scene | 4 | 20.1 / 18.8 ms | 15.2 / 15.5 ms | 16.5 |

The frame rate is set by the camera (it drops to ~16 fps in low light); with the CPU idle between frames the vendor
governor lowers the clocks, so per-frame latency is higher than in `bitino_bench` runs on a pinned, busy core
(11.1 ms at 1 thread on the X3 core, 5.8 ms at 4 threads, same model; `results/speed_oneplus-open.json`). Right
after launch, at full clocks, the app showed 9.2 ms inference and 6.2 ms preprocessing at 1 thread. ReActNet-A
(~35 ms) and ResNet-18 XNOR (BGR / OMZ preprocessing, ~27 ms) also ran in the app, side-loaded (1 thread, dark
scene).

`docs/img/android_demo.gif` (not yet recorded: point the phone at an object and run the script) is made with `android/tools/record_demo.sh` (frames captured by the app itself
with `--ei capture_seconds N`: `adb shell screenrecord` is blocked on this phone), switching from 1 to 4 threads
half way.

### Building

Needs JDK 17+, the Android SDK (platform 36, build-tools 35.0.0, CMake 3.31.6) and NDK r27c (27.2.12479018). An
NDK outside the SDK directory is used when given as `bitino.ndkPath=<dir>` in `android/local.properties` or as
`$ANDROID_NDK`; `sdk.dir` goes in `android/local.properties` too (or `$ANDROID_HOME`). The native build compiles the
runtime from this checkout (`android/bitino/src/main/cpp/CMakeLists.txt` adds the top-level `CMakeLists.txt`, which
carries the per-ABI kernel flags), so the library is always built from a repository checkout. `-Pbitino.nativeJobs=N`
bounds the parallel compile jobs per ABI (peak memory).

```bash
cd android
./gradlew :bitino:testDebugUnitTest          # preprocessing tests (JVM)
./gradlew :bitino:assembleRelease            # AAR
./gradlew :demo:assembleRelease              # demo APK
```

### Publishing the AAR (prepared, not done)

Coordinates: `io.github.fedeloper:bitino-android:<VERSION_NAME>` (`android/gradle.properties`). The publishing setup
uses the [vanniktech maven-publish plugin](https://github.com/vanniktech/gradle-maven-publish-plugin) (POM, sources
and javadoc jars, signing, Central Portal upload); nothing has been published.

**Maven Central** (Sonatype Central Portal):

1. Sign in at <https://central.sonatype.com> with the GitHub account `fedeloper`: the namespace
   `io.github.fedeloper` is verified automatically for GitHub logins (otherwise verify it under "Namespaces").
2. Generate a user token (Account -> Generate User Token) and a GPG key; publish the public key
   (`gpg --keyserver keyserver.ubuntu.com --send-keys <KEY_ID>`).
3. Put the credentials in `~/.gradle/gradle.properties` (never in the repository):
   ```properties
   mavenCentralUsername=<token user>
   mavenCentralPassword=<token password>
   signingInMemoryKey=<output of: gpg --export-secret-keys --armor <KEY_ID> (newlines as \n)>
   signingInMemoryKeyId=<last 8 hex digits of the key id>
   signingInMemoryKeyPassword=<passphrase>
   ```
4. Set `VERSION_NAME` and tag the release, then
   `android/gradlew -p android :bitino:publishToMavenCentral` uploads a deployment to the Central Portal for review;
   check it under "Deployments" and press "Publish" (or use `publishAndReleaseToMavenCentral` to release directly).
   `-SNAPSHOT` versions go to the snapshot repository instead.

**JitPack** (no account; builds from a GitHub tag once the repository is public): `jitpack.yml` at the repository
root builds the library with `publishToMavenLocal`. Consumers add `maven("https://jitpack.io")` to their repositories
and depend on `com.github.fedeloper.bitino:bitino-android:<tag>` (JitPack's coordinates for a module of a
multi-module build; the build log at <https://jitpack.io/#fedeloper/bitino> shows them). JitPack's build machines
must provide NDK r27c and CMake 3.31.6; `jitpack.yml` installs them with `sdkmanager` first. The first build of a tag
takes several minutes.

## Deepfake screen detector app (`android/deepfake`)

A second app runs a "Faster Than Lies" binary deepfake detector (Lanzino et al., CVPRW 2024; BNext backbone with the
FFT and LBP features computed inside the model, converted by `tools/convert_ftl_deepfake.py`) on whatever the phone
shows. A foreground service mirrors the screen with MediaProjection into a small virtual display (the largest centred square
of the screen becomes 3·252 = 756 pixels). On every new frame:

1. **Find the images** with classical layout analysis (`RectFinder`, no learned model). This is a recursive X-Y cut on
   the capture halved: UI backgrounds are flat, so whole rows or columns of one colour separate a screen's elements
   (margins, gaps between posts, the space between a thumbnail and its text). A row or column is uniform when at most
   max(2, length / 64) of its pixels differ from its majority colour, so thin lines across the gaps (a toolbar shadow
   touching a grid of images, dividers) do not glue a whole grid into one block; on a Chrome image-search page the
   strict version saw one block, this one the 9 images. A block is split along its uniform rows,
   else its uniform columns (which also trims a card down to its contents), until neither exists. Leaves of at least
   64×64 capture pixels with at least 24 colours (4 bits per channel) and no luminance band on more than 75 % of the
   pixels are photos or videos; text, buttons and flat graphics are not. Up to 24 are followed. On synthetic feed
   screens (light and dark, with text, cards, buttons, full-width, inset and thumbnail photos) 92.5 % of the photos were
   found with IoU >= 0.8 (mean IoU 0.988).
2. **Run the detector on 4 images per frame, in rotation**: images never analysed first, then those that waited
   longest. When the next frame shows the same images, the next 4 get their turn; once all have had one, the rotation
   starts again. Android sends a new frame only when the screen changes, so on a still screen the last frame is kept and
   analysed again until every image has had 3 turns (an alert needs 3 fake results in a row). Each image is preprocessed like the authors' test loader (`Resample`): shorter side resized to
   252 with PIL's antialiased bilinear filter, centre crop 224, u8 / 255. Tested against PIL itself: within one level
   on every sample, at least 95 % exact. When no image is found, run it on the **whole view**, the largest centred square
   (3×3 box filter to 252, centre crop 224).
3. **Flag where the fake is, and keep the flag.** Images are followed from frame to frame (`Tracker`): an image whose
   four edges all moved by at most max(8, width / 100) capture pixels is the same image (it keeps its score and its
   place in the rotation); one that moved further (a scroll) is analysed again; one missing for more than 2 frames in a
   row is gone. When an image has 3 fake results in a row (threshold 50–99 %, default 70 %), it is tinted red with a
   dashed red and white border and its P(fake), and it stays so until it moves beyond that tolerance or disappears. A flagged image is
   not analysed again, so the tint never reaches the detector. The whole view (largest centred square) is analysed only
   while no image is tinted; when it fires, a red banner at the top of the screen stays until the screen changes
   (different images found, or the centre's 8×8 luminance changes by more than 6 levels on average).

**Each found image shows its P(fake)** in a small pill in its top-left corner (grey "…" until analysed, green below the
threshold, orange above). The detector's input leaves out the strip along the top of every image tall enough to carry
the pill (20 dp), whether the pill is drawn or not, so the label never reaches the detector.

**Found images are outlined** (a setting, on by default) with a dashed line in two contrasting colours: grey and white
until the image is analysed, green and white while P(fake) is below the threshold, orange and black above it. Whatever the overlay draws is captured too, so the
outline is drawn just inside the image's own edge (the background around it stays uniform), is never a uniform line
(so it is never trimmed as a border), and appears only for images that have stayed in place for 2 frames (nothing is
left behind on scrolling content). The detector's centre crop drops those edge pixels; the whole-view input sees them,
3× smaller. Everything the overlay draws inside an image starts 6 capture pixels from its edge (`OverlayGeometry`): on the
OnePlus a 2 dp inset was 2 capture pixels, and once the screen was scaled into the capture and halved, the outlines
bled into the 3-pixel separators of an image grid, so the grid alternated between 20 tiles and one block every frame
(seen as flicker). Flagged boxes follow their image within the tolerance, and only images found in the current frame
are drawn.

The detector never sees what the overlay draws: each image is analysed without a fixed margin from its edges (wider
than the outline band: 2 dp inset + 2 dp stroke + 2 capture pixels), applied whether an outline is drawn or not, and the
whole view is analysed only when no image is on screen (it is the fallback for screens without images found). The
overlay redraws only when what it shows changes (every redraw is a new captured frame).

The finder is deterministic: its result is a function of the captured pixels only (no state carries over between
frames). The overlay draws only inside images already found (and the banner above the content), and never a
single-colour line (inside an image it would read as a gap and the image would shrink a little on every frame): outlines
and the fake border are dashed in two colours. JVM tests check that the same pixels give the same rectangles, that no
state carries over between screens, and that the overlay redrawn frame after frame (outlines, a red-tinted image, a
wrong screen-sized outline) leaves the result equal to the clean screen's; checked locally on a real Chrome
image-search screenshot too (9 images, unchanged with the overlay over 3 frames).

The notification shows, once per second, the images found and their P(fake), the whole-view P(fake), the time per
frame and the frames analysed per second. With 3 images on screen a frame takes 115 ms on the OnePlus Open at 1 thread
(3.6 ms to find the images, 12.3 ms of crops, 4 detector runs); more threads lower it. Everything runs on the phone.

```bash
# local build with detectors bundled (the weights have no redistribution licence yet: do not distribute this APK)
android/gradlew -p android :deepfake:assembleRelease \
    -Pbitino.deepfakeModels=artifacts/ftl-deepfake-bnext-t-cocofake/model.bitino,artifacts/ftl-deepfake-bnext-t-dffd/model.bitino
adb install -r android/deepfake/build/outputs/apk/release/deepfake-release.apk
```

Without `-Pbitino.deepfakeModels` the APK contains no detector; "Add a converted model file…" imports one (any
`ftl-deepfake-*` file). COCOFake detectors target generated images in general, DFFD detectors target manipulated faces.
On start Android asks for "Display over other apps" (once) and for screen-capture consent (every time; choose the
entire screen). The service stops from its notification or the app.

Tests:
- `:deepfake:testDebugUnitTest` (JVM): image finder on synthetic screens (every photo with IoU >= 0.97, nothing else;
  a grid against a toolbar shadow; outlines drawn into the capture), resampler against PIL
  (`android/tools/make_resample_fixture.py`), tracker (a few pixels = the same image, a scroll = a new view, a flag
  stays put, missed frames), capture geometry, alert logic.
- On a phone, without using its screen (`python3 android/tools/make_deepfake_testdata.py`, then
  `:deepfake:assembleDebug :deepfake:assembleDebugAndroidTest` with the same `-Pbitino.deepfakeModels`, install both
  APKs and `adb shell am instrument -w io.github.fedeloper.bitino.deepfake.test/androidx.test.runner.AndroidJUnitRunner`):
  each bundled detector, run through the app's library on 8 CIFAKE eval images, against the reference phone's outputs.
  On the OnePlus Open (2026-11-16) both BNext-T detectors were bit-identical to the reference, at 25–26 ms per frame
  (1 thread, unpinned, inside the app).

## Command-line runtime (adb)

Requirements: Android NDK (tested r27c), CMake ≥ 3.20, Ninja, adb. No root access and no
device modifications are needed.

```bash
scripts/build_android.sh                       # Release, arm64-v8a, API 30+, static libc++
BUILD_TYPE=Debug scripts/build_android.sh build/android-arm64-O0   # -O0 build (benchmark stage 1)
scripts/android_run_tests.sh                   # pushes and runs bitino_tests on the device
```

The build baseline is `-march=armv8.2-a+dotprod`. Kernels that need more (e.g. I8MM) live in
separate translation units and are selected only when `getauxval(AT_HWCAP2)` reports the feature.

32-bit phones (`ANDROID_ABI=armeabi-v7a scripts/build_android.sh build/android-armv7`, e.g. the Redmi 9A,
whose Android is 32-bit only on 64-bit Cortex-A53 cores): the NDK's armeabi-v7a baseline is ARMv7-A + NEON
(VFPv3), and the binary, packing and float_opt NEON kernels are built for AArch32 under the same names and
tiers (`runtime/kernels/neon_a32.h` provides the AArch64-only intrinsics they use; the 16 accumulators of the
4x4 popcount tile are two 2x4 asm passes, the float GEMM tile 4x8). `kFeatNeon` comes from `AT_HWCAP`
(`HWCAP_NEON`); VFPv4 FMA (`fma` in the CPU string, used by the float_opt transcendental leaves) and the
integer divide are runtime checks, so the binary runs on any NEON armeabi-v7a phone. The int8 stages have AArch32
kernels as well: the int8_full NEON kernels, and VMULL.S8 GEMM tiles (`runtime/kernels/int8_neon/int8_gemm_a32.h`)
in place of SDOT / I8MM for int8_full (`*.neon_a32_mull`, `i8matmul.neon_a32`) and optimized_int8
(`conv2d / dense.int8_neon_a32`). AArch32 traps unaligned accesses carrying alignment hints (lane loads / stores
through `uint32_t*` on byte buffers): such code uses `memcpy` there; qemu-arm does not trap them, so run
`bitino_tests` on the phone. The C++ tests build for armeabi-v7a too
(push and run `bitino_tests` as on arm64). Numerics: see "32-bit ARM (armeabi-v7a)" in
`docs/numerical-correctness.md`.

### Running a model

```bash
ADB=~/Android/Sdk/platform-tools/adb; D=/data/local/tmp/bitino
$ADB shell mkdir -p $D
$ADB push build/android-arm64/bitino_bench $D/
$ADB push artifacts/larq-zoo-quicknet-small/model.bitino $D/model.bitino
$ADB shell $D/bitino_bench --model $D/model.bitino --threads 1 --affinity 7 \
    --warmup 20 --iterations 100 --json $D/single_thread.json
$ADB pull $D/single_thread.json .
```

Correctness on the device (same inputs as the host):

```bash
$ADB push .cache/data/eval128/larq_zoo.f32 $D/inputs.f32
$ADB shell $D/bitino_bench --model $D/model.bitino --inputs $D/inputs.f32 --outputs $D/out.f32 --iterations 1 --warmup 0
$ADB pull $D/out.f32 .
python3 tools/validate_model.py --artifacts artifacts/larq-zoo-quicknet-small --eval-dir .cache/data/eval128 \
    --outputs-file out.f32 --label device:all
```

The model file's data section is SHA-256-verified at load (`--no-verify-hash` disables it, and the
report then records that). Both the `--inputs` file size and the model's input shape are checked.

### Native code without the AAR

Other native code can link the runtime (`libbitino.a`, built by `scripts/build_android.sh`) and use the C API
(`runtime/include/bitino/bitino_c.h`); `examples/android_jni/` is a minimal Java JNI wrapper over it.

## Device facts (verified 2026-10-29)

See the development log (not published) (`./bitino device inspect`): SM-S901U1, SM8450,
cpu0–3 Cortex-A510 1.79 GHz, cpu4–6 Cortex-A710 2.50 GHz, cpu7 Cortex-X2 2.99 GHz, `walt` governor,
asimddp + i8mm + bf16 + fp16, no SVE, `perf_event_paranoid=-1` (simpleperf: perf_event_open is denied for adb shell on this phone, 2026-11-04).
