# Android deployment

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
integer divide are runtime checks, so the binary runs on any NEON armeabi-v7a phone. int8 SDOT / I8MM and
int8_full kernels stay AArch64 only (the portable kernels run). The C++ tests build for armeabi-v7a too
(push and run `bitino_tests` as on arm64). Numerics: see "32-bit ARM (armeabi-v7a)" in
`docs/numerical-correctness.md`.

## Running a model

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

## App integration

Use the C API (`runtime/include/bitino/bitino_c.h`) or the JNI example (`examples/android_jni/`).
The APK path has not been built here (no Gradle/SDK build-tools installed). See the example README.

## Device facts (verified 2026-10-29)

See the development log (not published) (`./bitino device inspect`): SM-S901U1, SM8450,
cpu0–3 Cortex-A510 1.79 GHz, cpu4–6 Cortex-A710 2.50 GHz, cpu7 Cortex-X2 2.99 GHz, `walt` governor,
asimddp + i8mm + bf16 + fp16, no SVE, `perf_event_paranoid=-1` (simpleperf: perf_event_open is denied for adb shell on this phone, 2026-11-04).
