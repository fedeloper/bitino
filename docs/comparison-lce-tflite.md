# Bitino vs Larq Compute Engine vs TensorFlow Lite: 1-thread latency on phones

This page compares the latency of Bitino (this repository), Larq Compute Engine (LCE) and plain TensorFlow Lite
(TFLite). All three run the same Larq Zoo models with the same weights. Every number was measured on a real phone
with 1 thread, batch 1, pinned to one core. No emulated timing appears anywhere.

Reproduce it with `scripts/h2h_lce_tflite.sh` (steps `download`, `build`, `models`, `run`, `report`). It calls:

* `tools/h2h_models.py`: rebuilds, verifies and converts the models.
* `tools/h2h_bench.py`: runs the device benchmarks and writes the report.
* `tools/h2h_run.c`: the on-device launcher that every runtime runs under.
* `tools/h2h_tflite_names.py`: renames the input tensor.

The raw data are in `reports/h2h/2026-11-12/`:

* One directory per phone, holding:
  * `plan.json`: device, binaries with SHA-256, policy and the invocations.
  * `raw/<model>__<runtime>__r<repeat>.json`: the command, launcher report (placement, peak RSS), timing, cap and
    thermal state before and after the round, and the round order. Bitino runs also keep the full `bitino_bench`
    JSON.
  * `outputs/`: the output of every runtime on the timed input (binary; kept in the development log, not published here).
* `models/<model>.json`: the model build reports.
* `summary.json` and `summary.md`: the aggregates.

## Contents

1. [Results](#results)
2. [What runs](#what-runs)
3. [Models: same weights, verified](#models-same-weights-verified)
4. [Measurement method](#measurement-method)
5. [Differences in setup, caveats](#differences-in-setup-caveats)
6. [What could not be compared](#what-could-not-be-compared)

## Results

**Summary.**

* **Bitino vs LCE, same weights and core:**
  * OnePlus Open (Cortex-X3): Bitino is faster on all 11 models, by 1.09–1.71× (geometric mean 1.29×). The
    smallest margins are on the QuickNets; the largest are on Bi-Real/ResNet-E18 (1.47×) and XNOR-Net (1.71×).
  * Redmi 9A (32-bit Cortex-A53): Bitino is faster on all 5 models run, by 1.06–1.17× (geometric mean 1.11×).
  * Moto e13 (Cortex-A75): the two are about even (geometric mean 0.97×). LCE takes 6–10 % less time on the QuickNets,
    Bi-Real, ResNet-E18 and AlexNet; on XNOR-Net LCE takes 12 % longer; the DenseNets are within ±2 %.
* **Plain TFLite fp32**, what Android developers get by default, is 4.7–13× slower than Bitino on every phone
  (geometric mean 7.3× OnePlus, 6.2× Moto, 10.2× Redmi).
* **TFLite full-int8 PTQ** is 1.4–4.8× slower than Bitino, and it breaks accuracy: top-1 agreement with the
  reference is 0.45–0.66, and 0 on the DenseNets.
* **Model size:** Bitino and LCE files are within 1 % of each other, except XNOR-Net, where LCE's file is 89 MB vs
  Bitino's 24 MB. The TFLite fp32 files are 5–31× larger than Bitino's, and the int8 files 1.2–8× larger.
* **Peak RSS:** Bitino 14–54 MB, LCE 17–189 MB, TFLite fp32 83–496 MB.

One-sentence summary:

> On the same Larq Zoo binary networks and weights, at 1 thread, Bitino is 1.29× faster than Larq Compute Engine on
> a Cortex-X3 (all 11 models, up to 1.71×), 1.11× faster on a 32-bit Cortex-A53, and on par on a Cortex-A75 (LCE
> 3 % ahead on average), while plain TensorFlow Lite is 6–10× slower than Bitino in fp32 and loses most of the
> accuracy in int8.

**OnePlus Open (X3 @2.23 GHz cap)**, median ms (1 thread, batch 1); the last column is LCE ÷ Bitino (> 1: Bitino faster)

| model | Bitino | LCE 0.16 | TFLite fp32 | TFLite int8 | LCE ÷ Bitino |
|---|---:|---:|---:|---:|---:|
| quicknet-small | **4.90** | 5.33 | 45.33 | 11.45 | 1.09 |
| quicknet | **7.62** | 8.57 | 70.25 | 18.00 | 1.12 |
| quicknet-large | **13.57** | 15.22 | 136.47 | 34.53 | 1.12 |
| birealnet | **10.62** | 15.65 | 69.73 | 18.55 | 1.47 |
| binary-resnet-e18 | **10.68** | 15.77 | 69.96 | 18.68 | 1.48 |
| xnornet | **7.91** | 13.53 | 48.88 | 11.28 | 1.71 |
| binary-alexnet | **5.48** | 7.71 | 37.53 | 8.50 | 1.41 |
| binary-densenet28 | **22.74** | 29.40 | 143.10 | 42.62 | 1.29 |
| binary-densenet37 | **26.43** | 33.54 | 176.41 | 52.19 | 1.27 |
| binary-densenet37-dilated | **47.63** | 56.38 | 306.67 | 87.25 | 1.18 |
| binary-densenet45 | **35.72** | 42.81 | 247.64 | 72.24 | 1.20 |
| geometric mean | | | | | 1.29 |

**Moto e13 (A75 @1.61 GHz)**, median ms (1 thread, batch 1); the last column is LCE ÷ Bitino (> 1: Bitino faster)

| model | Bitino | LCE 0.16 | TFLite fp32 | TFLite int8 | LCE ÷ Bitino |
|---|---:|---:|---:|---:|---:|
| quicknet-small | 26.53 | **23.95** | 220.78 | 114.81 | 0.90 |
| quicknet | 43.37 | **40.18** | 352.83 | 182.57 | 0.93 |
| quicknet-large | 74.37 | **68.77** | 690.25 | 356.82 | 0.92 |
| birealnet | 69.25 | **63.13** | 343.61 | 176.74 | 0.91 |
| binary-resnet-e18 | 69.53 | **63.52** | 343.95 | 176.83 | 0.91 |
| xnornet | **46.98** | 52.55 | 223.34 | 110.90 | 1.12 |
| binary-alexnet | 32.44 | **30.38** | 172.86 | 84.45 | 0.94 |
| binary-densenet28 | **134.54** | 137.62 | 749.02 | 390.92 | 1.02 |
| binary-densenet37 | **153.75** | 155.97 | 941.82 | 485.39 | 1.01 |
| binary-densenet37-dilated | 281.79 | **277.10** | 1681.72 | 847.54 | 0.98 |
| binary-densenet45 | **204.98** | 206.86 | 1317.65 | 687.83 | 1.01 |
| geometric mean | | | | | 0.97 |

**Redmi 9A (A53 @2.0 GHz, 32-bit)**, median ms (1 thread, batch 1); the last column is LCE ÷ Bitino (> 1: Bitino faster)

| model | Bitino | LCE 0.16 | TFLite fp32 | TFLite int8 | LCE ÷ Bitino |
|---|---:|---:|---:|---:|---:|
| quicknet-small | **58.48** | 64.39 | 762.27 | 243.58 | 1.10 |
| quicknet | **96.35** | 102.24 | 1040.73 | 385.49 | 1.06 |
| quicknet-large | **163.34** | 174.54 | 2131.32 | 748.69 | 1.07 |
| birealnet | **132.56** | 155.41 | 1022.69 | 392.67 | 1.17 |
| binary-resnet-e18 | **134.52** | 156.20 | 1020.98 | 393.96 | 1.16 |
| geometric mean | | | | | 1.11 |


### Full tables

Generated by `tools/h2h_bench.py report` (`reports/h2h/2026-11-12/summary.md`).

* `bitino_neon`: Bitino's binary NEON kernels with reference float layers.
* `cph2551-control`: the same fp32 `.tflite` under the TFLite nightly binary and the LCE 0.16 binary.
* Peak RSS is `ru_maxrss`.
* On-device outputs are the probabilities for eval image 0 compared with the reference.

#### CPH2551 (cph2551-control), cpu7, arm64

| model | tflite_fp32 | lcebin_fp32 | LCE / Bitino (all) |
|---|---:|---:|---:|
| quicknet-small | 45.48 | 45.59 | n/a |
| quicknet | 70.54 | 70.37 | n/a |
| quicknet-large | 136.60 | 137.13 | n/a |
| birealnet | 69.93 | 69.82 | n/a |
| binary-resnet-e18 | 70.21 | 69.94 | n/a |

Peak RSS (MB, median over repeats) and run-to-run spread (max/min of the per-run means) on cph2551-control:

| model | tflite_fp32 RSS | lcebin_fp32 RSS | worst spread |
|---|---:|---:|---:|
| quicknet-small | 117.0 | 116.1 | 1.008 |
| quicknet | 123.7 | 123.4 | 1.004 |
| quicknet-large | 202.4 | 201.6 | 1.095 |
| birealnet | 116.2 | 115.3 | 1.004 |
| binary-resnet-e18 | 117.0 | 116.1 | 1.003 |

#### CPH2551 (cph2551), cpu7, arm64

| model | bitino_all | bitino_neon | lce | tflite_fp32 | tflite_int8 | LCE / Bitino (all) |
|---|---:|---:|---:|---:|---:|---:|
| quicknet-small | 4.90 | 23.32 | 5.33 | 45.33 | 11.45 | 1.09x |
| quicknet | 7.62 | 40.78 | 8.57 | 70.25 | 18.00 | 1.12x |
| quicknet-large | 13.57 | 50.17 | 15.22 | 136.47 | 34.53 | 1.12x |
| birealnet | 10.62 | 174.69 | 15.65 | 69.73 | 18.55 | 1.47x |
| binary-resnet-e18 | 10.68 | 174.94 | 15.77 | 69.96 | 18.68 | 1.48x |
| xnornet | 7.91 | 147.56 | 13.53 | 48.88 | 11.28 | 1.71x |
| binary-alexnet | 5.48 | 96.02 | 7.71 | 37.53 | 8.50 | 1.41x |
| binary-densenet28 | 22.74 | 270.77 | 29.40 | 143.10 | 42.62 | 1.29x |
| binary-densenet37 | 26.43 | 277.73 | 33.54 | 176.41 | 52.19 | 1.27x |
| binary-densenet37-dilated | 47.63 | 535.99 | 56.38 | 306.67 | 87.25 | 1.18x |
| binary-densenet45 | 35.72 | 337.09 | 42.81 | 247.64 | 72.24 | 1.20x |

Peak RSS (MB, median over repeats) and run-to-run spread (max/min of the per-run means) on cph2551:

| model | bitino_all RSS | bitino_neon RSS | lce RSS | tflite_fp32 RSS | tflite_int8 RSS | worst spread |
|---|---:|---:|---:|---:|---:|---:|
| quicknet-small | 15.4 | 13.7 | 25.6 | 117.0 | 43.7 | 1.016 |
| quicknet | 16.7 | 15.2 | 28.0 | 123.8 | 46.5 | 1.011 |
| quicknet-large | 20.1 | 18.6 | 30.8 | 202.4 | 67.8 | 1.063 |
| birealnet | 19.0 | 17.1 | 33.6 | 116.1 | 46.2 | 1.024 |
| binary-resnet-e18 | 19.0 | 17.1 | 34.4 | 116.9 | 46.4 | 1.007 |
| xnornet | 54.0 | 45.0 | 189.3 | 495.8 | 138.0 | 1.072 |
| binary-alexnet | 28.9 | 28.8 | 33.9 | 491.1 | 137.0 | 1.041 |
| binary-densenet28 | 27.7 | 24.2 | 53.4 | 85.2 | 45.1 | 1.008 |
| binary-densenet37 | 30.2 | 26.1 | 55.3 | 113.2 | 53.1 | 1.020 |
| binary-densenet37-dilated | 31.4 | 27.2 | 55.4 | 113.1 | 53.0 | 1.032 |
| binary-densenet45 | 35.4 | 29.8 | 59.3 | 153.3 | 63.6 | 1.017 |

#### moto e13 (moto-e13), cpu7, arm64

| model | bitino_all | bitino_neon | lce | tflite_fp32 | tflite_int8 | LCE / Bitino (all) |
|---|---:|---:|---:|---:|---:|---:|
| quicknet-small | 26.53 | 76.87 | 23.95 | 220.78 | 114.81 | 0.90x |
| quicknet | 43.37 | 137.46 | 40.18 | 352.83 | 182.57 | 0.93x |
| quicknet-large | 74.37 | 172.40 | 68.77 | 690.25 | 356.82 | 0.92x |
| birealnet | 69.25 | 616.62 | 63.13 | 343.61 | 176.74 | 0.91x |
| binary-resnet-e18 | 69.53 | 617.22 | 63.52 | 343.95 | 176.83 | 0.91x |
| xnornet | 46.98 | 489.69 | 52.55 | 223.34 | 110.90 | 1.12x |
| binary-densenet28 | 134.54 | 826.07 | 137.62 | 749.02 | 390.92 | 1.02x |
| binary-densenet37 | 153.75 | 847.87 | 155.97 | 941.82 | 485.39 | 1.01x |
| binary-alexnet | 32.44 | 323.88 | 30.38 | 172.86 | 84.45 | 0.94x |
| binary-densenet37-dilated | 281.79 | 1438.76 | 277.10 | 1681.72 | 847.54 | 0.98x |
| binary-densenet45 | 204.98 | 989.45 | 206.86 | 1317.65 | 687.83 | 1.01x |

Peak RSS (MB, median over repeats) and run-to-run spread (max/min of the per-run means) on moto-e13:

| model | bitino_all RSS | bitino_neon RSS | lce RSS | tflite_fp32 RSS | tflite_int8 RSS | worst spread |
|---|---:|---:|---:|---:|---:|---:|
| quicknet-small | 14.9 | 13.2 | 23.8 | 115.2 | 41.8 | 1.007 |
| quicknet | 16.1 | 14.6 | 26.2 | 122.0 | 44.6 | 1.017 |
| quicknet-large | 19.6 | 18.1 | 29.1 | 200.5 | 65.9 | 1.016 |
| birealnet | 18.5 | 16.6 | 31.9 | 114.6 | 44.5 | 1.007 |
| binary-resnet-e18 | 18.5 | 16.4 | 32.7 | 115.3 | 44.8 | 1.009 |
| xnornet | 53.6 | 44.5 | 187.6 | 494.2 | 136.2 | 1.008 |
| binary-densenet28 | 26.9 | 23.8 | 51.9 | 83.4 | 43.4 | 1.007 |
| binary-densenet37 | 29.6 | 25.4 | 53.6 | 111.3 | 51.0 | 1.008 |
| binary-alexnet | 28.3 | 28.3 | 32.4 | 489.9 | 135.0 | 1.006 |
| binary-densenet37-dilated | 30.7 | 26.6 | 53.7 | 111.4 | 51.2 | 1.018 |
| binary-densenet45 | 34.9 | 29.1 | 57.8 | 151.7 | 62.0 | 1.004 |

#### M2006C3LG (redmi9a), cpu0, armv7

| model | bitino_all | bitino_neon | lce | tflite_fp32 | tflite_int8 | LCE / Bitino (all) |
|---|---:|---:|---:|---:|---:|---:|
| quicknet-small | 58.48 | 343.39 | 64.39 | 762.27 | 243.58 | 1.10x |
| quicknet | 96.35 | 608.02 | 102.24 | 1040.73 | 385.49 | 1.06x |
| quicknet-large | 163.34 | 778.08 | 174.54 | 2131.32 | 748.69 | 1.07x |
| birealnet | 132.56 | 2165.16 | 155.41 | 1022.69 | 392.67 | 1.17x |
| binary-resnet-e18 | 134.52 | 2182.89 | 156.20 | 1020.98 | 393.96 | 1.16x |

Peak RSS (MB, median over repeats) and run-to-run spread (max/min of the per-run means) on redmi9a:

| model | bitino_all RSS | bitino_neon RSS | lce RSS | tflite_fp32 RSS | tflite_int8 RSS | worst spread |
|---|---:|---:|---:|---:|---:|---:|
| quicknet-small | 13.7 | 12.0 | 17.0 | 110.9 | 37.2 | 1.022 |
| quicknet | 14.9 | 13.5 | 19.8 | 117.8 | 40.0 | 1.019 |
| quicknet-large | 18.3 | 16.7 | 22.4 | 195.7 | 60.6 | 1.008 |
| birealnet | 17.2 | 15.3 | 23.7 | 108.7 | 38.2 | 1.008 |
| binary-resnet-e18 | 17.2 | 15.4 | 24.1 | 109.4 | 38.5 | 1.012 |

#### Model files and output agreement on the 128-image eval set (host interpreters)

| model | Bitino .bitino MB | LCE .tflite MB | TFLite fp32 MB | TFLite int8 MB | top-1 acc ref / LCE / fp32 / int8 | top-1 agreement with ref: LCE / fp32 / int8 | notes |
|---|---:|---:|---:|---:|---|---|---|
| binary-alexnet | 7.91 | 7.88 | 247.37 | 62.14 | 0.289 / 0.289 / 0.289 / 0.312 | 1.000 / 1.000 / 0.539 | kernel quantizers pre-applied (default TFLite fp32 conversion has 16 SIGN for 7 binarized inputs and a TRANSPOSE (weight quantizer not constant-folded)) |
| binary-densenet28 | 4.28 | 4.28 | 20.57 | 5.25 | 0.609 / 0.609 / 0.609 / 0.000 | 0.992 / 0.992 / 0.000 |  |
| binary-densenet37 | 5.43 | 5.43 | 34.87 | 8.86 | 0.672 / 0.672 / 0.672 / 0.000 | 0.938 / 0.938 / 0.000 |  |
| binary-densenet37-dilated | 5.43 | 5.43 | 34.87 | 8.87 | 0.664 / 0.656 / 0.656 / 0.000 | 0.984 / 0.984 / 0.000 |  |
| binary-densenet45 | 7.76 | 7.76 | 55.63 | 14.08 | 0.617 / 0.633 / 0.633 / 0.000 | 0.961 / 0.961 / 0.000 |  |
| binary-resnet-e18 | 4.21 | 4.20 | 46.76 | 11.85 | 0.547 / 0.547 / 0.562 / 0.484 | 0.984 / 0.969 / 0.656 |  |
| birealnet | 4.21 | 4.20 | 46.76 | 11.85 | 0.578 / 0.586 / 0.578 / 0.508 | 0.961 / 0.922 / 0.664 |  |
| quicknet | 4.39 | 4.39 | 52.97 | 13.42 | 0.656 / 0.641 / 0.664 / 0.422 | 0.969 / 0.992 / 0.516 |  |
| quicknet-large | 5.71 | 5.69 | 93.43 | 23.66 | 0.719 / 0.688 / 0.695 / 0.516 | 0.883 / 0.961 / 0.641 |  |
| quicknet-small | 4.22 | 4.21 | 50.66 | 12.83 | 0.594 / 0.586 / 0.594 / 0.359 | 0.992 / 1.000 / 0.453 |  |
| xnornet | 23.89 | 88.87 | 249.52 | 62.65 | 0.344 / 0.344 / 0.344 / 0.398 | 1.000 / 1.000 / 0.648 | kernel quantizers pre-applied (default LCE conversion has 4 LceBconv2d for 6 binary QuantConv2D layers (weight quantizer not constant-folded)); LCE: 5 of 6 binary convolutions are LceBconv2d |

#### On-device outputs on the timed input (eval image 0)

max |Δ probability| vs the reference (original Keras/larq model, TF 2.15.1), top-1 class in brackets (the reference top-1 is given in the model column); `=host` marks a device output identical within 1e-5 to the host interpreter's output of the same .tflite.

cph2551-control:

| model (ref top-1) | tflite_fp32 | lcebin_fp32 |
|---|---|---|
| quicknet-small (15) | 4.3e-02 (15) =host | 4.3e-02 (15) =host |
| quicknet (15) | 1.1e-06 (15) =host | 8.0e-07 (15) =host |
| quicknet-large (15) | 8.9e-07 (15) =host | 8.0e-07 (15) =host |
| birealnet (15) | 7.1e-06 (15) =host | 7.1e-06 (15) =host |
| binary-resnet-e18 (10) | 8.0e-07 (10) =host | 7.9e-07 (10) =host |

cph2551:

| model (ref top-1) | bitino_all | bitino_neon | lce | tflite_fp32 | tflite_int8 |
|---|---|---|---|---|---|
| quicknet-small (15) | 4.3e-02 (15) | 3.3e-07 (15) | 1.4e-01 (15) =host | 4.3e-02 (15) =host | 2.8e-01 (621) =host |
| quicknet (15) | 5.7e-07 (15) | 2.2e-07 (15) | 1.2e-05 (15) =host | 1.1e-06 (15) =host | 2.5e-01 (15) =host |
| quicknet-large (15) | 4.2e-07 (15) | 7.5e-07 (15) | 1.6e-01 (15) =host | 8.9e-07 (15) =host | 3.5e-01 (15) =host |
| birealnet (15) | 2.7e-06 (15) | 2.9e-06 (15) | 1.6e-06 (15) =host | 7.1e-06 (15) =host | 2.2e-01 (15) =host |
| binary-resnet-e18 (10) | 3.0e-07 (10) | 2.4e-07 (10) | 2.8e-07 (10) =host | 8.0e-07 (10) =host | 3.3e-01 (86) =host |
| xnornet (330) | 2.4e-07 (330) | 4.8e-07 (330) | 2.6e-06 (330) =host | 2.6e-06 (330) =host | 4.5e-01 (15) =host |
| binary-alexnet (330) | 6.0e-08 (330) | 6.0e-08 (330) | 2.2e-07 (330) =host | 4.6e-06 (330) =host | 2.1e-01 (86) |
| binary-densenet28 (15) | 4.3e-02 (15) | 4.3e-02 (15) | 3.3e-02 (15) =host | 3.3e-02 (15) =host | 2.8e-01 (446) =host |
| binary-densenet37 (15) | 5.5e-02 (15) | 5.5e-02 (15) | 5.5e-02 (15) =host | 5.5e-02 (15) =host | 4.2e-01 (530) =host |
| binary-densenet37-dilated (15) | 8.5e-02 (15) | 8.5e-02 (15) | 9.5e-02 (15) =host | 9.5e-02 (15) =host | 6.4e-01 (814) =host |
| binary-densenet45 (15) | 4.1e-02 (15) | 6.0e-07 (15) | 4.1e-02 (15) =host | 4.1e-02 (15) =host | 3.6e-01 (451) =host |

moto-e13:

| model (ref top-1) | bitino_all | bitino_neon | lce | tflite_fp32 | tflite_int8 |
|---|---|---|---|---|---|
| quicknet-small (15) | 4.3e-02 (15) | 3.3e-07 (15) | 1.4e-01 (15) =host | 4.3e-02 (15) =host | 2.8e-01 (621) =host |
| quicknet (15) | 5.7e-07 (15) | 2.2e-07 (15) | 1.2e-05 (15) =host | 1.1e-06 (15) =host | 2.5e-01 (15) =host |
| quicknet-large (15) | 4.2e-07 (15) | 7.5e-07 (15) | 1.6e-01 (15) =host | 8.9e-07 (15) =host | 3.5e-01 (15) =host |
| birealnet (15) | 2.7e-06 (15) | 2.9e-06 (15) | 1.6e-06 (15) =host | 7.1e-06 (15) =host | 2.2e-01 (15) =host |
| binary-resnet-e18 (10) | 3.0e-07 (10) | 2.4e-07 (10) | 2.8e-07 (10) =host | 8.0e-07 (10) =host | 3.3e-01 (86) =host |
| xnornet (330) | 2.4e-07 (330) | 4.8e-07 (330) | 2.6e-06 (330) =host | 2.6e-06 (330) =host | 4.5e-01 (15) =host |
| binary-densenet28 (15) | 4.3e-02 (15) | 4.3e-02 (15) | 3.3e-02 (15) =host | 3.3e-02 (15) =host | 2.8e-01 (446) =host |
| binary-densenet37 (15) | 5.5e-02 (15) | 5.5e-02 (15) | 5.5e-02 (15) =host | 5.5e-02 (15) =host | 4.2e-01 (530) =host |
| binary-alexnet (330) | 6.0e-08 (330) | 6.0e-08 (330) | 2.2e-07 (330) =host | 4.6e-06 (330) =host | 2.1e-01 (86) |
| binary-densenet37-dilated (15) | 8.5e-02 (15) | 8.5e-02 (15) | 9.5e-02 (15) =host | - | 6.4e-01 (814) =host |
| binary-densenet45 (15) | 4.1e-02 (15) | 6.0e-07 (15) | 4.1e-02 (15) =host | 4.1e-02 (15) =host | 3.6e-01 (451) =host |

redmi9a:

| model (ref top-1) | bitino_all | bitino_neon | lce | tflite_fp32 | tflite_int8 |
|---|---|---|---|---|---|
| quicknet-small (15) | 4.3e-02 (15) | 3.3e-07 (15) | 4.3e-02 (15) | 4.3e-02 (15) | 2.8e-01 (621) =host |
| quicknet (15) | 9.5e-07 (15) | 2.2e-07 (15) | 1.3e-05 (15) =host | 1.0e-06 (15) =host | 2.5e-01 (15) =host |
| quicknet-large (15) | 4.8e-07 (15) | 7.5e-07 (15) | 2.4e-01 (15) | 8.6e-07 (15) =host | 3.5e-01 (15) =host |
| birealnet (15) | 2.7e-06 (15) | 2.9e-06 (15) | 1.6e-06 (15) =host | 7.7e-06 (15) =host | 2.2e-01 (15) =host |
| binary-resnet-e18 (10) | 2.8e-07 (10) | 2.4e-07 (10) | 3.0e-07 (10) =host | 8.4e-07 (10) =host | 3.3e-01 (86) =host |



## What runs

| runtime | binary | version / source | how the model is produced |
|---|---|---|---|
| **Bitino, stage `all`** | `bitino_bench` (this repo at the commit of this page, NDK r27c, `-O3`, `c++_static`) | `build/android-arm64`, `build/android-armv7` (`scripts/build_android.sh`) | `artifacts/<model>/model.bitino` (read-only, the published Bitino model). Every kernel tier allowed: binary NEON kernels plus `float_opt` fp32 kernels; no INT8 |
| Bitino, stage `neon` | same | same | same file. Binary NEON kernels, but float layers on the portable reference kernels: the "binary kernels only" stage |
| **LCE** | `lce_benchmark_model_android_arm64` | GitHub release v0.16.0 (TF submodule = TensorFlow v2.16.1) | `lce.convert_keras_model(model)` (LCE 0.16.0 converter, target `arm`) |
| **TFLite fp32** | `android_aarch64_benchmark_model` / `android_arm_benchmark_model` | TFLite nightly "latest" object, Last-Modified 2025-02-07 | `tf.lite.TFLiteConverter.from_keras_model(model)` (TF 2.16.1): binarization stays as float `SIGN` ops, convolutions are float |
| **TFLite int8** | same as fp32 | same | the same converter with full-integer post-training quantization (`Optimize.DEFAULT`, int8 input/output, `TFLITE_BUILTINS_INT8`), calibrated on 128 ImageNet-100 **training** images (`.cache/data/dev1000_imagenet100train_larq`, disjoint from the eval set). `SIGN` has no int8 kernel: it runs in float between `DEQUANTIZE`/`QUANTIZE` |

Binary URLs and SHA-256, all downloaded on 2026-11-12:

| file | URL | SHA-256 |
|---|---|---|
| lce_benchmark_model_android_arm64 | https://github.com/larq/compute-engine/releases/download/v0.16.0/lce_benchmark_model_android_arm64 | `4068e0c40b3ef51d167be1ab2677304c3bf3fcb8ff2651ef81f8295e025d1c90` |
| lce_benchmark_model_aarch32 (Redmi, see below) | https://github.com/larq/compute-engine/releases/download/v0.16.0/lce_benchmark_model_aarch32 | `8670b8a96b9a8617c635f1e47b24b54734e802b925f7e167fdba25d557158877` |
| android_aarch64_benchmark_model | https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model | `5d45116eb86e3a23ea03318e93bfc8efb4dd0149d4692c127e7b0343faba1374` |
| android_arm_benchmark_model | https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_arm_benchmark_model | `3f3c0ed3436d1e506d78bf11ed5342c93acf64cff3145f1237ebd8a9047de068` |
| libc6 2.36-9+deb12u14 armhf (Redmi LCE only) | https://deb.debian.org/debian/pool/main/g/glibc/libc6_2.36-9+deb12u14_armhf.deb | `758c68b92654747025b48476a45c315bf16df1321224abdd9be672bfd120be45` |
| libstdc++6 12.2.0-14+deb12u1 armhf | https://deb.debian.org/debian/pool/main/g/gcc-12/libstdc++6_12.2.0-14+deb12u1_armhf.deb | `66d53ddeb10a37f22e455611c2c515f9d444e523b55d873ff48bde47267963db` |
| libgcc-s1 12.2.0-14+deb12u1 armhf | https://deb.debian.org/debian/pool/main/g/gcc-12/libgcc-s1_12.2.0-14+deb12u1_armhf.deb | `f58562bf01efd6112b914182147ee149e0d4c90869046e31844da8803669eeb3` |

The ELF notes give the build toolchains. The LCE Android binary was built by NDK r25c for Android 30. The TFLite
binaries were built by NDK r25b for Android 21. The TFLite binaries come from a "nightly/latest" object, so a
re-download can return a different build. The script checks the pinned SHA-256 and warns when it changed.

Host conversion environment, used read-only: conda env `lce`, with Python 3.11, tensorflow 2.16.1 + tf_keras 2.16.0
(`TF_USE_LEGACY_KERAS=1`), larq 0.14.0, larq_compute_engine 0.16.0 and h5py 3.16.

## Models: same weights, verified

The 11 Larq Zoo models that both Bitino and LCE support are QuickNet-Small/QuickNet/QuickNet-Large (v1.0),
Bi-Real Net, Binary ResNet-E18, XNOR-Net, Binary AlexNet (v0.3.0) and Binary DenseNet 28/37/37-dilated/45.
DoReFaNet is excluded because LCE does not support it. MeliusNet22 and Real-to-Binary Net were left out of this
round. Both are supported by Bitino and listed by LCE, but rebuilding their Lambda/SE topologies by hand was not
done.

* **No larq_zoo code was executed** (project rule, the development log (not published)). `tools/h2h_models.py`
  re-writes each topology from the larq_zoo 2.4.0 source, which was only read. It uses only the `larq` package's
  layers and quantizers. XNOR-Net's weight quantizer (clip, per-output-channel mean |W| × sign) is re-written in
  the same few lines.
* **Weights:** the original `.h5` files (`.cache/checkpoints/<model>/`, SHA-256 equal to the hashes in
  larq_zoo) are read with h5py. They are assigned layer by layer in the stored `layer_names` order. The layer
  types must match in order, every tensor shape must match, and the full Keras auto-names match too, except for
  QuickNet-Large, whose file was saved with a session counter offset (`quant_conv2d_63`, ...).
* **Check against Bitino's reference:** each rebuilt Keras model is run on the 128-image eval set
  (`.cache/data/eval128/larq_zoo.f32`, the inputs of `artifacts/<model>/reference_*.f32`). It is compared with
  `reference_logits.f32` and `reference_probs.f32`, which were produced by the original larq_zoo models under
  TF 2.15.1 / larq 0.13.3. **All 11 models reproduce the reference logits exactly** (max |Δ logit| = 0, 128/128
  top-1). The build fails below 98 % top-1 agreement.
* **Conversion:** the same Keras model is converted three ways (LCE, TFLite fp32, TFLite int8).
  * **Kernel quantizers pre-applied, XNOR-Net and Binary AlexNet.** The converters do not constant-fold the weight
    quantizer of very large kernels, so clip/abs/mean/sign/transpose stay in the graph and run at every inference.
    The affected kernels are XNOR-Net's 6×6×256×4096 conv and AlexNet's 9216×4096 "dense" conv. In XNOR-Net's
    case LCE then also misses 2 of the 6 binary convolutions. For these two models every kernel quantizer is
    applied once in TF (same function, same weights) and stored as the constant kernel with
    `kernel_quantizer=None`. The math is identical (the Keras check above is repeated on the frozen model: exact
    again), and it matches what Bitino's converter does offline. The rule is automatic and recorded in each build
    report (`weights_prequantized`): it triggers when LCE leaves binary convolutions unconverted or when TFLite fp32
    has more `SIGN` ops than 2 per binarized input, or a `TRANSPOSE`. The other 9 models convert completely as is.
  * **Input tensor renamed.** LCE 0.16's benchmark tool cannot parse `--input_layer_value_files` for the
    converter's input name `serving_default_input_1:0`. `tools/h2h_tflite_names.py` renames it in place to
    `serving_default_input_1_0`: the same byte length, nothing else changes. The build reports keep the
    converter's SHA-256 next to the final one.
  * **Determinism.** Rebuilding QuickNet-Small and DenseNet28 from scratch reproduces every `.tflite` and input
    file byte for byte.

## Measurement method

* **Same input, same work.** Every runtime is timed on eval image 0, the same float32 NHWC tensor. Bitino reads it
  with `--inputs`, and the TFLite tools read it with `--input_layer_value_files`; the int8 model gets the same image
  quantized with its input scale and zero point. All runtimes run batch 1 with 1 thread.
* **Same warm-up and iterations:** 10 untimed runs, then 50 timed runs.
  * TFLite and LCE get `--warmup_runs=10 --warmup_min_secs=0 --num_runs=50 --min_secs=0 --max_secs=100000`.
  * Bitino gets 1 correctness run on the input, plus `--warmup 9 --iterations 50`.
  * The timed region is `Interpreter::Invoke()` for TFLite and `Session::run()` for Bitino. Model load and
    preparation are excluded from both and recorded separately.
* **Statistic.** `benchmark_model` prints only the mean of the timed runs (plus min, max and std), so the per-run
  number is **the mean of the 50 timed runs** for every runtime. The table shows the **median over repeats** of
  these means. Bitino's per-run medians are kept in the raw data.
* **Same affinity mechanism, verified placement.** Every runtime runs as `h2h_run --cpus <c> -- <binary> ...`, and
  none of the benchmark binaries is asked to pin itself: Bitino runs without `--affinity`.
  * The launcher calls `sched_setaffinity()` in its child before `exec`, reads the mask back and accepts it only
    when it equals the requested CPU.
  * On the OnePlus Open, Qualcomm core_ctl *pauses* the idle Cortex-X3. `sched_setaffinity(cpu7)` then fails with
    `EINVAL`, or returns 0 while the kernel substitutes other CPUs (read back as `0-5`). The launcher busy-spins for
    20 ms and retries until the mask reads back as `7`. The X3 typically wakes after 2–4 tries (40–80 ms), and the
    wait is recorded per run (`pin_wait`). The same procedure applies to all runtimes.
  * Placement is then verified during the run. Every 25 ms the launcher reads the core each thread of the benchmark
    process last ran on (`/proc/<pid>/task/<tid>/stat` field 39) and the process's `Cpus_allowed_list`.
  * A run counts only when every sample of every thread is on the requested CPU and the allowed list never changed.
    Otherwise it is repeated, up to 3 attempts.
  * Result: **all accepted runs on all phones show 100 % of samples on the pinned core** (`off_cpu_samples` = 0 in
    `summary.json`), and every runtime ran a single thread.
* **Interleaving.** One round = one model with every runtime run once. The order is rotated by (repeat + model
  index), and the rounds cycle over the models for each repeat. The phone's device lock
  (`/tmp/bitino-device-<serial>.lock`, `flock(2)`, shared with the other jobs on these phones) is held for the
  whole round, so no other job runs between two runtimes of the same round.
* **Clocks and thermals.** No governor change, no reboot, USB only.
  * Every phone has a **reference cap**: the `scaling_max_freq` of the pinned core's cpufreq policy at the start of
    its first invocation. The values are 2227200 kHz on the OnePlus cpu7, 1612000 kHz on the Moto cpu7 and
    2001000 kHz on the Redmi cpu0.
  * Before every round, the gate (`PinnedGate` in `tools/h2h_bench.py`) waits, for at most 600 s, until the Android
    thermal status is 0 and that cap equals the reference.
  * The thermal status and the `scaling_max_freq` of every policy are recorded before and after every round. A
    round counts as off-reference when either reading differs. Such a round is marked `*` in the tables and
    excluded from the medians whenever reference rounds exist.
  * A second pass with `--redo-throttled` measures off-reference rounds again. The old records are kept under
    `raw/throttled/`.
  * The caps of the other clusters are recorded but not gated on, because a 1-thread run on the pinned core does not
    use them. An earlier gate compared every cluster with the highest cap ever seen. A short screen-on period lifted
    that reference to 2.96 GHz, after which the gate flagged normal screen-off rounds, so it was replaced.
  * Result: **every accepted round on every phone ran at its reference cap with thermal status 0.** One OnePlus
    AlexNet round ended with cpu7 capped at 1.98 GHz; it was re-measured, and its old record is in
    `cph2551/raw/throttled/`.
* **Peak RSS** is `ru_maxrss` of the benchmark process from `wait4()`, the same measurement for every runtime. It
  includes the tool itself, its mapped binary and the mapped model file.
* **Model size** is the size of the file each runtime loads.

## Differences in setup, caveats

These are the known differences between the runtimes' setups, and what each one does to the numbers.

* **The clocks are vendor-capped, the same for every runtime.**
  * OnePlus Open: with the screen off, the vendor power policy holds the Cortex-X3 at 2.23 GHz of 3.19 GHz (and
    the A715/A710 at 1.79 of 2.80), at thermal status 0. Every accepted OnePlus round ran at the 2.23 GHz cap.
    Absolute OnePlus milliseconds are therefore *not* peak-clock numbers. The ratios between runtimes are
    unaffected.
  * Moto e13: every core reports 1.61 GHz as both the cap and the maximum.
  * Redmi 9A: cpu0–3 run at 2.0 GHz and cpu4–7 at 1.5 GHz, uncapped.
* **The TFLite runtime versions differ, but this does not move the float numbers.** LCE 0.16.0 bundles the TFLite
  runtime of TensorFlow 2.16.1 (2024-03); the TFLite columns use the nightly `benchmark_model` of 2025-02. A
  control on the OnePlus ran the same `tflite_fp32.tflite` under both binaries (`cph2551-control`: 5 models,
  3 repeats). The two agree within 0.4 %, for example QuickNet 70.54 vs 70.37 ms and ResNet-E18 70.21 vs 69.94 ms.
  So the LCE vs TFLite-fp32 gap is not a runtime-version effect.
* **XNNPACK.** Both TFLite tools delegate the float ops to XNNPACK by default; `delegate` in `summary.json` holds
  the node counts. For example, LCE QuickNet-Small delegates 31 of 63 nodes; the LCE binary ops run on LCE's own
  kernels. Bitino uses only its own kernels.
* **Zero-padded binary convolutions.** In larq, `padding="same"` with the default `pad_values=0` lets border taps
  see 0, a third value. LCE runs such convolutions only on its slower path: the ARM target with float output and
  no fused activation (`bconv2d.cc`, see the development log (not published)). Bi-Real Net, ResNet-E18, XNOR-Net,
  AlexNet and the DenseNets are built this way; QuickNet's `pad_values=1` is LCE's fast path. This is a property
  of the models, identical for both runtimes, and it explains part of the larger Bitino margin on those models.
* **Model-level conversion details.**
  * LCE keeps XNOR-Net's 1×1 4096→4096 binary convolution as a float `CONV_2D` with ±α weights: 5 of 6 binary
    convolutions become `LceBconv2d`, even with the pre-applied kernel quantizers. That explains LCE's 89 MB
    XNOR-Net file and its higher peak RSS.
  * XNOR-Net and AlexNet are converted from Keras models with pre-applied kernel quantizers (see above).
* **Bitino's graphs have two outputs.** They return the logits (the softmax input) as well as the probabilities.
  The LCE/TFLite graphs return only the probabilities. No extra compute is involved: the logits tensor exists in
  both graphs.
* **Bitino stage `all` uses `float_fast` (approximate) softmax.** It is bit-exact with the `neon` stage only on
  the binary part. Its probabilities match the reference within the tolerances of `docs/numerical-correctness.md`
  (see the agreement table).
* **The untimed runs differ slightly.** Bitino's first untimed run is the correctness pass on the input, followed
  by 9 warm-up runs; the TFLite tools do 10 warm-up runs. All runtimes do 10 untimed runs on the same input.
* **Statistic.** Each run reports the mean of its 50 timed runs. Over all Bitino runs, the mean/median ratio is
  0.998–1.010. The tables show the median over repeats.
  * Repeats: the OnePlus used 5 for every model. The Moto used 5 for the first five models and 3 for XNOR-Net,
    AlexNet and the DenseNets. The Redmi used 3.
  * The max/min spread of the per-run means is in the RSS tables.
  * Runs longer than 1 s (Moto DenseNet-37-dilated TFLite fp32) were first recorded as failed: the early parser did
    not read `avg=1.68347e+06`. The report re-parses the kept tool output of such runs. Only runs that exited 0,
    had verified placement and completed 50 timed runs are re-parsed, and they are marked `reparsed`. The first
    of these runs had no output pulled, so that one cell of the output table is empty.
* **Peak RSS includes each tool.** That covers the mapped executable (bitino_bench is 30 MB unstripped, the
  TFLite tools 6–7 MB) and whatever pages of the model file were touched. Bitino maps the model and releases the
  packed constants after prepacking. TFLite fp32 holds all float weights, so its RSS grows with the float model
  size.
* **TFLite int8 is a latency reference, not a usable model.** Plain post-training int8 quantization of a
  binarized network loses most of its accuracy on this eval set. Top-1 agreement with the reference is 0.45–0.66
  for QuickNet/Bi-Real/ResNet-E/XNOR/AlexNet and 0 for every DenseNet; see the model table. The `SIGN` ops stay in
  float. It is listed because it is what an Android developer gets from the default TFLite int8 path.
* **Other jobs shared the phones.** Other benchmark and correctness suites ran on the same phones between rounds,
  under the same lock, never inside a round. Several batches waited up to about 2 h for a suite that held a phone's
  lock. Every round was gated on the reference cap (see above). The Redmi run started only after its correctness
  suite had ended.
* **Redmi 9A (32-bit only): LCE runs the Linux AArch32 build through glibc.** No armeabi-v7a Android build of
  `lce_benchmark_model` is published. The v0.16.0 release ships `lce_benchmark_model_aarch32`, a glibc Linux
  binary (`--config=rpi3`: `-march=armv7-a -mfpu=neon-vfpv4`), and no Bazel build was attempted.
  * It runs on the phone through Debian bookworm's armhf loader and libraries, pushed to
    `/data/local/tmp/h2h/glibc`: `ld-linux-armhf.so.3 --library-path glibc lce_benchmark_model ...`. URLs and
    hashes are in the binaries table.
  * Same kernel, same core, same launcher and placement checks. Only the C library and allocator differ (glibc
    instead of bionic), and the inference loop barely uses either.
  * The Redmi LCE output agrees with the host LCE interpreter to within 1e-5 on 3 of 5 models. On QuickNet-Small
    and QuickNet-Large it differs, but it equals the TFLite-fp32 output on QuickNet-Small (4.3e-2) and keeps the
    reference top-1 on both: the 32-bit build rounds float differently, and these binary networks amplify that.
  * Bitino runs its armeabi-v7a build (`ANDROID_ABI=armeabi-v7a scripts/build_android.sh`, AArch32 NEON kernels),
    and TFLite runs the official `android_arm_benchmark_model`.
  * Only the five most-used models were run on the Redmi (QuickNet ×3, Bi-Real, ResNet-E18; 3 repeats), because
    its TFLite fp32 runs take 1–2 s per inference.

## What could not be compared

* **DoReFaNet:** LCE does not support it (2-bit activations).
* **MeliusNet22, Real-to-Binary Net:** not rebuilt in Keras in this round (Lambda split/concat/pad stem and the
  data-dependent SE scaling), so there is no LCE/TFLite model for them yet.
* **QuickNet-XL (legacy):** not in the current Larq Zoo, and LCE conversion is unverified.
* **Redmi 9A, other 6 models** (XNOR-Net, AlexNet, DenseNets): not run, for time.
* **LCE on 32-bit Android as an Android binary:** none is published, so the Linux AArch32 build was used via glibc
  (above). Building `lce_benchmark_model` for `android_arm` with Bazel was not attempted.
* **Multi-thread latency, GPU/NNAPI delegates, Samsung S22:** out of scope. The S22 was not connected.
