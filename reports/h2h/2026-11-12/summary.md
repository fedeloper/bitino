### CPH2551 (cph2551-control), cpu7, arm64

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

### CPH2551 (cph2551), cpu7, arm64

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

### moto e13 (moto-e13), cpu7, arm64

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

### M2006C3LG (redmi9a), cpu0, armv7

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

### Model files and output agreement on the 128-image eval set (host interpreters)

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

### On-device outputs on the timed input (eval image 0)

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
