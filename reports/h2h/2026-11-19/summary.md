### CPH2551 (oneplus-a510), cpu0, arm64

| model | bitino_all | bitino_neon | lce | tflite_fp32 | tflite_int8 | LCE / Bitino (all) |
|---|---:|---:|---:|---:|---:|---:|
| quicknet-small | 23.08 | 115.48 | 25.78 | 230.80 | 53.92 | 1.12x |
| quicknet | 36.39 | 205.76 | 41.35 | 349.85 | 84.87 | 1.14x |
| quicknet-large | 65.17 | 249.70 | 72.89 | 675.72 | 162.41 | 1.12x |
| birealnet | 53.64 | 1068.55 | 70.07 | 340.29 | 87.33 | 1.31x |
| binary-resnet-e18 | 54.77 | 1071.94 | 70.99 | 340.50 | 88.08 | 1.30x |
| xnornet | 41.39 | 899.13 | 55.91 | 218.58 | 52.84 | 1.35x |
| binary-alexnet | 28.77 | 595.56 | 33.68 | 167.36 | 39.96 | 1.17x |
| binary-densenet28 | 108.50 | 1381.19 | 141.47 | 704.74 | 212.41 | 1.30x |
| binary-densenet37 | 124.17 | 1412.34 | 161.75 | 878.21 | 260.58 | 1.30x |
| binary-densenet37-dilated | 224.68 | 2201.14 | 279.66 | 1547.50 | 445.94 | 1.24x |
| binary-densenet45 | 164.42 | 1596.73 | 214.91 | 1255.05 | 365.50 | 1.31x |

Peak RSS (MB, median over repeats) and run-to-run spread (max/min of the per-run means) on oneplus-a510:

| model | bitino_all RSS | bitino_neon RSS | lce RSS | tflite_fp32 RSS | tflite_int8 RSS | worst spread |
|---|---:|---:|---:|---:|---:|---:|
| quicknet-small | 16.0 | 14.5 | 25.6 | 116.9 | 43.7 | 1.012 |
| quicknet | 17.3 | 16.0 | 27.9 | 123.8 | 46.5 | 1.011 |
| quicknet-large | 20.7 | 19.5 | 30.7 | 202.3 | 67.7 | 1.011 |
| birealnet | 19.5 | 17.9 | 33.5 | 116.1 | 46.2 | 1.070 |
| binary-resnet-e18 | 19.6 | 17.9 | 34.3 | 116.9 | 46.2 | 1.046 |
| xnornet | 54.2 | 50.2 | 189.2 | 495.8 | 138.0 | 1.022 |
| binary-alexnet | 29.1 | 33.9 | 33.9 | 491.3 | 136.9 | 1.014 |
| binary-densenet28 | 28.6 | 24.9 | 53.4 | 85.2 | 45.1 | 1.014 |
| binary-densenet37 | 31.8 | 26.8 | 55.3 | 112.9 | 53.0 | 1.016 |
| binary-densenet37-dilated | 32.6 | 27.9 | 55.3 | 113.0 | 53.0 | 1.004 |
| binary-densenet45 | 37.6 | 30.4 | 59.3 | 153.1 | 63.6 | 1.005 |

### CPH2551 (oneplus-a715), cpu4, arm64

| model | bitino_all | bitino_neon | lce | tflite_fp32 | tflite_int8 | LCE / Bitino (all) |
|---|---:|---:|---:|---:|---:|---:|
| quicknet-small | 10.80 | 39.95 | 11.55 | 93.54 | 23.93 | 1.07x |
| quicknet | 17.41 | 70.39 | 18.39 | 147.68 | 37.14 | 1.06x |
| quicknet-large | 31.48 | 88.00 | 32.99 | 289.57 | 70.54 | 1.05x |
| birealnet | 24.36 | 292.97 | 29.55 | 144.84 | 37.20 | 1.21x |
| binary-resnet-e18 | 24.47 | 293.45 | 29.69 | 145.03 | 37.41 | 1.21x |
| xnornet | 18.67 | 235.51 | 24.13 | 98.51 | 22.01 | 1.29x |
| binary-alexnet | 12.83 | 154.25 | 14.92 | 74.92 | 16.40 | 1.16x |
| binary-densenet28 | 49.28 | 430.28 | 55.47 | 308.38 | 88.01 | 1.13x |
| binary-densenet37 | 56.77 | 441.46 | 63.70 | 380.25 | 108.70 | 1.12x |
| binary-densenet37-dilated | 102.69 | 819.98 | 113.96 | 670.30 | 186.32 | 1.11x |
| binary-densenet45 | 76.16 | 530.12 | 84.76 | 536.33 | 152.57 | 1.11x |

Peak RSS (MB, median over repeats) and run-to-run spread (max/min of the per-run means) on oneplus-a715:

| model | bitino_all RSS | bitino_neon RSS | lce RSS | tflite_fp32 RSS | tflite_int8 RSS | worst spread |
|---|---:|---:|---:|---:|---:|---:|
| quicknet-small | 16.0 | 14.5 | 25.6 | 116.9 | 43.6 | 1.007 |
| quicknet | 17.3 | 16.0 | 28.0 | 123.7 | 46.5 | 1.005 |
| quicknet-large | 20.6 | 19.4 | 30.7 | 202.3 | 67.7 | 1.003 |
| birealnet | 19.6 | 17.9 | 33.6 | 116.2 | 46.1 | 1.005 |
| binary-resnet-e18 | 19.5 | 17.9 | 34.5 | 116.8 | 46.3 | 1.005 |
| xnornet | 54.2 | 50.2 | 189.2 | 495.8 | 137.9 | 1.005 |
| binary-alexnet | 29.1 | 34.0 | 33.9 | 491.2 | 137.0 | 1.012 |
| binary-densenet28 | 28.6 | 24.8 | 53.4 | 85.2 | 45.2 | 1.014 |
| binary-densenet37 | 31.7 | 26.7 | 55.4 | 113.1 | 53.0 | 1.033 |
| binary-densenet37-dilated | 32.6 | 27.9 | 55.3 | 113.1 | 53.0 | 1.010 |
| binary-densenet45 | 37.6 | 30.4 | 59.2 | 153.2 | 63.6 | 1.007 |

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

oneplus-a510:

| model (ref top-1) | bitino_all | bitino_neon | lce | tflite_fp32 | tflite_int8 |
|---|---|---|---|---|---|
| quicknet-small (15) | 4.3e-02 (15) | 3.3e-07 (15) | 1.4e-01 (15) | 4.3e-02 (15) | 2.8e-01 (621) |
| quicknet (15) | 5.7e-07 (15) | 2.2e-07 (15) | 1.2e-05 (15) | 1.1e-06 (15) | 2.5e-01 (15) |
| quicknet-large (15) | 4.2e-07 (15) | 7.5e-07 (15) | 1.6e-01 (15) | 8.9e-07 (15) | 3.5e-01 (15) |
| birealnet (15) | 2.7e-06 (15) | 2.9e-06 (15) | 1.6e-06 (15) | 7.1e-06 (15) | 2.2e-01 (15) |
| binary-resnet-e18 (10) | 3.0e-07 (10) | 2.4e-07 (10) | 2.8e-07 (10) | 8.0e-07 (10) | 3.3e-01 (86) |
| xnornet (330) | 2.4e-07 (330) | 4.8e-07 (330) | 2.6e-06 (330) | 2.6e-06 (330) | 4.5e-01 (15) |
| binary-alexnet (330) | 6.0e-08 (330) | 6.0e-08 (330) | 2.2e-07 (330) | 4.6e-06 (330) | 2.1e-01 (86) |
| binary-densenet28 (15) | 4.3e-02 (15) | 4.3e-02 (15) | 3.3e-02 (15) | 3.3e-02 (15) | 2.8e-01 (446) |
| binary-densenet37 (15) | 5.5e-02 (15) | 5.5e-02 (15) | 5.5e-02 (15) | 5.5e-02 (15) | 4.2e-01 (530) |
| binary-densenet37-dilated (15) | 8.5e-02 (15) | 8.5e-02 (15) | 9.5e-02 (15) | 9.5e-02 (15) | 6.4e-01 (814) |
| binary-densenet45 (15) | 4.1e-02 (15) | 6.0e-07 (15) | 4.1e-02 (15) | 4.1e-02 (15) | 3.6e-01 (451) |

oneplus-a715:

| model (ref top-1) | bitino_all | bitino_neon | lce | tflite_fp32 | tflite_int8 |
|---|---|---|---|---|---|
| quicknet-small (15) | 4.3e-02 (15) | 3.3e-07 (15) | 1.4e-01 (15) | 4.3e-02 (15) | 2.8e-01 (621) |
| quicknet (15) | 5.7e-07 (15) | 2.2e-07 (15) | 1.2e-05 (15) | 1.1e-06 (15) | 2.5e-01 (15) |
| quicknet-large (15) | 4.2e-07 (15) | 7.5e-07 (15) | 1.6e-01 (15) | 8.9e-07 (15) | 3.5e-01 (15) |
| birealnet (15) | 2.7e-06 (15) | 2.9e-06 (15) | 1.6e-06 (15) | 7.1e-06 (15) | 2.2e-01 (15) |
| binary-resnet-e18 (10) | 3.0e-07 (10) | 2.4e-07 (10) | 2.8e-07 (10) | 8.0e-07 (10) | 3.3e-01 (86) |
| xnornet (330) | 2.4e-07 (330) | 4.8e-07 (330) | 2.6e-06 (330) | 2.6e-06 (330) | 4.5e-01 (15) |
| binary-alexnet (330) | 6.0e-08 (330) | 6.0e-08 (330) | 2.2e-07 (330) | 4.6e-06 (330) | 2.1e-01 (86) |
| binary-densenet28 (15) | 4.3e-02 (15) | 4.3e-02 (15) | 3.3e-02 (15) | 3.3e-02 (15) | 2.8e-01 (446) |
| binary-densenet37 (15) | 5.5e-02 (15) | 5.5e-02 (15) | 5.5e-02 (15) | 5.5e-02 (15) | 4.2e-01 (530) |
| binary-densenet37-dilated (15) | 8.5e-02 (15) | 8.5e-02 (15) | 9.5e-02 (15) | 9.5e-02 (15) | 6.4e-01 (814) |
| binary-densenet45 (15) | 4.1e-02 (15) | 6.0e-07 (15) | 4.1e-02 (15) | 4.1e-02 (15) | 3.6e-01 (451) |
