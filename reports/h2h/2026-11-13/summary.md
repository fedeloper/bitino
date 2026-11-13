### CPH2551 (cph2551), cpu7, arm64

| model | bitino_all | lce | LCE / Bitino (all) |
|---|---:|---:|---:|
| quicknet-small | 4.66 | 5.33 | 1.14x |
| quicknet | 7.34 | 8.57 | 1.17x |
| quicknet-large | 13.21 | 15.21 | 1.15x |
| birealnet | 10.31 | 15.66 | 1.52x |
| binary-resnet-e18 | 10.37 | 15.75 | 1.52x |
| xnornet | 7.82 | 13.60 | 1.74x |
| binary-alexnet | 5.44 | 7.79 | 1.43x |
| binary-densenet28 | 21.17 | 29.44 | 1.39x |
| binary-densenet37 | 24.30 | 33.49 | 1.38x |
| binary-densenet37-dilated | 44.29 | 56.58 | 1.28x |
| binary-densenet45 | 32.83 | 42.75 | 1.30x |

Peak RSS (MB, median over repeats) and run-to-run spread (max/min of the per-run means) on cph2551:

| model | bitino_all RSS | lce RSS | worst spread |
|---|---:|---:|---:|
| quicknet-small | 15.7 | 25.7 | 1.005 |
| quicknet | 17.0 | 28.0 | 1.017 |
| quicknet-large | 20.5 | 30.8 | 1.003 |
| birealnet | 19.2 | 33.6 | 1.012 |
| binary-resnet-e18 | 19.2 | 34.4 | 1.039 |
| xnornet | 53.9 | 189.2 | 1.014 |
| binary-alexnet | 28.7 | 34.0 | 1.053 |
| binary-densenet28 | 28.3 | 53.3 | 1.010 |
| binary-densenet37 | 31.4 | 55.4 | 1.044 |
| binary-densenet37-dilated | 32.3 | 55.6 | 1.014 |
| binary-densenet45 | 37.4 | 59.5 | 1.024 |

### moto e13 (moto-e13), cpu7, arm64

| model | bitino_all | lce | LCE / Bitino (all) |
|---|---:|---:|---:|
| quicknet-small | 23.35 | 23.96 | 1.03x |
| quicknet | 38.87 | 40.11 | 1.03x |
| quicknet-large | 66.20 | 68.55 | 1.04x |
| birealnet | 59.28 | 63.05 | 1.06x |
| binary-resnet-e18 | 59.27 | 63.59 | 1.07x |
| xnornet | 41.90 | 52.50 | 1.25x |
| binary-alexnet | 29.36 | 30.39 | 1.03x |
| binary-densenet28 | 113.62 | 137.18 | 1.21x |
| binary-densenet37 | 129.69 | 155.74 | 1.20x |
| binary-densenet37-dilated | 239.65 | 280.02 | 1.17x |
| binary-densenet45 | 172.03 | 208.15 | 1.21x |

Peak RSS (MB, median over repeats) and run-to-run spread (max/min of the per-run means) on moto-e13:

| model | bitino_all RSS | lce RSS | worst spread |
|---|---:|---:|---:|
| quicknet-small | 15.2 | 23.8 | 1.015 |
| quicknet | 16.5 | 26.4 | 1.029 |
| quicknet-large | 20.0 | 29.1 | 1.020 |
| birealnet | 18.8 | 31.9 | 1.013 |
| binary-resnet-e18 | 18.8 | 32.8 | 1.013 |
| xnornet | 53.5 | 187.7 | 1.009 |
| binary-alexnet | 28.5 | 32.3 | 1.006 |
| binary-densenet28 | 27.6 | 51.8 | 1.009 |
| binary-densenet37 | 30.8 | 53.8 | 1.011 |
| binary-densenet37-dilated | 31.6 | 53.7 | 1.026 |
| binary-densenet45 | 36.5 | 57.7 | 1.010 |

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

cph2551:

| model (ref top-1) | bitino_all | lce |
|---|---|---|
| quicknet-small (15) | 4.3e-02 (15) | 1.4e-01 (15) =host |
| quicknet (15) | 5.7e-07 (15) | 1.2e-05 (15) =host |
| quicknet-large (15) | 4.2e-07 (15) | 1.6e-01 (15) =host |
| birealnet (15) | 2.7e-06 (15) | 1.6e-06 (15) =host |
| binary-resnet-e18 (10) | 3.0e-07 (10) | 2.8e-07 (10) =host |
| xnornet (330) | 2.4e-07 (330) | 2.6e-06 (330) =host |
| binary-alexnet (330) | 6.0e-08 (330) | 2.2e-07 (330) =host |
| binary-densenet28 (15) | 4.3e-02 (15) | 3.3e-02 (15) =host |
| binary-densenet37 (15) | 5.5e-02 (15) | 5.5e-02 (15) =host |
| binary-densenet37-dilated (15) | 8.5e-02 (15) | 9.5e-02 (15) =host |
| binary-densenet45 (15) | 4.1e-02 (15) | 4.1e-02 (15) =host |

moto-e13:

| model (ref top-1) | bitino_all | lce |
|---|---|---|
| quicknet-small (15) | 4.3e-02 (15) | 1.4e-01 (15) =host |
| quicknet (15) | 5.7e-07 (15) | 1.2e-05 (15) =host |
| quicknet-large (15) | 4.2e-07 (15) | 1.6e-01 (15) =host |
| birealnet (15) | 2.7e-06 (15) | 1.6e-06 (15) =host |
| binary-resnet-e18 (10) | 3.0e-07 (10) | 2.8e-07 (10) =host |
| xnornet (330) | 2.4e-07 (330) | 2.6e-06 (330) =host |
| binary-alexnet (330) | 6.0e-08 (330) | 2.2e-07 (330) =host |
| binary-densenet28 (15) | 4.3e-02 (15) | 3.3e-02 (15) =host |
| binary-densenet37 (15) | 5.5e-02 (15) | 5.5e-02 (15) =host |
| binary-densenet37-dilated (15) | 8.5e-02 (15) | 9.5e-02 (15) =host |
| binary-densenet45 (15) | 4.1e-02 (15) | 4.1e-02 (15) =host |
