# daBNN vs Bitino: head-to-head, 1 thread, batch 1

Per configuration: median over rounds of the per-run mean of 50 timed runs (ms); `*` = only off-reference-cap rounds available. Ratio = daBNN / Bitino (> 1: Bitino faster). Method: docs/comparison-dabnn.md.

| phone | core | model | daBNN | Bitino all | daBNN / Bitino all | Bitino int8_full | daBNN / Bitino int8_full | daBNN own harness (gbench) | rounds ok (dabnn/all/int8) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| CPH2551 (cph2551) | cpu0 (Cortex-A510) | dabnn-bireal18 | 109.44 | 54.19 | 2.02x | 43.77 | 2.50x | 108.52 | 5/5/5 |
| CPH2551 (cph2551) | cpu0 (Cortex-A510) | dabnn-bireal18-stem | 82.90 | 40.57 | 2.04x | 33.87 | 2.45x | 84.49 | 5/5/5 |
| CPH2551 (cph2551) | cpu4 (Cortex-A715) | dabnn-bireal18 | 39.55 | 24.33 | 1.63x | 18.99 | 2.08x | 39.88 | 5/5/5 |
| CPH2551 (cph2551) | cpu4 (Cortex-A715) | dabnn-bireal18-stem | 30.08 | 18.75 | 1.60x | 15.37 | 1.96x | 30.51 | 5/5/5 |
| CPH2551 (cph2551) | cpu7 (Cortex-X3) | dabnn-bireal18 | 21.19 | 10.33 | 2.05x | 8.51 | 2.49x | 21.29 | 5/5/5 |
| CPH2551 (cph2551) | cpu7 (Cortex-X3) | dabnn-bireal18-stem | 15.34 | 7.97 | 1.92x | 6.70 | 2.29x | 15.41 | 5/5/5 |
| moto e13 (moto-e13) | cpu7 (Cortex-A75) | dabnn-bireal18 | 101.95 | 59.90 | 1.70x | 48.28 | 2.11x | 103.92 | 5/5/5 |
| moto e13 (moto-e13) | cpu7 (Cortex-A75) | dabnn-bireal18-stem | 72.16 | 45.12 | 1.60x | 34.82 | 2.07x | 72.07 | 5/5/5 |
| M2006C3LG (redmi9a) | cpu0 (Cortex-A53) | dabnn-bireal18 | 826.87 | 128.21 | 6.45x | 112.39 | 7.36x | 841.58 | 5/5/5 |
| M2006C3LG (redmi9a) | cpu0 (Cortex-A53) | dabnn-bireal18-stem | 773.73 | 102.65 | 7.54x | 80.14 | 9.66x | 786.65 | 5/5/5 |

Run-to-run spread (max/min of the per-run means), median-of-medians and peak RSS:

| phone/core | model | runtime | median of medians ms | min..max of means ms | peak RSS MB | off-cpu samples |
|---|---|---|---:|---|---:|---:|
| cph2551-cpu0 | dabnn-bireal18 | bitino_all | 54.04 | 53.86..54.52 | 19.5 | 0 |
| cph2551-cpu0 | dabnn-bireal18 | bitino_int8_full | 43.78 | 43.31..43.90 | 13.6 | 0 |
| cph2551-cpu0 | dabnn-bireal18 | dabnn | 109.36 | 107.14..110.03 | 34.4 | 0 |
| cph2551-cpu0 | dabnn-bireal18 | dabnn_gbench | - | 107.74..110.33 | 38.2 | 0 |
| cph2551-cpu0 | dabnn-bireal18-stem | bitino_all | 40.60 | 40.55..40.62 | 18.1 | 0 |
| cph2551-cpu0 | dabnn-bireal18-stem | bitino_int8_full | 33.90 | 33.78..34.32 | 12.9 | 0 |
| cph2551-cpu0 | dabnn-bireal18-stem | dabnn | 82.71 | 82.35..83.23 | 28.3 | 0 |
| cph2551-cpu0 | dabnn-bireal18-stem | dabnn_gbench | - | 83.21..86.20 | 32.3 | 0 |
| cph2551-cpu4 | dabnn-bireal18 | bitino_all | 24.31 | 24.27..24.35 | 19.5 | 0 |
| cph2551-cpu4 | dabnn-bireal18 | bitino_int8_full | 18.98 | 18.95..19.02 | 13.6 | 0 |
| cph2551-cpu4 | dabnn-bireal18 | dabnn | 39.65 | 39.36..39.86 | 34.4 | 0 |
| cph2551-cpu4 | dabnn-bireal18 | dabnn_gbench | - | 39.43..40.21 | 42.2 | 0 |
| cph2551-cpu4 | dabnn-bireal18-stem | bitino_all | 18.76 | 18.70..18.79 | 18.0 | 0 |
| cph2551-cpu4 | dabnn-bireal18-stem | bitino_int8_full | 15.37 | 15.35..15.38 | 13.0 | 0 |
| cph2551-cpu4 | dabnn-bireal18-stem | dabnn | 30.18 | 29.87..30.16 | 28.3 | 0 |
| cph2551-cpu4 | dabnn-bireal18-stem | dabnn_gbench | - | 30.23..30.61 | 36.1 | 0 |
| cph2551-cpu7 | dabnn-bireal18 | bitino_all | 10.31 | 10.27..10.38 | 19.6 | 0 |
| cph2551-cpu7 | dabnn-bireal18 | bitino_int8_full | 8.51 | 8.51..8.54 | 13.6 | 0 |
| cph2551-cpu7 | dabnn-bireal18 | dabnn | 21.17 | 20.93..21.22 | 34.4 | 0 |
| cph2551-cpu7 | dabnn-bireal18 | dabnn_gbench | - | 21.22..21.31 | 42.3 | 0 |
| cph2551-cpu7 | dabnn-bireal18-stem | bitino_all | 7.97 | 7.95..7.99 | 18.1 | 0 |
| cph2551-cpu7 | dabnn-bireal18-stem | bitino_int8_full | 6.70 | 6.69..6.71 | 12.9 | 0 |
| cph2551-cpu7 | dabnn-bireal18-stem | dabnn | 15.30 | 14.97..15.51 | 28.2 | 0 |
| cph2551-cpu7 | dabnn-bireal18-stem | dabnn_gbench | - | 15.16..15.48 | 36.1 | 0 |
| moto-e13-cpu7 | dabnn-bireal18 | bitino_all | 59.95 | 59.88..60.07 | 19.0 | 0 |
| moto-e13-cpu7 | dabnn-bireal18 | bitino_int8_full | 48.25 | 48.22..48.33 | 12.9 | 0 |
| moto-e13-cpu7 | dabnn-bireal18 | dabnn | 102.06 | 101.83..103.14 | 33.8 | 0 |
| moto-e13-cpu7 | dabnn-bireal18 | dabnn_gbench | - | 103.03..105.60 | 37.4 | 0 |
| moto-e13-cpu7 | dabnn-bireal18-stem | bitino_all | 45.10 | 44.73..45.55 | 17.5 | 0 |
| moto-e13-cpu7 | dabnn-bireal18-stem | bitino_int8_full | 34.82 | 34.64..34.94 | 11.9 | 0 |
| moto-e13-cpu7 | dabnn-bireal18-stem | dabnn | 72.05 | 70.53..72.40 | 27.5 | 0 |
| moto-e13-cpu7 | dabnn-bireal18-stem | dabnn_gbench | - | 66.54..72.52 | 31.2 | 0 |
| redmi9a-cpu0 | dabnn-bireal18 | bitino_all | 128.14 | 127.43..129.11 | 17.7 | 0 |
| redmi9a-cpu0 | dabnn-bireal18 | bitino_int8_full | 112.42 | 112.16..112.54 | 11.4 | 0 |
| redmi9a-cpu0 | dabnn-bireal18 | dabnn | 826.77 | 825.97..838.67 | 32.6 | 0 |
| redmi9a-cpu0 | dabnn-bireal18 | dabnn_gbench | - | 840.68..842.48 | 25.2 | 0 |
| redmi9a-cpu0 | dabnn-bireal18-stem | bitino_all | 102.72 | 101.86..103.44 | 16.2 | 0 |
| redmi9a-cpu0 | dabnn-bireal18-stem | bitino_int8_full | 80.15 | 80.09..80.37 | 10.5 | 0 |
| redmi9a-cpu0 | dabnn-bireal18-stem | dabnn | 773.73 | 772.65..774.18 | 26.3 | 0 |
| redmi9a-cpu0 | dabnn-bireal18-stem | dabnn_gbench | - | 786.18..790.59 | 24.4 | 0 |

### Correctness on the 128-image eval set (device, untimed)

| phone | model | top-1 acc daBNN / Bitino all / int8_full / fp64 ref | top-1 agreement with fp64 ref: daBNN / Bitino all / int8_full | Bitino all vs daBNN: top-1 agree, max abs, median per-sample max abs | int8_full vs daBNN: top-1 agree, max abs | daBNN device vs daBNN emulated |
|---|---|---|---|---|---|---|
| cph2551 | dabnn-bireal18 | 0.5859 / 0.5859 / 0.5469 / 0.5859 | 0.9609 / 0.9688 / 0.7812 | 0.9609, 3.17, 7.58e-05 | 0.7656, 6.06 | bit-identical |
| cph2551 | dabnn-bireal18-stem | 0.5703 / 0.5781 / 0.5547 / 0.5859 | 0.9766 / 0.9844 / 0.7031 | 0.9609, 3.26, 7.58e-05 | 0.7188, 6.59 | bit-identical |
| moto-e13 | dabnn-bireal18 | 0.5859 / 0.5859 / 0.5469 / 0.5859 | 0.9609 / 0.9688 / 0.7812 | 0.9609, 3.17, 7.58e-05 | 0.7656, 6.06 | bit-identical |
| moto-e13 | dabnn-bireal18-stem | 0.5703 / 0.5781 / 0.5547 / 0.5859 | 0.9766 / 0.9844 / 0.7031 | 0.9609, 3.26, 7.58e-05 | 0.7188, 6.59 | bit-identical |
| redmi9a | dabnn-bireal18 | 0.5703 / 0.5859 / 0.5469 / 0.5859 | 0.9453 / 0.9688 / 0.7812 | 0.9453, 2.65, 7.65e-05 | 0.7500, 6.06 | max abs 3.17, top-1 0.9531 |
| redmi9a | dabnn-bireal18-stem | 0.5859 / 0.5859 / 0.5547 / 0.5859 | 0.9844 / 0.9922 / 0.7031 | 0.9766, 3.26, 7.63e-05 | 0.7188, 6.86 | max abs 3.25, top-1 0.9844 |

### Timed input (eval image 0), outputs of repeat 0

| phone/core | model | runtime | top-1 (fp64 ref top-1) | max abs vs fp64 ref | max abs vs daBNN |
|---|---|---|---|---:|---:|
| cph2551-cpu0 | dabnn-bireal18 | dabnn | 134 (134) | 0.93 | nan |
| cph2551-cpu0 | dabnn-bireal18 | bitino_all | 86 (134) | 1.59 | 1.95 |
| cph2551-cpu0 | dabnn-bireal18 | bitino_int8_full | 15 (134) | 4.01 | 4.08 |
| cph2551-cpu0 | dabnn-bireal18-stem | dabnn | 15 (15) | 7.57e-05 | nan |
| cph2551-cpu0 | dabnn-bireal18-stem | bitino_all | 15 (15) | 2.61e-06 | 7.68e-05 |
| cph2551-cpu0 | dabnn-bireal18-stem | bitino_int8_full | 86 (15) | 4.88 | 4.88 |
| cph2551-cpu4 | dabnn-bireal18 | dabnn | 134 (134) | 0.93 | nan |
| cph2551-cpu4 | dabnn-bireal18 | bitino_all | 86 (134) | 1.59 | 1.95 |
| cph2551-cpu4 | dabnn-bireal18 | bitino_int8_full | 15 (134) | 4.01 | 4.08 |
| cph2551-cpu4 | dabnn-bireal18-stem | dabnn | 15 (15) | 7.57e-05 | nan |
| cph2551-cpu4 | dabnn-bireal18-stem | bitino_all | 15 (15) | 2.61e-06 | 7.68e-05 |
| cph2551-cpu4 | dabnn-bireal18-stem | bitino_int8_full | 86 (15) | 4.88 | 4.88 |
| cph2551-cpu7 | dabnn-bireal18 | dabnn | 134 (134) | 0.93 | nan |
| cph2551-cpu7 | dabnn-bireal18 | bitino_all | 86 (134) | 1.59 | 1.95 |
| cph2551-cpu7 | dabnn-bireal18 | bitino_int8_full | 15 (134) | 4.01 | 4.08 |
| cph2551-cpu7 | dabnn-bireal18-stem | dabnn | 15 (15) | 7.57e-05 | nan |
| cph2551-cpu7 | dabnn-bireal18-stem | bitino_all | 15 (15) | 2.61e-06 | 7.68e-05 |
| cph2551-cpu7 | dabnn-bireal18-stem | bitino_int8_full | 86 (15) | 4.88 | 4.88 |
| moto-e13-cpu7 | dabnn-bireal18 | dabnn | 134 (134) | 0.93 | nan |
| moto-e13-cpu7 | dabnn-bireal18 | bitino_all | 86 (134) | 1.59 | 1.95 |
| moto-e13-cpu7 | dabnn-bireal18 | bitino_int8_full | 15 (134) | 4.01 | 4.08 |
| moto-e13-cpu7 | dabnn-bireal18-stem | dabnn | 15 (15) | 7.57e-05 | nan |
| moto-e13-cpu7 | dabnn-bireal18-stem | bitino_all | 15 (15) | 2.61e-06 | 7.68e-05 |
| moto-e13-cpu7 | dabnn-bireal18-stem | bitino_int8_full | 86 (15) | 4.88 | 4.88 |
| redmi9a-cpu0 | dabnn-bireal18 | dabnn | 134 (134) | 2.07 | nan |
| redmi9a-cpu0 | dabnn-bireal18 | bitino_all | 86 (134) | 1.59 | 1.58 |
| redmi9a-cpu0 | dabnn-bireal18 | bitino_int8_full | 15 (134) | 4.01 | 3.98 |
| redmi9a-cpu0 | dabnn-bireal18-stem | dabnn | 15 (15) | 7.1e-05 | nan |
| redmi9a-cpu0 | dabnn-bireal18-stem | bitino_all | 15 (15) | 3.57e-06 | 7.15e-05 |
| redmi9a-cpu0 | dabnn-bireal18-stem | bitino_int8_full | 86 (15) | 4.88 | 4.88 |
