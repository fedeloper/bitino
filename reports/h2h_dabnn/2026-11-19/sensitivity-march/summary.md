# daBNN vs Bitino: head-to-head, 1 thread, batch 1

Per configuration: median over rounds of the per-run mean of 50 timed runs (ms); `*` = only off-reference-cap rounds available. Ratio = daBNN / Bitino (> 1: Bitino faster). Method: docs/comparison-dabnn.md.

| phone | core | model | daBNN | Bitino all | daBNN / Bitino all | Bitino int8_full | daBNN / Bitino int8_full | daBNN own harness (gbench) | rounds ok (dabnn/all/int8) |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| CPH2551 (cph2551) | cpu7 (Cortex-X3) | dabnn-bireal18 | 21.26 | 10.33 | 2.06x | n/a | n/a | n/a | 3/3/0 |
| CPH2551 (cph2551) | cpu7 (Cortex-X3) | dabnn-bireal18-stem | 15.43 | 7.97 | 1.94x | n/a | n/a | n/a | 3/3/0 |
| moto e13 (moto-e13) | cpu7 (Cortex-A75) | dabnn-bireal18 | 101.33 | 59.60 | 1.70x | n/a | n/a | n/a | 3/3/0 |
| moto e13 (moto-e13) | cpu7 (Cortex-A75) | dabnn-bireal18-stem | 72.47 | 45.00 | 1.61x | n/a | n/a | n/a | 3/3/0 |

Run-to-run spread (max/min of the per-run means), median-of-medians and peak RSS:

| phone/core | model | runtime | median of medians ms | min..max of means ms | peak RSS MB | off-cpu samples |
|---|---|---|---:|---|---:|---:|
| cph2551-cpu7 | dabnn-bireal18 | bitino_all | 10.31 | 10.31..10.33 | 19.5 | 0 |
| cph2551-cpu7 | dabnn-bireal18 | dabnn | 21.24 | 21.14..21.27 | 34.4 | 0 |
| cph2551-cpu7 | dabnn-bireal18 | dabnn_v82 | 21.27 | 21.22..21.37 | 34.4 | 0 |
| cph2551-cpu7 | dabnn-bireal18-stem | bitino_all | 7.97 | 7.97..8.04 | 18.0 | 0 |
| cph2551-cpu7 | dabnn-bireal18-stem | dabnn | 15.40 | 15.41..15.46 | 28.3 | 0 |
| cph2551-cpu7 | dabnn-bireal18-stem | dabnn_v82 | 15.42 | 15.43..15.45 | 28.2 | 0 |
| moto-e13-cpu7 | dabnn-bireal18 | bitino_all | 59.68 | 59.53..59.88 | 18.8 | 0 |
| moto-e13-cpu7 | dabnn-bireal18 | dabnn | 101.57 | 99.41..102.79 | 33.8 | 0 |
| moto-e13-cpu7 | dabnn-bireal18 | dabnn_v82 | 103.00 | 102.46..102.96 | 33.8 | 0 |
| moto-e13-cpu7 | dabnn-bireal18-stem | bitino_all | 45.03 | 44.99..45.34 | 17.4 | 0 |
| moto-e13-cpu7 | dabnn-bireal18-stem | dabnn | 72.50 | 72.46..72.47 | 27.6 | 0 |
| moto-e13-cpu7 | dabnn-bireal18-stem | dabnn_v82 | 72.30 | 71.31..72.53 | 27.5 | 0 |

### Timed input (eval image 0), outputs of repeat 0

| phone/core | model | runtime | top-1 (fp64 ref top-1) | max abs vs fp64 ref | max abs vs daBNN |
|---|---|---|---|---:|---:|
| cph2551-cpu7 | dabnn-bireal18 | dabnn | 134 (134) | 0.93 | nan |
| cph2551-cpu7 | dabnn-bireal18 | bitino_all | 86 (134) | 1.59 | 1.95 |
| cph2551-cpu7 | dabnn-bireal18-stem | dabnn | 15 (15) | 7.57e-05 | nan |
| cph2551-cpu7 | dabnn-bireal18-stem | bitino_all | 15 (15) | 2.61e-06 | 7.68e-05 |
| moto-e13-cpu7 | dabnn-bireal18 | dabnn | 134 (134) | 0.93 | nan |
| moto-e13-cpu7 | dabnn-bireal18 | bitino_all | 86 (134) | 1.59 | 1.95 |
| moto-e13-cpu7 | dabnn-bireal18-stem | dabnn | 15 (15) | 7.57e-05 | nan |
| moto-e13-cpu7 | dabnn-bireal18-stem | bitino_all | 15 (15) | 2.61e-06 | 7.68e-05 |
