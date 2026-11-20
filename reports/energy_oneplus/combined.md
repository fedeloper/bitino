# Measured energy per inference, OnePlus Open (combined, partial)

Plugged-in ballast method (tools/energy_oneplus.py). Rows from `2026-11-19/` except the four QuickNet-Small single-core rows,
taken from `2026-11-19_rerun/`: in the first run they ran before the vendor caps settled (X3 2.23 -> 1.59 GHz, A510 2.02 -> 1.46 GHz
during the A510 row, whose two repeats then disagreed 650 vs 222 mW). Every row below ran with scaling_max_freq
X3 1.594 / A715 1.786 / A510 1.459 GHz (screen off, thermal status 2-3). Invalid = ballast-only discharge below 50 mA
(USB input not provably saturated): all 4-thread rows and ReActNet-18 at 2 threads.

| config | cores | median ms | extra power (mW) | mJ / inference | valid | source |
|---|---|---:|---:|---:|---:|---|
| larq-zoo-quicknet-small__x3 | 7 | 4.98 | 1172 | 5.95 | 2/2 | 2026-11-19_rerun |
| larq-zoo-quicknet-small__a715 | 3 | 10.81 | 367 | 3.99 | 2/2 | 2026-11-19_rerun |
| larq-zoo-quicknet-small__a510 | 0 | 32.39 | 231 | 7.68 | 2/2 | 2026-11-19_rerun |
| larq-zoo-quicknet-small__t2 | 7,3 | 4.01 | 1497 | 6.10 | 2/2 | 2026-11-19 |
| larq-zoo-quicknet-small__t4 | 7,3,4,5 | 2.60 | – | – | 0/2 | 2026-11-19 |
| larq-zoo-quicknet-small__x3_int8_full | 7 | 5.01 | 1101 | 5.69 | 2/2 | 2026-11-19_rerun |
| larq-zoo-quicknet__x3 | 7 | 7.89 | 1204 | 9.68 | 2/2 | 2026-11-19 |
| larq-zoo-quicknet__a715 | 3 | 17.41 | 377 | 6.60 | 2/2 | 2026-11-19 |
| larq-zoo-quicknet__a510 | 0 | 51.11 | 224 | 11.71 | 2/2 | 2026-11-19 |
| larq-zoo-quicknet__t2 | 7,3 | 6.09 | 1550 | 9.63 | 2/2 | 2026-11-19 |
| larq-zoo-quicknet__t4 | 7,3,4,5 | 4.09 | – | – | 0/2 | 2026-11-19 |
| larq-zoo-quicknet__x3_int8_full | 7 | 8.20 | 1078 | 8.99 | 2/2 | 2026-11-19 |
| ab-bnn-imagenet-reactnet18__x3 | 7 | 11.83 | 1285 | 15.46 | 2/2 | 2026-11-19 |
| ab-bnn-imagenet-reactnet18__a715 | 3 | 24.99 | 419 | 10.57 | 2/2 | 2026-11-19 |
| ab-bnn-imagenet-reactnet18__a510 | 0 | 77.00 | 247 | 19.42 | 2/2 | 2026-11-19 |
| ab-bnn-imagenet-reactnet18__t2 | 7,3 | 8.61 | – | – | 0/2 | 2026-11-19 |
| ab-bnn-imagenet-reactnet18__t4 | 7,3,4,5 | 5.70 | – | – | 0/2 | 2026-11-19 |
| ab-bnn-imagenet-reactnet18__x3_int8_full | 7 | 8.33 | 1119 | 9.51 | 2/2 | 2026-11-19 |
