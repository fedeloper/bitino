#!/usr/bin/env bash
# Runs a command in a systemd user scope with a hard memory cap and no swap, so a runaway job (many qemu
# instances, large evaluations) is killed instead of exhausting the host. The host has crashed from memory use
# before: keep the sum of all caps <= ~40 GB and check `free -g` first.
#   scripts/capped.sh 16G nice -n 10 scripts/emulated_reference_outputs.sh
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: $0 <MemoryMax, e.g. 16G> <command...>" >&2; exit 2; }
cap="$1"; shift
exec systemd-run --user --scope --quiet -p MemoryMax="$cap" -p MemorySwapMax=0 "$@"
