#!/usr/bin/env bash
# Measures Bitino on your Android phone and writes one JSON file for the community speed table.
#
#   scripts/bench_my_phone.sh                  # 4 model files of the latest release (about 40 MB, 5-10 minutes)
#   scripts/bench_my_phone.sh --quick          # 1 model file (ReActNet-18, 4 MB, 1-2 minutes)
#   scripts/bench_my_phone.sh --all            # all 22 redistributable model files of the release (about 150 MB)
#   scripts/bench_my_phone.sh --serial ID      # pick the phone when several are connected (the ID is not recorded)
#   scripts/bench_my_phone.sh --binary PATH --local DIR   # your own bitino_bench build and model files
#
# Needs: adb (Android platform-tools) and a phone with USB debugging enabled; bash, awk, curl or wget, sha256sum or
# shasum on the computer. No root, no app. Guide and privacy note: docs/benchmark-your-phone.md.
#
# What it does:
#   1. reads the phone model, SoC, Android version, ABI, CPU topology (cpufreq policies, maximum clocks), and refuses
#      emulators;
#   2. downloads bitino_bench for the phone's ABI and the model files from the GitHub release (checked against the
#      release's SHA256SUMS / MODELS-SHA256SUMS; cached in ~/.cache/bitino/bench), or uses --binary / --local;
#   3. output check: runs every file once on a fixed synthetic input at its exact stage and compares the SHA-256 of
#      the outputs with the reference computed by the maintainers (when the table below has one);
#   4. timing: 1 thread pinned to the fastest core, and 4 threads pinned to the 4 fastest cores (fastest first);
#      --repeats rounds (default 3), round-robin over the configurations, each of --warmup + --iterations runs
#      (default 10 + 50). Before each run it reads the cpufreq ceilings and Android's thermal status and waits (at
#      most 120 s) while the phone is throttling; the before/after state is recorded with every run;
#   5. writes bitino-bench-<phone model>-<date>.json (everything, raw reports included) and a .md summary to paste
#      into a "Speed on my phone" issue.
#
# What it never does: change governors, clocks or any setting; reboot; install anything; use the screen or the
# camera; read or write on the phone outside /data/local/tmp/bitino-community (created at the start, deleted at the
# end, also on Ctrl-C); record the adb serial number, IMEI, accounts, phone number or the build fingerprint.
set -euo pipefail
export LC_ALL=C   # decimal points in printf/awk/sort whatever the user locale

TOOL_VERSION=1
SCHEMA="bitino.community_bench.v1"
REPO="${BITINO_REPO:-fedeloper/bitino}"
TAG="${BITINO_RELEASE:-latest}"
DEV_DIR="/data/local/tmp/bitino-community"     # the only path on the phone this script touches
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/bitino/bench"
OUT_DIR="."
WARMUP=10
ITERATIONS=50
REPEATS=3
THROTTLE_MAX=120
BINARY=""
LOCAL_DIR=""
CHECK=1
SET="default"

# Release model files: <asset name without .bitino> <expected SHA-256 of the outputs on the check input, or ->.
# The check runs the file at its exact stage (float files: `neon`, bit-exact with the reference executor on every
# platform; int8_full files: `int8_full`, integer arithmetic) with 4 threads (threading never changes outputs).
MODEL_TABLE='
ab-bnn-imagenet-reactnet18 c7ad0130669fa53a54c1609e025721c6b2a5766143b7f67843523d5232535af3
ab-bnn-imagenet-reactnet18_int8_full b044a5cd9c33fa973260a44cc554cdf5ef8549ebfae8e37bae13689d3d140ff1
omz-resnet50-binary-0001 f24d015bda83595524640422b01fe15b8529b3ee79123713357c5ff4f04630f1
omz-resnet50-binary-0001_int8_full 2810a613e2aeb1f58c6c6d7615114dc1b9e714c0c946eb9eeff9f67b9f356c05
ab-bnn-imagenet-reactnet34 4bb14747110d6b3c633e1069c24b735f69e79ff3d98bc60bd7c1c774a92446f4
ab-bnn-imagenet-reactnet34_int8_full 9a968040137ccdb000bd02847db4e0365a417ddc05f0889aa509ed19918bcf20
ab-bnn-imagenet-reactneta ee31e41bb67b09ca2ef33a0133a2f2177afa55549374177888c0ea4c789206e0
ab-bnn-imagenet-reactneta_int8_full e576f4da810839b033b80ca419b76bbcf2efe5b9708395ab31285c3b1334c351
omz-resnet18-xnor-binary-onnx-0001 4534c2915829e951f4a8fe47a4bba5defaa9eae3c8957f117850e0fb116828a5
omz-resnet18-xnor-binary-onnx-0001_int8_full 94f2fb453012f13a0e0f67912260ca107286a7785dbc1c8832d31dbc12582e16
omz-resnet18-xnor-binary-onnx-0001-fp16 db1fa91a2e2ea336c05a3e6b4f364e451cc261f77efacb0ba28cad3f0ad72bc0
omz-resnet18-xnor-binary-onnx-0001-fp16_int8_full d671fd2434cb338aad14d75f0f350f1c5f7ac4d569dfe39dce94bd34d01020bd
omz-resnet50-binary-0001-fp16 3f5f1d5745c052b228a0c08c6a94dddb0d01d2a83e55c0f3dc617d66dc7c5356
omz-resnet50-binary-0001-fp16_int8_full 966a5de6f440fb2c9b1c806fae597043da4f685210790f254d8fc93aa8aac299
ab-bnn-cifar10-reactnet18 20c167569fb60e0af21dfde3e0ff00d191f225dc6dace904829277247f913053
ab-bnn-cifar10-reactnet18_int8_full 43b2bbf65d273026ccfb8884f051d45dd0989de46a674ba66a3a31779d91f1cd
ab-bnn-cifar10-reactneta a2eafa0efc55141fc3ae3bd43df24c7da2c7e8ca4e734149e38c5f8ac010ab9c
ab-bnn-cifar10-reactneta_int8_full af7eb8ec66db08aa1475dc806287ad55ca80ee6148bc652dddbd9c91ab3c890b
ab-bnn-cifar100-reactnet18 e028ff78dde6cb5117aaa45237641d8e18da71930590341ea53ff97d9bb2845a
ab-bnn-cifar100-reactnet18_int8_full 6e85764739fdae0fbca338f4ff6aee9c1a69bbe1354bbddbb279d1f4807fcb99
ab-bnn-cifar100-reactneta 906bffc5c2e2bb01f3382cc273a5a580cc3b357d80b43faaf32e54c0ddba60c5
ab-bnn-cifar100-reactneta_int8_full c23582619f3088be9ae68c7af4ca5b5eacd79c562f2f0c2a7c622a876dc547c3
'
DEFAULT_SET="ab-bnn-imagenet-reactnet18 ab-bnn-imagenet-reactnet18_int8_full omz-resnet50-binary-0001 omz-resnet50-binary-0001_int8_full"
QUICK_SET="ab-bnn-imagenet-reactnet18"

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; cat <<'EOF'

Options:
  --serial ID        adb serial of the phone (or set ANDROID_SERIAL); needed only with several phones connected
  --quick | --all    1 model file / all release model files (default: 4 headline files)
  --models A,B       asset names (see MODEL_TABLE in this script), e.g. ab-bnn-imagenet-reactnet18_int8_full
  --binary PATH      use this bitino_bench build instead of the release binary
  --local DIR        use local model files: DIR/<model-id>/model.bitino, DIR/<model-id>/model_int8_full.bitino
                     or DIR/<asset>.bitino, instead of downloading them
  --release TAG      release to download from (default: latest)
  --repeats N --iterations N --warmup N   (defaults 3, 50, 10)
  --no-check         skip the output check
  --out DIR          where to write the JSON and the summary (default: current directory)
EOF
}

die() { echo "error: $*" >&2; exit 1; }
say() { echo "== $*" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --serial) export ANDROID_SERIAL="$2"; shift 2 ;;
    --quick) SET="quick"; shift ;;
    --all) SET="all"; shift ;;
    --models) SET="$2"; shift 2 ;;
    --binary) BINARY="$2"; shift 2 ;;
    --local) LOCAL_DIR="$2"; shift 2 ;;
    --release) TAG="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --no-check) CHECK=0; shift ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option $1" ;;
  esac
done
for v in "$REPEATS" "$ITERATIONS" "$WARMUP"; do
  case "$v" in ''|*[!0-9]*) die "--repeats/--iterations/--warmup take a number" ;; esac
done
[ "$REPEATS" -ge 1 ] && [ "$ITERATIONS" -ge 1 ] || die "--repeats and --iterations must be at least 1"

# ------------------------------------------------------------------------------------------------ host tools
ADB="${ADB:-}"
if [ -z "$ADB" ]; then
  if command -v adb >/dev/null 2>&1; then ADB=adb
  elif [ -x "$HOME/Android/Sdk/platform-tools/adb" ]; then ADB="$HOME/Android/Sdk/platform-tools/adb"
  elif [ -x "$HOME/Library/Android/sdk/platform-tools/adb" ]; then ADB="$HOME/Library/Android/sdk/platform-tools/adb"
  else die "adb not found: install Android platform-tools (https://developer.android.com/tools/releases/platform-tools) or set ADB=/path/to/adb"
  fi
fi
command -v awk >/dev/null 2>&1 || die "awk is required"
if command -v sha256sum >/dev/null 2>&1; then sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else die "sha256sum or shasum is required"
fi
fetch() {   # <url> <dest>
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 2 -sS -o "$2.part" "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -O "$2.part" "$1"
  else die "curl or wget is required to download the release files (or use --binary and --local)"
  fi && mv "$2.part" "$2"
}
fsize() { wc -c < "$1" | tr -d ' '; }

TMP="$(mktemp -d 2>/dev/null || mktemp -d -t bitino-bench)"
DEVICE_READY=0
cleanup() {
  if [ "$DEVICE_READY" = 1 ]; then
    "$ADB" shell rm -rf "$DEV_DIR" >/dev/null 2>&1 || true   # our own directory only
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT
trap 'echo "interrupted; cleaning up" >&2; exit 130' INT TERM

# JSON string literal of $1
js() {
  local s="$1"
  s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\t'/ }; s=${s//$'\r'/}; s=${s//$'\n'/ }
  printf '"%s"' "$s"
}
jnum() { case "$1" in ''|*[!0-9.eE+-]*) printf 'null' ;; *) printf '%s' "$1" ;; esac; }   # number or null
jbool() { if [ "$1" = 1 ]; then printf 'true'; else printf 'false'; fi; }

# ------------------------------------------------------------------------------------------------ the phone
n_dev=$("$ADB" devices | awk 'NR > 1 && $2 == "device"' | wc -l | tr -d ' ')
if [ -z "${ANDROID_SERIAL:-}" ]; then
  [ "$n_dev" -ge 1 ] || { "$ADB" devices >&2; die "no phone found: connect it by USB, enable USB debugging and accept the prompt on the phone"; }
  [ "$n_dev" -eq 1 ] || die "$n_dev phones connected: pass --serial <id> (first column of 'adb devices')"
fi
[ "$("$ADB" get-state 2>/dev/null | tr -d '\r')" = "device" ] || die "the phone is not ready (adb get-state); unlock it and accept the USB debugging prompt"

sh1() { "$ADB" shell "$@" 2>/dev/null | tr -d '\r'; }
gp() { sh1 getprop "$1"; }

P_MODEL=$(gp ro.product.model); P_MANUF=$(gp ro.product.manufacturer); P_BRAND=$(gp ro.product.brand)
P_DEVICE=$(gp ro.product.device); P_SOC_M=$(gp ro.soc.manufacturer); P_SOC=$(gp ro.soc.model)
[ -n "$P_SOC" ] || P_SOC=$(gp ro.hardware.chipname)   # Android < 12 has no ro.soc.*; Samsung, some others
[ -n "$P_SOC" ] || P_SOC=$(gp ro.chipname)
P_BOARD=$(gp ro.board.platform); P_HW=$(gp ro.hardware); P_REL=$(gp ro.build.version.release)
P_API=$(gp ro.build.version.sdk); P_PATCH=$(gp ro.build.version.security_patch)
P_ABI=$(gp ro.product.cpu.abi); P_ABIS=$(gp ro.product.cpu.abilist)
Q1=$(gp ro.kernel.qemu); Q2=$(gp ro.boot.qemu); P_CHAR=$(gp ro.build.characteristics)
P_KERNEL=$(sh1 uname -r)

EMU=0
case "$P_HW" in goldfish*|ranchu*|vbox86*|cutf_cvm*) EMU=1 ;; esac
case "$P_MODEL" in *sdk_gphone*|*Emulator*|*"Android SDK built for"*|*"SDK built for"*) EMU=1 ;; esac
if [ "$Q1" = 1 ] || [ "$Q2" = 1 ]; then EMU=1; fi
case "$P_CHAR" in *emulator*) EMU=1 ;; esac
[ "$EMU" = 0 ] || die "this looks like an emulator ($P_MODEL, hardware $P_HW): the community table takes real phones only"
[ -n "$P_MODEL" ] || die "could not read ro.product.model"

case "$P_ABI" in
  arm64-v8a) BIN_ASSET="bitino_bench-android-arm64" ;;
  armeabi-v7a|armeabi) BIN_ASSET="bitino_bench-android-armv7" ;;
  *) die "unsupported ABI $P_ABI (Bitino's phone builds are arm64-v8a and armeabi-v7a)" ;;
esac
say "phone: $P_MANUF $P_MODEL ($P_DEVICE), SoC ${P_SOC:-?} (${P_BOARD:-?}), Android $P_REL (API $P_API), $P_ABI"

# CPU topology: one line per cpufreq policy "<policy> <cpuinfo_max_khz> <scaling_max_khz> <governor> <cpus...>"
TOPO=$(sh1 'for p in /sys/devices/system/cpu/cpufreq/policy*; do [ -d "$p" ] || continue; echo "${p##*/policy} $(cat $p/cpuinfo_max_freq 2>/dev/null || echo 0) $(cat $p/scaling_max_freq 2>/dev/null || echo 0) $(cat $p/scaling_governor 2>/dev/null || echo ?) $(cat $p/related_cpus 2>/dev/null)"; done')
PARTS=$(sh1 'grep -E "^(processor|CPU implementer|CPU part)" /proc/cpuinfo' | awk -F: '
  /^processor/ {c = $2 + 0} /CPU implementer/ {gsub(/ /, "", $2); imp[c] = $2} /CPU part/ {gsub(/ /, "", $2); part[c] = $2}
  END {for (k in part) print k, imp[k], part[k]}' | sort -n)
ONLINE=$(sh1 cat /sys/devices/system/cpu/online)
N_CPUS=$(sh1 'ls -d /sys/devices/system/cpu/cpu[0-9]*' | wc -l | tr -d ' ')
[ -n "$TOPO" ] || die "no cpufreq policies readable: cannot find the fastest core (real phones expose them; emulators do not)"
# CPU order: fastest policy first (highest cpuinfo_max_freq); its highest-numbered CPU first, then the other CPUs
ORDER=$(echo "$TOPO" | awk '{for (i = 5; i <= NF; i++) print $2, $i}' | sort -k1,1nr -k2,2n | awk '{print $2}')
FAST_POLICY_MAX=$(echo "$TOPO" | sort -k2,2nr | head -1 | awk '{print $2}')
FAST_CPU=$(echo "$TOPO" | sort -k2,2nr | head -1 | awk '{print $NF}')
FAST_POLICY=$(echo "$TOPO" | sort -k2,2nr | head -1 | awk '{print $1}')
T4_AFF=$( (echo "$FAST_CPU"; echo "$ORDER" | grep -vx "$FAST_CPU") | head -4 | paste -sd, -)
T4_THREADS=$(echo "$T4_AFF" | tr ',' '\n' | wc -l | tr -d ' ')   # 4, or every CPU of a phone with fewer
say "fastest core: cpu$FAST_CPU (policy $FAST_POLICY, max $(awk -v k="$FAST_POLICY_MAX" 'BEGIN{printf "%.2f", k / 1e6}') GHz); 4-thread cores: $T4_AFF"

# Clock ceilings + thermal status, parsed into: <json>|<capped 0/1>|<status or empty>|<new reference>|<throttled 0/1>
REF=""
CAPS_RAW_CMD='for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "P ${p##*/policy} $(cat $p/scaling_max_freq 2>/dev/null) $(cat $p/cpuinfo_max_freq 2>/dev/null)"; done; dumpsys thermalservice 2>/dev/null | grep -m1 "Thermal Status"'
read_caps() {
  local raw
  raw=$(sh1 "$CAPS_RAW_CMD")
  echo "$raw" | awk -v ref="$REF" '
    BEGIN { n = split(ref, a, " "); for (i = 1; i <= n; i++) { split(a[i], kv, ":"); R[kv[1]] = kv[2] } }
    $1 == "P" && NF == 4 && $3 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ { np++; pol[np] = $2; cur[$2] = $3; mx[$2] = $4 }
    /Thermal Status/ { st = $NF }
    END {
      capped = 0; thr = 0; js = ""; nr = ""
      for (i = 1; i <= np; i++) {
        p = pol[i]; if (cur[p] < mx[p]) capped = 1
        if (R[p] == "" || cur[p] > R[p]) R[p] = cur[p]
        if (cur[p] < R[p]) thr = 1
        js = js (i > 1 ? ", " : "") "\"" p "\": [" cur[p] ", " mx[p] "]"
      }
      for (p in R) nr = nr (nr == "" ? "" : " ") p ":" R[p]
      if (st != "" && st + 0 > 0) thr = 1
      printf "{\"policies\": {%s}, \"capped\": %s, \"thermal_status\": %s}|%d|%s|%s|%d\n", js, (capped ? "true" : "false"), (st == "" ? "null" : st), capped, st, nr, thr
    }'
}
CAPS_J=""; CAPS_CAPPED=0; CAPS_STATUS=""; CAPS_THR=0
caps() { IFS='|' read -r CAPS_J CAPS_CAPPED CAPS_STATUS REF CAPS_THR <<EOF
$(read_caps)
EOF
}
policy_cap() {   # <caps json> <policy> -> scaling_max_khz
  echo "$1" | awk -v p="\"$2\"" '{ i = index($0, p ": ["); if (i) { s = substr($0, i + length(p) + 3); split(s, v, ","); print v[1] + 0 } }'
}
BATTERY() { sh1 'dumpsys battery' | awk -F': ' '/^ *(AC powered|USB powered|Wireless powered|level|temperature|status):/ {gsub(/^ +/, "", $1); print $1 "=" $2}'; }
WAKE() { sh1 'dumpsys power' | awk -F= '/mWakefulness=/ {print $2; exit}'; }
battery_json() {
  local b; b=$(BATTERY)
  local lv tp ac usb wl
  lv=$(echo "$b" | awk -F= '$1 == "level" {print $2}'); tp=$(echo "$b" | awk -F= '$1 == "temperature" {print $2}')
  ac=$(echo "$b" | awk -F= '$1 == "AC powered" {print $2}'); usb=$(echo "$b" | awk -F= '$1 == "USB powered" {print $2}')
  wl=$(echo "$b" | awk -F= '$1 == "Wireless powered" {print $2}')
  printf '{"level_pct": %s, "temperature_c": %s, "ac_powered": %s, "usb_powered": %s, "wireless_powered": %s, "wakefulness": %s}' \
    "$(jnum "$lv")" "$( [ -n "$tp" ] && awk -v t="$tp" 'BEGIN{printf "%.1f", t / 10}' || printf null)" \
    "$( [ "$ac" = true ] && echo true || echo false)" "$( [ "$usb" = true ] && echo true || echo false)" \
    "$( [ "$wl" = true ] && echo true || echo false)" "$(js "$(WAKE)")"
}

# ------------------------------------------------------------------------------------------------ files
case "$SET" in
  default) ASSETS="$DEFAULT_SET" ;;
  quick) ASSETS="$QUICK_SET" ;;
  all) ASSETS=$(echo "$MODEL_TABLE" | awk 'NF {print $1}') ;;
  *) ASSETS=$(echo "$SET" | tr ',' ' ') ;;
esac
for a in $ASSETS; do
  echo "$MODEL_TABLE" | awk -v a="$a" '$1 == a {f = 1} END {exit !f}' || die "unknown model file $a (see MODEL_TABLE)"
done
model_id() { echo "${1%_int8_full}"; }
model_file() { case "$1" in *_int8_full) echo model_int8_full.bitino ;; *) echo model.bitino ;; esac; }
timed_stage() { case "$1" in *_int8_full) echo int8_full ;; *) echo all ;; esac; }
check_stage() { case "$1" in *_int8_full) echo int8_full ;; *) echo neon ;; esac; }
input_bytes() { case "$1" in *cifar*) echo 12288 ;; *) echo 602112 ;; esac; }   # f32 NHWC 1x32x32x3 / 1x224x224x3
expected_out() { echo "$MODEL_TABLE" | awk -v a="$1" '$1 == a && $2 != "-" {print $2}'; }

RELEASE_TAG=""   # set when something is downloaded
BASE_URL=""
release_init() {
  [ -z "$BASE_URL" ] || return 0
  RELEASE_TAG="$TAG"
  if [ "$TAG" = latest ] && command -v curl >/dev/null 2>&1; then
    local u; u=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest" 2>/dev/null || true)
    case "$u" in */releases/tag/*) RELEASE_TAG="${u##*/}" ;; esac
  fi
  if [ "$RELEASE_TAG" = latest ]; then BASE_URL="https://github.com/$REPO/releases/latest/download"
  else BASE_URL="https://github.com/$REPO/releases/download/$RELEASE_TAG"; fi
  mkdir -p "$CACHE/$RELEASE_TAG"
  say "release $RELEASE_TAG of $REPO (cache $CACHE/$RELEASE_TAG)"
  fetch "$BASE_URL/SHA256SUMS" "$TMP/SHA256SUMS" || die "could not download SHA256SUMS from $BASE_URL"
  fetch "$BASE_URL/MODELS-SHA256SUMS" "$TMP/MODELS-SHA256SUMS" || die "could not download MODELS-SHA256SUMS from $BASE_URL"
}
release_file() {   # <asset> <sums file> -> prints the local path
  local name="$1" sums="$2" want path
  want=$(awk -v n="$name" '$2 == n || $2 == "*" n {print $1}' "$sums")
  [ -n "$want" ] || return 2
  path="$CACHE/$RELEASE_TAG/$name"
  if [ ! -f "$path" ] || [ "$(sha256 "$path")" != "$want" ]; then
    say "downloading $name"
    fetch "$BASE_URL/$name" "$path" || die "download failed: $BASE_URL/$name"
    [ "$(sha256 "$path")" = "$want" ] || { rm -f "$path"; die "checksum mismatch for $name"; }
  fi
  echo "$path"
}

if [ -n "$BINARY" ]; then
  [ -f "$BINARY" ] || die "--binary $BINARY: no such file"
  BIN_PATH="$BINARY"; BIN_SOURCE="local"
else
  release_init
  BIN_PATH=$(release_file "$BIN_ASSET" "$TMP/SHA256SUMS") || {
    [ "$BIN_ASSET" = bitino_bench-android-armv7 ] && die "release $RELEASE_TAG has no 32-bit (armeabi-v7a) binary: build one with
  ANDROID_ABI=armeabi-v7a scripts/build_android.sh build/android-armv7   (see docs/android-deployment.md)
and pass --binary build/android-armv7/bitino_bench"
    die "$BIN_ASSET is not in release $RELEASE_TAG"; }
  BIN_SOURCE="release"
fi
BIN_SHA=$(sha256 "$BIN_PATH")

MODEL_PATHS=""   # lines "<asset> <path> <source>"
for a in $ASSETS; do
  mid=$(model_id "$a"); mf=$(model_file "$a")
  if [ -n "$LOCAL_DIR" ]; then
    p=""
    for c in "$LOCAL_DIR/$mid/$mf" "$LOCAL_DIR/$a.bitino"; do [ -f "$c" ] && { p="$c"; break; }; done
    [ -n "$p" ] || die "--local: neither $LOCAL_DIR/$mid/$mf nor $LOCAL_DIR/$a.bitino exists"
    MODEL_PATHS="$MODEL_PATHS$a $p local
"
  else
    release_init
    p=$(release_file "$a.bitino" "$TMP/MODELS-SHA256SUMS") || die "$a.bitino is not in release $RELEASE_TAG"
    MODEL_PATHS="$MODEL_PATHS$a $p release
"
  fi
done
mpath() { echo "$MODEL_PATHS" | awk -v a="$1" '$1 == a {print $2}'; }

# ------------------------------------------------------------------------------------------------ push
need_kb=$(( ($(fsize "$BIN_PATH") + 2 * 602112) / 1024 ))
for a in $ASSETS; do need_kb=$(( need_kb + $(fsize "$(mpath "$a")") / 1024 )); done
free_kb=$(sh1 'df -k /data/local/tmp' | awk 'NR == 2 {print $4}')
case "$free_kb" in ''|*[!0-9]*) ;; *) [ "$free_kb" -gt $(( need_kb + 51200 )) ] || die "not enough free space in /data/local/tmp (${free_kb} KB free, need ${need_kb} KB + 50 MB)" ;; esac

DEVICE_READY=1
sh1 "rm -rf $DEV_DIR && mkdir -p $DEV_DIR/m $DEV_DIR/out" >/dev/null
say "pushing the binary and $(echo $ASSETS | wc -w | tr -d ' ') model file(s) to $DEV_DIR"
"$ADB" push "$BIN_PATH" "$DEV_DIR/bitino_bench" >/dev/null 2>&1 || die "adb push failed"
sh1 "chmod 755 $DEV_DIR/bitino_bench" >/dev/null
for a in $ASSETS; do "$ADB" push "$(mpath "$a")" "$DEV_DIR/m/$a.bitino" >/dev/null 2>&1 || die "adb push $a failed"; done

dev_run() {   # <args...>: runs bitino_bench in DEV_DIR; prints its stdout+stderr, returns its exit status
  local out rc
  out=$("$ADB" shell "cd $DEV_DIR && ./bitino_bench $* 2>&1; echo BITINO_RC=\$?" 2>&1 | tr -d '\r')
  rc=$(echo "$out" | awk -F= '/^BITINO_RC=/ {print $2}' | tail -1)
  echo "$out" | grep -v '^BITINO_RC='
  [ "${rc:-1}" = 0 ]
}
ver_line=$(dev_run --model "m/$(echo $ASSETS | awk '{print $1}').bitino" --iterations 1 --warmup 0 --json out/probe.json) ||
  die "bitino_bench does not run on this phone: $ver_line"
"$ADB" pull "$DEV_DIR/out/probe.json" "$TMP/probe.json" >/dev/null 2>&1 || die "could not pull the probe report"
RUNTIME_VERSION=$(grep -m1 -E '"[a-z]+_version"' "$TMP/probe.json" | sed -E 's/.*: *"([^"]*)".*/\1/')
CPU_FEATURES=$(grep -m1 '"cpu_features"' "$TMP/probe.json" | sed -E 's/.*: *"([^"]*)".*/\1/')
say "runtime $RUNTIME_VERSION; CPU features: $CPU_FEATURES"

# ------------------------------------------------------------------------------------------------ output check
make_input() {   # <bytes> <path>: f32 little endian, the 7 values 0.5 -1.25 2 -0.75 1.5 -0.25 1 repeated
  printf '\000\000\000\077\000\000\240\277\000\000\000\100\000\000\100\277\000\000\300\077\000\000\200\276\000\000\200\077' > "$2.p"
  while [ "$(fsize "$2.p")" -lt "$1" ]; do cat "$2.p" "$2.p" > "$2.q"; mv "$2.q" "$2.p"; done
  head -c "$1" "$2.p" > "$2"; rm -f "$2.p"
}
CHECKS_JSON=""
CHECK_RESULT=""   # lines "<asset> <match|MISMATCH|unchecked|failed>"
if [ "$CHECK" = 1 ]; then
  for a in $ASSETS; do
    nb=$(input_bytes "$a"); st=$(check_stage "$a")
    if [ ! -f "$TMP/in_$nb.bin" ]; then
      make_input "$nb" "$TMP/in_$nb.bin"
      "$ADB" push "$TMP/in_$nb.bin" "$DEV_DIR/in_$nb.bin" >/dev/null 2>&1 || die "adb push failed"
    fi
    exp=$(expected_out "$a"); got=""; res="failed"
    if dev_run --model "m/$a.bitino" --stage "$st" --threads 4 --warmup 0 --iterations 1 \
         --inputs "in_$nb.bin" --outputs "out/$a.out" >/dev/null &&
       "$ADB" pull "$DEV_DIR/out/$a.out" "$TMP/$a.out" >/dev/null 2>&1; then
      got=$(sha256 "$TMP/$a.out")
      if [ -z "$exp" ]; then res="unchecked"; elif [ "$got" = "$exp" ]; then res="match"; else res="MISMATCH"; fi
    fi
    say "output check $a ($st): $res"
    CHECK_RESULT="$CHECK_RESULT$a $res
"
    CHECKS_JSON="$CHECKS_JSON${CHECKS_JSON:+,
}  {\"asset\": $(js "$a"), \"model_id\": $(js "$(model_id "$a")"), \"model_file\": $(js "$(model_file "$a")"), \"stage\": $(js "$st"), \"threads\": 4, \"input\": \"pattern-v1 f32 $nb bytes\", \"output_sha256\": $( [ -n "$got" ] && js "$got" || printf null), \"expected_sha256\": $( [ -n "$exp" ] && js "$exp" || printf null), \"result\": $(js "$res")}"
  done
fi

# ------------------------------------------------------------------------------------------------ timing
caps; START_CAPS="$CAPS_J"; START_BATT=$(battery_json)
RUNS_FILE="$TMP/runs.json"; : > "$RUNS_FILE"
TABLE="$TMP/table.txt"; : > "$TABLE"   # "<asset> <config> <threads> <affinity> <stage> <median_ms> <validity> <fast policy cap khz>"
GAVE_UP=0
n_cfg=$(( $(echo $ASSETS | wc -w) * 2 * REPEATS )); k=0
for rep in $(seq 0 $((REPEATS - 1))); do
  for a in $ASSETS; do
    for cfg in t1_fast t4_fast; do
      k=$((k + 1))
      if [ "$cfg" = t1_fast ]; then thr=1; aff="$FAST_CPU"; else thr="$T4_THREADS"; aff="$T4_AFF"; fi
      st=$(timed_stage "$a"); wu="$WARMUP"
      for attempt in 1 2; do
        # throttle gate: wait while the thermal status is > 0 or a ceiling is below the highest seen in this run
        t0=$(date +%s); limit=$THROTTLE_MAX; [ "$GAVE_UP" = 1 ] && limit=10
        caps
        while [ "$CAPS_THR" = 1 ] && [ $(( $(date +%s) - t0 )) -lt "$limit" ]; do sleep 2; caps; done
        waited=$(( $(date +%s) - t0 )); before="$CAPS_J"; thr_before="$CAPS_THR"; GAVE_UP="$CAPS_THR"
        label="${a}__${cfg}__r${rep}"
        line=$(dev_run --model "m/$a.bitino" --stage "$st" --threads "$thr" --affinity "$aff" --warmup "$wu" \
                 --iterations "$ITERATIONS" --json "out/r.json" --label "$label") || die "run $label failed: $line"
        caps; after="$CAPS_J"; thr_after="$CAPS_THR"
        "$ADB" pull "$DEV_DIR/out/r.json" "$TMP/r.json" >/dev/null 2>&1 || die "could not pull the report of $label"
        aerr=$(grep -m1 '"affinity_errors"' "$TMP/r.json" | tr -dc '0-9')
        if [ "${aerr:-0}" -gt $((wu + 1)) ] && [ "$attempt" = 1 ]; then
          say "[$k/$n_cfg] $label: thread still unpinned ($aerr pin failures; core paused by the vendor?), repeating with a longer warm-up"
          wu=$(( WARMUP * 5 > 50 ? WARMUP * 5 : 50 )); continue
        fi
        break
      done
      med=$(grep -m1 '"median_ms"' "$TMP/r.json" | sed -E 's/.*: *([0-9.eE+-]+).*/\1/')
      case "$med" in ''|*[!0-9.eE+-]*) die "run $label: no median in the report" ;; esac
      throttled=0; { [ "$thr_before" = 1 ] || [ "$thr_after" = 1 ]; } && throttled=1
      capped=0; { echo "$before$after" | grep -q '"capped": true'; } && capped=1
      st_max=$(echo "$before $after" | grep -o '"thermal_status": [0-9]*' | awk '{print $2}' | sort -n | tail -1)
      if [ "$capped" = 0 ] && [ "$throttled" = 0 ]; then val=clean
      elif [ "$throttled" = 1 ]; then val=thermal-throttled
      elif [ "${st_max:-x}" = 0 ]; then val=vendor-policy
      else val=thermal-throttled; fi
      fcap=$(policy_cap "$after" "$FAST_POLICY")
      echo "$a $cfg $thr $aff $st $med $val ${fcap:-0}" >> "$TABLE"
      printf '%s/%s %-44s %-8s %8.3f ms  %s\n' "$k" "$n_cfg" "$a" "$cfg" "$med" "$val" >&2
      {
        [ -s "$RUNS_FILE" ] && printf ',\n'
        printf '  {"asset": %s, "model_id": %s, "model_file": %s, "config": %s, "stage": %s, "threads": %s, "affinity": %s, "repeat": %s,\n' \
          "$(js "$a")" "$(js "$(model_id "$a")")" "$(js "$(model_file "$a")")" "$(js "$cfg")" "$(js "$st")" "$thr" "$(js "$aff")" "$rep"
        printf '   "warmup": %s, "iterations": %s, "median_ms": %s, "validity": %s,\n' "$wu" "$ITERATIONS" "$(jnum "$med")" "$(js "$val")"
        printf '   "throttle": {"mode": "throttle", "waited_s": %s, "timeout": %s, "throttled_before": %s, "throttled": %s, "before": %s, "after": %s},\n' \
          "$waited" "$(jbool "$thr_before")" "$(jbool "$thr_before")" "$(jbool "$throttled")" "$before" "$after"
        printf '   "report": '
        # the raw bitino_bench report, without the build fingerprint and the model manifest (the model is identified
        # by model.data_sha256)
        sed -E 's/("build_fingerprint": )"[^"]*"/\1null/' "$TMP/r.json" | awk '
          skip == 0 && /^  "manifest": \{$/ { skip = 1; next }
          skip == 1 { if ($0 ~ /^  \}/) { skip = 0; print "  \"manifest\": null" ($0 ~ /,$/ ? "," : "") }; next }
          { print }'
        printf '}'
      } >> "$RUNS_FILE"
    done
  done
done
caps; END_CAPS="$CAPS_J"; END_BATT=$(battery_json)

# ------------------------------------------------------------------------------------------------ summary
# per (asset, config): median of the repeat medians, min, max, worst validity, lowest ceiling of the fastest policy
SUMMARY=$(sort -k1,1 -k2,2 -k6,6g "$TABLE" | awk '
  function rank(v) { return v == "clean" ? 0 : v == "vendor-policy" ? 1 : 2 }
  { key = $1 " " $2; if (!(key in n)) { order[++nk] = key; th[key] = $3; af[key] = $4; sg[key] = $5; w[key] = "clean"; cap[key] = $8 }
    n[key]++; m[key, n[key]] = $6; if (rank($7) > rank(w[key])) w[key] = $7; if ($8 + 0 < cap[key] + 0) cap[key] = $8 }
  END { for (i = 1; i <= nk; i++) { k = order[i]; c = n[k]
          med = (c % 2) ? m[k, (c + 1) / 2] : (m[k, c / 2] + m[k, c / 2 + 1]) / 2
          print k, th[k], af[k], sg[k], med, m[k, 1], m[k, c], c, w[k], cap[k] } }')
# "<asset> <config> <threads> <affinity> <stage> <median> <min> <max> <repeats> <validity> <fast cap khz>"

CREATED=$(date -u +%Y-%m-%dT%H:%M:%SZ); DAY=$(date -u +%Y-%m-%d)
SLUG=$(echo "$P_MODEL" | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')
mkdir -p "$OUT_DIR"
JSON_OUT="$OUT_DIR/bitino-bench-$SLUG-$DAY.json"; MD_OUT="$OUT_DIR/bitino-bench-$SLUG-$DAY.md"

topo_json() {
  echo "$TOPO" | awk '{ cpus = ""; for (i = 5; i <= NF; i++) cpus = cpus (cpus == "" ? "" : ", ") $i
    printf "%s\"%s\": {\"cpus\": [%s], \"cpuinfo_max_khz\": %s, \"scaling_max_khz_at_start\": %s, \"governor\": \"%s\"}", (NR > 1 ? ", " : ""), $1, cpus, $2, $3, $4 }'
}
parts_json() { echo "$PARTS" | awk 'NF == 3 { printf "%s{\"cpu\": %s, \"implementer\": \"%s\", \"part\": \"%s\"}", (n++ ? ", " : ""), $1, $2, $3 }'; }
summary_json() {
  echo "$SUMMARY" | while read -r a cfg thr aff st med mn mx c val fcap; do
    [ -n "$a" ] || continue
    printf '%s  {"asset": %s, "model_id": %s, "model_file": %s, "config": %s, "stage": %s, "threads": %s, "affinity": %s, "median_ms": %s, "median_range_ms": [%s, %s], "repeats": %s, "validity": %s, "fastest_policy_ceiling_khz": %s}' \
      "${first:+,
}" "$(js "$a")" "$(js "$(model_id "$a")")" "$(js "$(model_file "$a")")" "$(js "$cfg")" "$(js "$st")" "$thr" "$(js "$aff")" "$med" "$mn" "$mx" "$c" "$(js "$val")" "$(jnum "$fcap")"
    first=1
  done
}

{
  printf '{\n "schema": "%s",\n "tool": {"name": "bench_my_phone.sh", "version": %s, "release": %s, "repo": %s},\n' "$SCHEMA" "$TOOL_VERSION" "$( [ -n "$RELEASE_TAG" ] && js "$RELEASE_TAG" || printf null)" "$(js "$REPO")"
  printf ' "created_utc": "%s",\n' "$CREATED"
  printf ' "device": {"model": %s, "manufacturer": %s, "brand": %s, "device": %s, "soc_manufacturer": %s, "soc_model": %s,\n' \
    "$(js "$P_MODEL")" "$(js "$P_MANUF")" "$(js "$P_BRAND")" "$(js "$P_DEVICE")" "$(js "$P_SOC_M")" "$(js "$P_SOC")"
  printf '  "board_platform": %s, "hardware": %s, "android_release": %s, "api_level": %s, "security_patch": %s, "abi": %s, "abi_list": %s,\n' \
    "$(js "$P_BOARD")" "$(js "$P_HW")" "$(js "$P_REL")" "$(js "$P_API")" "$(js "$P_PATCH")" "$(js "$P_ABI")" "$(js "$P_ABIS")"
  printf '  "kernel": %s, "cpu_features": %s, "online_cpus": %s, "n_cpus": %s,\n' "$(js "$P_KERNEL")" "$(js "$CPU_FEATURES")" "$(js "$ONLINE")" "$(jnum "$N_CPUS")"
  printf '  "emulator_markers": {"ro.kernel.qemu": %s, "ro.boot.qemu": %s, "ro.build.characteristics": %s},\n' "$(js "$Q1")" "$(js "$Q2")" "$(js "$P_CHAR")"
  printf '  "cpus": [%s],\n  "cpufreq_policies": {%s},\n' "$(parts_json)" "$(topo_json)"
  printf '  "fastest_cpu": "%s", "fastest_policy": "%s", "t4_affinity": "%s"},\n' "$FAST_CPU" "$FAST_POLICY" "$T4_AFF"
  printf ' "binary": {"source": "%s", "asset": %s, "sha256": "%s", "runtime_version": %s},\n' "$BIN_SOURCE" "$(js "$( [ "$BIN_SOURCE" = release ] && echo "$BIN_ASSET" || basename "$BIN_PATH")")" "$BIN_SHA" "$(js "$RUNTIME_VERSION")"
  printf ' "models": ['
  i=0; for a in $ASSETS; do p=$(mpath "$a"); src=$(echo "$MODEL_PATHS" | awk -v a="$a" '$1 == a {print $3}')
    printf '%s{"asset": %s, "model_id": %s, "model_file": %s, "source": "%s", "sha256": "%s"}' "$( [ $i -gt 0 ] && echo ', ')" "$(js "$a")" "$(js "$(model_id "$a")")" "$(js "$(model_file "$a")")" "$src" "$(sha256 "$p")"; i=1; done
  printf '],\n'
  printf ' "plan": {"configs": {"t1_fast": "1 thread pinned to the fastest core", "t4_fast": "4 threads pinned to the 4 fastest cores, fastest first"}, "warmup": %s, "iterations": %s, "repeats": %s, "order": "round-robin over files and configurations per repeat", "throttle_gate": {"max_wait_s": %s, "rule": "thermal status > 0 or scaling_max_freq below the highest seen in this run"}},\n' \
    "$WARMUP" "$ITERATIONS" "$REPEATS" "$THROTTLE_MAX"
  printf ' "conditions": {"start": {"caps": %s, "battery": %s}, "end": {"caps": %s, "battery": %s}},\n' "$START_CAPS" "$START_BATT" "$END_CAPS" "$END_BATT"
  printf ' "output_checks": [\n%s\n ],\n' "$CHECKS_JSON"
  printf ' "summary": [\n%s\n ],\n' "$(summary_json)"
  printf ' "runs": [\n'; cat "$RUNS_FILE"; printf '\n ]\n}\n'
} > "$JSON_OUT"

ghz() { awk -v k="$1" 'BEGIN { if (k + 0 > 0) printf "%.2f", k / 1e6; else printf "?" }'; }
lowcap=$(echo "$SUMMARY" | awk '$11 + 0 > 0 {print $11}' | sort -n | head -1)
{
  echo "### Bitino speed: $P_MANUF $P_MODEL"
  echo
  echo "| | |"
  echo "|---|---|"
  echo "| Phone | $P_MANUF $P_MODEL ($P_DEVICE) |"
  echo "| SoC | ${P_SOC:+${P_SOC_M:+$P_SOC_M }$P_SOC, }platform ${P_BOARD:-?} |"
  echo "| Android | $P_REL (API $P_API), $P_ABI |"
  echo "| Fastest core | cpu$FAST_CPU, max $(ghz "$FAST_POLICY_MAX") GHz; lowest ceiling during the runs $(ghz "${lowcap:-0}") GHz |"
  echo "| 4 threads on | cpu $T4_AFF |"
  echo "| Runtime | bitino $RUNTIME_VERSION ($BIN_SOURCE binary${RELEASE_TAG:+; files from release $RELEASE_TAG}) |"
  echo "| Battery | start $(echo "$START_BATT" | sed -E 's/.*"temperature_c": ([^,]*),.*/\1/') C, end $(echo "$END_BATT" | sed -E 's/.*"temperature_c": ([^,]*),.*/\1/') C; USB powered $(echo "$START_BATT" | sed -E 's/.*"usb_powered": ([a-z]*).*/\1/'); screen $(echo "$START_BATT" | sed -E 's/.*"wakefulness": "([^"]*)".*/\1/') |"
  echo
  echo "| model | file | stage | 1 thread (ms) | 4 threads (ms) | validity | output check |"
  echo "|---|---|---|---|---|---|---|"
  for a in $ASSETS; do
    t1=$(echo "$SUMMARY" | awk -v a="$a" '$1 == a && $2 == "t1_fast" {printf "%.2f", $6}')
    t4=$(echo "$SUMMARY" | awk -v a="$a" '$1 == a && $2 == "t4_fast" {printf "%.2f", $6}')
    v=$(echo "$SUMMARY" | awk -v a="$a" '$1 == a {print $10}' | sort -u | paste -sd/ -)
    oc=$(echo "$CHECK_RESULT" | awk -v a="$a" '$1 == a {print $2}')
    echo "| $(model_id "$a") | $(model_file "$a") | $(timed_stage "$a") | $t1 | $t4 | $v | ${oc:-not run} |"
  done
  echo
  echo "Median of $REPEATS repeats x $ITERATIONS timed runs (after $WARMUP warm-up runs), batch 1. Full data: \`$(basename "$JSON_OUT")\`."
} > "$MD_OUT"

cat "$MD_OUT"
echo
say "wrote $JSON_OUT and $MD_OUT"
if echo "$CHECK_RESULT" | grep -q MISMATCH; then
  echo "warning: an output check did not match the reference: please mention it in the issue" >&2
fi
say "submit: open a 'Speed on my phone' issue at https://github.com/$REPO/issues/new/choose, paste the .md and attach the .json"
