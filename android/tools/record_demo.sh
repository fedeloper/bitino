#!/usr/bin/env bash
# Records the demo app running on a connected phone and turns the recording into an optimized GIF (ffmpeg palette).
#
#   android/tools/record_demo.sh [out.gif]          (default docs/img/android_demo.gif)
#
# The frames are captured by the app itself (`--ei capture_seconds N`: PixelCopy of its window at ~10 fps, JPEG files
# in its external files directory), because `adb shell screenrecord` is blocked on some phones (e.g. OnePlus /
# ColorOS: "Unable to open ...: Permission denied"). The app must be installed with the camera permission granted and
# the screen unlocked (the script only wakes it).
# Env: ANDROID_SERIAL, ADB (default adb), FFMPEG (default ffmpeg), SECONDS_REC (14), THREADS (1, at start),
#      SWITCH_TAP="<x> <y>" (tapped at half time, e.g. the "4" threads button, to show the thread switch),
#      GIF_WIDTH (360), GIF_FPS (7), MAX_BYTES (4000000), FRAMES_DIR (keep the captured JPEG frames there).
set -euo pipefail
export LC_ALL=C
OUT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/docs/img/android_demo.gif}"
ADB="${ADB:-adb}"
FFMPEG="${FFMPEG:-ffmpeg}"
SECS="${SECONDS_REC:-14}"
PKG=io.github.fedeloper.bitino.demo
REMOTE=/sdcard/Android/data/$PKG/files/capture
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

"$ADB" shell input keyevent KEYCODE_WAKEUP
"$ADB" shell am force-stop "$PKG"
"$ADB" shell am start -n "$PKG/.MainActivity" --ei threads "${THREADS:-1}" --ei capture_seconds "$SECS" \
  --ei capture_width 540 >/dev/null
if [[ -n "${SWITCH_TAP:-}" ]]; then
  sleep $(( SECS / 2 + 1 ))
  "$ADB" shell input tap $SWITCH_TAP
fi
for _ in $(seq 1 $(( SECS + 30 ))); do
  "$ADB" shell test -e "$REMOTE/done" && break
  sleep 1
done
"$ADB" shell test -e "$REMOTE/done" || { echo "capture did not finish" >&2; exit 1; }
"$ADB" pull "$REMOTE" "$TMP/" >/dev/null 2>&1
"$ADB" shell am force-stop "$PKG"
"$ADB" shell rm -rf "$REMOTE"
[[ -n "${FRAMES_DIR:-}" ]] && { mkdir -p "$FRAMES_DIR"; cp "$TMP"/capture/*.jpg "$FRAMES_DIR/"; }

# concat list with the real frame times (file names are milliseconds since the capture started)
frames=$(ls "$TMP/capture/"*.jpg | sort)
prev=""
: > "$TMP/list.txt"
for f in $frames; do
  t=$(basename "$f" .jpg); t=$((10#$t))
  if [[ -n "$prev" ]]; then
    printf "file '%s'\nduration %.3f\n" "$pf" "$(echo "($t - $prev) / 1000" | bc -l)" >> "$TMP/list.txt"
  fi
  prev=$t; pf=$f
done
printf "file '%s'\nduration 0.1\nfile '%s'\n" "$pf" "$pf" >> "$TMP/list.txt"

mkdir -p "$(dirname "$OUT")"
w="${GIF_WIDTH:-360}"; fps="${GIF_FPS:-7}"; max="${MAX_BYTES:-4000000}"
while :; do
  f="fps=$fps,scale=$w:-1:flags=lanczos"
  "$FFMPEG" -v error -y -f concat -safe 0 -i "$TMP/list.txt" -vf "$f,palettegen=max_colors=128:stats_mode=diff" "$TMP/pal.png"
  "$FFMPEG" -v error -y -f concat -safe 0 -i "$TMP/list.txt" -i "$TMP/pal.png" \
    -lavfi "$f [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" -loop 0 "$OUT"
  size=$(stat -c %s "$OUT")
  echo "$OUT: $(echo "$frames" | wc -l) frames captured, GIF ${w}px wide at $fps fps, $size bytes"
  (( size <= max )) && break
  (( w <= 200 )) && { echo "cannot get under $max bytes" >&2; exit 1; }
  w=$(( w * 85 / 100 ))
done
