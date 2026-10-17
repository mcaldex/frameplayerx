#!/bin/bash
# Renders every FramePlayerX screen to flat PNGs for design review, with the
# desktop preview (no headset; any Vulkan driver, lavapipe on CI).
#
# Usage: tools/ui-preview.sh OUT_DIR
# Runs against a throwaway home with a small library made from the test
# clips, so it never touches your own settings or library.
set -euo pipefail
cd "$(dirname "$0")/.."
out="$(mkdir -p "${1:?usage: $0 OUT_DIR}" && cd "$1" && pwd)"
data=crates/fp-media/tests/data

home="$(mktemp -d)"
trap 'rm -rf "$home"' EXIT
mkdir -p "$home/Videos"
cp "$data/h264_aac_180_LR.mp4" "$home/Videos/Sunset Beach Walk_180_LR.mp4"
cp "$data/camera_clip.mp4" "$home/Videos/Garden Party.mp4"
cp "$data/vp9.webm" "$home/Videos/Mountain Lake_360.webm"
cp "$data/av1_opus.webm" "$home/Videos/City Lights_180_SBS.webm"
cp "$data/hevc10_tb.mkv" "$home/Videos/Aurora Timelapse_360_TB.mkv"
# Different dates, and a folder, for the browse shots' sort orders.
mkdir -p "$home/Videos/Holidays"
touch -d 2026-03-14 "$home/Videos/Sunset Beach Walk_180_LR.mp4"
touch -d 2026-09-02 "$home/Videos/Garden Party.mp4"
touch -d 2025-11-20 "$home/Videos/Mountain Lake_360.webm"
touch -d 2026-06-30 "$home/Videos/City Lights_180_SBS.webm"
touch -d 2026-01-08 "$home/Videos/Aurora Timelapse_360_TB.mkv"
# A source to lock with a PIN.
mkdir -p "$home/Private" "$home/.config/frameplayerx"
cp "$data/camera_clip.mp4" "$home/Private/Rooftop.mp4"
(umask 077 && cat >"$home/.config/frameplayerx/sources.json" <<EOF
[{"type": "local", "id": "private", "name": "Private videos", "root": "$home/Private"}]
EOF
)

script="$home/ui.fps"
cat >"$script" <<EOF
# Library scan and thumbnails. Nothing watched yet: Home offers sources.
wait 4
panel main home
screen history
panel main history-empty
# The headset's own page: its videos, then its folders.
open-source on-device
wait 1
panel main local
local-folders
panel main local-folders
screen sources
panel main sources
# Browsing a folder of the home source, by name and then newest first.
browse device-home $home/Videos
wait 1
panel main browse
browse-sort newest
panel main browse-newest
browse-sort name
# A source locked with a PIN: its row, the unlock form, its PIN settings.
lock private 2468
screen sources
panel main sources-locked
pin-form private
panel main source-unlock
pin-form private wrong
panel main source-unlock-wrong
unlock private 2468
pin-form private set
panel main source-pin
screen settings playback
panel main settings-playback
screen settings passthrough
panel main settings-passthrough-locked
unlocked
screen settings passthrough
panel main settings-passthrough
chroma-global
screen settings passthrough
frames 8
panel main settings-passthrough-on
screen settings controller
panel main settings-controller
screen settings controller-open
panel main settings-controller-open
screen settings controller-stick
panel main settings-controller-stick
screen settings device
panel main settings-device
haptics on
screen settings haptics
panel main settings-haptics
screen settings remote
panel main settings-remote
screen settings updates
panel main settings-updates
screen settings about
panel main settings-about
# Hover state on the first card of the headset's videos.
open-source on-device
point main 0.12 0.45
frames 8
panel main local-hover
# The keyboard, from the search field.
point main 0.12 0.30
click
frames 8
panel keyboard keyboard
# Its symbols page (the #+= key), then back to letters.
point keyboard 0.195 0.80
click
frames 8
panel keyboard keyboard-symbols
click
frames 3
button secondary
frames 3
# The player.
open $home/Videos/Sunset Beach Walk_180_LR.mp4
wait-playing
wait 1
pause
point bar 0.5 0.5
frames 8
panel bar player-bar
sbs player-scene
# A hover label: the Hide button.
point bar 0.883 0.69
frames 10
panel bar player-bar-tip
# The adjustments panel.
adjust
frames 8
panel adjust adjust
panel bar player-bar-adjust
# The Passthrough tab, with chroma key on (unlocked above).
adjust 4
chroma
mask
frames 8
panel adjust adjust-passthrough
# Software decoding falling behind.
slow-decode
frames 8
panel bar player-bar-slow-decode
# What was watched: the history, and Home's banner, over the paused video
# (Back closes the adjustments, then shows the browser).
button secondary
frames 3
button secondary
frames 3
screen history
frames 4
panel main history
screen home
frames 4
panel main home-watched
EOF

cargo build -p frameplayerx
HOME="$home" XDG_CONFIG_HOME="$home/.config" XDG_DATA_HOME="$home/.local/share" \
  XDG_CACHE_HOME="$home/.cache" \
  "${CARGO_TARGET_DIR:-target}/debug/frameplayerx" --preview "$out" --script "$script" --size 1400
ls -l "$out"
