#!/usr/bin/env bash
# Cross-compile workspace binaries for the Steam Frame (Linux ARM64).
#
# Targets glibc 2.28 so the binaries run on any SteamOS image, inside or
# outside the Steam Linux Runtime container. Uses zig as the cross linker, so
# no ARM64 sysroot or C toolchain is needed.
#
# One-time setup: zig and cargo-zigbuild from tools/requirements.txt (see
# there); rustup adds the ARM64 target from rust-toolchain.toml.
#
# Usage: tools/build-frame.sh [cargo args...]   e.g. tools/build-frame.sh -p frame-probe
set -euo pipefail
cd "$(dirname "$0")/.."

# CI can supply the key in the environment. Local builds can keep it in the ignored key file.
if [ -z "${FP_OPENSUBTITLES_API_KEY:-}" ] && [ -f opensubtitles.key ]; then
  FP_OPENSUBTITLES_API_KEY=$(<opensubtitles.key)
  export FP_OPENSUBTITLES_API_KEY
fi

target=aarch64-unknown-linux-gnu
cargo zigbuild --release --target "$target.2.28" "$@"
echo "Built into target/$target/release/"
