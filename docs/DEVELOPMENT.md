# Developing FramePlayerX

How to build, test and release FramePlayerX, and how the code fits
together. For using the app, see the [user guide](USER-GUIDE.md).

## Requirements

- **Linux** (Ubuntu or Debian), or **Windows with Ubuntu under WSL2**. The
  workspace doesn't build on Windows itself.
- Rust 1.95 or newer.
- For FFmpeg: meson 1.12 (1.3 can't use zig's linker; versions in between
  are untested), ninja, nasm and pkg-config.
- For builds for the Frame: zig (`pip install ziglang`) and `cargo-zigbuild`
  (`cargo install --locked cargo-zigbuild`), and
  `rustup target add aarch64-unknown-linux-gnu`.

On Ubuntu, `tools/local-xr.sh setup` installs Monado and the tools FFmpeg
needs with apt, then builds FFmpeg, in one go.

## Build and test

```bash
tools/build-ffmpeg.sh x86_64 aarch64      # once: FFmpeg 7.1 and dav1d into third_party/
export LD_LIBRARY_PATH=$PWD/third_party/ffmpeg/x86_64/lib
cargo test --workspace                    # GPU tests use any Vulkan device (lavapipe is fine)
cargo clippy --workspace --all-targets   # lints are in [workspace.lints]
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
(cd tools/frame-desktop-install && python3 -m unittest)   # the Desktop Mode installer
```

Under WSL, keep the Linux build off the Windows drive with
`export CARGO_TARGET_DIR=~/.cache/frameplayerx-target`, as
`tools/local-xr.sh` does.

## Code style

`cargo fmt` formats. Clippy, with the lints in `[workspace.lints]`, has to
pass with warnings denied, as it does in CI.

Comments say what the code can't:

- Each module starts with `//!`: what it is for, in a sentence or a short
  paragraph.
- `///` documents an item when its name and signature leave something out:
  units, what `None` means, that it blocks, what it costs. A field called
  `title` needs no `/// Title.`
- `//` inside a function says why: a hardware or server quirk, an
  invariant, a workaround and what it works around. Not what the next line
  does.
- Every `unsafe` block has a `// SAFETY:` comment saying why it is sound
  (clippy checks).
- No commented-out code. A known gap is an issue, not a `TODO`.

## Repository layout

| Crate | Purpose |
|---|---|
| `frameplayerx` | The app: the XR loop, the interface, and wiring everything together |
| `fp-core` | Shared types: formats and their detection, view settings, the source contract, data folders |
| `fp-ffmpeg-sys` | Bindings to the bundled FFmpeg 7.1, per architecture (`tools/gen-ffmpeg-bindings.sh`) |
| `fp-media` | Demuxing, decoding (V4L2 hardware, software), the A/V clock, ALSA audio, subtitles, thumbnails |
| `fp-render` | The Vulkan renderer: the projection shader for every format, egui panels, quads |
| `fp-xr` | OpenXR: the session and swapchains, controller input, passthrough |
| `fp-xr-loader` | Finds the active OpenXR runtime and loads it directly, with no Khronos loader to ship (shared with `frame-probe`) |
| `fp-library` | The SQLite database: folder scanning, the metadata and thumbnail worker, search, watch history |
| `fp-sources` | Sources: local folders, HTTP, WebDAV, DLNA, SMB, DeoVR and HereSphere feeds, Jellyfin and Emby |
| `fp-haptics` | Funscripts, the sync engine, and TCode, Buttplug and Handy devices |
| `fp-remote` | The DeoVR remote API and the phone web remote |
| `fp-updater` | Signed update manifests, download, install and rollback; the `fp-release` maintainer tool |
| `frameplayerx-install` | The PC installer: pairs with the headset and installs over SSH |
| `frame-probe` | Reports what a Steam Frame offers a native app (see [The platform probe](#the-platform-probe)) |

| Folder | What's in it |
|---|---|
| `assets/` | The logo (`logo.svg`), the Steam library artwork made from it, and the docs' install buttons |
| `site/` | The GitHub Pages site: the page the Frame Control button opens (see [Install buttons](#install-buttons)) |
| `packaging/` | What goes in the release zip besides the binary: the launcher, a README and the licences |
| `tools/` | Build, package, preview and test scripts |
| `tools/frame-desktop-install/` | The Desktop Mode installer: a Python script, and the install and uninstall `.desktop` launchers that download and run it (see its [README](../tools/frame-desktop-install/README.md)) |
| `vendor/smb/` | The `smb` crate with a credit-accounting fix (see `vendor/smb/PATCHES.md`) |

## How it fits together

**The app loop.** One loop owns the OpenXR session and the Vulkan device.
Each frame it reads the controllers, builds the interface with egui and
draws both eyes: the video through the projection shader, then the panels
as quads in the scene. Interface code doesn't change the app's state
itself: it returns `Action`s that `App::apply` carries out. Anything slow
(opening a video, browsing a server, scanning folders, checking for
updates) runs as a background job whose result comes back on a channel, so
a frame doesn't wait on the network.

**Video.** FFmpeg demuxes through fp-core's `ByteSource`, so every network
protocol is implemented in Rust and looks like a file to FFmpeg. H.264 and
HEVC go to the Frame's V4L2 stateful decoder; anything it can't take is
decoded in software. Frames are uploaded to Vulkan as YUV and converted in
the shader, which maps each pixel of each eye onto the video for the chosen
projection and stereo layout. HDR is tone mapped there too.

**Sources and browsing.** Every source implements one `Source` trait
(`list`, `open`, `details`). Each kind gets its own browser in `ui/`:
`local.rs`, `folders.rs`, `jellyfin.rs` and `feeds.rs`, built from the
shared pieces in `ui/page.rs`. Browsing a network source writes nothing to
the database; a video gets a row when it's opened.

**The database.** SQLite, with the schema version in `PRAGMA user_version`
and forward-only migrations in `fp-library/src/schema.rs`. The scanner
marks files missing rather than deleting them, so ratings, resume points
and adjustments survive a card being taken out.

**Data on the headset.**

| Folder | What's in it |
|---|---|
| `~/frameplayerx` | The app: binary, launcher, `lib/` (FFmpeg, dav1d), artwork, `VERSION` |
| `~/frameplayerx.old` | The previous version, kept for rollback |
| `~/.config/frameplayerx` | `settings.json`, `sources.json` (passwords, readable only by the user) and the optional `env` file |
| `~/.local/share/frameplayerx` | The database, `frameplayerx.log` (and `.log.old`), and `Interactive/`, searched for scripts not next to their video |
| `~/.cache/frameplayerx` | Thumbnails, scrub previews, and `feeds/`: each scene feed's last listing, shown at once the next time it's opened (readable only by the user: its URLs can carry the server's API key) |

## Running without a headset

### On this computer with Monado

`tools/local-xr.sh` runs FramePlayerX against [Monado](https://monado.freedesktop.org),
the open-source OpenXR runtime, so a change can be tried without building
and installing on the Frame. Monado shows both eyes in a desktop window and
fakes the headset and controllers from the keyboard and mouse. It works on
Ubuntu and Debian, and on Windows in Ubuntu under WSL2 (WSLg shows the
windows and plays the sound; Vulkan runs on llvmpipe there, so expect a low
frame rate).

```bash
tools/local-xr.sh setup                   # once: Monado, build tools, FFmpeg
tools/local-xr.sh run                     # Monado + FramePlayerX
tools/local-xr.sh run ~/Videos/clip.mp4   # ...opening a video
tools/local-xr.sh headless --exit-after 10 clip.mp4   # no window: smoke test
tools/local-xr.sh preview /tmp/shots      # every screen as PNGs (ui-preview.sh)
```

Click into Monado's debug window (*Monado!*) to drive it:

| Input | What it does |
|---|---|
| Keyboard and mouse | Move the headset |
| Left Ctrl / left Alt held | Move the left / right controller instead |
| WASD, Q/E | Move |
| Arrow keys, right-button drag | Turn |
| Left click | Trigger |
| Shift | Faster |

Its *Qwerty › Help* tab lists every key. The controllers present
themselves as Khronos simple controllers (select and menu), which
FramePlayerX binds.

- The window with the eyes opens at 1600×900; `LOCAL_XR_WINDOW=1280x720`
  picks another size, and it can be resized. Under WSL it's Monado's X11
  window, as the Wayland one stays black there.
- The eyes are 150% of the fake headset's 640×720.
  `XRT_COMPOSITOR_SCALE_PERCENTAGE=200 tools/local-xr.sh run` is sharper,
  `=60` smoother on a software renderer.
- Monado's log is in `/tmp/monado-service.log`; `-v` turns on
  FramePlayerX's debug log.

By hand, with Monado's simulated headset and no window:

```bash
XRT_COMPOSITOR_NULL=1 SIMULATED_ENABLE=1 monado-service &
XR_RUNTIME_JSON=/usr/share/openxr/1/openxr_monado.json \
  cargo run -p frameplayerx -- --exit-after 10 video.mp4
```

### The preview harness

`--preview` needs no OpenXR runtime at all: it renders offscreen with a
simulated head and right controller, driven by a script, and saves
screenshots. `tools/ui-preview.sh` uses it to render every screen; the
script commands are listed in `crates/frameplayerx/src/preview.rs`.

```bash
cat > /tmp/demo.fps <<'X'
wait 2
point main 0.24 0.04     # Watch history tab
click
shot history
open ~/Videos/clip_180_LR.mp4
wait-playing 20
shot playing
X
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo run -p frameplayerx -- --preview /tmp/shots --script /tmp/demo.fps
```

### Other modes

| Command | What it does |
|---|---|
| `frameplayerx --info FILE` | Prints what FramePlayerX detects about a video: format, streams, HDR |
| `frameplayerx --decode-bench FILE [--sw]` | Decodes without displaying and prints the decoder and frame rate |
| `frameplayerx --seek-test FILE\|URL` | Skips through a video headless, checking buffering and that hardware decoders are given back |

## On the headset

```bash
tools/build-frame.sh -p frameplayerx      # cross-compile for the Frame
tools/package.sh                          # release zip and manifests in dist/
frameplayerx-install --zip dist/frameplayerx-<version>-aarch64.zip
ssh frame tail -f .local/share/frameplayerx/frameplayerx.log
```

Builds target `aarch64-unknown-linux-gnu` with glibc 2.28, so they run on
any SteamOS image. `tools/package.sh` checks that the binary is ARM64,
needs nothing newer than glibc 2.28 and finds its libraries through
`$ORIGIN/lib`.

### Environment variables

Steam can't pass environment variables to an app, so the launcher reads
`KEY=value` lines from `~/.config/frameplayerx/env` before starting.

| Variable | Default | What it does |
|---|---|---|
| `FRAMEPLAYERX_REFRESH_RATE` | `90` | The refresh rate set in SteamVR's per-app settings for FramePlayerX, unless you chose one there. `0` leaves it alone |
| `FRAMEPLAYERX_RESOLUTION` | `2160` | Pixels per eye, set the same way (SteamVR's default is 1728). `0` leaves it alone |
| `FRAMEPLAYERX_DEBUG_INPUT` | | `1` logs each controller's input when it changes |
| `FRAMEPLAYERX_V4L2` | | `0` turns off the native V4L2 decoder |
| `FRAMEPLAYERX_V4L2_DEVICE` | | The V4L2 device to decode with, instead of the one found |

### Steam Frame notes

What the code relies on about the platform:

- SteamVR runs **on the headset** and is the OpenXR runtime. FramePlayerX
  finds the active runtime's manifest and talks to it directly, so there's
  no Khronos loader to ship.
- Native ARM64 apps run **outside the Steam Linux Runtime** container, so
  the release carries FFmpeg and dav1d in `lib/` and asks nothing newer
  than glibc 2.28 of the system.
- The Frame controllers have their own interaction profile,
  `/interaction_profiles/valve/frame_controller_valve`, from the
  `XR_VALVE_frame_controller_interaction` extension. Without it SteamVR
  presents them as Touch controllers.
- SteamVR's per-app video settings override what an app asks for through
  OpenXR, which is why the launcher sets the refresh rate and resolution
  there.
- The hardware decoder is a V4L2 stateful decoder for 8-bit 4:2:0 H.264
  and HEVC. It's shared with Steam's own streaming, so it can be busy.
- SteamOS only mounts ext4 microSD cards itself, and no USB drives;
  FramePlayerX mounts exFAT, FAT32 and NTFS itself under
  `/run/media/steamos/<name>`.
- Apps installed in Developer Mode appear in the Steam library as non-Steam
  games; `frameplayerx-install` writes that entry to `shortcuts.vdf`.

### The platform probe

`frame-probe` answers what a Frame offers a native app in one run: the
OpenXR runtime and its extensions, refresh rates, render size, swapchain
formats, controller profiles, V4L2 decoders, Vulkan Video and DMA-BUF
import. `tools/frame-probe.sh --build` builds it, runs it on a paired
headset and copies the text and JSON reports into `probe-reports/`.

## Releasing

1. Once: run `fp-release keygen`, paste the printed public key into
   `crates/fp-updater/src/lib.rs` (`RELEASE_PUBLIC_KEY`), and store the
   whole `.key` file as the `FP_RELEASE_KEY` repository secret. Keep the key
   offline otherwise.
2. Stable: push (or merge) to `main`. Beta: run the `ci` workflow on any
   other branch from the Actions tab (Run workflow). CI calculates the
   version, stamps it into `crates/frameplayerx/Cargo.toml`, builds,
   packages, signs `manifest.json` and publishes the GitHub release with
   the zip, `manifest.json(.sig)`, `framedrop.json`, `frameplayerx-install`
   for each PC, and the Desktop Mode installer and its install and uninstall launchers.

   - `main` gets the next stable `X.Y.Z` from the conventional commits
     since the last stable tag (`git cliff --bumped-version`): a breaking
     change (`feat!:`, `BREAKING CHANGE:`) bumps X, a `feat` bumps Y,
     anything else bumps Z. Beta tags don't count. With no stable tag yet
     it's `initial_tag` in `cliff.toml`.
   - Other branches get that version as `X.Y.Z-beta.N` (N one above the
     highest existing beta of it, and always above the latest stable),
     published as a GitHub pre-release. FramePlayerX offers it on the Beta
     update channel only.
   - Pushing a `vX.Y.Z` or `vX.Y.Z-anything` tag releases exactly that
     version (a pre-release if it has a `-`).

The app finds updates in GitHub's release list: the highest version on
the Stable channel without a `-` part, on Beta any version. In-app updates
and `frameplayerx-install` without `--zip` only install releases whose
manifest is signed with the release key.

### Install buttons

The README's install buttons are images in `assets/`, because GitHub strips
the inline styles and SVG of the HTML buttons FrameDrop and Frame Control
publish. They all point at the latest release, so they need no change per
release:

| Button | Opens |
|---|---|
| Desktop Mode / Uninstall | The release's `Install-FramePlayerX.desktop` / `Uninstall-FramePlayerX.desktop` |
| FrameDrop | `https://framedropvr.com/install?manifest=` with the release's `framedrop.json`, FrameDrop's documented link: it opens FrameDrop, or FrameDrop's site when it isn't installed |
| Frame Control | `site/frame-control.html` on GitHub Pages, which opens `frame-control://install?manifest=` with the release's `frameplayerx.json`. GitHub doesn't keep links to app schemes like `frame-control://` in Markdown, so a web page has to |

`.github/workflows/pages.yml` publishes `site/` (with the logo) when it
changes on `main`. Once, in the repository settings: **Pages › Build and
deployment › Source: GitHub Actions**.

## Artwork

`assets/logo.svg` is the logo. `python3 tools/make-artwork.py` renders the
Steam library artwork in `assets/steam/` from it (needs Pillow and
`rsvg-convert`); re-run it after changing the logo.
