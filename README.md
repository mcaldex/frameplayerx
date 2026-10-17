<p align="center">
  <img src="assets/logo.svg" alt="FramePlayerX Logo" width="160" />
</p>

# FramePlayerX

[![CI](https://github.com/remotesojourner/frameplayerx/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/remotesojourner/frameplayerx/actions/workflows/ci.yml)
[![Latest Release](https://img.shields.io/github/v/release/remotesojourner/frameplayerx?sort=semver&style=flat&logo=github&label=release)](https://github.com/remotesojourner/frameplayerx/releases/latest)
[![Latest Beta](https://img.shields.io/github/v/release/remotesojourner/frameplayerx?include_prereleases&sort=date&style=flat&logo=github&label=beta&color=orange)](https://github.com/remotesojourner/frameplayerx/releases)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat)](LICENSE)

A VR video player for the Valve Steam Frame. It plays flat, 180°, 360° and fisheye videos, in 2D or 3D, from the headset, a microSD card, a USB drive or a media server. It runs natively on the headset.

> [!NOTE]
> FramePlayerX is a fork of [FramePlayer](https://github.com/yellkell/frameplayer) by [yellkell](https://github.com/yellkell), who created the original player and released it as open source. Thank you, yellkell, for the work this project is built on.

---

## Features

- **VR and flat videos**: plays 180°, 360°, fisheye and flat videos, in 2D or 3D. The format is detected automatically, and you can change it for any video
- **Picture adjustments**: position, zoom, depth and picture settings, saved for each video
- **Videos on the headset**: finds videos in the Videos and Downloads folders and on microSD cards and USB drives, including cards formatted on Windows or a Mac
- **Media servers**: browse and play from Jellyfin, Emby, any server with a DeoVR feed or HereSphere API, network shares (SMB), WebDAV, DLNA servers and web folders. Jellyfin and Emby remember what you watched and where you stopped
- **Watch history**: carry on where you left off, and mark favorites and ratings
- **PIN locks**: hide a source behind a PIN until you unlock it
- **Passthrough videos**: see your room around the people in passthrough videos, and behind green or blue screens
- **Subtitles and spatial audio**: subtitles from a file or inside the video, and 360° audio that follows your head
- **Haptic devices**: plays funscripts on The Handy, devices that work with Intiface Central, and OSR2 and SR6
- **Phone remote**: scan a QR code to control playback and browse from your phone. Apps made for DeoVR's remote work too
- **Custom controls**: choose what each controller button does
- **Updates**: checks for a new version when it starts, and installs it from Settings when you choose to, on a Stable or a Beta channel
- **No accounts or tracking**: it only connects to the servers you add, and to GitHub to check for updates

---

## Requirements

| Requirement | Notes |
|---|---|
| Valve Steam Frame | Developer Mode only to install with Frame Control, FrameDrop or a PC |
| A PC (Windows, macOS or Linux) | Optional: only to install from a PC |
| Media server | Optional: Jellyfin or Emby, a DeoVR feed or HereSphere API, a NAS share, WebDAV or DLNA |

---

## Quick Start

1. **Install FramePlayerX** in one of three ways:

   - **On the headset, in Desktop Mode**: open this page in the headset's browser and press the button to download the installer. Open `Install-FramePlayerX.desktop` from your Downloads folder and confirm when Desktop Mode asks whether to trust it. A terminal window opens, installs the latest release and adds FramePlayerX to your Steam library. It may ask to close Steam for a moment; the app then shows up in Gaming Mode.

     <a href="https://github.com/remotesojourner/frameplayerx/releases/latest/download/Install-FramePlayerX.desktop"><img src="assets/install-button.svg" alt="Download the installer: Install-FramePlayerX.desktop" width="300" /></a>

   - **With FrameDrop or Frame Control, from a PC**: turn on Developer Mode (Steam Settings › System › Developer Mode) and pair the headset once with [FrameDrop](https://framedropvr.com) or [Frame Control](https://github.com/saphid/steam-frame). Then, in your PC's browser, press the button and the app installs the latest release on the headset:

     <a href="https://framedropvr.com/install?manifest=https%3A%2F%2Fgithub.com%2Fremotesojourner%2Fframeplayerx%2Freleases%2Flatest%2Fdownload%2Fframedrop.json"><img src="assets/framedrop-button.svg" alt="Install with FrameDrop" width="253" /></a>
     <a href="https://remotesojourner.github.io/frameplayerx/frame-control.html"><img src="assets/frame-control-button.svg" alt="Install with Frame Control" width="266" /></a>
   - **With `frameplayerx-install`, from a PC**: turn on Developer Mode, download `frameplayerx-install` for your PC from the [release page](https://github.com/remotesojourner/frameplayerx/releases/latest), then:

     ```bash
     frameplayerx-install pair 192.168.0.68    # once, with the headset's IP address
     frameplayerx-install                      # installs the latest release
     ```

     For `pair`, open Steam Settings › Developer › Pair new host on the headset and keep it showing.

2. **Start it** from the Steam library on the headset. FramePlayerX finds the videos on the headset and its cards and drives by itself. Add a media server under **Sources › Add source**.

To update later, open **Settings › Updates** and press **Download and install** when a new version is listed, then restart FramePlayerX. To remove it again, see [Uninstall](docs/USER-GUIDE.md#uninstall) in the user guide.

---

## Documentation

| Guide | What's in it |
|---|---|
| [User guide](docs/USER-GUIDE.md) | Installing, updating and uninstalling, sources, browsing, PIN locks, video formats, controls, passthrough, haptics, remote control and troubleshooting |
| [Development](docs/DEVELOPMENT.md) | Building and testing, how the code is organised, running without a headset (Monado and the preview harness), and releasing |

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Rust |
| VR | OpenXR, talking to the runtime directly (no Khronos loader) |
| Graphics | Vulkan through [ash](https://github.com/ash-rs/ash); the projections are drawn per pixel in a shader |
| Interface | [egui](https://github.com/emilk/egui) panels in VR, with the [Inter](https://rsms.me/inter/) font and [Phosphor](https://phosphoricons.com/) icons |
| Video | [FFmpeg](https://ffmpeg.org/) 7.1 for demuxing and software decoding, the Frame's V4L2 hardware decoder, [dav1d](https://code.videolan.org/videolan/dav1d) for AV1 |
| Audio | ALSA |
| Storage | SQLite (videos, watch history, adjustments), JSON settings |
| Builds | `cargo zigbuild` for Linux ARM64 against glibc 2.28 |

---

## License

FramePlayerX is licensed under the [Apache License, Version 2.0](LICENSE). It's a modified version of FramePlayer by yellkell; see [NOTICE](NOTICE). Bundled third-party libraries and fonts keep their own licences, and the controller pictures are renders of Valve's Steam Frame controller models; see [packaging/licenses](packaging/licenses/THIRD-PARTY.txt).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in FramePlayerX by you shall be licensed under the Apache License, Version 2.0, without any additional terms or conditions.

---

## FAQ

### Does it run on other headsets?

No. FramePlayerX is built only for the Steam Frame, so it can use the Frame's decoder, controllers and runtime directly instead of carrying everything a cross-platform player needs.

---

### Why does a video stutter?

Usually because it's 10-bit HEVC, 4:2:2 or AV1, which the Frame's hardware decoder can't play, so it's decoded on the CPU. The control bar says so. Converting it to 8-bit HEVC once on a PC makes it play smoothly; the [user guide](docs/USER-GUIDE.md#videos-that-stutter-10-bit-hevc-and-av1) has the commands.

---

### Does it work with my media server or NAS?

Yes, if it offers one of these: Jellyfin or Emby (sign in with Quick Connect or a password), a DeoVR feed or HereSphere API, SMB, WebDAV or DLNA. Add it under Sources › Add source.

---

### Is FramePlayerX made by Valve?

No. It's an independent project and isn't affiliated with or endorsed by Valve. Steam and Steam Frame are trademarks of Valve Corporation.

---

### Do you plan to add new features?

If I come across a new idea or receive a suggestion that fits a VR video player for the Frame, I will consider adding it, but there are no promises: this project exists to solve my own needs first.

---

## A Note on AI

This project was built with extensive help from [Claude Code](https://claude.com/claude-code).
