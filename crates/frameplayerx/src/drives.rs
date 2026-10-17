//! microSD cards and USB drives: which are mounted, and mounting those
//! SteamOS leaves alone.

use std::path::PathBuf;

/// Where SteamOS (udisks) mounts microSD cards and USB drives:
/// `/run/media/deck/<label>` (older images: `/run/media/<device>`).
pub const REMOVABLE_ROOT: &str = "/run/media";

/// Mount points of removable drives, from a `/proc/mounts` listing.
pub fn parse_removable_mounts(mounts: &str) -> Vec<PathBuf> {
    let prefix = format!("{REMOVABLE_ROOT}/");
    let mut out: Vec<PathBuf> = mounts
        .lines()
        .filter_map(|l| l.split_whitespace().nth(1))
        .map(unescape_mount)
        .filter(|p| p.starts_with(&prefix))
        .map(PathBuf::from)
        .collect();
    out.sort();
    out.dedup();
    out
}

/// `/proc/mounts` writes spaces and a few other bytes as `\ooo` octal.
fn unescape_mount(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let octal = b[i] == b'\\'
            && i + 3 < b.len()
            && b[i + 1..i + 4].iter().all(|c| (b'0'..=b'7').contains(c));
        if octal {
            out.push((b[i + 1] - b'0') * 64 + (b[i + 2] - b'0') * 8 + (b[i + 3] - b'0'));
            i += 4;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Currently mounted microSD cards and USB drives.
pub fn removable_mounts() -> Vec<PathBuf> {
    std::fs::read_to_string("/proc/mounts")
        .map(|m| parse_removable_mounts(&m))
        .unwrap_or_default()
}

/// A drive partition SteamOS left unmounted that FramePlayerX mounts itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnmountedDrive {
    /// Kernel name, e.g. `mmcblk0p1`.
    pub device: String,
    pub fs_type: String,
}

/// Whether FramePlayerX should mount this partition. The Frame's SteamOS
/// automounts only ext4 microSD cards (Steam libraries need symlinks) and
/// no USB drives at all (`sd*` also names its internal OS slots), so a card
/// formatted on Windows is never mounted. `sys_path` is the resolved
/// `/sys/class/block/<device>` link; USB drives are told apart from the
/// internal UFS disks by it.
pub fn should_mount(device: &str, sys_path: &str, fs_type: &str) -> bool {
    const FOREIGN: [&str; 4] = ["exfat", "vfat", "ntfs", "ntfs3"];
    if device.starts_with("mmcblk") {
        // ext4 cards are SteamOS's: it checks them before mounting.
        FOREIGN.contains(&fs_type)
    } else {
        sys_path.contains("/usb") && (FOREIGN.contains(&fs_type) || fs_type == "ext4")
    }
}

/// `ID_FS_TYPE` from a udev database record (`/run/udev/data/b<maj>:<min>`).
pub fn udev_fs_type(record: &str) -> Option<&str> {
    record
        .lines()
        .find_map(|l| l.strip_prefix("E:ID_FS_TYPE="))
        .filter(|t| !t.is_empty())
}

/// microSD cards and USB drives with a readable filesystem that nothing
/// has mounted.
pub fn unmounted_drives() -> Vec<UnmountedDrive> {
    let mounted = std::fs::read_to_string("/proc/mounts").unwrap_or_default();
    let mounted: Vec<&str> = mounted
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .collect();
    let Ok(dir) = std::fs::read_dir("/sys/class/block") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in dir.flatten() {
        let device = e.file_name().to_string_lossy().into_owned();
        if !device.starts_with("mmcblk") && !device.starts_with("sd") {
            continue;
        }
        let sys_path = std::fs::canonicalize(e.path())
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !device.starts_with("mmcblk") && !sys_path.contains("/usb") {
            continue;
        }
        let Ok(devnum) = std::fs::read_to_string(e.path().join("dev")) else {
            continue;
        };
        let Ok(record) = std::fs::read_to_string(format!("/run/udev/data/b{}", devnum.trim()))
        else {
            continue;
        };
        let Some(fs_type) = udev_fs_type(&record) else {
            continue;
        };
        if should_mount(&device, &sys_path, fs_type)
            && !mounted.contains(&format!("/dev/{device}").as_str())
        {
            out.push(UnmountedDrive {
                fs_type: fs_type.to_string(),
                device,
            });
        }
    }
    out.sort_by(|a, b| a.device.cmp(&b.device));
    out
}

/// Mounts a drive through udisks, as SteamOS does, under
/// `/run/media/<user>/<label>`. The Frame's polkit rules let the logged-in
/// user do this without a password. Blocking: call from a job thread.
pub fn mount_drive(drive: &UnmountedDrive) -> Result<PathBuf, String> {
    let out = std::process::Command::new("udisksctl")
        .args(["mount", "--no-user-interaction", "-o", "noatime", "-b"])
        .arg(format!("/dev/{}", drive.device))
        .output()
        .map_err(|e| format!("udisksctl: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(err.trim().to_string());
    }
    // "Mounted /dev/mmcblk0p1 at /run/media/steamos/SDCARD"
    text.trim()
        .split_once(" at ")
        .map(|(_, p)| PathBuf::from(p.trim_end_matches('.')))
        .ok_or_else(|| format!("unexpected udisksctl output: {}", text.trim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removable_mounts_from_proc() {
        let m = "/dev/nvme0n1p8 /home ext4 rw 0 0\n\
                 /dev/mmcblk0p1 /run/media/deck/SD\\040Card ext4 rw,nosuid 0 0\n\
                 /dev/sda1 /run/media/deck/USB exfat rw 0 0\n\
                 /dev/mmcblk0p1 /run/media/mmcblk0p1 ext4 rw 0 0\n\
                 tmpfs /run/user/1000 tmpfs rw 0 0\n";
        assert_eq!(
            parse_removable_mounts(m),
            vec![
                PathBuf::from("/run/media/deck/SD Card"),
                PathBuf::from("/run/media/deck/USB"),
                PathBuf::from("/run/media/mmcblk0p1")
            ]
        );
    }

    #[test]
    fn mounts_what_steamos_leaves_alone() {
        let card = "/sys/devices/platform/soc@0/8804000.mmc/mmc_host/mmc0/mmc0:d555/block/mmcblk0/mmcblk0p1";
        let usb = "/sys/devices/platform/soc@0/a600000.usb/xhci-hcd.1.auto/usb1/1-1/1-1:1.0/host0/target0:0:0/0:0:0:0/block/sde/sde1";
        let ufs =
            "/sys/devices/platform/soc@0/1d84000.ufshc/host0/target0:0:0/0:0:0:0/block/sda/sda8";
        assert!(should_mount("mmcblk0p1", card, "exfat"));
        assert!(should_mount("mmcblk0p1", card, "vfat"));
        assert!(!should_mount("mmcblk0p1", card, "ext4"));
        assert!(should_mount("sde1", usb, "exfat"));
        assert!(should_mount("sde1", usb, "ext4"));
        assert!(!should_mount("sda8", ufs, "ext4"));
        assert!(!should_mount("sda1", ufs, "vfat"));
        assert!(!should_mount("sde1", usb, "swap"));
    }

    #[test]
    fn fs_type_from_udev_record() {
        let r = "S:disk/by-label/SDCARD\nE:ID_FS_LABEL=SDCARD\nE:ID_FS_TYPE=exfat\nE:ID_FS_USAGE=filesystem\n";
        assert_eq!(udev_fs_type(r), Some("exfat"));
        assert_eq!(udev_fs_type("E:ID_FS_TYPE=\n"), None);
        assert_eq!(udev_fs_type("E:ID_PART_TABLE_TYPE=dos\n"), None);
    }
}
