//! Eye tracking: where the eyes look, as one line of sight from between
//! them, through `XR_EXT_eye_gaze_interaction`, which is how the Steam
//! Frame offers it. What follows the gaze is up to the app.
//!
//! The gaze is an action set of its own, bound to the extension's
//! interaction profile. The session attaches and syncs it together with
//! the controllers' set: a set left out of an xrSyncActions goes inactive.

use crate::context::XrContext;
use crate::{Result, XrContextExt};
use glam::{Quat, Vec3};
use openxr as xr;
use std::cell::Cell;

const PROFILE: &str = "/interaction_profiles/ext/eye_gaze_interaction";
const GAZE_POSE: &str = "/user/eyes_ext/input/gaze_ext/pose";

/// Where the eyes look, in the play space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gaze {
    /// Between the eyes, metres.
    pub origin: Vec3,
    /// The line of sight, a unit vector.
    pub direction: Vec3,
}

impl Gaze {
    /// The gaze of a pose that looks down its -Z, as OpenXR's does.
    pub fn from_pose(position: Vec3, orientation: Quat) -> Gaze {
        Gaze {
            origin: position,
            direction: (orientation * Vec3::NEG_Z).normalize(),
        }
    }
}

pub(crate) struct EyeTracker {
    pub set: xr::ActionSet,
    gaze: xr::Action<xr::Posef>,
    space: Option<xr::Space>,
    /// A gaze has been tracked this session, so the log says so once.
    seen: Cell<bool>,
}

impl EyeTracker {
    /// The tracker, when the runtime and headset have eye tracking; the log
    /// says why not otherwise. Its bindings are suggested here, so this
    /// comes before the session attaches its action sets.
    pub fn new(ctx: &XrContext) -> Option<EyeTracker> {
        if !ctx.enabled.ext_eye_gaze_interaction {
            log::info!("eye tracking: the runtime lacks XR_EXT_eye_gaze_interaction");
            return None;
        }
        match supported(&ctx.instance, ctx.system) {
            Ok(true) => {}
            Ok(false) => {
                log::info!("eye tracking: this headset has none");
                return None;
            }
            Err(e) => {
                log::warn!("eye tracking: xrGetSystemProperties: {e}");
                return None;
            }
        }
        match EyeTracker::create(&ctx.instance) {
            Ok(t) => {
                log::info!("eye tracking: offered (it reports once eye data is allowed)");
                Some(t)
            }
            Err(e) => {
                log::warn!("eye tracking: {e}");
                None
            }
        }
    }

    fn create(instance: &xr::Instance) -> Result<EyeTracker> {
        let set = instance
            .create_action_set("eyes", "Eye tracking", 0)
            .ctx("action set")?;
        let gaze = set
            .create_action::<xr::Posef>("gaze", "Where you look", &[])
            .ctx("action")?;
        let profile = instance.string_to_path(PROFILE).ctx("path")?;
        let pose = instance.string_to_path(GAZE_POSE).ctx("path")?;
        instance
            .suggest_interaction_profile_bindings(profile, &[xr::Binding::new(&gaze, pose)])
            .ctx("xrSuggestInteractionProfileBindings")?;
        Ok(EyeTracker {
            set,
            gaze,
            space: None,
            seen: Cell::new(false),
        })
    }

    /// Makes the gaze's space, once the session has attached [`Self::set`].
    pub fn attach<G>(&mut self, session: &xr::Session<G>) -> Result<()> {
        let space = self
            .gaze
            .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)
            .ctx("gaze space")?;
        self.space = Some(space);
        Ok(())
    }

    /// Where the eyes look at `time`, after this frame's xrSyncActions;
    /// `None` while they aren't tracked: closed, eye data not allowed, or
    /// the headset off.
    pub fn locate<G>(
        &self,
        session: &xr::Session<G>,
        base: &xr::Space,
        time: xr::Time,
    ) -> Option<Gaze> {
        let space = self.space.as_ref()?;
        if !self.gaze.is_active(session, xr::Path::NULL).ok()? {
            return None;
        }
        let loc = space.locate(base, time).ok()?;
        // Tracked, not only valid: a runtime may go on giving the last gaze,
        // or the head's forward, while the eyes are lost.
        let tracked = xr::SpaceLocationFlags::POSITION_VALID
            | xr::SpaceLocationFlags::ORIENTATION_VALID
            | xr::SpaceLocationFlags::ORIENTATION_TRACKED;
        if !loc.location_flags.contains(tracked) {
            return None;
        }
        if !self.seen.replace(true) {
            log::info!("eye tracking: tracking");
        }
        let (position, orientation) = crate::pose(loc.pose);
        Some(Gaze::from_pose(position, orientation))
    }
}

/// Whether the system can track the eyes
/// (`XrSystemEyeGazeInteractionPropertiesEXT`).
fn supported(
    instance: &xr::Instance,
    system: xr::SystemId,
) -> std::result::Result<bool, xr::sys::Result> {
    // SAFETY: the property structs are chained as xrGetSystemProperties
    // expects and outlive the call.
    unsafe {
        let mut eye = xr::sys::SystemEyeGazeInteractionPropertiesEXT {
            ty: xr::sys::SystemEyeGazeInteractionPropertiesEXT::TYPE,
            next: std::ptr::null_mut(),
            supports_eye_gaze_interaction: xr::sys::FALSE,
        };
        let mut p = xr::sys::SystemProperties {
            ty: xr::sys::SystemProperties::TYPE,
            next: &mut eye as *mut _ as *mut _,
            ..std::mem::zeroed()
        };
        let r = (instance.fp().get_system_properties)(instance.as_raw(), system, &mut p);
        if r.into_raw() < 0 {
            return Err(r);
        }
        Ok(eye.supports_eye_gaze_interaction != xr::sys::FALSE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pose_looks_down_its_minus_z() {
        let turned = Quat::from_rotation_y(90f32.to_radians());
        let g = Gaze::from_pose(Vec3::new(0.0, 1.6, 0.0), turned);
        assert_eq!(g.origin, Vec3::new(0.0, 1.6, 0.0));
        assert!(g.direction.distance(Vec3::NEG_X) < 1e-6, "{g:?}");
    }
}
