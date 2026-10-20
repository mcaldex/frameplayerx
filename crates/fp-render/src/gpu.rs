//! Vulkan instance and device.

use crate::{Error, Result, VkContext};
use ash::vk;
use fp_core::sync::lock;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::ffi::CString;
use std::mem::ManuallyDrop;
use std::sync::Mutex;

/// FramePlayerX needs Vulkan 1.3 (dynamic rendering, synchronization2).
pub const API_VERSION: u32 = vk::API_VERSION_1_3;

/// Creates the instance and device. OpenXR runtimes need to do this
/// themselves (XR_KHR_vulkan_enable2), so creation is pluggable.
pub trait Creator {
    fn create_instance(
        &self,
        entry: &ash::Entry,
        info: &vk::InstanceCreateInfo,
    ) -> Result<vk::Instance>;
    fn physical_device(&self, instance: &ash::Instance) -> Result<vk::PhysicalDevice>;
    fn create_device(
        &self,
        instance: &ash::Instance,
        pdev: vk::PhysicalDevice,
        info: &vk::DeviceCreateInfo,
    ) -> Result<vk::Device>;
}

/// Plain Vulkan: first device that supports Vulkan 1.3 with a graphics queue.
pub struct DirectCreator;

impl Creator for DirectCreator {
    fn create_instance(
        &self,
        entry: &ash::Entry,
        info: &vk::InstanceCreateInfo,
    ) -> Result<vk::Instance> {
        // SAFETY: valid create info.
        unsafe { entry.create_instance(info, None) }
            .map(|i| i.handle())
            .ctx("vkCreateInstance")
    }
    fn physical_device(&self, instance: &ash::Instance) -> Result<vk::PhysicalDevice> {
        // SAFETY: valid instance.
        let devices = unsafe { instance.enumerate_physical_devices() }.ctx("enumerate devices")?;
        devices
            .into_iter()
            .find(|&d| {
                // SAFETY: valid physical device.
                let p = unsafe { instance.get_physical_device_properties(d) };
                p.api_version >= API_VERSION && graphics_queue(instance, d).is_some()
            })
            .ok_or_else(|| Error::Unsupported("no Vulkan 1.3 device with a graphics queue".into()))
    }
    fn create_device(
        &self,
        instance: &ash::Instance,
        pdev: vk::PhysicalDevice,
        info: &vk::DeviceCreateInfo,
    ) -> Result<vk::Device> {
        // SAFETY: valid create info for this physical device.
        unsafe { instance.create_device(pdev, info, None) }
            .map(|d| d.handle())
            .ctx("vkCreateDevice")
    }
}

pub(crate) fn graphics_queue(instance: &ash::Instance, pdev: vk::PhysicalDevice) -> Option<u32> {
    // SAFETY: valid physical device.
    unsafe { instance.get_physical_device_queue_family_properties(pdev) }
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|i| i as u32)
}

pub struct Gpu {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub pdev: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub allocator: ManuallyDrop<Mutex<Allocator>>,
    pub device_name: String,
    pub driver: String,
    /// Turnip with what [`crate::q10c`] needs enabled: descriptor buffers,
    /// buffer device addresses, DRM format modifiers and 16-bit storage
    /// images.
    pub(crate) q10c_features: bool,
    /// DMA-BUF import is enabled, for [`crate::frames`].
    pub(crate) dma_buf: bool,
    /// The device has one queue, which the decoder thread's copies, the
    /// renderer and the OpenXR runtime share (see [`Gpu::lock_queue`]).
    queue_lock: Mutex<()>,
}

/// Device extensions [`crate::q10c`] needs.
const Q10C_EXTENSIONS: [&std::ffi::CStr; 2] = [
    ash::ext::image_drm_format_modifier::NAME,
    ash::ext::descriptor_buffer::NAME,
];

/// Device extensions [`crate::frames`] needs: importing the decoder's
/// buffers, and handing them back and forth with it.
const DMA_BUF_EXTENSIONS: [&std::ffi::CStr; 3] = [
    ash::khr::external_memory_fd::NAME,
    ash::ext::external_memory_dma_buf::NAME,
    ash::ext::queue_family_foreign::NAME,
];

/// Whether `pdev` is Turnip with everything [`crate::q10c`] needs, given
/// its extensions `has`.
fn q10c_supported(
    instance: &ash::Instance,
    pdev: vk::PhysicalDevice,
    driver: vk::DriverId,
    has: impl Fn(&std::ffi::CStr) -> bool,
) -> bool {
    if driver != vk::DriverId::MESA_TURNIP || !Q10C_EXTENSIONS.iter().all(|n| has(n)) {
        return false;
    }
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut fdb = vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f12)
        .push_next(&mut fdb);
    // SAFETY: valid physical device; the chain holds initialised structs.
    unsafe { instance.get_physical_device_features2(pdev, &mut f2) };
    let storage_formats = f2.features.shader_storage_image_extended_formats;
    f12.buffer_device_address == vk::TRUE
        && fdb.descriptor_buffer == vk::TRUE
        && storage_formats == vk::TRUE
}

impl Gpu {
    pub fn new(creator: &dyn Creator, app_name: &str) -> Result<Gpu> {
        // SAFETY: loads the system Vulkan loader.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| Error::Load(e.to_string()))?;
        let name = CString::new(app_name).unwrap_or_default();
        let app = vk::ApplicationInfo::default()
            .application_name(&name)
            .engine_name(&name)
            .api_version(API_VERSION);
        let ici = vk::InstanceCreateInfo::default().application_info(&app);
        let raw = creator.create_instance(&entry, &ici)?;
        // SAFETY: raw is a live instance created with this entry.
        let instance = unsafe { ash::Instance::load(entry.static_fn(), raw) };
        let pdev = creator.physical_device(&instance)?;
        let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut driver_props);
        // SAFETY: valid physical device.
        unsafe { instance.get_physical_device_properties2(pdev, &mut props2) };
        let props = props2.properties;
        if props.api_version < API_VERSION {
            return Err(Error::Unsupported("GPU does not support Vulkan 1.3".into()));
        }
        let device_name = props
            .device_name_as_c_str()
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_default();
        let driver = format!(
            "{} {}",
            driver_props
                .driver_name_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default(),
            driver_props
                .driver_info_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default()
        );
        let queue_family = graphics_queue(&instance, pdev)
            .ok_or_else(|| Error::Unsupported("no graphics queue".into()))?;
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&prio)];
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        // SAFETY: valid physical device.
        let available =
            unsafe { instance.enumerate_device_extension_properties(pdev) }.unwrap_or_default();
        let has = |n: &std::ffi::CStr| {
            available
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(n))
        };
        let q10c_features = q10c_supported(&instance, pdev, driver_props.driver_id, has);
        let dma_buf = DMA_BUF_EXTENSIONS.iter().all(|n| has(n));
        let mut exts = Vec::new();
        if q10c_features {
            exts.extend(Q10C_EXTENSIONS.map(std::ffi::CStr::as_ptr));
        }
        if dma_buf {
            exts.extend(DMA_BUF_EXTENSIONS.map(std::ffi::CStr::as_ptr));
        }
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(true);
        let mut fdb =
            vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);
        let features =
            vk::PhysicalDeviceFeatures::default().shader_storage_image_extended_formats(true);
        let mut dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&exts)
            .push_next(&mut f13);
        if q10c_features {
            dci = dci
                .enabled_features(&features)
                .push_next(&mut f12)
                .push_next(&mut fdb);
        }
        let raw_dev = creator.create_device(&instance, pdev, &dci)?;
        // SAFETY: raw_dev was created from this instance.
        let device = unsafe { ash::Device::load(instance.fp_v1_0(), raw_dev) };
        // SAFETY: queue 0 of the family was requested.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: pdev,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| Error::Alloc(e.to_string()))?;
        log::info!("GPU: {device_name} ({driver})");
        Ok(Gpu {
            entry,
            instance,
            pdev,
            device,
            queue,
            queue_family,
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            device_name,
            driver,
            q10c_features,
            dma_buf,
            queue_lock: Mutex::new(()),
        })
    }

    /// Holds the queue: while held, nothing else submits to it. The OpenXR
    /// runtime uses the queue in xrBeginFrame, xrEndFrame and when swapchain
    /// images are acquired or released, so those are called holding it.
    pub fn lock_queue(&self) -> std::sync::MutexGuard<'_, ()> {
        lock(&self.queue_lock)
    }

    /// `vkQueueSubmit`, holding the queue.
    pub fn submit(&self, submits: &[vk::SubmitInfo<'_>], fence: vk::Fence) -> Result<()> {
        let _queue = self.lock_queue();
        // SAFETY: the queue is held; the submissions are the caller's.
        unsafe { self.device.queue_submit(self.queue, submits, fence) }.ctx("submit")
    }

    /// Headless device for tests and tools.
    pub fn headless() -> Result<Gpu> {
        Gpu::new(&DirectCreator, "frameplayerx")
    }

    pub(crate) fn alloc(&self) -> std::sync::MutexGuard<'_, Allocator> {
        lock(&self.allocator)
    }

    pub fn wait_idle(&self) {
        // SAFETY: valid device.
        let _ = unsafe { self.device.device_wait_idle() };
    }

    pub(crate) fn shader(&self, spv: &[u8]) -> Result<vk::ShaderModule> {
        let words: Vec<u32> = spv
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect();
        // SAFETY: valid SPIR-V produced by naga at build time.
        unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .ctx("create shader")
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.wait_idle();
        // SAFETY: all resources were destroyed by their owners; the
        // allocator must go before the device, the device before the instance.
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
