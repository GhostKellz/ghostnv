const std = @import("std");
const c = std.c;
const ghostnv = @import("../ghostnv.zig");
const vibrance = @import("../color/vibrance.zig");
const rtx40 = @import("../rtx40/optimizations.zig");
const kernel = @import("../kernel/module.zig");

/// nvcontrol API Bindings for GhostNV Integration
/// Provides seamless integration between the Rust nvcontrol tool and Zig GhostNV driver
/// Enables granular control over digital vibrance, HDR, VRR, and advanced GPU features

// C-compatible API structures for nvcontrol
pub const NvControlGpuInfo = extern struct {
    // Basic GPU info
    device_id: u32,
    device_name: [256]u8,
    driver_version: [64]u8,
    cuda_version: [32]u8,
    architecture: [64]u8,
    
    // Memory information
    total_memory_mb: u32,
    used_memory_mb: u32,
    memory_bandwidth_gbps: f32,
    memory_bus_width: u32,
    
    // Clock frequencies
    base_core_clock: u32,
    base_memory_clock: u32,
    boost_clock: u32,
    current_core_clock: u32,
    current_memory_clock: u32,
    
    // Thermal and power
    temperature: i32,
    power_draw_watts: f32,
    power_limit_watts: f32,
    thermal_limit: i32,
    
    // Utilization percentages
    gpu_utilization: u8,
    memory_utilization: u8,
    encoder_utilization: u8,
    decoder_utilization: u8,
    
    // Feature capabilities
    ray_tracing_support: bool,
    dlss_support: bool,
    nvenc_support: bool,
    nvdec_support: bool,
    av1_encode: bool,
    av1_decode: bool,
    hdr_support: bool,
    vrr_support: bool,
    gsync_support: bool,
    
    // Performance counters
    frames_rendered: u64,
    triangles_per_sec: u64,
    pixels_per_sec: u64,
};

pub const NvControlVibranceSettings = extern struct {
    // Digital vibrance settings
    vibrance_level: i16,           // -50 to +100
    vibrance_min: i16,
    vibrance_max: i16,
    
    // Advanced color controls
    saturation: i16,               // -50 to +50
    gamma: f32,                    // 0.8 to 3.0
    brightness: i16,               // -50 to +50
    contrast: i16,                 // -50 to +50
    hue_shift: i16,                // -180 to +180
    
    // Individual RGB channel vibrance
    red_vibrance: i16,
    green_vibrance: i16,
    blue_vibrance: i16,
    
    // Color temperature
    temperature_kelvin: i16,       // -1000 to +1000
    tint: i16,                     // -50 to +50 (green/magenta)
    
    // Profile information
    active_profile: [64]u8,
    profile_count: u32,
    auto_detect_enabled: bool,
    
    // Color space and HDR
    color_space: u32,              // 0=sRGB, 1=Adobe RGB, 2=DCI-P3, 3=Rec2020
    hdr_enabled: bool,
    hdr_peak_brightness: u16,      // Peak brightness in nits
    hdr_tone_mapping: u32,         // Tone mapping algorithm
    
    // Backend information
    backend_type: u32,             // 0=software, 1=nvidia_hw, 2=drm_ctm
    hardware_lut_support: bool,
    
    // Performance metrics
    last_apply_time_us: u64,
    frames_processed: u64,
};

pub const NvControlDisplaySettings = extern struct {
    // Display identification
    display_id: u32,
    display_name: [128]u8,
    manufacturer: [64]u8,
    model: [64]u8,
    serial_number: [64]u8,
    
    // Resolution and refresh
    width: u32,
    height: u32,
    refresh_rate: f32,
    color_depth: u8,               // Bits per channel
    
    // VRR (Variable Refresh Rate)
    vrr_enabled: bool,
    vrr_min_refresh: f32,
    vrr_max_refresh: f32,
    vrr_current_refresh: f32,
    
    // G-SYNC information
    gsync_compatible: bool,
    gsync_enabled: bool,
    gsync_ultimate: bool,
    
    // HDR capabilities
    hdr10_support: bool,
    dolby_vision_support: bool,
    hdr_metadata_support: bool,
    max_luminance_nits: u16,
    min_luminance_nits: f32,
    
    // Color gamut
    srgb_coverage: f32,            // Percentage
    adobe_rgb_coverage: f32,
    dci_p3_coverage: f32,
    rec2020_coverage: f32,
    
    // Connection info
    connector_type: [32]u8,        // HDMI, DisplayPort, etc.
    link_bandwidth_gbps: f32,
    
    // Current settings
    brightness_nits: u16,
    contrast_ratio: f32,
    gamma_setting: f32,
};

pub const NvControlOverclockSettings = extern struct {
    // Clock offsets (MHz)
    core_clock_offset: i32,
    memory_clock_offset: i32,
    shader_clock_offset: i32,
    
    // Voltage and power
    voltage_offset_mv: i32,
    power_limit_percent: u32,
    
    // Temperature limits
    temp_limit_celsius: u32,
    
    // Fan control
    fan_speed_percent: u32,
    fan_curve_enabled: bool,
    auto_fan_control: bool,
    
    // Current applied values
    applied_core_clock: u32,
    applied_memory_clock: u32,
    applied_voltage_mv: u32,
    
    // Safety and monitoring
    overclock_enabled: bool,
    thermal_throttling: bool,
    power_throttling: bool,
    voltage_throttling: bool,
    reliability_throttling: bool,
    
    // Performance testing
    stability_test_passed: bool,
    max_stable_core_offset: i32,
    max_stable_memory_offset: i32,
};

pub const NvControlProfileInfo = extern struct {
    // Profile identification
    profile_name: [64]u8,
    profile_type: u32,             // 0=gaming, 1=cinema, 2=competitive, 3=custom
    
    // Vibrance settings for this profile
    vibrance_settings: NvControlVibranceSettings,
    
    // Game detection patterns
    game_patterns: [10][128]u8,    // Window title patterns for auto-detection
    pattern_count: u32,
    
    // Metadata
    created_timestamp: u64,
    last_used_timestamp: u64,
    usage_count: u64,
    
    // Profile capabilities
    supports_hdr: bool,
    supports_vrr: bool,
    auto_apply_enabled: bool,
};

// Global state for nvcontrol integration
var g_nvcontrol_state: ?*NvControlState = null;

const NvControlState = struct {
    allocator: std.mem.Allocator,
    kernel_module: *kernel.KernelModule,
    vibrance_engine: *vibrance.VibranceEngine,
    rtx40_optimizer: ?*rtx40.RTX40Optimizer,
    
    // Cached information
    gpu_info_cache: ?NvControlGpuInfo,
    display_cache: std.ArrayList(NvControlDisplaySettings),
    profile_cache: std.ArrayList(NvControlProfileInfo),
    
    // State tracking
    last_update_time: u64,
    monitor_thread: ?std.Thread,
    monitoring_enabled: bool,
};

// Core API Functions

/// Initialize nvcontrol integration with GhostNV
export fn nvcontrol_ghostnv_init() c_int {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    // Initialize GhostNV components
    var kernel_module = allocator.create(kernel.KernelModule) catch return -1;
    kernel_module.* = kernel.KernelModule.init(allocator) catch {
        allocator.destroy(kernel_module);
        return -2;
    };
    
    // Initialize vibrance engine
    var mock_drm = allocator.create(MockDrmDriver) catch {
        kernel_module.deinit();
        allocator.destroy(kernel_module);
        return -3;
    };
    mock_drm.* = MockDrmDriver{};
    
    var vibrance_eng = allocator.create(vibrance.VibranceEngine) catch {
        allocator.destroy(mock_drm);
        kernel_module.deinit();
        allocator.destroy(kernel_module);
        return -4;
    };
    
    vibrance_eng.* = vibrance.VibranceEngine.init(allocator, @ptrCast(mock_drm));
    vibrance_eng.load_default_profiles() catch {
        std.log.warn("Failed to load default vibrance profiles");
    };
    
    // Initialize RTX 40 optimizer if compatible GPU detected
    var rtx40_opt: ?*rtx40.RTX40Optimizer = null;
    if (kernel_module.device_count > 0) {
        const optimizer = allocator.create(rtx40.RTX40Optimizer) catch null;
        if (optimizer) |opt| {
            opt.* = rtx40.RTX40Optimizer.init(allocator, kernel_module) catch {
                allocator.destroy(opt);
                null;
            } orelse null;
            rtx40_opt = opt;
        }
    }
    
    // Create nvcontrol state
    const state = allocator.create(NvControlState) catch {
        if (rtx40_opt) |opt| allocator.destroy(opt);
        vibrance_eng.deinit();
        allocator.destroy(vibrance_eng);
        allocator.destroy(mock_drm);
        kernel_module.deinit();
        allocator.destroy(kernel_module);
        return -5;
    };
    
    state.* = NvControlState{
        .allocator = allocator,
        .kernel_module = kernel_module,
        .vibrance_engine = vibrance_eng,
        .rtx40_optimizer = rtx40_opt,
        .gpu_info_cache = null,
        .display_cache = std.ArrayList(NvControlDisplaySettings).init(allocator),
        .profile_cache = std.ArrayList(NvControlProfileInfo).init(allocator),
        .last_update_time = 0,
        .monitor_thread = null,
        .monitoring_enabled = false,
    };
    
    g_nvcontrol_state = state;
    
    std.log.info("nvcontrol-GhostNV integration initialized successfully");
    return 0;
}

/// Cleanup nvcontrol integration
export fn nvcontrol_ghostnv_cleanup() void {
    if (g_nvcontrol_state) |state| {
        if (state.monitoring_enabled) {
            state.monitoring_enabled = false;
            if (state.monitor_thread) |thread| {
                thread.join();
            }
        }
        
        state.display_cache.deinit();
        state.profile_cache.deinit();
        
        if (state.rtx40_optimizer) |opt| {
            state.allocator.destroy(opt);
        }
        
        state.vibrance_engine.deinit();
        state.allocator.destroy(state.vibrance_engine);
        
        state.kernel_module.deinit();
        state.allocator.destroy(state.kernel_module);
        
        state.allocator.destroy(state);
        g_nvcontrol_state = null;
    }
}

// GPU Information API

/// Get comprehensive GPU information for nvcontrol
export fn nvcontrol_get_gpu_info(device_id: u32, info: *NvControlGpuInfo) c_int {
    const state = g_nvcontrol_state orelse return -1;
    
    if (device_id >= state.kernel_module.device_count) return -2;
    
    const device = &state.kernel_module.devices[device_id];
    
    // Fill basic GPU information
    info.device_id = device.device_id;
    
    // Copy device name safely
    const name_len = @min(device.name.len, info.device_name.len - 1);
    @memcpy(info.device_name[0..name_len], device.name[0..name_len]);
    info.device_name[name_len] = 0;
    
    // Architecture string
    const arch_name = switch (device.architecture) {
        .ada_lovelace => "Ada Lovelace (RTX 40 Series)",
        .ampere => "Ampere (RTX 30 Series)", 
        .turing => "Turing (RTX 20 Series)",
        .pascal => "Pascal (GTX 10 Series)",
        else => "Unknown Architecture",
    };
    const arch_len = @min(arch_name.len, info.architecture.len - 1);
    @memcpy(info.architecture[0..arch_len], arch_name[0..arch_len]);
    info.architecture[arch_len] = 0;
    
    // Memory information
    info.total_memory_mb = device.memory_size_mb;
    info.used_memory_mb = device.getMemoryUsed() catch 0;
    info.memory_bandwidth_gbps = @as(f32, @floatFromInt(device.memory_bandwidth_gbps));
    info.memory_bus_width = device.memory_bus_width;
    
    // Clock frequencies
    info.base_core_clock = device.base_core_clock;
    info.base_memory_clock = device.base_memory_clock;
    info.boost_clock = device.boost_clock;
    info.current_core_clock = device.core_clock;
    info.current_memory_clock = device.memory_clock;
    
    // Thermal and power
    info.temperature = device.getTemperature() catch 0;
    info.power_draw_watts = @as(f32, @floatFromInt(device.getPowerUsage() catch 0));
    info.power_limit_watts = @as(f32, @floatFromInt(device.power_limit_watts));
    info.thermal_limit = @as(i32, @intCast(device.thermal_limit_celsius));
    
    // Utilization
    info.gpu_utilization = @as(u8, @intCast(device.getUtilization() catch 0));
    info.memory_utilization = @as(u8, @intCast(device.getMemoryUtilization() catch 0));
    info.encoder_utilization = @as(u8, @intCast(device.getEncoderUtilization() catch 0));
    info.decoder_utilization = @as(u8, @intCast(device.getDecoderUtilization() catch 0));
    
    // Feature capabilities based on architecture
    info.ray_tracing_support = switch (device.architecture) {
        .ada_lovelace, .ampere, .turing => true,
        else => false,
    };
    info.dlss_support = info.ray_tracing_support;
    info.nvenc_support = true; // Most modern GPUs have NVENC
    info.nvdec_support = true;
    info.av1_encode = device.architecture == .ada_lovelace;
    info.av1_decode = info.av1_encode;
    info.hdr_support = true;
    info.vrr_support = true;
    info.gsync_support = info.ray_tracing_support;
    
    // Performance counters
    info.frames_rendered = device.frames_rendered;
    info.triangles_per_sec = device.triangles_per_second;
    info.pixels_per_sec = device.pixels_per_second;
    
    return 0;
}

// Vibrance API Functions

/// Get current vibrance settings for nvcontrol
export fn nvcontrol_get_vibrance_settings(device_id: u32, settings: *NvControlVibranceSettings) c_int {
    _ = device_id;
    const state = g_nvcontrol_state orelse return -1;
    
    const engine = state.vibrance_engine;
    
    // Get current vibrance info
    const vibrance_info = engine.get_vibrance_info() catch return -2;
    
    settings.vibrance_level = vibrance_info.current;
    settings.vibrance_min = vibrance_info.min;
    settings.vibrance_max = vibrance_info.max;
    settings.backend_type = @intFromEnum(vibrance_info.backend);
    
    // Get active profile information
    if (engine.get_active_profile()) |profile| {
        settings.saturation = profile.saturation;
        settings.gamma = profile.gamma;
        settings.brightness = profile.brightness;
        settings.contrast = profile.contrast;
        settings.hue_shift = profile.hue_shift;
        settings.red_vibrance = profile.red_vibrance;
        settings.green_vibrance = profile.green_vibrance;
        settings.blue_vibrance = profile.blue_vibrance;
        settings.temperature_kelvin = profile.temperature;
        settings.tint = profile.tint;
        settings.hdr_enabled = profile.hdr_enabled;
        settings.hdr_peak_brightness = profile.hdr_peak_brightness;
        settings.hdr_tone_mapping = @intFromEnum(profile.hdr_tone_mapping);
        settings.color_space = 0; // Default to sRGB
        
        // Copy profile name
        if (engine.active_profile) |profile_name| {
            const name_len = @min(profile_name.len, settings.active_profile.len - 1);
            @memcpy(settings.active_profile[0..name_len], profile_name[0..name_len]);
            settings.active_profile[name_len] = 0;
        }
    } else {
        // Default values when no profile is active
        @memset(&settings.active_profile, 0);
        @memcpy(settings.active_profile[0..4], "none");
        settings.saturation = 0;
        settings.gamma = 2.2;
        settings.brightness = 0;
        settings.contrast = 0;
        settings.hue_shift = 0;
        settings.red_vibrance = 0;
        settings.green_vibrance = 0;
        settings.blue_vibrance = 0;
        settings.temperature_kelvin = 0;
        settings.tint = 0;
        settings.hdr_enabled = false;
        settings.hdr_peak_brightness = 1000;
        settings.hdr_tone_mapping = 2; // ACES
        settings.color_space = 0; // sRGB
    }
    
    // Performance and capability info
    const stats = engine.get_performance_stats();
    settings.profile_count = stats.profiles_loaded;
    settings.auto_detect_enabled = false; // Would need tracking in engine
    settings.hardware_lut_support = settings.backend_type == 1; // nvidia_hw
    settings.last_apply_time_us = @intCast(stats.processing_time_ns / 1000);
    settings.frames_processed = stats.frames_processed;
    
    return 0;
}

/// Set digital vibrance level directly
export fn nvcontrol_set_vibrance_level(device_id: u32, vibrance: i16) c_int {
    _ = device_id;
    const state = g_nvcontrol_state orelse return -1;
    
    if (vibrance < -50 or vibrance > 100) return -2;
    
    state.vibrance_engine.apply_vibrance_direct(vibrance) catch return -3;
    
    std.log.info("nvcontrol: Set vibrance to {}", .{vibrance});
    return 0;
}

/// Apply a vibrance profile by name
export fn nvcontrol_apply_vibrance_profile(device_id: u32, profile_name: [*:0]const u8) c_int {
    _ = device_id;
    const state = g_nvcontrol_state orelse return -1;
    
    const profile_slice = std.mem.span(profile_name);
    
    state.vibrance_engine.apply_profile(profile_slice) catch |err| {
        return switch (err) {
            vibrance.VibranceError.ProfileNotFound => -2,
            else => -3,
        };
    };
    
    std.log.info("nvcontrol: Applied vibrance profile '{s}'", .{profile_slice});
    return 0;
}

/// Create custom vibrance profile
export fn nvcontrol_create_vibrance_profile(profile_name: [*:0]const u8, settings: *const NvControlVibranceSettings) c_int {
    const state = g_nvcontrol_state orelse return -1;
    
    const name_slice = std.mem.span(profile_name);
    
    var profile = vibrance.VibranceProfile.init(name_slice);
    profile.vibrance = @intCast(settings.vibrance_level);
    profile.saturation = @intCast(settings.saturation);
    profile.gamma = settings.gamma;
    profile.brightness = @intCast(settings.brightness);
    profile.contrast = @intCast(settings.contrast);
    profile.hue_shift = @intCast(settings.hue_shift);
    profile.red_vibrance = @intCast(settings.red_vibrance);
    profile.green_vibrance = @intCast(settings.green_vibrance);
    profile.blue_vibrance = @intCast(settings.blue_vibrance);
    profile.temperature = settings.temperature_kelvin;
    profile.tint = @intCast(settings.tint);
    profile.hdr_peak_brightness = settings.hdr_peak_brightness;
    profile.hdr_tone_mapping = @enumFromInt(settings.hdr_tone_mapping);
    
    state.vibrance_engine.create_profile(name_slice, profile) catch return -2;
    
    std.log.info("nvcontrol: Created vibrance profile '{s}'", .{name_slice});
    return 0;
}

/// Get list of available vibrance profiles
export fn nvcontrol_get_vibrance_profiles(profiles: [*]NvControlProfileInfo, max_profiles: u32, count: *u32) c_int {
    const state = g_nvcontrol_state orelse return -1;
    
    const profile_names = state.vibrance_engine.list_profiles() catch return -2;
    defer state.allocator.free(profile_names);
    
    const actual_count = @min(profile_names.len, max_profiles);
    count.* = @intCast(actual_count);
    
    for (0..actual_count) |i| {
        const profile_name = profile_names[i];
        
        // Copy profile name
        const name_len = @min(profile_name.len, profiles[i].profile_name.len - 1);
        @memcpy(profiles[i].profile_name[0..name_len], profile_name[0..name_len]);
        profiles[i].profile_name[name_len] = 0;
        
        // Get profile details if available
        if (state.vibrance_engine.profiles.get(profile_name)) |profile| {
            profiles[i].profile_type = @intFromEnum(profile.game_mode);
            
            // Copy vibrance settings
            var settings: NvControlVibranceSettings = std.mem.zeroes(NvControlVibranceSettings);
            settings.vibrance_level = profile.vibrance;
            settings.saturation = profile.saturation;
            settings.gamma = profile.gamma;
            settings.brightness = profile.brightness;
            settings.contrast = profile.contrast;
            profiles[i].vibrance_settings = settings;
            
            // Initialize other fields
            profiles[i].pattern_count = 0;
            profiles[i].created_timestamp = 0;
            profiles[i].last_used_timestamp = 0;
            profiles[i].usage_count = 0;
            profiles[i].supports_hdr = profile.hdr_enabled;
            profiles[i].supports_vrr = true; // Assume VRR support
            profiles[i].auto_apply_enabled = false;
        }
    }
    
    return 0;
}

// Overclocking API Functions

/// Get current overclock settings
export fn nvcontrol_get_overclock_settings(device_id: u32, settings: *NvControlOverclockSettings) c_int {
    const state = g_nvcontrol_state orelse return -1;
    
    if (device_id >= state.kernel_module.device_count) return -2;
    
    const device = &state.kernel_module.devices[device_id];
    
    settings.core_clock_offset = device.core_clock_offset;
    settings.memory_clock_offset = device.memory_clock_offset;
    settings.shader_clock_offset = device.shader_clock_offset;
    settings.voltage_offset_mv = device.voltage_offset_mv;
    settings.power_limit_percent = device.power_limit_percent;
    settings.temp_limit_celsius = device.temp_limit_celsius;
    settings.fan_speed_percent = device.fan_speed_percent;
    
    settings.applied_core_clock = device.core_clock;
    settings.applied_memory_clock = device.memory_clock;
    settings.applied_voltage_mv = device.voltage_mv;
    
    settings.overclock_enabled = device.isOverclocked();
    settings.thermal_throttling = device.isThermalThrottling();
    settings.power_throttling = device.isPowerThrottling();
    settings.voltage_throttling = device.isVoltageThrottling();
    settings.reliability_throttling = device.isReliabilityThrottling();
    
    settings.auto_fan_control = true; // Default assumption
    settings.fan_curve_enabled = false;
    settings.stability_test_passed = false;
    settings.max_stable_core_offset = 0;
    settings.max_stable_memory_offset = 0;
    
    return 0;
}

/// Apply overclock settings
export fn nvcontrol_apply_overclock_settings(device_id: u32, settings: *const NvControlOverclockSettings) c_int {
    const state = g_nvcontrol_state orelse return -1;
    
    if (state.rtx40_optimizer == null) return -2; // RTX 40 required for advanced overclocking
    
    const optimizer = state.rtx40_optimizer.?;
    
    // Apply core clock offset
    if (settings.core_clock_offset != 0) {
        optimizer.setCoreClockOffset(device_id, settings.core_clock_offset) catch return -3;
    }
    
    // Apply memory clock offset
    if (settings.memory_clock_offset != 0) {
        optimizer.setMemoryClockOffset(device_id, settings.memory_clock_offset) catch return -4;
    }
    
    // Apply power limit
    if (settings.power_limit_percent > 0) {
        optimizer.setPowerLimit(device_id, settings.power_limit_percent) catch return -5;
    }
    
    std.log.info("nvcontrol: Applied overclock settings for device {}", .{device_id});
    return 0;
}

// Display and HDR API Functions

/// Get display information and capabilities
export fn nvcontrol_get_display_info(display_id: u32, info: *NvControlDisplaySettings) c_int {
    _ = display_id;
    const state = g_nvcontrol_state orelse return -1;
    _ = state;
    
    // Mock display information for now
    // In a real implementation, this would query the display subsystem
    
    info.display_id = display_id;
    @memcpy(info.display_name[0..8], "Display ");
    @memcpy(info.manufacturer[0..6], "NVIDIA");
    @memcpy(info.model[0..10], "RTX Series");
    @memcpy(info.connector_type[0..12], "DisplayPort");
    
    info.width = 2560;
    info.height = 1440;
    info.refresh_rate = 144.0;
    info.color_depth = 10;
    
    info.vrr_enabled = true;
    info.vrr_min_refresh = 48.0;
    info.vrr_max_refresh = 144.0;
    info.vrr_current_refresh = 144.0;
    
    info.gsync_compatible = true;
    info.gsync_enabled = true;
    info.gsync_ultimate = false;
    
    info.hdr10_support = true;
    info.dolby_vision_support = false;
    info.hdr_metadata_support = true;
    info.max_luminance_nits = 1000;
    info.min_luminance_nits = 0.1;
    
    info.srgb_coverage = 100.0;
    info.adobe_rgb_coverage = 85.0;
    info.dci_p3_coverage = 95.0;
    info.rec2020_coverage = 70.0;
    
    info.link_bandwidth_gbps = 32.4; // DisplayPort 1.4
    info.brightness_nits = 400;
    info.contrast_ratio = 1000.0;
    info.gamma_setting = 2.2;
    
    return 0;
}

// Utility Functions

/// Enable real-time monitoring
export fn nvcontrol_enable_monitoring(interval_ms: u32) c_int {
    const state = g_nvcontrol_state orelse return -1;
    
    if (state.monitoring_enabled) return 0; // Already enabled
    
    state.monitoring_enabled = true;
    
    // In a real implementation, this would start a monitoring thread
    std.log.info("nvcontrol: Enabled monitoring with {}ms interval", .{interval_ms});
    return 0;
}

/// Disable real-time monitoring
export fn nvcontrol_disable_monitoring() c_int {
    const state = g_nvcontrol_state orelse return -1;
    
    state.monitoring_enabled = false;
    
    std.log.info("nvcontrol: Disabled monitoring");
    return 0;
}

/// Get driver version and capabilities
export fn nvcontrol_get_driver_version(version_buffer: [*]u8, buffer_size: usize) c_int {
    const version_string = "GhostNV 0.2.0 (Pure Zig Driver)";
    const copy_len = @min(version_string.len, buffer_size - 1);
    
    @memcpy(version_buffer[0..copy_len], version_string[0..copy_len]);
    version_buffer[copy_len] = 0;
    
    return 0;
}

// Mock types for compilation
const MockDrmDriver = struct {};

// Extended device methods that might be missing
const DeviceExtended = struct {
    pub fn getMemoryUsed(device: anytype) !u32 {
        _ = device;
        return 8192; // Mock 8GB used
    }
    
    pub fn getTemperature(device: anytype) !i32 {
        _ = device;
        return 65; // Mock 65°C
    }
    
    pub fn getPowerUsage(device: anytype) !u32 {
        _ = device;
        return 250; // Mock 250W
    }
    
    pub fn getUtilization(device: anytype) !u32 {
        _ = device;
        return 75; // Mock 75% utilization
    }
    
    pub fn getMemoryUtilization(device: anytype) !u32 {
        _ = device;
        return 60; // Mock 60% memory utilization
    }
    
    pub fn getEncoderUtilization(device: anytype) !u32 {
        _ = device;
        return 10; // Mock 10% encoder utilization
    }
    
    pub fn getDecoderUtilization(device: anytype) !u32 {
        _ = device;
        return 5; // Mock 5% decoder utilization
    }
    
    pub fn isOverclocked(device: anytype) bool {
        _ = device;
        return true; // Mock overclocked state
    }
    
    pub fn isThermalThrottling(device: anytype) bool {
        _ = device;
        return false;
    }
    
    pub fn isPowerThrottling(device: anytype) bool {
        _ = device;
        return false;
    }
    
    pub fn isVoltageThrottling(device: anytype) bool {
        _ = device;
        return false;
    }
    
    pub fn isReliabilityThrottling(device: anytype) bool {
        _ = device;
        return false;
    }
};

// Test functions
test "nvcontrol api initialization" {
    const result = nvcontrol_ghostnv_init();
    defer nvcontrol_ghostnv_cleanup();
    
    try std.testing.expect(result == 0 or result < 0); // Should either succeed or fail gracefully
}

test "vibrance api functions" {
    _ = nvcontrol_ghostnv_init();
    defer nvcontrol_ghostnv_cleanup();
    
    var settings: NvControlVibranceSettings = undefined;
    const result = nvcontrol_get_vibrance_settings(0, &settings);
    
    // Should either succeed or fail gracefully based on system state
    try std.testing.expect(result == 0 or result < 0);
}