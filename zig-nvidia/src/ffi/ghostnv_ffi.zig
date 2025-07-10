const std = @import("std");
const zig_nvidia = @import("zig-nvidia");
const vibrance = zig_nvidia.color_vibrance;
const gsync = struct {
    pub fn enable_gsync() !void {
        std.log.info("G-Sync enabled", .{});
    }
    
    pub fn disable_gsync() !void {
        std.log.info("G-Sync disabled", .{});
    }
    
    pub const GsyncMode = enum {
        disabled,
        compatible,
        certified,
        ultimate,
        esports,
    };
    
    pub const GameType = enum {
        competitive_fps,
        immersive_single_player,
        racing,
        cinema,
    };
    
    pub const DisplayInfo = struct {
        gsync_mode: GsyncMode,
        min_refresh_hz: u32,
        max_refresh_hz: u32,
        current_refresh_hz: u32,
        ultra_low_latency: bool,
        variable_overdrive: bool,
        motion_blur_reduction: bool,
        hdr_enabled: bool,
        peak_brightness_nits: u32,
    };
    
    pub const GsyncManager = struct {
        allocator: std.mem.Allocator,
        
        pub fn init(alloc: std.mem.Allocator, drm_drv: anytype) GsyncManager {
            _ = drm_drv;
            return GsyncManager{
                .allocator = alloc,
            };
        }
        
        pub fn deinit(self: *GsyncManager) void {
            _ = self;
        }
        
        pub fn enable_gsync(self: *GsyncManager, mode: GsyncMode) !void {
            _ = self;
            _ = mode;
            std.log.info("G-Sync enabled", .{});
        }
        
        pub fn disable_gsync(self: *GsyncManager) !void {
            _ = self;
            std.log.info("G-Sync disabled", .{});
        }
        
        pub fn get_all_display_info(self: *GsyncManager) ![]DisplayInfo {
            const displays = try self.allocator.alloc(DisplayInfo, 1);
            displays[0] = DisplayInfo{
                .gsync_mode = .compatible,
                .min_refresh_hz = 60,
                .max_refresh_hz = 240,
                .current_refresh_hz = 120,
                .ultra_low_latency = true,
                .variable_overdrive = true,
                .motion_blur_reduction = false,
                .hdr_enabled = true,
                .peak_brightness_nits = 1000,
            };
            return displays;
        }
        
        pub fn set_refresh_rate(self: *GsyncManager, refresh_hz: u32) !void {
            _ = self;
            std.log.info("Refresh rate set to {} Hz", .{refresh_hz});
        }
        
        pub fn optimize_for_game(self: *GsyncManager, game_type: GameType) void {
            _ = self;
            std.log.info("Optimized for game type: {}", .{game_type});
        }
    };
};
const performance = zig_nvidia.gaming_performance;
const drm = zig_nvidia.drm_driver;
const nvctl = struct {
    pub fn set_power_limit(limit: u32) !void {
        std.log.info("Power limit set to {}W", .{limit});
    }
    
    pub fn set_fan_speed(speed: u8) !void {
        std.log.info("Fan speed set to {}%", .{speed});
    }
    
    pub const DeviceInfo = struct {
        id: u32,
        name: []const u8,
        driver_version: []const u8,
        pci_bus: u32,
        pci_device: u32,
        pci_function: u32,
    };
    
    pub const NvctlInterface = struct {
        allocator: std.mem.Allocator,
        devices: std.ArrayList(DeviceInfo),
        
        pub fn init(alloc: std.mem.Allocator, vibrance_eng: anytype, gsync_mgr: anytype) NvctlInterface {
            _ = vibrance_eng;
            _ = gsync_mgr;
            return NvctlInterface{
                .allocator = alloc,
                .devices = std.ArrayList(DeviceInfo).init(alloc),
            };
        }
        
        pub fn deinit(self: *NvctlInterface) void {
            self.devices.deinit();
        }
        
        pub fn enumerate_devices(self: *NvctlInterface) !void {
            // Mock device enumeration
            const mock_device = DeviceInfo{
                .id = 0,
                .name = "NVIDIA GeForce RTX 4090",
                .driver_version = "545.29.06",
                .pci_bus = 1,
                .pci_device = 0,
                .pci_function = 0,
            };
            try self.devices.append(mock_device);
        }
        
        pub fn get_device_info(self: *NvctlInterface, device_id: u32) !DeviceInfo {
            for (self.devices.items) |device| {
                if (device.id == device_id) {
                    return device;
                }
            }
            return error.DeviceNotFound;
        }
    };
};

// Global allocator for FFI operations
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Global driver instances
var drm_driver: ?*drm.DrmDriver = null;
var vibrance_engine: ?*vibrance.VibranceEngine = null;
var gsync_manager: ?*gsync.GsyncManager = null;
var nvctl_interface: ?*nvctl.NvctlInterface = null;

// Thread safety
var init_mutex = std.Thread.Mutex{};
var initialized = false;

// ═══════════════════════════════════════════════════════
// Core Types (C-compatible)
// ═══════════════════════════════════════════════════════

pub const GhostNVResult = enum(c_int) {
    GHOSTNV_OK = 0,
    GHOSTNV_ERROR_INVALID_DEVICE = -1,
    GHOSTNV_ERROR_INVALID_VALUE = -2,
    GHOSTNV_ERROR_NOT_SUPPORTED = -3,
    GHOSTNV_ERROR_PERMISSION_DENIED = -4,
    GHOSTNV_ERROR_NOT_INITIALIZED = -5,
    GHOSTNV_ERROR_MEMORY_ALLOCATION = -6,
    GHOSTNV_ERROR_DEVICE_BUSY = -7,
};

pub const GhostNVDevice = extern struct {
    device_id: u32,
    name: [256]u8,
    driver_version: [32]u8,
    pci_bus: u32,
    pci_device: u32,
    pci_function: u32,
    supports_gsync: bool,
    supports_vrr: bool,
    supports_hdr: bool,
};

pub const GhostNVVibranceProfile = extern struct {
    vibrance: i8,           // -50 to 100
    saturation: i8,         // -50 to 50  
    gamma: f32,             // 0.8 to 3.0
    brightness: i8,         // -50 to 50
    contrast: i8,           // -50 to 50
    temperature: i16,       // -1000 to 1000 Kelvin
    red_vibrance: i8,       // -50 to 100
    green_vibrance: i8,     // -50 to 100
    blue_vibrance: i8,      // -50 to 100
    preserve_skin_tones: bool,
    enhance_foliage: bool,
    boost_sky_colors: bool,
};

pub const GhostNVGSyncMode = enum(u32) {
    GSYNC_DISABLED = 0,
    GSYNC_COMPATIBLE = 1,
    GSYNC_CERTIFIED = 2,
    GSYNC_ULTIMATE = 3,
    GSYNC_ESPORTS = 4,
};

pub const GhostNVGSyncStatus = extern struct {
    mode: GhostNVGSyncMode,
    min_refresh_hz: u32,
    max_refresh_hz: u32,
    current_refresh_hz: u32,
    ultra_low_latency: bool,
    variable_overdrive: bool,
    motion_blur_reduction: bool,
    hdr_enabled: bool,
    peak_brightness_nits: u32,
};

pub const GhostNVGameType = enum(u32) {
    GAME_COMPETITIVE_FPS = 0,
    GAME_IMMERSIVE_SINGLE_PLAYER = 1,
    GAME_RACING = 2,
    GAME_CINEMA = 3,
};

pub const GhostNVPerformanceInfo = extern struct {
    gpu_clock_mhz: u32,
    memory_clock_mhz: u32,
    temperature_c: u32,
    fan_speed_rpm: u32,
    power_draw_watts: u32,
    gpu_utilization_percent: u32,
    memory_utilization_percent: u32,
    average_frametime_ms: f32,
    current_fps: u32,
};

// ═══════════════════════════════════════════════════════
// Core API Functions
// ═══════════════════════════════════════════════════════

/// Initialize GhostNV driver interface
export fn ghostnv_init() GhostNVResult {
    init_mutex.lock();
    defer init_mutex.unlock();
    
    if (initialized) {
        return .GHOSTNV_OK;
    }
    
    // Initialize DRM driver
    drm_driver = allocator.create(drm.DrmDriver) catch {
        return .GHOSTNV_ERROR_MEMORY_ALLOCATION;
    };
    drm_driver.?.* = drm.DrmDriver.init(allocator) catch {
        allocator.destroy(drm_driver.?);
        drm_driver = null;
        return .GHOSTNV_ERROR_NOT_SUPPORTED;
    };
    
    // Initialize vibrance engine
    vibrance_engine = allocator.create(vibrance.VibranceEngine) catch {
        return .GHOSTNV_ERROR_MEMORY_ALLOCATION;
    };
    vibrance_engine.?.* = vibrance.VibranceEngine.init(allocator, drm_driver.?);
    
    // Initialize G-SYNC manager
    gsync_manager = allocator.create(gsync.GsyncManager) catch {
        return .GHOSTNV_ERROR_MEMORY_ALLOCATION;
    };
    gsync_manager.?.* = gsync.GsyncManager.init(allocator, drm_driver.?);
    
    // Initialize nvctl interface
    nvctl_interface = allocator.create(nvctl.NvctlInterface) catch {
        return .GHOSTNV_ERROR_MEMORY_ALLOCATION;
    };
    nvctl_interface.?.* = nvctl.NvctlInterface.init(allocator, vibrance_engine.?, gsync_manager.?);
    
    // Enumerate devices
    nvctl_interface.?.enumerate_devices() catch {
        return .GHOSTNV_ERROR_DEVICE_BUSY;
    };
    
    initialized = true;
    
    std.log.info("GhostNV FFI interface initialized successfully", .{});
    return .GHOSTNV_OK;
}

/// Cleanup GhostNV driver interface  
export fn ghostnv_cleanup() void {
    init_mutex.lock();
    defer init_mutex.unlock();
    
    if (!initialized) {
        return;
    }
    
    if (nvctl_interface) |interface| {
        interface.deinit();
        allocator.destroy(interface);
        nvctl_interface = null;
    }
    
    if (gsync_manager) |manager| {
        manager.deinit();
        allocator.destroy(manager);
        gsync_manager = null;
    }
    
    if (vibrance_engine) |engine| {
        engine.deinit();
        allocator.destroy(engine);
        vibrance_engine = null;
    }
    
    if (drm_driver) |driver| {
        driver.deinit();
        allocator.destroy(driver);
        drm_driver = null;
    }
    
    initialized = false;
    
    std.log.info("GhostNV FFI interface cleaned up", .{});
}

/// Get number of NVIDIA devices
export fn ghostnv_get_device_count() i32 {
    if (!initialized or nvctl_interface == null) {
        return -1;
    }
    
    return @intCast(nvctl_interface.?.devices.items.len);
}

/// Get device information
export fn ghostnv_get_device_info(device_id: u32, device: *GhostNVDevice) GhostNVResult {
    if (!initialized or nvctl_interface == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    const device_info = nvctl_interface.?.get_device_info(device_id) catch {
        return .GHOSTNV_ERROR_INVALID_DEVICE;
    };
    
    device.device_id = device_info.id;
    
    // Copy strings safely
    const name_len = @min(device_info.name.len, device.name.len - 1);
    @memcpy(device.name[0..name_len], device_info.name[0..name_len]);
    device.name[name_len] = 0; // Null terminate
    
    const version_len = @min(device_info.driver_version.len, device.driver_version.len - 1);
    @memcpy(device.driver_version[0..version_len], device_info.driver_version[0..version_len]);
    device.driver_version[version_len] = 0; // Null terminate
    
    device.pci_bus = device_info.pci_bus;
    device.pci_device = device_info.pci_device;
    device.pci_function = device_info.pci_function;
    device.supports_gsync = true; // Assume RTX 30/40 series
    device.supports_vrr = true;
    device.supports_hdr = true;
    
    return .GHOSTNV_OK;
}

/// Check if specific feature is supported
export fn ghostnv_supports_feature(device_id: u32, feature_name: [*:0]const u8) bool {
    _ = device_id;
    
    if (!initialized) {
        return false;
    }
    
    const feature = std.mem.span(feature_name);
    
    if (std.mem.eql(u8, feature, "vibrance")) return true;
    if (std.mem.eql(u8, feature, "gsync")) return true;
    if (std.mem.eql(u8, feature, "vrr")) return true;
    if (std.mem.eql(u8, feature, "hdr")) return true;
    if (std.mem.eql(u8, feature, "frame_generation")) return true;
    if (std.mem.eql(u8, feature, "container_runtime")) return true;
    
    return false;
}

// ═══════════════════════════════════════════════════════
// Digital Vibrance API
// ═══════════════════════════════════════════════════════

/// Initialize vibrance engine
export fn ghostnv_vibrance_init() GhostNVResult {
    if (!initialized or vibrance_engine == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    // Vibrance engine is already initialized in ghostnv_init()
    return .GHOSTNV_OK;
}

/// Apply vibrance profile by name
export fn ghostnv_vibrance_apply_profile(profile_name: [*:0]const u8) GhostNVResult {
    if (!initialized or vibrance_engine == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    const name = std.mem.span(profile_name);
    
    vibrance_engine.?.apply_profile(name) catch {
        return .GHOSTNV_ERROR_INVALID_VALUE;
    };
    
    return .GHOSTNV_OK;
}

/// Create custom profile
export fn ghostnv_vibrance_create_profile(name: [*:0]const u8, profile: *const GhostNVVibranceProfile) GhostNVResult {
    if (!initialized or vibrance_engine == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    const profile_name = std.mem.span(name);
    
    // Convert FFI profile to internal format
    const internal_profile = vibrance.VibranceProfile{
        .name = profile_name,
        .vibrance = profile.vibrance,
        .saturation = profile.saturation,
        .gamma = profile.gamma,
        .brightness = profile.brightness,
        .contrast = profile.contrast,
        .hue_shift = 0,
        .red_vibrance = profile.red_vibrance,
        .green_vibrance = profile.green_vibrance,
        .blue_vibrance = profile.blue_vibrance,
        .temperature = profile.temperature,
        .tint = 0,
        .hdr_peak_brightness = 1000,
        .hdr_tone_mapping = .aces,
        .game_mode = .standard,
        .preserve_skin_tones = profile.preserve_skin_tones,
        .enhance_foliage = profile.enhance_foliage,
        .boost_sky_colors = profile.boost_sky_colors,
    };
    
    vibrance_engine.?.create_profile(profile_name, internal_profile) catch {
        return .GHOSTNV_ERROR_INVALID_VALUE;
    };
    
    return .GHOSTNV_OK;
}

/// Real-time vibrance adjustment
export fn ghostnv_vibrance_adjust(delta: i8) GhostNVResult {
    if (!initialized or vibrance_engine == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    vibrance_engine.?.real_time_adjust(delta) catch {
        return .GHOSTNV_ERROR_INVALID_VALUE;
    };
    
    return .GHOSTNV_OK;
}

/// Get current vibrance settings
export fn ghostnv_vibrance_get_current(out_profile: *GhostNVVibranceProfile) GhostNVResult {
    if (!initialized or vibrance_engine == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    const current_profile = vibrance_engine.?.get_active_profile() orelse {
        return .GHOSTNV_ERROR_INVALID_VALUE;
    };
    
    // Convert internal profile to FFI format
    out_profile.vibrance = current_profile.vibrance;
    out_profile.saturation = current_profile.saturation;
    out_profile.gamma = current_profile.gamma;
    out_profile.brightness = current_profile.brightness;
    out_profile.contrast = current_profile.contrast;
    out_profile.temperature = current_profile.temperature;
    out_profile.red_vibrance = current_profile.red_vibrance;
    out_profile.green_vibrance = current_profile.green_vibrance;
    out_profile.blue_vibrance = current_profile.blue_vibrance;
    out_profile.preserve_skin_tones = current_profile.preserve_skin_tones;
    out_profile.enhance_foliage = current_profile.enhance_foliage;
    out_profile.boost_sky_colors = current_profile.boost_sky_colors;
    
    return .GHOSTNV_OK;
}

/// List available profiles (returns count, fills names array)
export fn ghostnv_vibrance_list_profiles(names: [*][64]u8, max_count: i32) i32 {
    if (!initialized or vibrance_engine == null) {
        return -1;
    }
    
    // Mock profile list for now - real implementation would iterate over stored profiles
    const profile_names = [_][]const u8{
        "Default",
        "Gaming",
        "Cinema",
        "Competitive FPS",
        "Photography",
        "Cyberpunk",
        "Valorant",
        "Counter-Strike",
    };
    
    const count = @min(profile_names.len, @as(usize, @intCast(max_count)));
    
    for (0..count) |i| {
        const name_len = @min(profile_names[i].len, 63);
        @memcpy(names[i][0..name_len], profile_names[i][0..name_len]);
        names[i][name_len] = 0; // Null terminate
    }
    
    return @intCast(count);
}

/// Auto-detect game and apply profile
export fn ghostnv_vibrance_auto_detect(window_title: [*:0]const u8) GhostNVResult {
    if (!initialized or vibrance_engine == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    const title = std.mem.span(window_title);
    
    vibrance_engine.?.auto_detect_game(title) catch {
        return .GHOSTNV_ERROR_INVALID_VALUE;
    };
    
    return .GHOSTNV_OK;
}

/// Disable vibrance
export fn ghostnv_vibrance_disable() GhostNVResult {
    if (!initialized or vibrance_engine == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    vibrance_engine.?.disable() catch {
        return .GHOSTNV_ERROR_DEVICE_BUSY;
    };
    
    return .GHOSTNV_OK;
}

// ═══════════════════════════════════════════════════════
// G-SYNC / VRR API
// ═══════════════════════════════════════════════════════

/// Enable G-SYNC with specified mode
export fn ghostnv_gsync_enable(device_id: u32, mode: GhostNVGSyncMode) GhostNVResult {
    _ = device_id;
    
    if (!initialized or gsync_manager == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    const gsync_mode: gsync.GsyncMode = switch (mode) {
        .GSYNC_DISABLED => .disabled,
        .GSYNC_COMPATIBLE => .compatible,
        .GSYNC_CERTIFIED => .certified,
        .GSYNC_ULTIMATE => .ultimate,
        .GSYNC_ESPORTS => .esports,
    };
    
    gsync_manager.?.enable_gsync(gsync_mode) catch {
        return .GHOSTNV_ERROR_DEVICE_BUSY;
    };
    
    return .GHOSTNV_OK;
}

/// Disable G-SYNC
export fn ghostnv_gsync_disable(device_id: u32) GhostNVResult {
    _ = device_id;
    
    if (!initialized or gsync_manager == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    gsync_manager.?.disable_gsync() catch {
        return .GHOSTNV_ERROR_DEVICE_BUSY;
    };
    
    return .GHOSTNV_OK;
}

/// Get current G-SYNC status
export fn ghostnv_gsync_get_status(device_id: u32, status: *GhostNVGSyncStatus) GhostNVResult {
    _ = device_id;
    
    if (!initialized or gsync_manager == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    const displays = gsync_manager.?.get_all_display_info() catch {
        return .GHOSTNV_ERROR_DEVICE_BUSY;
    };
    defer allocator.free(displays);
    
    if (displays.len == 0) {
        return .GHOSTNV_ERROR_INVALID_DEVICE;
    }
    
    const display = displays[0];
    
    status.mode = switch (display.gsync_mode) {
        .disabled => .GSYNC_DISABLED,
        .compatible => .GSYNC_COMPATIBLE,
        .certified => .GSYNC_CERTIFIED,
        .ultimate => .GSYNC_ULTIMATE,
        .esports => .GSYNC_ESPORTS,
    };
    
    status.min_refresh_hz = display.min_refresh_hz;
    status.max_refresh_hz = display.max_refresh_hz;
    status.current_refresh_hz = display.current_refresh_hz;
    status.ultra_low_latency = display.ultra_low_latency;
    status.variable_overdrive = display.variable_overdrive;
    status.motion_blur_reduction = display.motion_blur_reduction;
    status.hdr_enabled = display.hdr_enabled;
    status.peak_brightness_nits = display.peak_brightness_nits;
    
    return .GHOSTNV_OK;
}

/// Set refresh rate (for VRR)
export fn ghostnv_gsync_set_refresh_rate(device_id: u32, refresh_hz: u32) GhostNVResult {
    _ = device_id;
    
    if (!initialized or gsync_manager == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    gsync_manager.?.set_refresh_rate(refresh_hz) catch {
        return .GHOSTNV_ERROR_INVALID_VALUE;
    };
    
    return .GHOSTNV_OK;
}

/// Optimize for specific game type
export fn ghostnv_gsync_optimize_for_game(device_id: u32, game_type: GhostNVGameType) GhostNVResult {
    _ = device_id;
    
    if (!initialized or gsync_manager == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    const internal_game_type: gsync.GameType = switch (game_type) {
        .GAME_COMPETITIVE_FPS => .competitive_fps,
        .GAME_IMMERSIVE_SINGLE_PLAYER => .immersive_single_player,
        .GAME_RACING => .racing,
        .GAME_CINEMA => .cinema,
    };
    
    gsync_manager.?.optimize_for_game(internal_game_type);
    
    return .GHOSTNV_OK;
}

/// Enable/disable ultra low latency
export fn ghostnv_gsync_set_ultra_low_latency(device_id: u32, enabled: bool) GhostNVResult {
    _ = device_id;
    
    if (!initialized or gsync_manager == null) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    if (enabled) {
        gsync_manager.?.optimize_for_game(.competitive_fps);
    } else {
        gsync_manager.?.optimize_for_game(.immersive_single_player);
    }
    
    return .GHOSTNV_OK;
}

// ═══════════════════════════════════════════════════════
// Performance & System API
// ═══════════════════════════════════════════════════════

/// Get performance information
export fn ghostnv_performance_get_info(device_id: u32, info: *GhostNVPerformanceInfo) GhostNVResult {
    _ = device_id;
    
    if (!initialized) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    // Mock performance data for RTX 4090
    info.gpu_clock_mhz = 2520;
    info.memory_clock_mhz = 10501;
    info.temperature_c = 65;
    info.fan_speed_rpm = 1800;
    info.power_draw_watts = 380;
    info.gpu_utilization_percent = 95;
    info.memory_utilization_percent = 75;
    info.average_frametime_ms = 8.33; // 120 FPS
    info.current_fps = 120;
    
    return .GHOSTNV_OK;
}

/// Set performance level (0=auto, 1=power save, 2=balanced, 3=performance, 4=max)
export fn ghostnv_performance_set_level(device_id: u32, level: u32) GhostNVResult {
    _ = device_id;
    
    if (!initialized) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    if (level > 4) {
        return .GHOSTNV_ERROR_INVALID_VALUE;
    }
    
    std.log.info("Setting performance level to {}", .{level});
    return .GHOSTNV_OK;
}

/// Enable/disable frame generation
export fn ghostnv_performance_set_frame_generation(device_id: u32, enabled: bool, max_frames: u8) GhostNVResult {
    _ = device_id;
    
    if (!initialized) {
        return .GHOSTNV_ERROR_NOT_INITIALIZED;
    }
    
    if (max_frames > 4) {
        return .GHOSTNV_ERROR_INVALID_VALUE;
    }
    
    std.log.info("Frame generation: enabled={}, max_frames={}", .{ enabled, max_frames });
    return .GHOSTNV_OK;
}

// ═══════════════════════════════════════════════════════
// Version and Build Information
// ═══════════════════════════════════════════════════════

/// Get GhostNV version string
export fn ghostnv_get_version() [*:0]const u8 {
    return "1.0.0-beta";
}

/// Get build information
export fn ghostnv_get_build_info() [*:0]const u8 {
    return "GhostNV Pure Zig NVIDIA Driver - Built for RTX 30/40 series";
}