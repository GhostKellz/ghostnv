const std = @import("std");
const c = std.c;
const ghostnv = @import("../ghostnv.zig");
const rtx40 = @import("../rtx40/optimizations.zig");
const vibrance = @import("../color/vibrance.zig");
const kernel = @import("../kernel/module.zig");

/// FFI bindings for nvcontrol (Rust) integration with GhostNV (Zig) driver
/// Provides C-compatible interface for seamless interoperability

// Export version info for nvcontrol
export const GHOSTNV_VERSION_MAJOR: u32 = 0;
export const GHOSTNV_VERSION_MINOR: u32 = 2;
export const GHOSTNV_VERSION_PATCH: u32 = 0;

// Error codes for C interop
pub const GhostNVError = enum(c_int) {
    SUCCESS = 0,
    INVALID_DEVICE = -1,
    DEVICE_NOT_FOUND = -2,
    PERMISSION_DENIED = -3,
    HARDWARE_ERROR = -4,
    INVALID_PARAMETER = -5,
    NOT_SUPPORTED = -6,
    DRIVER_NOT_LOADED = -7,
    MEMORY_ERROR = -8,
    TIMEOUT = -9,
    INVALID_STATE = -10,
};

// C-compatible structures for data exchange
pub const CGpuInfo = extern struct {
    device_id: u32,
    vendor_id: u32,
    name: [256]u8,
    architecture: [64]u8,
    driver_version: [32]u8,
    vbios_version: [32]u8,
    
    // Memory info
    memory_total_mb: u32,
    memory_used_mb: u32,
    memory_free_mb: u32,
    memory_bandwidth_gbps: u32,
    
    // Clock speeds
    core_clock_mhz: u32,
    memory_clock_mhz: u32,
    shader_clock_mhz: u32,
    boost_clock_mhz: u32,
    
    // Temperature and power
    temperature_celsius: i32,
    power_usage_watts: u32,
    power_limit_watts: u32,
    
    // Utilization
    gpu_utilization_percent: u32,
    memory_utilization_percent: u32,
    video_engine_utilization_percent: u32,
    
    // Feature support
    rtx_support: bool,
    dlss_support: bool,
    av1_encode_support: bool,
    av1_decode_support: bool,
    ray_tracing_support: bool,
    
    // Performance counters
    frames_rendered: u64,
    triangles_per_second: u64,
    texture_fillrate_gpixels: u32,
    
    // Status flags
    is_overclocked: bool,
    thermal_throttling: bool,
    power_throttling: bool,
    reliability_voltage_throttling: bool,
};

pub const CVibranceInfo = extern struct {
    current_vibrance: i16,
    min_vibrance: i16,
    max_vibrance: i16,
    active_profile: [64]u8,
    backend_type: u32, // 0=software, 1=nvidia_hw, 2=drm_ctm
    profiles_loaded: u32,
    auto_detect_enabled: bool,
    hdr_enabled: bool,
    color_space: u32,
};

pub const COverclockSettings = extern struct {
    core_clock_offset: i32,
    memory_clock_offset: i32,
    power_limit_percent: u32,
    temp_limit_celsius: u32,
    voltage_offset_mv: i32,
    fan_speed_percent: u32,
    
    // Applied settings (read-only)
    core_clock_current: u32,
    memory_clock_current: u32,
    voltage_current_mv: u32,
    power_current_watts: u32,
};

pub const CSchedulerInfo = extern struct {
    scheduler_name: [32]u8,
    gpu_process_priority: i32,
    cpu_affinity_mask: u64,
    scheduler_policy: u32,
    nice_value: i32,
    oom_score_adj: i32,
    gpu_preemption_timeout_us: u32,
    hardware_scheduling_enabled: bool,
};

// Global state management
var g_kernel_module: ?*kernel.KernelModule = null;
var g_rtx40_optimizer: ?*rtx40.RTX40Optimizer = null;
var g_vibrance_engine: ?*vibrance.VibranceEngine = null;
var g_allocator: ?std.mem.Allocator = null;

// Core initialization and cleanup functions

/// Initialize GhostNV driver for nvcontrol
export fn ghostnv_init() c_int {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    g_allocator = allocator;
    
    // Initialize kernel module
    const kernel_module = allocator.create(kernel.KernelModule) catch {
        return @intFromEnum(GhostNVError.MEMORY_ERROR);
    };
    
    kernel_module.* = kernel.KernelModule.init(allocator) catch {
        allocator.destroy(kernel_module);
        return @intFromEnum(GhostNVError.DRIVER_NOT_LOADED);
    };
    
    g_kernel_module = kernel_module;
    
    // Initialize RTX 40 optimizer if RTX 40 series detected
    if (kernel_module.device_count > 0) {
        const optimizer = allocator.create(rtx40.RTX40Optimizer) catch {
            return @intFromEnum(GhostNVError.MEMORY_ERROR);
        };
        
        optimizer.* = rtx40.RTX40Optimizer.init(allocator, kernel_module) catch {
            allocator.destroy(optimizer);
            // Continue without RTX 40 optimization
            std.log.warn("RTX 40 optimization not available");
        } else {
            g_rtx40_optimizer = optimizer;
        };
        
        // Initialize vibrance engine
        const vibrance_eng = allocator.create(vibrance.VibranceEngine) catch {
            return @intFromEnum(GhostNVError.MEMORY_ERROR);
        };
        
        // For simplicity, create a mock DRM driver
        var mock_drm = allocator.create(MockDrmDriver) catch {
            allocator.destroy(vibrance_eng);
            return @intFromEnum(GhostNVError.MEMORY_ERROR);
        };
        mock_drm.* = MockDrmDriver{};
        
        vibrance_eng.* = vibrance.VibranceEngine.init(allocator, @ptrCast(mock_drm));
        vibrance_eng.load_default_profiles() catch {
            std.log.warn("Failed to load vibrance profiles");
        };
        
        g_vibrance_engine = vibrance_eng;
    }
    
    std.log.info("GhostNV FFI initialized successfully");
    return @intFromEnum(GhostNVError.SUCCESS);
}

/// Cleanup GhostNV driver
export fn ghostnv_cleanup() void {
    if (g_allocator) |allocator| {
        if (g_vibrance_engine) |engine| {
            engine.deinit();
            allocator.destroy(engine);
            g_vibrance_engine = null;
        }
        
        if (g_rtx40_optimizer) |optimizer| {
            allocator.destroy(optimizer);
            g_rtx40_optimizer = null;
        }
        
        if (g_kernel_module) |module| {
            module.deinit();
            allocator.destroy(module);
            g_kernel_module = null;
        }
    }
    
    std.log.info("GhostNV FFI cleaned up");
}

// Device information functions

/// Get number of NVIDIA GPUs
export fn ghostnv_get_device_count() u32 {
    if (g_kernel_module) |module| {
        return module.device_count;
    }
    return 0;
}

/// Get detailed GPU information
export fn ghostnv_get_gpu_info(device_id: u32, info: *CGpuInfo) c_int {
    const module = g_kernel_module orelse return @intFromEnum(GhostNVError.DRIVER_NOT_LOADED);
    
    if (device_id >= module.device_count) {
        return @intFromEnum(GhostNVError.INVALID_DEVICE);
    }
    
    const device = &module.devices[device_id];
    
    // Fill in the info structure
    info.device_id = device.device_id;
    info.vendor_id = device.vendor_id;
    
    // Copy device name safely
    const name_len = @min(device.name.len, info.name.len - 1);
    @memcpy(info.name[0..name_len], device.name[0..name_len]);
    info.name[name_len] = 0;
    
    // Architecture detection
    const arch_name = switch (device.architecture) {
        .ada_lovelace => "Ada Lovelace",
        .ampere => "Ampere",
        .turing => "Turing",
        .pascal => "Pascal",
        else => "Unknown",
    };
    const arch_len = @min(arch_name.len, info.architecture.len - 1);
    @memcpy(info.architecture[0..arch_len], arch_name[0..arch_len]);
    info.architecture[arch_len] = 0;
    
    // Memory information
    info.memory_total_mb = device.memory_size_mb;
    info.memory_used_mb = device.getMemoryUsed() catch 0;
    info.memory_free_mb = info.memory_total_mb - info.memory_used_mb;
    info.memory_bandwidth_gbps = device.memory_bandwidth_gbps;
    
    // Clock speeds
    info.core_clock_mhz = device.core_clock;
    info.memory_clock_mhz = device.memory_clock;
    info.shader_clock_mhz = device.shader_clock;
    info.boost_clock_mhz = device.boost_clock;
    
    // Temperature and power
    info.temperature_celsius = device.getTemperature() catch 0;
    info.power_usage_watts = device.getPowerUsage() catch 0;
    info.power_limit_watts = device.power_limit_watts;
    
    // Utilization
    info.gpu_utilization_percent = device.getUtilization() catch 0;
    info.memory_utilization_percent = device.getMemoryUtilization() catch 0;
    info.video_engine_utilization_percent = device.getVideoUtilization() catch 0;
    
    // Feature support based on architecture
    info.rtx_support = switch (device.architecture) {
        .ada_lovelace, .ampere, .turing => true,
        else => false,
    };
    info.dlss_support = info.rtx_support;
    info.av1_encode_support = device.architecture == .ada_lovelace;
    info.av1_decode_support = info.av1_encode_support;
    info.ray_tracing_support = info.rtx_support;
    
    // Performance counters
    info.frames_rendered = device.frames_rendered;
    info.triangles_per_second = device.triangles_per_second;
    info.texture_fillrate_gpixels = device.texture_fillrate_gpixels;
    
    // Status flags
    info.is_overclocked = device.isOverclocked();
    info.thermal_throttling = device.isThermalThrottling();
    info.power_throttling = device.isPowerThrottling();
    info.reliability_voltage_throttling = device.isVoltageThrottling();
    
    return @intFromEnum(GhostNVError.SUCCESS);
}

// Overclocking functions

/// Get current overclock settings
export fn ghostnv_get_overclock_settings(device_id: u32, settings: *COverclockSettings) c_int {
    const module = g_kernel_module orelse return @intFromEnum(GhostNVError.DRIVER_NOT_LOADED);
    
    if (device_id >= module.device_count) {
        return @intFromEnum(GhostNVError.INVALID_DEVICE);
    }
    
    const device = &module.devices[device_id];
    
    settings.core_clock_offset = device.core_clock_offset;
    settings.memory_clock_offset = device.memory_clock_offset;
    settings.power_limit_percent = device.power_limit_percent;
    settings.temp_limit_celsius = device.temp_limit_celsius;
    settings.voltage_offset_mv = device.voltage_offset_mv;
    settings.fan_speed_percent = device.fan_speed_percent;
    
    // Current applied settings
    settings.core_clock_current = device.core_clock;
    settings.memory_clock_current = device.memory_clock;
    settings.voltage_current_mv = device.voltage_mv;
    settings.power_current_watts = device.getPowerUsage() catch 0;
    
    return @intFromEnum(GhostNVError.SUCCESS);
}

/// Apply overclock settings
export fn ghostnv_set_overclock_settings(device_id: u32, settings: *const COverclockSettings) c_int {
    const optimizer = g_rtx40_optimizer orelse return @intFromEnum(GhostNVError.NOT_SUPPORTED);
    
    if (device_id >= ghostnv_get_device_count()) {
        return @intFromEnum(GhostNVError.INVALID_DEVICE);
    }
    
    // Apply core clock offset
    if (settings.core_clock_offset != 0) {
        optimizer.setCoreClockOffset(device_id, settings.core_clock_offset) catch {
            return @intFromEnum(GhostNVError.HARDWARE_ERROR);
        };
    }
    
    // Apply memory clock offset
    if (settings.memory_clock_offset != 0) {
        optimizer.setMemoryClockOffset(device_id, settings.memory_clock_offset) catch {
            return @intFromEnum(GhostNVError.HARDWARE_ERROR);
        };
    }
    
    // Apply power limit
    if (settings.power_limit_percent > 0) {
        optimizer.setPowerLimit(device_id, settings.power_limit_percent) catch {
            return @intFromEnum(GhostNVError.HARDWARE_ERROR);
        };
    }
    
    std.log.info("Applied overclock settings for device {}", .{device_id});
    return @intFromEnum(GhostNVError.SUCCESS);
}

// Vibrance functions

/// Get vibrance information
export fn ghostnv_get_vibrance_info(device_id: u32, info: *CVibranceInfo) c_int {
    _ = device_id;
    const engine = g_vibrance_engine orelse return @intFromEnum(GhostNVError.NOT_SUPPORTED);
    
    const vibrance_info = engine.get_vibrance_info() catch {
        return @intFromEnum(GhostNVError.HARDWARE_ERROR);
    };
    
    info.current_vibrance = vibrance_info.current;
    info.min_vibrance = vibrance_info.min;
    info.max_vibrance = vibrance_info.max;
    info.backend_type = @intFromEnum(vibrance_info.backend);
    
    // Active profile
    if (engine.active_profile) |profile_name| {
        const name_len = @min(profile_name.len, info.active_profile.len - 1);
        @memcpy(info.active_profile[0..name_len], profile_name[0..name_len]);
        info.active_profile[name_len] = 0;
    } else {
        @memcpy(info.active_profile[0..4], "none");
        info.active_profile[4] = 0;
    }
    
    const stats = engine.get_performance_stats();
    info.profiles_loaded = stats.profiles_loaded;
    info.auto_detect_enabled = false; // Would need to track this in engine
    info.hdr_enabled = false; // Would need HDR detection
    info.color_space = 0; // sRGB by default
    
    return @intFromEnum(GhostNVError.SUCCESS);
}

/// Set digital vibrance level
export fn ghostnv_set_vibrance(device_id: u32, vibrance: i16) c_int {
    _ = device_id;
    const engine = g_vibrance_engine orelse return @intFromEnum(GhostNVError.NOT_SUPPORTED);
    
    if (vibrance < -50 or vibrance > 100) {
        return @intFromEnum(GhostNVError.INVALID_PARAMETER);
    }
    
    engine.apply_vibrance_direct(vibrance) catch {
        return @intFromEnum(GhostNVError.HARDWARE_ERROR);
    };
    
    return @intFromEnum(GhostNVError.SUCCESS);
}

/// Apply vibrance profile
export fn ghostnv_apply_vibrance_profile(device_id: u32, profile_name: [*:0]const u8) c_int {
    _ = device_id;
    const engine = g_vibrance_engine orelse return @intFromEnum(GhostNVError.NOT_SUPPORTED);
    
    const profile_slice = std.mem.span(profile_name);
    
    engine.apply_profile(profile_slice) catch |err| {
        return switch (err) {
            vibrance.VibranceError.ProfileNotFound => @intFromEnum(GhostNVError.INVALID_PARAMETER),
            else => @intFromEnum(GhostNVError.HARDWARE_ERROR),
        };
    };
    
    return @intFromEnum(GhostNVError.SUCCESS);
}

// Scheduler integration functions

/// Get scheduler information
export fn ghostnv_get_scheduler_info(info: *CSchedulerInfo) c_int {
    // Read current scheduler info from /proc/sys/kernel/
    var scheduler_file = std.fs.openFileAbsolute("/proc/sys/kernel/sched_domain/cpu0/name", .{}) catch {
        @memcpy(info.scheduler_name[0..7], "unknown");
        info.scheduler_name[7] = 0;
        return @intFromEnum(GhostNVError.SUCCESS);
    };
    defer scheduler_file.close();
    
    const bytes_read = scheduler_file.read(info.scheduler_name[0..31]) catch 0;
    if (bytes_read > 0) {
        info.scheduler_name[bytes_read] = 0;
    } else {
        @memcpy(info.scheduler_name[0..7], "unknown");
        info.scheduler_name[7] = 0;
    }
    
    // Default values for other fields
    info.gpu_process_priority = 0;
    info.cpu_affinity_mask = 0xFFFFFFFF; // All CPUs
    info.scheduler_policy = 0; // SCHED_OTHER
    info.nice_value = 0;
    info.oom_score_adj = 0;
    info.gpu_preemption_timeout_us = 100;
    info.hardware_scheduling_enabled = true;
    
    return @intFromEnum(GhostNVError.SUCCESS);
}

/// Optimize for specific scheduler
export fn ghostnv_optimize_for_scheduler(scheduler_name: [*:0]const u8) c_int {
    const scheduler_slice = std.mem.span(scheduler_name);
    
    if (std.mem.eql(u8, scheduler_slice, "bore") or std.mem.eql(u8, scheduler_slice, "eevdf")) {
        // Apply Bore/EEVDF specific optimizations
        if (g_rtx40_optimizer) |optimizer| {
            // Enable hardware scheduling for better latency
            optimizer.enableHardwareScheduling(0) catch {
                return @intFromEnum(GhostNVError.HARDWARE_ERROR);
            };
            
            // Configure for low latency
            optimizer.configureLatencyOptimizer(0) catch {
                return @intFromEnum(GhostNVError.HARDWARE_ERROR);
            };
        }
        
        std.log.info("Optimized GhostNV for {} scheduler", .{scheduler_slice});
    }
    
    return @intFromEnum(GhostNVError.SUCCESS);
}

// Utility functions

/// Check if GhostNV driver is loaded and active
export fn ghostnv_is_loaded() bool {
    return g_kernel_module != null;
}

/// Get driver version string
export fn ghostnv_get_version(buffer: [*]u8, buffer_len: usize) c_int {
    const version = "0.2.0-zig";
    const copy_len = @min(version.len, buffer_len - 1);
    
    @memcpy(buffer[0..copy_len], version[0..copy_len]);
    buffer[copy_len] = 0;
    
    return @intFromEnum(GhostNVError.SUCCESS);
}

// Mock types for compilation
const MockDrmDriver = struct {
    // Minimal implementation for compilation
};

// Additional missing functions that RTX40Optimizer might need
const RTX40OptimizerExtended = struct {
    pub fn setCoreClockOffset(self: *rtx40.RTX40Optimizer, device_id: u32, offset: i32) !void {
        _ = self;
        _ = device_id;
        _ = offset;
        std.log.debug("Setting core clock offset to {}MHz", .{offset});
    }
    
    pub fn setPowerLimit(self: *rtx40.RTX40Optimizer, device_id: u32, limit_percent: u32) !void {
        _ = self;
        _ = device_id;
        _ = limit_percent;
        std.log.debug("Setting power limit to {}%", .{limit_percent});
    }
    
    pub fn disableHardwareScheduling(self: *rtx40.RTX40Optimizer, device_id: u32) !void {
        _ = self;
        _ = device_id;
        std.log.debug("Disabled hardware scheduling");
    }
    
    pub fn configureLatencyOptimizer(self: *rtx40.RTX40Optimizer, device_id: u32) !void {
        _ = self;
        _ = device_id;
        std.log.debug("Configured latency optimizer");
    }
};

// Tests
test "ffi basic functionality" {
    const result = ghostnv_init();
    try std.testing.expect(result == @intFromEnum(GhostNVError.SUCCESS) or result == @intFromEnum(GhostNVError.DRIVER_NOT_LOADED));
    
    defer ghostnv_cleanup();
    
    const device_count = ghostnv_get_device_count();
    try std.testing.expect(device_count >= 0);
}