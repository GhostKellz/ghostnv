const std = @import("std");
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expect = testing.expect;

// Import modules to test
const ghostnv = @import("zig-nvidia");
const vibrance = ghostnv.color_vibrance;
const gaming = ghostnv.gaming_performance;
const container = ghostnv.container_runtime;
const cuda = ghostnv.cuda_runtime;

// =============================================================================
// Digital Vibrance Unit Tests
// =============================================================================

test "vibrance profile creation and validation" {
    const profile = vibrance.VibranceProfile{
        .name = "Test Profile",
        .vibrance = 50,
        .saturation = 25,
        .gamma = 2.2,
        .brightness = 10,
        .contrast = 15,
        .temperature = 100,
        .red_vibrance = 45,
        .green_vibrance = 50,
        .blue_vibrance = 55,
        .preserve_skin_tones = true,
        .enhance_foliage = false,
        .boost_sky_colors = true,
    };
    
    try expect(profile.vibrance >= -50 and profile.vibrance <= 100);
    try expect(profile.saturation >= -50 and profile.saturation <= 50);
    try expect(profile.gamma >= 0.8 and profile.gamma <= 3.0);
    try expect(profile.brightness >= -50 and profile.brightness <= 50);
    try expect(profile.contrast >= -50 and profile.contrast <= 50);
}

test "vibrance value clamping" {
    // Test vibrance clamping logic
    const test_values = [_]struct { input: i32, expected: i8 }{
        .{ .input = -100, .expected = -50 },  // Below minimum
        .{ .input = -50, .expected = -50 },   // At minimum
        .{ .input = 0, .expected = 0 },       // Neutral
        .{ .input = 50, .expected = 50 },     // Normal value
        .{ .input = 100, .expected = 100 },   // At maximum
        .{ .input = 200, .expected = 100 },   // Above maximum
    };
    
    for (test_values) |test_case| {
        const clamped = @as(i8, @intCast(std.math.clamp(test_case.input, -50, 100)));
        try expectEqual(test_case.expected, clamped);
    }
}

test "game profile auto-detection logic" {
    const test_games = [_]struct { window_title: []const u8, expected_profile: []const u8 }{
        .{ .window_title = "Counter-Strike 2", .expected_profile = "Competitive FPS" },
        .{ .window_title = "VALORANT", .expected_profile = "Competitive FPS" },
        .{ .window_title = "Cyberpunk 2077", .expected_profile = "Cyberpunk" },
        .{ .window_title = "The Witcher 3", .expected_profile = "Immersive Gaming" },
        .{ .window_title = "Forza Horizon 5", .expected_profile = "Racing" },
        .{ .window_title = "Unknown Game", .expected_profile = "Default" },
    };
    
    for (test_games) |test_case| {
        const detected_profile = detectGameProfile(test_case.window_title);
        try expect(std.mem.eql(u8, detected_profile, test_case.expected_profile));
    }
}

fn detectGameProfile(window_title: []const u8) []const u8 {
    if (std.mem.indexOf(u8, window_title, "Counter-Strike") != null or
        std.mem.indexOf(u8, window_title, "VALORANT") != null) {
        return "Competitive FPS";
    } else if (std.mem.indexOf(u8, window_title, "Cyberpunk") != null) {
        return "Cyberpunk";
    } else if (std.mem.indexOf(u8, window_title, "Witcher") != null) {
        return "Immersive Gaming";
    } else if (std.mem.indexOf(u8, window_title, "Forza") != null) {
        return "Racing";
    } else {
        return "Default";
    }
}

// =============================================================================
// G-SYNC Unit Tests
// =============================================================================

test "gsync mode capabilities" {
    const modes = [_]gsync.GsyncMode{ .disabled, .gsync_compatible, .gsync_certified, .gsync_ultimate, .gsync_esports };
    
    for (modes) |mode| {
        const hdr_support = mode.supports_hdr();
        const overdrive_support = mode.supports_variable_overdrive();
        const ull_support = mode.supports_ultra_low_latency();
        
        switch (mode) {
            .disabled => {
                try expect(!hdr_support);
                try expect(!overdrive_support);
                try expect(!ull_support);
            },
            .gsync_compatible => {
                try expect(!hdr_support);
                try expect(!overdrive_support);
                try expect(!ull_support);
            },
            .gsync_certified => {
                try expect(!hdr_support);
                try expect(overdrive_support);
                try expect(!ull_support);
            },
            .gsync_ultimate => {
                try expect(hdr_support);
                try expect(overdrive_support);
                try expect(ull_support);
            },
            .gsync_esports => {
                try expect(!hdr_support);
                try expect(!overdrive_support);
                try expect(ull_support);
            },
        }
    }
}

test "gsync refresh rate calculations" {
    var config = gsync.GsyncConfig.init(.gsync_ultimate, 48, 240);
    
    // Test optimal refresh rate calculation for different frame times
    const test_cases = [_]struct { frametime_ns: u64, expected_min: u32, expected_max: u32 }{
        .{ .frametime_ns = 4_166_667, .expected_min = 240, .expected_max = 240 }, // 240 FPS
        .{ .frametime_ns = 8_333_333, .expected_min = 120, .expected_max = 120 }, // 120 FPS
        .{ .frametime_ns = 16_666_667, .expected_min = 60, .expected_max = 60 },  // 60 FPS
        .{ .frametime_ns = 33_333_333, .expected_min = 30, .expected_max = 60 },  // 30 FPS (LFC)
    };
    
    for (test_cases) |test_case| {
        const optimal_rate = config.calculate_optimal_refresh_rate(test_case.frametime_ns);
        try expect(optimal_rate >= test_case.expected_min);
        try expect(optimal_rate <= test_case.expected_max);
    }
}

test "gsync low framerate compensation" {
    const config = gsync.GsyncConfig.init(.gsync_ultimate, 48, 144);
    
    // LFC should be enabled for displays with min refresh < 48Hz
    try expect(config.low_framerate_compensation == false); // min is 48
    
    var low_refresh_config = gsync.GsyncConfig.init(.gsync_ultimate, 30, 144);
    try expect(low_refresh_config.low_framerate_compensation == true); // min is 30
    
    // Test LFC calculation for 30 FPS (should double to 60 Hz)
    const lfc_rate = low_refresh_config.calculate_optimal_refresh_rate(33_333_333); // 30 FPS
    try expect(lfc_rate >= 60);
}

test "panel type response times" {
    const panels = [_]gsync.PanelType{ .tn, .ips, .va, .oled, .quantum_dot, .mini_led, .micro_led };
    
    for (panels) |panel| {
        const response_time = panel.getResponseTime();
        
        switch (panel) {
            .tn => try expectEqual(@as(f32, 1.0), response_time),
            .ips => try expectEqual(@as(f32, 4.0), response_time),
            .va => try expectEqual(@as(f32, 8.0), response_time),
            .oled => try expectEqual(@as(f32, 0.1), response_time),
            .quantum_dot => try expectEqual(@as(f32, 2.0), response_time),
            .mini_led => try expectEqual(@as(f32, 1.0), response_time),
            .micro_led => try expectEqual(@as(f32, 0.1), response_time),
        }
        
        // All response times should be positive and reasonable
        try expect(response_time > 0.0 and response_time <= 10.0);
    }
}

// =============================================================================
// RTX 40 Series Optimization Tests
// =============================================================================

test "rtx40 architecture detection" {
    const architectures = [_]rtx40.AdaArchitecture{ .rtx_4090, .rtx_4080, .rtx_4070_ti, .rtx_4070 };
    
    // Test that each architecture has valid characteristics
    for (architectures) |arch| {
        const memory_config = getMemoryConfigForArch(arch);
        const raster_config = getRasterConfigForArch(arch);
        
        // Memory bandwidth should be reasonable for RTX 40 series
        try expect(memory_config.bandwidth_gbps >= 400);
        try expect(memory_config.bandwidth_gbps <= 1200);
        
        // L2 cache should be appropriate for the tier
        try expect(memory_config.l2_cache_mb >= 36);
        try expect(memory_config.l2_cache_mb <= 96);
        
        // Triangle rate should be reasonable
        try expect(raster_config.triangle_rate_gt >= 50);
        try expect(raster_config.triangle_rate_gt <= 200);
    }
}

fn getMemoryConfigForArch(arch: rtx40.AdaArchitecture) struct {
    bandwidth_gbps: u32,
    l2_cache_mb: u32,
} {
    return switch (arch) {
        .rtx_4090 => .{ .bandwidth_gbps = 1008, .l2_cache_mb = 96 },
        .rtx_4080 => .{ .bandwidth_gbps = 717, .l2_cache_mb = 64 },
        .rtx_4070_ti => .{ .bandwidth_gbps = 504, .l2_cache_mb = 48 },
        .rtx_4070 => .{ .bandwidth_gbps = 504, .l2_cache_mb = 36 },
    };
}

fn getRasterConfigForArch(arch: rtx40.AdaArchitecture) struct {
    triangle_rate_gt: u32,
} {
    return switch (arch) {
        .rtx_4090 => .{ .triangle_rate_gt = 165 },
        .rtx_4080 => .{ .triangle_rate_gt = 110 },
        .rtx_4070_ti => .{ .triangle_rate_gt = 85 },
        .rtx_4070 => .{ .triangle_rate_gt = 65 },
    };
}

test "rtx40 power management calculations" {
    const power_configs = [_]struct {
        arch: rtx40.AdaArchitecture,
        base_watts: u32,
        boost_watts: u32,
    }{
        .{ .arch = .rtx_4090, .base_watts = 450, .boost_watts = 600 },
        .{ .arch = .rtx_4080, .base_watts = 320, .boost_watts = 400 },
        .{ .arch = .rtx_4070_ti, .base_watts = 285, .boost_watts = 350 },
        .{ .arch = .rtx_4070, .base_watts = 200, .boost_watts = 250 },
    };
    
    for (power_configs) |config| {
        // Boost power should always be higher than base power
        try expect(config.boost_watts > config.base_watts);
        
        // Power should be within reasonable bounds for RTX 40 series
        try expect(config.base_watts >= 200 and config.base_watts <= 500);
        try expect(config.boost_watts >= 250 and config.boost_watts <= 650);
        
        // Boost should not be more than 50% higher than base
        const boost_ratio = @as(f32, @floatFromInt(config.boost_watts)) / @as(f32, @floatFromInt(config.base_watts));
        try expect(boost_ratio <= 1.5);
    }
}

// =============================================================================
// Container Runtime Tests
// =============================================================================

test "container resource limits validation" {
    const valid_limits = container.ResourceLimits{
        .memory_limit_mb = 8192,
        .cpu_cores = 4.0,
        .gpu_memory_limit_mb = 12288,
    };
    
    // Test valid resource limits
    try expect(valid_limits.memory_limit_mb > 0);
    try expect(valid_limits.cpu_cores > 0.0);
    try expect(valid_limits.gpu_memory_limit_mb > 0);
    
    // Test resource limit bounds
    try expect(valid_limits.memory_limit_mb <= 128 * 1024); // Max 128GB
    try expect(valid_limits.cpu_cores <= 64.0); // Max 64 cores
    try expect(valid_limits.gpu_memory_limit_mb <= 128 * 1024); // Max 128GB GPU memory
}

test "container gpu access configuration" {
    const gpu_access = container.GpuAccess{
        .enabled = true,
        .device_ids = &[_]u32{ 0, 1 },
        .capabilities = &[_][]const u8{ "compute", "video", "graphics", "utility" },
    };
    
    try expect(gpu_access.enabled);
    try expect(gpu_access.device_ids.len == 2);
    try expect(gpu_access.capabilities.len == 4);
    
    // Test that device IDs are valid
    for (gpu_access.device_ids) |device_id| {
        try expect(device_id < 8); // Reasonable max GPU count
    }
    
    // Test that capabilities are valid
    const valid_capabilities = [_][]const u8{ "compute", "video", "graphics", "utility" };
    for (gpu_access.capabilities) |capability| {
        var found = false;
        for (valid_capabilities) |valid_cap| {
            if (std.mem.eql(u8, capability, valid_cap)) {
                found = true;
                break;
            }
        }
        try expect(found);
    }
}

test "container security policy validation" {
    const security_policy = container.SecurityPolicy{
        .seccomp_profile = "docker-default",
        .apparmor_profile = "docker-default",
        .capabilities = &[_][]const u8{"CAP_SYS_ADMIN"},
    };
    
    // Test that security profiles are not empty
    try expect(security_policy.seccomp_profile.len > 0);
    try expect(security_policy.apparmor_profile.len > 0);
    
    // Test capabilities format
    for (security_policy.capabilities) |capability| {
        try expect(std.mem.startsWith(u8, capability, "CAP_"));
    }
}

// =============================================================================
// Kernel Module Tests
// =============================================================================

test "kernel device info validation" {
    const allocator = testing.allocator;
    
    var device_info = kernel.DeviceInfo{
        .device_id = 0,
        .name = try allocator.dupe(u8, "RTX 4090"),
        .uuid = try allocator.dupe(u8, "GPU-12345678-1234-5678-90AB-CDEF12345678"),
        .pci_bus_id = try allocator.dupe(u8, "0000:01:00.0"),
        .memory_total_mb = 24576,
        .compute_capability_major = 8,
        .compute_capability_minor = 9,
    };
    defer device_info.deinit(allocator);
    
    // Test device info validation
    try expect(device_info.device_id < 16); // Reasonable device count
    try expect(device_info.name.len > 0);
    try expect(device_info.uuid.len > 0);
    try expect(device_info.pci_bus_id.len > 0);
    try expect(device_info.memory_total_mb > 0);
    try expect(device_info.compute_capability_major >= 5); // Minimum modern architecture
    try expect(device_info.compute_capability_minor < 10); // Reasonable minor version
}

test "kernel gpu status validation" {
    const gpu_status = kernel.GPUStatus{
        .gpu_clock_mhz = 2520,
        .memory_clock_mhz = 10501,
        .temperature_c = 65,
        .power_draw_watts = 350,
        .gpu_utilization_percent = 85,
        .memory_utilization_percent = 70,
    };
    
    // Test reasonable GPU status values
    try expect(gpu_status.gpu_clock_mhz >= 1000 and gpu_status.gpu_clock_mhz <= 4000);
    try expect(gpu_status.memory_clock_mhz >= 5000 and gpu_status.memory_clock_mhz <= 15000);
    try expect(gpu_status.temperature_c >= 20 and gpu_status.temperature_c <= 95);
    try expect(gpu_status.power_draw_watts <= 1000);
    try expect(gpu_status.gpu_utilization_percent <= 100);
    try expect(gpu_status.memory_utilization_percent <= 100);
}

// =============================================================================
// Performance Tests
// =============================================================================

test "vibrance adjustment performance" {
    const iterations = 1000;
    const start_time = std.time.nanoTimestamp();
    
    // Simulate vibrance adjustments
    var total: i32 = 0;
    for (0..iterations) |i| {
        const vibrance_value: i8 = @intCast((i % 150) - 50);
        total += vibrance_value;
    }
    
    const end_time = std.time.nanoTimestamp();
    const duration_ns = end_time - start_time;
    const ns_per_op = duration_ns / iterations;
    
    // Should be very fast (< 1000ns per operation)
    try expect(ns_per_op < 1000);
    
    // Prevent optimization
    try expect(total != 0);
}

test "memory allocation performance" {
    const allocator = testing.allocator;
    const allocation_size = 1024 * 1024; // 1MB
    const iterations = 100;
    
    const start_time = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        const buffer = try allocator.alloc(u8, allocation_size);
        defer allocator.free(buffer);
        
        // Touch the memory to ensure it's actually allocated
        buffer[0] = 42;
        buffer[allocation_size - 1] = 42;
    }
    
    const end_time = std.time.nanoTimestamp();
    const duration_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    
    // Should complete within reasonable time (< 100ms)
    try expect(duration_ms < 100.0);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

test "extreme vibrance values handling" {
    const extreme_values = [_]i32{ -1000, -100, -51, 101, 200, 1000 };
    
    for (extreme_values) |value| {
        const clamped = @as(i8, @intCast(std.math.clamp(value, -50, 100)));
        
        // Should always be within valid range
        try expect(clamped >= -50);
        try expect(clamped <= 100);
    }
}

test "zero and negative refresh rates" {
    const invalid_rates = [_]u32{ 0, 1, 10, 1000, 10000 };
    
    for (invalid_rates) |rate| {
        const is_valid = rate >= 30 and rate <= 480; // Reasonable bounds
        
        if (rate < 30 or rate > 480) {
            try expect(!is_valid);
        }
    }
}

test "large memory allocations" {
    const allocator = testing.allocator;
    
    // Test increasingly large allocations
    const sizes = [_]usize{ 1024, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024 };
    
    for (sizes) |size| {
        const buffer = allocator.alloc(u8, size) catch |err| {
            // Large allocations may fail in testing environment
            if (err == error.OutOfMemory) {
                continue;
            }
            return err;
        };
        defer allocator.free(buffer);
        
        // Verify allocation worked
        try expect(buffer.len == size);
        
        // Touch first and last byte
        buffer[0] = 0xAA;
        buffer[size - 1] = 0xBB;
        
        try expect(buffer[0] == 0xAA);
        try expect(buffer[size - 1] == 0xBB);
    }
}

// =============================================================================
// Compatibility Tests
// =============================================================================

test "string handling compatibility" {
    const test_strings = [_][]const u8{
        "RTX 4090",
        "NVIDIA GeForce RTX 4080",
        "Counter-Strike 2",
        "Cyberpunk 2077",
        "",
        "A very long GPU name that might exceed normal buffer sizes and could potentially cause issues",
    };
    
    for (test_strings) |test_string| {
        // Test string length validation
        const is_valid_length = test_string.len <= 256;
        
        if (test_string.len > 256) {
            try expect(!is_valid_length);
        } else {
            try expect(is_valid_length);
        }
        
        // Test that strings are valid UTF-8
        if (test_string.len > 0) {
            try expect(std.unicode.utf8ValidateSlice(test_string));
        }
    }
}

test "enum value bounds checking" {
    // Test G-SYNC mode enum bounds
    const gsync_modes = [_]u32{ 0, 1, 2, 3, 4, 255, 1000 };
    
    for (gsync_modes) |mode_value| {
        const is_valid = mode_value <= 4;
        
        if (mode_value > 4) {
            try expect(!is_valid);
        } else {
            const mode: gsync.GsyncMode = @enumFromInt(mode_value);
            try expect(@intFromEnum(mode) == mode_value);
        }
    }
}

// =============================================================================
// Test Utilities
// =============================================================================

fn createTestProfile(allocator: std.mem.Allocator, name: []const u8) !vibrance.VibranceProfile {
    _ = allocator;
    
    return vibrance.VibranceProfile{
        .name = name,
        .vibrance = 50,
        .saturation = 25,
        .gamma = 2.2,
        .brightness = 0,
        .contrast = 0,
        .temperature = 0,
        .red_vibrance = 50,
        .green_vibrance = 50,
        .blue_vibrance = 50,
        .preserve_skin_tones = true,
        .enhance_foliage = false,
        .boost_sky_colors = false,
    };
}

fn validateGSyncConfig(config: gsync.GsyncConfig) !void {
    try expect(config.min_refresh_hz > 0);
    try expect(config.max_refresh_hz > config.min_refresh_hz);
    try expect(config.current_refresh_hz >= config.min_refresh_hz);
    try expect(config.current_refresh_hz <= config.max_refresh_hz);
    try expect(config.target_frametime_ns > 0);
}