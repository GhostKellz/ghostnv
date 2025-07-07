const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const drm = @import("../drm/driver.zig");
const memory = @import("../hal/memory.zig");

pub const GsyncError = error{
    NotSupported,
    DisplayNotCompatible,
    InvalidRange,
    CalibrationFailed,
    InitializationFailed,
    InvalidMode,
    FirmwareError,
    OverdriveError,
};

pub const GsyncMode = enum(u8) {
    disabled = 0,
    gsync_compatible = 1,      // VESA Adaptive-Sync over DisplayPort
    gsync_certified = 2,       // NVIDIA G-SYNC certified displays
    gsync_ultimate = 3,        // G-SYNC Ultimate with HDR and variable overdrive
    gsync_esports = 4,         // G-SYNC Esports with ultra-low latency
    
    pub fn toString(self: GsyncMode) []const u8 {
        return switch (self) {
            .disabled => "Disabled",
            .gsync_compatible => "G-SYNC Compatible",
            .gsync_certified => "G-SYNC Certified",
            .gsync_ultimate => "G-SYNC Ultimate",
            .gsync_esports => "G-SYNC Esports",
        };
    }
    
    pub fn supports_hdr(self: GsyncMode) bool {
        return self == .gsync_ultimate;
    }
    
    pub fn supports_variable_overdrive(self: GsyncMode) bool {
        return self == .gsync_ultimate or self == .gsync_certified;
    }
    
    pub fn supports_ultra_low_latency(self: GsyncMode) bool {
        return self == .gsync_esports or self == .gsync_ultimate;
    }
};

pub const GsyncCapabilities = struct {
    mode: GsyncMode,
    min_refresh_hz: u32,
    max_refresh_hz: u32,
    supports_hdr: bool,
    supports_variable_overdrive: bool,
    supports_ultra_low_latency: bool,
    supports_low_framerate_compensation: bool,
    panel_type: PanelType,
    backlight_type: BacklightType,
    
    pub fn init(mode: GsyncMode, min_hz: u32, max_hz: u32) GsyncCapabilities {
        return GsyncCapabilities{
            .mode = mode,
            .min_refresh_hz = min_hz,
            .max_refresh_hz = max_hz,
            .supports_hdr = mode.supports_hdr(),
            .supports_variable_overdrive = mode.supports_variable_overdrive(),
            .supports_ultra_low_latency = mode.supports_ultra_low_latency(),
            .supports_low_framerate_compensation = min_hz < 48,
            .panel_type = .ips, // Default
            .backlight_type = .led, // Default
        };
    }
};

pub const PanelType = enum(u8) {
    tn = 0,
    ips = 1,
    va = 2,
    oled = 3,
    quantum_dot = 4,
    mini_led = 5,
    micro_led = 6,
    
    pub fn getResponseTime(self: PanelType) f32 {
        return switch (self) {
            .tn => 1.0,          // 1ms
            .ips => 4.0,         // 4ms
            .va => 8.0,          // 8ms
            .oled => 0.1,        // 0.1ms
            .quantum_dot => 2.0, // 2ms
            .mini_led => 1.0,    // 1ms
            .micro_led => 0.1,   // 0.1ms
        };
    }
};

pub const BacklightType = enum(u8) {
    led = 0,
    quantum_dot = 1,
    mini_led = 2,
    micro_led = 3,
    oled_self_emissive = 4,
    
    pub fn supportsLocalDimming(self: BacklightType) bool {
        return switch (self) {
            .mini_led, .micro_led, .oled_self_emissive => true,
            else => false,
        };
    }
};

pub const OverdriveLevel = enum(u8) {
    off = 0,
    normal = 1,
    fast = 2,
    faster = 3,
    extreme = 4,
    
    pub fn getMultiplier(self: OverdriveLevel) f32 {
        return switch (self) {
            .off => 1.0,
            .normal => 1.2,
            .fast => 1.5,
            .faster => 2.0,
            .extreme => 3.0,
        };
    }
};

pub const GsyncConfig = struct {
    mode: GsyncMode,
    min_refresh_hz: u32,
    max_refresh_hz: u32,
    current_refresh_hz: u32,
    target_frametime_ns: u64,
    
    // G-SYNC specific features
    low_framerate_compensation: bool,
    variable_overdrive: bool,
    overdrive_level: OverdriveLevel,
    ultra_low_latency: bool,
    adaptive_overdrive: bool,
    
    // HDR settings (for G-SYNC Ultimate)
    hdr_enabled: bool,
    peak_brightness_nits: u32,
    local_dimming_zones: u32,
    
    // Esports optimizations
    motion_blur_reduction: bool,
    input_lag_optimization: bool,
    
    pub fn init(mode: GsyncMode, min_hz: u32, max_hz: u32) GsyncConfig {
        return GsyncConfig{
            .mode = mode,
            .min_refresh_hz = min_hz,
            .max_refresh_hz = max_hz,
            .current_refresh_hz = max_hz,
            .target_frametime_ns = std.time.ns_per_s / max_hz,
            .low_framerate_compensation = min_hz < 48,
            .variable_overdrive = mode.supports_variable_overdrive(),
            .overdrive_level = .normal,
            .ultra_low_latency = mode.supports_ultra_low_latency(),
            .adaptive_overdrive = mode.supports_variable_overdrive(),
            .hdr_enabled = false,
            .peak_brightness_nits = 1000,
            .local_dimming_zones = 0,
            .motion_blur_reduction = false,
            .input_lag_optimization = mode == .gsync_esports,
        };
    }
    
    pub fn optimize_for_gaming(self: *GsyncConfig, game_type: GameType) void {
        switch (game_type) {
            .competitive_fps => {
                self.ultra_low_latency = true;
                self.motion_blur_reduction = true;
                self.input_lag_optimization = true;
                self.overdrive_level = .fast;
                self.hdr_enabled = false; // Disable HDR for lowest latency
            },
            .immersive_single_player => {
                self.hdr_enabled = self.mode.supports_hdr();
                self.motion_blur_reduction = false;
                self.overdrive_level = .normal;
                self.adaptive_overdrive = true;
            },
            .racing => {
                self.motion_blur_reduction = false; // Keep motion blur for realism
                self.ultra_low_latency = true;
                self.overdrive_level = .faster;
                self.input_lag_optimization = true;
            },
            .cinema => {
                self.hdr_enabled = self.mode.supports_hdr();
                self.motion_blur_reduction = false;
                self.overdrive_level = .normal;
                self.adaptive_overdrive = true;
            },
        }
    }
    
    pub fn calculate_optimal_refresh_rate(self: *GsyncConfig, frametime_ns: u64) u32 {
        var target_hz = std.time.ns_per_s / frametime_ns;
        
        // Apply low framerate compensation
        if (self.low_framerate_compensation and target_hz < 48) {
            var multiplier: u32 = 2;
            while (target_hz * multiplier < 48 and multiplier <= 4) {
                multiplier += 1;
            }
            target_hz = std.math.min(target_hz * multiplier, self.max_refresh_hz);
        }
        
        // Clamp to supported range
        return @intFromFloat(std.math.clamp(target_hz, self.min_refresh_hz, self.max_refresh_hz));
    }
    
    pub fn update_refresh_rate(self: *GsyncConfig, new_rate: u32) void {
        self.current_refresh_hz = std.math.clamp(new_rate, self.min_refresh_hz, self.max_refresh_hz);
        self.target_frametime_ns = std.time.ns_per_s / self.current_refresh_hz;
    }
};

pub const GameType = enum {
    competitive_fps,
    immersive_single_player,
    racing,
    cinema,
};

pub const GsyncDisplay = struct {
    allocator: Allocator,
    connector_id: u32,
    capabilities: GsyncCapabilities,
    config: GsyncConfig,
    calibrated: bool,
    
    // Hardware state
    panel_temperature: f32,
    backlight_zones: []u16, // For local dimming
    response_time_lut: [256]u8, // Look-up table for overdrive
    
    // Performance tracking
    frame_times: [120]f32,
    frame_time_index: u32,
    average_frametime: f32,
    stuttering_detected: bool,
    
    pub fn init(allocator: Allocator, connector_id: u32, mode: GsyncMode, min_hz: u32, max_hz: u32) !GsyncDisplay {
        const capabilities = GsyncCapabilities.init(mode, min_hz, max_hz);
        const config = GsyncConfig.init(mode, min_hz, max_hz);
        
        // Allocate backlight zones for local dimming
        const zone_count = if (capabilities.supports_hdr) 512 else 0;
        const backlight_zones = try allocator.alloc(u16, zone_count);
        
        // Initialize zones to full brightness
        for (backlight_zones) |*zone| {
            zone.* = 4095; // 12-bit brightness
        }
        
        return GsyncDisplay{
            .allocator = allocator,
            .connector_id = connector_id,
            .capabilities = capabilities,
            .config = config,
            .calibrated = false,
            .panel_temperature = 25.0, // Room temperature
            .backlight_zones = backlight_zones,
            .response_time_lut = generateResponseTimeLUT(),
            .frame_times = std.mem.zeroes([120]f32),
            .frame_time_index = 0,
            .average_frametime = 16.67,
            .stuttering_detected = false,
        };
    }
    
    pub fn deinit(self: *GsyncDisplay) void {
        self.allocator.free(self.backlight_zones);
    }
    
    pub fn calibrate(self: *GsyncDisplay) !void {
        std.log.info("Calibrating G-SYNC display {} for optimal performance", .{self.connector_id});
        
        // Measure panel response time
        try self.measure_response_time();
        
        // Calibrate overdrive levels
        try self.calibrate_overdrive();
        
        // Set up variable overdrive if supported
        if (self.capabilities.supports_variable_overdrive) {
            try self.setup_variable_overdrive();
        }
        
        // Configure HDR if supported
        if (self.capabilities.supports_hdr and self.config.hdr_enabled) {
            try self.configure_hdr();
        }
        
        self.calibrated = true;
        std.log.info("G-SYNC display {} calibration complete", .{self.connector_id});
    }
    
    fn measure_response_time(self: *GsyncDisplay) !void {
        // Simulate panel response time measurement
        const base_response_time = self.capabilities.panel_type.getResponseTime();
        const temperature_factor = 1.0 + (self.panel_temperature - 25.0) * 0.01;
        
        // Generate response time LUT based on measured characteristics
        for (self.response_time_lut, 0..) |*value, i| {
            const normalized = @as(f32, @floatFromInt(i)) / 255.0;
            const response_time = base_response_time * temperature_factor * (1.0 + normalized * 0.5);
            value.* = @intFromFloat(std.math.clamp(response_time * 10.0, 0, 255));
        }
    }
    
    fn calibrate_overdrive(self: *GsyncDisplay) !void {
        // Optimize overdrive based on panel characteristics
        switch (self.capabilities.panel_type) {
            .tn => self.config.overdrive_level = .fast,
            .ips => self.config.overdrive_level = .normal,
            .va => self.config.overdrive_level = .faster,
            .oled => self.config.overdrive_level = .off, // OLED doesn't need overdrive
            else => self.config.overdrive_level = .normal,
        }
    }
    
    fn setup_variable_overdrive(self: *GsyncDisplay) !void {
        if (!self.capabilities.supports_variable_overdrive) return;
        
        self.config.adaptive_overdrive = true;
        std.log.info("Variable overdrive enabled for G-SYNC display {}", .{self.connector_id});
    }
    
    fn configure_hdr(self: *GsyncDisplay) !void {
        if (!self.capabilities.supports_hdr) return;
        
        // Configure HDR metadata
        self.config.peak_brightness_nits = switch (self.capabilities.backlight_type) {
            .mini_led => 4000,
            .micro_led => 10000,
            .oled_self_emissive => 1000,
            else => 1000,
        };
        
        // Set up local dimming zones
        if (self.capabilities.backlight_type.supportsLocalDimming()) {
            self.config.local_dimming_zones = @intCast(self.backlight_zones.len);
        }
        
        std.log.info("HDR configured: {} nits, {} zones", .{ self.config.peak_brightness_nits, self.config.local_dimming_zones });
    }
    
    pub fn update_frame_timing(self: *GsyncDisplay, frametime_ms: f32) void {
        // Update frame time history
        self.frame_times[self.frame_time_index] = frametime_ms;
        self.frame_time_index = (self.frame_time_index + 1) % self.frame_times.len;
        
        // Calculate average frame time
        var sum: f32 = 0;
        for (self.frame_times) |ft| {
            sum += ft;
        }
        self.average_frametime = sum / @as(f32, @floatFromInt(self.frame_times.len));
        
        // Detect stuttering
        var variance: f32 = 0;
        for (self.frame_times) |ft| {
            const diff = ft - self.average_frametime;
            variance += diff * diff;
        }
        variance /= @as(f32, @floatFromInt(self.frame_times.len));
        
        self.stuttering_detected = variance > 25.0; // 5ms standard deviation threshold
        
        // Update refresh rate for G-SYNC
        const optimal_rate = self.config.calculate_optimal_refresh_rate(@intFromFloat(frametime_ms * std.time.ns_per_ms));
        if (optimal_rate != self.config.current_refresh_hz) {
            self.config.update_refresh_rate(optimal_rate);
        }
        
        // Adjust overdrive based on refresh rate if variable overdrive is enabled
        if (self.config.adaptive_overdrive) {
            self.adjust_overdrive_for_refresh_rate(optimal_rate);
        }
    }
    
    fn adjust_overdrive_for_refresh_rate(self: *GsyncDisplay, refresh_rate: u32) void {
        // Higher refresh rates need less overdrive
        const overdrive_factor = @as(f32, @floatFromInt(self.config.max_refresh_hz)) / @as(f32, @floatFromInt(refresh_rate));
        
        if (overdrive_factor < 1.2) {
            self.config.overdrive_level = .extreme;
        } else if (overdrive_factor < 1.5) {
            self.config.overdrive_level = .faster;
        } else if (overdrive_factor < 2.0) {
            self.config.overdrive_level = .fast;
        } else {
            self.config.overdrive_level = .normal;
        }
    }
    
    pub fn enable_ultra_low_latency(self: *GsyncDisplay) void {
        if (!self.capabilities.supports_ultra_low_latency) return;
        
        self.config.ultra_low_latency = true;
        self.config.input_lag_optimization = true;
        
        // Optimize settings for minimal latency
        if (self.capabilities.mode == .gsync_esports) {
            self.config.motion_blur_reduction = true;
            self.config.hdr_enabled = false; // Disable HDR for minimum latency
        }
        
        std.log.info("Ultra Low Latency enabled for G-SYNC display {}", .{self.connector_id});
    }
    
    pub fn disable_ultra_low_latency(self: *GsyncDisplay) void {
        self.config.ultra_low_latency = false;
        self.config.input_lag_optimization = false;
        self.config.motion_blur_reduction = false;
        
        std.log.info("Ultra Low Latency disabled for G-SYNC display {}", .{self.connector_id});
    }
    
    pub fn update_local_dimming(self: *GsyncDisplay, hdr_metadata: []const u16) void {
        if (!self.capabilities.backlight_type.supportsLocalDimming()) return;
        if (hdr_metadata.len != self.backlight_zones.len) return;
        
        // Update backlight zones based on HDR metadata
        for (self.backlight_zones, hdr_metadata) |*zone, metadata| {
            zone.* = metadata;
        }
    }
    
    pub fn get_display_info(self: *GsyncDisplay) DisplayInfo {
        return DisplayInfo{
            .connector_id = self.connector_id,
            .gsync_mode = self.config.mode,
            .current_refresh_hz = self.config.current_refresh_hz,
            .average_frametime_ms = self.average_frametime,
            .stuttering_detected = self.stuttering_detected,
            .ultra_low_latency = self.config.ultra_low_latency,
            .hdr_enabled = self.config.hdr_enabled,
            .overdrive_level = self.config.overdrive_level,
            .panel_temperature = self.panel_temperature,
            .calibrated = self.calibrated,
        };
    }
};

pub const DisplayInfo = struct {
    connector_id: u32,
    gsync_mode: GsyncMode,
    current_refresh_hz: u32,
    average_frametime_ms: f32,
    stuttering_detected: bool,
    ultra_low_latency: bool,
    hdr_enabled: bool,
    overdrive_level: OverdriveLevel,
    panel_temperature: f32,
    calibrated: bool,
};

fn generateResponseTimeLUT() [256]u8 {
    var lut: [256]u8 = undefined;
    
    for (&lut, 0..) |*value, i| {
        // Generate a curve that represents typical LCD response characteristics
        const normalized = @as(f32, @floatFromInt(i)) / 255.0;
        const response = 1.0 - std.math.exp(-normalized * 5.0); // Exponential response curve
        value.* = @intFromFloat(response * 255.0);
    }
    
    return lut;
}

pub const GsyncManager = struct {
    allocator: Allocator,
    displays: std.ArrayList(GsyncDisplay),
    drm_driver: *drm.DrmDriver,
    rtx_optimizer: ?*@import("../rtx40/optimizations.zig").RTX40Optimizer,
    
    pub fn init(allocator: Allocator, drm_driver: *drm.DrmDriver) GsyncManager {
        return GsyncManager{
            .allocator = allocator,
            .displays = std.ArrayList(GsyncDisplay).init(allocator),
            .drm_driver = drm_driver,
            .rtx_optimizer = null,
        };
    }
    
    /// Set RTX 40 series optimizer for enhanced VRR/G-SYNC performance
    pub fn setRTXOptimizer(self: *GsyncManager, optimizer: *@import("../rtx40/optimizations.zig").RTX40Optimizer) void {
        self.rtx_optimizer = optimizer;
        std.log.info("RTX 40 series G-SYNC optimizations enabled");
    }
    
    pub fn deinit(self: *GsyncManager) void {
        for (self.displays.items) |*display| {
            display.deinit();
        }
        self.displays.deinit();
    }
    
    pub fn detect_gsync_displays(self: *GsyncManager) !void {
        const vrr_caps = self.drm_driver.get_vrr_capabilities();
        
        if (vrr_caps.connected_vrr_displays == 0) {
            std.log.warn("No G-SYNC compatible displays detected");
            return;
        }
        
        // Create G-SYNC displays for each connected VRR-capable display
        for (0..vrr_caps.connected_vrr_displays) |i| {
            const mode: GsyncMode = if (vrr_caps.supports_gsync) .gsync_certified else .gsync_compatible;
            
            var display = try GsyncDisplay.init(
                self.allocator,
                @intCast(i + 1),
                mode,
                vrr_caps.min_refresh_rate,
                vrr_caps.max_refresh_rate
            );
            
            try display.calibrate();
            try self.displays.append(display);
            
            std.log.info("G-SYNC display {} detected: {} ({}-{}Hz)", 
                         .{ i + 1, mode.toString(), vrr_caps.min_refresh_rate, vrr_caps.max_refresh_rate });
        }
    }
    
    pub fn enable_gsync(self: *GsyncManager, mode: GsyncMode) !void {
        for (self.displays.items) |*display| {
            display.config.mode = mode;
            
            // Apply RTX 40 series specific optimizations
            if (self.rtx_optimizer) |optimizer| {
                try self.applyRTX40VRROptimizations(optimizer, display, mode);
            }
            
            // Enable G-SYNC on the display
            try self.drm_driver.enable_vrr(mode, display.config.min_refresh_hz, display.config.max_refresh_hz);
            
            std.log.info("G-SYNC enabled: {} on display {}", .{ mode.toString(), display.connector_id });
        }
    }
    
    /// Apply RTX 40 series specific VRR and G-SYNC optimizations
    fn applyRTX40VRROptimizations(self: *GsyncManager, optimizer: *@import("../rtx40/optimizations.zig").RTX40Optimizer, display: *GsyncDisplay, mode: GsyncMode) !void {
        std.log.info("Applying RTX 40 series VRR optimizations for {}", .{mode});
        
        // Configure display engine for optimal VRR performance
        try optimizer.configureVRROptimizations(0);
        
        // Optimize for different G-SYNC modes
        switch (mode) {
            .gsync_ultimate => {
                // Ultimate mode: Maximum features with optimal performance
                display.config.hdr_enabled = true;
                display.config.peak_brightness_nits = 4000; // RTX 40 can drive high-end displays
                display.config.adaptive_overdrive = true;
                display.config.variable_overdrive = true;
                
                // Enable Ada-specific display optimizations
                try optimizer.enableDisplayCompression(0);
                try optimizer.optimizeDisplayCache(0);
                try optimizer.enableHDROptimizations(0);
                
                std.log.info("RTX 40 G-SYNC Ultimate optimizations applied");
            },
            .gsync_esports => {
                // Esports mode: Ultra-low latency above all else
                display.config.ultra_low_latency = true;
                display.config.motion_blur_reduction = true;
                display.config.input_lag_optimization = true;
                display.config.hdr_enabled = false; // Disable HDR for minimum latency
                
                // RTX 40 ultra-low latency optimizations
                try optimizer.enableGameModeScheduling(0);
                try optimizer.configureLatencyOptimizer(0);
                try optimizer.optimizeContextSwitching(0);
                
                // Optimize memory subsystem for lowest latency
                try self.optimizeMemoryForLatency(optimizer, display);
                
                std.log.info("RTX 40 G-SYNC Esports optimizations applied");
            },
            .gsync_certified => {
                // Certified mode: Balanced performance and quality
                display.config.adaptive_overdrive = true;
                display.config.variable_overdrive = true;
                
                // Balanced RTX 40 optimizations
                try optimizer.configureVRROptimizations(0);
                try optimizer.optimizeDisplayCache(0);
                
                std.log.info("RTX 40 G-SYNC Certified optimizations applied");
            },
            .gsync_compatible => {
                // Compatible mode: Basic VRR with good performance
                display.config.adaptive_overdrive = false; // More conservative
                
                // Basic RTX 40 VRR optimizations
                try optimizer.configureVRROptimizations(0);
                
                std.log.info("RTX 40 G-SYNC Compatible optimizations applied");
            },
            .disabled => {
                // No optimizations needed
            },
        }
        
        // Apply RTX 40 specific refresh rate optimizations
        try self.optimizeRefreshRateForRTX40(optimizer, display);
    }
    
    /// Optimize memory subsystem for ultra-low latency VRR
    fn optimizeMemoryForLatency(self: *GsyncManager, optimizer: *@import("../rtx40/optimizations.zig").RTX40Optimizer, display: *GsyncDisplay) !void {
        _ = self;
        _ = display;
        
        // Configure GDDR6X for minimum latency
        try optimizer.setMemoryClockOffset(0, 0); // No overclock for stable latency
        try optimizer.configurePrefetching(0, .maximum); // Aggressive prefetching
        try optimizer.enableMemoryCompression(0, true); // Reduce bandwidth pressure
        
        std.log.debug("Memory optimized for ultra-low latency VRR");
    }
    
    /// Optimize refresh rate handling for RTX 40 series
    fn optimizeRefreshRateForRTX40(self: *GsyncManager, optimizer: *@import("../rtx40/optimizations.zig").RTX40Optimizer, display: *GsyncDisplay) !void {
        _ = self;
        
        // Configure display engine for high refresh rates
        if (display.config.max_refresh_hz >= 240) {
            // 240Hz+ optimization
            try optimizer.enableDisplayCompression(0);
            try optimizer.optimizeDisplayCache(0);
            
            // Adjust overdrive for high refresh rates
            display.config.overdrive_level = .fast;
            
            std.log.info("RTX 40 optimized for {}Hz high refresh rate", .{display.config.max_refresh_hz});
        } else if (display.config.max_refresh_hz >= 165) {
            // 165Hz optimization
            display.config.overdrive_level = .normal;
            
            std.log.info("RTX 40 optimized for {}Hz standard high refresh", .{display.config.max_refresh_hz});
        }
        
        // Enable variable refresh rate optimizations
        if (display.config.mode != .disabled) {
            // Configure VRR-specific optimizations
            display.config.low_framerate_compensation = display.config.min_refresh_hz < 48;
            
            // RTX 40 can handle more aggressive LFC
            if (display.config.low_framerate_compensation) {
                std.log.info("RTX 40 Low Framerate Compensation enabled for {}-{}Hz range", 
                             .{display.config.min_refresh_hz, display.config.max_refresh_hz});
            }
        }
    }
    
    pub fn disable_gsync(self: *GsyncManager) void {
        for (self.displays.items) |*display| {
            display.config.mode = .disabled;
        }
        
        self.drm_driver.disable_vrr();
        std.log.info("G-SYNC disabled on all displays");
    }
    
    pub fn optimize_for_game(self: *GsyncManager, game_type: GameType) void {
        for (self.displays.items) |*display| {
            display.config.optimize_for_gaming(game_type);
            
            if (game_type == .competitive_fps) {
                display.enable_ultra_low_latency();
            }
        }
        
        std.log.info("G-SYNC optimized for {}", .{game_type});
    }
    
    pub fn update_frame_timing(self: *GsyncManager, frametime_ms: f32) !void {
        for (self.displays.items) |*display| {
            display.update_frame_timing(frametime_ms);
            
            // Update DRM driver with new refresh rate
            try self.drm_driver.set_refresh_rate(display.config.current_refresh_hz);
        }
    }
    
    pub fn get_all_display_info(self: *GsyncManager) ![]DisplayInfo {
        const info_list = try self.allocator.alloc(DisplayInfo, self.displays.items.len);
        
        for (self.displays.items, info_list) |*display, *info| {
            info.* = display.get_display_info();
        }
        
        return info_list;
    }
};

// Test functions
test "gsync configuration" {
    var config = GsyncConfig.init(.gsync_ultimate, 48, 240);
    try std.testing.expect(config.min_refresh_hz == 48);
    try std.testing.expect(config.max_refresh_hz == 240);
    try std.testing.expect(config.variable_overdrive);
    
    config.optimize_for_gaming(.competitive_fps);
    try std.testing.expect(config.ultra_low_latency);
    try std.testing.expect(config.motion_blur_reduction);
}

test "gsync display initialization" {
    const allocator = std.testing.allocator;
    
    var display = try GsyncDisplay.init(allocator, 1, .gsync_ultimate, 48, 240);
    defer display.deinit();
    
    try std.testing.expect(display.capabilities.mode == .gsync_ultimate);
    try std.testing.expect(display.capabilities.supports_hdr);
    try std.testing.expect(display.capabilities.supports_variable_overdrive);
}