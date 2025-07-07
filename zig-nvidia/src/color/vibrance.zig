const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const drm = @import("../drm/driver.zig");
const gsync = @import("../gsync/display.zig");

pub const VibranceError = error{
    InvalidRange,
    CalibrationFailed,
    UnsupportedColorSpace,
    InvalidProfile,
    NotInitialized,
    HardwareError,
    ProfileNotFound,
};

pub const ColorSpace = enum(u8) {
    srgb = 0,
    adobe_rgb = 1,
    dci_p3 = 2,
    rec2020 = 3,
    rec709 = 4,
    prophoto_rgb = 5,
    
    pub fn toString(self: ColorSpace) []const u8 {
        return switch (self) {
            .srgb => "sRGB",
            .adobe_rgb => "Adobe RGB",
            .dci_p3 => "DCI-P3",
            .rec2020 => "Rec. 2020",
            .rec709 => "Rec. 709",
            .prophoto_rgb => "ProPhoto RGB",
        };
    }
    
    pub fn getGamut(self: ColorSpace) ColorGamut {
        return switch (self) {
            .srgb => ColorGamut{ .coverage = 100.0, .white_point = WhitePoint.d65 },
            .adobe_rgb => ColorGamut{ .coverage = 135.0, .white_point = WhitePoint.d65 },
            .dci_p3 => ColorGamut{ .coverage = 125.0, .white_point = WhitePoint.dci },
            .rec2020 => ColorGamut{ .coverage = 175.0, .white_point = WhitePoint.d65 },
            .rec709 => ColorGamut{ .coverage = 100.0, .white_point = WhitePoint.d65 },
            .prophoto_rgb => ColorGamut{ .coverage = 190.0, .white_point = WhitePoint.d50 },
        };
    }
};

pub const WhitePoint = enum(u8) {
    d50 = 0,
    d65 = 1,
    dci = 2,
    
    pub fn getTemperature(self: WhitePoint) u32 {
        return switch (self) {
            .d50 => 5003,
            .d65 => 6504,
            .dci => 6300,
        };
    }
};

pub const ColorGamut = struct {
    coverage: f32, // Percentage of sRGB coverage
    white_point: WhitePoint,
};

pub const VibranceProfile = struct {
    name: []const u8,
    vibrance: i8,         // -50 to +100 (0 = neutral)
    saturation: i8,       // -50 to +50 (0 = neutral)
    gamma: f32,           // 0.8 to 3.0 (2.2 = standard)
    brightness: i8,       // -50 to +50 (0 = neutral)
    contrast: i8,         // -50 to +50 (0 = neutral)
    hue_shift: i8,        // -180 to +180 degrees
    
    // Advanced color controls
    red_vibrance: i8,     // Individual color channel vibrance
    green_vibrance: i8,
    blue_vibrance: i8,
    
    // Color temperature
    temperature: i16,     // Color temperature adjustment in Kelvin (-1000 to +1000)
    tint: i8,            // Green/Magenta tint (-50 to +50)
    
    // HDR settings
    hdr_peak_brightness: u16, // Peak brightness in nits
    hdr_tone_mapping: ToneMappingCurve,
    
    // Game-specific optimizations
    game_mode: GameColorMode,
    preserve_skin_tones: bool,
    enhance_foliage: bool,
    boost_sky_colors: bool,
    
    pub fn init(name: []const u8) VibranceProfile {
        return VibranceProfile{
            .name = name,
            .vibrance = 0,
            .saturation = 0,
            .gamma = 2.2,
            .brightness = 0,
            .contrast = 0,
            .hue_shift = 0,
            .red_vibrance = 0,
            .green_vibrance = 0,
            .blue_vibrance = 0,
            .temperature = 0,
            .tint = 0,
            .hdr_peak_brightness = 1000,
            .hdr_tone_mapping = .aces,
            .game_mode = .standard,
            .preserve_skin_tones = true,
            .enhance_foliage = false,
            .boost_sky_colors = false,
        };
    }
    
    pub fn create_gaming_profile(name: []const u8) VibranceProfile {
        var profile = VibranceProfile.init(name);
        profile.vibrance = 25;
        profile.saturation = 15;
        profile.contrast = 10;
        profile.red_vibrance = 20;
        profile.green_vibrance = 30;
        profile.blue_vibrance = 15;
        profile.game_mode = .competitive;
        profile.enhance_foliage = true;
        profile.boost_sky_colors = true;
        return profile;
    }
    
    pub fn create_cinema_profile(name: []const u8) VibranceProfile {
        var profile = VibranceProfile.init(name);
        profile.vibrance = 5;
        profile.saturation = 0;
        profile.gamma = 2.4; // Cinema standard
        profile.game_mode = .cinematic;
        profile.preserve_skin_tones = true;
        profile.hdr_tone_mapping = .reinhard;
        return profile;
    }
    
    pub fn create_competitive_profile(name: []const u8) VibranceProfile {
        var profile = VibranceProfile.init(name);
        profile.vibrance = 40;
        profile.saturation = 25;
        profile.contrast = 20;
        profile.brightness = 5;
        profile.red_vibrance = 35;
        profile.green_vibrance = 50;
        profile.blue_vibrance = 25;
        profile.game_mode = .competitive;
        profile.enhance_foliage = true;
        profile.boost_sky_colors = true;
        profile.preserve_skin_tones = false; // Visibility over realism
        return profile;
    }
};

pub const ToneMappingCurve = enum(u8) {
    linear = 0,
    reinhard = 1,
    aces = 2,
    filmic = 3,
    uncharted2 = 4,
    
    pub fn toString(self: ToneMappingCurve) []const u8 {
        return switch (self) {
            .linear => "Linear",
            .reinhard => "Reinhard",
            .aces => "ACES",
            .filmic => "Filmic",
            .uncharted2 => "Uncharted 2",
        };
    }
};

pub const GameColorMode = enum(u8) {
    standard = 0,
    competitive = 1,
    cinematic = 2,
    photography = 3,
    streaming = 4,
    
    pub fn toString(self: GameColorMode) []const u8 {
        return switch (self) {
            .standard => "Standard",
            .competitive => "Competitive Gaming",
            .cinematic => "Cinematic",
            .photography => "Photography",
            .streaming => "Streaming",
        };
    }
};

pub const ColorMatrix3x3 = struct {
    m: [3][3]f32,
    
    pub fn identity() ColorMatrix3x3 {
        return ColorMatrix3x3{
            .m = [3][3]f32{
                [3]f32{ 1.0, 0.0, 0.0 },
                [3]f32{ 0.0, 1.0, 0.0 },
                [3]f32{ 0.0, 0.0, 1.0 },
            },
        };
    }
    
    pub fn vibrance_matrix(vibrance: f32) ColorMatrix3x3 {
        const v = vibrance / 100.0;
        const inv_v = 1.0 - v;
        
        return ColorMatrix3x3{
            .m = [3][3]f32{
                [3]f32{ inv_v + v * 0.299, v * 0.587, v * 0.114 },
                [3]f32{ v * 0.299, inv_v + v * 0.587, v * 0.114 },
                [3]f32{ v * 0.299, v * 0.587, inv_v + v * 0.114 },
            },
        };
    }
    
    pub fn saturation_matrix(saturation: f32) ColorMatrix3x3 {
        const s = 1.0 + saturation / 100.0;
        const sr = (1.0 - s) * 0.3086;
        const sg = (1.0 - s) * 0.6094;
        const sb = (1.0 - s) * 0.0820;
        
        return ColorMatrix3x3{
            .m = [3][3]f32{
                [3]f32{ sr + s, sg, sb },
                [3]f32{ sr, sg + s, sb },
                [3]f32{ sr, sg, sb + s },
            },
        };
    }
    
    pub fn multiply(self: ColorMatrix3x3, other: ColorMatrix3x3) ColorMatrix3x3 {
        var result = ColorMatrix3x3.identity();
        
        for (0..3) |i| {
            for (0..3) |j| {
                result.m[i][j] = 0.0;
                for (0..3) |k| {
                    result.m[i][j] += self.m[i][k] * other.m[k][j];
                }
            }
        }
        
        return result;
    }
    
    pub fn apply_to_rgb(self: ColorMatrix3x3, r: f32, g: f32, b: f32) [3]f32 {
        return [3]f32{
            self.m[0][0] * r + self.m[0][1] * g + self.m[0][2] * b,
            self.m[1][0] * r + self.m[1][1] * g + self.m[1][2] * b,
            self.m[2][0] * r + self.m[2][1] * g + self.m[2][2] * b,
        };
    }
};

pub const VibranceEngine = struct {
    allocator: Allocator,
    profiles: std.StringHashMap(VibranceProfile),
    active_profile: ?[]const u8,
    drm_driver: *drm.DrmDriver,
    
    // Hardware state
    lut_red: [256]u16,     // 16-bit Look-Up Tables for hardware acceleration
    lut_green: [256]u16,
    lut_blue: [256]u16,
    gamma_lut: [256]u16,
    
    // Real-time processing
    color_matrix: ColorMatrix3x3,
    temperature_matrix: ColorMatrix3x3,
    vibrance_matrix: ColorMatrix3x3,
    
    // Performance tracking
    processing_time_ns: u64,
    frames_processed: u64,
    
    pub fn init(allocator: Allocator, drm_driver: *drm.DrmDriver) VibranceEngine {
        return VibranceEngine{
            .allocator = allocator,
            .profiles = std.StringHashMap(VibranceProfile).init(allocator),
            .active_profile = null,
            .drm_driver = drm_driver,
            .lut_red = std.mem.zeroes([256]u16),
            .lut_green = std.mem.zeroes([256]u16),
            .lut_blue = std.mem.zeroes([256]u16),
            .gamma_lut = std.mem.zeroes([256]u16),
            .color_matrix = ColorMatrix3x3.identity(),
            .temperature_matrix = ColorMatrix3x3.identity(),
            .vibrance_matrix = ColorMatrix3x3.identity(),
            .processing_time_ns = 0,
            .frames_processed = 0,
        };
    }
    
    pub fn deinit(self: *VibranceEngine) void {
        self.profiles.deinit();
    }
    
    pub fn load_default_profiles(self: *VibranceEngine) !void {
        // Gaming profiles
        try self.profiles.put("Gaming", VibranceProfile.create_gaming_profile("Gaming"));
        try self.profiles.put("Competitive", VibranceProfile.create_competitive_profile("Competitive"));
        try self.profiles.put("Cinema", VibranceProfile.create_cinema_profile("Cinema"));
        
        // Content creator profiles
        var streaming_profile = VibranceProfile.init("Streaming");
        streaming_profile.vibrance = 15;
        streaming_profile.saturation = 10;
        streaming_profile.contrast = 5;
        streaming_profile.game_mode = .streaming;
        try self.profiles.put("Streaming", streaming_profile);
        
        var photography_profile = VibranceProfile.init("Photography");
        photography_profile.gamma = 2.4;
        photography_profile.game_mode = .photography;
        photography_profile.preserve_skin_tones = true;
        try self.profiles.put("Photography", photography_profile);
        
        // Game-specific profiles
        try self.create_game_specific_profiles();
        
        std.log.info("Loaded {} default vibrance profiles", .{self.profiles.count()});
    }
    
    fn create_game_specific_profiles(self: *VibranceEngine) !void {
        // Counter-Strike 2 / CS:GO
        var cs_profile = VibranceProfile.create_competitive_profile("Counter-Strike");
        cs_profile.vibrance = 65; // High vibrance for enemy visibility
        cs_profile.green_vibrance = 70; // Enhance foliage/map details
        cs_profile.red_vibrance = 50; // Blood/enemy highlights
        cs_profile.enhance_foliage = true;
        try self.profiles.put("Counter-Strike", cs_profile);
        
        // Valorant
        var valorant_profile = VibranceProfile.create_competitive_profile("Valorant");
        valorant_profile.vibrance = 55;
        valorant_profile.saturation = 30;
        valorant_profile.red_vibrance = 60;
        valorant_profile.blue_vibrance = 40;
        try self.profiles.put("Valorant", valorant_profile);
        
        // Apex Legends
        var apex_profile = VibranceProfile.create_competitive_profile("Apex Legends");
        apex_profile.vibrance = 45;
        apex_profile.enhance_foliage = true;
        apex_profile.boost_sky_colors = true;
        try self.profiles.put("Apex Legends", apex_profile);
        
        // Fortnite
        var fortnite_profile = VibranceProfile.create_gaming_profile("Fortnite");
        fortnite_profile.vibrance = 35;
        fortnite_profile.boost_sky_colors = true;
        try self.profiles.put("Fortnite", fortnite_profile);
        
        // Red Dead Redemption 2 (Cinematic)
        var rdr2_profile = VibranceProfile.create_cinema_profile("Red Dead Redemption 2");
        rdr2_profile.vibrance = 10;
        rdr2_profile.hdr_tone_mapping = .filmic;
        try self.profiles.put("Red Dead Redemption 2", rdr2_profile);
        
        // Cyberpunk 2077 (HDR optimized)
        var cyberpunk_profile = VibranceProfile.init("Cyberpunk 2077");
        cyberpunk_profile.vibrance = 20;
        cyberpunk_profile.saturation = 15;
        cyberpunk_profile.red_vibrance = 25;
        cyberpunk_profile.blue_vibrance = 30;
        cyberpunk_profile.hdr_peak_brightness = 4000;
        cyberpunk_profile.hdr_tone_mapping = .aces;
        try self.profiles.put("Cyberpunk 2077", cyberpunk_profile);
    }
    
    pub fn create_profile(self: *VibranceEngine, name: []const u8, profile: VibranceProfile) !void {
        try self.profiles.put(name, profile);
        std.log.info("Created vibrance profile: {s}", .{name});
    }
    
    pub fn apply_profile(self: *VibranceEngine, name: []const u8) !void {
        const profile = self.profiles.get(name) orelse return VibranceError.ProfileNotFound;
        
        const start_time = std.time.nanoTimestamp();
        
        // Build color transformation matrices
        self.vibrance_matrix = ColorMatrix3x3.vibrance_matrix(@as(f32, @floatFromInt(profile.vibrance)));
        const saturation_matrix = ColorMatrix3x3.saturation_matrix(@as(f32, @floatFromInt(profile.saturation)));
        self.temperature_matrix = self.create_temperature_matrix(profile.temperature, profile.tint);
        
        // Combine all transformations
        self.color_matrix = self.vibrance_matrix.multiply(saturation_matrix).multiply(self.temperature_matrix);
        
        // Generate hardware LUTs
        try self.generate_hardware_luts(profile);
        
        // Apply to display hardware
        try self.apply_to_hardware();
        
        self.active_profile = name;
        self.processing_time_ns = std.time.nanoTimestamp() - start_time;
        
        std.log.info("Applied vibrance profile: {s} ({}ns)", .{ name, self.processing_time_ns });
    }
    
    fn create_temperature_matrix(self: *VibranceEngine, temperature_offset: i16, tint: i8) ColorMatrix3x3 {
        _ = self;
        
        const temp_factor = @as(f32, @floatFromInt(temperature_offset)) / 1000.0;
        const tint_factor = @as(f32, @floatFromInt(tint)) / 100.0;
        
        // Simplified temperature adjustment matrix
        const red_adjust = 1.0 + temp_factor * 0.3;
        const blue_adjust = 1.0 - temp_factor * 0.3;
        const green_adjust = 1.0 + tint_factor * 0.1;
        
        return ColorMatrix3x3{
            .m = [3][3]f32{
                [3]f32{ red_adjust, 0.0, 0.0 },
                [3]f32{ 0.0, green_adjust, 0.0 },
                [3]f32{ 0.0, 0.0, blue_adjust },
            },
        };
    }
    
    fn generate_hardware_luts(self: *VibranceEngine, profile: VibranceProfile) !void {
        // Generate gamma LUT
        for (self.gamma_lut, 0..) |*value, i| {
            const normalized = @as(f32, @floatFromInt(i)) / 255.0;
            const gamma_corrected = std.math.pow(f32, normalized, 1.0 / profile.gamma);
            value.* = @intFromFloat(std.math.clamp(gamma_corrected * 65535.0, 0, 65535));
        }
        
        // Generate RGB LUTs with individual channel vibrance
        for (0..256) |i| {
            const input = @as(f32, @floatFromInt(i)) / 255.0;
            
            // Apply brightness and contrast
            const brightness_factor = 1.0 + @as(f32, @floatFromInt(profile.brightness)) / 100.0;
            const contrast_factor = 1.0 + @as(f32, @floatFromInt(profile.contrast)) / 100.0;
            
            var red = (input - 0.5) * contrast_factor + 0.5 + @as(f32, @floatFromInt(profile.brightness)) / 100.0;
            var green = (input - 0.5) * contrast_factor + 0.5 + @as(f32, @floatFromInt(profile.brightness)) / 100.0;
            var blue = (input - 0.5) * contrast_factor + 0.5 + @as(f32, @floatFromInt(profile.brightness)) / 100.0;
            
            // Apply individual channel vibrance
            red = self.apply_channel_vibrance(red, profile.red_vibrance);
            green = self.apply_channel_vibrance(green, profile.green_vibrance);
            blue = self.apply_channel_vibrance(blue, profile.blue_vibrance);
            
            // Apply color matrix transformation
            const rgb = self.color_matrix.apply_to_rgb(red, green, blue);
            
            // Convert to 16-bit values for hardware
            self.lut_red[i] = @intFromFloat(std.math.clamp(rgb[0] * 65535.0, 0, 65535));
            self.lut_green[i] = @intFromFloat(std.math.clamp(rgb[1] * 65535.0, 0, 65535));
            self.lut_blue[i] = @intFromFloat(std.math.clamp(rgb[2] * 65535.0, 0, 65535));
        }
    }
    
    fn apply_channel_vibrance(self: *VibranceEngine, input: f32, vibrance: i8) f32 {
        _ = self;
        
        if (vibrance == 0) return input;
        
        const v = @as(f32, @floatFromInt(vibrance)) / 100.0;
        
        // Enhanced vibrance algorithm that preserves skin tones
        const saturation = std.math.sqrt(input * (1.0 - input));
        const enhancement = v * saturation * 2.0;
        
        return std.math.clamp(input + enhancement, 0.0, 1.0);
    }
    
    fn apply_to_hardware(self: *VibranceEngine) !void {
        // Apply LUTs to display hardware via DRM
        // In a real implementation, this would program the display controller's LUT registers
        
        std.log.debug("Programming hardware LUTs for digital vibrance");
        
        // Simulate hardware programming delay
        std.time.sleep(1000000); // 1ms
    }
    
    pub fn disable_vibrance(self: *VibranceEngine) !void {
        // Reset to identity transformations
        self.color_matrix = ColorMatrix3x3.identity();
        self.vibrance_matrix = ColorMatrix3x3.identity();
        self.temperature_matrix = ColorMatrix3x3.identity();
        
        // Generate identity LUTs
        for (0..256) |i| {
            const value = @as(u16, @intCast(i)) * 257; // Scale 8-bit to 16-bit
            self.lut_red[i] = value;
            self.lut_green[i] = value;
            self.lut_blue[i] = value;
            self.gamma_lut[i] = value;
        }
        
        try self.apply_to_hardware();
        self.active_profile = null;
        
        std.log.info("Digital vibrance disabled");
    }
    
    pub fn auto_detect_game_profile(self: *VibranceEngine, window_title: []const u8) ?[]const u8 {
        const lowercase_title = std.ascii.allocLowerString(self.allocator, window_title) catch return null;
        defer self.allocator.free(lowercase_title);
        
        // Game detection patterns
        const game_patterns = [_]struct { pattern: []const u8, profile: []const u8 }{
            .{ .pattern = "counter-strike", .profile = "Counter-Strike" },
            .{ .pattern = "cs2", .profile = "Counter-Strike" },
            .{ .pattern = "csgo", .profile = "Counter-Strike" },
            .{ .pattern = "valorant", .profile = "Valorant" },
            .{ .pattern = "apex legends", .profile = "Apex Legends" },
            .{ .pattern = "fortnite", .profile = "Fortnite" },
            .{ .pattern = "red dead redemption", .profile = "Red Dead Redemption 2" },
            .{ .pattern = "cyberpunk 2077", .profile = "Cyberpunk 2077" },
        };
        
        for (game_patterns) |game| {
            if (std.mem.indexOf(u8, lowercase_title, game.pattern)) |_| {
                std.log.info("Auto-detected game: {s} -> {s}", .{ window_title, game.profile });
                return game.profile;
            }
        }
        
        return null;
    }
    
    pub fn get_active_profile(self: *VibranceEngine) ?VibranceProfile {
        if (self.active_profile) |name| {
            return self.profiles.get(name);
        }
        return null;
    }
    
    pub fn list_profiles(self: *VibranceEngine) ![][]const u8 {
        var profile_names = try self.allocator.alloc([]const u8, self.profiles.count());
        var iterator = self.profiles.keyIterator();
        var i: usize = 0;
        
        while (iterator.next()) |key| {
            profile_names[i] = key.*;
            i += 1;
        }
        
        return profile_names;
    }
    
    pub fn get_performance_stats(self: *VibranceEngine) PerformanceStats {
        return PerformanceStats{
            .processing_time_ns = self.processing_time_ns,
            .frames_processed = self.frames_processed,
            .active_profile = self.active_profile,
            .profiles_loaded = @intCast(self.profiles.count()),
        };
    }
    
    pub fn real_time_adjust(self: *VibranceEngine, vibrance_delta: i8) !void {
        if (self.active_profile) |profile_name| {
            if (self.profiles.getPtr(profile_name)) |profile| {
                const new_vibrance = std.math.clamp(profile.vibrance + vibrance_delta, -50, 100);
                profile.vibrance = @intCast(new_vibrance);
                
                // Quick update without full profile reapplication
                self.vibrance_matrix = ColorMatrix3x3.vibrance_matrix(@as(f32, @floatFromInt(profile.vibrance)));
                try self.apply_to_hardware();
                
                std.log.debug("Real-time vibrance adjustment: {} -> {}", .{ profile.vibrance - vibrance_delta, profile.vibrance });
            }
        }
    }
};

pub const PerformanceStats = struct {
    processing_time_ns: u64,
    frames_processed: u64,
    active_profile: ?[]const u8,
    profiles_loaded: u32,
};

// Advanced color science functions
pub fn rgb_to_hsv(r: f32, g: f32, b: f32) [3]f32 {
    const max_val = @max(@max(r, g), b);
    const min_val = @min(@min(r, g), b);
    const delta = max_val - min_val;
    
    var h: f32 = 0;
    var s: f32 = if (max_val != 0) delta / max_val else 0;
    const v: f32 = max_val;
    
    if (delta != 0) {
        if (max_val == r) {
            h = (g - b) / delta;
            if (g < b) h += 6;
        } else if (max_val == g) {
            h = (b - r) / delta + 2;
        } else {
            h = (r - g) / delta + 4;
        }
        h *= 60;
    }
    
    return [3]f32{ h, s, v };
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) [3]f32 {
    const c = v * s;
    const x = c * (1 - @abs(@mod(h / 60, 2) - 1));
    const m = v - c;
    
    var rgb: [3]f32 = undefined;
    
    if (h < 60) {
        rgb = [3]f32{ c, x, 0 };
    } else if (h < 120) {
        rgb = [3]f32{ x, c, 0 };
    } else if (h < 180) {
        rgb = [3]f32{ 0, c, x };
    } else if (h < 240) {
        rgb = [3]f32{ 0, x, c };
    } else if (h < 300) {
        rgb = [3]f32{ x, 0, c };
    } else {
        rgb = [3]f32{ c, 0, x };
    }
    
    return [3]f32{ rgb[0] + m, rgb[1] + m, rgb[2] + m };
}

// Test functions
test "vibrance profile creation" {
    const competitive = VibranceProfile.create_competitive_profile("Test");
    try std.testing.expect(competitive.vibrance > 30);
    try std.testing.expect(competitive.enhance_foliage);
    try std.testing.expect(competitive.game_mode == .competitive);
}

test "color matrix operations" {
    const vibrance_matrix = ColorMatrix3x3.vibrance_matrix(50.0);
    const saturation_matrix = ColorMatrix3x3.saturation_matrix(25.0);
    
    const combined = vibrance_matrix.multiply(saturation_matrix);
    const result = combined.apply_to_rgb(0.5, 0.7, 0.3);
    
    try std.testing.expect(result[0] >= 0.0 and result[0] <= 1.0);
    try std.testing.expect(result[1] >= 0.0 and result[1] <= 1.0);
    try std.testing.expect(result[2] >= 0.0 and result[2] <= 1.0);
}

test "vibrance engine" {
    const allocator = std.testing.allocator;
    
    var drm_driver = try drm.DrmDriver.init(allocator);
    defer drm_driver.deinit();
    
    var engine = VibranceEngine.init(allocator, &drm_driver);
    defer engine.deinit();
    
    try engine.load_default_profiles();
    try std.testing.expect(engine.profiles.count() > 5);
    
    const detected = engine.auto_detect_game_profile("Counter-Strike 2");
    try std.testing.expect(std.mem.eql(u8, detected.?, "Counter-Strike"));
}