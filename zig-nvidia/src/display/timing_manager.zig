const std = @import("std");
const display = @import("engine.zig");
const drm = @import("../drm/driver.zig");

/// Advanced Display Timing and EDID Management
/// Handles custom timings, HDR metadata, and DisplayID 2.0 tiled displays

pub const TimingError = error{
    InvalidTiming,
    UnsupportedRefreshRate,
    EdidParsingFailed,
    DisplayIdParsingFailed,
    TiledDisplayNotSupported,
    HdrMetadataInvalid,
    CustomTimingRejected,
    OutOfMemory,
};

pub const DisplayTiming = struct {
    // Basic timing parameters
    pixel_clock_khz: u32,
    h_addressable: u16,
    h_blanking: u16,
    h_front_porch: u16,
    h_sync_width: u16,
    h_back_porch: u16,
    v_addressable: u16,
    v_blanking: u16,
    v_front_porch: u16,
    v_sync_width: u16,
    v_back_porch: u16,
    
    // Sync polarities and scan type
    h_sync_positive: bool,
    v_sync_positive: bool,
    interlaced: bool,
    
    // Advanced features
    stereo_mode: StereoMode,
    color_depth: ColorDepth,
    colorimetry: Colorimetry,
    eotf: ElectroOpticalTransferFunction,
    
    // Variable refresh rate
    vrr_capable: bool,
    vrr_min_hz: u16,
    vrr_max_hz: u16,
    
    // HDR capabilities
    hdr_capable: bool,
    max_luminance_nits: u16,
    min_luminance_nits: f32,
    max_frame_average_nits: u16,
    
    pub fn calculateRefreshRate(self: DisplayTiming) f32 {
        const h_total = self.h_addressable + self.h_blanking;
        const v_total = self.v_addressable + self.v_blanking;
        const total_pixels = @as(u32, h_total) * @as(u32, v_total);
        
        return (@as(f32, @floatFromInt(self.pixel_clock_khz)) * 1000.0) / @as(f32, @floatFromInt(total_pixels));
    }
    
    pub fn calculateBandwidth(self: DisplayTiming) u64 {
        const bits_per_pixel = @as(u32, self.color_depth.getBitsPerPixel());
        return (@as(u64, self.pixel_clock_khz) * 1000 * bits_per_pixel) / 8; // Bytes per second
    }
    
    pub fn isWithinLimits(self: DisplayTiming, max_pixel_clock_khz: u32, max_bandwidth_gbps: f32) bool {
        const bandwidth_gbps = @as(f32, @floatFromInt(self.calculateBandwidth())) / 1_000_000_000.0;
        return self.pixel_clock_khz <= max_pixel_clock_khz and bandwidth_gbps <= max_bandwidth_gbps;
    }
    
    pub fn validateTiming(self: DisplayTiming) bool {
        // Basic validation checks
        if (self.h_addressable == 0 or self.v_addressable == 0) return false;
        if (self.h_sync_width == 0 or self.v_sync_width == 0) return false;
        if (self.pixel_clock_khz == 0) return false;
        
        // Check reasonable refresh rate range (30-500Hz)
        const refresh_rate = self.calculateRefreshRate();
        if (refresh_rate < 30.0 or refresh_rate > 500.0) return false;
        
        return true;
    }
};

pub const StereoMode = enum(u8) {
    none = 0,
    frame_sequential = 1,
    side_by_side = 2,
    top_bottom = 3,
    line_interleaved = 4,
    pixel_interleaved = 5,
    checkerboard = 6,
};

pub const ColorDepth = enum(u8) {
    bpc_6 = 6,
    bpc_8 = 8,
    bpc_10 = 10,
    bpc_12 = 12,
    bpc_14 = 14,
    bpc_16 = 16,
    
    pub fn getBitsPerPixel(self: ColorDepth) u8 {
        return @intFromEnum(self) * 3; // RGB channels
    }
    
    pub fn getMaxValue(self: ColorDepth) u32 {
        return (@as(u32, 1) << @intFromEnum(self)) - 1;
    }
};

pub const Colorimetry = enum(u8) {
    bt601 = 0,      // SDTV
    bt709 = 1,      // HDTV (sRGB)
    bt2020 = 2,     // UHDTV
    dci_p3 = 3,     // Digital Cinema
    adobe_rgb = 4,  // Adobe RGB
    
    pub fn getGamutCoverage(self: Colorimetry) f32 {
        return switch (self) {
            .bt601 => 0.45,      // 45% of visible spectrum
            .bt709 => 0.35,      // 35% (sRGB)
            .bt2020 => 0.76,     // 76% (wide gamut)
            .dci_p3 => 0.54,     // 54% (cinema)
            .adobe_rgb => 0.52,  // 52% (photo)
        };
    }
};

pub const ElectroOpticalTransferFunction = enum(u8) {
    bt709 = 0,      // Standard gamma 2.4
    pq = 1,         // Perceptual Quantizer (HDR10)
    hlg = 2,        // Hybrid Log-Gamma (HDR10+)
    linear = 3,     // Linear light
    
    pub fn isHdr(self: ElectroOpticalTransferFunction) bool {
        return self == .pq or self == .hlg;
    }
};

pub const EdidExtension = struct {
    tag: u8,
    revision: u8,
    data: []u8,
    
    pub const CEA_861_TAG: u8 = 0x02;
    pub const DISPLAY_ID_TAG: u8 = 0x70;
    pub const DISPLAY_ID_2_TAG: u8 = 0x81;
};

pub const HdrMetadata = struct {
    // Static HDR Metadata (HDR10)
    max_display_mastering_luminance: u16,  // cd/m²
    min_display_mastering_luminance: u16,  // 0.0001 cd/m² units
    max_content_light_level: u16,          // cd/m²
    max_frame_average_light_level: u16,    // cd/m²
    
    // Color primaries (CIE 1931 xy chromaticity)
    red_primary_x: u16,     // x * 50000
    red_primary_y: u16,     // y * 50000
    green_primary_x: u16,
    green_primary_y: u16,
    blue_primary_x: u16,
    blue_primary_y: u16,
    white_point_x: u16,
    white_point_y: u16,
    
    pub fn isValid(self: HdrMetadata) bool {
        // Basic validation of HDR metadata
        return self.max_display_mastering_luminance > 0 and
               self.max_display_mastering_luminance >= self.max_content_light_level and
               self.red_primary_x <= 50000 and self.red_primary_y <= 50000 and
               self.green_primary_x <= 50000 and self.green_primary_y <= 50000 and
               self.blue_primary_x <= 50000 and self.blue_primary_y <= 50000;
    }
    
    pub fn getBt2020Coverage(self: HdrMetadata) f32 {
        // Calculate how much of BT.2020 gamut is covered by these primaries
        // Simplified calculation
        const red_coverage = (@as(f32, @floatFromInt(self.red_primary_x)) / 50000.0) * 
                            (@as(f32, @floatFromInt(self.red_primary_y)) / 50000.0);
        const green_coverage = (@as(f32, @floatFromInt(self.green_primary_x)) / 50000.0) * 
                              (@as(f32, @floatFromInt(self.green_primary_y)) / 50000.0);
        const blue_coverage = (@as(f32, @floatFromInt(self.blue_primary_x)) / 50000.0) * 
                             (@as(f32, @floatFromInt(self.blue_primary_y)) / 50000.0);
        
        return (red_coverage + green_coverage + blue_coverage) / 3.0;
    }
};

pub const TiledDisplay = struct {
    tile_count_h: u8,
    tile_count_v: u8,
    tile_location_h: u8,
    tile_location_v: u8,
    native_resolution_h: u16,
    native_resolution_v: u16,
    bezel_left: u8,
    bezel_top: u8,
    bezel_right: u8,
    bezel_bottom: u8,
    
    pub fn getTotalResolution(self: TiledDisplay) struct { width: u32, height: u32 } {
        return .{
            .width = @as(u32, self.native_resolution_h) * @as(u32, self.tile_count_h),
            .height = @as(u32, self.native_resolution_v) * @as(u32, self.tile_count_v),
        };
    }
    
    pub fn isSingleTile(self: TiledDisplay) bool {
        return self.tile_count_h == 1 and self.tile_count_v == 1;
    }
    
    pub fn getTileIndex(self: TiledDisplay) u8 {
        return self.tile_location_v * self.tile_count_h + self.tile_location_h;
    }
};

pub const ParsedEdid = struct {
    allocator: std.mem.Allocator,
    
    // Basic display info
    manufacturer_id: [3]u8,
    product_code: u16,
    serial_number: u32,
    manufacture_week: u8,
    manufacture_year: u16,
    
    // Display characteristics  
    display_width_cm: u8,
    display_height_cm: u8,
    gamma: u8,            // Gamma * 100 (e.g., 220 = 2.20)
    
    // Supported features
    dpms_standby: bool,
    dpms_suspend: bool,
    dpms_off: bool,
    digital_input: bool,
    color_depth: ColorDepth,
    
    // Standard timings
    preferred_timing: ?DisplayTiming,
    standard_timings: std.ArrayList(DisplayTiming),
    detailed_timings: std.ArrayList(DisplayTiming),
    
    // Extensions
    extensions: std.ArrayList(EdidExtension),
    
    // HDR and advanced features
    hdr_metadata: ?HdrMetadata,
    vrr_capable: bool,
    vrr_range: struct { min: u16, max: u16 },
    
    // Tiled display info (DisplayID 2.0)
    tiled_display: ?TiledDisplay,
    
    pub fn init(allocator: std.mem.Allocator) ParsedEdid {
        return ParsedEdid{
            .allocator = allocator,
            .manufacturer_id = [_]u8{0} ** 3,
            .product_code = 0,
            .serial_number = 0,
            .manufacture_week = 0,
            .manufacture_year = 0,
            .display_width_cm = 0,
            .display_height_cm = 0,
            .gamma = 220, // Default 2.2 gamma
            .dpms_standby = false,
            .dpms_suspend = false,
            .dpms_off = false,
            .digital_input = false,
            .color_depth = .bpc_8,
            .preferred_timing = null,
            .standard_timings = std.ArrayList(DisplayTiming).init(allocator),
            .detailed_timings = std.ArrayList(DisplayTiming).init(allocator),
            .extensions = std.ArrayList(EdidExtension).init(allocator),
            .hdr_metadata = null,
            .vrr_capable = false,
            .vrr_range = .{ .min = 0, .max = 0 },
            .tiled_display = null,
        };
    }
    
    pub fn deinit(self: *ParsedEdid) void {
        self.standard_timings.deinit();
        self.detailed_timings.deinit();
        
        for (self.extensions.items) |ext| {
            self.allocator.free(ext.data);
        }
        self.extensions.deinit();
    }
    
    pub fn getMaxResolution(self: *const ParsedEdid) struct { width: u16, height: u16 } {
        var max_width: u16 = 0;
        var max_height: u16 = 0;
        
        if (self.preferred_timing) |timing| {
            max_width = timing.h_addressable;
            max_height = timing.v_addressable;
        }
        
        for (self.detailed_timings.items) |timing| {
            if (timing.h_addressable > max_width or timing.v_addressable > max_height) {
                max_width = timing.h_addressable;
                max_height = timing.v_addressable;
            }
        }
        
        return .{ .width = max_width, .height = max_height };
    }
    
    pub fn getMaxRefreshRate(self: *const ParsedEdid) f32 {
        var max_refresh: f32 = 0.0;
        
        if (self.preferred_timing) |timing| {
            max_refresh = timing.calculateRefreshRate();
        }
        
        for (self.detailed_timings.items) |timing| {
            const refresh = timing.calculateRefreshRate();
            if (refresh > max_refresh) {
                max_refresh = refresh;
            }
        }
        
        return max_refresh;
    }
    
    pub fn supportsHdr(self: *const ParsedEdid) bool {
        return self.hdr_metadata != null;
    }
    
    pub fn isTiledDisplay(self: *const ParsedEdid) bool {
        return self.tiled_display != null and !self.tiled_display.?.isSingleTile();
    }
};

pub const TimingManager = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    custom_timings: std.ArrayList(DisplayTiming),
    edid_cache: std.AutoHashMap(u32, ParsedEdid), // display_id -> parsed EDID
    timing_overrides: std.AutoHashMap(u32, DisplayTiming), // display_id -> custom timing
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .custom_timings = std.ArrayList(DisplayTiming).init(allocator),
            .edid_cache = std.AutoHashMap(u32, ParsedEdid).init(allocator),
            .timing_overrides = std.AutoHashMap(u32, DisplayTiming).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.custom_timings.deinit();
        
        var edid_iter = self.edid_cache.valueIterator();
        while (edid_iter.next()) |edid| {
            edid.deinit();
        }
        self.edid_cache.deinit();
        
        self.timing_overrides.deinit();
    }
    
    pub fn parseEdid(self: *Self, display_id: u32, edid_data: []const u8) !ParsedEdid {
        if (edid_data.len < 128) {
            return TimingError.EdidParsingFailed;
        }
        
        var parsed_edid = ParsedEdid.init(self.allocator);
        
        // Parse EDID header and basic info
        try self.parseEdidHeader(&parsed_edid, edid_data[0..128]);
        
        // Parse extensions
        const num_extensions = edid_data[126];
        var offset: usize = 128;
        
        for (0..num_extensions) |i| {
            if (offset + 128 > edid_data.len) break;
            
            const ext_data = edid_data[offset..offset + 128];
            try self.parseEdidExtension(&parsed_edid, ext_data);
            offset += 128;
            _ = i;
        }
        
        try self.edid_cache.put(display_id, parsed_edid);
        return parsed_edid;
    }
    
    fn parseEdidHeader(self: *Self, edid: *ParsedEdid, data: []const u8) !void {
        _ = self;
        
        // Verify EDID header
        const expected_header = [_]u8{ 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00 };
        if (!std.mem.eql(u8, data[0..8], &expected_header)) {
            return TimingError.EdidParsingFailed;
        }
        
        // Parse manufacturer ID (bytes 8-9)
        const mfg_id = (@as(u16, data[8]) << 8) | data[9];
        edid.manufacturer_id[0] = @as(u8, @truncate(((mfg_id >> 10) & 0x1F) + 'A' - 1));
        edid.manufacturer_id[1] = @as(u8, @truncate(((mfg_id >> 5) & 0x1F) + 'A' - 1));
        edid.manufacturer_id[2] = @as(u8, @truncate((mfg_id & 0x1F) + 'A' - 1));
        
        // Parse product code (bytes 10-11)
        edid.product_code = (@as(u16, data[11]) << 8) | data[10];
        
        // Parse serial number (bytes 12-15)
        edid.serial_number = (@as(u32, data[15]) << 24) | (@as(u32, data[14]) << 16) | 
                            (@as(u32, data[13]) << 8) | data[12];
        
        // Parse manufacture date (bytes 16-17)
        edid.manufacture_week = data[16];
        edid.manufacture_year = @as(u16, data[17]) + 1990;
        
        // Parse display characteristics (bytes 21-24)
        edid.display_width_cm = data[21];
        edid.display_height_cm = data[22];
        edid.gamma = data[23]; // Gamma = (value + 100) / 100
        
        // Parse feature support (byte 24)
        const features = data[24];
        edid.dpms_standby = (features & 0x80) != 0;
        edid.dpms_suspend = (features & 0x40) != 0;
        edid.dpms_off = (features & 0x20) != 0;
        edid.digital_input = (data[20] & 0x80) != 0;
        
        // Parse color depth from input definition
        if (edid.digital_input) {
            const depth_bits = (data[20] >> 4) & 0x07;
            edid.color_depth = switch (depth_bits) {
                1 => .bpc_6,
                2 => .bpc_8,
                3 => .bpc_10,
                4 => .bpc_12,
                5 => .bpc_14,
                6 => .bpc_16,
                else => .bpc_8,
            };
        }
        
        // Parse detailed timing descriptors (bytes 54-125)
        var offset: usize = 54;
        while (offset <= 125 - 18) {
            if (parseDetailedTiming(data[offset..offset + 18])) |timing| {
                if (edid.preferred_timing == null) {
                    edid.preferred_timing = timing;
                } else {
                    try edid.detailed_timings.append(timing);
                }
            }
            offset += 18;
        }
    }
    
    fn parseDetailedTiming(data: []const u8) ?DisplayTiming {
        if (data.len < 18) return null;
        
        // Parse pixel clock (bytes 0-1)
        const pixel_clock_khz = (@as(u32, data[1]) << 8) | data[0];
        if (pixel_clock_khz == 0) return null; // Not a timing descriptor
        
        // Parse horizontal timing (bytes 2-4)
        const h_addressable = @as(u16, data[2]) | ((@as(u16, data[4]) & 0xF0) << 4);
        const h_blanking = @as(u16, data[3]) | ((@as(u16, data[4]) & 0x0F) << 8);
        
        // Parse vertical timing (bytes 5-7)
        const v_addressable = @as(u16, data[5]) | ((@as(u16, data[7]) & 0xF0) << 4);
        const v_blanking = @as(u16, data[6]) | ((@as(u16, data[7]) & 0x0F) << 8);
        
        // Parse sync timing (bytes 8-11)
        const h_front_porch = @as(u16, data[8]) | ((@as(u16, data[11]) & 0xC0) << 2);
        const h_sync_width = @as(u16, data[9]) | ((@as(u16, data[11]) & 0x30) << 4);
        const v_front_porch = @as(u16, (data[10] >> 4)) | ((@as(u16, data[11]) & 0x0C) << 2);
        const v_sync_width = @as(u16, (data[10] & 0x0F)) | ((@as(u16, data[11]) & 0x03) << 4);
        
        // Parse sync polarities (byte 17)
        const sync_flags = data[17];
        const h_sync_positive = (sync_flags & 0x02) != 0;
        const v_sync_positive = (sync_flags & 0x04) != 0;
        const interlaced = (sync_flags & 0x80) != 0;
        
        return DisplayTiming{
            .pixel_clock_khz = pixel_clock_khz * 10, // EDID stores in 10kHz units
            .h_addressable = h_addressable,
            .h_blanking = h_blanking,
            .h_front_porch = h_front_porch,
            .h_sync_width = h_sync_width,
            .h_back_porch = h_blanking - h_front_porch - h_sync_width,
            .v_addressable = v_addressable,
            .v_blanking = v_blanking,
            .v_front_porch = v_front_porch,
            .v_sync_width = v_sync_width,
            .v_back_porch = v_blanking - v_front_porch - v_sync_width,
            .h_sync_positive = h_sync_positive,
            .v_sync_positive = v_sync_positive,
            .interlaced = interlaced,
            .stereo_mode = .none,
            .color_depth = .bpc_8,
            .colorimetry = .bt709,
            .eotf = .bt709,
            .vrr_capable = false,
            .vrr_min_hz = 0,
            .vrr_max_hz = 0,
            .hdr_capable = false,
            .max_luminance_nits = 0,
            .min_luminance_nits = 0,
            .max_frame_average_nits = 0,
        };
    }
    
    fn parseEdidExtension(self: *Self, edid: *ParsedEdid, ext_data: []const u8) !void {
        const tag = ext_data[0];
        
        const extension = EdidExtension{
            .tag = tag,
            .revision = ext_data[1],
            .data = try self.allocator.dupe(u8, ext_data[2..]),
        };
        
        switch (tag) {
            EdidExtension.CEA_861_TAG => {
                try self.parseCea861Extension(edid, extension.data);
            },
            EdidExtension.DISPLAY_ID_TAG, EdidExtension.DISPLAY_ID_2_TAG => {
                try self.parseDisplayIdExtension(edid, extension.data, tag == EdidExtension.DISPLAY_ID_2_TAG);
            },
            else => {
                // Unknown extension, store raw data
            },
        }
        
        try edid.extensions.append(extension);
    }
    
    fn parseCea861Extension(self: *Self, edid: *ParsedEdid, data: []const u8) !void {
        _ = self;
        if (data.len < 2) return;
        
        // Parse basic CEA-861 capabilities
        const dtd_offset = data[0]; // Offset to detailed timing descriptors
        const supports_underscan = (data[1] & 0x80) != 0;
        const supports_audio = (data[1] & 0x40) != 0;
        const supports_ycbcr444 = (data[1] & 0x20) != 0;
        const supports_ycbcr422 = (data[1] & 0x10) != 0;
        
        _ = supports_underscan;
        _ = supports_audio;
        _ = supports_ycbcr444;
        _ = supports_ycbcr422;
        
        // Parse data blocks (simplified)
        var offset: usize = 2;
        while (offset < data.len and offset < dtd_offset) {
            if (offset >= data.len) break;
            
            const block_header = data[offset];
            const block_tag = (block_header >> 5) & 0x07;
            const block_length = block_header & 0x1F;
            offset += 1;
            
            if (offset + block_length > data.len) break;
            
            switch (block_tag) {
                0x03 => { // Video Capability Data Block
                    if (block_length >= 1) {
                        const video_caps = data[offset];
                        // Check for VRR support (simplified)
                        if ((video_caps & 0x80) != 0) {
                            edid.vrr_capable = true;
                            edid.vrr_range.min = 48; // Typical minimum
                            edid.vrr_range.max = 144; // Common maximum
                        }
                    }
                },
                0x06 => { // HDR Static Metadata Data Block
                    if (block_length >= 3) {
                        var hdr_metadata = HdrMetadata{
                            .max_display_mastering_luminance = 0,
                            .min_display_mastering_luminance = 0,
                            .max_content_light_level = 0,
                            .max_frame_average_light_level = 0,
                            .red_primary_x = 0,
                            .red_primary_y = 0,
                            .green_primary_x = 0,
                            .green_primary_y = 0,
                            .blue_primary_x = 0,
                            .blue_primary_y = 0,
                            .white_point_x = 0,
                            .white_point_y = 0,
                        };
                        
                        // Parse HDR metadata (simplified)
                        const supported_eotf = data[offset + 1];
                        if ((supported_eotf & 0x04) != 0) { // PQ (HDR10)
                            if (block_length >= 5) {
                                hdr_metadata.max_display_mastering_luminance = data[offset + 2];
                                hdr_metadata.min_display_mastering_luminance = data[offset + 3];
                                hdr_metadata.max_content_light_level = data[offset + 4];
                            }
                            
                            edid.hdr_metadata = hdr_metadata;
                        }
                    }
                },
                else => {
                    // Skip unknown data blocks
                },
            }
            
            offset += block_length;
        }
    }
    
    fn parseDisplayIdExtension(self: *Self, edid: *ParsedEdid, data: []const u8, is_v2: bool) !void {
        _ = self;
        if (data.len < 5) return;
        
        // Parse DisplayID header
        const structure_version = data[0] >> 4;
        const structure_revision = data[0] & 0x0F;
        const primary_use_case = data[1];
        const extension_count = data[2];
        
        _ = structure_version;
        _ = structure_revision;
        _ = primary_use_case;
        _ = extension_count;
        
        // Parse data blocks
        var offset: usize = 5;
        while (offset < data.len - 1) { // -1 for checksum
            if (offset + 3 > data.len) break;
            
            const block_tag = data[offset];
            const revision_and_length = (@as(u16, data[offset + 2]) << 8) | data[offset + 1];
            const block_length = revision_and_length & 0x1FFF;
            
            offset += 3;
            if (offset + block_length > data.len) break;
            
            switch (block_tag) {
                0x12 => { // Tiled Display Topology Data Block
                    if (block_length >= 22 and is_v2) {
                        const tiled = TiledDisplay{
                            .tile_count_h = (data[offset + 2] & 0x0F) + 1,
                            .tile_count_v = (data[offset + 2] >> 4) + 1,
                            .tile_location_h = data[offset + 4] & 0x0F,
                            .tile_location_v = data[offset + 4] >> 4,
                            .native_resolution_h = (@as(u16, data[offset + 6]) << 8) | data[offset + 5],
                            .native_resolution_v = (@as(u16, data[offset + 8]) << 8) | data[offset + 7],
                            .bezel_left = data[offset + 12],
                            .bezel_top = data[offset + 13],
                            .bezel_right = data[offset + 14],
                            .bezel_bottom = data[offset + 15],
                        };
                        
                        edid.tiled_display = tiled;
                    }
                },
                else => {
                    // Skip unknown data blocks
                },
            }
            
            offset += block_length;
        }
    }
    
    pub fn addCustomTiming(self: *Self, timing: DisplayTiming) !void {
        if (!timing.validateTiming()) {
            return TimingError.InvalidTiming;
        }
        
        try self.custom_timings.append(timing);
    }
    
    pub fn setTimingOverride(self: *Self, display_id: u32, timing: DisplayTiming) !void {
        if (!timing.validateTiming()) {
            return TimingError.InvalidTiming;
        }
        
        try self.timing_overrides.put(display_id, timing);
    }
    
    pub fn getOptimalTiming(self: *Self, display_id: u32, requested_width: u16, requested_height: u16, requested_refresh: f32) ?DisplayTiming {
        // Check for override first
        if (self.timing_overrides.get(display_id)) |override_timing| {
            return override_timing;
        }
        
        // Check EDID cache
        if (self.edid_cache.get(display_id)) |edid| {
            // Find best matching timing from EDID
            var best_timing: ?DisplayTiming = null;
            var best_score: f32 = 0.0;
            
            // Check preferred timing
            if (edid.preferred_timing) |timing| {
                const score = self.scoreTiming(timing, requested_width, requested_height, requested_refresh);
                if (score > best_score) {
                    best_score = score;
                    best_timing = timing;
                }
            }
            
            // Check detailed timings
            for (edid.detailed_timings.items) |timing| {
                const score = self.scoreTiming(timing, requested_width, requested_height, requested_refresh);
                if (score > best_score) {
                    best_score = score;
                    best_timing = timing;
                }
            }
            
            return best_timing;
        }
        
        return null;
    }
    
    fn scoreTiming(self: *Self, timing: DisplayTiming, target_width: u16, target_height: u16, target_refresh: f32) f32 {
        _ = self;
        
        // Calculate matching score (higher is better)
        var score: f32 = 0.0;
        
        // Resolution match (most important)
        if (timing.h_addressable == target_width and timing.v_addressable == target_height) {
            score += 100.0;
        } else {
            const width_ratio = @min(@as(f32, @floatFromInt(timing.h_addressable)) / @as(f32, @floatFromInt(target_width)),
                                    @as(f32, @floatFromInt(target_width)) / @as(f32, @floatFromInt(timing.h_addressable)));
            const height_ratio = @min(@as(f32, @floatFromInt(timing.v_addressable)) / @as(f32, @floatFromInt(target_height)),
                                     @as(f32, @floatFromInt(target_height)) / @as(f32, @floatFromInt(timing.v_addressable)));
            score += (width_ratio + height_ratio) * 25.0;
        }
        
        // Refresh rate match
        const actual_refresh = timing.calculateRefreshRate();
        const refresh_diff = @abs(actual_refresh - target_refresh);
        if (refresh_diff < 1.0) {
            score += 50.0;
        } else {
            score += @max(0.0, 50.0 - refresh_diff);
        }
        
        return score;
    }
    
    pub fn generateCustomTiming(self: *Self, width: u16, height: u16, refresh_hz: f32, reduced_blanking: bool) !DisplayTiming {
        _ = self;
        
        // Generate CVT (Coordinated Video Timings) standard timing
        var timing = DisplayTiming{
            .pixel_clock_khz = 0,
            .h_addressable = width,
            .h_blanking = 0,
            .h_front_porch = 0,
            .h_sync_width = 0,
            .h_back_porch = 0,
            .v_addressable = height,
            .v_blanking = 0,
            .v_front_porch = 0,
            .v_sync_width = 0,
            .v_back_porch = 0,
            .h_sync_positive = false,
            .v_sync_positive = true,
            .interlaced = false,
            .stereo_mode = .none,
            .color_depth = .bpc_8,
            .colorimetry = .bt709,
            .eotf = .bt709,
            .vrr_capable = false,
            .vrr_min_hz = 0,
            .vrr_max_hz = 0,
            .hdr_capable = false,
            .max_luminance_nits = 0,
            .min_luminance_nits = 0,
            .max_frame_average_nits = 0,
        };
        
        if (reduced_blanking) {
            // CVT Reduced Blanking
            timing.v_front_porch = 3;
            timing.v_sync_width = 4;
            timing.v_back_porch = 6;
            timing.v_blanking = timing.v_front_porch + timing.v_sync_width + timing.v_back_porch;
            
            timing.h_front_porch = 48;
            timing.h_sync_width = 32;
            timing.h_back_porch = 80;
            timing.h_blanking = timing.h_front_porch + timing.h_sync_width + timing.h_back_porch;
        } else {
            // Standard CVT
            const v_field_rate = refresh_hz;
            const interlace_factor: f32 = 1.0; // Progressive
            const h_period_estimate = (1.0 / v_field_rate - 550.0e-6) / (@as(f32, @floatFromInt(height)) + 3.0) * 1e6;
            
            timing.v_front_porch = 3;
            timing.v_sync_width = 4;
            timing.v_back_porch = @as(u16, @intFromFloat(550.0 / h_period_estimate + 0.5));
            timing.v_blanking = timing.v_front_porch + timing.v_sync_width + timing.v_back_porch;
            
            const h_period = h_period_estimate / interlace_factor;
            _ = h_period;
            const h_total_pixels = @as(f32, @floatFromInt(width)) * 8.0 / 8.0; // Character cell granularity
            timing.h_blanking = @as(u16, @intFromFloat(h_total_pixels * 0.3 / 0.7));
            timing.h_front_porch = timing.h_blanking / 2 - timing.h_sync_width;
            timing.h_sync_width = @as(u16, @intFromFloat(h_total_pixels * 0.08));
            timing.h_back_porch = timing.h_blanking - timing.h_front_porch - timing.h_sync_width;
        }
        
        // Calculate pixel clock
        const h_total = timing.h_addressable + timing.h_blanking;
        const v_total = timing.v_addressable + timing.v_blanking;
        timing.pixel_clock_khz = @as(u32, @intFromFloat(@as(f32, @floatFromInt(h_total)) * @as(f32, @floatFromInt(v_total)) * refresh_hz / 1000.0));
        
        if (!timing.validateTiming()) {
            return TimingError.InvalidTiming;
        }
        
        return timing;
    }
    
    pub fn getCachedEdid(self: *const Self, display_id: u32) ?ParsedEdid {
        return self.edid_cache.get(display_id);
    }
    
    pub fn clearTimingCache(self: *Self, display_id: u32) void {
        if (self.edid_cache.fetchRemove(display_id)) |entry| {
            var edid = entry.value;
            edid.deinit();
        }
        _ = self.timing_overrides.remove(display_id);
    }
};