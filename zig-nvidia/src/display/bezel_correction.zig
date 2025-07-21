const std = @import("std");
const display = @import("engine.zig");

/// Advanced Bezel Correction and Multi-Monitor Spanning Engine
/// Provides seamless desktop spanning with physical bezel compensation

pub const BezelError = error{
    InvalidDisplayLayout,
    UnsupportedResolution,
    CalibrationRequired,
    InsufficientDisplays,
    TopologyDetectionFailed,
    OutOfMemory,
};

pub const DisplayOrientation = enum(u8) {
    landscape = 0,
    portrait_left = 90,
    portrait_right = 270,
    landscape_flipped = 180,
    
    pub fn getRotationMatrix(self: DisplayOrientation) [4]f32 {
        return switch (self) {
            .landscape => [_]f32{ 1.0, 0.0, 0.0, 1.0 },
            .portrait_left => [_]f32{ 0.0, -1.0, 1.0, 0.0 },
            .portrait_right => [_]f32{ 0.0, 1.0, -1.0, 0.0 },
            .landscape_flipped => [_]f32{ -1.0, 0.0, 0.0, -1.0 },
        };
    }
    
    pub fn needsResolutionSwap(self: DisplayOrientation) bool {
        return self == .portrait_left or self == .portrait_right;
    }
};

pub const PhysicalDisplay = struct {
    // Identification
    display_id: u32,
    manufacturer: [4]u8, // EDID manufacturer ID
    product_code: u16,
    serial_number: [13]u8,
    
    // Physical properties
    width_mm: u16,
    height_mm: u16,
    bezel_left_mm: u8,
    bezel_right_mm: u8,
    bezel_top_mm: u8,
    bezel_bottom_mm: u8,
    
    // Resolution and timing
    native_width: u32,
    native_height: u32,
    refresh_rate: u16,
    pixel_density_ppi: u16,
    
    // Position in desktop
    position_x: i32,
    position_y: i32,
    orientation: DisplayOrientation,
    
    // Calibration data
    color_profile: ?[]u8, // ICC profile
    brightness_nits: u16,
    contrast_ratio: f32,
    gamma_curve: [256]u16,
    
    pub fn init(display_id: u32) PhysicalDisplay {
        return PhysicalDisplay{
            .display_id = display_id,
            .manufacturer = [_]u8{0} ** 4,
            .product_code = 0,
            .serial_number = [_]u8{0} ** 13,
            .width_mm = 0,
            .height_mm = 0,
            .bezel_left_mm = 5,    // Default 5mm bezels
            .bezel_right_mm = 5,
            .bezel_top_mm = 5,
            .bezel_bottom_mm = 5,
            .native_width = 1920,
            .native_height = 1080,
            .refresh_rate = 60,
            .pixel_density_ppi = 96,
            .position_x = 0,
            .position_y = 0,
            .orientation = .landscape,
            .color_profile = null,
            .brightness_nits = 300,
            .contrast_ratio = 1000.0,
            .gamma_curve = [_]u16{0} ** 256,
        };
    }
    
    pub fn getEffectiveResolution(self: *const PhysicalDisplay) struct { width: u32, height: u32 } {
        return if (self.orientation.needsResolutionSwap()) 
            .{ .width = self.native_height, .height = self.native_width }
        else 
            .{ .width = self.native_width, .height = self.native_height };
    }
    
    pub fn getPixelSize(self: *const PhysicalDisplay) struct { width_um: f32, height_um: f32 } {
        const res = self.getEffectiveResolution();
        const width_um = (@as(f32, @floatFromInt(self.width_mm)) * 1000.0) / @as(f32, @floatFromInt(res.width));
        const height_um = (@as(f32, @floatFromInt(self.height_mm)) * 1000.0) / @as(f32, @floatFromInt(res.height));
        return .{ .width_um = width_um, .height_um = height_um };
    }
    
    pub fn getBezelOffsets(self: *const PhysicalDisplay) struct { left: f32, right: f32, top: f32, bottom: f32 } {
        const pixel_size = self.getPixelSize();
        return .{
            .left = @as(f32, @floatFromInt(self.bezel_left_mm)) / pixel_size.width_um * 1000.0,
            .right = @as(f32, @floatFromInt(self.bezel_right_mm)) / pixel_size.width_um * 1000.0,
            .top = @as(f32, @floatFromInt(self.bezel_top_mm)) / pixel_size.height_um * 1000.0,
            .bottom = @as(f32, @floatFromInt(self.bezel_bottom_mm)) / pixel_size.height_um * 1000.0,
        };
    }
};

pub const DisplayTopology = enum(u8) {
    single = 1,
    horizontal_dual = 2,
    vertical_dual = 3,
    quad_2x2 = 4,
    triple_horizontal = 5,
    triple_vertical = 6,
    triple_l_shape = 7,
    surround_3x1 = 8,
    surround_5x1 = 9,
    custom = 255,
    
    pub fn getDisplayCount(self: DisplayTopology) u8 {
        return switch (self) {
            .single => 1,
            .horizontal_dual, .vertical_dual => 2,
            .quad_2x2, .triple_horizontal, .triple_vertical, .triple_l_shape => 3,
            .surround_3x1 => 3,
            .surround_5x1 => 5,
            .custom => 0, // Determined by actual configuration
        };
    }
    
    pub fn getRecommendedLayout(displays: []const PhysicalDisplay) DisplayTopology {
        switch (displays.len) {
            1 => return .single,
            2 => {
                // Determine if displays are arranged horizontally or vertically
                const display0 = displays[0];
                const display1 = displays[1];
                
                const horizontal_gap = @abs(display1.position_x - (display0.position_x + @as(i32, @intCast(display0.native_width))));
                const vertical_gap = @abs(display1.position_y - (display0.position_y + @as(i32, @intCast(display0.native_height))));
                
                return if (horizontal_gap < vertical_gap) .horizontal_dual else .vertical_dual;
            },
            3 => return .triple_horizontal, // Default to horizontal for 3 displays
            4 => return .quad_2x2,
            5 => return .surround_5x1,
            else => return .custom,
        }
    }
};

pub const DesktopSpanningMode = enum(u8) {
    clone = 0,      // Mirror same content on all displays
    extend = 1,     // Extend desktop across displays
    spanning = 2,   // Seamless spanning with bezel correction
    surround = 3,   // Gaming surround mode (single virtual display)
    
    pub fn requiresBezelCorrection(self: DesktopSpanningMode) bool {
        return self == .spanning or self == .surround;
    }
};

pub const BezelCorrectionMatrix = struct {
    display_count: u8,
    correction_matrices: [8][16]f32, // Up to 8 displays, 4x4 matrices each
    viewport_transforms: [8]struct {
        offset_x: f32,
        offset_y: f32,
        scale_x: f32,
        scale_y: f32,
    },
    total_virtual_width: u32,
    total_virtual_height: u32,
    
    pub fn init(displays: []const PhysicalDisplay, topology: DisplayTopology) !BezelCorrectionMatrix {
        var matrix = BezelCorrectionMatrix{
            .display_count = @intCast(displays.len),
            .correction_matrices = [_][16]f32{[_]f32{0} ** 16} ** 8,
            .viewport_transforms = [_]struct {
                offset_x: f32,
                offset_y: f32,
                scale_x: f32,
                scale_y: f32,
            }{.{ .offset_x = 0, .offset_y = 0, .scale_x = 1, .scale_y = 1 }} ** 8,
            .total_virtual_width = 0,
            .total_virtual_height = 0,
        };
        
        try matrix.calculateCorrectionMatrices(displays, topology);
        return matrix;
    }
    
    fn calculateCorrectionMatrices(self: *BezelCorrectionMatrix, displays: []const PhysicalDisplay, topology: DisplayTopology) !void {
        switch (topology) {
            .horizontal_dual => try self.calculateHorizontalDual(displays),
            .vertical_dual => try self.calculateVerticalDual(displays),
            .quad_2x2 => try self.calculateQuad2x2(displays),
            .triple_horizontal => try self.calculateTripleHorizontal(displays),
            .surround_3x1 => try self.calculateSurround3x1(displays),
            .surround_5x1 => try self.calculateSurround5x1(displays),
            else => try self.calculateCustomLayout(displays),
        }
    }
    
    fn calculateHorizontalDual(self: *BezelCorrectionMatrix, displays: []const PhysicalDisplay) !void {
        if (displays.len != 2) return BezelError.InvalidDisplayLayout;
        
        const left_display = &displays[0];
        const right_display = &displays[1];
        
        const left_res = left_display.getEffectiveResolution();
        const right_res = right_display.getEffectiveResolution();
        
        // Calculate bezel offsets in pixels
        const left_bezel = left_display.getBezelOffsets();
        const right_bezel = right_display.getBezelOffsets();
        
        // Calculate total virtual dimensions
        const bezel_gap_pixels = (left_bezel.right + right_bezel.left) / 2.0;
        self.total_virtual_width = left_res.width + right_res.width;
        self.total_virtual_height = @max(left_res.height, right_res.height);
        
        // Left display transform
        self.viewport_transforms[0] = .{
            .offset_x = 0,
            .offset_y = 0,
            .scale_x = 1.0,
            .scale_y = 1.0,
        };
        
        // Right display transform (account for bezel gap)
        self.viewport_transforms[1] = .{
            .offset_x = @as(f32, @floatFromInt(left_res.width)) - bezel_gap_pixels,
            .offset_y = 0,
            .scale_x = 1.0,
            .scale_y = 1.0,
        };
        
        // Create transformation matrices
        self.createIdentityMatrix(0);
        self.createTranslationMatrix(1, self.viewport_transforms[1].offset_x, 0);
    }
    
    fn calculateVerticalDual(self: *BezelCorrectionMatrix, displays: []const PhysicalDisplay) !void {
        if (displays.len != 2) return BezelError.InvalidDisplayLayout;
        
        const top_display = &displays[0];
        const bottom_display = &displays[1];
        
        const top_res = top_display.getEffectiveResolution();
        const bottom_res = bottom_display.getEffectiveResolution();
        
        const top_bezel = top_display.getBezelOffsets();
        const bottom_bezel = bottom_display.getBezelOffsets();
        
        const bezel_gap_pixels = (top_bezel.bottom + bottom_bezel.top) / 2.0;
        self.total_virtual_width = @max(top_res.width, bottom_res.width);
        self.total_virtual_height = top_res.height + bottom_res.height;
        
        self.viewport_transforms[0] = .{
            .offset_x = 0,
            .offset_y = 0,
            .scale_x = 1.0,
            .scale_y = 1.0,
        };
        
        self.viewport_transforms[1] = .{
            .offset_x = 0,
            .offset_y = @as(f32, @floatFromInt(top_res.height)) - bezel_gap_pixels,
            .scale_x = 1.0,
            .scale_y = 1.0,
        };
        
        self.createIdentityMatrix(0);
        self.createTranslationMatrix(1, 0, self.viewport_transforms[1].offset_y);
    }
    
    fn calculateQuad2x2(self: *BezelCorrectionMatrix, displays: []const PhysicalDisplay) !void {
        if (displays.len != 4) return BezelError.InvalidDisplayLayout;
        
        // Assume displays are arranged:
        // [0] [1]
        // [2] [3]
        
        const top_left = &displays[0];
        const top_right = &displays[1];
        const bottom_left = &displays[2];
        const bottom_right = &displays[3];
        
        const tl_res = top_left.getEffectiveResolution();
        const tr_res = top_right.getEffectiveResolution();
        const bl_res = bottom_left.getEffectiveResolution();
        _ = bottom_right.getEffectiveResolution();
        
        // Calculate bezel corrections
        const tl_bezel = top_left.getBezelOffsets();
        const tr_bezel = top_right.getBezelOffsets();
        const bl_bezel = bottom_left.getBezelOffsets();
        _ = bottom_right.getBezelOffsets();
        
        const horizontal_gap = (tl_bezel.right + tr_bezel.left) / 2.0;
        const vertical_gap = (tl_bezel.bottom + bl_bezel.top) / 2.0;
        
        self.total_virtual_width = tl_res.width + tr_res.width;
        self.total_virtual_height = tl_res.height + bl_res.height;
        
        // Top-left (reference)
        self.viewport_transforms[0] = .{ .offset_x = 0, .offset_y = 0, .scale_x = 1.0, .scale_y = 1.0 };
        
        // Top-right
        self.viewport_transforms[1] = .{
            .offset_x = @as(f32, @floatFromInt(tl_res.width)) - horizontal_gap,
            .offset_y = 0,
            .scale_x = 1.0,
            .scale_y = 1.0,
        };
        
        // Bottom-left
        self.viewport_transforms[2] = .{
            .offset_x = 0,
            .offset_y = @as(f32, @floatFromInt(tl_res.height)) - vertical_gap,
            .scale_x = 1.0,
            .scale_y = 1.0,
        };
        
        // Bottom-right
        self.viewport_transforms[3] = .{
            .offset_x = @as(f32, @floatFromInt(tl_res.width)) - horizontal_gap,
            .offset_y = @as(f32, @floatFromInt(tl_res.height)) - vertical_gap,
            .scale_x = 1.0,
            .scale_y = 1.0,
        };
        
        // Create transformation matrices
        for (0..4) |i| {
            const transform = self.viewport_transforms[i];
            self.createTranslationMatrix(@intCast(i), transform.offset_x, transform.offset_y);
        }
    }
    
    fn calculateTripleHorizontal(self: *BezelCorrectionMatrix, displays: []const PhysicalDisplay) !void {
        if (displays.len != 3) return BezelError.InvalidDisplayLayout;
        
        const left = &displays[0];
        const center = &displays[1];
        const right = &displays[2];
        
        const left_res = left.getEffectiveResolution();
        const center_res = center.getEffectiveResolution();
        const right_res = right.getEffectiveResolution();
        
        const left_bezel = left.getBezelOffsets();
        const center_bezel = center.getBezelOffsets();
        const right_bezel = right.getBezelOffsets();
        
        const left_center_gap = (left_bezel.right + center_bezel.left) / 2.0;
        const center_right_gap = (center_bezel.right + right_bezel.left) / 2.0;
        
        self.total_virtual_width = left_res.width + center_res.width + right_res.width;
        self.total_virtual_height = @max(@max(left_res.height, center_res.height), right_res.height);
        
        // Left display
        self.viewport_transforms[0] = .{ .offset_x = 0, .offset_y = 0, .scale_x = 1.0, .scale_y = 1.0 };
        
        // Center display
        self.viewport_transforms[1] = .{
            .offset_x = @as(f32, @floatFromInt(left_res.width)) - left_center_gap,
            .offset_y = 0,
            .scale_x = 1.0,
            .scale_y = 1.0,
        };
        
        // Right display
        self.viewport_transforms[2] = .{
            .offset_x = @as(f32, @floatFromInt(left_res.width)) + @as(f32, @floatFromInt(center_res.width)) - left_center_gap - center_right_gap,
            .offset_y = 0,
            .scale_x = 1.0,
            .scale_y = 1.0,
        };
        
        for (0..3) |i| {
            const transform = self.viewport_transforms[i];
            self.createTranslationMatrix(@intCast(i), transform.offset_x, transform.offset_y);
        }
    }
    
    fn calculateSurround3x1(self: *BezelCorrectionMatrix, displays: []const PhysicalDisplay) !void {
        // Special gaming surround mode for 3x1 setup
        try self.calculateTripleHorizontal(displays);
        
        // Apply additional surround-specific corrections
        // This creates a slight curve effect for immersive gaming
        for (0..3) |i| {
            const angle = switch (i) {
                0 => -15.0, // Left display angled inward
                1 => 0.0,   // Center display straight
                2 => 15.0,  // Right display angled inward
                else => 0.0,
            };
            
            self.applyPerspectiveCorrection(@intCast(i), angle);
        }
    }
    
    fn calculateSurround5x1(self: *BezelCorrectionMatrix, displays: []const PhysicalDisplay) !void {
        if (displays.len != 5) return BezelError.InvalidDisplayLayout;
        
        // Ultra-wide 5x1 surround setup
        var total_width: u32 = 0;
        var max_height: u32 = 0;
        var x_offset: f32 = 0;
        
        for (displays, 0..) |disp, i| {
            const res = disp.getEffectiveResolution();
            const bezel = disp.getBezelOffsets();
            
            self.viewport_transforms[i] = .{
                .offset_x = x_offset,
                .offset_y = 0,
                .scale_x = 1.0,
                .scale_y = 1.0,
            };
            
            // Account for bezel gap
            const next_gap = if (i < displays.len - 1) 
                (bezel.right + displays[i + 1].getBezelOffsets().left) / 2.0
            else 0;
                
            x_offset += @as(f32, @floatFromInt(res.width)) - next_gap;
            total_width += res.width;
            max_height = @max(max_height, res.height);
            
            // Apply surround angle correction
            const angle = switch (i) {
                0 => -30.0, // Far left
                1 => -15.0, // Left
                2 => 0.0,   // Center
                3 => 15.0,  // Right
                4 => 30.0,  // Far right
                else => 0.0,
            };
            
            self.createTranslationMatrix(@intCast(i), self.viewport_transforms[i].offset_x, 0);
            self.applyPerspectiveCorrection(@intCast(i), angle);
        }
        
        self.total_virtual_width = total_width;
        self.total_virtual_height = max_height;
    }
    
    fn calculateCustomLayout(self: *BezelCorrectionMatrix, displays: []const PhysicalDisplay) !void {
        // For custom layouts, use the actual position coordinates from EDID/configuration
        var min_x: i32 = std.math.maxInt(i32);
        var min_y: i32 = std.math.maxInt(i32);
        var max_x: i32 = std.math.minInt(i32);
        var max_y: i32 = std.math.minInt(i32);
        
        // Find bounding rectangle
        for (displays) |disp| {
            const res = disp.getEffectiveResolution();
            min_x = @min(min_x, disp.position_x);
            min_y = @min(min_y, disp.position_y);
            max_x = @max(max_x, disp.position_x + @as(i32, @intCast(res.width)));
            max_y = @max(max_y, disp.position_y + @as(i32, @intCast(res.height)));
        }
        
        self.total_virtual_width = @intCast(max_x - min_x);
        self.total_virtual_height = @intCast(max_y - min_y);
        
        // Set up transforms relative to bounding rectangle origin
        for (displays, 0..) |disp, i| {
            self.viewport_transforms[i] = .{
                .offset_x = @as(f32, @floatFromInt(disp.position_x - min_x)),
                .offset_y = @as(f32, @floatFromInt(disp.position_y - min_y)),
                .scale_x = 1.0,
                .scale_y = 1.0,
            };
            
            const transform = self.viewport_transforms[i];
            self.createTranslationMatrix(@intCast(i), transform.offset_x, transform.offset_y);
        }
    }
    
    fn createIdentityMatrix(self: *BezelCorrectionMatrix, display_index: u8) void {
        const identity = [_]f32{
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        };
        self.correction_matrices[display_index] = identity;
    }
    
    fn createTranslationMatrix(self: *BezelCorrectionMatrix, display_index: u8, tx: f32, ty: f32) void {
        const translation = [_]f32{
            1.0, 0.0, 0.0, tx,
            0.0, 1.0, 0.0, ty,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        };
        self.correction_matrices[display_index] = translation;
    }
    
    fn applyPerspectiveCorrection(self: *BezelCorrectionMatrix, display_index: u8, angle_degrees: f32) void {
        const angle_rad = angle_degrees * std.math.pi / 180.0;
        const cos_angle = @cos(angle_rad);
        const sin_angle = @sin(angle_rad);
        
        // Apply perspective transformation for surround gaming
        const perspective = [_]f32{
            cos_angle, -sin_angle * 0.1, 0.0, 0.0,
            sin_angle * 0.1, 1.0, 0.0, 0.0,
            0.0, 0.0, cos_angle, 0.0,
            0.0, 0.0, 0.0, 1.0,
        };
        
        // Multiply existing matrix with perspective correction
        self.multiplyMatrix(display_index, &perspective);
    }
    
    fn multiplyMatrix(self: *BezelCorrectionMatrix, display_index: u8, matrix_b: *const [16]f32) void {
        const matrix_a = &self.correction_matrices[display_index];
        var result = [_]f32{0} ** 16;
        
        for (0..4) |i| {
            for (0..4) |j| {
                var sum: f32 = 0;
                for (0..4) |k| {
                    sum += matrix_a[i * 4 + k] * matrix_b[k * 4 + j];
                }
                result[i * 4 + j] = sum;
            }
        }
        
        self.correction_matrices[display_index] = result;
    }
};

pub const BezelCorrector = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    displays: std.ArrayList(PhysicalDisplay),
    topology: DisplayTopology,
    spanning_mode: DesktopSpanningMode,
    correction_matrix: ?BezelCorrectionMatrix,
    calibration_data: std.AutoHashMap(u32, CalibrationData),
    
    const CalibrationData = struct {
        color_temperature: u16,
        brightness_offset: i16,
        gamma_correction: [3]f32, // RGB gamma
        uniformity_correction: ?[]u16, // Per-pixel uniformity data
    };
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .displays = std.ArrayList(PhysicalDisplay).init(allocator),
            .topology = .single,
            .spanning_mode = .extend,
            .correction_matrix = null,
            .calibration_data = std.AutoHashMap(u32, CalibrationData).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.displays.deinit();
        self.calibration_data.deinit();
    }
    
    pub fn addDisplay(self: *Self, phys_display: PhysicalDisplay) !void {
        try self.displays.append(phys_display);
        self.topology = DisplayTopology.getRecommendedLayout(self.displays.items);
    }
    
    pub fn removeDisplay(self: *Self, display_id: u32) void {
        for (self.displays.items, 0..) |disp, i| {
            if (disp.display_id == display_id) {
                _ = self.displays.orderedRemove(i);
                break;
            }
        }
        self.topology = DisplayTopology.getRecommendedLayout(self.displays.items);
    }
    
    pub fn detectDisplayTopology(self: *Self) !DisplayTopology {
        if (self.displays.items.len < 2) return .single;
        
        // Analyze physical positioning to determine topology
        var topology = DisplayTopology.getRecommendedLayout(self.displays.items);
        
        // Validate topology makes sense for the display arrangement
        if (!self.validateTopology(topology)) {
            topology = .custom;
        }
        
        self.topology = topology;
        return topology;
    }
    
    fn validateTopology(self: *const Self, topology: DisplayTopology) bool {
        const expected_count = topology.getDisplayCount();
        if (expected_count != 0 and expected_count != self.displays.items.len) {
            return false;
        }
        
        // Additional validation logic could go here
        return true;
    }
    
    pub fn setSpanningMode(self: *Self, mode: DesktopSpanningMode) !void {
        self.spanning_mode = mode;
        
        if (mode.requiresBezelCorrection()) {
            try self.calculateBezelCorrection();
        } else {
            self.correction_matrix = null;
        }
    }
    
    pub fn calculateBezelCorrection(self: *Self) !void {
        if (self.displays.items.len < 2) {
            return BezelError.InsufficientDisplays;
        }
        
        self.correction_matrix = try BezelCorrectionMatrix.init(self.displays.items, self.topology);
    }
    
    pub fn getCorrectionMatrix(self: *const Self, display_index: u8) ?[16]f32 {
        if (self.correction_matrix) |matrix| {
            if (display_index < matrix.display_count) {
                return matrix.correction_matrices[display_index];
            }
        }
        return null;
    }
    
    pub fn getViewportTransform(self: *const Self, display_index: u8) ?struct { offset_x: f32, offset_y: f32, scale_x: f32, scale_y: f32 } {
        if (self.correction_matrix) |matrix| {
            if (display_index < matrix.display_count) {
                return matrix.viewport_transforms[display_index];
            }
        }
        return null;
    }
    
    pub fn getVirtualDesktopSize(self: *const Self) struct { width: u32, height: u32 } {
        if (self.correction_matrix) |matrix| {
            return .{ .width = matrix.total_virtual_width, .height = matrix.total_virtual_height };
        }
        
        // Fallback: sum of all display resolutions
        var total_width: u32 = 0;
        var max_height: u32 = 0;
        
        for (self.displays.items) |disp| {
            const res = disp.getEffectiveResolution();
            total_width += res.width;
            max_height = @max(max_height, res.height);
        }
        
        return .{ .width = total_width, .height = max_height };
    }
    
    pub fn transformCursor(self: *const Self, global_x: i32, global_y: i32) struct { display_id: u32, local_x: i32, local_y: i32 } {
        // Transform global cursor coordinates to display-local coordinates
        for (self.displays.items, 0..) |disp, i| {
            if (self.getViewportTransform(@intCast(i))) |transform| {
                const local_x = @as(i32, @intFromFloat(@as(f32, @floatFromInt(global_x)) - transform.offset_x));
                const local_y = @as(i32, @intFromFloat(@as(f32, @floatFromInt(global_y)) - transform.offset_y));
                
                const res = disp.getEffectiveResolution();
                if (local_x >= 0 and local_x < res.width and local_y >= 0 and local_y < res.height) {
                    return .{ .display_id = disp.display_id, .local_x = local_x, .local_y = local_y };
                }
            }
        }
        
        // Default to first display if no match
        return .{ .display_id = self.displays.items[0].display_id, .local_x = global_x, .local_y = global_y };
    }
    
    pub fn calibrateDisplay(self: *Self, display_id: u32, calibration: CalibrationData) !void {
        try self.calibration_data.put(display_id, calibration);
    }
    
    pub fn getCalibration(self: *const Self, display_id: u32) ?CalibrationData {
        return self.calibration_data.get(display_id);
    }
    
    pub fn autoDetectBezels(self: *Self, display_id: u32) !void {
        // In real implementation, this would use test patterns and cameras/sensors
        // to automatically measure bezel sizes
        
        for (self.displays.items) |*disp| {
            if (disp.display_id == display_id) {
                // Default bezel sizes based on common monitor types
                if (disp.width_mm > 600) { // Large monitor (>24")
                    disp.bezel_left_mm = 8;
                    disp.bezel_right_mm = 8;
                    disp.bezel_top_mm = 12;
                    disp.bezel_bottom_mm = 20;
                } else { // Standard monitor
                    disp.bezel_left_mm = 5;
                    disp.bezel_right_mm = 5;
                    disp.bezel_top_mm = 8;
                    disp.bezel_bottom_mm = 15;
                }
                break;
            }
        }
    }
    
    pub fn optimizeForGaming(self: *Self) !void {
        // Optimize bezel correction for gaming (minimal latency, maximum FOV)
        if (self.displays.items.len >= 3) {
            self.topology = .surround_3x1;
            self.spanning_mode = .surround;
            try self.calculateBezelCorrection();
        }
    }
    
    pub fn optimizeForProductivity(self: *Self) !void {
        // Optimize for productivity (accurate positioning, color consistency)
        self.spanning_mode = .spanning;
        try self.calculateBezelCorrection();
        
        // Auto-calibrate color consistency across displays
        for (self.displays.items) |disp| {
            const calibration = CalibrationData{
                .color_temperature = 6500, // Standard D65
                .brightness_offset = 0,
                .gamma_correction = [_]f32{ 2.2, 2.2, 2.2 },
                .uniformity_correction = null,
            };
            self.calibrateDisplay(disp.display_id, calibration) catch {};
        }
    }
};