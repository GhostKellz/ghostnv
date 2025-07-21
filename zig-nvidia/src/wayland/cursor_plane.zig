const std = @import("std");
const drm = @import("../drm/driver.zig");
const memory = @import("../hal/memory.zig");
const display = @import("../display/engine.zig");
const linux = std.os.linux;

/// Hardware Cursor Plane Management for Wayland Compositor
/// Provides zero-copy cursor rendering with hardware acceleration

pub const CursorError = error{
    CursorPlaneNotFound,
    CursorSizeNotSupported,
    CursorFormatNotSupported,
    CursorUpdateFailed,
    HardwareError,
    OutOfMemory,
    PermissionDenied,
};

pub const CursorFormat = enum(u32) {
    ARGB8888 = 0x34325241,
    RGBA8888 = 0x34324152,
    RGB565 = 0x36314752,
    
    pub fn getBytesPerPixel(self: CursorFormat) u8 {
        return switch (self) {
            .ARGB8888, .RGBA8888 => 4,
            .RGB565 => 2,
        };
    }
    
    pub fn hasAlpha(self: CursorFormat) bool {
        return switch (self) {
            .ARGB8888, .RGBA8888 => true,
            .RGB565 => false,
        };
    }
};

pub const CursorSize = enum(u16) {
    size_16x16 = 16,
    size_32x32 = 32,
    size_64x64 = 64,
    size_128x128 = 128,
    size_256x256 = 256,
    
    pub fn isValid(width: u16, height: u16) bool {
        const size = @max(width, height);
        return size == 16 or size == 32 or size == 64 or size == 128 or size == 256;
    }
    
    pub fn getAlignment(self: CursorSize) u16 {
        return switch (self) {
            .size_16x16, .size_32x32 => 16,
            .size_64x64 => 32,
            .size_128x128 => 64,
            .size_256x256 => 128,
        };
    }
};

pub const CursorBuffer = struct {
    width: u16,
    height: u16,
    format: CursorFormat,
    stride: u32,
    size: usize,
    dma_buf: *memory.DmaBuffer,
    gpu_address: u64,
    hotspot_x: i16,
    hotspot_y: i16,
    
    pub fn init(allocator: std.mem.Allocator, width: u16, height: u16, format: CursorFormat, hotspot_x: i16, hotspot_y: i16) !CursorBuffer {
        if (!CursorSize.isValid(width, height)) {
            return CursorError.CursorSizeNotSupported;
        }
        
        const bpp = format.getBytesPerPixel();
        const stride = @as(u32, width) * bpp;
        const size = stride * height;
        
        // Allocate DMA buffer for cursor
        var dma_buf = try allocator.create(memory.DmaBuffer);
        dma_buf.* = try memory.DmaBuffer.init(allocator, size, .coherent);
        
        return CursorBuffer{
            .width = width,
            .height = height,
            .format = format,
            .stride = stride,
            .size = size,
            .dma_buf = dma_buf,
            .gpu_address = dma_buf.physical_address,
            .hotspot_x = hotspot_x,
            .hotspot_y = hotspot_y,
        };
    }
    
    pub fn deinit(self: *CursorBuffer, allocator: std.mem.Allocator) void {
        self.dma_buf.deinit();
        allocator.destroy(self.dma_buf);
    }
    
    pub fn updatePixelData(self: *CursorBuffer, pixel_data: []const u8) !void {
        if (pixel_data.len != self.size) {
            return CursorError.CursorUpdateFailed;
        }
        
        // Copy pixel data to DMA buffer
        @memcpy(self.dma_buf.virtual_address[0..self.size], pixel_data);
        
        // Ensure cache coherency
        try self.dma_buf.sync();
    }
    
    pub fn clear(self: *CursorBuffer) void {
        @memset(self.dma_buf.virtual_address[0..self.size], 0);
    }
};

pub const CursorState = struct {
    visible: bool,
    x: i32,
    y: i32,
    buffer: ?*CursorBuffer,
    last_update_frame: u64,
    
    pub fn init() CursorState {
        return CursorState{
            .visible = false,
            .x = 0,
            .y = 0,
            .buffer = null,
            .last_update_frame = 0,
        };
    }
    
    pub fn setPosition(self: *CursorState, x: i32, y: i32) void {
        self.x = x;
        self.y = y;
    }
    
    pub fn setBuffer(self: *CursorState, buffer: ?*CursorBuffer) void {
        self.buffer = buffer;
        self.visible = buffer != null;
    }
    
    pub fn getHotspotAdjustedPosition(self: *const CursorState) struct { x: i32, y: i32 } {
        if (self.buffer) |buf| {
            return .{
                .x = self.x - buf.hotspot_x,
                .y = self.y - buf.hotspot_y,
            };
        }
        return .{ .x = self.x, .y = self.y };
    }
};

pub const CursorPlane = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    plane_id: u32,
    crtc_id: u32,
    drm_fd: i32,
    supported_formats: []CursorFormat,
    max_width: u16,
    max_height: u16,
    current_state: CursorState,
    gamma_lut: ?[]u16, // Optional gamma correction LUT
    
    // Hardware cursor plane registers
    const CURSOR_BASE_ADDR: u32 = 0x14B000;
    const CURSOR_CONTROL: u32 = CURSOR_BASE_ADDR + 0x000;
    const CURSOR_POSITION: u32 = CURSOR_BASE_ADDR + 0x004;
    const CURSOR_SIZE: u32 = CURSOR_BASE_ADDR + 0x008;
    const CURSOR_SURFACE: u32 = CURSOR_BASE_ADDR + 0x00C;
    const CURSOR_HOTSPOT: u32 = CURSOR_BASE_ADDR + 0x010;
    const CURSOR_ALPHA: u32 = CURSOR_BASE_ADDR + 0x014;
    
    pub fn init(allocator: std.mem.Allocator, drm_fd: i32, crtc_id: u32) !Self {
        // Find cursor plane for this CRTC
        const plane_id = try Self.findCursorPlane(drm_fd, crtc_id);
        
        // Query supported formats and capabilities
        const formats = try Self.querySupportedFormats(allocator, drm_fd, plane_id);
        const caps = try Self.queryCapabilities(drm_fd, plane_id);
        
        return Self{
            .allocator = allocator,
            .plane_id = plane_id,
            .crtc_id = crtc_id,
            .drm_fd = drm_fd,
            .supported_formats = formats,
            .max_width = caps.max_width,
            .max_height = caps.max_height,
            .current_state = CursorState.init(),
            .gamma_lut = null,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.supported_formats);
        if (self.gamma_lut) |lut| {
            self.allocator.free(lut);
        }
    }
    
    fn findCursorPlane(drm_fd: i32, crtc_id: u32) !u32 {
        // Query DRM plane resources
        var plane_res: linux.drm.mode_get_plane_res = undefined;
        plane_res.plane_id_ptr = 0;
        plane_res.count_planes = 0;
        
        if (linux.syscall(.ioctl, @as(usize, @intCast(drm_fd)), linux.DRM.IOCTL_MODE_GETPLANERESOURCES, @intFromPtr(&plane_res)) != 0) {
            return CursorError.CursorPlaneNotFound;
        }
        
        // Allocate buffer for plane IDs
        const plane_ids = try std.heap.page_allocator.alloc(u32, plane_res.count_planes);
        defer std.heap.page_allocator.free(plane_ids);
        
        plane_res.plane_id_ptr = @intFromPtr(plane_ids.ptr);
        
        if (linux.syscall(.ioctl, @as(usize, @intCast(drm_fd)), linux.DRM.IOCTL_MODE_GETPLANERESOURCES, @intFromPtr(&plane_res)) != 0) {
            return CursorError.CursorPlaneNotFound;
        }
        
        // Find cursor plane that can be used with this CRTC
        for (plane_ids) |plane_id| {
            var plane: linux.drm.mode_get_plane = undefined;
            plane.plane_id = plane_id;
            
            if (linux.syscall(.ioctl, @as(usize, @intCast(drm_fd)), linux.DRM.IOCTL_MODE_GETPLANE, @intFromPtr(&plane)) == 0) {
                // Check if this plane supports cursor functionality and is compatible with CRTC
                if ((plane.possible_crtcs & (@as(u32, 1) << @truncate(crtc_id))) != 0) {
                    // Additional check for cursor plane type would go here
                    return plane_id;
                }
            }
        }
        
        return CursorError.CursorPlaneNotFound;
    }
    
    fn querySupportedFormats(allocator: std.mem.Allocator, drm_fd: i32, plane_id: u32) ![]CursorFormat {
        _ = drm_fd;
        _ = plane_id;
        
        // In real implementation, query actual hardware formats
        // For now, return common supported formats
        const formats = try allocator.alloc(CursorFormat, 3);
        formats[0] = .ARGB8888;
        formats[1] = .RGBA8888;
        formats[2] = .RGB565;
        
        return formats;
    }
    
    fn queryCapabilities(drm_fd: i32, plane_id: u32) !struct { max_width: u16, max_height: u16 } {
        _ = drm_fd;
        _ = plane_id;
        
        // Query hardware capabilities
        // For modern NVIDIA hardware, typical cursor sizes are 256x256
        return .{
            .max_width = 256,
            .max_height = 256,
        };
    }
    
    pub fn setCursor(self: *Self, buffer: ?*CursorBuffer, x: i32, y: i32) !void {
        self.current_state.setBuffer(buffer);
        self.current_state.setPosition(x, y);
        
        if (buffer) |buf| {
            // Validate cursor dimensions
            if (buf.width > self.max_width or buf.height > self.max_height) {
                return CursorError.CursorSizeNotSupported;
            }
            
            // Check format support
            var format_supported = false;
            for (self.supported_formats) |fmt| {
                if (fmt == buf.format) {
                    format_supported = true;
                    break;
                }
            }
            if (!format_supported) {
                return CursorError.CursorFormatNotSupported;
            }
            
            try self.updateHardwareCursor();
        } else {
            try self.hideCursor();
        }
    }
    
    pub fn moveCursor(self: *Self, x: i32, y: i32) !void {
        self.current_state.setPosition(x, y);
        
        if (self.current_state.visible) {
            try self.updateCursorPosition();
        }
    }
    
    pub fn showCursor(self: *Self) !void {
        if (self.current_state.buffer != null) {
            self.current_state.visible = true;
            try self.updateHardwareCursor();
        }
    }
    
    pub fn hideCursor(self: *Self) !void {
        self.current_state.visible = false;
        
        // Use atomic property to disable cursor plane
        try self.updateCursorVisibility(false);
    }
    
    fn updateHardwareCursor(self: *Self) !void {
        if (self.current_state.buffer) |buffer| {
            const pos = self.current_state.getHotspotAdjustedPosition();
            
            // Set cursor surface address
            try self.setProperty("FB_ID", @as(u64, buffer.gpu_address));
            
            // Set cursor position
            const position = (@as(u64, @bitCast(@as(i64, pos.y))) << 32) | @as(u32, @bitCast(pos.x));
            try self.setProperty("CRTC_X", @as(u64, @bitCast(@as(i64, pos.x))));
            try self.setProperty("CRTC_Y", @as(u64, @bitCast(@as(i64, pos.y))));
            
            // Set cursor size
            try self.setProperty("CRTC_W", buffer.width);
            try self.setProperty("CRTC_H", buffer.height);
            
            // Enable cursor
            try self.updateCursorVisibility(true);
        }
    }
    
    fn updateCursorPosition(self: *Self) !void {
        const pos = self.current_state.getHotspotAdjustedPosition();
        
        try self.setProperty("CRTC_X", @as(u64, @bitCast(@as(i64, pos.x))));
        try self.setProperty("CRTC_Y", @as(u64, @bitCast(@as(i64, pos.y))));
    }
    
    fn updateCursorVisibility(self: *Self, visible: bool) !void {
        const fb_id: u64 = if (visible and self.current_state.buffer != null) 
            self.current_state.buffer.?.gpu_address
        else 
            0;
            
        try self.setProperty("FB_ID", fb_id);
    }
    
    fn setProperty(self: *Self, name: []const u8, value: u64) !void {
        // Use DRM atomic properties to set cursor plane properties
        var prop: linux.drm.mode_obj_set_property = undefined;
        prop.obj_id = self.plane_id;
        prop.obj_type = linux.DRM.MODE_OBJECT_PLANE;
        prop.prop_id = try self.getPropertyId(name);
        prop.value = value;
        
        if (linux.syscall(.ioctl, @as(usize, @intCast(self.drm_fd)), linux.DRM.IOCTL_MODE_OBJ_SETPROPERTY, @intFromPtr(&prop)) != 0) {
            return CursorError.CursorUpdateFailed;
        }
    }
    
    fn getPropertyId(self: *Self, name: []const u8) !u32 {
        _ = self;
        _ = name;
        
        // In real implementation, query property ID by name
        // This is simplified
        return if (std.mem.eql(u8, name, "FB_ID"))
            1
        else if (std.mem.eql(u8, name, "CRTC_X"))
            2
        else if (std.mem.eql(u8, name, "CRTC_Y"))
            3
        else if (std.mem.eql(u8, name, "CRTC_W"))
            4
        else if (std.mem.eql(u8, name, "CRTC_H"))
            5
        else
            0;
    }
    
    pub fn commitCursor(self: *Self) !void {
        // Use atomic commit for cursor updates
        var req: linux.drm.mode_atomic = undefined;
        req.flags = linux.DRM.MODE_ATOMIC_NONBLOCK;
        req.count_objs = 1;
        
        // In real implementation, prepare atomic request with all cursor properties
        if (linux.syscall(.ioctl, @as(usize, @intCast(self.drm_fd)), linux.DRM.IOCTL_MODE_ATOMIC, @intFromPtr(&req)) != 0) {
            return CursorError.CursorUpdateFailed;
        }
    }
    
    pub fn getCurrentPosition(self: *const Self) struct { x: i32, y: i32 } {
        return .{ .x = self.current_state.x, .y = self.current_state.y };
    }
    
    pub fn isVisible(self: *const Self) bool {
        return self.current_state.visible;
    }
    
    pub fn getMaxSize(self: *const Self) struct { width: u16, height: u16 } {
        return .{ .width = self.max_width, .height = self.max_height };
    }
    
    pub fn isFormatSupported(self: *const Self, format: CursorFormat) bool {
        for (self.supported_formats) |fmt| {
            if (fmt == format) return true;
        }
        return false;
    }
    
    pub fn setGammaCorrection(self: *Self, lut: []const u16) !void {
        if (self.gamma_lut) |old_lut| {
            self.allocator.free(old_lut);
        }
        
        self.gamma_lut = try self.allocator.dupe(u16, lut);
        
        // Apply gamma LUT to hardware
        try self.setProperty("GAMMA_LUT", @intFromPtr(self.gamma_lut.?.ptr));
    }
    
    pub fn enableHardwareCursorAcceleration(self: *Self) !void {
        // Enable hardware acceleration features like cursor scaling, color correction
        try self.setProperty("scaling mode", 1); // Hardware scaling
        try self.setProperty("color_encoding", 1); // BT.709 color encoding
        try self.setProperty("color_range", 1); // Limited range
    }
};

/// Cursor Manager for Multi-Display Setups
pub const CursorManager = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    cursor_planes: std.AutoHashMap(u32, CursorPlane), // crtc_id -> CursorPlane
    buffer_pool: std.ArrayList(*CursorBuffer),
    active_cursor: ?*CursorBuffer,
    global_position: struct { x: i32, y: i32 },
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .cursor_planes = std.AutoHashMap(u32, CursorPlane).init(allocator),
            .buffer_pool = std.ArrayList(*CursorBuffer).init(allocator),
            .active_cursor = null,
            .global_position = .{ .x = 0, .y = 0 },
        };
    }
    
    pub fn deinit(self: *Self) void {
        // Clean up cursor planes
        var plane_iter = self.cursor_planes.valueIterator();
        while (plane_iter.next()) |plane| {
            plane.deinit();
        }
        self.cursor_planes.deinit();
        
        // Clean up buffer pool
        for (self.buffer_pool.items) |buffer| {
            buffer.deinit(self.allocator);
        }
        self.buffer_pool.deinit();
    }
    
    pub fn addDisplay(self: *Self, drm_fd: i32, crtc_id: u32) !void {
        const cursor_plane = try CursorPlane.init(self.allocator, drm_fd, crtc_id);
        try self.cursor_planes.put(crtc_id, cursor_plane);
    }
    
    pub fn removeDisplay(self: *Self, crtc_id: u32) void {
        if (self.cursor_planes.fetchRemove(crtc_id)) |kv| {
            kv.value.deinit();
        }
    }
    
    pub fn setCursor(self: *Self, buffer: ?*CursorBuffer) !void {
        self.active_cursor = buffer;
        
        // Update cursor on all displays
        var plane_iter = self.cursor_planes.valueIterator();
        while (plane_iter.next()) |plane| {
            try plane.setCursor(buffer, self.global_position.x, self.global_position.y);
        }
    }
    
    pub fn moveCursor(self: *Self, x: i32, y: i32) !void {
        self.global_position.x = x;
        self.global_position.y = y;
        
        // Update cursor position on all displays
        var plane_iter = self.cursor_planes.valueIterator();
        while (plane_iter.next()) |plane| {
            try plane.moveCursor(x, y);
        }
    }
    
    pub fn createCursorBuffer(self: *Self, width: u16, height: u16, format: CursorFormat, hotspot_x: i16, hotspot_y: i16) !*CursorBuffer {
        const buffer = try self.allocator.create(CursorBuffer);
        buffer.* = try CursorBuffer.init(self.allocator, width, height, format, hotspot_x, hotspot_y);
        
        try self.buffer_pool.append(buffer);
        return buffer;
    }
    
    pub fn destroyCursorBuffer(self: *Self, buffer: *CursorBuffer) void {
        // Remove from pool
        for (self.buffer_pool.items, 0..) |buf, i| {
            if (buf == buffer) {
                _ = self.buffer_pool.orderedRemove(i);
                break;
            }
        }
        
        buffer.deinit(self.allocator);
        self.allocator.destroy(buffer);
        
        // Clear active cursor if it was destroyed
        if (self.active_cursor == buffer) {
            self.active_cursor = null;
        }
    }
    
    pub fn commitAll(self: *Self) !void {
        var plane_iter = self.cursor_planes.valueIterator();
        while (plane_iter.next()) |plane| {
            try plane.commitCursor();
        }
    }
};