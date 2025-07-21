const std = @import("std");
const drm = @import("../drm/driver.zig");
const memory = @import("../hal/memory.zig");
const display = @import("../display/engine.zig");
const linux = std.os.linux;

/// Hardware Overlay Plane Management for Wayland Compositor
/// Provides efficient video overlay composition with zero-copy optimization

pub const OverlayError = error{
    OverlayPlaneNotFound,
    UnsupportedFormat,
    UnsupportedSize,
    ScalingNotSupported,
    OverlayUpdateFailed,
    PlaneInUse,
    HardwareError,
    OutOfMemory,
    PermissionDenied,
};

pub const OverlayFormat = enum(u32) {
    // RGB formats
    XRGB8888 = 0x34325258,
    ARGB8888 = 0x34325241,
    RGBA8888 = 0x34324152,
    RGBX8888 = 0x34324752,
    RGB565 = 0x36314752,
    
    // YUV formats for video
    YUV420 = 0x32315659,
    YUV422 = 0x36315659,
    YUV444 = 0x34325559,
    NV12 = 0x3231564E,
    NV21 = 0x3132564E,
    P010 = 0x30313050, // 10-bit YUV420
    P016 = 0x36313050, // 16-bit YUV420
    
    pub fn getBytesPerPixel(self: OverlayFormat) u8 {
        return switch (self) {
            .XRGB8888, .ARGB8888, .RGBA8888, .RGBX8888 => 4,
            .RGB565 => 2,
            .YUV420, .NV12, .NV21 => 1, // Planar, varies
            .YUV422 => 2,
            .YUV444 => 3,
            .P010, .P016 => 2, // 16-bit per component
        };
    }
    
    pub fn isYuv(self: OverlayFormat) bool {
        return switch (self) {
            .YUV420, .YUV422, .YUV444, .NV12, .NV21, .P010, .P016 => true,
            else => false,
        };
    }
    
    pub fn hasAlpha(self: OverlayFormat) bool {
        return switch (self) {
            .ARGB8888, .RGBA8888 => true,
            else => false,
        };
    }
    
    pub fn isHdr(self: OverlayFormat) bool {
        return switch (self) {
            .P010, .P016 => true,
            else => false,
        };
    }
};

pub const OverlayScaling = enum(u8) {
    none = 0,
    nearest = 1,
    bilinear = 2,
    bicubic = 3,
    lanczos = 4,
    
    pub fn getQuality(self: OverlayScaling) f32 {
        return switch (self) {
            .none => 1.0,
            .nearest => 0.5,
            .bilinear => 0.7,
            .bicubic => 0.85,
            .lanczos => 0.95,
        };
    }
};

pub const OverlayColorSpace = enum(u8) {
    bt601 = 0,     // SDTV
    bt709 = 1,     // HDTV
    bt2020 = 2,    // UHDTV/HDR
    dci_p3 = 3,    // Digital cinema
    
    pub fn getGamut(self: OverlayColorSpace) struct { r: [2]f32, g: [2]f32, b: [2]f32 } {
        return switch (self) {
            .bt601 => .{
                .r = .{ 0.640, 0.330 },
                .g = .{ 0.290, 0.600 },
                .b = .{ 0.150, 0.060 },
            },
            .bt709 => .{
                .r = .{ 0.640, 0.330 },
                .g = .{ 0.300, 0.600 },
                .b = .{ 0.150, 0.060 },
            },
            .bt2020 => .{
                .r = .{ 0.708, 0.292 },
                .g = .{ 0.170, 0.797 },
                .b = .{ 0.131, 0.046 },
            },
            .dci_p3 => .{
                .r = .{ 0.680, 0.320 },
                .g = .{ 0.265, 0.690 },
                .b = .{ 0.150, 0.060 },
            },
        };
    }
};

pub const OverlayBuffer = struct {
    width: u32,
    height: u32,
    format: OverlayFormat,
    stride: [4]u32,     // Stride for each plane (up to 4 planes for YUV)
    offset: [4]u32,     // Offset for each plane
    dma_buf: [4]?*memory.DmaBuffer, // DMA buffer for each plane
    gpu_addresses: [4]u64,
    plane_count: u8,
    
    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32, format: OverlayFormat) !OverlayBuffer {
        var buffer = OverlayBuffer{
            .width = width,
            .height = height,
            .format = format,
            .stride = [_]u32{0} ** 4,
            .offset = [_]u32{0} ** 4,
            .dma_buf = [_]?*memory.DmaBuffer{null} ** 4,
            .gpu_addresses = [_]u64{0} ** 4,
            .plane_count = 0,
        };
        
        // Calculate plane layout based on format
        switch (format) {
            .XRGB8888, .ARGB8888, .RGBA8888, .RGBX8888, .RGB565 => {
                // Single plane RGB
                buffer.plane_count = 1;
                buffer.stride[0] = width * format.getBytesPerPixel();
                const size = buffer.stride[0] * height;
                
                buffer.dma_buf[0] = try allocator.create(memory.DmaBuffer);
                buffer.dma_buf[0].?.* = try memory.DmaBuffer.init(allocator, size, .coherent);
                buffer.gpu_addresses[0] = buffer.dma_buf[0].?.physical_address;
            },
            
            .YUV420, .NV12, .NV21 => {
                // Two planes: Y + UV
                buffer.plane_count = 2;
                
                // Y plane (full resolution)
                buffer.stride[0] = width;
                const y_size = buffer.stride[0] * height;
                buffer.dma_buf[0] = try allocator.create(memory.DmaBuffer);
                buffer.dma_buf[0].?.* = try memory.DmaBuffer.init(allocator, y_size, .coherent);
                buffer.gpu_addresses[0] = buffer.dma_buf[0].?.physical_address;
                
                // UV plane (half resolution)
                buffer.stride[1] = width;
                buffer.offset[1] = y_size;
                const uv_size = buffer.stride[1] * (height / 2);
                buffer.dma_buf[1] = try allocator.create(memory.DmaBuffer);
                buffer.dma_buf[1].?.* = try memory.DmaBuffer.init(allocator, uv_size, .coherent);
                buffer.gpu_addresses[1] = buffer.dma_buf[1].?.physical_address;
            },
            
            .P010, .P016 => {
                // HDR YUV420 - 10/16-bit per component
                buffer.plane_count = 2;
                const bpp = if (format == .P010) 2 else 2; // 16-bit per component
                
                // Y plane
                buffer.stride[0] = width * bpp;
                const y_size = buffer.stride[0] * height;
                buffer.dma_buf[0] = try allocator.create(memory.DmaBuffer);
                buffer.dma_buf[0].?.* = try memory.DmaBuffer.init(allocator, y_size, .coherent);
                buffer.gpu_addresses[0] = buffer.dma_buf[0].?.physical_address;
                
                // UV plane
                buffer.stride[1] = width * bpp;
                buffer.offset[1] = y_size;
                const uv_size = buffer.stride[1] * (height / 2);
                buffer.dma_buf[1] = try allocator.create(memory.DmaBuffer);
                buffer.dma_buf[1].?.* = try memory.DmaBuffer.init(allocator, uv_size, .coherent);
                buffer.gpu_addresses[1] = buffer.dma_buf[1].?.physical_address;
            },
            
            else => {
                return OverlayError.UnsupportedFormat;
            },
        }
        
        return buffer;
    }
    
    pub fn deinit(self: *OverlayBuffer, allocator: std.mem.Allocator) void {
        for (0..self.plane_count) |i| {
            if (self.dma_buf[i]) |buf| {
                buf.deinit();
                allocator.destroy(buf);
            }
        }
    }
    
    pub fn updatePlane(self: *OverlayBuffer, plane_index: u8, data: []const u8) !void {
        if (plane_index >= self.plane_count) return OverlayError.OverlayUpdateFailed;
        
        if (self.dma_buf[plane_index]) |buf| {
            const expected_size = self.stride[plane_index] * switch (plane_index) {
                0 => self.height, // Y plane full height
                1 => self.height / 2, // UV plane half height for YUV420
                else => self.height,
            };
            
            if (data.len != expected_size) return OverlayError.OverlayUpdateFailed;
            
            @memcpy(buf.virtual_address[0..data.len], data);
            try buf.sync();
        }
    }
};

pub const OverlayPlane = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    plane_id: u32,
    crtc_id: u32,
    drm_fd: i32,
    z_order: u8,
    supported_formats: []OverlayFormat,
    max_width: u32,
    max_height: u32,
    min_scale: f32,
    max_scale: f32,
    current_buffer: ?*OverlayBuffer,
    src_rect: struct { x: u32, y: u32, w: u32, h: u32 },
    dst_rect: struct { x: i32, y: i32, w: u32, h: u32 },
    alpha: u8,
    color_space: OverlayColorSpace,
    scaling_mode: OverlayScaling,
    visible: bool,
    
    pub fn init(allocator: std.mem.Allocator, drm_fd: i32, crtc_id: u32, z_order: u8) !Self {
        const plane_id = try Self.findOverlayPlane(drm_fd, crtc_id, z_order);
        const formats = try Self.querySupportedFormats(allocator, drm_fd, plane_id);
        const caps = try Self.queryCapabilities(drm_fd, plane_id);
        
        return Self{
            .allocator = allocator,
            .plane_id = plane_id,
            .crtc_id = crtc_id,
            .drm_fd = drm_fd,
            .z_order = z_order,
            .supported_formats = formats,
            .max_width = caps.max_width,
            .max_height = caps.max_height,
            .min_scale = caps.min_scale,
            .max_scale = caps.max_scale,
            .current_buffer = null,
            .src_rect = .{ .x = 0, .y = 0, .w = 0, .h = 0 },
            .dst_rect = .{ .x = 0, .y = 0, .w = 0, .h = 0 },
            .alpha = 255,
            .color_space = .bt709,
            .scaling_mode = .bilinear,
            .visible = false,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.supported_formats);
    }
    
    fn findOverlayPlane(drm_fd: i32, crtc_id: u32, z_order: u8) !u32 {
        _ = z_order; // Would be used to find specific overlay plane by z-order
        
        // Query DRM plane resources
        var plane_res: linux.drm.mode_get_plane_res = undefined;
        plane_res.plane_id_ptr = 0;
        plane_res.count_planes = 0;
        
        if (linux.syscall(.ioctl, @as(usize, @intCast(drm_fd)), linux.DRM.IOCTL_MODE_GETPLANERESOURCES, @intFromPtr(&plane_res)) != 0) {
            return OverlayError.OverlayPlaneNotFound;
        }
        
        const plane_ids = try std.heap.page_allocator.alloc(u32, plane_res.count_planes);
        defer std.heap.page_allocator.free(plane_ids);
        
        plane_res.plane_id_ptr = @intFromPtr(plane_ids.ptr);
        
        if (linux.syscall(.ioctl, @as(usize, @intCast(drm_fd)), linux.DRM.IOCTL_MODE_GETPLANERESOURCES, @intFromPtr(&plane_res)) != 0) {
            return OverlayError.OverlayPlaneNotFound;
        }
        
        // Find overlay plane compatible with CRTC
        for (plane_ids) |plane_id| {
            var plane: linux.drm.mode_get_plane = undefined;
            plane.plane_id = plane_id;
            
            if (linux.syscall(.ioctl, @as(usize, @intCast(drm_fd)), linux.DRM.IOCTL_MODE_GETPLANE, @intFromPtr(&plane)) == 0) {
                if ((plane.possible_crtcs & (@as(u32, 1) << @truncate(crtc_id))) != 0) {
                    // Check if this is an overlay plane (not cursor or primary)
                    // In real implementation, check plane type property
                    return plane_id;
                }
            }
        }
        
        return OverlayError.OverlayPlaneNotFound;
    }
    
    fn querySupportedFormats(allocator: std.mem.Allocator, drm_fd: i32, plane_id: u32) ![]OverlayFormat {
        _ = drm_fd;
        _ = plane_id;
        
        // In real implementation, query actual hardware formats
        const formats = try allocator.alloc(OverlayFormat, 8);
        formats[0] = .XRGB8888;
        formats[1] = .ARGB8888;
        formats[2] = .YUV420;
        formats[3] = .NV12;
        formats[4] = .NV21;
        formats[5] = .P010;
        formats[6] = .RGB565;
        formats[7] = .YUV422;
        
        return formats;
    }
    
    fn queryCapabilities(drm_fd: i32, plane_id: u32) !struct { max_width: u32, max_height: u32, min_scale: f32, max_scale: f32 } {
        _ = drm_fd;
        _ = plane_id;
        
        return .{
            .max_width = 4096,
            .max_height = 4096,
            .min_scale = 0.25, // 4:1 downscale
            .max_scale = 4.0,  // 4:1 upscale
        };
    }
    
    pub fn setBuffer(self: *Self, buffer: ?*OverlayBuffer, src_rect: ?struct { x: u32, y: u32, w: u32, h: u32 }, dst_rect: ?struct { x: i32, y: i32, w: u32, h: u32 }) !void {
        self.current_buffer = buffer;
        
        if (buffer) |buf| {
            // Validate format support
            var format_supported = false;
            for (self.supported_formats) |fmt| {
                if (fmt == buf.format) {
                    format_supported = true;
                    break;
                }
            }
            if (!format_supported) {
                return OverlayError.UnsupportedFormat;
            }
            
            // Set source rectangle (portion of buffer to display)
            if (src_rect) |src| {
                self.src_rect = src;
            } else {
                self.src_rect = .{ .x = 0, .y = 0, .w = buf.width, .h = buf.height };
            }
            
            // Set destination rectangle (where to display on screen)
            if (dst_rect) |dst| {
                self.dst_rect = dst;
            } else {
                self.dst_rect = .{ .x = 0, .y = 0, .w = buf.width, .h = buf.height };
            }
            
            // Validate scaling ratio
            const scale_x = @as(f32, @floatFromInt(self.dst_rect.w)) / @as(f32, @floatFromInt(self.src_rect.w));
            const scale_y = @as(f32, @floatFromInt(self.dst_rect.h)) / @as(f32, @floatFromInt(self.src_rect.h));
            
            if (scale_x < self.min_scale or scale_x > self.max_scale or
                scale_y < self.min_scale or scale_y > self.max_scale) {
                return OverlayError.ScalingNotSupported;
            }
            
            self.visible = true;
            try self.updateHardwareOverlay();
        } else {
            self.visible = false;
            try self.hideOverlay();
        }
    }
    
    fn updateHardwareOverlay(self: *Self) !void {
        if (self.current_buffer) |buffer| {
            // Set buffer addresses for each plane
            for (0..buffer.plane_count) |i| {
                try self.setProperty("FB_ID", buffer.gpu_addresses[i]);
            }
            
            // Set source rectangle (fixed point 16.16)
            try self.setProperty("SRC_X", (@as(u64, self.src_rect.x) << 16));
            try self.setProperty("SRC_Y", (@as(u64, self.src_rect.y) << 16));
            try self.setProperty("SRC_W", (@as(u64, self.src_rect.w) << 16));
            try self.setProperty("SRC_H", (@as(u64, self.src_rect.h) << 16));
            
            // Set destination rectangle
            try self.setProperty("CRTC_X", @as(u64, @bitCast(@as(i64, self.dst_rect.x))));
            try self.setProperty("CRTC_Y", @as(u64, @bitCast(@as(i64, self.dst_rect.y))));
            try self.setProperty("CRTC_W", self.dst_rect.w);
            try self.setProperty("CRTC_H", self.dst_rect.h);
            
            // Set color space and encoding
            try self.setProperty("COLOR_ENCODING", @intFromEnum(self.color_space));
            try self.setProperty("COLOR_RANGE", 1); // Limited range
            
            // Set alpha blending
            try self.setProperty("alpha", self.alpha);
            
            // Set scaling filter
            try self.setProperty("scaling filter", @intFromEnum(self.scaling_mode));
        }
    }
    
    fn hideOverlay(self: *Self) !void {
        try self.setProperty("FB_ID", 0);
        try self.setProperty("CRTC_ID", 0);
    }
    
    fn setProperty(self: *Self, name: []const u8, value: u64) !void {
        var prop: linux.drm.mode_obj_set_property = undefined;
        prop.obj_id = self.plane_id;
        prop.obj_type = linux.DRM.MODE_OBJECT_PLANE;
        prop.prop_id = try self.getPropertyId(name);
        prop.value = value;
        
        if (linux.syscall(.ioctl, @as(usize, @intCast(self.drm_fd)), linux.DRM.IOCTL_MODE_OBJ_SETPROPERTY, @intFromPtr(&prop)) != 0) {
            return OverlayError.OverlayUpdateFailed;
        }
    }
    
    fn getPropertyId(self: *Self, name: []const u8) !u32 {
        _ = self;
        
        // Simplified property ID lookup
        return if (std.mem.eql(u8, name, "FB_ID"))
            1
        else if (std.mem.eql(u8, name, "CRTC_ID"))
            2
        else if (std.mem.eql(u8, name, "SRC_X"))
            3
        else if (std.mem.eql(u8, name, "SRC_Y"))
            4
        else if (std.mem.eql(u8, name, "SRC_W"))
            5
        else if (std.mem.eql(u8, name, "SRC_H"))
            6
        else if (std.mem.eql(u8, name, "CRTC_X"))
            7
        else if (std.mem.eql(u8, name, "CRTC_Y"))
            8
        else if (std.mem.eql(u8, name, "CRTC_W"))
            9
        else if (std.mem.eql(u8, name, "CRTC_H"))
            10
        else if (std.mem.eql(u8, name, "COLOR_ENCODING"))
            11
        else if (std.mem.eql(u8, name, "COLOR_RANGE"))
            12
        else if (std.mem.eql(u8, name, "alpha"))
            13
        else if (std.mem.eql(u8, name, "scaling filter"))
            14
        else
            return OverlayError.OverlayUpdateFailed;
    }
    
    pub fn setAlpha(self: *Self, alpha: u8) !void {
        self.alpha = alpha;
        if (self.visible) {
            try self.setProperty("alpha", alpha);
        }
    }
    
    pub fn setColorSpace(self: *Self, color_space: OverlayColorSpace) !void {
        self.color_space = color_space;
        if (self.visible) {
            try self.setProperty("COLOR_ENCODING", @intFromEnum(color_space));
        }
    }
    
    pub fn setScalingMode(self: *Self, scaling: OverlayScaling) !void {
        self.scaling_mode = scaling;
        if (self.visible) {
            try self.setProperty("scaling filter", @intFromEnum(scaling));
        }
    }
    
    pub fn move(self: *Self, x: i32, y: i32) !void {
        self.dst_rect.x = x;
        self.dst_rect.y = y;
        
        if (self.visible) {
            try self.setProperty("CRTC_X", @as(u64, @bitCast(@as(i64, x))));
            try self.setProperty("CRTC_Y", @as(u64, @bitCast(@as(i64, y))));
        }
    }
    
    pub fn resize(self: *Self, width: u32, height: u32) !void {
        if (self.current_buffer) |buffer| {
            // Validate scaling
            const scale_x = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(buffer.width));
            const scale_y = @as(f32, @floatFromInt(height)) / @as(f32, @floatFromInt(buffer.height));
            
            if (scale_x < self.min_scale or scale_x > self.max_scale or
                scale_y < self.min_scale or scale_y > self.max_scale) {
                return OverlayError.ScalingNotSupported;
            }
        }
        
        self.dst_rect.w = width;
        self.dst_rect.h = height;
        
        if (self.visible) {
            try self.setProperty("CRTC_W", width);
            try self.setProperty("CRTC_H", height);
        }
    }
    
    pub fn commitOverlay(self: *Self) !void {
        // Use atomic commit for overlay updates
        var req: linux.drm.mode_atomic = undefined;
        req.flags = linux.DRM.MODE_ATOMIC_NONBLOCK;
        req.count_objs = 1;
        
        if (linux.syscall(.ioctl, @as(usize, @intCast(self.drm_fd)), linux.DRM.IOCTL_MODE_ATOMIC, @intFromPtr(&req)) != 0) {
            return OverlayError.OverlayUpdateFailed;
        }
    }
    
    pub fn isFormatSupported(self: *const Self, format: OverlayFormat) bool {
        for (self.supported_formats) |fmt| {
            if (fmt == format) return true;
        }
        return false;
    }
    
    pub fn getScalingLimits(self: *const Self) struct { min: f32, max: f32 } {
        return .{ .min = self.min_scale, .max = self.max_scale };
    }
};

/// Multi-Plane Overlay Manager for Video and UI Composition
pub const OverlayManager = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    overlay_planes: std.ArrayList(OverlayPlane),
    buffer_pool: std.ArrayList(*OverlayBuffer),
    composition_order: []u8, // Z-order for composition
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .overlay_planes = std.ArrayList(OverlayPlane).init(allocator),
            .buffer_pool = std.ArrayList(*OverlayBuffer).init(allocator),
            .composition_order = &.{},
        };
    }
    
    pub fn deinit(self: *Self) void {
        for (self.overlay_planes.items) |*plane| {
            plane.deinit();
        }
        self.overlay_planes.deinit();
        
        for (self.buffer_pool.items) |buffer| {
            buffer.deinit(self.allocator);
        }
        self.buffer_pool.deinit();
        
        if (self.composition_order.len > 0) {
            self.allocator.free(self.composition_order);
        }
    }
    
    pub fn addOverlayPlane(self: *Self, drm_fd: i32, crtc_id: u32, z_order: u8) !u8 {
        const plane = try OverlayPlane.init(self.allocator, drm_fd, crtc_id, z_order);
        try self.overlay_planes.append(plane);
        return @as(u8, @intCast(self.overlay_planes.items.len - 1));
    }
    
    pub fn createOverlayBuffer(self: *Self, width: u32, height: u32, format: OverlayFormat) !*OverlayBuffer {
        const buffer = try self.allocator.create(OverlayBuffer);
        buffer.* = try OverlayBuffer.init(self.allocator, width, height, format);
        
        try self.buffer_pool.append(buffer);
        return buffer;
    }
    
    pub fn destroyOverlayBuffer(self: *Self, buffer: *OverlayBuffer) void {
        for (self.buffer_pool.items, 0..) |buf, i| {
            if (buf == buffer) {
                _ = self.buffer_pool.orderedRemove(i);
                break;
            }
        }
        
        buffer.deinit(self.allocator);
        self.allocator.destroy(buffer);
    }
    
    pub fn setOverlay(self: *Self, plane_index: u8, buffer: ?*OverlayBuffer, src_rect: ?struct { x: u32, y: u32, w: u32, h: u32 }, dst_rect: ?struct { x: i32, y: i32, w: u32, h: u32 }) !void {
        if (plane_index >= self.overlay_planes.items.len) return OverlayError.OverlayPlaneNotFound;
        
        try self.overlay_planes.items[plane_index].setBuffer(buffer, src_rect, dst_rect);
    }
    
    pub fn updateCompositionOrder(self: *Self, z_orders: []const u8) !void {
        if (self.composition_order.len > 0) {
            self.allocator.free(self.composition_order);
        }
        
        self.composition_order = try self.allocator.dupe(u8, z_orders);
        
        // Sort planes by z-order for optimal composition
        std.sort.insertion(u8, self.composition_order, {}, struct {
            fn lessThan(_: void, a: u8, b: u8) bool {
                return a < b;
            }
        }.lessThan);
    }
    
    pub fn commitAllOverlays(self: *Self) !void {
        // Commit overlays in composition order
        if (self.composition_order.len > 0) {
            for (self.composition_order) |plane_index| {
                if (plane_index < self.overlay_planes.items.len) {
                    try self.overlay_planes.items[plane_index].commitOverlay();
                }
            }
        } else {
            // Commit all planes in creation order
            for (self.overlay_planes.items) |*plane| {
                try plane.commitOverlay();
            }
        }
    }
    
    pub fn getPlaneCapabilities(self: *const Self, plane_index: u8) ?struct { 
        formats: []OverlayFormat, 
        max_size: struct { w: u32, h: u32 },
        scaling: struct { min: f32, max: f32 }
    } {
        if (plane_index >= self.overlay_planes.items.len) return null;
        
        const plane = &self.overlay_planes.items[plane_index];
        return .{
            .formats = plane.supported_formats,
            .max_size = .{ .w = plane.max_width, .h = plane.max_height },
            .scaling = .{ .min = plane.min_scale, .max = plane.max_scale },
        };
    }
    
    pub fn optimizeForVideo(self: *Self, plane_index: u8) !void {
        if (plane_index >= self.overlay_planes.items.len) return;
        
        var plane = &self.overlay_planes.items[plane_index];
        
        // Optimize for video playback
        try plane.setColorSpace(.bt709); // HDTV color space
        try plane.setScalingMode(.bicubic); // High quality scaling
    }
    
    pub fn optimizeForUI(self: *Self, plane_index: u8) !void {
        if (plane_index >= self.overlay_planes.items.len) return;
        
        var plane = &self.overlay_planes.items[plane_index];
        
        // Optimize for UI elements
        try plane.setScalingMode(.nearest); // Pixel-perfect scaling
    }
};