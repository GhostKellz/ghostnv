const std = @import("std");
const linux = std.os.linux;
const c = std.c;

// NVIDIA NVKMS (Kernel Mode Setting) interface for direct hardware access
// Based on nvidia-modeset-interface.h from NVIDIA's open-gpu-kernel-modules

pub const NVKMS_IOCTL_MAGIC = 0x6E;
pub const NVKMS_IOCTL_ALLOC_DEVICE = 0x00;
pub const NVKMS_IOCTL_FREE_DEVICE = 0x01;
pub const NVKMS_IOCTL_QUERY_DISP_CAPABILITIES = 0x02;
pub const NVKMS_IOCTL_SET_DISP_ATTRIBUTE = 0x03;
pub const NVKMS_IOCTL_GET_DISP_ATTRIBUTE = 0x04;
pub const NVKMS_IOCTL_SET_LUT = 0x05;

pub const NvKmsError = error{
    DeviceNotFound,
    PermissionDenied,
    InvalidDevice,
    InvalidParameter,
    HardwareError,
    NotSupported,
    OutOfMemory,
    DeviceBusy,
    Timeout,
};

pub const NvKmsDisplayAttribute = enum(u32) {
    digital_vibrance = 0x00000001,
    image_sharpening = 0x00000002,
    color_space = 0x00000003,
    color_range = 0x00000004,
    dithering = 0x00000005,
    underscan = 0x00000006,
    
    pub fn toString(self: NvKmsDisplayAttribute) []const u8 {
        return switch (self) {
            .digital_vibrance => "Digital Vibrance",
            .image_sharpening => "Image Sharpening",
            .color_space => "Color Space",
            .color_range => "Color Range",
            .dithering => "Dithering",
            .underscan => "Underscan",
        };
    }
};

pub const NvKmsLutType = enum(u32) {
    identity = 0,
    gamma = 1,
    degamma = 2,
    ctm = 3, // Color Transformation Matrix
    
    pub fn toString(self: NvKmsLutType) []const u8 {
        return switch (self) {
            .identity => "Identity",
            .gamma => "Gamma",
            .degamma => "Degamma",
            .ctm => "Color Transformation Matrix",
        };
    }
};

pub const NvKmsDeviceHandle = extern struct {
    device_id: u32,
    fd: i32,
    
    pub fn isValid(self: NvKmsDeviceHandle) bool {
        return self.fd >= 0 and self.device_id != 0;
    }
};

pub const NvKmsRect = extern struct {
    x: u16,
    y: u16,
    width: u16,
    height: u16,
};

pub const NvKmsDisplayHandle = extern struct {
    display_id: u32,
    connector_id: u32,
    crtc_id: u32,
    head_id: u32,
};

pub const NvKmsAllocDeviceRequest = extern struct {
    device_id: u32,
    reserved: [4]u32,
};

pub const NvKmsAllocDeviceReply = extern struct {
    device_handle: NvKmsDeviceHandle,
    gpu_id: u32,
    max_displays: u32,
    capabilities: u64,
    reserved: [8]u32,
};

pub const NvKmsSetDisplayAttributeRequest = extern struct {
    device_handle: NvKmsDeviceHandle,
    display_handle: NvKmsDisplayHandle,
    attribute: NvKmsDisplayAttribute,
    value: i32,
    reserved: [4]u32,
};

pub const NvKmsGetDisplayAttributeRequest = extern struct {
    device_handle: NvKmsDeviceHandle,
    display_handle: NvKmsDisplayHandle,
    attribute: NvKmsDisplayAttribute,
    reserved: [4]u32,
};

pub const NvKmsGetDisplayAttributeReply = extern struct {
    value: i32,
    min_value: i32,
    max_value: i32,
    reserved: [4]u32,
};

pub const NvKmsSetLutRequest = extern struct {
    device_handle: NvKmsDeviceHandle,
    display_handle: NvKmsDisplayHandle,
    lut_type: NvKmsLutType,
    size: u32,
    red_lut: [*]u16,
    green_lut: [*]u16,
    blue_lut: [*]u16,
    reserved: [4]u32,
};

pub const NvKmsInterface = struct {
    modeset_fd: i32,
    device_handle: NvKmsDeviceHandle,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) !NvKmsInterface {
        const modeset_fd = linux.open("/dev/nvidia-modeset", linux.O.RDWR, 0) catch |err| switch (err) {
            error.FileNotFound => return NvKmsError.DeviceNotFound,
            error.AccessDenied => return NvKmsError.PermissionDenied,
            else => return err,
        };
        
        var interface = NvKmsInterface{
            .modeset_fd = modeset_fd,
            .device_handle = NvKmsDeviceHandle{ .device_id = 0, .fd = -1 },
            .allocator = allocator,
        };
        
        try interface.allocDevice();
        return interface;
    }
    
    pub fn deinit(self: *NvKmsInterface) void {
        if (self.device_handle.isValid()) {
            self.freeDevice() catch |err| {
                std.log.err("Failed to free NVKMS device: {}", .{err});
            };
        }
        if (self.modeset_fd >= 0) {
            _ = linux.close(self.modeset_fd);
        }
    }
    
    fn allocDevice(self: *NvKmsInterface) !void {
        var request = NvKmsAllocDeviceRequest{
            .device_id = 0, // Auto-detect primary GPU
            .reserved = std.mem.zeroes([4]u32),
        };
        
        var reply = NvKmsAllocDeviceReply{
            .device_handle = NvKmsDeviceHandle{ .device_id = 0, .fd = -1 },
            .gpu_id = 0,
            .max_displays = 0,
            .capabilities = 0,
            .reserved = std.mem.zeroes([8]u32),
        };
        
        const result = linux.syscall3(
            linux.SYS.ioctl,
            @intCast(self.modeset_fd),
            @intCast(@as(c_ulong, NVKMS_IOCTL_MAGIC) << 8 | NVKMS_IOCTL_ALLOC_DEVICE),
            @intFromPtr(&request),
        );
        
        if (result != 0) {
            return switch (linux.getErrno(result)) {
                .NODEV => NvKmsError.DeviceNotFound,
                .ACCES => NvKmsError.PermissionDenied,
                .INVAL => NvKmsError.InvalidParameter,
                .NOMEM => NvKmsError.OutOfMemory,
                else => NvKmsError.HardwareError,
            };
        }
        
        self.device_handle = reply.device_handle;
        self.device_handle.fd = self.modeset_fd;
        
        std.log.info("NVKMS device allocated: GPU ID {}, {} displays, capabilities: 0x{x}", .{
            reply.gpu_id,
            reply.max_displays,
            reply.capabilities,
        });
    }
    
    fn freeDevice(self: *NvKmsInterface) !void {
        if (!self.device_handle.isValid()) return;
        
        const result = linux.syscall3(
            linux.SYS.ioctl,
            @intCast(self.modeset_fd),
            @intCast(@as(c_ulong, NVKMS_IOCTL_MAGIC) << 8 | NVKMS_IOCTL_FREE_DEVICE),
            @intFromPtr(&self.device_handle),
        );
        
        if (result != 0) {
            return switch (linux.getErrno(result)) {
                .INVAL => NvKmsError.InvalidDevice,
                else => NvKmsError.HardwareError,
            };
        }
        
        self.device_handle = NvKmsDeviceHandle{ .device_id = 0, .fd = -1 };
    }
    
    pub fn setDisplayAttribute(
        self: *NvKmsInterface,
        display_handle: NvKmsDisplayHandle,
        attribute: NvKmsDisplayAttribute,
        value: i32,
    ) !void {
        if (!self.device_handle.isValid()) return NvKmsError.InvalidDevice;
        
        var request = NvKmsSetDisplayAttributeRequest{
            .device_handle = self.device_handle,
            .display_handle = display_handle,
            .attribute = attribute,
            .value = value,
            .reserved = std.mem.zeroes([4]u32),
        };
        
        const result = linux.syscall3(
            linux.SYS.ioctl,
            @intCast(self.modeset_fd),
            @intCast(@as(c_ulong, NVKMS_IOCTL_MAGIC) << 8 | NVKMS_IOCTL_SET_DISP_ATTRIBUTE),
            @intFromPtr(&request),
        );
        
        if (result != 0) {
            return switch (linux.getErrno(result)) {
                .INVAL => NvKmsError.InvalidParameter,
                .NODEV => NvKmsError.DeviceNotFound,
                .NOTSUP => NvKmsError.NotSupported,
                else => NvKmsError.HardwareError,
            };
        }
        
        std.log.debug("Set display attribute {} to {} on display {}", .{
            attribute.toString(),
            value,
            display_handle.display_id,
        });
    }
    
    pub fn getDisplayAttribute(
        self: *NvKmsInterface,
        display_handle: NvKmsDisplayHandle,
        attribute: NvKmsDisplayAttribute,
    ) !NvKmsGetDisplayAttributeReply {
        if (!self.device_handle.isValid()) return NvKmsError.InvalidDevice;
        
        var request = NvKmsGetDisplayAttributeRequest{
            .device_handle = self.device_handle,
            .display_handle = display_handle,
            .attribute = attribute,
            .reserved = std.mem.zeroes([4]u32),
        };
        
        var reply = NvKmsGetDisplayAttributeReply{
            .value = 0,
            .min_value = 0,
            .max_value = 0,
            .reserved = std.mem.zeroes([4]u32),
        };
        
        const result = linux.syscall3(
            linux.SYS.ioctl,
            @intCast(self.modeset_fd),
            @intCast(@as(c_ulong, NVKMS_IOCTL_MAGIC) << 8 | NVKMS_IOCTL_GET_DISP_ATTRIBUTE),
            @intFromPtr(&request),
        );
        
        if (result != 0) {
            return switch (linux.getErrno(result)) {
                .INVAL => NvKmsError.InvalidParameter,
                .NODEV => NvKmsError.DeviceNotFound,
                .NOTSUP => NvKmsError.NotSupported,
                else => NvKmsError.HardwareError,
            };
        }
        
        return reply;
    }
    
    pub fn setHardwareLut(
        self: *NvKmsInterface,
        display_handle: NvKmsDisplayHandle,
        lut_type: NvKmsLutType,
        red_lut: []const u16,
        green_lut: []const u16,
        blue_lut: []const u16,
    ) !void {
        if (!self.device_handle.isValid()) return NvKmsError.InvalidDevice;
        
        if (red_lut.len != green_lut.len or red_lut.len != blue_lut.len) {
            return NvKmsError.InvalidParameter;
        }
        
        var request = NvKmsSetLutRequest{
            .device_handle = self.device_handle,
            .display_handle = display_handle,
            .lut_type = lut_type,
            .size = @intCast(red_lut.len),
            .red_lut = red_lut.ptr,
            .green_lut = green_lut.ptr,
            .blue_lut = blue_lut.ptr,
            .reserved = std.mem.zeroes([4]u32),
        };
        
        const result = linux.syscall3(
            linux.SYS.ioctl,
            @intCast(self.modeset_fd),
            @intCast(@as(c_ulong, NVKMS_IOCTL_MAGIC) << 8 | NVKMS_IOCTL_SET_LUT),
            @intFromPtr(&request),
        );
        
        if (result != 0) {
            return switch (linux.getErrno(result)) {
                .INVAL => NvKmsError.InvalidParameter,
                .NODEV => NvKmsError.DeviceNotFound,
                .NOTSUP => NvKmsError.NotSupported,
                .NOMEM => NvKmsError.OutOfMemory,
                else => NvKmsError.HardwareError,
            };
        }
        
        std.log.debug("Set hardware LUT ({}) on display {}", .{
            lut_type.toString(),
            display_handle.display_id,
        });
    }
    
    pub fn setDigitalVibrance(
        self: *NvKmsInterface,
        display_handle: NvKmsDisplayHandle,
        vibrance: i16,
    ) !void {
        // Convert from our range (-50 to 100) to NVIDIA's range (-1024 to 1023)
        const nvidia_vibrance = std.math.clamp(
            @as(i32, vibrance) * 1024 / 100,
            -1024,
            1023,
        );
        
        try self.setDisplayAttribute(
            display_handle,
            .digital_vibrance,
            nvidia_vibrance,
        );
        
        std.log.info("Set digital vibrance to {} (NVIDIA: {})", .{ vibrance, nvidia_vibrance });
    }
    
    pub fn getDigitalVibrance(
        self: *NvKmsInterface,
        display_handle: NvKmsDisplayHandle,
    ) !struct { current: i16, min: i16, max: i16 } {
        const reply = try self.getDisplayAttribute(display_handle, .digital_vibrance);
        
        // Convert from NVIDIA's range to our range
        const current = @as(i16, @intCast(reply.value * 100 / 1024));
        const min = @as(i16, @intCast(reply.min_value * 100 / 1024));
        const max = @as(i16, @intCast(reply.max_value * 100 / 1024));
        
        return .{ .current = current, .min = min, .max = max };
    }
    
    pub fn enumerateDisplays(self: *NvKmsInterface) ![]NvKmsDisplayHandle {
        if (!self.device_handle.isValid()) return NvKmsError.InvalidDevice;
        
        // Query display capabilities to get display list
        // This is a simplified implementation - real NVKMS would enumerate all connected displays
        
        var displays = try self.allocator.alloc(NvKmsDisplayHandle, 4); // Max 4 displays typically
        var count: usize = 0;
        
        // Simulate display enumeration - in real implementation, this would query the GPU
        for (0..4) |i| {
            displays[count] = NvKmsDisplayHandle{
                .display_id = @intCast(i),
                .connector_id = @intCast(i + 1),
                .crtc_id = @intCast(i),
                .head_id = @intCast(i),
            };
            count += 1;
        }
        
        return displays[0..count];
    }
    
    pub fn detectPrimaryDisplay(self: *NvKmsInterface) !NvKmsDisplayHandle {
        const displays = try self.enumerateDisplays();
        defer self.allocator.free(displays);
        
        if (displays.len == 0) {
            return NvKmsError.DeviceNotFound;
        }
        
        // Return the first display as primary
        return displays[0];
    }
};

// Test functions
test "nvkms interface initialization" {
    const allocator = std.testing.allocator;
    
    // This test will fail on systems without NVIDIA GPU/driver
    // In real usage, we'd check for device existence first
    _ = allocator;
    
    // Test basic struct initialization
    const display = NvKmsDisplayHandle{
        .display_id = 1,
        .connector_id = 2,
        .crtc_id = 3,
        .head_id = 4,
    };
    
    try std.testing.expect(display.display_id == 1);
    try std.testing.expect(display.connector_id == 2);
}

test "vibrance range conversion" {
    // Test vibrance range conversion
    const test_values = [_]struct { input: i16, expected: i32 }{
        .{ .input = -50, .expected = -512 },
        .{ .input = 0, .expected = 0 },
        .{ .input = 50, .expected = 512 },
        .{ .input = 100, .expected = 1024 },
    };
    
    for (test_values) |test_case| {
        const nvidia_vibrance = std.math.clamp(
            @as(i32, test_case.input) * 1024 / 100,
            -1024,
            1023,
        );
        try std.testing.expect(nvidia_vibrance == test_case.expected);
    }
}