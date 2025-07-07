const std = @import("std");
const linux = std.os.linux;
const fs = std.fs;
const Allocator = std.mem.Allocator;

// PCI Hardware Abstraction Layer for NVIDIA GPU detection and management

pub const PciError = error{
    DeviceNotFound,
    AccessDenied,
    InvalidDevice,
    ParseError,
    OutOfMemory,
    IoError,
    InvalidConfig,
    ResourceAllocationFailed,
};

pub const NVIDIA_VENDOR_ID: u16 = 0x10DE;

pub const PciDeviceId = struct {
    vendor_id: u16,
    device_id: u16,
    subsystem_vendor_id: u16,
    subsystem_device_id: u16,
    
    pub fn isNvidia(self: PciDeviceId) bool {
        return self.vendor_id == NVIDIA_VENDOR_ID;
    }
    
    pub fn toString(self: PciDeviceId, allocator: Allocator) ![]u8 {
        return std.fmt.allocPrint(allocator, "{x:0>4}:{x:0>4}:{x:0>4}:{x:0>4}", .{
            self.vendor_id,
            self.device_id,
            self.subsystem_vendor_id,
            self.subsystem_device_id,
        });
    }
};

pub const PciDeviceClass = enum(u8) {
    display_controller = 0x03,
    multimedia_controller = 0x04,
    bridge_device = 0x06,
    processing_accelerator = 0x12,
    
    pub fn toString(self: PciDeviceClass) []const u8 {
        return switch (self) {
            .display_controller => "Display Controller",
            .multimedia_controller => "Multimedia Controller",
            .bridge_device => "Bridge Device",
            .processing_accelerator => "Processing Accelerator",
        };
    }
};

pub const NvidiaArchitecture = enum {
    unknown,
    kepler,    // GTX 600/700 series
    maxwell,   // GTX 900 series
    pascal,    // GTX 10 series
    volta,     // Titan V
    turing,    // RTX 20 series
    ampere,    // RTX 30 series
    ada,       // RTX 40 series
    hopper,    // H100 series
    blackwell, // RTX 50 series
    
    pub fn toString(self: NvidiaArchitecture) []const u8 {
        return switch (self) {
            .unknown => "Unknown",
            .kepler => "Kepler",
            .maxwell => "Maxwell",
            .pascal => "Pascal",
            .volta => "Volta",
            .turing => "Turing",
            .ampere => "Ampere",
            .ada => "Ada Lovelace",
            .hopper => "Hopper",
            .blackwell => "Blackwell",
        };
    }
};

pub const PciDevice = struct {
    bus: u8,
    slot: u8,
    function: u8,
    device_id: PciDeviceId,
    class_code: u8,
    subclass_code: u8,
    prog_if: u8,
    revision_id: u8,
    bar0: u64,
    bar1: u64,
    bar2: u64,
    bar3: u64,
    bar4: u64,
    bar5: u64,
    irq: u8,
    path: []const u8,
    
    // NVIDIA-specific fields
    architecture: NvidiaArchitecture,
    memory_size: u64,
    compute_capability: struct { major: u8, minor: u8 },
    
    pub fn init(allocator: Allocator, bus: u8, slot: u8, function: u8, path: []const u8) !PciDevice {
        var device = PciDevice{
            .bus = bus,
            .slot = slot,
            .function = function,
            .device_id = PciDeviceId{
                .vendor_id = 0,
                .device_id = 0,
                .subsystem_vendor_id = 0,
                .subsystem_device_id = 0,
            },
            .class_code = 0,
            .subclass_code = 0,
            .prog_if = 0,
            .revision_id = 0,
            .bar0 = 0,
            .bar1 = 0,
            .bar2 = 0,
            .bar3 = 0,
            .bar4 = 0,
            .bar5 = 0,
            .irq = 0,
            .path = try allocator.dupe(u8, path),
            .architecture = .unknown,
            .memory_size = 0,
            .compute_capability = .{ .major = 0, .minor = 0 },
        };
        
        try device.readPciConfig(allocator);
        return device;
    }
    
    pub fn deinit(self: *PciDevice, allocator: Allocator) void {
        allocator.free(self.path);
    }
    
    fn readPciConfig(self: *PciDevice, allocator: Allocator) !void {
        const config_path = try std.fmt.allocPrint(allocator, "{s}/config", .{self.path});
        defer allocator.free(config_path);
        
        const config_file = fs.openFileAbsolute(config_path, .{}) catch |err| switch (err) {
            error.FileNotFound => return PciError.DeviceNotFound,
            error.AccessDenied => return PciError.AccessDenied,
            else => return err,
        };
        defer config_file.close();
        
        var config_data: [256]u8 = undefined;
        const bytes_read = try config_file.readAll(&config_data);
        if (bytes_read < 64) return PciError.InvalidDevice;
        
        // Parse PCI configuration header
        self.device_id.vendor_id = std.mem.readInt(u16, config_data[0..2], .little);
        self.device_id.device_id = std.mem.readInt(u16, config_data[2..4], .little);
        self.revision_id = config_data[8];
        self.prog_if = config_data[9];
        self.subclass_code = config_data[10];
        self.class_code = config_data[11];
        
        // Read BARs
        self.bar0 = std.mem.readInt(u32, config_data[16..20], .little);
        self.bar1 = std.mem.readInt(u32, config_data[20..24], .little);
        self.bar2 = std.mem.readInt(u32, config_data[24..28], .little);
        self.bar3 = std.mem.readInt(u32, config_data[28..32], .little);
        self.bar4 = std.mem.readInt(u32, config_data[32..36], .little);
        self.bar5 = std.mem.readInt(u32, config_data[36..40], .little);
        
        // Read subsystem IDs
        if (bytes_read >= 48) {
            self.device_id.subsystem_vendor_id = std.mem.readInt(u16, config_data[44..46], .little);
            self.device_id.subsystem_device_id = std.mem.readInt(u16, config_data[46..48], .little);
        }
        
        // Determine NVIDIA architecture if this is an NVIDIA device
        if (self.device_id.isNvidia()) {
            self.architecture = self.detectArchitecture();
            try self.readNvidiaSpecificInfo(allocator);
        }
    }
    
    fn detectArchitecture(self: *PciDevice) NvidiaArchitecture {
        const device_id = self.device_id.device_id;
        
        // Device ID ranges for different architectures
        // This is a simplified mapping - real implementation would be more comprehensive
        return switch (device_id) {
            0x1000...0x12FF => .kepler,
            0x1300...0x13FF => .maxwell,
            0x1B00...0x1BFF => .pascal,
            0x1C00...0x1CFF => .pascal,
            0x1D00...0x1DFF => .volta,
            0x1E00...0x1FFF => .turing,
            0x2000...0x25FF => .ampere,
            0x2600...0x26FF => .ada,
            0x2700...0x27FF => .hopper,
            0x2800...0x28FF => .blackwell,
            else => .unknown,
        };
    }
    
    fn readNvidiaSpecificInfo(self: *PciDevice, allocator: Allocator) !void {
        // Read NVIDIA-specific information
        
        // Try to read memory size from resource0
        const resource_path = try std.fmt.allocPrint(allocator, "{s}/resource0", .{self.path});
        defer allocator.free(resource_path);
        
        if (fs.openFileAbsolute(resource_path, .{})) |resource_file| {
            defer resource_file.close();
            const stat = try resource_file.stat();
            self.memory_size = stat.size;
        } else |_| {
            // Fall back to BAR0 size estimation
            self.memory_size = 0;
        }
        
        // Set compute capability based on architecture
        self.compute_capability = switch (self.architecture) {
            .kepler => .{ .major = 3, .minor = 5 },
            .maxwell => .{ .major = 5, .minor = 0 },
            .pascal => .{ .major = 6, .minor = 0 },
            .volta => .{ .major = 7, .minor = 0 },
            .turing => .{ .major = 7, .minor = 5 },
            .ampere => .{ .major = 8, .minor = 0 },
            .ada => .{ .major = 8, .minor = 9 },
            .hopper => .{ .major = 9, .minor = 0 },
            .blackwell => .{ .major = 10, .minor = 0 },
            else => .{ .major = 0, .minor = 0 },
        };
    }
    
    pub fn isDisplayController(self: PciDevice) bool {
        return self.class_code == @intFromEnum(PciDeviceClass.display_controller);
    }
    
    pub fn isComputeAccelerator(self: PciDevice) bool {
        return self.class_code == @intFromEnum(PciDeviceClass.processing_accelerator);
    }
    
    pub fn getDeviceName(self: PciDevice, allocator: Allocator) ![]const u8 {
        if (!self.device_id.isNvidia()) {
            return try std.fmt.allocPrint(allocator, "Unknown Device {x:0>4}:{x:0>4}", .{
                self.device_id.vendor_id,
                self.device_id.device_id,
            });
        }
        
        // NVIDIA device name mapping (simplified)
        const device_id = self.device_id.device_id;
        return switch (device_id) {
            0x2684 => try allocator.dupe(u8, "GeForce RTX 4090"),
            0x2782 => try allocator.dupe(u8, "GeForce RTX 4080"),
            0x2783 => try allocator.dupe(u8, "GeForce RTX 4070"),
            0x2786 => try allocator.dupe(u8, "GeForce RTX 4060"),
            0x2204 => try allocator.dupe(u8, "GeForce RTX 3090"),
            0x2206 => try allocator.dupe(u8, "GeForce RTX 3080"),
            0x2484 => try allocator.dupe(u8, "GeForce RTX 3070"),
            0x2487 => try allocator.dupe(u8, "GeForce RTX 3060"),
            else => try std.fmt.allocPrint(allocator, "NVIDIA GPU {x:0>4}", .{device_id}),
        };
    }
    
    pub fn getBusAddress(self: PciDevice) [12]u8 {
        var addr: [12]u8 = undefined;
        _ = std.fmt.bufPrint(&addr, "{x:0>4}:{x:0>2}:{x:0>2}.{x}", .{
            0x0000, // Domain (usually 0000)
            self.bus,
            self.slot,
            self.function,
        }) catch unreachable;
        return addr;
    }
    
    pub fn enable_bus_master(self: *PciDevice, allocator: Allocator) !void {
        const cmd_path = try std.fmt.allocPrint(allocator, "{s}/config", .{self.path});
        defer allocator.free(cmd_path);
        
        const config_file = fs.openFileAbsolute(cmd_path, .{ .mode = .read_write }) catch |err| switch (err) {
            error.FileNotFound => return PciError.DeviceNotFound,
            error.AccessDenied => return PciError.AccessDenied,
            else => return err,
        };
        defer config_file.close();
        
        // Read command register (offset 0x04)
        try config_file.seekTo(0x04);
        var cmd_data: [2]u8 = undefined;
        _ = try config_file.readAll(&cmd_data);
        
        var cmd = std.mem.readInt(u16, &cmd_data, .little);
        
        // Set bus master enable bit (bit 2)
        cmd |= 0x0004;
        
        // Write back
        try config_file.seekTo(0x04);
        std.mem.writeInt(u16, &cmd_data, cmd, .little);
        _ = try config_file.writeAll(&cmd_data);
    }
    
    pub fn get_bar_size(self: *PciDevice, allocator: Allocator, bar_num: u8) !u64 {
        if (bar_num > 5) return PciError.InvalidDevice;
        
        const resource_path = try std.fmt.allocPrint(allocator, "{s}/resource{}", .{ self.path, bar_num });
        defer allocator.free(resource_path);
        
        if (fs.openFileAbsolute(resource_path, .{})) |resource_file| {
            defer resource_file.close();
            const stat = try resource_file.stat();
            return @intCast(stat.size);
        } else |_| {
            return 0;
        }
    }
};

pub const PciEnumerator = struct {
    allocator: Allocator,
    devices: std.ArrayList(PciDevice),
    
    pub fn init(allocator: Allocator) PciEnumerator {
        return PciEnumerator{
            .allocator = allocator,
            .devices = std.ArrayList(PciDevice).init(allocator),
        };
    }
    
    pub fn deinit(self: *PciEnumerator) void {
        for (self.devices.items) |*device| {
            device.deinit(self.allocator);
        }
        self.devices.deinit();
    }
    
    pub fn scanPciDevices(self: *PciEnumerator) !void {
        var pci_dir = fs.openDirAbsolute("/sys/bus/pci/devices", .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => return PciError.DeviceNotFound,
            error.AccessDenied => return PciError.AccessDenied,
            else => return err,
        };
        defer pci_dir.close();
        
        var iterator = pci_dir.iterate();
        while (try iterator.next()) |entry| {
            if (entry.kind != .directory) continue;
            
            // Parse PCI address from directory name (format: 0000:xx:xx.x)
            if (entry.name.len != 12) continue;
            
            const bus = std.fmt.parseInt(u8, entry.name[5..7], 16) catch continue;
            const slot = std.fmt.parseInt(u8, entry.name[8..10], 16) catch continue;
            const function = std.fmt.parseInt(u8, entry.name[11..12], 16) catch continue;
            
            const device_path = try std.fmt.allocPrint(self.allocator, "/sys/bus/pci/devices/{s}", .{entry.name});
            defer self.allocator.free(device_path);
            
            if (PciDevice.init(self.allocator, bus, slot, function, device_path)) |device| {
                try self.devices.append(device);
            } else |err| {
                std.log.debug("Failed to read PCI device {s}: {}", .{ entry.name, err });
            }
        }
        
        std.log.info("Scanned {} PCI devices", .{self.devices.items.len});
    }
    
    pub fn findNvidiaGpus(self: *PciEnumerator) ![]PciDevice {
        var nvidia_devices = std.ArrayList(PciDevice).init(self.allocator);
        defer nvidia_devices.deinit();
        
        for (self.devices.items) |device| {
            if (device.device_id.isNvidia() and device.isDisplayController()) {
                try nvidia_devices.append(device);
            }
        }
        
        return nvidia_devices.toOwnedSlice();
    }
    
    pub fn findPrimaryGpu(self: *PciEnumerator) !?PciDevice {
        // Find the primary GPU (first NVIDIA display controller)
        for (self.devices.items) |device| {
            if (device.device_id.isNvidia() and device.isDisplayController()) {
                return device;
            }
        }
        return null;
    }
    
    pub fn printDeviceInfo(self: *PciEnumerator) !void {
        std.log.info("=== PCI Device Information ===");
        
        for (self.devices.items) |device| {
            if (device.device_id.isNvidia()) {
                const name = try device.getDeviceName(self.allocator);
                defer self.allocator.free(name);
                
                const bus_addr = device.getBusAddress();
                
                std.log.info("NVIDIA Device: {s}", .{name});
                std.log.info("  Bus Address: {s}", .{bus_addr});
                std.log.info("  Architecture: {s}", .{device.architecture.toString()});
                std.log.info("  Device ID: {x:0>4}:{x:0>4}", .{ device.device_id.vendor_id, device.device_id.device_id });
                std.log.info("  Memory Size: {} MB", .{device.memory_size / (1024 * 1024)});
                std.log.info("  Compute Capability: {}.{}", .{ device.compute_capability.major, device.compute_capability.minor });
                std.log.info("  Class: {s}", .{(@as(PciDeviceClass, @enumFromInt(device.class_code))).toString()});
                std.log.info("");
            }
        }
    }
    
    // Legacy compatibility functions
    pub fn enumerate_nvidia_devices(self: *PciEnumerator) !u32 {
        try self.scanPciDevices();
        var count: u32 = 0;
        for (self.devices.items) |device| {
            if (device.device_id.isNvidia()) {
                count += 1;
            }
        }
        return count;
    }
    
    pub fn get_device_by_index(self: *PciEnumerator, index: u32) !PciDevice {
        var current_index: u32 = 0;
        for (self.devices.items) |device| {
            if (device.device_id.isNvidia()) {
                if (current_index == index) {
                    return device;
                }
                current_index += 1;
            }
        }
        return PciError.DeviceNotFound;
    }
};

// Test functions
test "pci device enumeration" {
    const allocator = std.testing.allocator;
    
    var enumerator = PciEnumerator.init(allocator);
    defer enumerator.deinit();
    
    // Test basic initialization
    try std.testing.expect(enumerator.devices.items.len == 0);
    
    // Test device ID creation
    const device_id = PciDeviceId{
        .vendor_id = 0x10DE,
        .device_id = 0x2684,
        .subsystem_vendor_id = 0x1458,
        .subsystem_device_id = 0x4090,
    };
    
    try std.testing.expect(device_id.isNvidia());
    
    const id_str = try device_id.toString(allocator);
    defer allocator.free(id_str);
    try std.testing.expectEqualStrings("10de:2684:1458:4090", id_str);
}

test "nvidia architecture detection" {
    const test_cases = [_]struct { device_id: u16, expected: NvidiaArchitecture }{
        .{ .device_id = 0x1000, .expected = .kepler },
        .{ .device_id = 0x1380, .expected = .maxwell },
        .{ .device_id = 0x1B80, .expected = .pascal },
        .{ .device_id = 0x1E00, .expected = .turing },
        .{ .device_id = 0x2204, .expected = .ampere },
        .{ .device_id = 0x2684, .expected = .ada },
    };
    
    for (test_cases) |case| {
        var device = PciDevice{
            .bus = 0,
            .slot = 0,
            .function = 0,
            .device_id = PciDeviceId{
                .vendor_id = 0x10DE,
                .device_id = case.device_id,
                .subsystem_vendor_id = 0,
                .subsystem_device_id = 0,
            },
            .class_code = 0,
            .subclass_code = 0,
            .prog_if = 0,
            .revision_id = 0,
            .bar0 = 0,
            .bar1 = 0,
            .bar2 = 0,
            .bar3 = 0,
            .bar4 = 0,
            .bar5 = 0,
            .irq = 0,
            .path = "",
            .architecture = .unknown,
            .memory_size = 0,
            .compute_capability = .{ .major = 0, .minor = 0 },
        };
        
        const detected = device.detectArchitecture();
        try std.testing.expect(detected == case.expected);
    }
}

// Legacy compatibility functions for existing code
pub fn enumerate_nvidia_devices() !u32 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    var enumerator = PciEnumerator.init(allocator);
    defer enumerator.deinit();
    
    return enumerator.enumerate_nvidia_devices();
}

pub fn get_device_by_index(index: u32) !PciDevice {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    
    var enumerator = PciEnumerator.init(allocator);
    defer enumerator.deinit();
    
    return enumerator.get_device_by_index(index);
}