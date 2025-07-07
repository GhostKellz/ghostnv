const std = @import("std");
const linux = std.os.linux;
const fs = std.fs;
const Allocator = std.mem.Allocator;
const print = std.debug.print;
const pci = @import("../hal/pci.zig");

// Import gaming performance module for VRR types
const VrrMode = enum(u8) {
    disabled = 0,
    adaptive_sync = 1,
    gsync_compatible = 2,
    gsync_ultimate = 3,
    freesync = 4,
    freesync_premium = 5,
};

pub const VrrCapabilities = struct {
    supports_adaptive_sync: bool,
    supports_gsync: bool,
    supports_freesync: bool,
    min_refresh_rate: u32,
    max_refresh_rate: u32,
    connected_vrr_displays: u32,
};

pub const DrmError = error{
    RegistrationFailed,
    InitializationFailed,
    InvalidMode,
    ResourceBusy,
    NotSupported,
    VrrNotSupported,
    InvalidRefreshRate,
    DisplayNotConnected,
    DeviceNotFound,
    AccessDenied,
    OutOfMemory,
    InvalidDevice,
    HardwareError,
};

pub const DrmConnectorType = enum {
    Unknown,
    VGA,
    DVID,
    DVII,
    DVIA,
    Composite,
    SVIDEO,
    LVDS,
    Component,
    NinePinDIN,
    DisplayPort,
    HDMIA,
    HDMIB,
    TV,
    eDP,
    VIRTUAL,
    DSI,
    DPI,
    WRITEBACK,
    SPI,
    USB,
};

pub const DrmMode = struct {
    hdisplay: u16,
    vdisplay: u16,
    refresh_rate: u32,
    flags: u32,
    name: [32]u8,
    
    pub fn init(width: u16, height: u16, refresh: u32) DrmMode {
        var mode = DrmMode{
            .hdisplay = width,
            .vdisplay = height,
            .refresh_rate = refresh,
            .flags = 0,
            .name = [_]u8{0} ** 32,
        };
        
        _ = std.fmt.bufPrint(&mode.name, "{}-{}-{}", .{width, height, refresh}) catch {};
        return mode;
    }
};

pub const DrmConnector = struct {
    id: u32,
    connector_type: DrmConnectorType,
    connected: bool,
    modes: std.ArrayList(DrmMode),
    current_mode: ?DrmMode,
    supports_vrr: bool,
    vrr_enabled: bool,
    vrr_min_refresh: u32,
    vrr_max_refresh: u32,
    
    pub fn init(allocator: std.mem.Allocator, id: u32, connector_type: DrmConnectorType) DrmConnector {
        const supports_vrr = switch (connector_type) {
            .DisplayPort, .HDMIA, .HDMIB, .eDP => true,
            else => false,
        };
        
        return DrmConnector{
            .id = id,
            .connector_type = connector_type,
            .connected = false,
            .modes = std.ArrayList(DrmMode).init(allocator),
            .current_mode = null,
            .supports_vrr = supports_vrr,
            .vrr_enabled = false,
            .vrr_min_refresh = 48,
            .vrr_max_refresh = 165,
        };
    }
    
    pub fn deinit(self: *DrmConnector) void {
        self.modes.deinit();
    }
    
    pub fn detect(self: *DrmConnector) !bool {
        // Simulate connector detection
        // In real implementation, this would probe the display
        print("nvzig: Detecting connector {} (type: {})\n", .{self.id, self.connector_type});
        
        switch (self.connector_type) {
            .DisplayPort, .HDMIA, .HDMIB => {
                self.connected = true;
                try self.add_standard_modes();
            },
            else => {
                self.connected = false;
            }
        }
        
        return self.connected;
    }
    
    fn add_standard_modes(self: *DrmConnector) !void {
        // Add common display modes
        const standard_modes = [_]struct { u16, u16, u32 }{
            .{ 1920, 1080, 60 },
            .{ 2560, 1440, 60 },
            .{ 3840, 2160, 60 },
            .{ 1680, 1050, 60 },
            .{ 1280, 1024, 60 },
            .{ 1024, 768, 60 },
        };
        
        for (standard_modes) |mode_info| {
            const mode = DrmMode.init(mode_info[0], mode_info[1], mode_info[2]);
            try self.modes.append(mode);
        }
        
        print("nvzig: Added {} modes to connector {}\n", .{self.modes.items.len, self.id});
    }
    
    pub fn set_mode(self: *DrmConnector, mode: DrmMode) !void {
        // Validate mode is supported
        var found = false;
        for (self.modes.items) |supported_mode| {
            if (supported_mode.hdisplay == mode.hdisplay and 
                supported_mode.vdisplay == mode.vdisplay and
                supported_mode.refresh_rate == mode.refresh_rate) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            return DrmError.InvalidMode;
        }
        
        self.current_mode = mode;
        print("nvzig: Set mode {}x{} @ {}Hz on connector {}\n", 
              .{mode.hdisplay, mode.vdisplay, mode.refresh_rate, self.id});
    }
    
    pub fn enable_vrr(self: *DrmConnector, min_refresh: u32, max_refresh: u32) !void {
        if (!self.supports_vrr) {
            return DrmError.VrrNotSupported;
        }
        
        if (min_refresh >= max_refresh or min_refresh < 24 or max_refresh > 500) {
            return DrmError.InvalidRefreshRate;
        }
        
        self.vrr_enabled = true;
        self.vrr_min_refresh = min_refresh;
        self.vrr_max_refresh = max_refresh;
        
        print("nvzig: VRR enabled on connector {}: {}-{}Hz\n", 
              .{self.id, min_refresh, max_refresh});
    }
    
    pub fn disable_vrr(self: *DrmConnector) void {
        self.vrr_enabled = false;
        print("nvzig: VRR disabled on connector {}\n", .{self.id});
    }
    
    pub fn set_refresh_rate(self: *DrmConnector, refresh_rate: u32) !void {
        if (!self.vrr_enabled) {
            return DrmError.VrrNotSupported;
        }
        
        if (refresh_rate < self.vrr_min_refresh or refresh_rate > self.vrr_max_refresh) {
            return DrmError.InvalidRefreshRate;
        }
        
        if (self.current_mode) |*mode| {
            mode.refresh_rate = refresh_rate;
            print("nvzig: VRR refresh rate set to {}Hz on connector {}\n", 
                  .{refresh_rate, self.id});
        }
    }
};

pub const DrmCrtc = struct {
    id: u32,
    active: bool,
    mode: ?DrmMode,
    x: u32,
    y: u32,
    
    pub fn init(id: u32) DrmCrtc {
        return DrmCrtc{
            .id = id,
            .active = false,
            .mode = null,
            .x = 0,
            .y = 0,
        };
    }
    
    pub fn set_config(self: *DrmCrtc, mode: DrmMode, x: u32, y: u32) !void {
        self.mode = mode;
        self.x = x;
        self.y = y;
        self.active = true;
        
        print("nvzig: CRTC {} configured: {}x{} @ {}Hz, offset ({}, {})\n",
              .{self.id, mode.hdisplay, mode.vdisplay, mode.refresh_rate, x, y});
    }
    
    pub fn disable(self: *DrmCrtc) void {
        self.active = false;
        self.mode = null;
        print("nvzig: CRTC {} disabled\n", .{self.id});
    }
};

pub const DrmFramebuffer = struct {
    id: u32,
    width: u32,
    height: u32,
    format: u32,
    pitch: u32,
    size: usize,
    physical_address: usize,
    virtual_address: ?usize,
    
    pub fn init(id: u32, width: u32, height: u32, format: u32) DrmFramebuffer {
        const bpp = 4; // Assume 32-bit RGBA for now
        const pitch = width * bpp;
        const size = pitch * height;
        
        return DrmFramebuffer{
            .id = id,
            .width = width,
            .height = height,
            .format = format,
            .pitch = pitch,
            .size = size,
            .physical_address = 0,
            .virtual_address = null,
        };
    }
    
    pub fn allocate_memory(self: *DrmFramebuffer, phys_addr: usize) !void {
        self.physical_address = phys_addr;
        // In real implementation, would map framebuffer memory
        print("nvzig: Allocated framebuffer {} memory: 0x{X} ({}x{}, {} bytes)\n",
              .{self.id, phys_addr, self.width, self.height, self.size});
    }
    
    pub fn map_virtual(self: *DrmFramebuffer) !void {
        if (self.physical_address == 0) return DrmError.InitializationFailed;
        
        // In real implementation, would use ioremap or similar
        self.virtual_address = self.physical_address;
        print("nvzig: Mapped framebuffer {} virtual address: 0x{X}\n", 
              .{self.id, self.virtual_address.?});
    }
};

pub const DrmDevice = struct {
    dev_node: []const u8,     // e.g., "/dev/dri/card0"
    sysfs_path: []const u8,   // e.g., "/sys/class/drm/card0"
    major: u32,
    minor: u32,
    
    pub fn init(allocator: Allocator, card_index: u32) !DrmDevice {
        const dev_node = try std.fmt.allocPrint(allocator, "/dev/dri/card{}", .{card_index});
        const sysfs_path = try std.fmt.allocPrint(allocator, "/sys/class/drm/card{}", .{card_index});
        
        return DrmDevice{
            .dev_node = dev_node,
            .sysfs_path = sysfs_path,
            .major = 226, // DRM major device number
            .minor = card_index,
        };
    }
    
    pub fn deinit(self: *DrmDevice, allocator: Allocator) void {
        allocator.free(self.dev_node);
        allocator.free(self.sysfs_path);
    }
    
    pub fn exists(self: *DrmDevice) bool {
        return fs.accessAbsolute(self.dev_node, .{}) catch false;
    }
};

pub const DrmDriver = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    major_version: u32,
    minor_version: u32,
    connectors: std.ArrayList(DrmConnector),
    crtcs: std.ArrayList(DrmCrtc),
    framebuffers: std.ArrayList(DrmFramebuffer),
    next_id: u32,
    registered: bool,
    
    // Hardware integration
    pci_device: ?pci.PciDevice,
    drm_device: ?DrmDevice,
    
    // Memory management
    vram_size: u64,
    vram_used: u64,
    system_memory_used: u64,
    
    pub fn init(allocator: std.mem.Allocator) !DrmDriver {
        return DrmDriver{
            .allocator = allocator,
            .name = "ghostnv",
            .major_version = 1,
            .minor_version = 0,
            .connectors = std.ArrayList(DrmConnector).init(allocator),
            .crtcs = std.ArrayList(DrmCrtc).init(allocator),
            .framebuffers = std.ArrayList(DrmFramebuffer).init(allocator),
            .next_id = 1,
            .registered = false,
            .pci_device = null,
            .drm_device = null,
            .vram_size = 0,
            .vram_used = 0,
            .system_memory_used = 0,
        };
    }
    
    pub fn initWithPciDevice(allocator: std.mem.Allocator, pci_dev: pci.PciDevice) !DrmDriver {
        var driver = try DrmDriver.init(allocator);
        driver.pci_device = pci_dev;
        driver.vram_size = pci_dev.memory_size;
        return driver;
    }
    
    pub fn deinit(self: *DrmDriver) void {
        if (self.drm_device) |*device| {
            device.deinit(self.allocator);
        }
        
        if (self.pci_device) |*device| {
            device.deinit(self.allocator);
        }
        
        for (self.connectors.items) |*connector| {
            connector.deinit();
        }
        self.connectors.deinit();
        self.crtcs.deinit();
        self.framebuffers.deinit();
    }
    
    pub fn register(self: *DrmDriver) !void {
        if (self.registered) return;
        
        print("GhostNV: Registering DRM driver '{}' v{}.{}\n", 
              .{self.name, self.major_version, self.minor_version});
        
        // Find available DRM card slot
        const card_index = try self.findAvailableDrmCard();
        self.drm_device = try DrmDevice.init(self.allocator, card_index);
        
        // Initialize PCI device if not already done
        if (self.pci_device == null) {
            try self.detectPciDevice();
        }
        
        // Initialize hardware resources
        try self.init_hardware();
        
        // Create DRM device node
        try self.createDrmDeviceNode();
        
        // Create connectors and CRTCs
        try self.create_connectors();
        try self.create_crtcs();
        
        // Register with DRM subsystem
        try self.registerWithDrmSubsystem();
        
        self.registered = true;
        print("GhostNV: DRM driver registration complete on {s}\n", .{self.drm_device.?.dev_node});
    }
    
    fn findAvailableDrmCard(self: *DrmDriver) !u32 {
        // Find the first available DRM card slot
        for (0..16) |i| {
            const card_index = @as(u32, @intCast(i));
            var test_device = try DrmDevice.init(self.allocator, card_index);
            defer test_device.deinit(self.allocator);
            
            if (!test_device.exists()) {
                return card_index;
            }
        }
        return DrmError.ResourceBusy;
    }
    
    fn detectPciDevice(self: *DrmDriver) !void {
        var enumerator = pci.PciEnumerator.init(self.allocator);
        defer enumerator.deinit();
        
        try enumerator.scanPciDevices();
        
        if (try enumerator.findPrimaryGpu()) |primary_gpu| {
            self.pci_device = primary_gpu;
            self.vram_size = primary_gpu.memory_size;
            
            const name = try primary_gpu.getDeviceName(self.allocator);
            defer self.allocator.free(name);
            
            print("GhostNV: Detected primary GPU: {s} ({s})\n", .{
                name,
                primary_gpu.architecture.toString(),
            });
        } else {
            return DrmError.DeviceNotFound;
        }
    }
    
    fn createDrmDeviceNode(self: *DrmDriver) !void {
        if (self.drm_device == null) return DrmError.InitializationFailed;
        
        const device = &self.drm_device.?;
        
        // In real kernel implementation, this would:
        // 1. Call device_create() to create sysfs entry
        // 2. Register character device with DRM major number
        // 3. Create /dev/dri/cardN device node
        
        print("GhostNV: Creating DRM device node {s}\n", .{device.dev_node});
        
        // Simulate device node creation
        // In real implementation: mknod(device.dev_node, S_IFCHR | 0666, makedev(device.major, device.minor))
    }
    
    fn registerWithDrmSubsystem(self: *DrmDriver) !void {
        // In real kernel implementation, this would:
        // 1. Call drm_dev_alloc() to allocate DRM device
        // 2. Set up driver callbacks (open, close, ioctl, etc.)
        // 3. Call drm_dev_register() to register with DRM core
        // 4. Register framebuffer if console support needed
        
        print("GhostNV: Registering with DRM subsystem\n");
        
        // Set up driver capabilities
        const capabilities = [_][]const u8{
            "DRM_PRIME",
            "DRM_RENDER",
            "DRM_MODESET",
            "DRM_ATOMIC",
            "DRM_GEM",
            "DRM_SYNCOBJ",
        };
        
        for (capabilities) |cap| {
            print("GhostNV: Capability: {s}\n", .{cap});
        }
    }
    
    pub fn unregister(self: *DrmDriver) void {
        if (!self.registered) return;
        
        print("nvzig: Unregistering DRM driver\n");
        
        // Clean up all resources
        for (self.crtcs.items) |*crtc| {
            crtc.disable();
        }
        
        // In real implementation, would call drm_dev_unregister
        self.registered = false;
        print("nvzig: DRM driver unregistered\n");
    }
    
    fn init_hardware(self: *DrmDriver) !void {
        print("nvzig: Initializing DRM hardware\n");
        
        // TODO: Initialize display controller
        // TODO: Set up display PLLs
        // TODO: Initialize DACs/encoders
        
        print("nvzig: DRM hardware initialization complete\n");
    }
    
    fn create_connectors(self: *DrmDriver) !void {
        // Create common connector types for modern GPUs
        const connector_configs = [_]DrmConnectorType{
            .DisplayPort,
            .DisplayPort,
            .HDMIA,
            .HDMIA,
        };
        
        for (connector_configs) |connector_type| {
            var connector = DrmConnector.init(self.allocator, self.next_id, connector_type);
            self.next_id += 1;
            
            // Detect if anything is connected
            _ = try connector.detect();
            
            try self.connectors.append(connector);
            print("nvzig: Created connector {} (type: {})\n", .{connector.id, connector_type});
        }
    }
    
    fn create_crtcs(self: *DrmDriver) !void {
        // Modern GPUs typically have 4-6 CRTCs
        const crtc_count = 4;
        
        for (0..crtc_count) |i| {
            const crtc = DrmCrtc.init(self.next_id);
            self.next_id += 1;
            
            try self.crtcs.append(crtc);
            print("nvzig: Created CRTC {}\n", .{crtc.id});
        }
    }
    
    pub fn create_framebuffer(self: *DrmDriver, width: u32, height: u32, format: u32) !*DrmFramebuffer {
        var fb = DrmFramebuffer.init(self.next_id, width, height, format);
        self.next_id += 1;
        
        // Allocate memory for framebuffer
        try fb.allocate_memory(0x80000000); // Simulate VRAM address
        try fb.map_virtual();
        
        try self.framebuffers.append(fb);
        print("nvzig: Created framebuffer {} ({}x{})\n", .{fb.id, width, height});
        
        return &self.framebuffers.items[self.framebuffers.items.len - 1];
    }
    
    pub fn set_mode(self: *DrmDriver, connector_id: u32, crtc_id: u32, mode: DrmMode) !void {
        // Find connector
        var connector: ?*DrmConnector = null;
        for (self.connectors.items) |*conn| {
            if (conn.id == connector_id) {
                connector = conn;
                break;
            }
        }
        
        if (connector == null) return DrmError.InvalidMode;
        
        // Find CRTC
        var crtc: ?*DrmCrtc = null;
        for (self.crtcs.items) |*c| {
            if (c.id == crtc_id) {
                crtc = c;
                break;
            }
        }
        
        if (crtc == null) return DrmError.InvalidMode;
        
        // Set the mode
        try connector.?.set_mode(mode);
        try crtc.?.set_config(mode, 0, 0);
        
        print("nvzig: Mode set complete: connector {} -> CRTC {}\n", .{connector_id, crtc_id});
    }
    
    // VRR (Variable Refresh Rate) functions
    pub fn enable_vrr(self: *DrmDriver, vrr_mode: VrrMode, min_refresh: u32, max_refresh: u32) !void {
        for (self.connectors.items) |*connector| {
            if (connector.connected and connector.supports_vrr) {
                try connector.enable_vrr(min_refresh, max_refresh);
                print("nvzig: VRR enabled ({}): {}-{}Hz on connector {}\n", 
                      .{vrr_mode, min_refresh, max_refresh, connector.id});
            }
        }
    }
    
    pub fn disable_vrr(self: *DrmDriver) void {
        for (self.connectors.items) |*connector| {
            if (connector.vrr_enabled) {
                connector.disable_vrr();
            }
        }
    }
    
    pub fn set_refresh_rate(self: *DrmDriver, refresh_rate: u32) !void {
        for (self.connectors.items) |*connector| {
            if (connector.connected and connector.vrr_enabled) {
                try connector.set_refresh_rate(refresh_rate);
            }
        }
    }
    
    pub fn get_vrr_capabilities(self: *DrmDriver) VrrCapabilities {
        var caps = VrrCapabilities{
            .supports_adaptive_sync = false,
            .supports_gsync = false,
            .supports_freesync = false,
            .min_refresh_rate = 60,
            .max_refresh_rate = 60,
            .connected_vrr_displays = 0,
        };
        
        for (self.connectors.items) |*connector| {
            if (connector.connected and connector.supports_vrr) {
                caps.supports_adaptive_sync = true;
                caps.supports_gsync = true; // Assume G-SYNC compatible
                caps.supports_freesync = true; // VESA Adaptive-Sync
                caps.min_refresh_rate = @min(caps.min_refresh_rate, connector.vrr_min_refresh);
                caps.max_refresh_rate = @max(caps.max_refresh_rate, connector.vrr_max_refresh);
                caps.connected_vrr_displays += 1;
            }
        }
        
        return caps;
    }
    
    // Wayland-optimized functions
    pub fn atomic_commit(self: *DrmDriver) !void {
        print("nvzig: Performing atomic commit (Wayland-optimized)\n");
        // TODO: Implement atomic modesetting for smooth Wayland experience
    }
    
    pub fn get_plane_capabilities(self: *DrmDriver) void {
        _ = self;
        print("nvzig: Querying display plane capabilities\n");
        // TODO: Return overlay/cursor plane information
    }
};

test "drm driver initialization" {
    const allocator = std.testing.allocator;
    var driver = try DrmDriver.init(allocator);
    defer driver.deinit();
    
    try std.testing.expect(!driver.registered);
    try driver.register();
    try std.testing.expect(driver.registered);
    
    driver.unregister();
    try std.testing.expect(!driver.registered);
}

test "drm mode creation" {
    const mode = DrmMode.init(1920, 1080, 60);
    try std.testing.expect(mode.hdisplay == 1920);
    try std.testing.expect(mode.vdisplay == 1080);
    try std.testing.expect(mode.refresh_rate == 60);
}

test "framebuffer creation" {
    const allocator = std.testing.allocator;
    var driver = try DrmDriver.init(allocator);
    defer driver.deinit();
    
    const fb = try driver.create_framebuffer(1920, 1080, 0x34325258); // DRM_FORMAT_XRGB8888
    try std.testing.expect(fb.width == 1920);
    try std.testing.expect(fb.height == 1080);
    try std.testing.expect(fb.size == 1920 * 1080 * 4);
}