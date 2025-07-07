const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const vibrance = @import("../color/vibrance.zig");
const gsync = @import("../gsync/display.zig");

pub const NvctlError = error{
    InvalidAttribute,
    PermissionDenied,
    DeviceNotFound,
    InvalidValue,
    NotSupported,
    InitializationFailed,
};

pub const NvctlAttribute = enum(u32) {
    // Digital Vibrance Attributes
    digital_vibrance = 0,
    color_saturation = 1,
    gamma_correction = 2,
    brightness = 3,
    contrast = 4,
    color_temperature = 5,
    
    // G-SYNC Attributes  
    gsync_mode = 100,
    gsync_min_refresh = 101,
    gsync_max_refresh = 102,
    gsync_current_refresh = 103,
    gsync_ultra_low_latency = 104,
    gsync_variable_overdrive = 105,
    
    // VRR Attributes
    vrr_enabled = 200,
    vrr_min_refresh = 201,
    vrr_max_refresh = 202,
    vrr_current_refresh = 203,
    vrr_low_framerate_compensation = 204,
    
    // Performance Attributes
    performance_level = 300,
    power_mizer_mode = 301,
    gpu_boost_clock = 302,
    memory_boost_clock = 303,
    
    // Thermal Attributes
    gpu_temperature = 400,
    gpu_fan_speed = 401,
    gpu_fan_control_mode = 402,
    
    pub fn toString(self: NvctlAttribute) []const u8 {
        return switch (self) {
            .digital_vibrance => "DigitalVibrance",
            .color_saturation => "ColorSaturation", 
            .gamma_correction => "GammaCorrection",
            .brightness => "Brightness",
            .contrast => "Contrast",
            .color_temperature => "ColorTemperature",
            .gsync_mode => "GSyncMode",
            .gsync_min_refresh => "GSyncMinRefresh",
            .gsync_max_refresh => "GSyncMaxRefresh",
            .gsync_current_refresh => "GSyncCurrentRefresh",
            .gsync_ultra_low_latency => "GSyncUltraLowLatency",
            .gsync_variable_overdrive => "GSyncVariableOverdrive",
            .vrr_enabled => "VRREnabled",
            .vrr_min_refresh => "VRRMinRefresh",
            .vrr_max_refresh => "VRRMaxRefresh",
            .vrr_current_refresh => "VRRCurrentRefresh",
            .vrr_low_framerate_compensation => "VRRLowFramerateCompensation",
            .performance_level => "PerformanceLevel",
            .power_mizer_mode => "PowerMizerMode",
            .gpu_boost_clock => "GPUBoostClock",
            .memory_boost_clock => "MemoryBoostClock",
            .gpu_temperature => "GPUCoreTemp",
            .gpu_fan_speed => "GPUFanSpeed",
            .gpu_fan_control_mode => "GPUFanControlMode",
        };
    }
};

pub const NvctlAttributeValue = union(enum) {
    integer: i32,
    float: f32,
    string: []const u8,
    boolean: bool,
    
    pub fn fromInteger(value: i32) NvctlAttributeValue {
        return NvctlAttributeValue{ .integer = value };
    }
    
    pub fn fromFloat(value: f32) NvctlAttributeValue {
        return NvctlAttributeValue{ .float = value };
    }
    
    pub fn fromBoolean(value: bool) NvctlAttributeValue {
        return NvctlAttributeValue{ .boolean = value };
    }
};

pub const NvctlInterface = struct {
    allocator: Allocator,
    vibrance_engine: *vibrance.VibranceEngine,
    gsync_manager: *gsync.GsyncManager,
    
    // Device tracking
    devices: std.ArrayList(NvctlDevice),
    
    pub fn init(allocator: Allocator, vibrance_engine: *vibrance.VibranceEngine, gsync_manager: *gsync.GsyncManager) NvctlInterface {
        return NvctlInterface{
            .allocator = allocator,
            .vibrance_engine = vibrance_engine,
            .gsync_manager = gsync_manager,
            .devices = std.ArrayList(NvctlDevice).init(allocator),
        };
    }
    
    pub fn deinit(self: *NvctlInterface) void {
        self.devices.deinit();
    }
    
    pub fn enumerate_devices(self: *NvctlInterface) !void {
        // Enumerate NVIDIA GPUs for nvctl interface
        const device_count = 1; // Simplified for RTX 4090
        
        for (0..device_count) |i| {
            const device = NvctlDevice{
                .id = @intCast(i),
                .name = "NVIDIA GeForce RTX 4090",
                .driver_version = "575.48.00",
                .cuda_version = "12.3",
                .pci_bus = 0x01,
                .pci_device = 0x00,
                .pci_function = 0x00,
            };
            
            try self.devices.append(device);
        }
        
        std.log.info("nvctl: Enumerated {} NVIDIA devices", .{device_count});
    }
    
    pub fn get_attribute(self: *NvctlInterface, device_id: u32, attribute: NvctlAttribute) !NvctlAttributeValue {
        _ = device_id; // For now, assume single GPU
        
        return switch (attribute) {
            // Digital Vibrance attributes
            .digital_vibrance => blk: {
                if (self.vibrance_engine.get_active_profile()) |profile| {
                    break :blk NvctlAttributeValue.fromInteger(profile.vibrance);
                }
                break :blk NvctlAttributeValue.fromInteger(0);
            },
            .color_saturation => blk: {
                if (self.vibrance_engine.get_active_profile()) |profile| {
                    break :blk NvctlAttributeValue.fromInteger(profile.saturation);
                }
                break :blk NvctlAttributeValue.fromInteger(0);
            },
            .gamma_correction => blk: {
                if (self.vibrance_engine.get_active_profile()) |profile| {
                    break :blk NvctlAttributeValue.fromFloat(profile.gamma);
                }
                break :blk NvctlAttributeValue.fromFloat(2.2);
            },
            .brightness => blk: {
                if (self.vibrance_engine.get_active_profile()) |profile| {
                    break :blk NvctlAttributeValue.fromInteger(profile.brightness);
                }
                break :blk NvctlAttributeValue.fromInteger(0);
            },
            .contrast => blk: {
                if (self.vibrance_engine.get_active_profile()) |profile| {
                    break :blk NvctlAttributeValue.fromInteger(profile.contrast);
                }
                break :blk NvctlAttributeValue.fromInteger(0);
            },
            .color_temperature => blk: {
                if (self.vibrance_engine.get_active_profile()) |profile| {
                    break :blk NvctlAttributeValue.fromInteger(profile.temperature);
                }
                break :blk NvctlAttributeValue.fromInteger(0);
            },
            
            // G-SYNC attributes
            .gsync_mode => blk: {
                const displays = try self.gsync_manager.get_all_display_info();
                defer self.allocator.free(displays);
                
                if (displays.len > 0) {
                    break :blk NvctlAttributeValue.fromInteger(@intFromEnum(displays[0].gsync_mode));
                }
                break :blk NvctlAttributeValue.fromInteger(0);
            },
            .gsync_current_refresh => blk: {
                const displays = try self.gsync_manager.get_all_display_info();
                defer self.allocator.free(displays);
                
                if (displays.len > 0) {
                    break :blk NvctlAttributeValue.fromInteger(@intCast(displays[0].current_refresh_hz));
                }
                break :blk NvctlAttributeValue.fromInteger(60);
            },
            .gsync_ultra_low_latency => blk: {
                const displays = try self.gsync_manager.get_all_display_info();
                defer self.allocator.free(displays);
                
                if (displays.len > 0) {
                    break :blk NvctlAttributeValue.fromBoolean(displays[0].ultra_low_latency);
                }
                break :blk NvctlAttributeValue.fromBoolean(false);
            },
            
            // Performance attributes
            .performance_level => NvctlAttributeValue.fromInteger(3), // High performance
            .gpu_temperature => NvctlAttributeValue.fromInteger(65), // Mock temperature
            .gpu_fan_speed => NvctlAttributeValue.fromInteger(1800), // Mock fan RPM
            
            else => return NvctlError.InvalidAttribute,
        };
    }
    
    pub fn set_attribute(self: *NvctlInterface, device_id: u32, attribute: NvctlAttribute, value: NvctlAttributeValue) !void {
        _ = device_id; // For now, assume single GPU
        
        switch (attribute) {
            // Digital Vibrance controls
            .digital_vibrance => {
                const vibrance_value = switch (value) {
                    .integer => |v| @as(i8, @intCast(std.math.clamp(v, -50, 100))),
                    else => return NvctlError.InvalidValue,
                };
                
                try self.vibrance_engine.real_time_adjust(vibrance_value);
                std.log.info("nvctl: Set digital vibrance to {}", .{vibrance_value});
            },
            
            .color_saturation => {
                // Create or modify active profile
                var profile = if (self.vibrance_engine.get_active_profile()) |p| p else vibrance.VibranceProfile.init("nvctl_custom");
                
                profile.saturation = switch (value) {
                    .integer => |v| @as(i8, @intCast(std.math.clamp(v, -50, 50))),
                    else => return NvctlError.InvalidValue,
                };
                
                try self.vibrance_engine.create_profile("nvctl_custom", profile);
                try self.vibrance_engine.apply_profile("nvctl_custom");
                std.log.info("nvctl: Set color saturation to {}", .{profile.saturation});
            },
            
            .gamma_correction => {
                var profile = if (self.vibrance_engine.get_active_profile()) |p| p else vibrance.VibranceProfile.init("nvctl_custom");
                
                profile.gamma = switch (value) {
                    .float => |v| std.math.clamp(v, 0.8, 3.0),
                    .integer => |v| std.math.clamp(@as(f32, @floatFromInt(v)) / 100.0, 0.8, 3.0),
                    else => return NvctlError.InvalidValue,
                };
                
                try self.vibrance_engine.create_profile("nvctl_custom", profile);
                try self.vibrance_engine.apply_profile("nvctl_custom");
                std.log.info("nvctl: Set gamma correction to {:.2}", .{profile.gamma});
            },
            
            .brightness => {
                var profile = if (self.vibrance_engine.get_active_profile()) |p| p else vibrance.VibranceProfile.init("nvctl_custom");
                
                profile.brightness = switch (value) {
                    .integer => |v| @as(i8, @intCast(std.math.clamp(v, -50, 50))),
                    else => return NvctlError.InvalidValue,
                };
                
                try self.vibrance_engine.create_profile("nvctl_custom", profile);
                try self.vibrance_engine.apply_profile("nvctl_custom");
                std.log.info("nvctl: Set brightness to {}", .{profile.brightness});
            },
            
            .contrast => {
                var profile = if (self.vibrance_engine.get_active_profile()) |p| p else vibrance.VibranceProfile.init("nvctl_custom");
                
                profile.contrast = switch (value) {
                    .integer => |v| @as(i8, @intCast(std.math.clamp(v, -50, 50))),
                    else => return NvctlError.InvalidValue,
                };
                
                try self.vibrance_engine.create_profile("nvctl_custom", profile);
                try self.vibrance_engine.apply_profile("nvctl_custom");
                std.log.info("nvctl: Set contrast to {}", .{profile.contrast});
            },
            
            // G-SYNC controls
            .gsync_mode => {
                const gsync_mode = switch (value) {
                    .integer => |v| @as(gsync.GsyncMode, @enumFromInt(@as(u8, @intCast(std.math.clamp(v, 0, 4))))),
                    else => return NvctlError.InvalidValue,
                };
                
                try self.gsync_manager.enable_gsync(gsync_mode);
                std.log.info("nvctl: Set G-SYNC mode to {}", .{gsync_mode});
            },
            
            .gsync_ultra_low_latency => {
                const enabled = switch (value) {
                    .boolean => |v| v,
                    .integer => |v| v != 0,
                    else => return NvctlError.InvalidValue,
                };
                
                if (enabled) {
                    self.gsync_manager.optimize_for_game(.competitive_fps);
                    std.log.info("nvctl: Enabled G-SYNC Ultra Low Latency");
                } else {
                    self.gsync_manager.optimize_for_game(.immersive_single_player);
                    std.log.info("nvctl: Disabled G-SYNC Ultra Low Latency");
                }
            },
            
            else => return NvctlError.InvalidAttribute,
        }
    }
    
    pub fn list_attributes(self: *NvctlInterface, device_id: u32) ![]NvctlAttribute {
        _ = device_id;
        
        // Return all supported attributes
        const attributes = [_]NvctlAttribute{
            .digital_vibrance,
            .color_saturation,
            .gamma_correction,
            .brightness,
            .contrast,
            .color_temperature,
            .gsync_mode,
            .gsync_min_refresh,
            .gsync_max_refresh,
            .gsync_current_refresh,
            .gsync_ultra_low_latency,
            .gsync_variable_overdrive,
            .vrr_enabled,
            .vrr_min_refresh,
            .vrr_max_refresh,
            .vrr_current_refresh,
            .performance_level,
            .gpu_temperature,
            .gpu_fan_speed,
        };
        
        return try self.allocator.dupe(NvctlAttribute, &attributes);
    }
    
    pub fn apply_vibrance_profile(self: *NvctlInterface, profile_name: []const u8) !void {
        try self.vibrance_engine.apply_profile(profile_name);
        std.log.info("nvctl: Applied vibrance profile '{s}'", .{profile_name});
    }
    
    pub fn create_vibrance_profile_from_current(self: *NvctlInterface, profile_name: []const u8) !void {
        if (self.vibrance_engine.get_active_profile()) |current_profile| {
            try self.vibrance_engine.create_profile(profile_name, current_profile);
            std.log.info("nvctl: Created vibrance profile '{s}' from current settings", .{profile_name});
        } else {
            return NvctlError.InvalidValue;
        }
    }
    
    pub fn get_device_info(self: *NvctlInterface, device_id: u32) !NvctlDevice {
        if (device_id >= self.devices.items.len) {
            return NvctlError.DeviceNotFound;
        }
        
        return self.devices.items[device_id];
    }
};

pub const NvctlDevice = struct {
    id: u32,
    name: []const u8,
    driver_version: []const u8,
    cuda_version: []const u8,
    pci_bus: u8,
    pci_device: u8,
    pci_function: u8,
};

// CLI command compatibility with existing nvctl
pub const NvctlCLI = struct {
    interface: *NvctlInterface,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, interface: *NvctlInterface) NvctlCLI {
        return NvctlCLI{
            .interface = interface,
            .allocator = allocator,
        };
    }
    
    pub fn execute_command(self: *NvctlCLI, args: [][]const u8) !void {
        if (args.len < 2) {
            try self.print_help();
            return;
        }
        
        const command = args[1];
        
        if (std.mem.eql(u8, command, "vibrance")) {
            try self.handle_vibrance_command(args[2..]);
        } else if (std.mem.eql(u8, command, "gsync")) {
            try self.handle_gsync_command(args[2..]);
        } else if (std.mem.eql(u8, command, "get")) {
            try self.handle_get_command(args[2..]);
        } else if (std.mem.eql(u8, command, "set")) {
            try self.handle_set_command(args[2..]);
        } else {
            std.debug.print("Unknown command: {s}\n", .{command});
            try self.print_help();
        }
    }
    
    fn handle_vibrance_command(self: *NvctlCLI, args: [][]const u8) !void {
        if (args.len == 0) {
            const current = try self.interface.get_attribute(0, .digital_vibrance);
            std.debug.print("Current digital vibrance: {}\n", .{current.integer});
            return;
        }
        
        const value = std.fmt.parseInt(i32, args[0], 10) catch {
            std.debug.print("Invalid vibrance value: {s}\n", .{args[0]});
            return;
        };
        
        try self.interface.set_attribute(0, .digital_vibrance, NvctlAttributeValue.fromInteger(value));
        std.debug.print("Set digital vibrance to: {}\n", .{value});
    }
    
    fn handle_gsync_command(self: *NvctlCLI, args: [][]const u8) !void {
        if (args.len == 0) {
            const mode = try self.interface.get_attribute(0, .gsync_mode);
            const refresh = try self.interface.get_attribute(0, .gsync_current_refresh);
            const ull = try self.interface.get_attribute(0, .gsync_ultra_low_latency);
            
            std.debug.print("G-SYNC Mode: {}\n", .{mode.integer});
            std.debug.print("Current Refresh: {}Hz\n", .{refresh.integer});
            std.debug.print("Ultra Low Latency: {}\n", .{ull.boolean});
            return;
        }
        
        const subcommand = args[0];
        if (std.mem.eql(u8, subcommand, "enable")) {
            try self.interface.set_attribute(0, .gsync_mode, NvctlAttributeValue.fromInteger(2)); // G-SYNC Certified
            std.debug.print("G-SYNC enabled\n");
        } else if (std.mem.eql(u8, subcommand, "disable")) {
            try self.interface.set_attribute(0, .gsync_mode, NvctlAttributeValue.fromInteger(0)); // Disabled
            std.debug.print("G-SYNC disabled\n");
        } else if (std.mem.eql(u8, subcommand, "ull")) {
            if (args.len > 1) {
                const enabled = std.mem.eql(u8, args[1], "1") or std.mem.eql(u8, args[1], "true");
                try self.interface.set_attribute(0, .gsync_ultra_low_latency, NvctlAttributeValue.fromBoolean(enabled));
                std.debug.print("G-SYNC Ultra Low Latency: {}\n", .{enabled});
            }
        }
    }
    
    fn handle_get_command(self: *NvctlCLI, args: [][]const u8) !void {
        if (args.len == 0) {
            std.debug.print("Usage: nvctl get <attribute>\n");
            return;
        }
        
        const attr_name = args[0];
        const attribute = self.parse_attribute_name(attr_name) orelse {
            std.debug.print("Unknown attribute: {s}\n", .{attr_name});
            return;
        };
        
        const value = try self.interface.get_attribute(0, attribute);
        std.debug.print("{s}: ", .{attribute.toString()});
        
        switch (value) {
            .integer => |v| std.debug.print("{}\n", .{v}),
            .float => |v| std.debug.print("{:.2}\n", .{v}),
            .boolean => |v| std.debug.print("{}\n", .{v}),
            .string => |v| std.debug.print("{s}\n", .{v}),
        }
    }
    
    fn handle_set_command(self: *NvctlCLI, args: [][]const u8) !void {
        if (args.len < 2) {
            std.debug.print("Usage: nvctl set <attribute> <value>\n");
            return;
        }
        
        const attr_name = args[0];
        const value_str = args[1];
        
        const attribute = self.parse_attribute_name(attr_name) orelse {
            std.debug.print("Unknown attribute: {s}\n", .{attr_name});
            return;
        };
        
        const value = self.parse_attribute_value(attribute, value_str) catch {
            std.debug.print("Invalid value for {s}: {s}\n", .{ attr_name, value_str });
            return;
        };
        
        try self.interface.set_attribute(0, attribute, value);
        std.debug.print("Set {s} to {s}\n", .{ attr_name, value_str });
    }
    
    fn parse_attribute_name(self: *NvctlCLI, name: []const u8) ?NvctlAttribute {
        _ = self;
        
        if (std.mem.eql(u8, name, "vibrance") or std.mem.eql(u8, name, "DigitalVibrance")) {
            return .digital_vibrance;
        } else if (std.mem.eql(u8, name, "saturation") or std.mem.eql(u8, name, "ColorSaturation")) {
            return .color_saturation;
        } else if (std.mem.eql(u8, name, "gamma") or std.mem.eql(u8, name, "GammaCorrection")) {
            return .gamma_correction;
        } else if (std.mem.eql(u8, name, "brightness") or std.mem.eql(u8, name, "Brightness")) {
            return .brightness;
        } else if (std.mem.eql(u8, name, "contrast") or std.mem.eql(u8, name, "Contrast")) {
            return .contrast;
        } else if (std.mem.eql(u8, name, "gsync") or std.mem.eql(u8, name, "GSyncMode")) {
            return .gsync_mode;
        } else if (std.mem.eql(u8, name, "temp") or std.mem.eql(u8, name, "GPUCoreTemp")) {
            return .gpu_temperature;
        }
        
        return null;
    }
    
    fn parse_attribute_value(self: *NvctlCLI, attribute: NvctlAttribute, value_str: []const u8) !NvctlAttributeValue {
        _ = self;
        
        return switch (attribute) {
            .digital_vibrance, .color_saturation, .brightness, .contrast, .gsync_mode, .gpu_temperature, .gpu_fan_speed => {
                const int_val = try std.fmt.parseInt(i32, value_str, 10);
                return NvctlAttributeValue.fromInteger(int_val);
            },
            .gamma_correction => {
                const float_val = try std.fmt.parseFloat(f32, value_str);
                return NvctlAttributeValue.fromFloat(float_val);
            },
            .gsync_ultra_low_latency, .vrr_enabled => {
                const bool_val = std.mem.eql(u8, value_str, "1") or 
                                std.mem.eql(u8, value_str, "true") or 
                                std.mem.eql(u8, value_str, "on");
                return NvctlAttributeValue.fromBoolean(bool_val);
            },
            else => return NvctlError.InvalidValue,
        };
    }
    
    fn print_help(self: *NvctlCLI) !void {
        _ = self;
        
        std.debug.print(
            \\nvctl - NVIDIA Control Interface for GhostNV
            \\
            \\Usage:
            \\  nvctl vibrance [value]           Get/set digital vibrance (-50 to 100)
            \\  nvctl gsync [enable|disable|ull] Control G-SYNC settings
            \\  nvctl get <attribute>            Get attribute value
            \\  nvctl set <attribute> <value>    Set attribute value
            \\
            \\Common Attributes:
            \\  vibrance, saturation, gamma, brightness, contrast
            \\  gsync, temp, fan
            \\
            \\Examples:
            \\  nvctl vibrance 50               Set vibrance to 50%
            \\  nvctl gsync enable              Enable G-SYNC
            \\  nvctl gsync ull true            Enable Ultra Low Latency
            \\  nvctl set brightness 10         Set brightness to +10
            \\  nvctl get temp                  Get GPU temperature
            \\
        );
    }
};

// Test functions
test "nvctl interface" {
    const allocator = std.testing.allocator;
    
    var drm_driver = try @import("../drm/driver.zig").DrmDriver.init(allocator);
    defer drm_driver.deinit();
    
    var vibrance_engine = vibrance.VibranceEngine.init(allocator, &drm_driver);
    defer vibrance_engine.deinit();
    
    var gsync_manager = gsync.GsyncManager.init(allocator, &drm_driver);
    defer gsync_manager.deinit();
    
    var interface = NvctlInterface.init(allocator, &vibrance_engine, &gsync_manager);
    defer interface.deinit();
    
    try interface.enumerate_devices();
    
    // Test vibrance control
    try interface.set_attribute(0, .digital_vibrance, NvctlAttributeValue.fromInteger(50));
    const vibrance_val = try interface.get_attribute(0, .digital_vibrance);
    try std.testing.expect(vibrance_val.integer == 50);
}