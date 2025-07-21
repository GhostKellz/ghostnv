const std = @import("std");
const Allocator = std.mem.Allocator;
const pci = @import("pci.zig");
const linux = std.os.linux;

/// GPU Clock Control Hardware Abstraction Layer
/// Manages GPU core clocks, memory clocks, boost states, and performance levels

pub const ClockError = error{
    UnsupportedArchitecture,
    InvalidClockDomain,
    ClockOutOfRange,
    ThermalThrottling,
    PowerLimitExceeded,
    HardwareError,
    PermissionDenied,
    OutOfMemory,
};

pub const ClockDomain = enum(u8) {
    graphics_core = 0,
    graphics_memory = 1,
    shader_core = 2,
    video_decode = 3,
    video_encode = 4,
    display = 5,
    pcie = 6,
    system = 7,
    
    pub fn toString(self: ClockDomain) []const u8 {
        return switch (self) {
            .graphics_core => "Graphics Core",
            .graphics_memory => "Graphics Memory",
            .shader_core => "Shader Core", 
            .video_decode => "Video Decode",
            .video_encode => "Video Encode",
            .display => "Display",
            .pcie => "PCIe",
            .system => "System",
        };
    }
};

pub const PerformanceLevel = enum(u8) {
    p0 = 0, // Maximum performance
    p1 = 1,
    p2 = 2,
    p3 = 3,
    p4 = 4,
    p5 = 5,
    p6 = 6,
    p7 = 7,
    p8 = 8, // Minimum performance
    idle = 255,
    
    pub fn getPowerBudget(self: PerformanceLevel) u32 {
        return switch (self) {
            .p0 => 300, // watts
            .p1 => 250,
            .p2 => 200,
            .p3 => 150,
            .p4 => 125,
            .p5 => 100,
            .p6 => 75,
            .p7 => 50,
            .p8 => 25,
            .idle => 10,
        };
    }
};

pub const ClockState = struct {
    base_clock_mhz: u32,
    boost_clock_mhz: u32,
    memory_clock_mhz: u32,
    shader_clock_mhz: u32,
    current_pstate: PerformanceLevel,
    target_pstate: PerformanceLevel,
    thermal_throttled: bool,
    power_throttled: bool,
    voltage_mv: u32,
};

pub const ClockLimits = struct {
    min_graphics_mhz: u32,
    max_graphics_mhz: u32,
    min_memory_mhz: u32,
    max_memory_mhz: u32,
    min_shader_mhz: u32,
    max_shader_mhz: u32,
    thermal_limit_c: u8,
    power_limit_w: u32,
    voltage_min_mv: u32,
    voltage_max_mv: u32,
};

pub const BoostManager = struct {
    const Self = @This();
    
    enabled: bool,
    temperature_limit: u8,
    power_limit: u32,
    current_boost_offset: i32,
    max_boost_offset: i32,
    boost_duration_ms: u32,
    thermal_headroom: u8,
    
    pub fn init() Self {
        return Self{
            .enabled = true,
            .temperature_limit = 83, // 83°C typical GPU boost limit
            .power_limit = 300, // 300W typical high-end GPU
            .current_boost_offset = 0,
            .max_boost_offset = 200, // +200MHz max boost
            .boost_duration_ms = 1000,
            .thermal_headroom = 10, // 10°C headroom for boost
        };
    }
    
    pub fn canBoost(self: *Self, temperature: u8, power_usage: u32) bool {
        return self.enabled and 
               temperature < (self.temperature_limit - self.thermal_headroom) and
               power_usage < self.power_limit;
    }
    
    pub fn calculateBoostOffset(self: *Self, load_percent: u8, temperature: u8) i32 {
        if (!self.canBoost(temperature, 0)) return 0;
        
        const temp_factor = @as(f32, @floatFromInt(self.temperature_limit - temperature)) / 20.0;
        const load_factor = @as(f32, @floatFromInt(load_percent)) / 100.0;
        
        const boost_offset = @as(i32, @intFromFloat(
            @as(f32, @floatFromInt(self.max_boost_offset)) * temp_factor * load_factor
        ));
        
        return @min(boost_offset, self.max_boost_offset);
    }
};

pub const ClockManager = struct {
    const Self = @This();
    
    allocator: Allocator,
    device: pci.PciDevice,
    current_state: ClockState,
    limits: ClockLimits,
    boost_manager: BoostManager,
    architecture: pci.GpuArchitecture,
    mmio_base: ?*volatile u8,
    
    // Clock control registers (architecture-specific offsets)
    const CLOCK_CONTROL_BASE: u32 = 0x137000;
    const PSTATE_CONTROL: u32 = CLOCK_CONTROL_BASE + 0x00;
    const GRAPHICS_CLOCK: u32 = CLOCK_CONTROL_BASE + 0x04;
    const MEMORY_CLOCK: u32 = CLOCK_CONTROL_BASE + 0x08;
    const SHADER_CLOCK: u32 = CLOCK_CONTROL_BASE + 0x0C;
    const VOLTAGE_CONTROL: u32 = CLOCK_CONTROL_BASE + 0x10;
    const THERMAL_STATUS: u32 = CLOCK_CONTROL_BASE + 0x14;
    const POWER_STATUS: u32 = CLOCK_CONTROL_BASE + 0x18;
    
    pub fn init(allocator: Allocator, device: pci.PciDevice) !Self {
        const architecture = try pci.detectGpuArchitecture(device.device_id);
        
        var self = Self{
            .allocator = allocator,
            .device = device,
            .current_state = undefined,
            .limits = undefined,
            .boost_manager = BoostManager.init(),
            .architecture = architecture,
            .mmio_base = null,
        };
        
        try self.mapMemoryRegions();
        try self.initializeClockLimits();
        try self.readCurrentState();
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        if (self.mmio_base) |base| {
            // Unmap MMIO regions
            _ = linux.munmap(@ptrCast(base), 0x1000000); // 16MB typical GPU MMIO
        }
    }
    
    fn mapMemoryRegions(self: *Self) !void {
        // Map GPU MMIO BAR0 for register access
        const bar0_fd = try std.fs.openFileAbsolute("/sys/bus/pci/devices/0000:01:00.0/resource0", .{});
        defer bar0_fd.close();
        
        const mmio_ptr = linux.mmap(
            null,
            0x1000000, // 16MB
            linux.PROT.READ | linux.PROT.WRITE,
            linux.MAP.SHARED,
            bar0_fd.handle,
            0
        );
        
        if (mmio_ptr == linux.MAP.FAILED) {
            return ClockError.PermissionDenied;
        }
        
        self.mmio_base = @ptrCast(@alignCast(mmio_ptr));
    }
    
    fn initializeClockLimits(self: *Self) !void {
        // Set architecture-specific clock limits
        self.limits = switch (self.architecture) {
            .ada_lovelace => ClockLimits{
                .min_graphics_mhz = 300,
                .max_graphics_mhz = 2800,
                .min_memory_mhz = 405,
                .max_memory_mhz = 1313, // 21Gbps effective
                .min_shader_mhz = 600,
                .max_shader_mhz = 2800,
                .thermal_limit_c = 83,
                .power_limit_w = 450,
                .voltage_min_mv = 700,
                .voltage_max_mv = 1100,
            },
            .ampere => ClockLimits{
                .min_graphics_mhz = 300,
                .max_graphics_mhz = 2100,
                .min_memory_mhz = 405,
                .max_memory_mhz = 1219, // 19.5Gbps effective
                .min_shader_mhz = 600,
                .max_shader_mhz = 2100,
                .thermal_limit_c = 83,
                .power_limit_w = 350,
                .voltage_min_mv = 700,
                .voltage_max_mv = 1100,
            },
            .turing => ClockLimits{
                .min_graphics_mhz = 300,
                .max_graphics_mhz = 2000,
                .min_memory_mhz = 405,
                .max_memory_mhz = 875, // 14Gbps effective
                .min_shader_mhz = 600,
                .max_shader_mhz = 2000,
                .thermal_limit_c = 83,
                .power_limit_w = 280,
                .voltage_min_mv = 700,
                .voltage_max_mv = 1100,
            },
            else => return ClockError.UnsupportedArchitecture,
        };
    }
    
    fn readRegister(self: *Self, offset: u32) u32 {
        if (self.mmio_base) |base| {
            const reg_ptr: *volatile u32 = @ptrCast(@alignCast(base + offset));
            return reg_ptr.*;
        }
        return 0;
    }
    
    fn writeRegister(self: *Self, offset: u32, value: u32) void {
        if (self.mmio_base) |base| {
            const reg_ptr: *volatile u32 = @ptrCast(@alignCast(base + offset));
            reg_ptr.* = value;
        }
    }
    
    pub fn readCurrentState(self: *Self) !void {
        // Read current clock frequencies from hardware
        const graphics_reg = self.readRegister(GRAPHICS_CLOCK);
        const memory_reg = self.readRegister(MEMORY_CLOCK);
        const shader_reg = self.readRegister(SHADER_CLOCK);
        const pstate_reg = self.readRegister(PSTATE_CONTROL);
        const voltage_reg = self.readRegister(VOLTAGE_CONTROL);
        const thermal_reg = self.readRegister(THERMAL_STATUS);
        const power_reg = self.readRegister(POWER_STATUS);
        
        self.current_state = ClockState{
            .base_clock_mhz = (graphics_reg & 0xFFFF),
            .boost_clock_mhz = ((graphics_reg >> 16) & 0xFFFF),
            .memory_clock_mhz = (memory_reg & 0xFFFF),
            .shader_clock_mhz = (shader_reg & 0xFFFF),
            .current_pstate = @enumFromInt(@as(u8, @truncate(pstate_reg & 0xFF))),
            .target_pstate = @enumFromInt(@as(u8, @truncate((pstate_reg >> 8) & 0xFF))),
            .thermal_throttled = (thermal_reg & 0x1) != 0,
            .power_throttled = (power_reg & 0x1) != 0,
            .voltage_mv = (voltage_reg & 0xFFFF),
        };
    }
    
    pub fn setPerformanceLevel(self: *Self, pstate: PerformanceLevel) !void {
        // Validate the performance state is supported
        if (@intFromEnum(pstate) > 8 and pstate != .idle) {
            return ClockError.InvalidClockDomain;
        }
        
        // Set target P-state
        const pstate_reg = (@as(u32, @intFromEnum(pstate)) << 8) | @intFromEnum(self.current_state.current_pstate);
        self.writeRegister(PSTATE_CONTROL, pstate_reg);
        
        // Wait for transition
        var timeout: u32 = 1000;
        while (timeout > 0) {
            try self.readCurrentState();
            if (self.current_state.current_pstate == pstate) break;
            timeout -= 1;
            std.time.sleep(1_000_000); // 1ms
        }
        
        if (timeout == 0) {
            return ClockError.HardwareError;
        }
    }
    
    pub fn setClockFrequency(self: *Self, domain: ClockDomain, frequency_mhz: u32) !void {
        // Validate frequency is within limits
        const valid = switch (domain) {
            .graphics_core => frequency_mhz >= self.limits.min_graphics_mhz and frequency_mhz <= self.limits.max_graphics_mhz,
            .graphics_memory => frequency_mhz >= self.limits.min_memory_mhz and frequency_mhz <= self.limits.max_memory_mhz,
            .shader_core => frequency_mhz >= self.limits.min_shader_mhz and frequency_mhz <= self.limits.max_shader_mhz,
            else => return ClockError.InvalidClockDomain,
        };
        
        if (!valid) {
            return ClockError.ClockOutOfRange;
        }
        
        // Apply clock frequency
        const register_offset = switch (domain) {
            .graphics_core => GRAPHICS_CLOCK,
            .graphics_memory => MEMORY_CLOCK,
            .shader_core => SHADER_CLOCK,
            else => return ClockError.InvalidClockDomain,
        };
        
        self.writeRegister(register_offset, frequency_mhz);
        
        // Verify the change took effect
        std.time.sleep(10_000_000); // 10ms settle time
        try self.readCurrentState();
    }
    
    pub fn enableBoost(self: *Self, enable: bool) void {
        self.boost_manager.enabled = enable;
    }
    
    pub fn updateBoostState(self: *Self, temperature: u8, load_percent: u8) !void {
        if (!self.boost_manager.enabled) return;
        
        const boost_offset = self.boost_manager.calculateBoostOffset(load_percent, temperature);
        
        if (boost_offset != self.boost_manager.current_boost_offset) {
            self.boost_manager.current_boost_offset = boost_offset;
            
            // Apply boost offset to graphics clock
            const new_boost_clock = @as(u32, @intCast(
                @as(i32, @intCast(self.current_state.base_clock_mhz)) + boost_offset
            ));
            
            if (new_boost_clock <= self.limits.max_graphics_mhz) {
                try self.setClockFrequency(.graphics_core, new_boost_clock);
            }
        }
    }
    
    pub fn getClockInfo(self: *Self) ClockState {
        return self.current_state;
    }
    
    pub fn getClockLimits(self: *Self) ClockLimits {
        return self.limits;
    }
    
    pub fn handleThermalThrottling(self: *Self, temperature: u8) !void {
        const thermal_limit = self.limits.thermal_limit_c;
        
        if (temperature >= thermal_limit) {
            // Emergency thermal throttling
            self.current_state.thermal_throttled = true;
            
            // Reduce to P4 (75% performance) or lower
            var target_pstate = PerformanceLevel.p4;
            
            if (temperature >= thermal_limit + 5) {
                target_pstate = PerformanceLevel.p6; // 50% performance
            }
            if (temperature >= thermal_limit + 10) {
                target_pstate = PerformanceLevel.p8; // Minimum performance
            }
            
            try self.setPerformanceLevel(target_pstate);
        } else if (temperature < thermal_limit - 5 and self.current_state.thermal_throttled) {
            // Recovery from thermal throttling
            self.current_state.thermal_throttled = false;
            try self.setPerformanceLevel(PerformanceLevel.p0);
        }
    }
    
    pub fn handlePowerThrottling(self: *Self, power_usage: u32) !void {
        const power_limit = self.limits.power_limit_w;
        
        if (power_usage >= power_limit) {
            self.current_state.power_throttled = true;
            
            // Reduce performance level based on power overage
            const overage_percent = ((power_usage - power_limit) * 100) / power_limit;
            
            var target_pstate = PerformanceLevel.p2;
            if (overage_percent > 10) target_pstate = PerformanceLevel.p4;
            if (overage_percent > 20) target_pstate = PerformanceLevel.p6;
            
            try self.setPerformanceLevel(target_pstate);
        } else if (power_usage < (power_limit * 90 / 100) and self.current_state.power_throttled) {
            // Recovery from power throttling
            self.current_state.power_throttled = false;
            try self.setPerformanceLevel(PerformanceLevel.p0);
        }
    }
};

pub fn initClockManager(allocator: Allocator, device: pci.PciDevice) !ClockManager {
    return ClockManager.init(allocator, device);
}