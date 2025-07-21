const std = @import("std");
const Allocator = std.mem.Allocator;
const pci = @import("pci.zig");
const clock = @import("clock.zig");
const linux = std.os.linux;

/// GPU Power Management Hardware Abstraction Layer
/// Manages GPU power states, DVFS, power monitoring, and power budgets

pub const PowerError = error{
    UnsupportedPowerState,
    PowerStateTransitionFailed,
    PowerBudgetExceeded,
    VoltageOutOfRange,
    HardwareError,
    PermissionDenied,
    ThermalEmergency,
    OutOfMemory,
};

pub const PowerState = enum(u8) {
    d0 = 0,     // Full power
    d1 = 1,     // Light sleep
    d2 = 2,     // Deep sleep  
    d3_hot = 3, // System suspend
    d3_cold = 4, // System off
    gc6 = 5,    // NVIDIA GPU Context Save/Restore
    
    pub fn toString(self: PowerState) []const u8 {
        return switch (self) {
            .d0 => "D0 - Full Power",
            .d1 => "D1 - Light Sleep",
            .d2 => "D2 - Deep Sleep",
            .d3_hot => "D3 Hot - System Suspend",
            .d3_cold => "D3 Cold - System Off", 
            .gc6 => "GC6 - GPU Context Save",
        };
    }
    
    pub fn getPowerConsumption(self: PowerState) u32 {
        return switch (self) {
            .d0 => 300,     // Full TGP
            .d1 => 50,      // Idle power
            .d2 => 20,      // Deep idle
            .d3_hot => 10,  // Suspend power
            .d3_cold => 0,  // No power
            .gc6 => 5,      // Context preserved
        };
    }
};

pub const VoltageRail = enum(u8) {
    core = 0,
    memory = 1,
    pll = 2,
    io = 3,
    
    pub fn getDefaultVoltage(self: VoltageRail) u32 {
        return switch (self) {
            .core => 900,    // 0.9V typical GPU core
            .memory => 1350, // 1.35V GDDR6
            .pll => 1800,    // 1.8V PLL
            .io => 1200,     // 1.2V I/O
        };
    }
};

pub const PowerMetrics = struct {
    total_power_w: f32,
    core_power_w: f32,
    memory_power_w: f32,
    pcie_power_w: f32,
    cooling_power_w: f32,
    efficiency_percent: f32,
    temperature_c: u8,
    fan_rpm: u16,
    power_limit_w: u32,
    voltage_core_mv: u32,
    voltage_memory_mv: u32,
    power_state: PowerState,
    throttle_reasons: ThrottleReason,
};

pub const ThrottleReason = packed struct {
    thermal: bool,
    power: bool,
    voltage: bool,
    current: bool,
    reliability: bool,
    external: bool,
    _padding: u2 = 0,
};

pub const DVFSPoint = struct {
    voltage_mv: u32,
    frequency_mhz: u32,
    power_w: f32,
    temperature_limit_c: u8,
    
    pub fn efficiency(self: DVFSPoint) f32 {
        if (self.power_w == 0) return 0;
        return @as(f32, @floatFromInt(self.frequency_mhz)) / self.power_w;
    }
};

pub const PowerProfile = struct {
    name: []const u8,
    max_power_w: u32,
    dvfs_points: []DVFSPoint,
    aggressive_boost: bool,
    thermal_target_c: u8,
    min_fan_speed_percent: u8,
};

pub const PowerManager = struct {
    const Self = @This();
    
    allocator: Allocator,
    device: pci.PciDevice,
    current_state: PowerState,
    target_state: PowerState,
    current_metrics: PowerMetrics,
    active_profile: PowerProfile,
    mmio_base: ?*volatile u8,
    clock_manager: ?*clock.ClockManager,
    
    // Power management registers
    const POWER_CONTROL_BASE: u32 = 0x138000;
    const POWER_STATE_CONTROL: u32 = POWER_CONTROL_BASE + 0x00;
    const VOLTAGE_CONTROL: u32 = POWER_CONTROL_BASE + 0x04;
    const POWER_SENSORS: u32 = POWER_CONTROL_BASE + 0x08;
    const POWER_LIMITS: u32 = POWER_CONTROL_BASE + 0x0C;
    const DVFS_CONTROL: u32 = POWER_CONTROL_BASE + 0x10;
    const THERMAL_CONTROL: u32 = POWER_CONTROL_BASE + 0x14;
    const FAN_CONTROL: u32 = POWER_CONTROL_BASE + 0x18;
    const GC6_CONTROL: u32 = POWER_CONTROL_BASE + 0x1C;
    
    // Default power profiles
    const PERFORMANCE_PROFILE = PowerProfile{
        .name = "Performance",
        .max_power_w = 450,
        .dvfs_points = &[_]DVFSPoint{
            .{ .voltage_mv = 1100, .frequency_mhz = 2800, .power_w = 400, .temperature_limit_c = 83 },
            .{ .voltage_mv = 1000, .frequency_mhz = 2400, .power_w = 300, .temperature_limit_c = 80 },
            .{ .voltage_mv = 900, .frequency_mhz = 2000, .power_w = 200, .temperature_limit_c = 75 },
        },
        .aggressive_boost = true,
        .thermal_target_c = 83,
        .min_fan_speed_percent = 30,
    };
    
    const BALANCED_PROFILE = PowerProfile{
        .name = "Balanced",
        .max_power_w = 300,
        .dvfs_points = &[_]DVFSPoint{
            .{ .voltage_mv = 1000, .frequency_mhz = 2200, .power_w = 280, .temperature_limit_c = 80 },
            .{ .voltage_mv = 925, .frequency_mhz = 1900, .power_w = 200, .temperature_limit_c = 75 },
            .{ .voltage_mv = 850, .frequency_mhz = 1600, .power_w = 140, .temperature_limit_c = 70 },
        },
        .aggressive_boost = false,
        .thermal_target_c = 75,
        .min_fan_speed_percent = 25,
    };
    
    const EFFICIENCY_PROFILE = PowerProfile{
        .name = "Efficiency", 
        .max_power_w = 200,
        .dvfs_points = &[_]DVFSPoint{
            .{ .voltage_mv = 900, .frequency_mhz = 1800, .power_w = 180, .temperature_limit_c = 70 },
            .{ .voltage_mv = 825, .frequency_mhz = 1500, .power_w = 120, .temperature_limit_c = 65 },
            .{ .voltage_mv = 750, .frequency_mhz = 1200, .power_w = 80, .temperature_limit_c = 60 },
        },
        .aggressive_boost = false,
        .thermal_target_c = 65,
        .min_fan_speed_percent = 20,
    };
    
    pub fn init(allocator: Allocator, device: pci.PciDevice, clock_mgr: ?*clock.ClockManager) !Self {
        var self = Self{
            .allocator = allocator,
            .device = device,
            .current_state = PowerState.d0,
            .target_state = PowerState.d0,
            .current_metrics = undefined,
            .active_profile = BALANCED_PROFILE,
            .mmio_base = null,
            .clock_manager = clock_mgr,
        };
        
        try self.mapMemoryRegions();
        try self.initializePowerSubsystem();
        try self.readCurrentMetrics();
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        // Ensure GPU is in safe power state before cleanup
        self.setPowerState(.d1) catch {};
        
        if (self.mmio_base) |base| {
            _ = linux.munmap(@ptrCast(base), 0x1000000);
        }
    }
    
    fn mapMemoryRegions(self: *Self) !void {
        // Reuse mapping from clock manager or create new one
        if (self.clock_manager) |clock_mgr| {
            self.mmio_base = clock_mgr.mmio_base;
        } else {
            const bar0_fd = try std.fs.openFileAbsolute("/sys/bus/pci/devices/0000:01:00.0/resource0", .{});
            defer bar0_fd.close();
            
            const mmio_ptr = linux.mmap(
                null,
                0x1000000,
                linux.PROT.READ | linux.PROT.WRITE,
                linux.MAP.SHARED,
                bar0_fd.handle,
                0
            );
            
            if (mmio_ptr == linux.MAP.FAILED) {
                return PowerError.PermissionDenied;
            }
            
            self.mmio_base = @ptrCast(@alignCast(mmio_ptr));
        }
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
    
    fn initializePowerSubsystem(self: *Self) !void {
        // Initialize power management hardware
        
        // Set default power limits
        const power_limit_reg = (self.active_profile.max_power_w & 0xFFFF) |
                               ((self.active_profile.thermal_target_c & 0xFF) << 16);
        self.writeRegister(POWER_LIMITS, power_limit_reg);
        
        // Configure voltage regulators
        try self.setVoltage(.core, VoltageRail.core.getDefaultVoltage());
        try self.setVoltage(.memory, VoltageRail.memory.getDefaultVoltage());
        
        // Initialize fan control
        const fan_control = (self.active_profile.min_fan_speed_percent & 0xFF) |
                           (1 << 31); // Enable automatic fan control
        self.writeRegister(FAN_CONTROL, fan_control);
        
        // Enable GC6 context save/restore
        self.writeRegister(GC6_CONTROL, 0x1);
    }
    
    pub fn readCurrentMetrics(self: *Self) !void {
        // Read power sensors
        const power_sensors = self.readRegister(POWER_SENSORS);
        const thermal_status = self.readRegister(THERMAL_CONTROL);
        const voltage_status = self.readRegister(VOLTAGE_CONTROL);
        const fan_status = self.readRegister(FAN_CONTROL);
        const power_state_reg = self.readRegister(POWER_STATE_CONTROL);
        
        // Parse power consumption (watts as fixed-point)
        const total_power_raw = (power_sensors & 0xFFFF);
        const core_power_raw = ((power_sensors >> 16) & 0xFFFF);
        
        // Parse temperature and fan speed
        const temperature = @as(u8, @truncate(thermal_status & 0xFF));
        const fan_rpm = @as(u16, @truncate((thermal_status >> 16) & 0xFFFF));
        
        // Parse voltages (millivolts)
        const voltage_core = (voltage_status & 0xFFFF);
        const voltage_memory = ((voltage_status >> 16) & 0xFFFF);
        
        // Parse power state
        const power_state = @as(PowerState, @enumFromInt(@as(u8, @truncate(power_state_reg & 0xFF))));
        
        // Calculate derived metrics
        const efficiency = if (total_power_raw > 0) 
            (@as(f32, @floatFromInt(self.getCurrentFrequency())) / @as(f32, @floatFromInt(total_power_raw))) * 10.0
        else 0.0;
        
        self.current_metrics = PowerMetrics{
            .total_power_w = @as(f32, @floatFromInt(total_power_raw)) / 10.0, // Fixed-point conversion
            .core_power_w = @as(f32, @floatFromInt(core_power_raw)) / 10.0,
            .memory_power_w = @as(f32, @floatFromInt(total_power_raw - core_power_raw)) / 10.0,
            .pcie_power_w = 25.0, // Estimate
            .cooling_power_w = @as(f32, @floatFromInt(fan_rpm)) / 1000.0, // Rough estimate
            .efficiency_percent = efficiency,
            .temperature_c = temperature,
            .fan_rpm = fan_rpm,
            .power_limit_w = self.active_profile.max_power_w,
            .voltage_core_mv = voltage_core,
            .voltage_memory_mv = voltage_memory,
            .power_state = power_state,
            .throttle_reasons = self.parseThrottleReasons(),
        };
        
        self.current_state = power_state;
    }
    
    fn parseThrottleReasons(self: *Self) ThrottleReason {
        const thermal_reg = self.readRegister(THERMAL_CONTROL);
        const power_reg = self.readRegister(POWER_SENSORS);
        
        return ThrottleReason{
            .thermal = (thermal_reg & 0x100) != 0,
            .power = (power_reg & 0x80000000) != 0,
            .voltage = false, // Would need additional registers
            .current = false,
            .reliability = false,
            .external = false,
        };
    }
    
    fn getCurrentFrequency(self: *Self) u32 {
        if (self.clock_manager) |clock_mgr| {
            return clock_mgr.current_state.base_clock_mhz;
        }
        return 1500; // Default estimate
    }
    
    pub fn setPowerState(self: *Self, target_state: PowerState) !void {
        if (self.current_state == target_state) return;
        
        self.target_state = target_state;
        
        // Perform state-specific transitions
        switch (target_state) {
            .d0 => try self.transitionToD0(),
            .d1 => try self.transitionToD1(),
            .d2 => try self.transitionToD2(),
            .d3_hot => try self.transitionToD3Hot(),
            .d3_cold => try self.transitionToD3Cold(),
            .gc6 => try self.transitionToGC6(),
        }
        
        // Verify transition completed
        var timeout: u32 = 1000;
        while (timeout > 0) {
            try self.readCurrentMetrics();
            if (self.current_state == target_state) break;
            timeout -= 1;
            std.time.sleep(1_000_000); // 1ms
        }
        
        if (timeout == 0) {
            return PowerError.PowerStateTransitionFailed;
        }
    }
    
    fn transitionToD0(self: *Self) !void {
        // Full power state
        self.writeRegister(POWER_STATE_CONTROL, 0x0);
        
        // Restore clocks if clock manager available
        if (self.clock_manager) |clock_mgr| {
            try clock_mgr.setPerformanceLevel(.p0);
        }
        
        // Enable all voltage rails
        try self.setVoltage(.core, self.active_profile.dvfs_points[0].voltage_mv);
    }
    
    fn transitionToD1(self: *Self) !void {
        // Light sleep - reduce clocks but keep context
        self.writeRegister(POWER_STATE_CONTROL, 0x1);
        
        if (self.clock_manager) |clock_mgr| {
            try clock_mgr.setPerformanceLevel(.p4);
        }
        
        // Reduce core voltage
        try self.setVoltage(.core, 800); // 0.8V for idle
    }
    
    fn transitionToD2(self: *Self) !void {
        // Deep sleep - minimal power
        self.writeRegister(POWER_STATE_CONTROL, 0x2);
        
        if (self.clock_manager) |clock_mgr| {
            try clock_mgr.setPerformanceLevel(.p8);
        }
        
        // Minimum voltage
        try self.setVoltage(.core, 700); // 0.7V minimum
    }
    
    fn transitionToD3Hot(self: *Self) !void {
        // System suspend - save context
        self.writeRegister(POWER_STATE_CONTROL, 0x3);
        
        // Power down most subsystems but maintain PCIe link
        if (self.clock_manager) |clock_mgr| {
            try clock_mgr.setPerformanceLevel(.idle);
        }
    }
    
    fn transitionToD3Cold(self: *Self) !void {
        // Complete power off
        self.writeRegister(POWER_STATE_CONTROL, 0x4);
        
        // This would typically require motherboard/PSU coordination
        // Just signal the intent to hardware
    }
    
    fn transitionToGC6(self: *Self) !void {
        // GPU Context Save/Restore state
        // Enable GC6 with context preservation
        self.writeRegister(GC6_CONTROL, 0x3); // Enable GC6 + context save
        self.writeRegister(POWER_STATE_CONTROL, 0x5);
    }
    
    pub fn setVoltage(self: *Self, rail: VoltageRail, voltage_mv: u32) !void {
        // Validate voltage range (typically 0.6V to 1.2V for GPU cores)
        if (voltage_mv < 600 or voltage_mv > 1200) {
            return PowerError.VoltageOutOfRange;
        }
        
        const rail_offset = @intFromEnum(rail) * 4;
        const voltage_reg = self.readRegister(VOLTAGE_CONTROL + rail_offset);
        const new_voltage_reg = (voltage_reg & 0xFFFF0000) | (voltage_mv & 0xFFFF);
        
        self.writeRegister(VOLTAGE_CONTROL + rail_offset, new_voltage_reg);
        
        // Wait for voltage regulator to stabilize
        std.time.sleep(5_000_000); // 5ms
    }
    
    pub fn setPowerProfile(self: *Self, profile: PowerProfile) !void {
        self.active_profile = profile;
        
        // Apply new power limits
        const power_limit_reg = (profile.max_power_w & 0xFFFF) |
                               ((profile.thermal_target_c & 0xFF) << 16);
        self.writeRegister(POWER_LIMITS, power_limit_reg);
        
        // Update fan control
        const fan_control = (profile.min_fan_speed_percent & 0xFF) |
                           (1 << 31);
        self.writeRegister(FAN_CONTROL, fan_control);
        
        // Apply optimal DVFS point for current load
        if (profile.dvfs_points.len > 0) {
            const optimal_point = self.selectOptimalDVFSPoint();
            try self.setVoltage(.core, optimal_point.voltage_mv);
            
            if (self.clock_manager) |clock_mgr| {
                try clock_mgr.setClockFrequency(.graphics_core, optimal_point.frequency_mhz);
            }
        }
    }
    
    fn selectOptimalDVFSPoint(self: *Self) DVFSPoint {
        // Select DVFS point based on current temperature and power budget
        for (self.active_profile.dvfs_points) |point| {
            if (self.current_metrics.temperature_c <= point.temperature_limit_c and
                point.power_w <= @as(f32, @floatFromInt(self.active_profile.max_power_w))) {
                return point;
            }
        }
        
        // Fallback to lowest power point
        return self.active_profile.dvfs_points[self.active_profile.dvfs_points.len - 1];
    }
    
    pub fn handleThermalEmergency(self: *Self, temperature: u8) !void {
        if (temperature >= 95) { // Critical temperature
            // Emergency shutdown
            try self.setPowerState(.d3_hot);
            return PowerError.ThermalEmergency;
        } else if (temperature >= 85) {
            // Aggressive throttling
            try self.setPowerProfile(Self.EFFICIENCY_PROFILE);
            if (self.clock_manager) |clock_mgr| {
                try clock_mgr.setPerformanceLevel(.p8);
            }
        }
    }
    
    pub fn updatePowerManagement(self: *Self) !void {
        try self.readCurrentMetrics();
        
        // Handle thermal conditions
        if (self.current_metrics.temperature_c >= self.active_profile.thermal_target_c) {
            try self.handleThermalEmergency(self.current_metrics.temperature_c);
        }
        
        // Handle power budget
        if (self.current_metrics.total_power_w > @as(f32, @floatFromInt(self.active_profile.max_power_w))) {
            // Power throttling
            const optimal_point = self.selectOptimalDVFSPoint();
            try self.setVoltage(.core, optimal_point.voltage_mv);
            
            if (self.clock_manager) |clock_mgr| {
                try clock_mgr.handlePowerThrottling(@as(u32, @intFromFloat(self.current_metrics.total_power_w)));
            }
        }
    }
    
    pub fn getPowerMetrics(self: *Self) PowerMetrics {
        return self.current_metrics;
    }
    
    pub fn getCurrentPowerState(self: *Self) PowerState {
        return self.current_state;
    }
    
    pub fn getEfficiencyScore(self: *Self) f32 {
        return self.current_metrics.efficiency_percent;
    }
};

pub fn initPowerManager(allocator: Allocator, device: pci.PciDevice, clock_mgr: ?*clock.ClockManager) !PowerManager {
    return PowerManager.init(allocator, device, clock_mgr);
}