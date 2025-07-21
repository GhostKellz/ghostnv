const std = @import("std");
const Allocator = std.mem.Allocator;
const pci = @import("pci.zig");
const power = @import("power.zig");
const clock = @import("clock.zig");
const linux = std.os.linux;

/// GPU Thermal Management Hardware Abstraction Layer
/// Monitors temperatures, manages thermal throttling, and controls cooling

pub const ThermalError = error{
    SensorNotFound,
    SensorReadFailed,
    ThermalEmergency,
    FanControlFailed,
    InvalidTemperature,
    CoolingInsufficient,
    HardwareError,
    OutOfMemory,
    PermissionDenied,
};

pub const ThermalSensor = enum(u8) {
    gpu_core = 0,
    gpu_memory = 1,
    gpu_hotspot = 2,
    vrm_core = 3,
    vrm_memory = 4,
    ambient = 5,
    memory_junction = 6,
    power_supply = 7,
    
    pub fn toString(self: ThermalSensor) []const u8 {
        return switch (self) {
            .gpu_core => "GPU Core",
            .gpu_memory => "GPU Memory",
            .gpu_hotspot => "GPU Hotspot",
            .vrm_core => "VRM Core",
            .vrm_memory => "VRM Memory",
            .ambient => "Ambient",
            .memory_junction => "Memory Junction",
            .power_supply => "Power Supply",
        };
    }
    
    pub fn getCriticalTemp(self: ThermalSensor) u8 {
        return switch (self) {
            .gpu_core => 95,
            .gpu_memory => 105,
            .gpu_hotspot => 100,
            .vrm_core => 125,
            .vrm_memory => 115,
            .ambient => 85,
            .memory_junction => 110,
            .power_supply => 85,
        };
    }
    
    pub fn getThrottleTemp(self: ThermalSensor) u8 {
        return self.getCriticalTemp() - 10;
    }
};

pub const ThermalZone = struct {
    sensor: ThermalSensor,
    current_temp: u8,
    max_temp: u8,
    min_temp: u8,
    average_temp: u8,
    critical_temp: u8,
    throttle_temp: u8,
    hysteresis: u8,
    last_reading_time: u64,
    reading_count: u32,
    temp_history: [16]u8, // Rolling history for averaging
    
    pub fn init(sensor: ThermalSensor) ThermalZone {
        return ThermalZone{
            .sensor = sensor,
            .current_temp = 0,
            .max_temp = 0,
            .min_temp = 255,
            .average_temp = 0,
            .critical_temp = sensor.getCriticalTemp(),
            .throttle_temp = sensor.getThrottleTemp(),
            .hysteresis = 5, // 5°C hysteresis
            .last_reading_time = 0,
            .reading_count = 0,
            .temp_history = [_]u8{0} ** 16,
        };
    }
    
    pub fn updateTemperature(self: *ThermalZone, temp: u8, timestamp: u64) void {
        self.current_temp = temp;
        self.last_reading_time = timestamp;
        
        // Update min/max
        if (temp > self.max_temp) self.max_temp = temp;
        if (temp < self.min_temp) self.min_temp = temp;
        
        // Update rolling history
        const index = self.reading_count % 16;
        self.temp_history[index] = temp;
        self.reading_count += 1;
        
        // Calculate average
        const count = @min(self.reading_count, 16);
        var sum: u32 = 0;
        for (0..count) |i| {
            sum += self.temp_history[i];
        }
        self.average_temp = @as(u8, @truncate(sum / count));
    }
    
    pub fn needsThrottling(self: *ThermalZone) bool {
        return self.current_temp >= self.throttle_temp;
    }
    
    pub fn isEmergency(self: *ThermalZone) bool {
        return self.current_temp >= self.critical_temp;
    }
    
    pub fn canRecoverFromThrottling(self: *ThermalZone) bool {
        return self.current_temp <= (self.throttle_temp - self.hysteresis);
    }
};

pub const FanCurve = struct {
    temperature_points: [8]u8,
    fan_speed_percent: [8]u8,
    
    pub const SILENT_CURVE = FanCurve{
        .temperature_points = [_]u8{ 30, 40, 50, 60, 70, 75, 80, 85 },
        .fan_speed_percent = [_]u8{ 20, 25, 35, 45, 60, 70, 85, 100 },
    };
    
    pub const BALANCED_CURVE = FanCurve{
        .temperature_points = [_]u8{ 30, 45, 55, 65, 72, 78, 83, 88 },
        .fan_speed_percent = [_]u8{ 25, 30, 40, 55, 70, 80, 90, 100 },
    };
    
    pub const AGGRESSIVE_CURVE = FanCurve{
        .temperature_points = [_]u8{ 25, 35, 45, 55, 65, 70, 75, 80 },
        .fan_speed_percent = [_]u8{ 30, 40, 50, 60, 75, 85, 95, 100 },
    };
    
    pub fn getFanSpeed(self: *const FanCurve, temperature: u8) u8 {
        if (temperature <= self.temperature_points[0]) {
            return self.fan_speed_percent[0];
        }
        
        if (temperature >= self.temperature_points[7]) {
            return self.fan_speed_percent[7];
        }
        
        // Linear interpolation between points
        for (0..7) |i| {
            if (temperature <= self.temperature_points[i + 1]) {
                const temp_range = self.temperature_points[i + 1] - self.temperature_points[i];
                const fan_range = self.fan_speed_percent[i + 1] - self.fan_speed_percent[i];
                const temp_offset = temperature - self.temperature_points[i];
                
                const interpolated_fan = self.fan_speed_percent[i] + 
                    @as(u8, @truncate((@as(u32, fan_range) * temp_offset) / temp_range));
                
                return interpolated_fan;
            }
        }
        
        return self.fan_speed_percent[7]; // Fallback to maximum
    }
};

pub const CoolingPolicy = enum(u8) {
    silent = 0,
    balanced = 1,
    performance = 2,
    custom = 3,
    
    pub fn getFanCurve(self: CoolingPolicy) FanCurve {
        return switch (self) {
            .silent => FanCurve.SILENT_CURVE,
            .balanced => FanCurve.BALANCED_CURVE,
            .performance => FanCurve.AGGRESSIVE_CURVE,
            .custom => FanCurve.BALANCED_CURVE, // Default fallback
        };
    }
};

pub const ThermalStatus = struct {
    zones: []ThermalZone,
    overall_temp: u8,
    hottest_zone: ThermalSensor,
    throttling_active: bool,
    emergency_shutdown: bool,
    fan_speed_percent: u8,
    fan_rpm: u16,
    cooling_policy: CoolingPolicy,
    power_limit_active: bool,
    clock_throttle_active: bool,
};

pub const ThermalManager = struct {
    const Self = @This();
    
    allocator: Allocator,
    device: pci.PciDevice,
    zones: std.ArrayList(ThermalZone),
    thermal_status: ThermalStatus,
    cooling_policy: CoolingPolicy,
    custom_fan_curve: FanCurve,
    mmio_base: ?*volatile u8,
    power_manager: ?*power.PowerManager,
    clock_manager: ?*clock.ClockManager,
    monitoring_enabled: bool,
    emergency_callback: ?fn() void,
    
    // Thermal control registers
    const THERMAL_BASE: u32 = 0x139000;
    const THERMAL_SENSORS: u32 = THERMAL_BASE + 0x00;
    const THERMAL_STATUS_REG: u32 = THERMAL_BASE + 0x04;
    const THERMAL_THRESHOLDS: u32 = THERMAL_BASE + 0x08;
    const FAN_CONTROL: u32 = THERMAL_BASE + 0x0C;
    const FAN_TACHOMETER: u32 = THERMAL_BASE + 0x10;
    const THERMAL_POLICY: u32 = THERMAL_BASE + 0x14;
    const THERMAL_EMERGENCY: u32 = THERMAL_BASE + 0x18;
    
    // Sensor register offsets (each sensor has 4 bytes)
    const SENSOR_DATA_SIZE: u32 = 0x04;
    const MAX_SENSORS: u8 = 8;
    
    pub fn init(allocator: Allocator, device: pci.PciDevice, power_mgr: ?*power.PowerManager, clock_mgr: ?*clock.ClockManager) !Self {
        var self = Self{
            .allocator = allocator,
            .device = device,
            .zones = std.ArrayList(ThermalZone).init(allocator),
            .thermal_status = undefined,
            .cooling_policy = .balanced,
            .custom_fan_curve = FanCurve.BALANCED_CURVE,
            .mmio_base = null,
            .power_manager = power_mgr,
            .clock_manager = clock_mgr,
            .monitoring_enabled = false,
            .emergency_callback = null,
        };
        
        try self.mapMemoryRegions();
        try self.initializeThermalZones();
        try self.initializeCoolingSubsystem();
        try self.readThermalStatus();
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        self.monitoring_enabled = false;
        
        // Set fans to safe speed before cleanup
        self.setFanSpeed(50) catch {}; // 50% safe speed
        
        self.zones.deinit();
        
        if (self.mmio_base) |base| {
            _ = linux.munmap(@ptrCast(base), 0x1000000);
        }
    }
    
    fn mapMemoryRegions(self: *Self) !void {
        // Reuse mapping from power/clock managers or create new one
        if (self.power_manager) |power_mgr| {
            self.mmio_base = power_mgr.mmio_base;
        } else if (self.clock_manager) |clock_mgr| {
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
                return ThermalError.PermissionDenied;
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
    
    fn initializeThermalZones(self: *Self) !void {
        // Initialize thermal zones for all available sensors
        const sensor_availability = self.readRegister(THERMAL_SENSORS);
        
        const all_sensors = [_]ThermalSensor{
            .gpu_core, .gpu_memory, .gpu_hotspot, .vrm_core,
            .vrm_memory, .ambient, .memory_junction, .power_supply
        };
        
        for (all_sensors, 0..) |sensor, i| {
            // Check if sensor is available
            if ((sensor_availability & (@as(u32, 1) << @truncate(i))) != 0) {
                var zone = ThermalZone.init(sensor);
                try self.zones.append(zone);
            }
        }
        
        if (self.zones.items.len == 0) {
            return ThermalError.SensorNotFound;
        }
        
        // Set thermal thresholds in hardware
        for (self.zones.items, 0..) |zone, i| {
            const threshold_reg = (@as(u32, zone.critical_temp) << 16) | zone.throttle_temp;
            self.writeRegister(THERMAL_THRESHOLDS + @as(u32, @truncate(i)) * 4, threshold_reg);
        }
    }
    
    fn initializeCoolingSubsystem(self: *Self) !void {
        // Initialize fan control
        const fan_control_reg = (1 << 31) | // Enable automatic fan control
                               (@as(u32, @intFromEnum(self.cooling_policy)) << 24) |
                               (50); // 50% initial fan speed
        self.writeRegister(FAN_CONTROL, fan_control_reg);
        
        // Set thermal policy
        self.writeRegister(THERMAL_POLICY, @intFromEnum(self.cooling_policy));
        
        // Enable thermal monitoring
        self.writeRegister(THERMAL_STATUS_REG, 0x1);
        
        self.monitoring_enabled = true;
    }
    
    pub fn readThermalStatus(self: *Self) !void {
        const timestamp = std.time.microTimestamp();
        
        // Read all sensor temperatures
        for (self.zones.items, 0..) |*zone, i| {
            const sensor_reg = self.readRegister(THERMAL_SENSORS + (@as(u32, @truncate(i)) + 1) * SENSOR_DATA_SIZE);
            const temperature = @as(u8, @truncate(sensor_reg & 0xFF));
            
            if (temperature > 0 and temperature < 200) { // Sanity check
                zone.updateTemperature(temperature, @intCast(timestamp));
            } else {
                return ThermalError.SensorReadFailed;
            }
        }
        
        // Find hottest zone and calculate overall temperature
        var hottest_temp: u8 = 0;
        var hottest_sensor: ThermalSensor = .gpu_core;
        var total_temp: u32 = 0;
        
        for (self.zones.items) |zone| {
            total_temp += zone.current_temp;
            if (zone.current_temp > hottest_temp) {
                hottest_temp = zone.current_temp;
                hottest_sensor = zone.sensor;
            }
        }
        
        // Read fan status
        const fan_status = self.readRegister(FAN_TACHOMETER);
        const fan_rpm = @as(u16, @truncate((fan_status >> 16) & 0xFFFF));
        const fan_percent = @as(u8, @truncate(fan_status & 0xFF));
        
        // Read thermal control status
        const thermal_reg = self.readRegister(THERMAL_STATUS_REG);
        const throttling = (thermal_reg & 0x2) != 0;
        const emergency = (thermal_reg & 0x4) != 0;
        
        self.thermal_status = ThermalStatus{
            .zones = self.zones.items,
            .overall_temp = @as(u8, @truncate(total_temp / @as(u32, @truncate(self.zones.items.len)))),
            .hottest_zone = hottest_sensor,
            .throttling_active = throttling,
            .emergency_shutdown = emergency,
            .fan_speed_percent = fan_percent,
            .fan_rpm = fan_rpm,
            .cooling_policy = self.cooling_policy,
            .power_limit_active = false, // Will be updated by power manager
            .clock_throttle_active = false, // Will be updated by clock manager
        };
    }
    
    pub fn updateThermalManagement(self: *Self) !void {
        try self.readThermalStatus();
        
        // Check for emergency conditions
        for (self.zones.items) |zone| {
            if (zone.isEmergency()) {
                try self.handleThermalEmergency(zone);
                return;
            }
        }
        
        // Check for throttling conditions
        var needs_throttling = false;
        for (self.zones.items) |zone| {
            if (zone.needsThrottling()) {
                needs_throttling = true;
                break;
            }
        }
        
        if (needs_throttling and !self.thermal_status.throttling_active) {
            try self.enableThermalThrottling();
        } else if (!needs_throttling and self.thermal_status.throttling_active) {
            // Check if we can recover from throttling
            var can_recover = true;
            for (self.zones.items) |zone| {
                if (!zone.canRecoverFromThrottling()) {
                    can_recover = false;
                    break;
                }
            }
            if (can_recover) {
                try self.disableThermalThrottling();
            }
        }
        
        // Update fan speed based on temperature
        try self.updateFanSpeed();
    }
    
    fn handleThermalEmergency(self: *Self, zone: ThermalZone) !void {
        // Set emergency flag
        self.writeRegister(THERMAL_EMERGENCY, 0x1);
        
        // Maximum fan speed
        try self.setFanSpeed(100);
        
        // Emergency throttling
        if (self.power_manager) |power_mgr| {
            try power_mgr.handleThermalEmergency(zone.current_temp);
        }
        
        if (self.clock_manager) |clock_mgr| {
            try clock_mgr.handleThermalThrottling(zone.current_temp);
        }
        
        // Call emergency callback if set
        if (self.emergency_callback) |callback| {
            callback();
        }
        
        // For critical temperatures, initiate shutdown
        if (zone.current_temp >= 100) {
            return ThermalError.ThermalEmergency;
        }
    }
    
    fn enableThermalThrottling(self: *Self) !void {
        // Set throttling flag
        const status_reg = self.readRegister(THERMAL_STATUS_REG);
        self.writeRegister(THERMAL_STATUS_REG, status_reg | 0x2);
        
        // Coordinate with power and clock managers
        if (self.power_manager) |power_mgr| {
            try power_mgr.setPowerProfile(power.PowerManager.EFFICIENCY_PROFILE);
        }
        
        if (self.clock_manager) |clock_mgr| {
            try clock_mgr.setPerformanceLevel(.p4); // Reduce to 75% performance
        }
        
        // Increase fan speed
        const hottest_temp = self.getHottestTemperature();
        const fan_curve = self.cooling_policy.getFanCurve();
        const target_fan_speed = fan_curve.getFanSpeed(hottest_temp + 10); // +10°C offset for cooling
        try self.setFanSpeed(target_fan_speed);
    }
    
    fn disableThermalThrottling(self: *Self) !void {
        // Clear throttling flag
        const status_reg = self.readRegister(THERMAL_STATUS_REG);
        self.writeRegister(THERMAL_STATUS_REG, status_reg & ~@as(u32, 0x2));
        
        // Restore normal operation
        if (self.power_manager) |power_mgr| {
            try power_mgr.setPowerProfile(power.PowerManager.BALANCED_PROFILE);
        }
        
        if (self.clock_manager) |clock_mgr| {
            try clock_mgr.setPerformanceLevel(.p0);
        }
    }
    
    fn updateFanSpeed(self: *Self) !void {
        const hottest_temp = self.getHottestTemperature();
        const fan_curve = if (self.cooling_policy == .custom) 
            self.custom_fan_curve 
        else 
            self.cooling_policy.getFanCurve();
            
        const target_speed = fan_curve.getFanSpeed(hottest_temp);
        
        // Only update if speed change is significant (>5% difference)
        const current_speed = self.thermal_status.fan_speed_percent;
        if (@abs(@as(i16, target_speed) - @as(i16, current_speed)) > 5) {
            try self.setFanSpeed(target_speed);
        }
    }
    
    pub fn setFanSpeed(self: *Self, speed_percent: u8) !void {
        const clamped_speed = @min(speed_percent, 100);
        
        const fan_control = (1 << 31) | // Keep automatic control enabled
                           (@as(u32, @intFromEnum(self.cooling_policy)) << 24) |
                           clamped_speed;
        
        self.writeRegister(FAN_CONTROL, fan_control);
        
        // Verify fan response
        std.time.sleep(100_000_000); // 100ms for fan to respond
        
        const fan_status = self.readRegister(FAN_TACHOMETER);
        const actual_speed = @as(u8, @truncate(fan_status & 0xFF));
        
        // Allow 10% tolerance for fan speed
        if (@abs(@as(i16, actual_speed) - @as(i16, clamped_speed)) > 10) {
            return ThermalError.FanControlFailed;
        }
    }
    
    pub fn setCoolingPolicy(self: *Self, policy: CoolingPolicy) !void {
        self.cooling_policy = policy;
        
        // Update hardware policy register
        self.writeRegister(THERMAL_POLICY, @intFromEnum(policy));
        
        // Update fan control with new policy
        const fan_control = (1 << 31) |
                           (@as(u32, @intFromEnum(policy)) << 24) |
                           self.thermal_status.fan_speed_percent;
        self.writeRegister(FAN_CONTROL, fan_control);
    }
    
    pub fn setCustomFanCurve(self: *Self, curve: FanCurve) !void {
        self.custom_fan_curve = curve;
        try self.setCoolingPolicy(.custom);
    }
    
    pub fn setEmergencyCallback(self: *Self, callback: fn() void) void {
        self.emergency_callback = callback;
    }
    
    pub fn getTemperature(self: *Self, sensor: ThermalSensor) ?u8 {
        for (self.zones.items) |zone| {
            if (zone.sensor == sensor) {
                return zone.current_temp;
            }
        }
        return null;
    }
    
    pub fn getHottestTemperature(self: *Self) u8 {
        var hottest: u8 = 0;
        for (self.zones.items) |zone| {
            if (zone.current_temp > hottest) {
                hottest = zone.current_temp;
            }
        }
        return hottest;
    }
    
    pub fn getThermalStatus(self: *Self) ThermalStatus {
        return self.thermal_status;
    }
    
    pub fn isThrottling(self: *Self) bool {
        return self.thermal_status.throttling_active;
    }
    
    pub fn getAverageTemperature(self: *Self, sensor: ThermalSensor) ?u8 {
        for (self.zones.items) |zone| {
            if (zone.sensor == sensor) {
                return zone.average_temp;
            }
        }
        return null;
    }
    
    pub fn getThermalMargin(self: *Self, sensor: ThermalSensor) ?i8 {
        for (self.zones.items) |zone| {
            if (zone.sensor == sensor) {
                return @as(i8, @intCast(zone.throttle_temp)) - @as(i8, @intCast(zone.current_temp));
            }
        }
        return null;
    }
    
    pub fn resetThermalHistory(self: *Self) void {
        for (self.zones.items) |*zone| {
            zone.max_temp = zone.current_temp;
            zone.min_temp = zone.current_temp;
            zone.reading_count = 1;
            zone.temp_history = [_]u8{zone.current_temp} ** 16;
        }
    }
};

pub fn initThermalManager(allocator: Allocator, device: pci.PciDevice, power_mgr: ?*power.PowerManager, clock_mgr: ?*clock.ClockManager) !ThermalManager {
    return ThermalManager.init(allocator, device, power_mgr, clock_mgr);
}