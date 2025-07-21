const std = @import("std");
const Allocator = std.mem.Allocator;
const pci = @import("pci.zig");
const linux = std.os.linux;

/// GPU GPIO and I2C Hardware Abstraction Layer
/// Manages GPIO pins, I2C buses, and external device communication

pub const GpioError = error{
    PinNotFound,
    InvalidDirection,
    InvalidValue,
    I2cBusError,
    I2cTimeout,
    DeviceNotResponding,
    HardwareError,
    PermissionDenied,
    OutOfMemory,
    InvalidAddress,
};

pub const GpioPin = enum(u8) {
    // Display connector pins
    ddc_sda = 0,        // Display Data Channel SDA
    ddc_scl = 1,        // Display Data Channel SCL
    hpd_0 = 2,          // Hot Plug Detect 0
    hpd_1 = 3,          // Hot Plug Detect 1
    hpd_2 = 4,          // Hot Plug Detect 2
    hpd_3 = 5,          // Hot Plug Detect 3
    
    // Power management
    power_good = 8,     // Power good signal
    power_enable = 9,   // Power enable control
    reset_n = 10,       // Reset signal (active low)
    
    // Fan control
    fan_tach_0 = 16,    // Fan tachometer 0
    fan_tach_1 = 17,    // Fan tachometer 1
    fan_pwm_0 = 18,     // Fan PWM control 0
    fan_pwm_1 = 19,     // Fan PWM control 1
    
    // LED control
    led_power = 24,     // Power LED
    led_activity = 25,  // Activity LED
    
    // General purpose
    gpio_0 = 32,        // General purpose GPIO 0
    gpio_1 = 33,        // General purpose GPIO 1
    gpio_2 = 34,        // General purpose GPIO 2
    gpio_3 = 35,        // General purpose GPIO 3
    
    pub fn toString(self: GpioPin) []const u8 {
        return switch (self) {
            .ddc_sda => "DDC SDA",
            .ddc_scl => "DDC SCL",
            .hpd_0 => "HPD 0",
            .hpd_1 => "HPD 1",
            .hpd_2 => "HPD 2",
            .hpd_3 => "HPD 3",
            .power_good => "Power Good",
            .power_enable => "Power Enable",
            .reset_n => "Reset#",
            .fan_tach_0 => "Fan Tach 0",
            .fan_tach_1 => "Fan Tach 1",
            .fan_pwm_0 => "Fan PWM 0",
            .fan_pwm_1 => "Fan PWM 1",
            .led_power => "Power LED",
            .led_activity => "Activity LED",
            .gpio_0 => "GPIO 0",
            .gpio_1 => "GPIO 1",
            .gpio_2 => "GPIO 2",
            .gpio_3 => "GPIO 3",
        };
    }
};

pub const GpioDirection = enum(u8) {
    input = 0,
    output = 1,
    
    pub fn toString(self: GpioDirection) []const u8 {
        return switch (self) {
            .input => "Input",
            .output => "Output",
        };
    }
};

pub const GpioValue = enum(u8) {
    low = 0,
    high = 1,
    
    pub fn toBool(self: GpioValue) bool {
        return self == .high;
    }
    
    pub fn fromBool(value: bool) GpioValue {
        return if (value) .high else .low;
    }
};

pub const I2cBus = enum(u8) {
    ddc_0 = 0,          // Display Data Channel 0
    ddc_1 = 1,          // Display Data Channel 1
    ddc_2 = 2,          // Display Data Channel 2
    ddc_3 = 3,          // Display Data Channel 3
    pmbus = 4,          // Power Management Bus
    thermal = 5,        // Thermal sensors
    eeprom = 6,         // EEPROM/SPD
    general = 7,        // General purpose I2C
    
    pub fn toString(self: I2cBus) []const u8 {
        return switch (self) {
            .ddc_0 => "DDC 0",
            .ddc_1 => "DDC 1", 
            .ddc_2 => "DDC 2",
            .ddc_3 => "DDC 3",
            .pmbus => "PMBus",
            .thermal => "Thermal",
            .eeprom => "EEPROM",
            .general => "General",
        };
    }
    
    pub fn getDefaultSpeed(self: I2cBus) u32 {
        return switch (self) {
            .ddc_0, .ddc_1, .ddc_2, .ddc_3=> 100_000, // 100kHz for DDC
            .pmbus => 400_000,                          // 400kHz for PMBus
            .thermal => 400_000,                        // 400kHz for thermal sensors
            .eeprom => 100_000,                         // 100kHz for EEPROM
            .general => 100_000,                        // 100kHz default
        };
    }
};

pub const I2cDevice = struct {
    bus: I2cBus,
    address: u8,        // 7-bit I2C address
    speed_hz: u32,
    name: []const u8,
    
    pub fn init(bus: I2cBus, address: u8, name: []const u8) I2cDevice {
        return I2cDevice{
            .bus = bus,
            .address = address,
            .speed_hz = bus.getDefaultSpeed(),
            .name = name,
        };
    }
};

pub const EdidData = struct {
    header: [8]u8,
    manufacturer_id: [2]u8,
    product_code: [2]u8,
    serial_number: [4]u8,
    week_of_manufacture: u8,
    year_of_manufacture: u8,
    edid_version: u8,
    edid_revision: u8,
    display_params: [5]u8,
    chromaticity: [10]u8,
    established_timings: [3]u8,
    standard_timings: [16]u8,
    detailed_timings: [72]u8,
    extension_flag: u8,
    checksum: u8,
    
    pub fn isValid(self: *const EdidData) bool {
        // Check EDID header
        const expected_header = [_]u8{ 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00 };
        return std.mem.eql(u8, &self.header, &expected_header);
    }
    
    pub fn calculateChecksum(self: *const EdidData) u8 {
        const data: [*]const u8 = @ptrCast(self);
        var sum: u32 = 0;
        for (0..127) |i| {
            sum += data[i];
        }
        return @as(u8, @truncate(256 - (sum % 256)));
    }
    
    pub fn verifyChecksum(self: *const EdidData) bool {
        return self.calculateChecksum() == self.checksum;
    }
};

pub const GpioManager = struct {
    const Self = @This();
    
    allocator: Allocator,
    device: pci.PciDevice,
    mmio_base: ?*volatile u8,
    pin_directions: [64]GpioDirection, // Support up to 64 GPIO pins
    pin_values: [64]GpioValue,
    i2c_devices: std.ArrayList(I2cDevice),
    
    // GPIO control registers
    const GPIO_BASE: u32 = 0x13A000;
    const GPIO_DIRECTION: u32 = GPIO_BASE + 0x00;  // Direction control (2 registers for 64 pins)
    const GPIO_OUTPUT: u32 = GPIO_BASE + 0x08;     // Output values (2 registers)
    const GPIO_INPUT: u32 = GPIO_BASE + 0x10;      // Input values (2 registers)
    const GPIO_ENABLE: u32 = GPIO_BASE + 0x18;     // Pin enable (2 registers)
    
    // I2C control registers (8 buses, each with 16-byte register space)
    const I2C_BASE: u32 = 0x13B000;
    const I2C_CONTROL: u32 = 0x00;      // Control register offset
    const I2C_STATUS: u32 = 0x04;       // Status register offset
    const I2C_DATA: u32 = 0x08;         // Data register offset
    const I2C_ADDRESS: u32 = 0x0C;      // Address register offset
    
    pub fn init(allocator: Allocator, device: pci.PciDevice) !Self {
        var self = Self{
            .allocator = allocator,
            .device = device,
            .mmio_base = null,
            .pin_directions = [_]GpioDirection{.input} ** 64,
            .pin_values = [_]GpioValue{.low} ** 64,
            .i2c_devices = std.ArrayList(I2cDevice).init(allocator),
        };
        
        try self.mapMemoryRegions();
        try self.initializeGpio();
        try self.initializeI2c();
        try self.discoverI2cDevices();
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        // Set all output pins to safe states
        for (0..64) |pin| {
            if (self.pin_directions[pin] == .output) {
                self.setGpioValue(@enumFromInt(pin), .low) catch {};
            }
        }
        
        self.i2c_devices.deinit();
        
        if (self.mmio_base) |base| {
            _ = linux.munmap(@ptrCast(base), 0x1000000);
        }
    }
    
    fn mapMemoryRegions(self: *Self) !void {
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
            return GpioError.PermissionDenied;
        }
        
        self.mmio_base = @ptrCast(@alignCast(mmio_ptr));
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
    
    fn initializeGpio(self: *Self) !void {
        // Read current GPIO configuration from hardware
        const direction_low = self.readRegister(GPIO_DIRECTION);
        const direction_high = self.readRegister(GPIO_DIRECTION + 4);
        const input_low = self.readRegister(GPIO_INPUT);
        const input_high = self.readRegister(GPIO_INPUT + 4);
        
        // Update internal state
        for (0..32) |i| {
            const pin_bit = @as(u32, 1) << @truncate(i);
            self.pin_directions[i] = if ((direction_low & pin_bit) != 0) .output else .input;
            self.pin_values[i] = if ((input_low & pin_bit) != 0) .high else .low;
        }
        
        for (32..64) |i| {
            const pin_bit = @as(u32, 1) << @truncate(i - 32);
            self.pin_directions[i] = if ((direction_high & pin_bit) != 0) .output else .input;
            self.pin_values[i] = if ((input_high & pin_bit) != 0) .high else .low;
        }
        
        // Enable commonly used GPIO pins
        const common_pins = [_]GpioPin{
            .hpd_0, .hpd_1, .hpd_2, .hpd_3,
            .power_good, .led_power, .led_activity
        };
        
        for (common_pins) |pin| {
            try self.enableGpioPin(pin);
        }
    }
    
    fn initializeI2c(self: *Self) !void {
        // Initialize all I2C buses
        for (0..8) |bus_id| {
            const bus_base = I2C_BASE + @as(u32, @truncate(bus_id)) * 0x10;
            
            // Reset bus
            self.writeRegister(bus_base + I2C_CONTROL, 0x80); // Reset bit
            std.time.sleep(1_000_000); // 1ms
            
            // Configure bus speed (default 100kHz)
            const speed_config = 100_000 / 1000; // Simplified speed calculation
            self.writeRegister(bus_base + I2C_CONTROL, speed_config | 0x1); // Enable bit
        }
    }
    
    fn discoverI2cDevices(self: *Self) !void {
        // Discover common I2C devices
        const common_devices = [_]I2cDevice{
            I2cDevice.init(.ddc_0, 0x50, "EDID EEPROM 0"),
            I2cDevice.init(.ddc_1, 0x50, "EDID EEPROM 1"),
            I2cDevice.init(.ddc_2, 0x50, "EDID EEPROM 2"),
            I2cDevice.init(.ddc_3, 0x50, "EDID EEPROM 3"),
            I2cDevice.init(.thermal, 0x48, "Temperature Sensor"),
            I2cDevice.init(.thermal, 0x49, "Temperature Sensor 2"),
            I2cDevice.init(.pmbus, 0x58, "Power Management IC"),
            I2cDevice.init(.eeprom, 0x57, "Configuration EEPROM"),
        };
        
        for (common_devices) |device| {
            // Test if device responds
            if (self.i2cPing(device.bus, device.address)) {
                try self.i2c_devices.append(device);
            }
        }
    }
    
    pub fn enableGpioPin(self: *Self, pin: GpioPin) !void {
        const pin_num = @intFromEnum(pin);
        if (pin_num >= 64) return GpioError.PinNotFound;
        
        const reg_offset = if (pin_num < 32) GPIO_ENABLE else GPIO_ENABLE + 4;
        const bit_offset = @truncate(pin_num % 32);
        
        const current_val = self.readRegister(reg_offset);
        const new_val = current_val | (@as(u32, 1) << bit_offset);
        self.writeRegister(reg_offset, new_val);
    }
    
    pub fn setGpioDirection(self: *Self, pin: GpioPin, direction: GpioDirection) !void {
        const pin_num = @intFromEnum(pin);
        if (pin_num >= 64) return GpioError.PinNotFound;
        
        const reg_offset = if (pin_num < 32) GPIO_DIRECTION else GPIO_DIRECTION + 4;
        const bit_offset = @truncate(pin_num % 32);
        
        const current_val = self.readRegister(reg_offset);
        const new_val = if (direction == .output)
            current_val | (@as(u32, 1) << bit_offset)
        else
            current_val & ~(@as(u32, 1) << bit_offset);
        
        self.writeRegister(reg_offset, new_val);
        self.pin_directions[pin_num] = direction;
    }
    
    pub fn setGpioValue(self: *Self, pin: GpioPin, value: GpioValue) !void {
        const pin_num = @intFromEnum(pin);
        if (pin_num >= 64) return GpioError.PinNotFound;
        
        if (self.pin_directions[pin_num] != .output) {
            return GpioError.InvalidDirection;
        }
        
        const reg_offset = if (pin_num < 32) GPIO_OUTPUT else GPIO_OUTPUT + 4;
        const bit_offset = @truncate(pin_num % 32);
        
        const current_val = self.readRegister(reg_offset);
        const new_val = if (value == .high)
            current_val | (@as(u32, 1) << bit_offset)
        else
            current_val & ~(@as(u32, 1) << bit_offset);
        
        self.writeRegister(reg_offset, new_val);
        self.pin_values[pin_num] = value;
    }
    
    pub fn getGpioValue(self: *Self, pin: GpioPin) !GpioValue {
        const pin_num = @intFromEnum(pin);
        if (pin_num >= 64) return GpioError.PinNotFound;
        
        const reg_offset = if (pin_num < 32) GPIO_INPUT else GPIO_INPUT + 4;
        const bit_offset = @truncate(pin_num % 32);
        
        const current_val = self.readRegister(reg_offset);
        const pin_state = (current_val & (@as(u32, 1) << bit_offset)) != 0;
        
        const value = GpioValue.fromBool(pin_state);
        self.pin_values[pin_num] = value;
        return value;
    }
    
    pub fn getHotPlugDetect(self: *Self, connector: u8) !bool {
        const pin = switch (connector) {
            0 => GpioPin.hpd_0,
            1 => GpioPin.hpd_1,
            2 => GpioPin.hpd_2,
            3 => GpioPin.hpd_3,
            else => return GpioError.PinNotFound,
        };
        
        const value = try self.getGpioValue(pin);
        return value.toBool();
    }
    
    fn getBusRegisterBase(self: *Self, bus: I2cBus) u32 {
        return I2C_BASE + @as(u32, @intFromEnum(bus)) * 0x10;
    }
    
    fn i2cWaitForCompletion(self: *Self, bus: I2cBus) !void {
        const bus_base = self.getBusRegisterBase(bus);
        var timeout: u32 = 1000;
        
        while (timeout > 0) {
            const status = self.readRegister(bus_base + I2C_STATUS);
            if ((status & 0x1) == 0) return; // Transaction complete
            if ((status & 0x2) != 0) return GpioError.I2cBusError; // Error occurred
            
            timeout -= 1;
            std.time.sleep(1000); // 1μs
        }
        
        return GpioError.I2cTimeout;
    }
    
    pub fn i2cPing(self: *Self, bus: I2cBus, device_addr: u8) bool {
        const bus_base = self.getBusRegisterBase(bus);
        
        // Set device address
        self.writeRegister(bus_base + I2C_ADDRESS, @as(u32, device_addr) << 1);
        
        // Send ping (0-byte write)
        self.writeRegister(bus_base + I2C_CONTROL, 0x3); // Start + Write
        
        // Check for ACK
        std.time.sleep(10_000); // 10μs
        const status = self.readRegister(bus_base + I2C_STATUS);
        
        return (status & 0x8) == 0; // No NACK
    }
    
    pub fn i2cWrite(self: *Self, bus: I2cBus, device_addr: u8, register: u8, data: []const u8) !void {
        if (data.len == 0 or data.len > 32) return GpioError.InvalidAddress;
        
        const bus_base = self.getBusRegisterBase(bus);
        
        // Set device address
        self.writeRegister(bus_base + I2C_ADDRESS, (@as(u32, device_addr) << 1));
        
        // Write register address
        self.writeRegister(bus_base + I2C_DATA, register);
        self.writeRegister(bus_base + I2C_CONTROL, 0x3); // Start + Write
        try self.i2cWaitForCompletion(bus);
        
        // Write data bytes
        for (data) |byte| {
            self.writeRegister(bus_base + I2C_DATA, byte);
            self.writeRegister(bus_base + I2C_CONTROL, 0x1); // Continue write
            try self.i2cWaitForCompletion(bus);
        }
        
        // Send stop condition
        self.writeRegister(bus_base + I2C_CONTROL, 0x4); // Stop
    }
    
    pub fn i2cRead(self: *Self, bus: I2cBus, device_addr: u8, register: u8, data: []u8) !void {
        if (data.len == 0 or data.len > 32) return GpioError.InvalidAddress;
        
        const bus_base = self.getBusRegisterBase(bus);
        
        // Set device address for write (to send register address)
        self.writeRegister(bus_base + I2C_ADDRESS, (@as(u32, device_addr) << 1));
        self.writeRegister(bus_base + I2C_DATA, register);
        self.writeRegister(bus_base + I2C_CONTROL, 0x3); // Start + Write
        try self.i2cWaitForCompletion(bus);
        
        // Restart with read
        self.writeRegister(bus_base + I2C_ADDRESS, (@as(u32, device_addr) << 1) | 0x1);
        self.writeRegister(bus_base + I2C_CONTROL, 0x5); // Restart + Read
        try self.i2cWaitForCompletion(bus);
        
        // Read data bytes
        for (data, 0..) |*byte, i| {
            if (i == data.len - 1) {
                self.writeRegister(bus_base + I2C_CONTROL, 0x6); // Read + NACK + Stop
            } else {
                self.writeRegister(bus_base + I2C_CONTROL, 0x2); // Read + ACK
            }
            try self.i2cWaitForCompletion(bus);
            
            const data_reg = self.readRegister(bus_base + I2C_DATA);
            byte.* = @as(u8, @truncate(data_reg & 0xFF));
        }
    }
    
    pub fn readEdid(self: *Self, connector: u8) !EdidData {
        const bus = switch (connector) {
            0 => I2cBus.ddc_0,
            1 => I2cBus.ddc_1,
            2 => I2cBus.ddc_2,
            3 => I2cBus.ddc_3,
            else => return GpioError.InvalidAddress,
        };
        
        // Check if display is connected
        const hpd = try self.getHotPlugDetect(connector);
        if (!hpd) {
            return GpioError.DeviceNotResponding;
        }
        
        // Read EDID data (128 bytes)
        var edid_bytes: [128]u8 = undefined;
        
        // EDID is typically at I2C address 0x50
        for (0..8) |block| { // Read in 16-byte blocks
            const start_addr = @as(u8, @truncate(block * 16));
            try self.i2cRead(bus, 0x50, start_addr, edid_bytes[block * 16..(block + 1) * 16]);
        }
        
        // Parse EDID structure
        const edid: *EdidData = @ptrCast(@alignCast(&edid_bytes));
        
        if (!edid.isValid()) {
            return GpioError.DeviceNotResponding;
        }
        
        if (!edid.verifyChecksum()) {
            return GpioError.I2cBusError;
        }
        
        return edid.*;
    }
    
    pub fn setLed(self: *Self, led: GpioPin, on: bool) !void {
        if (led != .led_power and led != .led_activity) {
            return GpioError.InvalidValue;
        }
        
        try self.setGpioDirection(led, .output);
        try self.setGpioValue(led, GpioValue.fromBool(on));
    }
    
    pub fn getPowerGood(self: *Self) !bool {
        const value = try self.getGpioValue(.power_good);
        return value.toBool();
    }
    
    pub fn getI2cDevices(self: *Self) []const I2cDevice {
        return self.i2c_devices.items;
    }
    
    pub fn addI2cDevice(self: *Self, device: I2cDevice) !void {
        // Verify device responds
        if (!self.i2cPing(device.bus, device.address)) {
            return GpioError.DeviceNotResponding;
        }
        
        try self.i2c_devices.append(device);
    }
    
    pub fn findI2cDevice(self: *Self, name: []const u8) ?I2cDevice {
        for (self.i2c_devices.items) |device| {
            if (std.mem.eql(u8, device.name, name)) {
                return device;
            }
        }
        return null;
    }
    
    pub fn scanI2cBus(self: *Self, bus: I2cBus, found_addresses: *[128]u8) u8 {
        var count: u8 = 0;
        
        for (0x08..0x78) |addr| { // Standard I2C address range
            if (self.i2cPing(bus, @as(u8, @truncate(addr)))) {
                found_addresses[count] = @as(u8, @truncate(addr));
                count += 1;
                if (count >= 128) break;
            }
        }
        
        return count;
    }
};

pub fn initGpioManager(allocator: Allocator, device: pci.PciDevice) !GpioManager {
    return GpioManager.init(allocator, device);
}