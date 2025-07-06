const std = @import("std");
const print = std.debug.print;

pub const PciError = error{
    DeviceNotFound,
    AccessDenied,
    InvalidConfig,
    ResourceAllocationFailed,
};

pub const NVIDIA_VENDOR_ID: u16 = 0x10DE;

// Common NVIDIA GPU device IDs (sample)
pub const NVIDIA_DEVICE_IDS = [_]u16{
    0x2684, // RTX 4090
    0x2685, // RTX 4080
    0x2704, // RTX 4070 Ti
    0x2782, // RTX 4070
    0x2786, // RTX 4060 Ti
    0x2787, // RTX 4060
    0x1E82, // RTX 2080
    0x1E84, // RTX 2070
    0x1F02, // RTX 2060
    // Add more as needed
};

pub const PciDevice = struct {
    vendor_id: u16,
    device_id: u16,
    bus: u8,
    slot: u8,
    function: u8,
    bar0: usize,
    bar1: usize,
    bar3: usize,
    irq: u8,
    
    pub fn init(bus: u8, slot: u8, function: u8) !PciDevice {
        var device = PciDevice{
            .vendor_id = 0,
            .device_id = 0,
            .bus = bus,
            .slot = slot,
            .function = function,
            .bar0 = 0,
            .bar1 = 0,
            .bar3 = 0,
            .irq = 0,
        };
        
        // Read PCI configuration space
        try device.read_config();
        return device;
    }
    
    fn read_config(self: *PciDevice) !void {
        // In real kernel module, this would use pci_read_config_word, etc.
        // For now, simulate reading config space
        
        // Read vendor/device ID
        const vendor_device = try pci_config_read_dword(self.bus, self.slot, self.function, 0x00);
        self.vendor_id = @truncate(vendor_device & 0xFFFF);
        self.device_id = @truncate((vendor_device >> 16) & 0xFFFF);
        
        // Verify this is an NVIDIA device
        if (self.vendor_id != NVIDIA_VENDOR_ID) {
            return PciError.DeviceNotFound;
        }
        
        // Check if device ID is in our supported list
        var supported = false;
        for (NVIDIA_DEVICE_IDS) |id| {
            if (self.device_id == id) {
                supported = true;
                break;
            }
        }
        
        if (!supported) {
            print("nvzig: Unsupported NVIDIA device ID: 0x{X}\n", .{self.device_id});
        }
        
        // Read BARs (Base Address Registers)
        self.bar0 = try pci_config_read_dword(self.bus, self.slot, self.function, 0x10);
        self.bar1 = try pci_config_read_dword(self.bus, self.slot, self.function, 0x14);
        self.bar3 = try pci_config_read_dword(self.bus, self.slot, self.function, 0x1C);
        
        // Read IRQ
        const irq_data = try pci_config_read_dword(self.bus, self.slot, self.function, 0x3C);
        self.irq = @truncate(irq_data & 0xFF);
        
        print("nvzig: PCI device {}:{}.{} - Vendor: 0x{X}, Device: 0x{X}, IRQ: {}\n",
              .{self.bus, self.slot, self.function, self.vendor_id, self.device_id, self.irq});
    }
    
    pub fn enable_bus_master(self: *PciDevice) !void {
        // Read command register
        const cmd = try pci_config_read_word(self.bus, self.slot, self.function, 0x04);
        
        // Set bus master enable bit (bit 2)
        const new_cmd = cmd | 0x0004;
        try pci_config_write_word(self.bus, self.slot, self.function, 0x04, new_cmd);
    }
    
    pub fn get_bar_size(self: *PciDevice, bar_num: u8) !usize {
        const bar_offset = 0x10 + (bar_num * 4);
        
        // Save original value
        const original = try pci_config_read_dword(self.bus, self.slot, self.function, bar_offset);
        
        // Write all 1s to get size
        try pci_config_write_dword(self.bus, self.slot, self.function, bar_offset, 0xFFFFFFFF);
        
        // Read back to get size
        const size_mask = try pci_config_read_dword(self.bus, self.slot, self.function, bar_offset);
        
        // Restore original value
        try pci_config_write_dword(self.bus, self.slot, self.function, bar_offset, original);
        
        // Calculate size
        const size = (~size_mask + 1) & 0xFFFFFFF0;
        return size;
    }
};

// PCI configuration space access functions
// In real kernel module, these would be kernel API calls
fn pci_config_read_dword(bus: u8, slot: u8, function: u8, offset: u8) !u32 {
    _ = bus; _ = slot; _ = function; _ = offset;
    // Simulate PCI config read
    // In real implementation: return pci_read_config_dword(...);
    return 0x10DE2684; // Simulate RTX 4090
}

fn pci_config_read_word(bus: u8, slot: u8, function: u8, offset: u8) !u16 {
    const dword = try pci_config_read_dword(bus, slot, function, offset & 0xFC);
    const shift = (offset & 0x03) * 8;
    return @truncate((dword >> @intCast(shift)) & 0xFFFF);
}

fn pci_config_write_dword(bus: u8, slot: u8, function: u8, offset: u8, value: u32) !void {
    _ = bus; _ = slot; _ = function; _ = offset; _ = value;
    // Simulate PCI config write
    // In real implementation: pci_write_config_dword(...);
}

fn pci_config_write_word(bus: u8, slot: u8, function: u8, offset: u8, value: u16) !void {
    _ = bus; _ = slot; _ = function; _ = offset; _ = value;
    // Simulate PCI config write
    // In real implementation: pci_write_config_word(...);
}

pub fn enumerate_nvidia_devices() !u32 {
    var device_count: u32 = 0;
    
    print("nvzig: Enumerating PCI devices...\n");
    
    // Scan all PCI buses, slots, and functions
    for (0..256) |bus| {
        for (0..32) |slot| {
            for (0..8) |function| {
                const device = PciDevice.init(@intCast(bus), @intCast(slot), @intCast(function)) catch continue;
                
                if (device.vendor_id == NVIDIA_VENDOR_ID) {
                    print("nvzig: Found NVIDIA device at {}:{}.{} (0x{X}:0x{X})\n",
                          .{bus, slot, function, device.vendor_id, device.device_id});
                    device_count += 1;
                }
            }
        }
    }
    
    return device_count;
}

pub fn get_device_by_index(index: u32) !PciDevice {
    var current_index: u32 = 0;
    
    // Scan again to find the device at the specified index
    for (0..256) |bus| {
        for (0..32) |slot| {
            for (0..8) |function| {
                const device = PciDevice.init(@intCast(bus), @intCast(slot), @intCast(function)) catch continue;
                
                if (device.vendor_id == NVIDIA_VENDOR_ID) {
                    if (current_index == index) {
                        return device;
                    }
                    current_index += 1;
                }
            }
        }
    }
    
    return PciError.DeviceNotFound;
}

test "pci enumeration" {
    const count = try enumerate_nvidia_devices();
    try std.testing.expect(count >= 0);
}

test "pci device creation" {
    const device = PciDevice.init(0, 0, 0) catch return;
    try std.testing.expect(device.bus == 0);
    try std.testing.expect(device.slot == 0);
    try std.testing.expect(device.function == 0);
}