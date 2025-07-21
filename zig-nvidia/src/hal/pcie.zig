const std = @import("std");
const Allocator = std.mem.Allocator;
const pci = @import("pci.zig");
const linux = std.os.linux;

/// PCIe Link Management Hardware Abstraction Layer  
/// Manages PCIe link state, error handling, and power management

pub const PcieError = error{
    LinkTrainingFailed,
    UnsupportedSpeed,
    UnsupportedWidth,
    LinkDownError,
    AerError,
    ConfigSpaceError,
    PowerManagementError,
    HotplugError,
    OutOfMemory,
    PermissionDenied,
};

pub const PcieLinkSpeed = enum(u8) {
    gen1 = 1,  // 2.5 GT/s
    gen2 = 2,  // 5.0 GT/s  
    gen3 = 3,  // 8.0 GT/s
    gen4 = 4,  // 16.0 GT/s
    gen5 = 5,  // 32.0 GT/s
    gen6 = 6,  // 64.0 GT/s
    
    pub fn getBandwidthGbps(self: PcieLinkSpeed) f32 {
        return switch (self) {
            .gen1 => 2.5,
            .gen2 => 5.0,
            .gen3 => 8.0,
            .gen4 => 16.0,
            .gen5 => 32.0,
            .gen6 => 64.0,
        };
    }
    
    pub fn toString(self: PcieLinkSpeed) []const u8 {
        return switch (self) {
            .gen1 => "PCIe 1.0 (2.5 GT/s)",
            .gen2 => "PCIe 2.0 (5.0 GT/s)",
            .gen3 => "PCIe 3.0 (8.0 GT/s)",
            .gen4 => "PCIe 4.0 (16.0 GT/s)",
            .gen5 => "PCIe 5.0 (32.0 GT/s)",
            .gen6 => "PCIe 6.0 (64.0 GT/s)",
        };
    }
};

pub const PcieLinkWidth = enum(u8) {
    x1 = 1,
    x2 = 2,
    x4 = 4,
    x8 = 8,
    x16 = 16,
    x32 = 32,
    
    pub fn toString(self: PcieLinkWidth) []const u8 {
        return switch (self) {
            .x1 => "x1",
            .x2 => "x2",
            .x4 => "x4",
            .x8 => "x8",
            .x16 => "x16",
            .x32 => "x32",
        };
    }
    
    pub fn getLaneCount(self: PcieLinkWidth) u8 {
        return @intFromEnum(self);
    }
};

pub const PcieLinkState = enum(u8) {
    l0 = 0,        // Active
    l0s = 1,       // Standby
    l1 = 2,        // Low power active
    l1_1 = 3,      // L1 sub-state 1
    l1_2 = 4,      // L1 sub-state 2
    l2 = 5,        // Auxiliary power
    l3 = 6,        // Off
    
    pub fn toString(self: PcieLinkState) []const u8 {
        return switch (self) {
            .l0 => "L0 - Active",
            .l0s => "L0s - Standby", 
            .l1 => "L1 - Low Power Active",
            .l1_1 => "L1.1 - Sub-state 1",
            .l1_2 => "L1.2 - Sub-state 2", 
            .l2 => "L2 - Auxiliary Power",
            .l3 => "L3 - Off",
        };
    }
    
    pub fn getPowerSavings(self: PcieLinkState) f32 {
        return switch (self) {
            .l0 => 0.0,    // No savings
            .l0s => 0.1,   // 10% savings
            .l1 => 0.5,    // 50% savings
            .l1_1 => 0.7,  // 70% savings
            .l1_2 => 0.9,  // 90% savings
            .l2 => 0.95,   // 95% savings
            .l3 => 1.0,    // 100% savings (off)
        };
    }
};

pub const AerErrorType = enum(u8) {
    correctable = 0,
    non_fatal = 1,
    fatal = 2,
    
    pub fn toString(self: AerErrorType) []const u8 {
        return switch (self) {
            .correctable => "Correctable",
            .non_fatal => "Non-Fatal", 
            .fatal => "Fatal",
        };
    }
};

pub const AerError = struct {
    error_type: AerErrorType,
    error_code: u32,
    timestamp: u64,
    source_id: u16,
    description: []const u8,
    
    pub fn init(error_type: AerErrorType, code: u32, source: u16, desc: []const u8) AerError {
        return AerError{
            .error_type = error_type,
            .error_code = code,
            .timestamp = @intCast(std.time.microTimestamp()),
            .source_id = source,
            .description = desc,
        };
    }
};

pub const LinkCapabilities = struct {
    max_speed: PcieLinkSpeed,
    max_width: PcieLinkWidth,
    supported_speeds: [6]bool, // Gen 1-6 support
    aspm_support: bool,        // Active State Power Management
    l1_substates: bool,        // L1 sub-states support
    hot_plug: bool,            // Hot plug support
    surprise_down: bool,       // Surprise removal support
    power_indicator: bool,     // Power indicator support
    attention_button: bool,    // Attention button support
    mrl_sensor: bool,          // MRL sensor support
    power_controller: bool,    // Power controller support
    ari_support: bool,         // Alternative Routing-ID
    
    pub fn getTotalBandwidthGbps(self: *const LinkCapabilities) f32 {
        return self.max_speed.getBandwidthGbps() * @as(f32, @floatFromInt(self.max_width.getLaneCount()));
    }
};

pub const LinkStatus = struct {
    current_speed: PcieLinkSpeed,
    current_width: PcieLinkWidth,
    link_state: PcieLinkState,
    link_training: bool,
    slot_clock_config: bool,
    data_link_layer_active: bool,
    link_bandwidth_notification: bool,
    link_autonomous_bandwidth_status: bool,
    negotiated_speed: PcieLinkSpeed,
    negotiated_width: PcieLinkWidth,
    equalization_complete: bool,
    
    pub fn getCurrentBandwidthGbps(self: *const LinkStatus) f32 {
        return self.current_speed.getBandwidthGbps() * @as(f32, @floatFromInt(self.current_width.getLaneCount()));
    }
    
    pub fn getEfficiencyPercent(self: *const LinkStatus, capabilities: *const LinkCapabilities) f32 {
        const max_bandwidth = capabilities.getTotalBandwidthGbps();
        const current_bandwidth = self.getCurrentBandwidthGbps();
        return (current_bandwidth / max_bandwidth) * 100.0;
    }
};

pub const PowerManagementState = struct {
    current_state: u8,         // D0, D1, D2, D3
    pme_enable: bool,          // Power Management Event enable
    data_select: u8,           // Data select for power consumption
    data_scale: u8,            // Data scale for power consumption
    power_consumed: f32,       // Current power consumption (watts)
    
    pub fn getStateName(self: *const PowerManagementState) []const u8 {
        return switch (self.current_state) {
            0 => "D0 - Fully On",
            1 => "D1 - Low Power",
            2 => "D2 - Standby",
            3 => "D3hot - Suspend",
            4 => "D3cold - Off",
            else => "Unknown",
        };
    }
};

pub const PcieManager = struct {
    const Self = @This();
    
    allocator: Allocator,
    device: pci.PciDevice,
    config_space: ?*volatile u8,
    capabilities: LinkCapabilities,
    current_status: LinkStatus,
    power_state: PowerManagementState,
    aer_errors: std.ArrayList(AerError),
    
    // PCIe capability register offsets
    const PCIE_CAP_LIST: u8 = 0x34;
    const PCIE_CAP_ID_PCIE: u8 = 0x10;
    const PCIE_CAP_ID_PM: u8 = 0x01;
    const PCIE_CAP_ID_AER: u16 = 0x0001;
    
    // PCIe registers (relative to PCIe capability)
    const PCIE_LINK_CAP: u8 = 0x0C;
    const PCIE_LINK_CTRL: u8 = 0x10;
    const PCIE_LINK_STATUS: u8 = 0x12;
    const PCIE_LINK_CAP2: u8 = 0x2C;
    const PCIE_LINK_CTRL2: u8 = 0x30;
    const PCIE_LINK_STATUS2: u8 = 0x32;
    
    // Power Management registers
    const PM_CAP: u8 = 0x02;
    const PM_CTRL_STATUS: u8 = 0x04;
    const PM_DATA: u8 = 0x07;
    
    pub fn init(allocator: Allocator, device: pci.PciDevice) !Self {
        var self = Self{
            .allocator = allocator,
            .device = device,
            .config_space = null,
            .capabilities = undefined,
            .current_status = undefined,
            .power_state = undefined,
            .aer_errors = std.ArrayList(AerError).init(allocator),
        };
        
        try self.mapConfigSpace();
        try self.readCapabilities();
        try self.readCurrentStatus();
        try self.initializeAer();
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        // Ensure device is in safe state before cleanup
        self.setLinkState(.l1) catch {};
        
        self.aer_errors.deinit();
        
        if (self.config_space) |space| {
            _ = linux.munmap(@ptrCast(space), 4096);
        }
    }
    
    fn mapConfigSpace(self: *Self) !void {
        // Map PCI configuration space
        const config_path = try std.fmt.allocPrint(
            self.allocator, 
            "/sys/bus/pci/devices/{04x}:{02x}:{02x}.{x}/config",
            .{0, 1, 0, 0} // Assuming GPU at 01:00.0
        );
        defer self.allocator.free(config_path);
        
        const config_fd = try std.fs.openFileAbsolute(config_path, .{});
        defer config_fd.close();
        
        const config_ptr = linux.mmap(
            null,
            4096, // Standard PCI config space size
            linux.PROT.READ | linux.PROT.WRITE,
            linux.MAP.SHARED,
            config_fd.handle,
            0
        );
        
        if (config_ptr == linux.MAP.FAILED) {
            return PcieError.PermissionDenied;
        }
        
        self.config_space = @ptrCast(@alignCast(config_ptr));
    }
    
    fn readConfig8(self: *Self, offset: u16) u8 {
        if (self.config_space) |space| {
            return space[offset];
        }
        return 0;
    }
    
    fn readConfig16(self: *Self, offset: u16) u16 {
        if (self.config_space) |space| {
            const ptr: *volatile u16 = @ptrCast(@alignCast(space + offset));
            return ptr.*;
        }
        return 0;
    }
    
    fn readConfig32(self: *Self, offset: u16) u32 {
        if (self.config_space) |space| {
            const ptr: *volatile u32 = @ptrCast(@alignCast(space + offset));
            return ptr.*;
        }
        return 0;
    }
    
    fn writeConfig16(self: *Self, offset: u16, value: u16) void {
        if (self.config_space) |space| {
            const ptr: *volatile u16 = @ptrCast(@alignCast(space + offset));
            ptr.* = value;
        }
    }
    
    fn writeConfig32(self: *Self, offset: u16, value: u32) void {
        if (self.config_space) |space| {
            const ptr: *volatile u32 = @ptrCast(@alignCast(space + offset));
            ptr.* = value;
        }
    }
    
    fn findCapability(self: *Self, cap_id: u8) ?u8 {
        var cap_ptr = self.readConfig8(PCIE_CAP_LIST);
        
        while (cap_ptr != 0) {
            const cap_id_found = self.readConfig8(cap_ptr);
            if (cap_id_found == cap_id) {
                return cap_ptr;
            }
            cap_ptr = self.readConfig8(cap_ptr + 1);
        }
        
        return null;
    }
    
    fn readCapabilities(self: *Self) !void {
        // Find PCIe capability
        const pcie_cap = self.findCapability(PCIE_CAP_ID_PCIE) orelse {
            return PcieError.ConfigSpaceError;
        };
        
        // Read link capabilities
        const link_cap = self.readConfig32(pcie_cap + PCIE_LINK_CAP);
        const link_cap2 = self.readConfig32(pcie_cap + PCIE_LINK_CAP2);
        
        // Parse maximum speed
        const max_speed_val = @as(u8, @truncate(link_cap & 0xF));
        const max_speed = if (max_speed_val >= 1 and max_speed_val <= 6) 
            @as(PcieLinkSpeed, @enumFromInt(max_speed_val))
        else 
            PcieLinkSpeed.gen3;
        
        // Parse maximum width
        const max_width_val = @as(u8, @truncate((link_cap >> 4) & 0x3F));
        const max_width = switch (max_width_val) {
            1 => PcieLinkWidth.x1,
            2 => PcieLinkWidth.x2,
            4 => PcieLinkWidth.x4,
            8 => PcieLinkWidth.x8,
            16 => PcieLinkWidth.x16,
            32 => PcieLinkWidth.x32,
            else => PcieLinkWidth.x16,
        };
        
        // Parse supported speeds from Link Capabilities 2
        var supported_speeds = [_]bool{false} ** 6;
        for (1..7) |i| {
            supported_speeds[i-1] = (link_cap2 & (@as(u32, 1) << @truncate(i-1))) != 0;
        }
        
        self.capabilities = LinkCapabilities{
            .max_speed = max_speed,
            .max_width = max_width,
            .supported_speeds = supported_speeds,
            .aspm_support = (link_cap & 0x00C00000) != 0,
            .l1_substates = (link_cap2 & 0x00000040) != 0,
            .hot_plug = (link_cap & 0x00000040) != 0,
            .surprise_down = (link_cap & 0x00000080) != 0,
            .power_indicator = (link_cap & 0x00000100) != 0,
            .attention_button = (link_cap & 0x00000200) != 0,
            .mrl_sensor = (link_cap & 0x00000400) != 0,
            .power_controller = (link_cap & 0x00000800) != 0,
            .ari_support = (link_cap2 & 0x00000020) != 0,
        };
    }
    
    fn readCurrentStatus(self: *Self) !void {
        const pcie_cap = self.findCapability(PCIE_CAP_ID_PCIE) orelse {
            return PcieError.ConfigSpaceError;
        };
        
        // Read link status
        const link_status = self.readConfig16(pcie_cap + PCIE_LINK_STATUS);
        const link_status2 = self.readConfig16(pcie_cap + PCIE_LINK_STATUS2);
        
        // Parse current speed
        const current_speed_val = @as(u8, @truncate(link_status & 0xF));
        const current_speed = if (current_speed_val >= 1 and current_speed_val <= 6) 
            @as(PcieLinkSpeed, @enumFromInt(current_speed_val))
        else 
            PcieLinkSpeed.gen1;
        
        // Parse current width
        const current_width_val = @as(u8, @truncate((link_status >> 4) & 0x3F));
        const current_width = switch (current_width_val) {
            1 => PcieLinkWidth.x1,
            2 => PcieLinkWidth.x2,
            4 => PcieLinkWidth.x4,
            8 => PcieLinkWidth.x8,
            16 => PcieLinkWidth.x16,
            32 => PcieLinkWidth.x32,
            else => PcieLinkWidth.x1,
        };
        
        self.current_status = LinkStatus{
            .current_speed = current_speed,
            .current_width = current_width,
            .link_state = .l0, // Would need ASPM status to determine actual state
            .link_training = (link_status & 0x0800) != 0,
            .slot_clock_config = (link_status & 0x1000) != 0,
            .data_link_layer_active = (link_status & 0x2000) != 0,
            .link_bandwidth_notification = (link_status & 0x4000) != 0,
            .link_autonomous_bandwidth_status = (link_status & 0x8000) != 0,
            .negotiated_speed = current_speed,
            .negotiated_width = current_width,
            .equalization_complete = (link_status2 & 0x0002) != 0,
        };
        
        // Read power management status
        if (self.findCapability(PCIE_CAP_ID_PM)) |pm_cap| {
            const pm_ctrl_status = self.readConfig16(pm_cap + PM_CTRL_STATUS);
            const pm_data = self.readConfig8(pm_cap + PM_DATA);
            
            self.power_state = PowerManagementState{
                .current_state = @as(u8, @truncate(pm_ctrl_status & 0x3)),
                .pme_enable = (pm_ctrl_status & 0x0100) != 0,
                .data_select = @as(u8, @truncate((pm_ctrl_status >> 9) & 0xF)),
                .data_scale = @as(u8, @truncate((pm_ctrl_status >> 13) & 0x3)),
                .power_consumed = @as(f32, @floatFromInt(pm_data)) * 0.1, // Estimated scaling
            };
        } else {
            self.power_state = PowerManagementState{
                .current_state = 0,
                .pme_enable = false,
                .data_select = 0,
                .data_scale = 0,
                .power_consumed = 25.0, // Default PCIe slot power
            };
        }
    }
    
    fn initializeAer(self: *Self) !void {
        // Advanced Error Reporting initialization would go here
        // This is simplified since AER requires extended config space access
    }
    
    pub fn setLinkSpeed(self: *Self, target_speed: PcieLinkSpeed) !void {
        const pcie_cap = self.findCapability(PCIE_CAP_ID_PCIE) orelse {
            return PcieError.ConfigSpaceError;
        };
        
        // Check if target speed is supported
        const speed_index = @intFromEnum(target_speed) - 1;
        if (speed_index >= 6 or !self.capabilities.supported_speeds[speed_index]) {
            return PcieError.UnsupportedSpeed;
        }
        
        // Set target link speed in Link Control 2 register
        var link_ctrl2 = self.readConfig16(pcie_cap + PCIE_LINK_CTRL2);
        link_ctrl2 = (link_ctrl2 & 0xFFF0) | @intFromEnum(target_speed);
        self.writeConfig16(pcie_cap + PCIE_LINK_CTRL2, link_ctrl2);
        
        // Retrain the link
        var link_ctrl = self.readConfig16(pcie_cap + PCIE_LINK_CTRL);
        link_ctrl |= 0x0020; // Set Retrain Link bit
        self.writeConfig16(pcie_cap + PCIE_LINK_CTRL, link_ctrl);
        
        // Wait for link training to complete
        try self.waitForLinkTraining();
        
        // Verify the speed change
        try self.readCurrentStatus();
        if (self.current_status.current_speed != target_speed) {
            return PcieError.LinkTrainingFailed;
        }
    }
    
    pub fn setLinkState(self: *Self, target_state: PcieLinkState) !void {
        const pcie_cap = self.findCapability(PCIE_CAP_ID_PCIE) orelse {
            return PcieError.ConfigSpaceError;
        };
        
        var link_ctrl = self.readConfig16(pcie_cap + PCIE_LINK_CTRL);
        
        // Configure ASPM (Active State Power Management)
        link_ctrl &= 0xFFFC; // Clear ASPM control bits
        
        switch (target_state) {
            .l0 => {
                // No ASPM - stay in L0
                link_ctrl |= 0x0000;
            },
            .l0s => {
                // L0s entry enabled
                link_ctrl |= 0x0001;
            },
            .l1 => {
                // L1 entry enabled
                link_ctrl |= 0x0002;
            },
            .l1_1, .l1_2 => {
                // L1 sub-states require additional configuration
                link_ctrl |= 0x0002;
                // L1 sub-state configuration would go here
            },
            .l2, .l3 => {
                // These require coordination with power management
                return self.setPowerState(@intFromEnum(target_state) - 3);
            },
        }
        
        self.writeConfig16(pcie_cap + PCIE_LINK_CTRL, link_ctrl);
        
        // Allow time for state transition
        std.time.sleep(10_000_000); // 10ms
        
        self.current_status.link_state = target_state;
    }
    
    pub fn setPowerState(self: *Self, d_state: u8) !void {
        if (d_state > 3) return PcieError.PowerManagementError;
        
        const pm_cap = self.findCapability(PCIE_CAP_ID_PM) orelse {
            return PcieError.ConfigSpaceError;
        };
        
        var pm_ctrl_status = self.readConfig16(pm_cap + PM_CTRL_STATUS);
        pm_ctrl_status = (pm_ctrl_status & 0xFFFC) | d_state;
        self.writeConfig16(pm_cap + PM_CTRL_STATUS, pm_ctrl_status);
        
        // Wait for power state transition
        std.time.sleep(100_000_000); // 100ms
        
        self.power_state.current_state = d_state;
    }
    
    fn waitForLinkTraining(self: *Self) !void {
        const pcie_cap = self.findCapability(PCIE_CAP_ID_PCIE) orelse {
            return PcieError.ConfigSpaceError;
        };
        
        var timeout: u32 = 1000; // 1 second timeout
        
        while (timeout > 0) {
            const link_status = self.readConfig16(pcie_cap + PCIE_LINK_STATUS);
            if ((link_status & 0x0800) == 0) { // Link training bit cleared
                return;
            }
            
            timeout -= 1;
            std.time.sleep(1_000_000); // 1ms
        }
        
        return PcieError.LinkTrainingFailed;
    }
    
    pub fn enableAspm(self: *Self, enable_l0s: bool, enable_l1: bool) !void {
        if (!self.capabilities.aspm_support) {
            return PcieError.PowerManagementError;
        }
        
        const pcie_cap = self.findCapability(PCIE_CAP_ID_PCIE) orelse {
            return PcieError.ConfigSpaceError;
        };
        
        var link_ctrl = self.readConfig16(pcie_cap + PCIE_LINK_CTRL);
        link_ctrl &= 0xFFFC; // Clear ASPM bits
        
        if (enable_l0s) link_ctrl |= 0x0001;
        if (enable_l1) link_ctrl |= 0x0002;
        
        self.writeConfig16(pcie_cap + PCIE_LINK_CTRL, link_ctrl);
    }
    
    pub fn getAerErrors(self: *Self) []const AerError {
        return self.aer_errors.items;
    }
    
    pub fn clearAerErrors(self: *Self) void {
        self.aer_errors.clearRetainingCapacity();
    }
    
    pub fn getLinkCapabilities(self: *Self) LinkCapabilities {
        return self.capabilities;
    }
    
    pub fn getLinkStatus(self: *Self) LinkStatus {
        return self.current_status;
    }
    
    pub fn getPowerState(self: *Self) PowerManagementState {
        return self.power_state;
    }
    
    pub fn getCurrentBandwidthUtilization(self: *Self) f32 {
        // This would require monitoring PCIe performance counters
        // Simplified implementation returns efficiency percentage
        return self.current_status.getEfficiencyPercent(&self.capabilities);
    }
    
    pub fn isLinkUp(self: *Self) bool {
        return self.current_status.data_link_layer_active and 
               !self.current_status.link_training;
    }
    
    pub fn supportsHotplug(self: *Self) bool {
        return self.capabilities.hot_plug;
    }
    
    pub fn updateStatus(self: *Self) !void {
        try self.readCurrentStatus();
    }
    
    pub fn resetLink(self: *Self) !void {
        // Trigger hot reset through bridge control (if available)
        // This is a simplified implementation
        const pcie_cap = self.findCapability(PCIE_CAP_ID_PCIE) orelse {
            return PcieError.ConfigSpaceError;
        };
        
        // Disable then re-enable the link
        var link_ctrl = self.readConfig16(pcie_cap + PCIE_LINK_CTRL);
        link_ctrl |= 0x0010; // Link disable
        self.writeConfig16(pcie_cap + PCIE_LINK_CTRL, link_ctrl);
        
        std.time.sleep(100_000_000); // 100ms
        
        link_ctrl &= ~@as(u16, 0x0010); // Link enable
        self.writeConfig16(pcie_cap + PCIE_LINK_CTRL, link_ctrl);
        
        try self.waitForLinkTraining();
    }
};

pub fn initPcieManager(allocator: Allocator, device: pci.PciDevice) !PcieManager {
    return PcieManager.init(allocator, device);
}