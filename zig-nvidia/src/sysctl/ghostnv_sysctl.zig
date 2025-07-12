const std = @import("std");
const linux = std.os.linux;
const kernel = @import("../kernel/module.zig");
const rtx40 = @import("../rtx40/optimizations.zig");
const vibrance = @import("../color/vibrance.zig");

/// GhostNV Sysctl Interface for Runtime Tuning
/// Provides /proc/sys/ghostnv/* entries for real-time configuration
pub const GhostNVSysctl = struct {
    allocator: std.mem.Allocator,
    kernel_module: *kernel.KernelModule,
    optimizer: ?*rtx40.RTX40Optimizer,
    vibrance_engine: ?*vibrance.VibranceEngine,
    
    // Sysctl entries
    proc_entries: std.ArrayList(SysctlEntry),
    proc_dir: ?*ProcDir,
    
    // Current configuration state
    config: SysctlConfig,
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, kernel_module: *kernel.KernelModule) !Self {
        return Self{
            .allocator = allocator,
            .kernel_module = kernel_module,
            .optimizer = null,
            .vibrance_engine = null,
            .proc_entries = std.ArrayList(SysctlEntry).init(allocator),
            .proc_dir = null,
            .config = SysctlConfig.default(),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.unregister_proc_entries();
        self.proc_entries.deinit();
    }
    
    /// Initialize sysctl interface with hardware optimizers
    pub fn init_with_hardware(self: *Self, optimizer: *rtx40.RTX40Optimizer, vibrance_engine: *vibrance.VibranceEngine) !void {
        self.optimizer = optimizer;
        self.vibrance_engine = vibrance_engine;
        
        try self.register_proc_entries();
        
        std.log.info("GhostNV sysctl interface initialized with hardware support");
    }
    
    /// Register all /proc/sys/ghostnv/* entries
    fn register_proc_entries(self: *Self) !void {
        // Create /proc/sys/ghostnv directory
        self.proc_dir = try self.create_proc_directory("ghostnv");
        
        // Performance tuning entries
        try self.add_proc_entry("memory_clock_offset", .{ .read = proc_read_memory_clock, .write = proc_write_memory_clock });
        try self.add_proc_entry("gpu_boost_clock", .{ .read = proc_read_boost_clock, .write = proc_write_boost_clock });
        try self.add_proc_entry("power_limit", .{ .read = proc_read_power_limit, .write = proc_write_power_limit });
        try self.add_proc_entry("fan_curve", .{ .read = proc_read_fan_curve, .write = proc_write_fan_curve });
        try self.add_proc_entry("thermal_target", .{ .read = proc_read_thermal_target, .write = proc_write_thermal_target });
        
        // Vibrance and display
        try self.add_proc_entry("digital_vibrance", .{ .read = proc_read_vibrance, .write = proc_write_vibrance });
        try self.add_proc_entry("active_profile", .{ .read = proc_read_active_profile, .write = proc_write_active_profile });
        try self.add_proc_entry("auto_game_detect", .{ .read = proc_read_auto_detect, .write = proc_write_auto_detect });
        
        // Advanced features  
        try self.add_proc_entry("hardware_scheduling", .{ .read = proc_read_hw_sched, .write = proc_write_hw_sched });
        try self.add_proc_entry("preemption_timeout", .{ .read = proc_read_preempt_timeout, .write = proc_write_preempt_timeout });
        try self.add_proc_entry("memory_compression", .{ .read = proc_read_mem_compression, .write = proc_write_mem_compression });
        
        // Ray tracing optimizations
        try self.add_proc_entry("rt_core_boost", .{ .read = proc_read_rt_boost, .write = proc_write_rt_boost });
        try self.add_proc_entry("tensor_performance", .{ .read = proc_read_tensor_perf, .write = proc_write_tensor_perf });
        
        // Game-specific optimizations
        try self.add_proc_entry("game_mode", .{ .read = proc_read_game_mode, .write = proc_write_game_mode });
        try self.add_proc_entry("latency_mode", .{ .read = proc_read_latency_mode, .write = proc_write_latency_mode });
        
        // Monitoring and stats
        try self.add_proc_entry("gpu_stats", .{ .read = proc_read_gpu_stats, .write = null });
        try self.add_proc_entry("temperature", .{ .read = proc_read_temperature, .write = null });
        try self.add_proc_entry("power_usage", .{ .read = proc_read_power_usage, .write = null });
        try self.add_proc_entry("memory_usage", .{ .read = proc_read_memory_usage, .write = null });
        
        std.log.info("Registered {} sysctl entries under /proc/sys/ghostnv/", .{self.proc_entries.items.len});
    }
    
    fn unregister_proc_entries(self: *Self) void {
        if (self.proc_dir) |dir| {
            // In a real kernel module, this would call remove_proc_entry()
            self.remove_proc_directory(dir);
            self.proc_dir = null;
        }
        
        self.proc_entries.clearAndFree();
        std.log.info("Unregistered sysctl entries");
    }
    
    fn add_proc_entry(self: *Self, name: []const u8, operations: ProcOperations) !void {
        const entry = SysctlEntry{
            .name = try self.allocator.dupe(u8, name),
            .operations = operations,
            .proc_entry = try self.create_proc_entry(name, operations),
        };
        
        try self.proc_entries.append(entry);
    }
    
    // Proc entry creation (kernel-specific implementation would go here)
    fn create_proc_directory(self: *Self, name: []const u8) !*ProcDir {
        _ = self;
        std.log.debug("Created proc directory: /proc/sys/{s}/", .{name});
        return @as(*ProcDir, @ptrFromInt(0x1000)); // Mock pointer
    }
    
    fn remove_proc_directory(self: *Self, dir: *ProcDir) void {
        _ = self;
        _ = dir;
        std.log.debug("Removed proc directory");
    }
    
    fn create_proc_entry(self: *Self, name: []const u8, operations: ProcOperations) !*ProcEntry {
        _ = self;
        _ = operations;
        std.log.debug("Created proc entry: {s}", .{name});
        return @as(*ProcEntry, @ptrFromInt(0x2000)); // Mock pointer
    }
    
    // Sysctl read/write handlers
    
    fn proc_read_memory_clock(self: *Self, buffer: []u8) !usize {
        const offset = self.config.memory_clock_offset;
        return std.fmt.bufPrint(buffer, "{}\n", .{offset}) catch 0;
    }
    
    fn proc_write_memory_clock(self: *Self, buffer: []const u8) !void {
        const offset = std.fmt.parseInt(i32, std.mem.trim(u8, buffer, " \n\r\t"), 10) catch return;
        
        if (offset < -1000 or offset > 2000) {
            return error.InvalidValue;
        }
        
        self.config.memory_clock_offset = offset;
        
        if (self.optimizer) |opt| {
            try opt.setMemoryClockOffset(0, offset);
        }
        
        std.log.info("Memory clock offset set to: {}MHz", .{offset});
    }
    
    fn proc_read_boost_clock(self: *Self, buffer: []u8) !usize {
        const boost = self.config.gpu_boost_clock;
        return std.fmt.bufPrint(buffer, "{}\n", .{boost}) catch 0;
    }
    
    fn proc_write_boost_clock(self: *Self, buffer: []const u8) !void {
        const boost = std.fmt.parseInt(u32, std.mem.trim(u8, buffer, " \n\r\t"), 10) catch return;
        
        if (boost < 1000 or boost > 3000) {
            return error.InvalidValue;
        }
        
        self.config.gpu_boost_clock = boost;
        
        // Apply boost clock via hardware registers
        if (self.optimizer) |opt| {
            try opt.setBoostClock(0, boost);
        }
        
        std.log.info("GPU boost clock set to: {}MHz", .{boost});
    }
    
    fn proc_read_power_limit(self: *Self, buffer: []u8) !usize {
        const power = self.config.power_limit_watts;
        return std.fmt.bufPrint(buffer, "{}\n", .{power}) catch 0;
    }
    
    fn proc_write_power_limit(self: *Self, buffer: []const u8) !void {
        const power = std.fmt.parseInt(u32, std.mem.trim(u8, buffer, " \n\r\t"), 10) catch return;
        
        if (power < 100 or power > 800) {
            return error.InvalidValue;
        }
        
        self.config.power_limit_watts = power;
        
        // Apply power limit
        if (self.optimizer) |opt| {
            try opt.setPowerLimit(0, power);
        }
        
        std.log.info("Power limit set to: {}W", .{power});
    }
    
    fn proc_read_vibrance(self: *Self, buffer: []u8) !usize {
        if (self.vibrance_engine) |engine| {
            const info = try engine.get_vibrance_info();
            return std.fmt.bufPrint(buffer, "{}\n", .{info.current}) catch 0;
        }
        return std.fmt.bufPrint(buffer, "0\n", .{}) catch 0;
    }
    
    fn proc_write_vibrance(self: *Self, buffer: []const u8) !void {
        const vibrance = std.fmt.parseInt(i16, std.mem.trim(u8, buffer, " \n\r\t"), 10) catch return;
        
        if (vibrance < -50 or vibrance > 100) {
            return error.InvalidValue;
        }
        
        if (self.vibrance_engine) |engine| {
            try engine.apply_vibrance_direct(vibrance);
        }
        
        std.log.info("Digital vibrance set to: {}", .{vibrance});
    }
    
    fn proc_read_active_profile(self: *Self, buffer: []u8) !usize {
        if (self.vibrance_engine) |engine| {
            if (engine.active_profile) |profile| {
                return std.fmt.bufPrint(buffer, "{s}\n", .{profile}) catch 0;
            }
        }
        return std.fmt.bufPrint(buffer, "none\n", .{}) catch 0;
    }
    
    fn proc_write_active_profile(self: *Self, buffer: []const u8) !void {
        const profile_name = std.mem.trim(u8, buffer, " \n\r\t");
        
        if (self.vibrance_engine) |engine| {
            try engine.apply_profile(profile_name);
        }
        
        std.log.info("Applied vibrance profile: {s}", .{profile_name});
    }
    
    fn proc_read_hw_sched(self: *Self, buffer: []u8) !usize {
        const enabled = if (self.config.hardware_scheduling) "1" else "0";
        return std.fmt.bufPrint(buffer, "{s}\n", .{enabled}) catch 0;
    }
    
    fn proc_write_hw_sched(self: *Self, buffer: []const u8) !void {
        const value = std.mem.trim(u8, buffer, " \n\r\t");
        const enabled = std.mem.eql(u8, value, "1") or std.ascii.eqlIgnoreCase(value, "true");
        
        self.config.hardware_scheduling = enabled;
        
        if (self.optimizer) |opt| {
            if (enabled) {
                try opt.enableHardwareScheduling(0);
            } else {
                try opt.disableHardwareScheduling(0);
            }
        }
        
        std.log.info("Hardware scheduling: {}", .{enabled});
    }
    
    fn proc_read_gpu_stats(self: *Self, buffer: []u8) !usize {
        if (self.kernel_module.device_count == 0) {
            return std.fmt.bufPrint(buffer, "No GPU devices\n", .{}) catch 0;
        }
        
        const device = &self.kernel_module.devices[0];
        
        // Read GPU statistics
        const gpu_util = device.getUtilization() catch 0;
        const mem_util = device.getMemoryUtilization() catch 0;
        const temp = device.getTemperature() catch 0;
        const power = device.getPowerUsage() catch 0;
        
        return std.fmt.bufPrint(buffer, 
            \\GPU Utilization: {}%
            \\Memory Utilization: {}%
            \\Temperature: {}°C
            \\Power Usage: {}W
            \\Clock Speed: {}MHz
            \\Memory Clock: {}MHz
            \\
        , .{ gpu_util, mem_util, temp, power, device.core_clock, device.memory_clock }) catch 0;
    }
    
    fn proc_read_temperature(self: *Self, buffer: []u8) !usize {
        if (self.kernel_module.device_count > 0) {
            const temp = self.kernel_module.devices[0].getTemperature() catch 0;
            return std.fmt.bufPrint(buffer, "{}\n", .{temp}) catch 0;
        }
        return std.fmt.bufPrint(buffer, "0\n", .{}) catch 0;
    }
    
    fn proc_read_power_usage(self: *Self, buffer: []u8) !usize {
        if (self.kernel_module.device_count > 0) {
            const power = self.kernel_module.devices[0].getPowerUsage() catch 0;
            return std.fmt.bufPrint(buffer, "{}\n", .{power}) catch 0;
        }
        return std.fmt.bufPrint(buffer, "0\n", .{}) catch 0;
    }
    
    fn proc_read_memory_usage(self: *Self, buffer: []u8) !usize {
        if (self.kernel_module.device_count > 0) {
            const device = &self.kernel_module.devices[0];
            const used = device.getMemoryUsed() catch 0;
            const total = device.memory_size_mb * 1024 * 1024;
            const usage_percent = if (total > 0) (used * 100) / total else 0;
            
            return std.fmt.bufPrint(buffer, 
                "Used: {} MB\nTotal: {} MB\nUsage: {}%\n", 
                .{ used / (1024 * 1024), total / (1024 * 1024), usage_percent }
            ) catch 0;
        }
        return std.fmt.bufPrint(buffer, "No device\n", .{}) catch 0;
    }
    
    // Default implementations for unimplemented handlers
    fn proc_read_fan_curve(self: *Self, buffer: []u8) !usize {
        _ = self;
        return std.fmt.bufPrint(buffer, "auto\n", .{}) catch 0;
    }
    
    fn proc_write_fan_curve(self: *Self, buffer: []const u8) !void {
        _ = self;
        _ = buffer;
        std.log.debug("Fan curve configuration not yet implemented");
    }
    
    fn proc_read_thermal_target(self: *Self, buffer: []u8) !usize {
        const target = self.config.thermal_target;
        return std.fmt.bufPrint(buffer, "{}\n", .{target}) catch 0;
    }
    
    fn proc_write_thermal_target(self: *Self, buffer: []const u8) !void {
        const target = std.fmt.parseInt(u32, std.mem.trim(u8, buffer, " \n\r\t"), 10) catch return;
        if (target < 50 or target > 95) return error.InvalidValue;
        self.config.thermal_target = target;
        std.log.info("Thermal target set to: {}°C", .{target});
    }
    
    fn proc_read_auto_detect(self: *Self, buffer: []u8) !usize {
        const enabled = if (self.config.auto_game_detect) "1" else "0";
        return std.fmt.bufPrint(buffer, "{s}\n", .{enabled}) catch 0;
    }
    
    fn proc_write_auto_detect(self: *Self, buffer: []const u8) !void {
        const value = std.mem.trim(u8, buffer, " \n\r\t");
        self.config.auto_game_detect = std.mem.eql(u8, value, "1") or std.ascii.eqlIgnoreCase(value, "true");
        std.log.info("Auto game detection: {}", .{self.config.auto_game_detect});
    }
    
    fn proc_read_preempt_timeout(self: *Self, buffer: []u8) !usize {
        const timeout = self.config.preemption_timeout_us;
        return std.fmt.bufPrint(buffer, "{}\n", .{timeout}) catch 0;
    }
    
    fn proc_write_preempt_timeout(self: *Self, buffer: []const u8) !void {
        const timeout = std.fmt.parseInt(u32, std.mem.trim(u8, buffer, " \n\r\t"), 10) catch return;
        if (timeout < 10 or timeout > 10000) return error.InvalidValue;
        self.config.preemption_timeout_us = timeout;
        std.log.info("Preemption timeout set to: {}μs", .{timeout});
    }
    
    fn proc_read_mem_compression(self: *Self, buffer: []u8) !usize {
        const enabled = if (self.config.memory_compression) "1" else "0";
        return std.fmt.bufPrint(buffer, "{s}\n", .{enabled}) catch 0;
    }
    
    fn proc_write_mem_compression(self: *Self, buffer: []const u8) !void {
        const value = std.mem.trim(u8, buffer, " \n\r\t");
        const enabled = std.mem.eql(u8, value, "1") or std.ascii.eqlIgnoreCase(value, "true");
        self.config.memory_compression = enabled;
        
        if (self.optimizer) |opt| {
            try opt.enableMemoryCompression(0, enabled);
        }
        
        std.log.info("Memory compression: {}", .{enabled});
    }
    
    fn proc_read_rt_boost(self: *Self, buffer: []u8) !usize {
        const boost = self.config.rt_core_boost;
        return std.fmt.bufPrint(buffer, "{}\n", .{boost}) catch 0;
    }
    
    fn proc_write_rt_boost(self: *Self, buffer: []const u8) !void {
        const boost = std.fmt.parseInt(u8, std.mem.trim(u8, buffer, " \n\r\t"), 10) catch return;
        if (boost > 100) return error.InvalidValue;
        self.config.rt_core_boost = boost;
        std.log.info("RT Core boost set to: {}%", .{boost});
    }
    
    fn proc_read_tensor_perf(self: *Self, buffer: []u8) !usize {
        const perf = self.config.tensor_performance;
        return std.fmt.bufPrint(buffer, "{}\n", .{perf}) catch 0;
    }
    
    fn proc_write_tensor_perf(self: *Self, buffer: []const u8) !void {
        const perf = std.fmt.parseInt(u8, std.mem.trim(u8, buffer, " \n\r\t"), 10) catch return;
        if (perf > 100) return error.InvalidValue;
        self.config.tensor_performance = perf;
        std.log.info("Tensor performance set to: {}%", .{perf});
    }
    
    fn proc_read_game_mode(self: *Self, buffer: []u8) !usize {
        const mode = self.config.game_mode.toString();
        return std.fmt.bufPrint(buffer, "{s}\n", .{mode}) catch 0;
    }
    
    fn proc_write_game_mode(self: *Self, buffer: []const u8) !void {
        const mode_str = std.mem.trim(u8, buffer, " \n\r\t");
        const mode = std.meta.stringToEnum(GameMode, mode_str) orelse return error.InvalidValue;
        self.config.game_mode = mode;
        std.log.info("Game mode set to: {s}", .{mode.toString()});
    }
    
    fn proc_read_latency_mode(self: *Self, buffer: []u8) !usize {
        const mode = self.config.latency_mode.toString();
        return std.fmt.bufPrint(buffer, "{s}\n", .{mode}) catch 0;
    }
    
    fn proc_write_latency_mode(self: *Self, buffer: []const u8) !void {
        const mode_str = std.mem.trim(u8, buffer, " \n\r\t");
        const mode = std.meta.stringToEnum(LatencyMode, mode_str) orelse return error.InvalidValue;
        self.config.latency_mode = mode;
        std.log.info("Latency mode set to: {s}", .{mode.toString()});
    }
};

// Configuration structure
const SysctlConfig = struct {
    // Performance settings
    memory_clock_offset: i32,
    gpu_boost_clock: u32,
    power_limit_watts: u32,
    thermal_target: u32,
    
    // Feature toggles
    hardware_scheduling: bool,
    memory_compression: bool,
    auto_game_detect: bool,
    
    // Timing settings
    preemption_timeout_us: u32,
    
    // Performance boosts
    rt_core_boost: u8,        // 0-100%
    tensor_performance: u8,   // 0-100%
    
    // Mode settings
    game_mode: GameMode,
    latency_mode: LatencyMode,
    
    pub fn default() SysctlConfig {
        return SysctlConfig{
            .memory_clock_offset = 0,
            .gpu_boost_clock = 2500, // Default boost clock
            .power_limit_watts = 350, // Conservative default
            .thermal_target = 83,     // Default target temp
            .hardware_scheduling = true,
            .memory_compression = true,
            .auto_game_detect = false,
            .preemption_timeout_us = 100,
            .rt_core_boost = 80,
            .tensor_performance = 90,
            .game_mode = .balanced,
            .latency_mode = .normal,
        };
    }
};

const GameMode = enum {
    power_save,
    balanced,
    performance,
    extreme,
    
    pub fn toString(self: GameMode) []const u8 {
        return switch (self) {
            .power_save => "power_save",
            .balanced => "balanced", 
            .performance => "performance",
            .extreme => "extreme",
        };
    }
};

const LatencyMode = enum {
    normal,
    low,
    ultra_low,
    
    pub fn toString(self: LatencyMode) []const u8 {
        return switch (self) {
            .normal => "normal",
            .low => "low",
            .ultra_low => "ultra_low",
        };
    }
};

// Proc filesystem types
const SysctlEntry = struct {
    name: []const u8,
    operations: ProcOperations,
    proc_entry: *ProcEntry,
};

const ProcOperations = struct {
    read: ?*const fn (self: *GhostNVSysctl, buffer: []u8) anyerror!usize,
    write: ?*const fn (self: *GhostNVSysctl, buffer: []const u8) anyerror!void,
};

// Mock kernel types (in real implementation these would be kernel types)
const ProcDir = opaque {};
const ProcEntry = opaque {};

// Test functions
test "sysctl config" {
    const config = SysctlConfig.default();
    try std.testing.expect(config.memory_clock_offset == 0);
    try std.testing.expect(config.hardware_scheduling == true);
    try std.testing.expect(config.game_mode == .balanced);
}

test "sysctl interface" {
    const allocator = std.testing.allocator;
    
    var mock_kernel = kernel.KernelModule{
        .allocator = allocator,
        .nvidia_fd = -1,
        .nvidia_ctl_fd = -1,
        .nvidia_uvm_fd = -1,
        .device_count = 0,
        .devices = undefined,
    };
    
    var sysctl = try GhostNVSysctl.init(allocator, &mock_kernel);
    defer sysctl.deinit();
    
    try std.testing.expect(sysctl.config.game_mode == .balanced);
}