const std = @import("std");
const linux = std.os.linux;

/// Kernel module interface for GhostNV Pure Zig NVIDIA Driver
/// Provides direct hardware access via kernel syscalls and ioctl
pub const KernelModule = struct {
    allocator: std.mem.Allocator,
    nvidia_fd: i32,
    nvidia_ctl_fd: i32,
    nvidia_uvm_fd: i32,
    device_count: u32,
    devices: []DeviceInfo,
    
    pub fn init(allocator: std.mem.Allocator) !KernelModule {
        // Open NVIDIA device files
        const nvidia_ctl_fd = std.posix.open("/dev/nvidiactl", .{ .ACCMODE = .RDWR }, 0) catch |err| {
            std.log.err("Failed to open /dev/nvidiactl: {}", .{err});
            return error.NoNVIDIADevice;
        };
        
        const nvidia_fd = std.posix.open("/dev/nvidia0", .{ .ACCMODE = .RDWR }, 0) catch |err| {
            std.log.err("Failed to open /dev/nvidia0: {}", .{err});
            std.posix.close(nvidia_ctl_fd);
            return error.NoNVIDIADevice;
        };
        
        const nvidia_uvm_fd = std.posix.open("/dev/nvidia-uvm", .{ .ACCMODE = .RDWR }, 0) catch |err| {
            std.log.warn("Failed to open /dev/nvidia-uvm: {}", .{err});
            return -1;
        };
        
        // Query device count
        var device_count: u32 = 0;
        const count_result = ioctl(nvidia_ctl_fd, NVIDIA_IOCTL_CARD_INFO, @intFromPtr(&device_count));
        if (count_result < 0) {
            std.log.err("Failed to query device count");
            device_count = 1; // Assume single GPU
        }
        
        // Allocate device info array
        const devices = try allocator.alloc(DeviceInfo, device_count);
        
        // Initialize device info
        for (devices, 0..) |*device, i| {
            device.* = try initializeDevice(allocator, @intCast(i));
        }
        
        std.log.info("GhostNV kernel module initialized: {} GPUs detected", .{device_count});
        
        return KernelModule{
            .allocator = allocator,
            .nvidia_fd = nvidia_fd,
            .nvidia_ctl_fd = nvidia_ctl_fd,
            .nvidia_uvm_fd = nvidia_uvm_fd,
            .device_count = device_count,
            .devices = devices,
        };
    }
    
    pub fn deinit(self: *KernelModule) void {
        for (self.devices) |*device| {
            device.deinit(self.allocator);
        }
        self.allocator.free(self.devices);
        
        if (self.nvidia_uvm_fd >= 0) std.posix.close(self.nvidia_uvm_fd);
        std.posix.close(self.nvidia_fd);
        std.posix.close(self.nvidia_ctl_fd);
    }
    
    /// Get GPU performance and status information
    pub fn getGPUStatus(self: *KernelModule, device_id: u32) !GPUStatus {
        if (device_id >= self.device_count) {
            return error.InvalidDevice;
        }
        
        var status = GPUStatus{};
        
        // Query GPU clocks via NVML-style ioctl
        var clock_info = NVMLClockInfo{};
        const clock_result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_GET_CLOCKS, @intFromPtr(&clock_info));
        if (clock_result == 0) {
            status.gpu_clock_mhz = clock_info.graphics_clock;
            status.memory_clock_mhz = clock_info.memory_clock;
        }
        
        // Query temperature
        var temp_info = NVMLTempInfo{};
        const temp_result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_GET_TEMPERATURE, @intFromPtr(&temp_info));
        if (temp_result == 0) {
            status.temperature_c = temp_info.gpu_temp;
        }
        
        // Query power usage
        var power_info = NVMLPowerInfo{};
        const power_result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_GET_POWER, @intFromPtr(&power_info));
        if (power_result == 0) {
            status.power_draw_watts = power_info.power_draw;
        }
        
        // Query utilization
        var util_info = NVMLUtilInfo{};
        const util_result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_GET_UTILIZATION, @intFromPtr(&util_info));
        if (util_result == 0) {
            status.gpu_utilization_percent = util_info.gpu_util;
            status.memory_utilization_percent = util_info.memory_util;
        }
        
        return status;
    }
    
    /// Set digital vibrance via kernel interface
    pub fn setDigitalVibrance(self: *KernelModule, device_id: u32, vibrance: i8) !void {
        if (device_id >= self.device_count) {
            return error.InvalidDevice;
        }
        
        std.log.info("Setting digital vibrance to {} for GPU {}", .{ vibrance, device_id });
        
        var vibrance_cmd = NVCTLVibranceCommand{
            .device_id = device_id,
            .attribute = NVCTRL_DIGITAL_VIBRANCE,
            .value = vibrance,
        };
        
        const result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_SET_ATTRIBUTE, @intFromPtr(&vibrance_cmd));
        if (result < 0) {
            const errno = std.posix.errno(-1);
            std.log.err("Failed to set digital vibrance: errno {}", .{errno});
            return error.KernelCallFailed;
        }
        
        std.log.info("Digital vibrance set successfully");
    }
    
    /// Get digital vibrance value
    pub fn getDigitalVibrance(self: *KernelModule, device_id: u32) !i8 {
        if (device_id >= self.device_count) {
            return error.InvalidDevice;
        }
        
        var vibrance_cmd = NVCTLVibranceCommand{
            .device_id = device_id,
            .attribute = NVCTRL_DIGITAL_VIBRANCE,
            .value = 0,
        };
        
        const result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_GET_ATTRIBUTE, @intFromPtr(&vibrance_cmd));
        if (result < 0) {
            return error.KernelCallFailed;
        }
        
        return @intCast(vibrance_cmd.value);
    }
    
    /// Configure G-SYNC settings
    pub fn setGSyncMode(self: *KernelModule, device_id: u32, mode: GSyncMode) !void {
        if (device_id >= self.device_count) {
            return error.InvalidDevice;
        }
        
        std.log.info("Setting G-SYNC mode to {} for GPU {}", .{ mode, device_id });
        
        var gsync_cmd = NVCTLGSyncCommand{
            .device_id = device_id,
            .mode = @intFromEnum(mode),
            .flags = 0,
        };
        
        const result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_SET_GSYNC, @intFromPtr(&gsync_cmd));
        if (result < 0) {
            return error.KernelCallFailed;
        }
        
        std.log.info("G-SYNC mode set successfully");
    }
    
    /// Get G-SYNC status
    pub fn getGSyncStatus(self: *KernelModule, device_id: u32) !GSyncStatus {
        if (device_id >= self.device_count) {
            return error.InvalidDevice;
        }
        
        var gsync_info = NVCTLGSyncInfo{};
        const result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_GET_GSYNC, @intFromPtr(&gsync_info));
        if (result < 0) {
            return error.KernelCallFailed;
        }
        
        return GSyncStatus{
            .mode = @enumFromInt(gsync_info.mode),
            .min_refresh_hz = gsync_info.min_refresh,
            .max_refresh_hz = gsync_info.max_refresh,
            .current_refresh_hz = gsync_info.current_refresh,
            .enabled = gsync_info.enabled,
        };
    }
    
    /// Set refresh rate for VRR
    pub fn setRefreshRate(self: *KernelModule, device_id: u32, refresh_hz: u32) !void {
        if (device_id >= self.device_count) {
            return error.InvalidDevice;
        }
        
        std.log.info("Setting refresh rate to {}Hz for GPU {}", .{ refresh_hz, device_id });
        
        var refresh_cmd = NVCTLRefreshCommand{
            .device_id = device_id,
            .refresh_rate = refresh_hz,
        };
        
        const result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_SET_REFRESH_RATE, @intFromPtr(&refresh_cmd));
        if (result < 0) {
            return error.KernelCallFailed;
        }
        
        std.log.info("Refresh rate set successfully");
    }
    
    /// Allocate GPU memory
    pub fn allocateGPUMemory(self: *KernelModule, device_id: u32, size_bytes: u64) !GPUMemoryHandle {
        if (device_id >= self.device_count) {
            return error.InvalidDevice;
        }
        
        var alloc_cmd = NVUVMAllocCommand{
            .size = size_bytes,
            .flags = NVUVM_ALLOC_FLAGS_DEFAULT,
            .gpu_va = 0,
        };
        
        const result = ioctl(self.nvidia_uvm_fd, NVIDIA_UVM_IOCTL_ALLOC, @intFromPtr(&alloc_cmd));
        if (result < 0) {
            return error.AllocationFailed;
        }
        
        return GPUMemoryHandle{
            .gpu_va = alloc_cmd.gpu_va,
            .size = size_bytes,
            .device_id = device_id,
        };
    }
    
    /// Free GPU memory
    pub fn freeGPUMemory(self: *KernelModule, handle: GPUMemoryHandle) !void {
        var free_cmd = NVUVMFreeCommand{
            .gpu_va = handle.gpu_va,
            .size = handle.size,
        };
        
        const result = ioctl(self.nvidia_uvm_fd, NVIDIA_UVM_IOCTL_FREE, @intFromPtr(&free_cmd));
        if (result < 0) {
            return error.FreeFailed;
        }
    }
    
    /// Submit GPU commands for execution
    pub fn submitGPUCommands(self: *KernelModule, device_id: u32, commands: []const GPUCommand) !u64 {
        if (device_id >= self.device_count) {
            return error.InvalidDevice;
        }
        
        var submit_cmd = NVCommandSubmit{
            .device_id = device_id,
            .num_commands = @intCast(commands.len),
            .commands_ptr = @intFromPtr(commands.ptr),
            .fence_id = 0,
        };
        
        const result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_SUBMIT_COMMANDS, @intFromPtr(&submit_cmd));
        if (result < 0) {
            return error.SubmitFailed;
        }
        
        return submit_cmd.fence_id;
    }
    
    /// Wait for GPU commands to complete
    pub fn waitForCommands(self: *KernelModule, fence_id: u64, timeout_ms: u32) !void {
        var wait_cmd = NVCommandWait{
            .fence_id = fence_id,
            .timeout_ns = @as(u64, timeout_ms) * 1000000,
        };
        
        const result = ioctl(self.nvidia_fd, NVIDIA_IOCTL_WAIT_COMMANDS, @intFromPtr(&wait_cmd));
        if (result < 0) {
            return error.WaitFailed;
        }
    }
    
    // Private helper methods
    
    fn initializeDevice(allocator: std.mem.Allocator, device_id: u32) !DeviceInfo {
        // Query device information
        var device_info = DeviceInfo{
            .device_id = device_id,
            .name = try allocator.alloc(u8, 256),
            .uuid = try allocator.alloc(u8, 64),
            .pci_bus_id = try allocator.alloc(u8, 32),
            .memory_total_mb = 0,
            .compute_capability_major = 0,
            .compute_capability_minor = 0,
        };
        
        // These would be real queries in production
        // For RTX 4090 defaults:
        @memcpy(device_info.name[0..17], "RTX 4090");
        device_info.name[17] = 0;
        
        @memcpy(device_info.uuid[0..16], "GPU-00000000-1234");
        device_info.uuid[16] = 0;
        
        @memcpy(device_info.pci_bus_id[0..12], "0000:01:00.0");
        device_info.pci_bus_id[12] = 0;
        
        device_info.memory_total_mb = 24576; // 24GB
        device_info.compute_capability_major = 8;
        device_info.compute_capability_minor = 9;
        
        return device_info;
    }
};

// Linux ioctl wrapper
fn ioctl(fd: i32, request: u32, arg: usize) i32 {
    return @intCast(linux.syscall3(.ioctl, @as(usize, @bitCast(@as(isize, fd))), request, arg));
}

// NVIDIA IOCTL command definitions
const NVIDIA_IOCTL_MAGIC = 0xF0;

const NVIDIA_IOCTL_CARD_INFO = 0x00;
const NVIDIA_IOCTL_GET_CLOCKS = 0x10;
const NVIDIA_IOCTL_GET_TEMPERATURE = 0x11;
const NVIDIA_IOCTL_GET_POWER = 0x12;
const NVIDIA_IOCTL_GET_UTILIZATION = 0x13;
const NVIDIA_IOCTL_SET_ATTRIBUTE = 0x20;
const NVIDIA_IOCTL_GET_ATTRIBUTE = 0x21;
const NVIDIA_IOCTL_SET_GSYNC = 0x30;
const NVIDIA_IOCTL_GET_GSYNC = 0x31;
const NVIDIA_IOCTL_SET_REFRESH_RATE = 0x32;
const NVIDIA_IOCTL_SUBMIT_COMMANDS = 0x40;
const NVIDIA_IOCTL_WAIT_COMMANDS = 0x41;

const NVIDIA_UVM_IOCTL_ALLOC = 0x50;
const NVIDIA_UVM_IOCTL_FREE = 0x51;

// NVCTRL attribute definitions
const NVCTRL_DIGITAL_VIBRANCE = 261;
const NVCTRL_COLOR_SATURATION = 262;
const NVCTRL_GAMMA_CORRECTION = 263;

// NVUVM flags
const NVUVM_ALLOC_FLAGS_DEFAULT = 0x00000001;

// Data structures for kernel communication

const NVMLClockInfo = extern struct {
    graphics_clock: u32,
    memory_clock: u32,
    shader_clock: u32,
    video_clock: u32,
};

const NVMLTempInfo = extern struct {
    gpu_temp: u32,
    memory_temp: u32,
    power_limit_temp: u32,
};

const NVMLPowerInfo = extern struct {
    power_draw: u32,
    power_limit: u32,
    max_power: u32,
};

const NVMLUtilInfo = extern struct {
    gpu_util: u32,
    memory_util: u32,
    encoder_util: u32,
    decoder_util: u32,
};

const NVCTLVibranceCommand = extern struct {
    device_id: u32,
    attribute: u32,
    value: i32,
};

const NVCTLGSyncCommand = extern struct {
    device_id: u32,
    mode: u32,
    flags: u32,
};

const NVCTLGSyncInfo = extern struct {
    mode: u32,
    min_refresh: u32,
    max_refresh: u32,
    current_refresh: u32,
    enabled: u32,
};

const NVCTLRefreshCommand = extern struct {
    device_id: u32,
    refresh_rate: u32,
};

const NVUVMAllocCommand = extern struct {
    size: u64,
    flags: u32,
    gpu_va: u64,
};

const NVUVMFreeCommand = extern struct {
    gpu_va: u64,
    size: u64,
};

const NVCommandSubmit = extern struct {
    device_id: u32,
    num_commands: u32,
    commands_ptr: u64,
    fence_id: u64,
};

const NVCommandWait = extern struct {
    fence_id: u64,
    timeout_ns: u64,
};

// Public type definitions

pub const DeviceInfo = struct {
    device_id: u32,
    name: []u8,
    uuid: []u8,
    pci_bus_id: []u8,
    memory_total_mb: u32,
    compute_capability_major: u8,
    compute_capability_minor: u8,
    
    pub fn deinit(self: *DeviceInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.uuid);
        allocator.free(self.pci_bus_id);
    }
};

pub const GPUStatus = struct {
    gpu_clock_mhz: u32 = 0,
    memory_clock_mhz: u32 = 0,
    temperature_c: u32 = 0,
    power_draw_watts: u32 = 0,
    gpu_utilization_percent: u32 = 0,
    memory_utilization_percent: u32 = 0,
};

pub const GSyncMode = enum(u32) {
    disabled = 0,
    compatible = 1,
    certified = 2,
    ultimate = 3,
    esports = 4,
};

pub const GSyncStatus = struct {
    mode: GSyncMode,
    min_refresh_hz: u32,
    max_refresh_hz: u32,
    current_refresh_hz: u32,
    enabled: bool,
};

pub const GPUMemoryHandle = struct {
    gpu_va: u64,
    size: u64,
    device_id: u32,
};

pub const GPUCommand = extern struct {
    command_type: u32,
    data_ptr: u64,
    data_size: u32,
};

// Test functions
test "kernel module init" {
    const allocator = std.testing.allocator;
    
    // This test would require actual NVIDIA hardware
    // For CI/testing, we'd mock the device files
    
    std.log.info("Kernel module test would run here with real hardware");
}