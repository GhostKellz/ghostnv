const std = @import("std");
const linux = std.os.linux;
const print = std.debug.print;

const hal = @import("hal/pci.zig");
const device = @import("device/state.zig");
const memory = @import("hal/memory.zig");
const drm = @import("drm/driver.zig");

// Public exports for tools and tests
pub const hal_pci = hal;
pub const hal_memory = memory;
pub const hal_command = @import("hal/command.zig");
pub const hal_interrupt = @import("hal/interrupt.zig");
pub const device_state = device;
pub const drm_driver = drm;
pub const color_vibrance = @import("color/vibrance.zig");
pub const cuda_runtime = @import("cuda/runtime.zig");
pub const nvenc_encoder = @import("nvenc/encoder.zig");
pub const gaming_performance = @import("gaming/performance.zig");
pub const container_runtime = @import("container/runtime.zig");

pub const NVZIG_MODULE_NAME = "nvzig";
pub const NVZIG_VERSION_MAJOR = 1;
pub const NVZIG_VERSION_MINOR = 0;
pub const NVZIG_VERSION_PATCH = 0;

pub const NvzigError = error{
    InitializationFailed,
    DeviceNotFound,
    MemoryAllocationFailed,
    PciAccessFailed,
    InterruptRegistrationFailed,
    DrmRegistrationFailed,
    UnsupportedHardware,
};

pub const ModuleState = struct {
    initialized: bool = false,
    device_count: u32 = 0,
    devices: ?[]device.NvzigDevice = null,
    memory_manager: memory.MemoryManager,
    drm_driver: ?drm.DrmDriver = null,
    
    pub fn init(allocator: std.mem.Allocator) ModuleState {
        return ModuleState{
            .memory_manager = memory.MemoryManager.init(allocator),
        };
    }
    
    pub fn deinit(self: *ModuleState) void {
        if (self.devices) |devices| {
            for (devices) |*dev| {
                dev.deinit();
            }
            self.memory_manager.allocator.free(devices);
        }
        self.memory_manager.deinit();
    }
};

var global_state: ModuleState = undefined;

export fn nvzig_init_module() callconv(.C) c_int {
    print("nvzig: Initializing pure Zig NVIDIA driver v{}.{}.{}\n", .{
        NVZIG_VERSION_MAJOR, NVZIG_VERSION_MINOR, NVZIG_VERSION_PATCH
    });
    
    // Initialize global state with a simple allocator for kernel module
    // In real kernel module, we'd use kernel allocators
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    global_state = ModuleState.init(allocator);
    
    // Phase 1: PCI device enumeration
    const device_count = hal.enumerate_nvidia_devices() catch |err| {
        print("nvzig: Failed to enumerate NVIDIA devices: {}\n", .{err});
        return -1;
    };
    
    if (device_count == 0) {
        print("nvzig: No compatible NVIDIA devices found\n", .{});
        return -1;
    }
    
    print("nvzig: Found {} NVIDIA device(s)\n", .{device_count});
    global_state.device_count = device_count;
    
    // Phase 2: Initialize devices
    global_state.devices = allocator.alloc(device.NvzigDevice, device_count) catch {
        print("nvzig: Failed to allocate device array\n", .{});
        return -1;
    };
    
    for (global_state.devices.?, 0..) |*dev, i| {
        dev.* = device.NvzigDevice.init(allocator, @intCast(i)) catch |err| {
            print("nvzig: Failed to initialize device {}: {}\n", .{i, err});
            return -1;
        };
    }
    
    // Phase 3: Register DRM driver (optional, for display support)
    global_state.drm_driver = drm.DrmDriver.init(allocator) catch |err| {
        print("nvzig: Warning - Failed to initialize DRM driver: {}\n", .{err});
        // Continue without DRM support
    };
    
    if (global_state.drm_driver) |*driver| {
        driver.register() catch |err| {
            print("nvzig: Warning - Failed to register DRM driver: {}\n", .{err});
        };
    }
    
    global_state.initialized = true;
    print("nvzig: Driver initialization complete\n", .{});
    return 0;
}

export fn nvzig_exit_module() callconv(.C) void {
    print("nvzig: Shutting down pure Zig NVIDIA driver\n", .{});
    
    if (!global_state.initialized) return;
    
    // Unregister DRM driver
    if (global_state.drm_driver) |*driver| {
        driver.unregister();
    }
    
    // Clean up devices
    if (global_state.devices) |devices| {
        for (devices) |*dev| {
            dev.shutdown();
        }
    }
    
    global_state.deinit();
    print("nvzig: Driver shutdown complete\n", .{});
}

// Kernel module macros equivalent
// Functions are already exported with export fn declarations above

// Module information
export const __module_license: [*:0]const u8 = "GPL v2";
export const __module_author: [*:0]const u8 = "GhostNV Team";
export const __module_description: [*:0]const u8 = "Pure Zig NVIDIA Open Driver";
export const __module_version: [*:0]const u8 = "1.0.0";

// Test functions for userspace testing
pub fn main() !void {
    print("nvzig: Running in userspace test mode\n", .{});
    
    // Test initialization
    const result = nvzig_init_module();
    if (result != 0) {
        print("nvzig: Initialization failed with code {}\n", .{result});
        return;
    }
    
    // Simulate running for a bit
    std.time.sleep(1000000000); // 1 second
    
    // Test cleanup
    nvzig_exit_module();
}

test "module lifecycle" {
    // Test module init/exit cycle
    const result = nvzig_init_module();
    defer nvzig_exit_module();
    
    try std.testing.expect(result == 0);
    try std.testing.expect(global_state.initialized);
}