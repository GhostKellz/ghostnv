const std = @import("std");

// Main GhostNV module for external kernel integration
pub const kernel = @import("kernel/integration.zig");
pub const driver = @import("kernel/module.zig");
pub const display = @import("display/engine.zig");
pub const video = @import("video/processor.zig");
pub const audio = @import("audio/rtx_voice.zig");
pub const cuda = @import("cuda/runtime.zig");
pub const memory = @import("hal/memory.zig");
pub const command = @import("hal/command.zig");
pub const arch_config = @import("arch/config.zig");

// Re-export key types for GhostKernel integration
pub const GhostKernelIntegration = kernel.GhostKernelIntegration;
pub const GhostKernelAPI = kernel.GhostKernelAPI;
pub const NvidiaKernelModule = driver.NvidiaKernelModule;
// Driver selector is in the root directory - not accessible from remote dependencies
// pub const DriverSelector = @import("../driver_selector.zig").DriverSelector;

// Module metadata
pub const MODULE_NAME = "ghostnv";
pub const MODULE_VERSION = "575.0.0-ghost";
pub const MODULE_LICENSE = "GPL-compatible";

// Main initialization function for GhostKernel
pub fn init_for_ghostkernel(allocator: std.mem.Allocator, kernel_ctx: *anyopaque) !*GhostKernelIntegration {
    _ = allocator;
    return try kernel.GhostKernelAPI.ghostnv_init(kernel_ctx);
}

// Test function
test "ghostnv module exports" {
    const testing = std.testing;
    
    // Test that all modules compile
    _ = kernel;
    _ = driver;
    _ = display;
    _ = video;
    _ = audio;
    _ = cuda;
    _ = memory;
    _ = command;
    _ = arch_config;
    
    // Test module metadata
    try testing.expect(std.mem.eql(u8, MODULE_NAME, "ghostnv"));
    try testing.expect(std.mem.startsWith(u8, MODULE_VERSION, "575.0.0"));
}