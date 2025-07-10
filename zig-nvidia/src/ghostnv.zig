const std = @import("std");

// Main GhostNV module for external kernel integration
pub const kernel = @import("kernel/integration.zig");
pub const driver = @import("kernel/module.zig");
pub const display = @import("display/engine.zig");
pub const video = @import("video/processor.zig");
pub const audio = @import("audio/pipewire_integration.zig");
pub const cuda = @import("cuda/runtime.zig");
pub const memory = @import("hal/memory.zig");
pub const command = @import("hal/command.zig");
pub const arch_config = @import("arch/config.zig");
pub const color_vibrance = @import("color/vibrance.zig");
pub const container_runtime = @import("container/runtime.zig");
pub const drm_driver = struct {
    pub const DisplayInfo = struct {
        name: []const u8,
        width: u32,
        height: u32,
        refresh_rate: u32,
        connected: bool,
    };
    
    pub const DrmDriver = struct {
        allocator: std.mem.Allocator,
        
        pub fn init(allocator: std.mem.Allocator) !DrmDriver {
            return DrmDriver{ .allocator = allocator };
        }
        
        pub fn deinit(self: *DrmDriver) void {
            _ = self;
        }
        
        pub fn get_displays(self: *DrmDriver) ![]DisplayInfo {
            _ = self;
            return &[_]DisplayInfo{};
        }
        
        pub fn register(self: *DrmDriver) !void {
            _ = self;
            // Stub implementation
        }
    };
    
    pub fn get_displays() ![]DisplayInfo {
        return &[_]DisplayInfo{};
    }
};

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

// Driver version information
pub const DriverVersion = struct {
    major: u32,
    minor: u32,
    patch: u32,
};

// Patch type enumeration
pub const PatchType = enum {
    performance,
    realtime,
    audio,
    debug,
    common,
};

// Build mode enumeration
pub const BuildMode = enum {
    legacy,
    patched,
    realtime,
    audio,
    debug,
};

// Patch manager for handling driver patches
pub const PatchManager = struct {
    allocator: std.mem.Allocator,
    patches_dir: []const u8,
    version: DriverVersion,

    pub fn init(allocator: std.mem.Allocator, patches_dir: []const u8, version: DriverVersion) PatchManager {
        return PatchManager{
            .allocator = allocator,
            .patches_dir = patches_dir,
            .version = version,
        };
    }
    
    pub fn detect_version(self: *PatchManager) !DriverVersion {
        _ = self;
        return DriverVersion{ .major = 575, .minor = 0, .patch = 0 };
    }
    
    pub fn apply_patches(self: *PatchManager, patch_types: []const PatchType) !void {
        _ = self;
        _ = patch_types;
        // Stub implementation for now
    }
    
    pub fn validate_patches(self: *PatchManager) !void {
        _ = self;
        // Stub implementation for now
    }
};

// Kernel builder for building driver modules  
pub const KernelBuilder = struct {
    allocator: std.mem.Allocator,
    mode: BuildMode,
    patch_manager: PatchManager,

    pub fn init(allocator: std.mem.Allocator, mode: BuildMode, patch_manager: PatchManager) KernelBuilder {
        return KernelBuilder{
            .allocator = allocator,
            .mode = mode,
            .patch_manager = patch_manager,
        };
    }

    pub fn build(self: *KernelBuilder) !void {
        _ = self;
        // Stub implementation for now
    }
};

// Audio manager for PipeWire integration
pub const AudioManager = struct {
    allocator: std.mem.Allocator,
    enabled: bool,

    pub fn init(allocator: std.mem.Allocator) AudioManager {
        return AudioManager{
            .allocator = allocator,
            .enabled = false,
        };
    }

    pub fn enable_pipewire_integration(self: *AudioManager) !void {
        self.enabled = true;
    }

    pub fn setup_hdmi_audio(self: *AudioManager) !void {
        _ = self;
        // Stub implementation
    }
};

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