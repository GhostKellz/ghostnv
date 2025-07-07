# GhostNV Kernel Integration Guide

## Overview

This document outlines how the GhostNV pure Zig NVIDIA driver integrates with the Ghost Kernel (linux-ghost) and how GhostShell should pull and utilize the driver directly.

## Architecture Integration

### Direct Kernel Compilation

Unlike traditional DKMS-based drivers, GhostNV is compiled directly into the Ghost Kernel:

```zig
// Ghost Kernel build.zig integration
const ghostnv = b.dependency("ghostnv", .{
    .target = target,
    .optimize = optimize,
    .enable_gaming_optimizations = true,
    .gpu_generations = &.{ .ada, .ampere, .turing },
});

kernel.root_module.addImport("ghostnv", ghostnv.module("ghostnv"));
```

### Driver Loading Sequence

1. **Early Boot Detection** - PCI enumeration identifies NVIDIA GPUs
2. **Architecture Selection** - Chooses optimal driver path per GPU
3. **Memory Management Setup** - Initializes VRAM pools and GART
4. **Command Pipeline Init** - Sets up GPU command submission
5. **Display Engine Start** - Enables output with VRR support

## GhostShell Integration

### Direct Driver Access

GhostShell can access the pure Zig driver through multiple interfaces:

#### 1. Kernel Module Interface
```zig
// Direct kernel module access
const ghostnv = @import("ghostnv");

pub fn initNvidiaControl() !void {
    const driver = try ghostnv.getDriver(0); // Primary GPU
    try driver.setDigitalVibrance(50);
    try driver.enableVRR(48, 165);
}
```

#### 2. Sysfs Interface
```bash
# Digital vibrance control
echo "50" > /sys/kernel/ghostnv/vibrance
echo "1" > /sys/kernel/ghostnv/vrr_enable
echo "48:165" > /sys/kernel/ghostnv/vrr_range
```

#### 3. Direct Hardware Access
```zig
// Low-level hardware control
const nvkms = @import("ghostnv").nvkms;

pub fn setVibranceHardware(vibrance: i16) !void {
    var interface = try nvkms.NvKmsInterface.init();
    defer interface.deinit();
    
    const displays = try interface.enumerateDisplays();
    for (displays) |display| {
        try interface.setDigitalVibrance(display, vibrance);
    }
}
```

### FFI Bridge for Existing Tools

For compatibility with existing nvcontrol Rust projects:

```c
// Generated C FFI header (ghostnv_ffi.h)
typedef struct {
    int8_t vibrance;           // -50 to 100
    int8_t saturation;         // -50 to 50  
    float gamma;               // 0.8 to 3.0
    bool preserve_skin_tones;
} GhostNVVibranceProfile;

typedef enum {
    GHOSTNV_GSYNC_DISABLED = 0,
    GHOSTNV_GSYNC_COMPATIBLE = 1,
    GHOSTNV_GSYNC_ULTIMATE = 2,
    GHOSTNV_FREESYNC = 3,
} GhostNVGSyncMode;

// Core API functions
GhostNVResult ghostnv_init(void);
GhostNVResult ghostnv_vibrance_set(int8_t vibrance);
GhostNVResult ghostnv_vibrance_get(int8_t* vibrance);
GhostNVResult ghostnv_gsync_enable(uint32_t display_id, GhostNVGSyncMode mode);
GhostNVResult ghostnv_gsync_disable(uint32_t display_id);
GhostNVResult ghostnv_vrr_set_range(uint32_t min_hz, uint32_t max_hz);
```

## Driver Components

### 1. Hardware Abstraction Layer (HAL)

**Location**: `zig-nvidia/src/hal/`

- **PCI Management** (`pci.zig`): GPU detection and enumeration
- **Memory Management** (`memory.zig`): VRAM and system memory pools
- **Command Pipeline** (`command.zig`): GPU work submission
- **Interrupt Handling** (`interrupt.zig`): GPU event processing

### 2. DRM Integration

**Location**: `zig-nvidia/src/drm/`

- **Driver Registration** (`driver.zig`): DRM subsystem integration
- **VRR Support**: Variable refresh rate for gaming
- **Atomic Modesetting**: Wayland-optimized display management

### 3. NVKMS Interface

**Location**: `zig-nvidia/src/nvkms/`

- **Hardware Interface** (`interface.zig`): Direct NVIDIA hardware access
- **Digital Vibrance**: Hardware LUT programming
- **Display Management**: Native nvidia-modeset communication

### 4. Color Management

**Location**: `zig-nvidia/src/color/`

- **Vibrance Engine** (`vibrance.zig`): Hardware-accelerated color processing
- **Profile Management**: Save/load vibrance profiles
- **Real-time Adjustment**: Zero-latency color changes

## Build Integration

### Ghost Kernel Build Configuration

```zig
// linux-ghost/build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // Enable NVIDIA support
    const nvidia_support = b.option(bool, "nvidia", "Enable NVIDIA GPU support") orelse true;
    const gaming_optimized = b.option(bool, "gaming", "Enable gaming optimizations") orelse false;
    
    const kernel = b.addExecutable(.{
        .name = "linux-ghost",
        .root_source_file = b.path("src/kernel.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    if (nvidia_support) {
        const ghostnv = b.dependency("ghostnv", .{
            .target = target,
            .optimize = optimize,
            .gaming_optimized = gaming_optimized,
        });
        
        kernel.root_module.addImport("ghostnv", ghostnv.module("ghostnv"));
        
        // Add gaming-specific optimizations
        if (gaming_optimized) {
            kernel.defineCMacro("GHOSTNV_GAMING_OPTIMIZED", "1");
            kernel.defineCMacro("GHOSTNV_VRR_ENABLED", "1");
            kernel.defineCMacro("GHOSTNV_LOW_LATENCY", "1");
        }
    }
}
```

### GhostShell Integration

```zig
// ghostshell/src/gpu.zig
const std = @import("std");
const ghostnv = @import("ghostnv");

pub const GpuManager = struct {
    driver: *ghostnv.Driver,
    
    pub fn init() !GpuManager {
        const driver = try ghostnv.getDriver(0);
        return GpuManager{ .driver = driver };
    }
    
    pub fn setVibrance(self: *GpuManager, vibrance: i8) !void {
        try self.driver.setDigitalVibrance(vibrance);
    }
    
    pub fn enableGaming(self: *GpuManager) !void {
        try self.driver.enableVRR(48, 165);
        try self.driver.setLatencyMode(.ultra_low);
        try self.driver.setPowerMode(.maximum_performance);
    }
};
```

## Performance Optimizations

### Gaming Mode

```zig
pub const GamingConfig = struct {
    vrr_enabled: bool = true,           // Variable refresh rate
    low_latency: bool = true,           // Ultra-low latency mode
    direct_submission: bool = true,     // Bypass CPU scheduling
    frame_pacing: FramePacing = .adaptive,
    power_mode: PowerMode = .maximum_performance,
};
```

### Memory Management

- **Zero-copy operations** between CPU and GPU
- **Unified memory addressing** for seamless access
- **Intelligent caching** with hardware-aware algorithms
- **NUMA-aware allocation** for multi-GPU systems

### Container Integration

```dockerfile
# Native GPU container support
FROM nvidia/cuda:12.3-devel-ubuntu22.04
# No additional runtime needed - built into kernel
RUN nvidia-smi  # Works directly with GhostNV
```

## API Reference

### Core Driver API

```zig
pub const Driver = struct {
    // Digital vibrance control (-50 to 100)
    pub fn setDigitalVibrance(self: *Driver, vibrance: i8) !void;
    pub fn getDigitalVibrance(self: *Driver) !i8;
    
    // VRR/G-SYNC control
    pub fn enableVRR(self: *Driver, min_hz: u32, max_hz: u32) !void;
    pub fn disableVRR(self: *Driver) !void;
    pub fn setRefreshRate(self: *Driver, hz: u32) !void;
    
    // Performance modes
    pub fn setLatencyMode(self: *Driver, mode: LatencyMode) !void;
    pub fn setPowerMode(self: *Driver, mode: PowerMode) !void;
    
    // Memory management
    pub fn allocateVRAM(self: *Driver, size: usize) !*VRAMBuffer;
    pub fn freeVRAM(self: *Driver, buffer: *VRAMBuffer) !void;
    
    // Command submission
    pub fn submitWork(self: *Driver, work: *GPUWork) !void;
    pub fn waitForCompletion(self: *Driver, timeout_ms: u32) !void;
};
```

### Configuration Management

```zig
pub const Config = struct {
    vibrance: i8 = 0,
    vrr_enabled: bool = false,
    vrr_min_hz: u32 = 48,
    vrr_max_hz: u32 = 165,
    gaming_mode: bool = false,
    
    pub fn load() !Config;
    pub fn save(self: Config) !void;
    pub fn apply(self: Config, driver: *Driver) !void;
};
```

## Testing Integration

### Unit Tests

```zig
test "driver initialization" {
    const driver = try ghostnv.getDriver(0);
    try std.testing.expect(driver.isReady());
}

test "vibrance control" {
    const driver = try ghostnv.getDriver(0);
    try driver.setDigitalVibrance(50);
    const vibrance = try driver.getDigitalVibrance();
    try std.testing.expect(vibrance == 50);
}
```

### Hardware-in-the-Loop Testing

```bash
# Automated GPU testing
./test-runner --gpu-required --test-all-generations
```

## Deployment

### Kernel Installation

```bash
# Build Ghost Kernel with GhostNV
cd linux-ghost
zig build -Dnvidia=true -Dgaming=true
sudo make install

# Reboot into Ghost Kernel
sudo grub-set-default "Ghost Kernel with GhostNV"
sudo reboot
```

### Runtime Verification

```bash
# Check driver status
dmesg | grep ghostnv
cat /proc/ghostnv/status

# Test GPU functionality
nvidia-smi  # Should work immediately
nvctl vibrance 50  # Hardware vibrance control
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - Check PCI enumeration: `lspci | grep NVIDIA`
   - Verify kernel module loading: `dmesg | grep ghostnv`

2. **Performance Issues**
   - Enable gaming mode: `echo gaming > /sys/kernel/ghostnv/mode`
   - Check power management: `cat /sys/kernel/ghostnv/power_mode`

3. **Display Issues**
   - Verify VRR support: `cat /sys/kernel/ghostnv/vrr_caps`
   - Check display connectivity: `cat /sys/kernel/ghostnv/displays`

### Debug Interface

```bash
# Enable debug logging
echo 1 > /sys/kernel/ghostnv/debug
dmesg -w | grep ghostnv

# Performance monitoring
cat /sys/kernel/ghostnv/stats
```

## Future Enhancements

### Planned Features

1. **AI Integration**: CUDA runtime optimization
2. **Video Acceleration**: Enhanced NVENC support
3. **Multi-GPU**: Advanced SLI/NVLink management
4. **Container Runtime**: Native OCI GPU runtime
5. **Wayland Protocol**: Native Wayland GPU acceleration

### Contribution Guidelines

- Follow Zig style guidelines
- Include comprehensive tests
- Document hardware interactions
- Maintain compatibility with existing tools

---

**Note**: This integration guide assumes Ghost Kernel environment. For traditional Linux kernels, use the DKMS compatibility layer provided in `dkms/` directory.