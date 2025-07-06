# Zig-NVIDIA: Pure Zig NVIDIA Open Driver

**Zig-NVIDIA** is a pure Zig implementation of the NVIDIA open driver, part of the GhostNV project. This experimental driver aims to provide blazing-fast performance and stability for Linux Wayland environments.

## 🎯 Features

- **Pure Zig Implementation**: Complete rewrite of NVIDIA open drivers in Zig
- **Wayland-Optimized**: Zero-copy buffer sharing and direct scanout
- **Memory Safe**: Zig's memory safety eliminates common driver bugs
- **Performance-First**: Optimized for minimal latency and maximum throughput
- **Modular Design**: Clean separation of HAL, DRM, and Wayland components

## 🏗️ Architecture

```
zig-nvidia/
├── src/
│   ├── main.zig              # Kernel module entry point
│   ├── hal/                  # Hardware Abstraction Layer
│   │   ├── pci.zig          # PCI device management
│   │   └── memory.zig       # GPU memory management
│   ├── device/               # Device Management
│   │   └── state.zig        # Device state and character interface
│   ├── drm/                  # DRM/KMS Implementation
│   │   └── driver.zig       # DRM driver and modesetting
│   └── wayland/              # Wayland Optimization
│       └── compositor.zig   # Zero-copy compositor interface
```

## 🚀 Build & Usage

### From GhostNV Root

```bash
# Build pure Zig driver
zig build pure-zig

# Test Wayland functionality
zig build wayland-test

# Run all tests including Zig driver
zig build test
```

### Direct Build

```bash
cd zig-nvidia
zig build
zig build test
```

## 🎮 Supported Features

### ✅ Implemented
- PCI device enumeration and management
- Basic GPU memory management (VRAM allocation)
- DRM/KMS framework with connector/CRTC management
- Wayland buffer management with dmabuf support
- Zero-copy direct scanout for fullscreen applications
- Character device interface (`/dev/nvidia*`)

### 🚧 In Progress
- GPU command submission and scheduling
- Hardware-accelerated composition
- Power management and clock gating
- CUDA compute support

### 📋 Planned
- Full OpenGL/Vulkan support
- Hardware video encoding/decoding
- Multi-GPU support
- Suspend/resume functionality

## 🔧 Kernel Integration

This driver is designed to be embedded directly into the Linux-Ghost kernel for maximum performance. Key integration points:

- **Zero DKMS**: Compiled directly into kernel
- **Optimized Syscalls**: Direct kernel function calls
- **Memory Efficiency**: Zig's compile-time optimizations
- **Type Safety**: Eliminates entire classes of kernel bugs

## 🏎️ Performance Goals

- **Sub-millisecond latency** for display updates
- **Zero-copy** buffer operations where possible
- **Direct scanout** for compatible applications
- **GPU-accelerated composition** for complex scenes
- **Wayland-native** optimization paths

## 🧪 Testing

The driver includes comprehensive tests for all major components:

```bash
# Test PCI enumeration
zig test src/hal/pci.zig

# Test memory management
zig test src/hal/memory.zig

# Test DRM functionality
zig test src/drm/driver.zig

# Test Wayland compositor
zig test src/wayland/compositor.zig
```

## 🔬 Experimental Status

**This is an experimental pure Zig implementation.** While functional for testing, it's not yet production-ready. Use the legacy driver (`zig build legacy`) for stable systems.

## 🌟 Why Zig?

- **Memory Safety**: Eliminates segfaults and memory leaks
- **Performance**: Zero-cost abstractions and compile-time optimizations
- **Simplicity**: Clean, readable code without C preprocessor magic
- **Debugging**: Better error messages and stack traces
- **Future-Proof**: Modern language designed for systems programming

## 🤝 Contributing

This is part of the larger GhostNV project. See the main [CLAUDE.md](../CLAUDE.md) for contribution guidelines and project goals.

## 📄 License

GPL v2 (matching NVIDIA open driver license)