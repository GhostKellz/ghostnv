# GhostNV Documentation

Welcome to **GhostNV** by CK Technology - a modern, open-source NVIDIA GPU driver framework built with Zig for optimal Linux performance.

## 🚀 Getting Started

### Prerequisites

- **Zig 0.15.0-dev** or later
- Linux kernel headers (matching your running kernel)
- GCC/Clang for kernel module compilation
- Git for source management
- Root access for driver installation

### Quick Setup

```bash
# Clone the repository
git clone <your-repo-url> ghostnv
cd ghostnv

# Initialize submodules (if any)
git submodule update --init --recursive

# Build the project
zig build

# Install (requires root)
sudo zig build install
```

## 📋 Project Structure

```
ghostnv/
├── kernel-open/          # NVIDIA open-gpu-kernel-modules source
├── nouveau/              # Legacy NVIDIA OSS driver components
├── src/                  # Core NVIDIA module source code
├── zig/
│   └── ghostnv.zig       # Zig build orchestrator and glue code
├── patches/              # Driver patches organized by version
├── build.zig             # Main Zig build configuration
├── CLAUDE.md             # AI assistant instructions
├── DOCS.md               # This documentation file
├── README.md             # Project overview
└── LICENSE               # License information
```

## 🔧 Build System

GhostNV uses a Zig-based build system that replaces traditional Makefiles:

### Build Commands

```bash
# Standard build
zig build

# Debug build with verbose output
zig build -Doptimize=Debug

# Release build
zig build -Doptimize=ReleaseFast

# Clean build artifacts
zig build clean
```

### Build Modes

- **legacy**: Standard NVIDIA open driver build
- **patched**: Applies performance/compatibility patches
- **real-time**: Optimized for low-latency applications

## 🩹 Patch System

The modular patch system supports different NVIDIA driver versions:

### Patch Organization

```
patches/
├── v575/                 # NVIDIA driver version 575.x
│   ├── performance.patch
│   ├── wayland.patch
│   └── audio.patch
├── v580/                 # NVIDIA driver version 580.x
└── common/               # Universal patches
```

### Applying Patches

Patches are automatically applied during the build process based on the detected NVIDIA driver version.

## 🎮 Features

### Core Driver Features

- **Wayland Optimization**: Enhanced performance for modern Linux desktops
- **Real-time Scheduling**: Low-latency support for gaming and streaming
- **Memory Management**: Improved GPU memory allocation and cleanup
- **Power Management**: Better power efficiency and thermal control

### Optional Features

- **RTX Audio**: GPU-accelerated audio filtering (experimental)
- **NVENC Integration**: Hardware-accelerated video encoding
- **CUDA Optimization**: Enhanced compute performance

## 🛠️ Development

### Setting Up Development Environment

1. **Install Zig**: Download from [ziglang.org](https://ziglang.org)
2. **Clone Repository**: `git clone <repo-url>`
3. **Install Dependencies**: Ensure kernel headers are installed
4. **Test Build**: Run `zig build` to verify setup

### Code Style

- Follow Zig's standard formatting (`zig fmt`)
- Use descriptive variable names
- Comment complex algorithms
- Prefer explicit over implicit code

### Testing

```bash
# Run unit tests
zig build test

# Test specific module
zig build test -- --filter "module_name"

# Integration tests (requires hardware)
zig build test-integration
```

## 📊 Performance Tuning

### GPU Configuration

```bash
# Check current GPU settings
ghostnv status

# Enable performance mode
ghostnv mode performance

# Enable power-saving mode
ghostnv mode powersave

# Real-time mode for gaming
ghostnv mode realtime
```

### Memory Optimization

- Monitor GPU memory usage with `ghostnv memory`
- Adjust buffer sizes in configuration
- Use memory pools for frequent allocations

## 🐛 Troubleshooting

### Common Issues

**Build Failures**
- Ensure Zig version compatibility
- Check kernel headers are installed
- Verify GCC/Clang availability

**Driver Loading Issues**
- Check kernel module dependencies
- Verify NVIDIA hardware compatibility
- Review dmesg output for errors

**Performance Problems**
- Monitor GPU utilization
- Check power management settings
- Verify Wayland compositor compatibility

### Debug Information

```bash
# Enable debug logging
ghostnv debug on

# View system information
ghostnv info

# Check driver status
ghostnv status --verbose
```

## 🤝 Contributing

### Getting Started

1. **Fork the Repository**: Create your own copy
2. **Create Feature Branch**: `git checkout -b feature/new-feature`
3. **Make Changes**: Implement your improvements
4. **Test Changes**: Ensure all tests pass
5. **Submit Pull Request**: Describe your changes

### Development Workflow

1. **Issue Discussion**: Discuss major changes in issues first
2. **Code Review**: All changes require peer review
3. **Testing**: Comprehensive testing on multiple configurations
4. **Documentation**: Update docs for user-facing changes

### Code Guidelines

- **Safety First**: Kernel code must be robust and safe
- **Performance**: Optimize for common use cases
- **Compatibility**: Support multiple NVIDIA generations
- **Documentation**: Comment complex kernel interactions

## 📚 Resources

### NVIDIA Documentation

- [NVIDIA Open GPU Kernel Modules](https://github.com/NVIDIA/open-gpu-kernel-modules)
- [NVIDIA Developer Documentation](https://docs.nvidia.com/cuda/)
- [Linux Kernel Module Programming](https://www.kernel.org/doc/html/latest/)

### Zig Resources

- [Zig Language Reference](https://ziglang.org/documentation/master/)
- [Zig Standard Library](https://ziglang.org/documentation/master/std/)
- [Zig Build System](https://ziglang.org/learn/build-system/)

### Linux Graphics

- [Wayland Protocol](https://wayland.freedesktop.org/docs/html/)
- [DRM/KMS Documentation](https://www.kernel.org/doc/html/latest/gpu/drm-kms.html)
- [Linux GPU Driver Development](https://www.kernel.org/doc/html/latest/gpu/)

## 🔒 Security

### Reporting Security Issues

- **Email**: security@ck-technology.com
- **GPG Key**: Available on project website
- **Response Time**: 48 hours for acknowledgment

### Security Best Practices

- Regular security audits
- Minimal privilege requirements
- Secure coding practices
- Memory safety emphasis

## 📄 License

GhostNV is licensed under [LICENSE](LICENSE). This project is completely free and open-source, reflecting CK Technology's commitment to the Linux community.

## 🏢 About CK Technology

CK Technology is dedicated to advancing open-source GPU driver technology because Linux deserves better. Our mission is to provide the community with high-quality, performance-optimized drivers that push the boundaries of what's possible with open-source graphics technology.

---

*For additional support, visit our community forums or submit issues on GitHub.*