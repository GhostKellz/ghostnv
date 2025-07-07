# GhostNV Installation Guide

**Complete installation instructions for the Pure Zig NVIDIA Driver**

---

## 🎯 Quick Install (Recommended)

```bash
# Clone and build
git clone https://github.com/yourusername/ghostnv.git
cd ghostnv

# Build everything
zig build -Doptimize=ReleaseFast

# Build FFI library for Rust integration
zig build ffi

# Generate C headers
zig build ffi-headers

# Install system-wide
sudo ./install.sh
```

---

## 📋 Prerequisites

### System Requirements
- **Linux Kernel**: 6.0+ (optimized for 6.15.x)
- **Zig Compiler**: 0.13.0 or newer
- **GPU**: RTX 30/40 series (primary focus), RTX 20 series supported
- **RAM**: 4GB minimum, 8GB recommended
- **Architecture**: x86_64

### Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential linux-headers-$(uname -r) dkms git curl

# Arch Linux
sudo pacman -S base-devel linux-headers dkms git curl

# Fedora/RHEL
sudo dnf groupinstall "Development Tools"
sudo dnf install kernel-devel dkms git curl

# Install Zig (if not already installed)
curl -L https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz | tar -xJ
sudo mv zig-linux-x86_64-0.13.0 /usr/local/zig
echo 'export PATH="/usr/local/zig:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## 🛠️ Build Options

### Basic Build
```bash
# Standard release build
zig build -Doptimize=ReleaseFast

# Debug build with symbols
zig build -Doptimize=Debug -Ddebug-driver=true
```

### Feature-Specific Builds
```bash
# Gaming optimizations (RTX 40 series)
zig build -Doptimize=ReleaseFast -Dgaming=true -Dvrr=true -Dframe-gen=true

# Pure Zig driver only
zig build -Doptimize=ReleaseFast -Dpure-zig=true

# Container runtime support
zig build container
zig build oci

# All tools and features
zig build -Doptimize=ReleaseFast -Dcuda=true -Dnvenc=true -Dgaming=true -Dvrr=true
```

### Legacy/Compatibility Builds
```bash
# Legacy C driver fallback
zig build -Doptimize=ReleaseFast -Ddriver-mode=legacy_c

# Hybrid mode (C + Zig)
zig build -Doptimize=ReleaseFast -Ddriver-mode=hybrid
```

---

## 🏗️ Component Installation

### 1. Core Driver Module

```bash
# Build kernel modules
zig build modules

# Manual module installation
sudo insmod zig-out/nvidia.ko
sudo insmod zig-out/nvidia-uvm.ko
sudo insmod zig-out/nvidia-modeset.ko

# Or use DKMS (recommended)
sudo dkms add .
sudo dkms build ghostnv/1.0.0
sudo dkms install ghostnv/1.0.0
```

### 2. FFI Shared Library

```bash
# Build and install FFI library for Rust integration
zig build ffi
zig build ffi-headers

# Install system-wide
sudo cp zig-out/lib/libghostnv.so /usr/local/lib/
sudo cp zig-out/lib/libghostnv.so.1.0.0 /usr/local/lib/
sudo cp zig-out/include/ghostnv_ffi.h /usr/local/include/
sudo cp zig-out/ghostnv.pc /usr/local/lib/pkgconfig/
sudo ldconfig

# Verify installation
pkg-config --modversion ghostnv
```

### 3. Tools and Utilities

```bash
# Build all tools
zig build ghostvibrance
zig build container
zig build oci

# Install to system
sudo cp zig-out/bin/ghostvibrance /usr/local/bin/
sudo cp zig-out/bin/ghostnv-container /usr/local/bin/
sudo cp zig-out/bin/ghostnv-container-oci /usr/local/bin/

# Add to PATH (if needed)
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
```

### 4. Container Runtime Setup

```bash
# Register with Docker
sudo ghostnv-container-oci register

# Or manually configure Docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "ghostnv": {
      "path": "/usr/local/bin/ghostnv-container-oci"
    }
  }
}
EOF

sudo systemctl restart docker

# Test container runtime
docker run --runtime=ghostnv --gpus all nvidia/cuda:12.3-devel nvidia-smi
```

---

## 🔧 Integration with Existing Projects

### nvcontrol (Rust) Integration

If you have the existing nvcontrol Rust project:

```bash
# In your nvcontrol project directory
cd nvcontrol

# Add to Cargo.toml
cat >> Cargo.toml << EOF

[dependencies]
libc = "0.2"
bindgen = "0.69"

[build-dependencies]
bindgen = "0.69"
EOF

# Create build.rs (see NVCONTROL_INTEGRATION.md for full content)
cat > build.rs << 'EOF'
use bindgen;
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-link-lib=ghostnv");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    
    let bindings = bindgen::Builder::default()
        .header("/usr/local/include/ghostnv_ffi.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ghostnv_bindings.rs"))
        .expect("Couldn't write bindings!");
}
EOF

# Rebuild nvcontrol with GhostNV support
cargo build --release

# Test integration
nvctl vibrance 50
nvctl gsync enable
```

---

## ⚙️ Configuration

### Driver Selection

Create `/etc/ghostnv/config.toml`:
```toml
[driver]
# auto, pure_zig, hybrid, legacy_c
mode = "auto"

# Force specific GPU support
force_rtx_40_optimizations = true

[features]
digital_vibrance = true
gsync_support = true
vrr_support = true
container_runtime = true
frame_generation = true

[performance]
# power_save, balanced, performance, max
level = "performance"

# Enable for competitive gaming
ultra_low_latency = true

[logging]
level = "info"  # debug, info, warn, error
file = "/var/log/ghostnv.log"
```

### SystemD Service

Create `/etc/systemd/system/ghostnv.service`:
```ini
[Unit]
Description=GhostNV Pure Zig NVIDIA Driver
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/ghostnv-driver-init
ExecStop=/usr/local/bin/ghostnv-driver-cleanup
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ghostnv
sudo systemctl start ghostnv
```

### Udev Rules

Create `/etc/udev/rules.d/99-ghostnv.rules`:
```
# GhostNV device permissions
KERNEL=="nvidia*", GROUP="video", MODE="0664"
KERNEL=="nvidiactl", GROUP="video", MODE="0664"
KERNEL=="nvidia-uvm", GROUP="video", MODE="0664"
KERNEL=="nvidia-modeset", GROUP="video", MODE="0664"

# Auto-load modules
ACTION=="add", SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{class}=="0x030000", RUN+="/sbin/modprobe nvidia"
```

Reload udev rules:
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

---

## 🧪 Testing Installation

### Basic Functionality Test

```bash
# Check driver loaded
lsmod | grep nvidia

# Check devices
ls -la /dev/nvidia*

# Test digital vibrance
ghostvibrance --test

# Test G-SYNC
ghostnv gsync status

# Test container runtime
ghostnv-container devices
```

### Performance Benchmarks

```bash
# Build and run benchmarks
zig build benchmarks
zig-out/bin/ghostnv-benchmarks

# CUDA test
zig build cuda-test
zig-out/bin/cuda-test

# Gaming performance test
zig build gaming-test
zig-out/bin/gaming-test

# VRR test
zig build vrr-test
zig-out/bin/vrr-test
```

### Integration Tests

```bash
# Test Rust FFI integration
cd tests/rust-integration
cargo test

# Test Docker container runtime
docker run --runtime=ghostnv --gpus all \
  -v $(pwd)/tests:/tests \
  nvidia/cuda:12.3-devel \
  /tests/container-test.sh

# Test nvcontrol integration (if available)
nvctl vibrance 75
nvctl gsync enable
nvctl get temp
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Module Loading Fails
```bash
# Check for conflicts
lsmod | grep nvidia
sudo rmmod nvidia*

# Check kernel headers
ls /lib/modules/$(uname -r)/build

# Rebuild modules
zig build modules
sudo dkms remove ghostnv/1.0.0 --all
sudo dkms install ghostnv/1.0.0
```

#### 2. FFI Library Not Found
```bash
# Check library installation
ldconfig -p | grep ghostnv

# Reinstall library
sudo rm /usr/local/lib/libghostnv*
zig build ffi
sudo cp zig-out/lib/libghostnv* /usr/local/lib/
sudo ldconfig
```

#### 3. Permission Denied
```bash
# Add user to video group
sudo usermod -aG video $USER

# Fix device permissions
sudo chmod 666 /dev/nvidia*

# Check SELinux/AppArmor
sudo setenforce 0  # Temporarily disable SELinux
```

#### 4. Container Runtime Issues
```bash
# Check Docker daemon
sudo systemctl status docker

# Verify OCI runtime
ghostnv-container-oci --version

# Re-register runtime
sudo ghostnv-container-oci register
sudo systemctl restart docker
```

### Debug Mode

```bash
# Enable debug logging
export GHOSTNV_DEBUG=1
export GHOSTNV_LOG_LEVEL=debug

# Run with debug output
ghostnv version
ghostvibrance --debug 50
ghostnv-container --debug devices
```

### Collect Debug Information

```bash
# Generate debug report
sudo ghostnv debug-report > /tmp/ghostnv-debug.log

# System information
uname -a
lspci | grep -i nvidia
nvidia-smi  # If available
dmesg | grep -i nvidia
journalctl -u ghostnv
```

---

## 🔄 Uninstallation

### Complete Removal

```bash
# Stop services
sudo systemctl stop ghostnv
sudo systemctl disable ghostnv

# Remove modules
sudo rmmod nvidia*
sudo dkms remove ghostnv/1.0.0 --all

# Remove files
sudo rm -f /usr/local/bin/ghostnv*
sudo rm -f /usr/local/lib/libghostnv*
sudo rm -f /usr/local/include/ghostnv_ffi.h
sudo rm -f /usr/local/lib/pkgconfig/ghostnv.pc
sudo rm -rf /etc/ghostnv/
sudo rm -f /etc/systemd/system/ghostnv.service
sudo rm -f /etc/udev/rules.d/99-ghostnv.rules

# Update system
sudo ldconfig
sudo systemctl daemon-reload
sudo udevadm control --reload-rules
```

### Restore Original NVIDIA Driver

```bash
# Reinstall official NVIDIA driver
sudo apt install nvidia-driver-545  # Ubuntu/Debian
sudo pacman -S nvidia                # Arch Linux
sudo dnf install nvidia-driver       # Fedora

# Or download from NVIDIA website
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/545.29.06/NVIDIA-Linux-x86_64-545.29.06.run
sudo chmod +x NVIDIA-Linux-x86_64-545.29.06.run
sudo ./NVIDIA-Linux-x86_64-545.29.06.run
```

---

## 📞 Support

### Getting Help

- **Documentation**: Check README.md and all *.md files in the project
- **Issues**: Report bugs on GitHub Issues
- **Logs**: Always include debug logs with issue reports
- **Hardware**: Specify exact GPU model and driver version

### Contributing

- **Bug Reports**: Use the provided templates
- **Feature Requests**: Explain use case and benefit
- **Pull Requests**: Follow coding standards and include tests
- **Testing**: Help test on different hardware configurations

This installation guide provides comprehensive instructions for getting GhostNV running on your system with all features enabled!