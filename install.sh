#!/bin/bash

# GhostNV Installation Script
# Installs the Pure Zig NVIDIA Driver with all components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INSTALL_PREFIX="/usr/local"
CONFIG_DIR="/etc/ghostnv"
LOG_FILE="/var/log/ghostnv-install.log"
SERVICE_NAME="ghostnv"

# Logging
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_dependencies() {
    echo_info "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for required commands
    for cmd in zig make gcc; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$cmd")
        fi
    done
    
    # Check for kernel headers
    if [[ ! -d "/lib/modules/$(uname -r)/build" ]]; then
        missing_deps+=("linux-headers-$(uname -r)")
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo_error "Missing dependencies: ${missing_deps[*]}"
        echo_info "Please install them first:"
        echo_info "  Ubuntu/Debian: sudo apt install build-essential linux-headers-\$(uname -r) zig"
        echo_info "  Arch Linux: sudo pacman -S base-devel linux-headers zig"
        echo_info "  Fedora: sudo dnf groupinstall \"Development Tools\" && sudo dnf install kernel-devel zig"
        exit 1
    fi
    
    echo_success "All dependencies satisfied"
}

check_gpu() {
    echo_info "Checking for NVIDIA GPU..."
    
    if ! lspci | grep -i nvidia &> /dev/null; then
        echo_warning "No NVIDIA GPU detected"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo_success "NVIDIA GPU detected"
        lspci | grep -i nvidia
    fi
}

backup_existing() {
    echo_info "Backing up existing NVIDIA drivers..."
    
    local backup_dir="/var/backups/ghostnv-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup existing modules
    if [[ -f "/lib/modules/$(uname -r)/kernel/drivers/video/nvidia.ko" ]]; then
        cp "/lib/modules/$(uname -r)/kernel/drivers/video/nvidia"* "$backup_dir/" 2>/dev/null || true
        echo_info "Backed up existing NVIDIA modules to $backup_dir"
    fi
    
    # Backup existing configuration
    if [[ -d "/etc/nvidia" ]]; then
        cp -r "/etc/nvidia" "$backup_dir/" 2>/dev/null || true
    fi
    
    echo "$backup_dir" > /var/lib/ghostnv-backup-location
}

build_ghostnv() {
    echo_info "Building GhostNV components..."
    
    # Build main driver
    echo_info "Building Pure Zig NVIDIA driver..."
    zig build -Doptimize=ReleaseFast -Dpure-zig=true -Dgaming=true -Dvrr=true
    
    # Build FFI library
    echo_info "Building FFI shared library..."
    zig build ffi
    
    # Build tools
    echo_info "Building tools and utilities..."
    zig build ghostvibrance
    zig build container
    zig build oci
    
    # Generate headers
    echo_info "Generating C headers..."
    zig build ffi-headers
    
    # Build kernel modules
    echo_info "Building kernel modules..."
    zig build modules
    
    echo_success "Build completed successfully"
}

install_components() {
    echo_info "Installing GhostNV components..."
    
    # Create directories
    mkdir -p "$INSTALL_PREFIX/bin"
    mkdir -p "$INSTALL_PREFIX/lib"
    mkdir -p "$INSTALL_PREFIX/include"
    mkdir -p "$INSTALL_PREFIX/lib/pkgconfig"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "/var/lib/ghostnv"
    
    # Install binaries
    if [[ -f "zig-out/bin/ghostnv" ]]; then
        cp zig-out/bin/ghostnv "$INSTALL_PREFIX/bin/"
        chmod +x "$INSTALL_PREFIX/bin/ghostnv"
    fi
    
    if [[ -f "zig-out/bin/ghostvibrance" ]]; then
        cp zig-out/bin/ghostvibrance "$INSTALL_PREFIX/bin/"
        chmod +x "$INSTALL_PREFIX/bin/ghostvibrance"
    fi
    
    if [[ -f "zig-out/bin/ghostnv-container" ]]; then
        cp zig-out/bin/ghostnv-container "$INSTALL_PREFIX/bin/"
        chmod +x "$INSTALL_PREFIX/bin/ghostnv-container"
    fi
    
    if [[ -f "zig-out/bin/ghostnv-container-oci" ]]; then
        cp zig-out/bin/ghostnv-container-oci "$INSTALL_PREFIX/bin/"
        chmod +x "$INSTALL_PREFIX/bin/ghostnv-container-oci"
    fi
    
    # Install libraries
    if [[ -f "zig-out/lib/libghostnv.so" ]]; then
        cp zig-out/lib/libghostnv.so* "$INSTALL_PREFIX/lib/" 2>/dev/null || true
    fi
    
    # Install headers
    if [[ -f "zig-out/include/ghostnv_ffi.h" ]]; then
        cp zig-out/include/ghostnv_ffi.h "$INSTALL_PREFIX/include/"
    fi
    
    # Install pkg-config file
    if [[ -f "zig-out/ghostnv.pc" ]]; then
        cp zig-out/ghostnv.pc "$INSTALL_PREFIX/lib/pkgconfig/"
    fi
    
    # Update library cache
    ldconfig
    
    echo_success "Components installed successfully"
}

install_kernel_modules() {
    echo_info "Installing kernel modules..."
    
    # Remove existing NVIDIA modules
    echo_info "Removing existing NVIDIA modules..."
    rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null || true
    
    # Install new modules using DKMS
    if command -v dkms &> /dev/null; then
        echo_info "Installing with DKMS..."
        
        # Create DKMS configuration
        cat > dkms.conf << EOF
PACKAGE_NAME="ghostnv"
PACKAGE_VERSION="1.0.0"
CLEAN="make clean"
MAKE[0]="zig build modules"
BUILT_MODULE_NAME[0]="nvidia"
BUILT_MODULE_LOCATION[0]="zig-out/"
DEST_MODULE_LOCATION[0]="/kernel/drivers/video/"
AUTOINSTALL="yes"
EOF
        
        # Register with DKMS
        dkms add .
        dkms build ghostnv/1.0.0
        dkms install ghostnv/1.0.0
        
        echo_success "Kernel modules installed with DKMS"
    else
        echo_warning "DKMS not available, installing modules manually"
        
        # Manual installation
        local kernel_dir="/lib/modules/$(uname -r)/kernel/drivers/video"
        mkdir -p "$kernel_dir"
        
        if [[ -f "zig-out/nvidia.ko" ]]; then
            cp zig-out/nvidia*.ko "$kernel_dir/"
            depmod -a
            echo_success "Kernel modules installed manually"
        else
            echo_error "Kernel modules not found in zig-out/"
            return 1
        fi
    fi
}

configure_system() {
    echo_info "Configuring system..."
    
    # Create default configuration
    cat > "$CONFIG_DIR/config.toml" << EOF
[driver]
mode = "auto"
force_rtx_40_optimizations = true

[features]
digital_vibrance = true
gsync_support = true
vrr_support = true
container_runtime = true
frame_generation = true

[performance]
level = "performance"
ultra_low_latency = true

[logging]
level = "info"
file = "/var/log/ghostnv.log"
EOF
    
    # Create udev rules
    cat > /etc/udev/rules.d/99-ghostnv.rules << EOF
# GhostNV device permissions
KERNEL=="nvidia*", GROUP="video", MODE="0664"
KERNEL=="nvidiactl", GROUP="video", MODE="0664"
KERNEL=="nvidia-uvm", GROUP="video", MODE="0664"
KERNEL=="nvidia-modeset", GROUP="video", MODE="0664"

# Auto-load modules
ACTION=="add", SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{class}=="0x030000", RUN+="/sbin/modprobe nvidia"
EOF
    
    # Create systemd service
    cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=GhostNV Pure Zig NVIDIA Driver
After=multi-user.target

[Service]
Type=oneshot
ExecStart=$INSTALL_PREFIX/bin/ghostnv version
ExecReload=/bin/kill -HUP \$MAINPID
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and udev
    systemctl daemon-reload
    udevadm control --reload-rules
    udevadm trigger
    
    echo_success "System configuration completed"
}

setup_docker_integration() {
    echo_info "Setting up Docker integration..."
    
    if command -v docker &> /dev/null; then
        # Register GhostNV runtime
        if [[ -x "$INSTALL_PREFIX/bin/ghostnv-container-oci" ]]; then
            "$INSTALL_PREFIX/bin/ghostnv-container-oci" register 2>/dev/null || {
                echo_warning "Failed to auto-register Docker runtime"
                echo_info "Manual Docker configuration may be required"
            }
            
            # Restart Docker if running
            if systemctl is-active --quiet docker; then
                echo_info "Restarting Docker to apply runtime changes..."
                systemctl restart docker
                echo_success "Docker integration configured"
            fi
        fi
    else
        echo_info "Docker not found, skipping container runtime setup"
    fi
}

start_services() {
    echo_info "Starting GhostNV services..."
    
    # Load kernel modules
    modprobe nvidia
    modprobe nvidia_uvm
    modprobe nvidia_modeset
    
    # Enable and start service
    systemctl enable "$SERVICE_NAME"
    systemctl start "$SERVICE_NAME"
    
    echo_success "Services started successfully"
}

run_tests() {
    echo_info "Running basic functionality tests..."
    
    # Test driver loading
    if lsmod | grep -q nvidia; then
        echo_success "NVIDIA modules loaded"
    else
        echo_warning "NVIDIA modules not loaded"
    fi
    
    # Test device files
    if [[ -e "/dev/nvidia0" ]]; then
        echo_success "GPU device files present"
    else
        echo_warning "GPU device files not found"
    fi
    
    # Test tools
    if command -v ghostvibrance &> /dev/null; then
        if ghostvibrance --version &> /dev/null; then
            echo_success "GhostVibrance tool working"
        fi
    fi
    
    # Test FFI library
    if ldconfig -p | grep -q ghostnv; then
        echo_success "FFI library installed and cached"
    else
        echo_warning "FFI library not found in cache"
    fi
    
    echo_info "Basic tests completed"
}

print_summary() {
    echo
    echo_success "=== GhostNV Installation Complete ==="
    echo
    echo_info "Installed components:"
    echo "  • Pure Zig NVIDIA Driver"
    echo "  • Digital Vibrance Engine (GhostVibrance)"
    echo "  • G-SYNC and VRR Support"
    echo "  • Container Runtime"
    echo "  • FFI Library for Rust integration"
    echo
    echo_info "Next steps:"
    echo "  1. Reboot your system: sudo reboot"
    echo "  2. Test installation: ghostvibrance --version"
    echo "  3. Configure digital vibrance: ghostvibrance 50"
    echo "  4. Test G-SYNC: ghostnv gsync status"
    echo "  5. Test containers: ghostnv-container devices"
    echo
    echo_info "Configuration files:"
    echo "  • Driver config: $CONFIG_DIR/config.toml"
    echo "  • Logs: /var/log/ghostnv.log"
    echo "  • Service: systemctl status ghostnv"
    echo
    echo_info "For nvcontrol integration, see: NVCONTROL_INTEGRATION.md"
    echo_info "For troubleshooting, see: INSTALL.md"
    echo
    echo_success "Enjoy the best Linux NVIDIA driver experience! 🚀"
}

cleanup_on_error() {
    echo_error "Installation failed. Cleaning up..."
    
    # Remove installed files
    rm -f "$INSTALL_PREFIX/bin/ghostnv"*
    rm -f "$INSTALL_PREFIX/lib/libghostnv"*
    rm -f "$INSTALL_PREFIX/include/ghostnv_ffi.h"
    rm -f "$INSTALL_PREFIX/lib/pkgconfig/ghostnv.pc"
    rm -f "/etc/systemd/system/$SERVICE_NAME.service"
    rm -f "/etc/udev/rules.d/99-ghostnv.rules"
    rm -rf "$CONFIG_DIR"
    
    # Reload system configuration
    ldconfig
    systemctl daemon-reload
    udevadm control --reload-rules
    
    echo_info "Cleanup completed"
    exit 1
}

main() {
    echo_info "Starting GhostNV installation..."
    echo_info "Installation log: $LOG_FILE"
    
    # Set error trap
    trap cleanup_on_error ERR
    
    # Check prerequisites
    check_root
    check_dependencies
    check_gpu
    
    # Installation steps
    backup_existing
    build_ghostnv
    install_components
    install_kernel_modules
    configure_system
    setup_docker_integration
    start_services
    run_tests
    
    # Success
    print_summary
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "GhostNV Installation Script"
            echo "Usage: sudo ./install.sh [options]"
            echo "Options:"
            echo "  --help, -h    Show this help message"
            echo "  --force       Force installation without prompts"
            exit 0
            ;;
        --force)
            export FORCE_INSTALL=1
            shift
            ;;
        *)
            echo_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main installation
main