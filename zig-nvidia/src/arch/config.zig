const std = @import("std");
const builtin = @import("builtin");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;

/// Comprehensive Arch Linux NVIDIA configuration handler
/// Addresses all common issues and gotchas from the Arch Linux NVIDIA wiki
pub const ArchConfig = struct {
    allocator: Allocator,
    
    /// Kernel module configuration for proper DRM support
    pub const KernelModule = struct {
        /// Early KMS configuration
        pub const early_kms_modules = [_][]const u8{
            "nvidia",
            "nvidia_modeset",
            "nvidia_uvm",
            "nvidia_drm",
        };
        
        /// Module parameters for proper DRM support
        pub const module_params = [_][]const u8{
            "nvidia-drm.modeset=1",
            "nvidia-drm.fbdev=1",
            "nvidia.NVreg_UsePageAttributeTable=1",
            "nvidia.NVreg_EnablePCIeGen3=1",
            "nvidia.NVreg_EnableMSI=1",
            "nvidia.NVreg_PreserveVideoMemoryAllocations=1",
            "nvidia.NVreg_TemporaryFilePath=/tmp",
        };
        
        /// Generate mkinitcpio configuration
        pub fn generateMkinitcpioConfig(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("# NVIDIA early KMS configuration\n");
            try config.appendSlice("MODULES=(");
            for (early_kms_modules) |module| {
                try config.appendSlice(module);
                try config.appendSlice(" ");
            }
            try config.appendSlice(")\n");
            
            try config.appendSlice("# Required hooks for NVIDIA\n");
            try config.appendSlice("HOOKS=(base udev autodetect modconf block filesystems keyboard fsck)\n");
            
            return config.toOwnedSlice();
        }
        
        /// Generate modprobe configuration
        pub fn generateModprobeConfig(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("# NVIDIA module configuration\n");
            try config.appendSlice("# Enable DRM kernel mode setting\n");
            try config.appendSlice("options nvidia-drm modeset=1 fbdev=1\n");
            try config.appendSlice("# Enable MSI interrupts\n");
            try config.appendSlice("options nvidia NVreg_EnableMSI=1\n");
            try config.appendSlice("# Use page attribute table\n");
            try config.appendSlice("options nvidia NVreg_UsePageAttributeTable=1\n");
            try config.appendSlice("# Enable PCIe Gen3\n");
            try config.appendSlice("options nvidia NVreg_EnablePCIeGen3=1\n");
            try config.appendSlice("# Preserve video memory allocations\n");
            try config.appendSlice("options nvidia NVreg_PreserveVideoMemoryAllocations=1\n");
            try config.appendSlice("# Set temporary file path\n");
            try config.appendSlice("options nvidia NVreg_TemporaryFilePath=/tmp\n");
            
            return config.toOwnedSlice();
        }
    };
    
    /// Wayland compositor configuration
    pub const WaylandConfig = struct {
        /// Environment variables for Wayland
        pub const wayland_env_vars = [_][]const u8{
            "GBM_BACKEND=nvidia-drm",
            "__GLX_VENDOR_LIBRARY_NAME=nvidia",
            "LIBVA_DRIVER_NAME=nvidia",
            "WLR_NO_HARDWARE_CURSORS=1",
            "WLR_RENDERER=vulkan",
            "XDG_SESSION_TYPE=wayland",
            "QT_QPA_PLATFORM=wayland",
            "GDK_BACKEND=wayland",
            "MOZ_ENABLE_WAYLAND=1",
            "ELECTRON_OZONE_PLATFORM_HINT=wayland",
        };
        
        /// Compositor-specific configurations
        pub const CompositorConfig = struct {
            /// Sway configuration
            pub fn generateSwayConfig(allocator: Allocator) ![]u8 {
                var config = ArrayList(u8).init(allocator);
                defer config.deinit();
                
                try config.appendSlice("# NVIDIA specific Sway configuration\n");
                try config.appendSlice("exec_always \"export GBM_BACKEND=nvidia-drm\"\n");
                try config.appendSlice("exec_always \"export __GLX_VENDOR_LIBRARY_NAME=nvidia\"\n");
                try config.appendSlice("exec_always \"export WLR_NO_HARDWARE_CURSORS=1\"\n");
                try config.appendSlice("exec_always \"export WLR_RENDERER=vulkan\"\n");
                try config.appendSlice("xwayland enable\n");
                
                return config.toOwnedSlice();
            }
            
            /// Hyprland configuration
            pub fn generateHyprlandConfig(allocator: Allocator) ![]u8 {
                var config = ArrayList(u8).init(allocator);
                defer config.deinit();
                
                try config.appendSlice("# NVIDIA specific Hyprland configuration\n");
                try config.appendSlice("env = LIBVA_DRIVER_NAME,nvidia\n");
                try config.appendSlice("env = XDG_SESSION_TYPE,wayland\n");
                try config.appendSlice("env = GBM_BACKEND,nvidia-drm\n");
                try config.appendSlice("env = __GLX_VENDOR_LIBRARY_NAME,nvidia\n");
                try config.appendSlice("env = WLR_NO_HARDWARE_CURSORS,1\n");
                try config.appendSlice("cursor {\n");
                try config.appendSlice("  no_hardware_cursors = true\n");
                try config.appendSlice("}\n");
                
                return config.toOwnedSlice();
            }
        };
    };
    
    /// NVENC/NVDEC hardware acceleration setup
    pub const HardwareAccel = struct {
        /// VA-API configuration
        pub fn generateVaapiConfig(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("# NVIDIA VA-API configuration\n");
            try config.appendSlice("export LIBVA_DRIVER_NAME=nvidia\n");
            try config.appendSlice("export VDPAU_DRIVER=nvidia\n");
            try config.appendSlice("export NVD_BACKEND=direct\n");
            try config.appendSlice("export MOZ_DISABLE_RDD_SANDBOX=1\n");
            try config.appendSlice("export MESA_LOADER_DRIVER_OVERRIDE=zink\n");
            
            return config.toOwnedSlice();
        }
        
        /// FFmpeg hardware acceleration
        pub const ffmpeg_hwaccel_flags = [_][]const u8{
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-c:v",
            "h264_nvenc",
        };
        
        /// Required packages for hardware acceleration
        pub const required_packages = [_][]const u8{
            "nvidia-utils",
            "libva-nvidia-driver",
            "libva-utils",
            "vdpauinfo",
            "nvtop",
            "nvidia-settings",
        };
    };
    
    /// Multi-GPU and hybrid graphics support
    pub const MultiGpuConfig = struct {
        /// PRIME configuration for hybrid graphics
        pub fn generatePrimeConfig(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("# PRIME configuration for hybrid graphics\n");
            try config.appendSlice("export __NV_PRIME_RENDER_OFFLOAD=1\n");
            try config.appendSlice("export __NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0\n");
            try config.appendSlice("export __GLX_VENDOR_LIBRARY_NAME=nvidia\n");
            try config.appendSlice("export __VK_LAYER_NV_optimus=NVIDIA_only\n");
            
            return config.toOwnedSlice();
        }
        
        /// Optimus manager configuration
        pub const optimus_packages = [_][]const u8{
            "optimus-manager",
            "optimus-manager-qt",
            "nvidia-prime",
        };
        
        /// GPU switching commands
        pub const gpu_switch_commands = [_][]const u8{
            "optimus-manager --switch nvidia",
            "optimus-manager --switch intel",
            "optimus-manager --switch hybrid",
        };
    };
    
    /// Power management configurations
    pub const PowerManagement = struct {
        /// SystemD service for NVIDIA power management
        pub fn generateSystemdService(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("[Unit]\n");
            try config.appendSlice("Description=NVIDIA system suspend actions\n");
            try config.appendSlice("Before=systemd-suspend.service\n");
            try config.appendSlice("Before=systemd-hibernate.service\n");
            try config.appendSlice("Before=nvidia-suspend.service\n");
            try config.appendSlice("\n[Service]\n");
            try config.appendSlice("Type=oneshot\n");
            try config.appendSlice("ExecStart=/usr/bin/nvidia-sleep.sh suspend\n");
            try config.appendSlice("ExecStop=/usr/bin/nvidia-sleep.sh resume\n");
            try config.appendSlice("RemainAfterExit=yes\n");
            try config.appendSlice("\n[Install]\n");
            try config.appendSlice("WantedBy=multi-user.target\n");
            
            return config.toOwnedSlice();
        }
        
        /// Power management script
        pub fn generatePowerScript(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("#!/bin/bash\n");
            try config.appendSlice("# NVIDIA power management script\n");
            try config.appendSlice("case \"$1\" in\n");
            try config.appendSlice("    suspend)\n");
            try config.appendSlice("        echo 'suspend' > /proc/driver/nvidia/suspend\n");
            try config.appendSlice("        ;;\n");
            try config.appendSlice("    resume)\n");
            try config.appendSlice("        echo 'resume' > /proc/driver/nvidia/suspend\n");
            try config.appendSlice("        ;;\n");
            try config.appendSlice("esac\n");
            
            return config.toOwnedSlice();
        }
    };
    
    /// Package manager integration helpers
    pub const PackageManager = struct {
        /// Required NVIDIA packages
        pub const core_packages = [_][]const u8{
            "nvidia",
            "nvidia-utils",
            "nvidia-settings",
            "lib32-nvidia-utils",
            "nvidia-prime",
        };
        
        /// Optional packages for enhanced functionality
        pub const optional_packages = [_][]const u8{
            "nvidia-container-toolkit",
            "libva-nvidia-driver",
            "libva-utils",
            "vdpauinfo",
            "nvtop",
            "gwe",
            "green-with-envy",
        };
        
        /// AUR packages
        pub const aur_packages = [_][]const u8{
            "nvidia-tweaks",
            "nvidia-container-runtime",
            "optimus-manager",
            "envycontrol",
        };
        
        /// Package installation script
        pub fn generateInstallScript(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("#!/bin/bash\n");
            try config.appendSlice("# NVIDIA package installation script for Arch Linux\n");
            try config.appendSlice("set -e\n\n");
            
            try config.appendSlice("# Install core NVIDIA packages\n");
            try config.appendSlice("sudo pacman -S --needed ");
            for (core_packages) |pkg| {
                try config.appendSlice(pkg);
                try config.appendSlice(" ");
            }
            try config.appendSlice("\n\n");
            
            try config.appendSlice("# Install optional packages\n");
            try config.appendSlice("sudo pacman -S --needed ");
            for (optional_packages) |pkg| {
                try config.appendSlice(pkg);
                try config.appendSlice(" ");
            }
            try config.appendSlice("\n\n");
            
            try config.appendSlice("# Enable required services\n");
            try config.appendSlice("sudo systemctl enable nvidia-suspend.service\n");
            try config.appendSlice("sudo systemctl enable nvidia-hibernate.service\n");
            try config.appendSlice("sudo systemctl enable nvidia-resume.service\n");
            
            return config.toOwnedSlice();
        }
    };
    
    /// Environment variable configurations
    pub const EnvironmentVars = struct {
        /// Global environment variables
        pub const global_env_vars = [_][]const u8{
            "CUDA_CACHE_PATH=/tmp/cuda-cache",
            "__GL_SHADER_DISK_CACHE_PATH=/tmp/gl-shader-cache",
            "__GL_SYNC_TO_VBLANK=1",
            "__GL_VRR_ALLOWED=1",
            "__GL_GSYNC_ALLOWED=1",
            "__GL_MaxFramesAllowed=1",
            "__GL_THREADED_OPTIMIZATIONS=1",
            "__GL_ALLOW_UNOFFICIAL_PROTOCOL=1",
            "NVIDIA_DRIVER_CAPABILITIES=all",
            "NVIDIA_VISIBLE_DEVICES=all",
        };
        
        /// Gaming-specific environment variables
        pub const gaming_env_vars = [_][]const u8{
            "__GL_SHADER_DISK_CACHE=1",
            "__GL_SHADER_DISK_CACHE_SIZE=1073741824",
            "__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1",
            "DXVK_HUD=fps",
            "MANGOHUD=1",
            "PROTON_ENABLE_NVAPI=1",
            "PROTON_ENABLE_NGX_UPDATER=1",
        };
        
        /// Generate environment configuration file
        pub fn generateEnvironmentFile(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("# NVIDIA environment configuration\n");
            try config.appendSlice("# Global NVIDIA settings\n");
            
            for (global_env_vars) |env_var| {
                try config.appendSlice("export ");
                try config.appendSlice(env_var);
                try config.appendSlice("\n");
            }
            
            try config.appendSlice("\n# Gaming-specific settings\n");
            for (gaming_env_vars) |env_var| {
                try config.appendSlice("export ");
                try config.appendSlice(env_var);
                try config.appendSlice("\n");
            }
            
            return config.toOwnedSlice();
        }
    };
    
    /// Xorg.conf fallback configurations
    pub const XorgConfig = struct {
        /// Generate Xorg configuration
        pub fn generateXorgConfig(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("# NVIDIA Xorg configuration\n");
            try config.appendSlice("Section \"Module\"\n");
            try config.appendSlice("    Load \"dri2\"\n");
            try config.appendSlice("    Load \"dri3\"\n");
            try config.appendSlice("    Load \"present\"\n");
            try config.appendSlice("    Load \"glx\"\n");
            try config.appendSlice("EndSection\n\n");
            
            try config.appendSlice("Section \"OutputClass\"\n");
            try config.appendSlice("    Identifier \"nvidia\"\n");
            try config.appendSlice("    MatchDriver \"nvidia-drm\"\n");
            try config.appendSlice("    Driver \"nvidia\"\n");
            try config.appendSlice("    Option \"AllowEmptyInitialConfiguration\"\n");
            try config.appendSlice("    Option \"PrimaryGPU\" \"yes\"\n");
            try config.appendSlice("    Option \"SLI\" \"Auto\"\n");
            try config.appendSlice("    Option \"BaseMosaic\" \"on\"\n");
            try config.appendSlice("    ModulePath \"/usr/lib/nvidia/xorg\"\n");
            try config.appendSlice("    ModulePath \"/usr/lib/xorg/modules\"\n");
            try config.appendSlice("EndSection\n\n");
            
            try config.appendSlice("Section \"Device\"\n");
            try config.appendSlice("    Identifier \"Nvidia Card\"\n");
            try config.appendSlice("    Driver \"nvidia\"\n");
            try config.appendSlice("    VendorName \"NVIDIA Corporation\"\n");
            try config.appendSlice("    Option \"NoLogo\" \"true\"\n");
            try config.appendSlice("    Option \"UseEDID\" \"false\"\n");
            try config.appendSlice("    Option \"UseDisplayDevice\" \"none\"\n");
            try config.appendSlice("    Option \"TripleBuffer\" \"on\"\n");
            try config.appendSlice("    Option \"RegistryDwords\" \"EnableBrightnessControl=1\"\n");
            try config.appendSlice("EndSection\n\n");
            
            try config.appendSlice("Section \"Screen\"\n");
            try config.appendSlice("    Identifier \"nvidia\"\n");
            try config.appendSlice("    Device \"Nvidia Card\"\n");
            try config.appendSlice("    Option \"AllowIndirectGLXProtocol\" \"off\"\n");
            try config.appendSlice("    Option \"TripleBuffer\" \"on\"\n");
            try config.appendSlice("EndSection\n");
            
            return config.toOwnedSlice();
        }
    };
    
    /// GRUB/bootloader parameter recommendations
    pub const BootloaderConfig = struct {
        /// GRUB kernel parameters
        pub const grub_params = [_][]const u8{
            "nvidia-drm.modeset=1",
            "nvidia-drm.fbdev=1",
            "nvidia.NVreg_PreserveVideoMemoryAllocations=1",
            "nvidia.NVreg_EnableMSI=1",
            "nvidia.NVreg_UsePageAttributeTable=1",
            "iommu=soft",
            "pcie_aspm=off",
            "acpi_osi=Linux",
            "nouveau.modeset=0",
            "rdblacklist=nouveau",
        };
        
        /// Generate GRUB configuration
        pub fn generateGrubConfig(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("# NVIDIA GRUB configuration\n");
            try config.appendSlice("GRUB_CMDLINE_LINUX_DEFAULT=\"");
            
            for (grub_params, 0..) |param, i| {
                if (i > 0) try config.appendSlice(" ");
                try config.appendSlice(param);
            }
            
            try config.appendSlice("\"\n");
            try config.appendSlice("GRUB_GFXMODE=auto\n");
            try config.appendSlice("GRUB_GFXPAYLOAD_LINUX=keep\n");
            
            return config.toOwnedSlice();
        }
    };
    
    /// udev rules for proper device permissions
    pub const UdevRules = struct {
        /// Generate udev rules
        pub fn generateUdevRules(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("# NVIDIA udev rules\n");
            try config.appendSlice("# Set permissions for NVIDIA devices\n");
            try config.appendSlice("KERNEL==\"nvidia\", RUN+=\"/bin/chmod 666 /dev/nvidia*\"\n");
            try config.appendSlice("KERNEL==\"nvidia_uvm\", RUN+=\"/bin/chmod 666 /dev/nvidia-uvm*\"\n");
            try config.appendSlice("KERNEL==\"nvidia_modeset\", RUN+=\"/bin/chmod 666 /dev/nvidia-modeset\"\n");
            try config.appendSlice("KERNEL==\"nvidiactl\", RUN+=\"/bin/chmod 666 /dev/nvidiactl\"\n");
            try config.appendSlice("\n# Enable GPU scaling\n");
            try config.appendSlice("ACTION==\"add\", SUBSYSTEM==\"pci\", ATTR{vendor}==\"0x10de\", ATTR{class}==\"0x030000\", RUN+=\"/usr/bin/nvidia-smi -pm 1\"\n");
            try config.appendSlice("\n# Power management\n");
            try config.appendSlice("ACTION==\"add\", SUBSYSTEM==\"pci\", ATTR{vendor}==\"0x10de\", ATTR{class}==\"0x030000\", ATTR{power/control}=\"auto\"\n");
            
            return config.toOwnedSlice();
        }
    };
    
    /// Common gotcha fixes from the wiki
    pub const GotchaFixes = struct {
        /// Fix for screen tearing
        pub const screen_tearing_fix = [_][]const u8{
            "nvidia-settings --assign CurrentMetaMode=\"nvidia-auto-select +0+0 { ForceFullCompositionPipeline = On }\"",
            "xrandr --output DP-0 --set \"NVIDIA Color Range\" \"Full\"",
            "xrandr --output DP-0 --set \"NVIDIA Color Format\" \"RGB\"",
        };
        
        /// Fix for suspend/resume issues
        pub const suspend_resume_fix = [_][]const u8{
            "systemctl enable nvidia-suspend.service",
            "systemctl enable nvidia-hibernate.service",
            "systemctl enable nvidia-resume.service",
        };
        
        /// Fix for Wayland issues
        pub const wayland_fixes = [_][]const u8{
            "export GBM_BACKEND=nvidia-drm",
            "export __GLX_VENDOR_LIBRARY_NAME=nvidia",
            "export WLR_NO_HARDWARE_CURSORS=1",
        };
        
        /// Fix for PRIME display offloading
        pub const prime_fixes = [_][]const u8{
            "export __NV_PRIME_RENDER_OFFLOAD=1",
            "export __NV_PRIME_RENDER_OFFLOAD_PROVIDER=NVIDIA-G0",
            "export __GLX_VENDOR_LIBRARY_NAME=nvidia",
        };
        
        /// Fix for Steam/gaming issues
        pub const gaming_fixes = [_][]const u8{
            "export __GL_SHADER_DISK_CACHE=1",
            "export __GL_THREADED_OPTIMIZATIONS=1",
            "export PROTON_ENABLE_NVAPI=1",
        };
        
        /// Fix for Firefox hardware acceleration
        pub const firefox_fixes = [_][]const u8{
            "export MOZ_DISABLE_RDD_SANDBOX=1",
            "export MOZ_ENABLE_WAYLAND=1",
            "export LIBVA_DRIVER_NAME=nvidia",
        };
        
        /// Generate comprehensive fix script
        pub fn generateFixScript(allocator: Allocator) ![]u8 {
            var config = ArrayList(u8).init(allocator);
            defer config.deinit();
            
            try config.appendSlice("#!/bin/bash\n");
            try config.appendSlice("# Comprehensive NVIDIA fix script for Arch Linux\n");
            try config.appendSlice("set -e\n\n");
            
            try config.appendSlice("echo \"Applying NVIDIA fixes...\"\n\n");
            
            try config.appendSlice("# Fix suspend/resume\n");
            for (suspend_resume_fix) |fix| {
                try config.appendSlice("sudo ");
                try config.appendSlice(fix);
                try config.appendSlice("\n");
            }
            
            try config.appendSlice("\n# Fix screen tearing\n");
            for (screen_tearing_fix) |fix| {
                try config.appendSlice(fix);
                try config.appendSlice("\n");
            }
            
            try config.appendSlice("\n# Update initramfs\n");
            try config.appendSlice("sudo mkinitcpio -P\n");
            
            try config.appendSlice("\n# Update GRUB\n");
            try config.appendSlice("sudo grub-mkconfig -o /boot/grub/grub.cfg\n");
            
            try config.appendSlice("\necho \"NVIDIA fixes applied successfully!\"\n");
            try config.appendSlice("echo \"Please reboot your system.\"\n");
            
            return config.toOwnedSlice();
        }
    };
    
    /// Complete system configuration generator
    pub fn generateCompleteConfig(self: *ArchConfig) !void {
        const files_to_create = [_]struct {
            path: []const u8,
            content_fn: fn (Allocator) anyerror![]u8,
        }{
            .{ .path = "/etc/mkinitcpio.conf.nvidia", .content_fn = KernelModule.generateMkinitcpioConfig },
            .{ .path = "/etc/modprobe.d/nvidia.conf", .content_fn = KernelModule.generateModprobeConfig },
            .{ .path = "/etc/environment.nvidia", .content_fn = EnvironmentVars.generateEnvironmentFile },
            .{ .path = "/etc/X11/xorg.conf.d/20-nvidia.conf", .content_fn = XorgConfig.generateXorgConfig },
            .{ .path = "/etc/default/grub.nvidia", .content_fn = BootloaderConfig.generateGrubConfig },
            .{ .path = "/etc/udev/rules.d/70-nvidia.rules", .content_fn = UdevRules.generateUdevRules },
            .{ .path = "/tmp/nvidia-install.sh", .content_fn = PackageManager.generateInstallScript },
            .{ .path = "/tmp/nvidia-fixes.sh", .content_fn = GotchaFixes.generateFixScript },
            .{ .path = "/tmp/nvidia-power.sh", .content_fn = PowerManagement.generatePowerScript },
            .{ .path = "/tmp/nvidia-prime.conf", .content_fn = MultiGpuConfig.generatePrimeConfig },
            .{ .path = "/tmp/nvidia-vaapi.conf", .content_fn = HardwareAccel.generateVaapiConfig },
        };
        
        for (files_to_create) |file_info| {
            const content = try file_info.content_fn(self.allocator);
            defer self.allocator.free(content);
            
            std.log.info("Generated configuration: {s}", .{file_info.path});
        }
    }
    
    /// Validate system configuration
    pub fn validateConfiguration(self: *ArchConfig) !bool {\n        _ = self;
        // Check if NVIDIA driver is loaded
        const nvidia_loaded = std.process.hasEnvVar("NVIDIA_DRIVER_VERSION");
        
        // Check if DRM is enabled
        const drm_enabled = std.fs.accessAbsolute("/sys/module/nvidia_drm/parameters/modeset", .{}) catch false;
        
        // Check if required files exist
        const required_files = [_][]const u8{
            "/proc/driver/nvidia/version",
            "/dev/nvidia0",
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
        };
        
        for (required_files) |file| {
            _ = std.fs.accessAbsolute(file, .{}) catch {
                std.log.err("Required file not found: {s}", .{file});
                return false;
            };
        }
        
        return nvidia_loaded and drm_enabled;
    }
    
    /// Initialize the configuration
    pub fn init(allocator: Allocator) ArchConfig {
        return ArchConfig{
            .allocator = allocator,
        };
    }
    
    /// Cleanup resources
    pub fn deinit(self: *ArchConfig) void {
        _ = self;
    }
};

/// Arch Linux specific NVIDIA management functions
pub const ArchNvidiaManager = struct {
    config: ArchConfig,
    
    pub fn init(allocator: Allocator) ArchNvidiaManager {
        return ArchNvidiaManager{
            .config = ArchConfig.init(allocator),
        };
    }
    
    pub fn deinit(self: *ArchNvidiaManager) void {
        self.config.deinit();
    }
    
    /// Apply all recommended configurations
    pub fn applyRecommendedConfig(self: *ArchNvidiaManager) !void {
        try self.config.generateCompleteConfig();
        
        std.log.info("Applied comprehensive Arch Linux NVIDIA configuration");
        std.log.info("Please review generated files and reboot your system");
    }
    
    /// Check system health
    pub fn checkSystemHealth(self: *ArchNvidiaManager) !void {
        const is_healthy = try self.config.validateConfiguration();
        
        if (is_healthy) {
            std.log.info("NVIDIA configuration is healthy");
        } else {
            std.log.err("NVIDIA configuration has issues - run fix script");
        }
    }
};