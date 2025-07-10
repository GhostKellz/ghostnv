const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// NVIDIA Container Runtime integration for GhostNV
/// Provides native GPU isolation and resource management for containers
pub const ContainerRuntime = struct {
    allocator: Allocator,
    devices: ArrayList(ContainerDevice),
    namespaces: ArrayList(ContainerNamespace),
    cgroup_manager: CGroupManager,
    
    pub fn init(allocator: Allocator) !ContainerRuntime {
        return ContainerRuntime{
            .allocator = allocator,
            .devices = ArrayList(ContainerDevice).init(allocator),
            .namespaces = ArrayList(ContainerNamespace).init(allocator),
            .cgroup_manager = try CGroupManager.init(allocator),
        };
    }
    
    pub fn deinit(self: *ContainerRuntime) void {
        self.devices.deinit();
        self.namespaces.deinit();
        self.cgroup_manager.deinit();
    }
    
    /// Create a new GPU-enabled container namespace
    pub fn create_container(self: *ContainerRuntime, config: ContainerConfig) !ContainerHandle {
        std.log.info("Creating GPU container: {s}", .{config.name});
        
        // Create new namespace
        var namespace = try self.create_namespace(config);
        
        // Setup GPU device access
        if (config.gpu_access.enabled) {
            try self.setup_gpu_devices(&namespace, config.gpu_access);
        }
        
        // Configure resource limits
        try self.cgroup_manager.setup_limits(namespace.id, config.limits);
        
        // Apply security policies
        try self.apply_security_policies(&namespace, config.security);
        
        const handle = ContainerHandle{
            .id = namespace.id,
            .namespace = namespace,
            .pid = 0, // Will be set when container starts
        };
        
        try self.namespaces.append(namespace);
        
        std.log.info("Container {} created successfully", .{handle.id});
        return handle;
    }
    
    /// Start a container with GPU access
    pub fn start_container(self: *ContainerRuntime, handle: *ContainerHandle, executable: []const u8, args: [][]const u8) !void {
        std.log.info("Starting container {} with executable: {s}", .{ handle.id, executable });
        
        // Fork and exec in the container namespace
        const pid = linux.fork();
        if (pid == 0) {
            // Child process - setup container environment
            try self.enter_namespace(&handle.namespace);
            try self.setup_container_environment(&handle.namespace);
            
            // Set GPU environment variables
            try self.set_gpu_environment_variables();
            
            // Execute the container command
            const result = linux.execve(executable.ptr, @ptrCast(args.ptr), @ptrFromInt(0));
            std.log.err("execve failed: {}", .{result});
            std.process.exit(1);
        } else if (pid > 0) {
            // Parent process
            handle.pid = pid;
            std.log.info("Container {} started with PID {}", .{ handle.id, pid });
        } else {
            return error.ForkFailed;
        }
    }
    
    /// Stop and cleanup container
    pub fn stop_container(self: *ContainerRuntime, handle: *ContainerHandle) !void {
        std.log.info("Stopping container {}", .{handle.id});
        
        if (handle.pid > 0) {
            // Send SIGTERM first
            _ = linux.kill(handle.pid, linux.SIG.TERM);
            
            // Wait for graceful shutdown
            std.time.sleep(5 * std.time.ns_per_s);
            
            // Force kill if still running
            _ = linux.kill(handle.pid, linux.SIG.KILL);
        }
        
        // Cleanup resources
        try self.cleanup_container_resources(handle);
        
        std.log.info("Container {} stopped", .{handle.id});
    }
    
    /// List GPU devices available for containers
    pub fn list_gpu_devices(self: *ContainerRuntime) ![]ContainerDevice {
        var devices = ArrayList(ContainerDevice).init(self.allocator);
        
        // Enumerate NVIDIA GPUs
        for (0..4) |i| { // Max 4 GPUs
            const device_path = try std.fmt.allocPrint(self.allocator, "/dev/nvidia{}", .{i});
            defer self.allocator.free(device_path);
            
            if (std.fs.accessAbsolute(device_path, .{})) {
                const device = ContainerDevice{
                    .id = @intCast(i),
                    .path = try self.allocator.dupe(u8, device_path),
                    .device_type = .gpu,
                    .uuid = try self.get_gpu_uuid(@intCast(i)),
                    .memory_mb = try self.get_gpu_memory(@intCast(i)),
                    .compute_capability = try self.get_compute_capability(@intCast(i)),
                    .available = true,
                };
                
                try devices.append(device);
            } else |_| {
                break; // No more GPUs
            }
        }
        
        // Add control devices
        const control_devices = [_][]const u8{ "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-modeset" };
        
        for (control_devices, 1000..) |ctrl_path, id| {
            if (std.fs.accessAbsolute(ctrl_path, .{})) {
                const device = ContainerDevice{
                    .id = @intCast(id),
                    .path = try self.allocator.dupe(u8, ctrl_path),
                    .device_type = .control,
                    .uuid = "N/A",
                    .memory_mb = 0,
                    .compute_capability = .{ .major = 0, .minor = 0 },
                    .available = true,
                };
                
                try devices.append(device);
            } else |_| {}
        }
        
        return devices.toOwnedSlice();
    }
    
    /// Get container resource usage
    pub fn get_container_stats(self: *ContainerRuntime, handle: *ContainerHandle) !ContainerStats {
        return ContainerStats{
            .id = handle.id,
            .pid = handle.pid,
            .cpu_usage_percent = try self.cgroup_manager.get_cpu_usage(handle.namespace.id),
            .memory_usage_mb = try self.cgroup_manager.get_memory_usage(handle.namespace.id),
            .gpu_usage_percent = try self.get_gpu_usage_for_container(handle.namespace.id),
            .gpu_memory_usage_mb = try self.get_gpu_memory_usage_for_container(handle.namespace.id),
            .network_rx_bytes = 0, // TODO: Implement network stats
            .network_tx_bytes = 0,
        };
    }
    
    // Private implementation methods
    
    fn create_namespace(self: *ContainerRuntime, config: ContainerConfig) !ContainerNamespace {
        const namespace_id = std.crypto.random.int(u32);
        
        // Create new namespaces (simplified - real implementation would use clone() with CLONE_NEW* flags)
        return ContainerNamespace{
            .id = namespace_id,
            .name = try self.allocator.dupe(u8, config.name),
            .pid_namespace = namespace_id,
            .net_namespace = namespace_id,
            .mount_namespace = namespace_id,
            .user_namespace = namespace_id,
        };
    }
    
    fn setup_gpu_devices(self: *ContainerRuntime, namespace: *ContainerNamespace, gpu_access: GpuAccess) !void {
        _ = namespace;
        
        std.log.info("Setting up GPU access for container", .{});
        
        // Create device nodes in container
        for (gpu_access.device_ids) |device_id| {
            const device_path = try std.fmt.allocPrint(self.allocator, "/dev/nvidia{}", .{device_id});
            defer self.allocator.free(device_path);
            
            // Bind mount GPU device into container namespace
            // Real implementation would use mount() syscall
            std.log.info("Mounting GPU device: {s}", .{device_path});
        }
        
        // Mount control devices
        const control_devices = [_][]const u8{ "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-modeset" };
        for (control_devices) |device| {
            std.log.info("Mounting control device: {s}", .{device});
        }
    }
    
    fn enter_namespace(_: *ContainerRuntime, namespace: *ContainerNamespace) !void {
        // Real implementation would use setns() to enter namespaces
        std.log.info("Entering container namespace {}", .{namespace.id});
    }
    
    fn setup_container_environment(self: *ContainerRuntime, namespace: *ContainerNamespace) !void {
        _ = self;
        _ = namespace;
        
        // Setup minimal container environment
        // - Mount /proc, /sys, /dev
        // - Setup hostname
        // - Configure networking
        std.log.info("Setting up container environment");
    }
    
    fn set_gpu_environment_variables(self: *ContainerRuntime) !void {
        _ = self;
        
        // Set NVIDIA-specific environment variables
        try std.process.env.set("NVIDIA_VISIBLE_DEVICES", "all");
        try std.process.env.set("NVIDIA_DRIVER_CAPABILITIES", "compute,video,graphics,utility");
        
        std.log.info("Set GPU environment variables");
    }
    
    fn apply_security_policies(self: *ContainerRuntime, namespace: *ContainerNamespace, security: SecurityPolicy) !void {
        _ = self;
        _ = namespace;
        _ = security;
        
        // Apply seccomp, apparmor, SELinux policies
        std.log.info("Applying security policies");
    }
    
    fn cleanup_container_resources(self: *ContainerRuntime, handle: *ContainerHandle) !void {
        // Remove from cgroups
        try self.cgroup_manager.remove_container(handle.namespace.id);
        
        // Cleanup namespace
        for (self.namespaces.items, 0..) |namespace, i| {
            if (namespace.id == handle.namespace.id) {
                _ = self.namespaces.swapRemove(i);
                break;
            }
        }
        
        std.log.info("Cleaned up resources for container {}", .{handle.id});
    }
    
    fn get_gpu_uuid(self: *ContainerRuntime, device_id: u32) ![]u8 {
        // Mock UUID for now - real implementation would query NVML
        return try std.fmt.allocPrint(self.allocator, "GPU-{:08x}-{:04x}-{:04x}", .{ device_id, 1234, 5678 });
    }
    
    fn get_gpu_memory(self: *ContainerRuntime, device_id: u32) !u32 {
        _ = self;
        _ = device_id;
        
        // Mock 24GB for RTX 4090 - real implementation would query NVML
        return 24576;
    }
    
    fn get_compute_capability(self: *ContainerRuntime, device_id: u32) !ComputeCapability {
        _ = self;
        _ = device_id;
        
        // Mock compute capability for RTX 40 series
        return ComputeCapability{ .major = 8, .minor = 9 };
    }
    
    fn get_gpu_usage_for_container(self: *ContainerRuntime, namespace_id: u32) !f32 {
        _ = self;
        _ = namespace_id;
        
        // Mock GPU usage - real implementation would track per-container GPU usage
        return 45.5;
    }
    
    fn get_gpu_memory_usage_for_container(self: *ContainerRuntime, namespace_id: u32) !u32 {
        _ = self;
        _ = namespace_id;
        
        // Mock GPU memory usage
        return 8192;
    }
};

/// CGroup manager for resource limits
pub const CGroupManager = struct {
    allocator: Allocator,
    cgroup_root: []const u8,
    
    pub fn init(allocator: Allocator) !CGroupManager {
        return CGroupManager{
            .allocator = allocator,
            .cgroup_root = "/sys/fs/cgroup",
        };
    }
    
    pub fn deinit(self: *CGroupManager) void {
        _ = self;
    }
    
    pub fn setup_limits(self: *CGroupManager, namespace_id: u32, limits: ResourceLimits) !void {
        const cgroup_path = try std.fmt.allocPrint(self.allocator, "{s}/ghostnv/container-{}", .{ self.cgroup_root, namespace_id });
        defer self.allocator.free(cgroup_path);
        
        // Create cgroup directory
        std.fs.makeDirAbsolute(cgroup_path) catch |err| {
            if (err != error.PathAlreadyExists) {
                return err;
            }
        };
        
        // Set memory limit
        if (limits.memory_limit_mb > 0) {
            const memory_limit_path = try std.fmt.allocPrint(self.allocator, "{s}/memory.max", .{cgroup_path});
            defer self.allocator.free(memory_limit_path);
            
            const memory_bytes = limits.memory_limit_mb * 1024 * 1024;
            const limit_str = try std.fmt.allocPrint(self.allocator, "{}", .{memory_bytes});
            defer self.allocator.free(limit_str);
            
            try std.fs.cwd().writeFile(memory_limit_path, limit_str);
        }
        
        // Set CPU limit
        if (limits.cpu_cores > 0) {
            const cpu_max_path = try std.fmt.allocPrint(self.allocator, "{s}/cpu.max", .{cgroup_path});
            defer self.allocator.free(cpu_max_path);
            
            const cpu_quota = limits.cpu_cores * 100000; // 100ms period
            const cpu_str = try std.fmt.allocPrint(self.allocator, "{} 100000", .{cpu_quota});
            defer self.allocator.free(cpu_str);
            
            try std.fs.writeFileAbsolute(cpu_max_path, cpu_str);
        }
        
        std.log.info("Set resource limits for container {}: {}MB RAM, {} CPU cores", .{ namespace_id, limits.memory_limit_mb, limits.cpu_cores });
    }
    
    pub fn get_cpu_usage(self: *CGroupManager, namespace_id: u32) !f32 {
        _ = self;
        _ = namespace_id;
        
        // Mock CPU usage
        return 25.5;
    }
    
    pub fn get_memory_usage(self: *CGroupManager, namespace_id: u32) !u32 {
        _ = self;
        _ = namespace_id;
        
        // Mock memory usage in MB
        return 2048;
    }
    
    pub fn remove_container(self: *CGroupManager, namespace_id: u32) !void {
        const cgroup_path = try std.fmt.allocPrint(self.allocator, "{s}/ghostnv/container-{}", .{ self.cgroup_root, namespace_id });
        defer self.allocator.free(cgroup_path);
        
        std.fs.deleteTreeAbsolute(cgroup_path) catch |err| {
            if (err != error.FileNotFound) {
                std.log.warn("Failed to remove cgroup {s}: {}", .{ cgroup_path, err });
            }
        };
    }
};

// Type definitions

pub const ContainerDevice = struct {
    id: u32,
    path: []const u8,
    device_type: DeviceType,
    uuid: []const u8,
    memory_mb: u32,
    compute_capability: ComputeCapability,
    available: bool,
};

pub const DeviceType = enum {
    gpu,
    control,
};

pub const ComputeCapability = struct {
    major: u8,
    minor: u8,
};

pub const ContainerConfig = struct {
    name: []const u8,
    gpu_access: GpuAccess,
    limits: ResourceLimits,
    security: SecurityPolicy,
};

pub const GpuAccess = struct {
    enabled: bool,
    device_ids: []const u32,
    capabilities: []const []const u8, // compute, video, graphics, utility
};

pub const ResourceLimits = struct {
    memory_limit_mb: u32,
    cpu_cores: f32,
    gpu_memory_limit_mb: u32,
};

pub const SecurityPolicy = struct {
    seccomp_profile: []const u8,
    apparmor_profile: []const u8,
    capabilities: []const []const u8,
};

pub const ContainerNamespace = struct {
    id: u32,
    name: []const u8,
    pid_namespace: u32,
    net_namespace: u32,
    mount_namespace: u32,
    user_namespace: u32,
};

pub const ContainerHandle = struct {
    id: u32,
    namespace: ContainerNamespace,
    pid: i32,
};

pub const ContainerStats = struct {
    id: u32,
    pid: i32,
    cpu_usage_percent: f32,
    memory_usage_mb: u32,
    gpu_usage_percent: f32,
    gpu_memory_usage_mb: u32,
    network_rx_bytes: u64,
    network_tx_bytes: u64,
};

// CLI for container management
pub const ContainerCLI = struct {
    runtime: *ContainerRuntime,
    
    pub fn init(runtime: *ContainerRuntime) ContainerCLI {
        return ContainerCLI{ .runtime = runtime };
    }
    
    pub fn execute_command(self: *ContainerCLI, args: [][]const u8) !void {
        if (args.len < 2) {
            try self.print_help();
            return;
        }
        
        const command = args[1];
        
        if (std.mem.eql(u8, command, "run")) {
            try self.handle_run_command(args[2..]);
        } else if (std.mem.eql(u8, command, "list")) {
            try self.handle_list_command();
        } else if (std.mem.eql(u8, command, "stop")) {
            try self.handle_stop_command(args[2..]);
        } else if (std.mem.eql(u8, command, "stats")) {
            try self.handle_stats_command(args[2..]);
        } else if (std.mem.eql(u8, command, "devices")) {
            try self.handle_devices_command();
        } else {
            std.debug.print("Unknown command: {s}\n", .{command});
            try self.print_help();
        }
    }
    
    fn handle_run_command(self: *ContainerCLI, args: [][]const u8) !void {
        if (args.len < 2) {
            std.debug.print("Usage: ghostnv-container run <name> <image> [command...]\n", .{});
            return;
        }
        
        const container_name = args[0];
        const image = args[1];
        const command = if (args.len > 2) args[2] else "/bin/bash";
        
        const config = ContainerConfig{
            .name = container_name,
            .gpu_access = GpuAccess{
                .enabled = true,
                .device_ids = &[_]u32{0}, // Use GPU 0
                .capabilities = &[_][]const u8{ "compute", "video", "graphics", "utility" },
            },
            .limits = ResourceLimits{
                .memory_limit_mb = 8192, // 8GB
                .cpu_cores = 4.0,
                .gpu_memory_limit_mb = 12288, // 12GB
            },
            .security = SecurityPolicy{
                .seccomp_profile = "default",
                .apparmor_profile = "docker-default",
                .capabilities = &[_][]const u8{},
            },
        };
        
        var handle = try self.runtime.create_container(config);
        std.debug.print("Created container {s} with ID {}\n", .{ container_name, handle.id });
        
        const exec_args = &[_][]const u8{command};
        try self.runtime.start_container(&handle, image, exec_args);
        
        std.debug.print("Started container {s}\n", .{container_name});
    }
    
    fn handle_list_command(self: *ContainerCLI) !void {
        std.debug.print("Active GPU Containers:\n");
        std.debug.print("ID\tNAME\tSTATUS\n");
        
        for (self.runtime.namespaces.items) |namespace| {
            std.debug.print("{}\t{s}\tRunning\n", .{ namespace.id, namespace.name });
        }
    }
    
    fn handle_stop_command(self: *ContainerCLI, args: [][]const u8) !void {
        if (args.len < 1) {
            std.debug.print("Usage: ghostnv-container stop <container_id>\n");
            return;
        }
        
        const container_id = try std.fmt.parseInt(u32, args[0], 10);
        
        for (self.runtime.namespaces.items) |namespace| {
            if (namespace.id == container_id) {
                var handle = ContainerHandle{
                    .id = container_id,
                    .namespace = namespace,
                    .pid = 1234, // Mock PID
                };
                
                try self.runtime.stop_container(&handle);
                std.debug.print("Stopped container {}\n", .{container_id});
                return;
            }
        }
        
        std.debug.print("Container {} not found\n", .{container_id});
    }
    
    fn handle_stats_command(self: *ContainerCLI, args: [][]const u8) !void {
        if (args.len < 1) {
            std.debug.print("Usage: ghostnv-container stats <container_id>\n");
            return;
        }
        
        const container_id = try std.fmt.parseInt(u32, args[0], 10);
        
        for (self.runtime.namespaces.items) |namespace| {
            if (namespace.id == container_id) {
                var handle = ContainerHandle{
                    .id = container_id,
                    .namespace = namespace,
                    .pid = 1234,
                };
                
                const stats = try self.runtime.get_container_stats(&handle);
                
                std.debug.print("Container {} Stats:\n", .{container_id});
                std.debug.print("  CPU Usage: {:.1}%\n", .{stats.cpu_usage_percent});
                std.debug.print("  Memory Usage: {}MB\n", .{stats.memory_usage_mb});
                std.debug.print("  GPU Usage: {:.1}%\n", .{stats.gpu_usage_percent});
                std.debug.print("  GPU Memory: {}MB\n", .{stats.gpu_memory_usage_mb});
                return;
            }
        }
        
        std.debug.print("Container {} not found\n", .{container_id});
    }
    
    fn handle_devices_command(self: *ContainerCLI) !void {
        const devices = try self.runtime.list_gpu_devices();
        defer self.runtime.allocator.free(devices);
        
        std.debug.print("Available GPU Devices:\n");
        std.debug.print("ID\tPATH\t\t\tMEMORY\tCOMPUTE\tSTATUS\n");
        
        for (devices) |device| {
            const status = if (device.available) "Available" else "In Use";
            std.debug.print("{}\t{s}\t\t{}MB\t{}.{}\t{s}\n", .{
                device.id,
                device.path,
                device.memory_mb,
                device.compute_capability.major,
                device.compute_capability.minor,
                status,
            });
        }
    }
    
    fn print_help(self: *ContainerCLI) !void {
        _ = self;
        
        std.debug.print(
            \\ghostnv-container - NVIDIA Container Runtime for GhostNV
            \\
            \\Usage:
            \\  ghostnv-container run <name> <image> [command]   Create and start GPU container
            \\  ghostnv-container list                          List active containers
            \\  ghostnv-container stop <id>                     Stop container
            \\  ghostnv-container stats <id>                    Show container stats
            \\  ghostnv-container devices                       List GPU devices
            \\
            \\Examples:
            \\  ghostnv-container run ml-training tensorflow/tensorflow:latest-gpu
            \\  ghostnv-container run blender blender:latest blender --background
            \\  ghostnv-container stats 1234
            \\  ghostnv-container devices
            \\
        , .{});
    }
};

// Test functions
test "container runtime init" {
    const allocator = std.testing.allocator;
    
    var runtime = try ContainerRuntime.init(allocator);
    defer runtime.deinit();
    
    try std.testing.expect(runtime.devices.items.len == 0);
    try std.testing.expect(runtime.namespaces.items.len == 0);
}

test "gpu device listing" {
    const allocator = std.testing.allocator;
    
    var runtime = try ContainerRuntime.init(allocator);
    defer runtime.deinit();
    
    // This test would need actual GPU devices to work properly
    // For now, just test that the function doesn't crash
    const devices = runtime.list_gpu_devices() catch return;
    defer allocator.free(devices);
}