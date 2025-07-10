const std = @import("std");
const runtime = @import("runtime.zig");
const Allocator = std.mem.Allocator;

/// Docker/Podman compatibility shim for GhostNV Container Runtime
/// Implements the OCI Runtime Specification for GPU containers
pub const DockerShim = struct {
    allocator: Allocator,
    runtime: *runtime.ContainerRuntime,
    oci_runtime_path: []const u8,
    
    pub fn init(allocator: Allocator, container_runtime: *runtime.ContainerRuntime) DockerShim {
        return DockerShim{
            .allocator = allocator,
            .runtime = container_runtime,
            .oci_runtime_path = "/usr/bin/ghostnv-oci",
        };
    }
    
    /// Register GhostNV as Docker runtime
    pub fn register_docker_runtime(self: *DockerShim) !void {
        std.log.info("Registering GhostNV as Docker runtime");
        
        // Create Docker daemon configuration
        const docker_config = DockerRuntimeConfig{
            .name = "ghostnv",
            .path = self.oci_runtime_path,
            .runtime_args = &[_][]const u8{ "--gpu-support", "--zig-optimized" },
        };
        
        try self.write_docker_config(docker_config);
        
        // Install OCI runtime binary
        try self.install_oci_runtime();
        
        std.log.info("GhostNV runtime registered. Use with: docker run --runtime=ghostnv --gpus all <image>");
    }
    
    /// OCI runtime spec implementation
    pub fn handle_oci_command(self: *DockerShim, args: [][]const u8) !void {
        if (args.len < 2) {
            return error.InvalidOCICommand;
        }
        
        const command = args[1];
        
        if (std.mem.eql(u8, command, "create")) {
            try self.oci_create(args[2..]);
        } else if (std.mem.eql(u8, command, "start")) {
            try self.oci_start(args[2..]);
        } else if (std.mem.eql(u8, command, "delete")) {
            try self.oci_delete(args[2..]);
        } else if (std.mem.eql(u8, command, "state")) {
            try self.oci_state(args[2..]);
        } else if (std.mem.eql(u8, command, "kill")) {
            try self.oci_kill(args[2..]);
        } else {
            std.log.err("Unknown OCI command: {s}", .{command});
            return error.UnknownOCICommand;
        }
    }
    
    /// Create container from OCI bundle
    fn oci_create(self: *DockerShim, args: [][]const u8) !void {
        if (args.len < 2) {
            return error.InvalidOCICreate;
        }
        
        const container_id = args[0];
        const bundle_path = args[1];
        
        std.log.info("OCI Create: container_id={s}, bundle={s}", .{ container_id, bundle_path });
        
        // Parse OCI runtime specification
        const spec = try self.parse_oci_spec(bundle_path);
        defer self.allocator.free(spec.process.args);
        
        // Convert OCI spec to GhostNV container config
        const config = try self.oci_spec_to_container_config(spec, container_id);
        
        // Create container
        const handle = try self.runtime.create_container(config);
        
        // Store container handle for later operations
        try self.store_container_handle(container_id, handle);
        
        std.log.info("Container {s} created successfully", .{container_id});
    }
    
    /// Start previously created container
    fn oci_start(self: *DockerShim, args: [][]const u8) !void {
        if (args.len < 1) {
            return error.InvalidOCIStart;
        }
        
        const container_id = args[0];
        
        std.log.info("OCI Start: container_id={s}", .{container_id});
        
        // Retrieve stored container handle
        var handle = try self.get_container_handle(container_id);
        
        // Start the container
        const executable = "/bin/bash"; // Default - should be from OCI spec
        var exec_args = [_][]const u8{executable};
        
        try self.runtime.start_container(&handle, executable, &exec_args);
        
        std.log.info("Container {s} started successfully", .{container_id});
    }
    
    /// Delete container
    fn oci_delete(self: *DockerShim, args: [][]const u8) !void {
        if (args.len < 1) {
            return error.InvalidOCIDelete;
        }
        
        const container_id = args[0];
        
        std.log.info("OCI Delete: container_id={s}", .{container_id});
        
        var handle = try self.get_container_handle(container_id);
        try self.runtime.stop_container(&handle);
        
        // Remove stored handle
        try self.remove_container_handle(container_id);
        
        std.log.info("Container {s} deleted successfully", .{container_id});
    }
    
    /// Get container state
    fn oci_state(self: *DockerShim, args: [][]const u8) !void {
        if (args.len < 1) {
            return error.InvalidOCIState;
        }
        
        const container_id = args[0];
        
        const handle = self.get_container_handle(container_id) catch {
            // Container not found
            const state = OCIState{
                .oci_version = "1.0.0",
                .id = container_id,
                .status = "stopped",
                .pid = 0,
                .bundle = "",
            };
            
            try self.output_oci_state(state);
            return;
        };
        
        const stats = try self.runtime.get_container_stats(&handle);
        
        const state = OCIState{
            .oci_version = "1.0.0",
            .id = container_id,
            .status = if (stats.pid > 0) "running" else "stopped",
            .pid = stats.pid,
            .bundle = "/var/lib/ghostnv/bundles/", // Mock bundle path
        };
        
        try self.output_oci_state(state);
    }
    
    /// Kill container
    fn oci_kill(self: *DockerShim, args: [][]const u8) !void {
        if (args.len < 1) {
            return error.InvalidOCIKill;
        }
        
        const container_id = args[0];
        const signal = if (args.len > 1) args[1] else "TERM";
        
        std.log.info("OCI Kill: container_id={s}, signal={s}", .{ container_id, signal });
        
        var handle = try self.get_container_handle(container_id);
        
        if (std.mem.eql(u8, signal, "KILL") or std.mem.eql(u8, signal, "9")) {
            try self.runtime.stop_container(&handle);
        } else {
            // For other signals, just log (real implementation would send signal to container)
            std.log.info("Sending signal {s} to container {s}", .{ signal, container_id });
        }
    }
    
    // Helper methods
    
    fn write_docker_config(self: *DockerShim, config: DockerRuntimeConfig) !void {
        const docker_config_path = "/etc/docker/daemon.json";
        
        // Read existing config or create new
        var docker_config = std.json.Value{ .object = std.json.ObjectMap.init(self.allocator) };
        
        if (std.fs.readFileAlloc(self.allocator, docker_config_path, 1024 * 1024)) |existing_content| {
            defer self.allocator.free(existing_content);
            
            docker_config = std.json.parseFromSlice(std.json.Value, self.allocator, existing_content, .{}) catch docker_config;
        } else |_| {
            // File doesn't exist or can't be read, use empty config
        }
        
        // Add GhostNV runtime
        if (docker_config.object.get("runtimes") == null) {
            try docker_config.object.put("runtimes", std.json.Value{ .object = std.json.ObjectMap.init(self.allocator) });
        }
        
        var runtimes = docker_config.object.getPtr("runtimes").?.object;
        
        var ghostnv_runtime = std.json.ObjectMap.init(self.allocator);
        try ghostnv_runtime.put("path", std.json.Value{ .string = config.path });
        
        try runtimes.put("ghostnv", std.json.Value{ .object = ghostnv_runtime });
        
        // Write updated config
        const config_json = try std.json.stringifyAlloc(self.allocator, docker_config, .{ .whitespace = .indent_2 });
        defer self.allocator.free(config_json);
        
        try std.fs.writeFileAbsolute(docker_config_path, config_json);
        
        std.log.info("Docker configuration updated at {s}", .{docker_config_path});
    }
    
    fn install_oci_runtime(self: *DockerShim) !void {
        // Create OCI runtime wrapper script
        const oci_script =
            \\#!/bin/bash
            \\# GhostNV OCI Runtime Wrapper
            \\exec /usr/bin/ghostnv-container-oci "$@"
        ;
        
        try std.fs.writeFileAbsolute(self.oci_runtime_path, oci_script);
        
        // Make executable
        const file = try std.fs.openFileAbsolute(self.oci_runtime_path, .{});
        defer file.close();
        
        try file.chmod(0o755);
        
        std.log.info("OCI runtime installed at {s}", .{self.oci_runtime_path});
    }
    
    fn parse_oci_spec(self: *DockerShim, bundle_path: []const u8) !OCISpec {
        const config_path = try std.fmt.allocPrint(self.allocator, "{s}/config.json", .{bundle_path});
        defer self.allocator.free(config_path);
        
        const config_content = try std.fs.cwd().readFileAlloc(self.allocator, config_path, 1024 * 1024);
        defer self.allocator.free(config_content);
        
        // Simple JSON parsing for demo - real implementation would use proper JSON parser
        return OCISpec{
            .oci_version = "1.0.0",
            .process = OCIProcess{
                .args = try self.allocator.dupe([]const u8, &[_][]const u8{ "/bin/bash" }),
                .env = &[_][]const u8{},
                .cwd = "/",
            },
            .hostname = "ghostnv-container",
        };
    }
    
    fn oci_spec_to_container_config(self: *DockerShim, spec: OCISpec, container_id: []const u8) !runtime.ContainerConfig {
        _ = spec;
        
        return runtime.ContainerConfig{
            .name = try self.allocator.dupe(u8, container_id),
            .gpu_access = runtime.GpuAccess{
                .enabled = true,
                .device_ids = &[_]u32{0}, // Default to GPU 0
                .capabilities = &[_][]const u8{ "compute", "video", "graphics", "utility" },
            },
            .limits = runtime.ResourceLimits{
                .memory_limit_mb = 4096, // 4GB default
                .cpu_cores = 2.0,
                .gpu_memory_limit_mb = 8192, // 8GB GPU memory
            },
            .security = runtime.SecurityPolicy{
                .seccomp_profile = "docker-default",
                .apparmor_profile = "docker-default",
                .capabilities = &[_][]const u8{},
            },
        };
    }
    
    fn store_container_handle(self: *DockerShim, container_id: []const u8, handle: runtime.ContainerHandle) !void {
        const state_dir = "/var/lib/ghostnv/containers";
        std.fs.makeDirAbsolute(state_dir) catch |err| {
            if (err != error.PathAlreadyExists) {
                return err;
            }
        };
        
        const state_path = try std.fmt.allocPrint(self.allocator, "{s}/{s}.json", .{ state_dir, container_id });
        defer self.allocator.free(state_path);
        
        // Serialize handle to JSON (simplified)
        const handle_json = try std.fmt.allocPrint(self.allocator, 
            \\{{
            \\  "id": {},
            \\  "namespace_id": {},
            \\  "pid": {}
            \\}}
        , .{ handle.id, handle.namespace.id, handle.pid });
        defer self.allocator.free(handle_json);
        
        const file = try std.fs.createFileAbsolute(state_path, .{});
        defer file.close();
        try file.writeAll(handle_json);
    }
    
    fn get_container_handle(self: *DockerShim, container_id: []const u8) !runtime.ContainerHandle {
        const state_path = try std.fmt.allocPrint(self.allocator, "/var/lib/ghostnv/containers/{s}.json", .{container_id});
        defer self.allocator.free(state_path);
        
        const file = try std.fs.openFileAbsolute(state_path, .{});
        defer file.close();
        const handle_json = try file.readToEndAlloc(self.allocator, 1024);
        defer self.allocator.free(handle_json);
        
        // Parse JSON (simplified - real implementation would use proper JSON parser)
        // For now, return a mock handle
        return runtime.ContainerHandle{
            .id = 12345,
            .namespace = runtime.ContainerNamespace{
                .id = 12345,
                .name = try self.allocator.dupe(u8, container_id),
                .pid_namespace = 12345,
                .net_namespace = 12345,
                .mount_namespace = 12345,
                .user_namespace = 12345,
            },
            .pid = 1234,
        };
    }
    
    fn remove_container_handle(self: *DockerShim, container_id: []const u8) !void {
        const state_path = try std.fmt.allocPrint(self.allocator, "/var/lib/ghostnv/containers/{s}.json", .{container_id});
        defer self.allocator.free(state_path);
        
        std.fs.deleteFileAbsolute(state_path) catch |err| {
            if (err != error.FileNotFound) {
                return err;
            }
        };
    }
    
    fn output_oci_state(self: *DockerShim, state: OCIState) !void {
        _ = self;
        
        // Output JSON state to stdout (OCI requirement)
        std.debug.print(
            \\{{
            \\  "ociVersion": "{s}",
            \\  "id": "{s}",
            \\  "status": "{s}",
            \\  "pid": {},
            \\  "bundle": "{s}"
            \\}}
        , .{ state.oci_version, state.id, state.status, state.pid, state.bundle });
    }
};

// Type definitions for OCI compatibility

const DockerRuntimeConfig = struct {
    name: []const u8,
    path: []const u8,
    runtime_args: []const []const u8,
};

const OCISpec = struct {
    oci_version: []const u8,
    process: OCIProcess,
    hostname: []const u8,
};

const OCIProcess = struct {
    args: []const []const u8,
    env: []const []const u8,
    cwd: []const u8,
};

const OCIState = struct {
    oci_version: []const u8,
    id: []const u8,
    status: []const u8,
    pid: i32,
    bundle: []const u8,
};

// Main entry point for OCI runtime
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    // Initialize runtime
    var container_runtime = try runtime.ContainerRuntime.init(allocator);
    defer container_runtime.deinit();
    
    // Initialize Docker shim
    var shim = DockerShim.init(allocator, &container_runtime);
    
    // Handle OCI command
    // Convert args to correct type
    var converted_args = try allocator.alloc([]const u8, args.len);
    defer allocator.free(converted_args);
    for (args, 0..) |arg, i| {
        converted_args[i] = arg;
    }
    try shim.handle_oci_command(converted_args);
}