const std = @import("std");
const ghostnv = @import("ghostnv");
const container = ghostnv.container_runtime;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Parse command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 2) {
        try print_help();
        return;
    }
    
    // Initialize container runtime
    var runtime = try container.ContainerRuntime.init(allocator);
    defer runtime.deinit();
    
    // Initialize CLI interface
    var cli = container.ContainerCLI.init(&runtime);
    
    // Execute command
    try cli.execute_command(args);
}

fn print_help() !void {
    std.debug.print(
        \\GhostNV Container Runtime - Native GPU Container Support
        \\
        \\Usage:
        \\  ghostnv-container <command> [options]
        \\
        \\Commands:
        \\  run <name> <image> [cmd]     Create and start GPU-enabled container
        \\  list                         List active containers
        \\  stop <id>                    Stop running container
        \\  stats <id>                   Show container resource usage
        \\  devices                      List available GPU devices
        \\  help                         Show this help message
        \\
        \\Examples:
        \\  # Run TensorFlow with GPU support
        \\  ghostnv-container run ml-training tensorflow/tensorflow:latest-gpu python train.py
        \\
        \\  # Run Blender headless rendering
        \\  ghostnv-container run render blender:latest blender --background --render-output /output/
        \\
        \\  # Run CUDA development environment
        \\  ghostnv-container run cuda-dev nvidia/cuda:12.3-devel-ubuntu22.04 bash
        \\
        \\  # Show container performance
        \\  ghostnv-container stats 1234
        \\
        \\  # List GPU devices
        \\  ghostnv-container devices
        \\
        \\Features:
        \\  • Native Zig implementation for maximum performance
        \\  • Zero-overhead GPU passthrough
        \\  • Fine-grained resource control
        \\  • Secure namespace isolation
        \\  • Real-time GPU monitoring
        \\  • Multi-GPU support
        \\
    );
}