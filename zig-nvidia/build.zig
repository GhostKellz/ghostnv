const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build options
    const gpu_generation = b.option([]const u8, "gpu_generation", "Target GPU generation (turing, ampere, ada)") orelse "ada";
    const hybrid_mode = b.option(bool, "hybrid_mode", "Enable hybrid Zig/C driver mode") orelse false;
    const legacy_mode = b.option(bool, "legacy_mode", "Enable legacy compatibility mode") orelse false;
    const gaming_optimized = b.option(bool, "gaming_optimized", "Enable gaming-specific optimizations") orelse false;

    // Main library
    const ghostnv = b.addStaticLibrary(.{
        .name = "ghostnv",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Build configuration
    const config = b.addOptions();
    config.addOption([]const u8, "gpu_generation", gpu_generation);
    config.addOption(bool, "hybrid_mode", hybrid_mode);
    config.addOption(bool, "legacy_mode", legacy_mode);
    config.addOption(bool, "gaming_optimized", gaming_optimized);

    ghostnv.root_module.addOptions("config", config);
    b.installArtifact(ghostnv);

    // Tools
    const tools = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "gpu-test", .path = "tools/gpu-test.zig" },
        .{ .name = "ghostvibrance", .path = "tools/ghostvibrance.zig" },
        .{ .name = "test-nvenc", .path = "tools/test-nvenc.zig" },
        .{ .name = "test-rtx", .path = "tools/test-rtx.zig" },
        .{ .name = "ghostnv-container", .path = "tools/ghostnv-container.zig" },
        .{ .name = "memory-bandwidth", .path = "benchmarks/memory_bandwidth.zig" },
        .{ .name = "benchmark", .path = "benchmarks/main.zig" },
    };

    for (tools) |tool| {
        const exe = b.addExecutable(.{
            .name = tool.name,
            .root_source_file = b.path(tool.path),
            .target = target,
            .optimize = optimize,
        });
        exe.root_module.addOptions("config", config);
        b.installArtifact(exe);
    }

    // Tests
    const test_step = b.step("test", "Run unit tests");
    
    const test_files = [_][]const u8{
        "src/main.zig",
        "src/hal/pci.zig",
        "src/hal/memory.zig", 
        "src/hal/command.zig",
        "src/hal/interrupt.zig",
        "src/drm/driver.zig",
        "src/color/vibrance.zig",
    };

    for (test_files) |test_file| {
        const unit_tests = b.addTest(.{
            .root_source_file = b.path(test_file),
            .target = target,
            .optimize = optimize,
        });
        unit_tests.root_module.addOptions("config", config);
        
        const run_unit_tests = b.addRunArtifact(unit_tests);
        test_step.dependOn(&run_unit_tests.step);
    }

    // Integration tests
    const integration_tests = b.addTest(.{
        .root_source_file = b.path("tests/integration_tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    integration_tests.root_module.addOptions("config", config);
    
    const run_integration_tests = b.addRunArtifact(integration_tests);
    test_step.dependOn(&run_integration_tests.step);

    // Benchmarks
    const bench_step = b.step("bench", "Run benchmarks");
    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("benchmarks/main.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    benchmark.root_module.addOptions("config", config);
    
    const run_benchmark = b.addRunArtifact(benchmark);
    bench_step.dependOn(&run_benchmark.step);

    // Documentation
    const docs_step = b.step("docs", "Generate documentation");
    const docs = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    docs.root_module.addOptions("config", config);
    
    const docs_install = b.addInstallDirectory(.{
        .source_dir = docs.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&docs_install.step);

    // Default run step
    const run_cmd = b.addRunArtifact(b.addExecutable(.{
        .name = "ghostnv-demo",
        .root_source_file = b.path("tools/gpu-test.zig"),
        .target = target,
        .optimize = optimize,
    }));
    run_cmd.root_module.addOptions("config", config);

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run GPU test demo");
    run_step.dependOn(&run_cmd.step);
}