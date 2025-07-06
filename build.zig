const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build mode options
    const legacy_mode = b.option(bool, "legacy", "Build using legacy NVIDIA driver mode") orelse false;
    const patched_mode = b.option(bool, "patched", "Build with performance patches applied") orelse false;
    const realtime_mode = b.option(bool, "realtime", "Build with real-time optimizations") orelse false;
    const audio_mode = b.option(bool, "audio", "Build with RTX Voice/Audio enhancements") orelse false;
    const debug_mode = b.option(bool, "debug-driver", "Build with debug patches and symbols") orelse false;
    const pure_zig_mode = b.option(bool, "pure-zig", "Build pure Zig NVIDIA driver (experimental)") orelse false;

    // GhostNV module
    const ghostnv_mod = b.addModule("ghostnv", .{
        .root_source_file = b.path("zig/ghostnv.zig"),
        .target = target,
    });

    // Main executable
    const exe = b.addExecutable(.{
        .name = "ghostnv",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "ghostnv", .module = ghostnv_mod },
            },
        }),
    });

    b.installArtifact(exe);

    // Pure Zig NVIDIA driver module
    const zig_nvidia_mod = b.addModule("zig-nvidia", .{
        .root_source_file = b.path("zig-nvidia/src/main.zig"),
        .target = target,
    });

    // Build steps
    const modules_step = b.step("modules", "Build NVIDIA kernel modules");
    const clean_step = b.step("clean", "Clean build artifacts");
    const patch_step = b.step("patch", "Apply patches based on configuration");
    const validate_step = b.step("validate", "Validate patches and checksums");
    const pure_zig_step = b.step("pure-zig", "Build pure Zig NVIDIA driver");
    const wayland_test_step = b.step("wayland-test", "Test Wayland compositor functionality");

    // Legacy build step - calls original Makefile
    const legacy_cmd = b.addSystemCommand(&.{ "make", "modules" });
    legacy_cmd.cwd = b.path(".");
    const legacy_build_step = b.step("legacy", "Build using legacy Makefile system");
    legacy_build_step.dependOn(&legacy_cmd.step);

    // Patch application step
    const patch_cmd = b.addRunArtifact(exe);
    patch_cmd.addArgs(&.{"patch"});
    if (patched_mode) patch_cmd.addArgs(&.{"--performance"});
    if (realtime_mode) patch_cmd.addArgs(&.{"--realtime"});
    if (audio_mode) patch_cmd.addArgs(&.{"--audio"});
    if (debug_mode) patch_cmd.addArgs(&.{"--debug"});
    patch_step.dependOn(&patch_cmd.step);

    // Validation step
    const validate_cmd = b.addRunArtifact(exe);
    validate_cmd.addArgs(&.{"validate"});
    validate_step.dependOn(&validate_cmd.step);

    // Module build step with patch dependency
    const build_cmd = b.addSystemCommand(&.{ "make", "modules" });
    build_cmd.cwd = b.path(".");
    modules_step.dependOn(&patch_cmd.step);
    modules_step.dependOn(&build_cmd.step);

    // Pure Zig driver build step
    const zig_nvidia_exe = b.addExecutable(.{
        .name = "zig-nvidia-driver",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    const install_zig_nvidia = b.addInstallArtifact(zig_nvidia_exe, .{});
    pure_zig_step.dependOn(&install_zig_nvidia.step);

    // Wayland test executable
    const wayland_test = b.addExecutable(.{
        .name = "wayland-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/src/wayland/compositor.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    const run_wayland_test = b.addRunArtifact(wayland_test);
    wayland_test_step.dependOn(&run_wayland_test.step);

    // Clean step
    const clean_cmd = b.addSystemCommand(&.{ "make", "clean" });
    clean_cmd.cwd = b.path(".");
    clean_step.dependOn(&clean_cmd.step);

    // Run step
    const run_step = b.step("run", "Run the GhostNV CLI");
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    run_step.dependOn(&run_cmd.step);

    // Test step
    const test_step = b.step("test", "Run tests");
    const mod_tests = b.addTest(.{
        .root_module = ghostnv_mod,
    });
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const zig_nvidia_tests = b.addTest(.{
        .root_module = zig_nvidia_mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const run_exe_tests = b.addRunArtifact(exe_tests);
    const run_zig_nvidia_tests = b.addRunArtifact(zig_nvidia_tests);
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_zig_nvidia_tests.step);
}
