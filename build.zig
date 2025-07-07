const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Driver architecture options
    const driver_mode = b.option(enum { hybrid, pure_zig, legacy_c, auto }, "driver-mode", "Driver architecture selection") orelse .auto;
    const legacy_mode = b.option(bool, "legacy", "Build using legacy NVIDIA driver mode") orelse false;
    const patched_mode = b.option(bool, "patched", "Build with performance patches applied") orelse false;
    const realtime_mode = b.option(bool, "realtime", "Build with real-time optimizations") orelse false;
    const audio_mode = b.option(bool, "audio", "Build with RTX Voice/Audio enhancements") orelse false;
    const debug_mode = b.option(bool, "debug-driver", "Build with debug patches and symbols") orelse false;
    const pure_zig_mode = b.option(bool, "pure-zig", "Build pure Zig NVIDIA driver (experimental)") orelse false;
    
    // New feature options
    const cuda_mode = b.option(bool, "cuda", "Enable CUDA compute support") orelse true;
    const nvenc_mode = b.option(bool, "nvenc", "Enable NVENC video encoding") orelse true;
    const vrr_mode = b.option(bool, "vrr", "Enable Variable Refresh Rate support") orelse true;
    const gaming_mode = b.option(bool, "gaming", "Enable gaming performance optimizations") orelse true;
    const frame_gen_mode = b.option(bool, "frame-gen", "Enable AI frame generation") orelse false;

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
    
    // New feature build steps
    const cuda_test_step = b.step("cuda-test", "Test CUDA compute functionality");
    const nvenc_test_step = b.step("nvenc-test", "Test NVENC video encoding");
    const gaming_test_step = b.step("gaming-test", "Test gaming performance optimizations");
    const vrr_test_step = b.step("vrr-test", "Test Variable Refresh Rate functionality");
    const benchmarks_step = b.step("benchmarks", "Run performance benchmarks");

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
    if (cuda_mode) patch_cmd.addArgs(&.{"--cuda"});
    if (nvenc_mode) patch_cmd.addArgs(&.{"--nvenc"});
    if (gaming_mode) patch_cmd.addArgs(&.{"--gaming"});
    if (vrr_mode) patch_cmd.addArgs(&.{"--vrr"});
    if (frame_gen_mode) patch_cmd.addArgs(&.{"--frame-gen"});
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

    // CUDA test executable
    const cuda_test = b.addExecutable(.{
        .name = "cuda-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/src/cuda/runtime.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    const run_cuda_test = b.addRunArtifact(cuda_test);
    cuda_test_step.dependOn(&run_cuda_test.step);

    // NVENC test executable
    const nvenc_test = b.addExecutable(.{
        .name = "nvenc-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/src/nvenc/encoder.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    const run_nvenc_test = b.addRunArtifact(nvenc_test);
    nvenc_test_step.dependOn(&run_nvenc_test.step);

    // Gaming performance test executable
    const gaming_test = b.addExecutable(.{
        .name = "gaming-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/src/gaming/performance.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    const run_gaming_test = b.addRunArtifact(gaming_test);
    gaming_test_step.dependOn(&run_gaming_test.step);

    // VRR test (uses DRM driver)
    const vrr_test = b.addExecutable(.{
        .name = "vrr-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/src/drm/driver.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    const run_vrr_test = b.addRunArtifact(vrr_test);
    vrr_test_step.dependOn(&run_vrr_test.step);

    // Comprehensive benchmark suite
    const benchmark_exe = b.addExecutable(.{
        .name = "ghostnv-benchmarks",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/benchmarks/main.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    const run_benchmarks = b.addRunArtifact(benchmark_exe);
    benchmarks_step.dependOn(&run_benchmarks.step);

    // GhostVibrance CLI tool
    const ghostvibrance_exe = b.addExecutable(.{
        .name = "ghostvibrance",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/tools/ghostvibrance.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    b.installArtifact(ghostvibrance_exe);
    
    const ghostvibrance_step = b.step("ghostvibrance", "Build GhostVibrance CLI tool");
    ghostvibrance_step.dependOn(&b.addInstallArtifact(ghostvibrance_exe, .{}).step);

    // GhostNV Container Runtime
    const container_exe = b.addExecutable(.{
        .name = "ghostnv-container",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/tools/ghostnv-container.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    b.installArtifact(container_exe);
    
    const container_step = b.step("container", "Build GhostNV Container Runtime");
    container_step.dependOn(&b.addInstallArtifact(container_exe, .{}).step);

    // OCI Runtime for Docker compatibility
    const oci_exe = b.addExecutable(.{
        .name = "ghostnv-container-oci",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/src/container/docker_shim.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    b.installArtifact(oci_exe);
    
    const oci_step = b.step("oci", "Build OCI Runtime for Docker/Podman compatibility");
    oci_step.dependOn(&b.addInstallArtifact(oci_exe, .{}).step);

    // FFI Shared Library for Rust integration
    const ffi_lib = b.addSharedLibrary(.{
        .name = "ghostnv",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/src/ffi/ghostnv_ffi.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    // Set shared library version
    ffi_lib.version = .{ .major = 1, .minor = 0, .patch = 0 };
    
    b.installArtifact(ffi_lib);
    
    const ffi_step = b.step("ffi", "Build FFI shared library for Rust integration");
    ffi_step.dependOn(&b.addInstallArtifact(ffi_lib, .{}).step);

    // Generate C header file
    const header_step = b.step("ffi-headers", "Generate C headers for FFI");
    const header_cmd = b.addRunArtifact(exe);
    header_cmd.addArgs(&.{"generate-headers"});
    header_step.dependOn(&header_cmd.step);

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

    // Test step - comprehensive test suite
    const test_step = b.step("test", "Run all tests");
    
    // Unit tests
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/tests/unit_tests.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    // Integration tests
    const integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig-nvidia/tests/integration_tests.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zig-nvidia", .module = zig_nvidia_mod },
            },
        }),
    });
    
    // Module tests
    const mod_tests = b.addTest(.{
        .root_module = ghostnv_mod,
    });
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });
    const zig_nvidia_tests = b.addTest(.{
        .root_module = zig_nvidia_mod,
    });
    
    // Run all tests
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const run_integration_tests = b.addRunArtifact(integration_tests);
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const run_exe_tests = b.addRunArtifact(exe_tests);
    const run_zig_nvidia_tests = b.addRunArtifact(zig_nvidia_tests);
    
    test_step.dependOn(&run_unit_tests.step);
    test_step.dependOn(&run_integration_tests.step);
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_zig_nvidia_tests.step);
    
    // Separate test steps for different test types
    const unit_test_step = b.step("test-unit", "Run unit tests only");
    unit_test_step.dependOn(&run_unit_tests.step);
    
    const integration_test_step = b.step("test-integration", "Run integration tests only");
    integration_test_step.dependOn(&run_integration_tests.step);
}
