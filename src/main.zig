const std = @import("std");
const ghostnv = @import("ghostnv");
const print = std.debug.print;
const ArrayList = std.ArrayList;

const Command = enum {
    help,
    patch,
    validate,
    build,
    legacy,
    realtime,
    audio,
    version,
    clean,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try print_help();
        return;
    }

    const command = std.meta.stringToEnum(Command, args[1]) orelse {
        print("Unknown command: {s}\n", .{args[1]});
        try print_help();
        return;
    };

    switch (command) {
        .help => try print_help(),
        .patch => try handle_patch(allocator, args[2..]),
        .validate => try handle_validate(allocator),
        .build => try handle_build(allocator, args[2..]),
        .legacy => try handle_legacy(allocator),
        .realtime => try handle_realtime(allocator),
        .audio => try handle_audio(allocator),
        .version => try handle_version(allocator),
        .clean => try handle_clean(allocator),
    }
}

fn print_help() !void {
    print(
        \\GhostNV - NVIDIA Open Driver Zig Integration
        \\
        \\Usage: ghostnv <command> [options]
        \\
        \\Commands:
        \\  help                 Show this help message
        \\  patch [--options]    Apply patches based on configuration
        \\  validate             Validate patches and checksums
        \\  build [mode]         Build kernel modules
        \\  legacy               Build using legacy Makefile system
        \\  realtime             Build with real-time optimizations
        \\  audio                Build with RTX Voice/Audio enhancements
        \\  version              Show driver version information
        \\  clean                Clean build artifacts
        \\
        \\Patch Options:
        \\  --performance        Apply performance patches
        \\  --realtime           Apply real-time optimizations
        \\  --audio              Apply RTX Voice/Audio patches
        \\  --debug              Apply debug patches
        \\
        \\Examples:
        \\  ghostnv patch --performance --realtime
        \\  ghostnv build realtime
        \\  ghostnv audio
        \\
    );
}

fn handle_patch(allocator: std.mem.Allocator, args: [][]const u8) !void {
    // Detect NVIDIA version
    const version = ghostnv.DriverVersion{ .major = 575, .minor = 0, .patch = 0 };
    var patch_manager = ghostnv.PatchManager.init(allocator, "patches", version);
    
    // Try to detect actual version
    const detected_version = patch_manager.detect_version() catch version;
    patch_manager.version = detected_version;

    var patch_types = ArrayList(ghostnv.PatchType).init(allocator);
    defer patch_types.deinit();

    // Parse patch arguments
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--performance")) {
            try patch_types.append(.performance);
        } else if (std.mem.eql(u8, arg, "--realtime")) {
            try patch_types.append(.realtime);
        } else if (std.mem.eql(u8, arg, "--audio")) {
            try patch_types.append(.audio);
        } else if (std.mem.eql(u8, arg, "--debug")) {
            try patch_types.append(.debug);
        }
    }

    // Apply common patches by default
    try patch_types.append(.common);

    if (patch_types.items.len == 1) {
        print("No specific patches requested, applying common patches only\n");
    }

    try patch_manager.apply_patches(patch_types.items);
}

fn handle_validate(allocator: std.mem.Allocator) !void {
    const version = ghostnv.DriverVersion{ .major = 575, .minor = 0, .patch = 0 };
    var patch_manager = ghostnv.PatchManager.init(allocator, "patches", version);
    
    const detected_version = patch_manager.detect_version() catch version;
    patch_manager.version = detected_version;

    try patch_manager.validate_patches();
}

fn handle_build(allocator: std.mem.Allocator, args: [][]const u8) !void {
    const mode = if (args.len > 0) blk: {
        break :blk std.meta.stringToEnum(ghostnv.BuildMode, args[0]) orelse .legacy;
    } else .legacy;

    const version = ghostnv.DriverVersion{ .major = 575, .minor = 0, .patch = 0 };
    var patch_manager = ghostnv.PatchManager.init(allocator, "patches", version);
    
    const detected_version = patch_manager.detect_version() catch version;
    patch_manager.version = detected_version;

    var builder = ghostnv.KernelBuilder.init(allocator, mode, patch_manager);
    try builder.build();
}

fn handle_legacy(allocator: std.mem.Allocator) !void {
    const version = ghostnv.DriverVersion{ .major = 575, .minor = 0, .patch = 0 };
    var patch_manager = ghostnv.PatchManager.init(allocator, "patches", version);
    var builder = ghostnv.KernelBuilder.init(allocator, .legacy, patch_manager);
    try builder.build();
}

fn handle_realtime(allocator: std.mem.Allocator) !void {
    const version = ghostnv.DriverVersion{ .major = 575, .minor = 0, .patch = 0 };
    var patch_manager = ghostnv.PatchManager.init(allocator, "patches", version);
    
    const detected_version = patch_manager.detect_version() catch version;
    patch_manager.version = detected_version;

    var builder = ghostnv.KernelBuilder.init(allocator, .realtime, patch_manager);
    try builder.build();
}

fn handle_audio(allocator: std.mem.Allocator) !void {
    print("Initializing RTX Voice/Audio support\n");
    
    var audio_manager = ghostnv.AudioManager.init(allocator);
    try audio_manager.enable_rtx_voice();
    try audio_manager.setup_noise_cancellation();

    const version = ghostnv.DriverVersion{ .major = 575, .minor = 0, .patch = 0 };
    var patch_manager = ghostnv.PatchManager.init(allocator, "patches", version);
    
    const detected_version = patch_manager.detect_version() catch version;
    patch_manager.version = detected_version;

    var builder = ghostnv.KernelBuilder.init(allocator, .audio, patch_manager);
    try builder.build();
}

fn handle_version(allocator: std.mem.Allocator) !void {
    print("GhostNV - NVIDIA Open Driver Zig Integration\n");
    print("Zig Version: {s}\n", .{@import("builtin").zig_version_string});
    
    const version = ghostnv.DriverVersion{ .major = 575, .minor = 0, .patch = 0 };
    var patch_manager = ghostnv.PatchManager.init(allocator, "patches", version);
    
    const detected_version = patch_manager.detect_version() catch version;
    const version_str = try detected_version.to_string(allocator);
    defer allocator.free(version_str);
    
    print("Detected NVIDIA Version: {s}\n", .{version_str});
}

fn handle_clean(allocator: std.mem.Allocator) !void {
    _ = allocator;
    print("Cleaning build artifacts\n");
    
    // This would typically call make clean or remove specific directories
    print("Clean operation would be implemented here\n");
    print("For now, run: make clean\n");
}
