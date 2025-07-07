const std = @import("std");
const print = std.debug.print;
const ghostnv = @import("ghostnv");
const vibrance = ghostnv.color_vibrance;
const drm = ghostnv.drm_driver;

const CliError = error{
    InvalidArgument,
    ProfileNotFound,
    PermissionDenied,
    HardwareNotSupported,
};

const Command = enum {
    help,
    list,
    apply,
    create,
    delete,
    auto,
    adjust,
    disable,
    status,
    monitor,
};

const CliArgs = struct {
    command: Command,
    profile_name: ?[]const u8 = null,
    vibrance: ?i8 = null,
    saturation: ?i8 = null,
    gamma: ?f32 = null,
    brightness: ?i8 = null,
    contrast: ?i8 = null,
    temperature: ?i16 = null,
    game_mode: ?vibrance.GameColorMode = null,
    auto_detect: bool = false,
    real_time: bool = false,
    
    pub fn parse(allocator: std.mem.Allocator, args: [][]const u8) !CliArgs {
        if (args.len < 2) {
            return CliArgs{ .command = .help };
        }
        
        const command_str = args[1];
        const command = std.meta.stringToEnum(Command, command_str) orelse {
            print("Unknown command: {s}\n", .{command_str});
            return CliArgs{ .command = .help };
        };
        
        var cli_args = CliArgs{ .command = command };
        
        var i: usize = 2;
        while (i < args.len) {
            const arg = args[i];
            
            if (std.mem.eql(u8, arg, "--profile") and i + 1 < args.len) {
                cli_args.profile_name = args[i + 1];
                i += 2;
            } else if (std.mem.eql(u8, arg, "--vibrance") and i + 1 < args.len) {
                cli_args.vibrance = try std.fmt.parseInt(i8, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, arg, "--saturation") and i + 1 < args.len) {
                cli_args.saturation = try std.fmt.parseInt(i8, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, arg, "--gamma") and i + 1 < args.len) {
                cli_args.gamma = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 2;
            } else if (std.mem.eql(u8, arg, "--brightness") and i + 1 < args.len) {
                cli_args.brightness = try std.fmt.parseInt(i8, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, arg, "--contrast") and i + 1 < args.len) {
                cli_args.contrast = try std.fmt.parseInt(i8, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, arg, "--temperature") and i + 1 < args.len) {
                cli_args.temperature = try std.fmt.parseInt(i16, args[i + 1], 10);
                i += 2;
            } else if (std.mem.eql(u8, arg, "--auto")) {
                cli_args.auto_detect = true;
                i += 1;
            } else if (std.mem.eql(u8, arg, "--real-time")) {
                cli_args.real_time = true;
                i += 1;
            } else if (std.mem.eql(u8, arg, "--game-mode") and i + 1 < args.len) {
                cli_args.game_mode = std.meta.stringToEnum(vibrance.GameColorMode, args[i + 1]);
                i += 2;
            } else {
                i += 1;
            }
        }
        
        return cli_args;
    }
};

fn print_help() void {
    print(
        \\GhostVibrance - Advanced Digital Vibrance Control for NVIDIA GPUs
        \\
        \\Usage: ghostvibrance <command> [options]
        \\
        \\Commands:
        \\  help                     Show this help message
        \\  list                     List all available profiles
        \\  apply <profile>          Apply a vibrance profile
        \\  create <profile>         Create a new profile with specified settings
        \\  delete <profile>         Delete a profile
        \\  auto                     Auto-detect and apply game profile
        \\  adjust <value>           Real-time vibrance adjustment (-50 to +100)
        \\  disable                  Disable digital vibrance
        \\  status                   Show current status and active profile
        \\  monitor                  Monitor and auto-apply profiles (daemon mode)
        \\
        \\Profile Creation Options:
        \\  --vibrance <-50 to 100>  Digital vibrance level (0 = neutral)
        \\  --saturation <-50 to 50> Color saturation (-50 to +50)
        \\  --gamma <0.8 to 3.0>     Gamma correction (2.2 = standard)
        \\  --brightness <-50 to 50> Brightness adjustment
        \\  --contrast <-50 to 50>   Contrast adjustment
        \\  --temperature <-1000 to 1000> Color temperature in Kelvin
        \\  --game-mode <mode>       Game optimization mode
        \\
        \\Game Modes:
        \\  standard                 Standard color reproduction
        \\  competitive              Enhanced visibility for competitive gaming
        \\  cinematic                Cinema-accurate colors
        \\  photography              Photo editing optimized
        \\  streaming                Content creation optimized
        \\
        \\Other Options:
        \\  --auto                   Enable automatic game detection
        \\  --real-time              Real-time adjustment mode
        \\
        \\Examples:
        \\  ghostvibrance apply Gaming
        \\  ghostvibrance create MyProfile --vibrance 50 --saturation 20
        \\  ghostvibrance auto
        \\  ghostvibrance adjust +10
        \\  ghostvibrance monitor --auto
        \\
        \\Pre-loaded Profiles:
        \\  Gaming, Competitive, Cinema, Streaming, Photography
        \\  Counter-Strike, Valorant, Apex Legends, Fortnite, Cyberpunk 2077
        \\
    );
}

fn initialize_vibrance_engine(allocator: std.mem.Allocator) !vibrance.VibranceEngine {
    // Initialize DRM driver
    var drm_driver = try drm.DrmDriver.init(allocator);
    try drm_driver.register();
    
    // Initialize vibrance engine
    var engine = vibrance.VibranceEngine.init(allocator, &drm_driver);
    try engine.load_default_profiles();
    
    return engine;
}

fn handle_list_command(engine: *vibrance.VibranceEngine) !void {
    const profiles = try engine.list_profiles();
    defer engine.allocator.free(profiles);
    
    print("📋 Available Vibrance Profiles:\n");
    print("═══════════════════════════════\n");
    
    for (profiles) |profile_name| {
        if (engine.profiles.get(profile_name)) |profile| {
            const active_marker = if (engine.active_profile != null and std.mem.eql(u8, engine.active_profile.?, profile_name)) " 🔥 ACTIVE" else "";
            print("  {s:<20} | Vibrance: {:3} | Mode: {s}{s}\n", 
                  .{ profile_name, profile.vibrance, profile.game_mode.toString(), active_marker });
        }
    }
    
    print("\nTotal profiles: {}\n", .{profiles.len});
}

fn handle_apply_command(engine: *vibrance.VibranceEngine, profile_name: []const u8) !void {
    engine.apply_profile(profile_name) catch |err| switch (err) {
        vibrance.VibranceError.ProfileNotFound => {
            print("❌ Profile '{s}' not found. Use 'ghostvibrance list' to see available profiles.\n", .{profile_name});
            return;
        },
        else => return err,
    };
    
    const stats = engine.get_performance_stats();
    print("✅ Applied profile '{s}' successfully!\n", .{profile_name});
    print("   Processing time: {:.2}ms\n", .{@as(f64, @floatFromInt(stats.processing_time_ns)) / 1_000_000.0});
}

fn handle_create_command(engine: *vibrance.VibranceEngine, args: CliArgs) !void {
    const profile_name = args.profile_name orelse {
        print("❌ Profile name required for create command\n");
        return CliError.InvalidArgument;
    };
    
    var profile = vibrance.VibranceProfile.init(profile_name);
    
    // Apply command line options
    if (args.vibrance) |v| profile.vibrance = v;
    if (args.saturation) |s| profile.saturation = s;
    if (args.gamma) |g| profile.gamma = g;
    if (args.brightness) |b| profile.brightness = b;
    if (args.contrast) |c| profile.contrast = c;
    if (args.temperature) |t| profile.temperature = t;
    if (args.game_mode) |gm| profile.game_mode = gm;
    
    try engine.create_profile(profile_name, profile);
    print("✅ Created profile '{s}' with:\n", .{profile_name});
    print("   Vibrance: {}, Saturation: {}, Gamma: {:.1}\n", .{ profile.vibrance, profile.saturation, profile.gamma });
    print("   Game Mode: {s}\n", .{profile.game_mode.toString()});
}

fn handle_auto_command(engine: *vibrance.VibranceEngine) !void {
    print("🔍 Auto-detecting active application...\n");
    
    // In a real implementation, this would scan active windows
    // For now, simulate with common game window titles
    const test_titles = [_][]const u8{
        "Counter-Strike 2",
        "VALORANT",
        "Apex Legends",
        "Fortnite",
        "Unknown Game",
    };
    
    for (test_titles) |title| {
        if (engine.auto_detect_game_profile(title)) |detected_profile| {
            print("🎮 Detected: {s}\n", .{title});
            try handle_apply_command(engine, detected_profile);
            return;
        }
    }
    
    print("ℹ️ No recognized game detected. Using 'Gaming' profile as fallback.\n");
    try handle_apply_command(engine, "Gaming");
}

fn handle_adjust_command(engine: *vibrance.VibranceEngine, adjustment_str: []const u8) !void {
    const adjustment = std.fmt.parseInt(i8, adjustment_str, 10) catch {
        print("❌ Invalid adjustment value. Use -50 to +100.\n");
        return CliError.InvalidArgument;
    };
    
    if (engine.active_profile == null) {
        print("❌ No active profile. Apply a profile first.\n");
        return;
    }
    
    try engine.real_time_adjust(adjustment);
    
    if (engine.get_active_profile()) |profile| {
        print("🎛️ Adjusted vibrance to {} ({}{})\n", .{ profile.vibrance, if (adjustment >= 0) "+" else "", adjustment });
    }
}

fn handle_status_command(engine: *vibrance.VibranceEngine) !void {
    print("📊 GhostVibrance Status\n");
    print("═══════════════════════\n");
    
    const stats = engine.get_performance_stats();
    
    if (stats.active_profile) |profile_name| {
        if (engine.get_active_profile()) |profile| {
            print("🔥 Active Profile: {s}\n", .{profile_name});
            print("   Vibrance: {}\n", .{profile.vibrance});
            print("   Saturation: {}\n", .{profile.saturation});
            print("   Gamma: {:.1}\n", .{profile.gamma});
            print("   Game Mode: {s}\n", .{profile.game_mode.toString()});
            print("   HDR: {s}\n", .{if (profile.hdr_enabled) "Enabled" else "Disabled"});
        }
    } else {
        print("💤 No active profile (Digital vibrance disabled)\n");
    }
    
    print("\n📈 Performance:\n");
    print("   Last processing time: {:.2}ms\n", .{@as(f64, @floatFromInt(stats.processing_time_ns)) / 1_000_000.0});
    print("   Profiles loaded: {}\n", .{stats.profiles_loaded});
    print("   Frames processed: {}\n", .{stats.frames_processed});
}

fn handle_monitor_command(engine: *vibrance.VibranceEngine, auto_detect: bool) !void {
    print("👁️ Starting GhostVibrance monitor (Press Ctrl+C to stop)\n");
    print("   Auto-detection: {s}\n", .{if (auto_detect) "Enabled" else "Disabled"});
    
    var last_window_title: [256]u8 = std.mem.zeroes([256]u8);
    
    while (true) {
        // Simulate window title detection
        // In real implementation, would use X11/Wayland APIs
        const current_title = "Counter-Strike 2"; // Simulated
        
        if (!std.mem.eql(u8, current_title, last_window_title[0..current_title.len])) {
            if (auto_detect) {
                if (engine.auto_detect_game_profile(current_title)) |profile| {
                    print("🔄 Window changed: {s} -> {s}\n", .{ current_title, profile });
                    try engine.apply_profile(profile);
                }
            }
            
            @memcpy(last_window_title[0..current_title.len], current_title);
        }
        
        // Sleep for 1 second
        std.time.sleep(1 * std.time.ns_per_s);
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    const cli_args = try CliArgs.parse(allocator, args);
    
    switch (cli_args.command) {
        .help => {
            print_help();
            return;
        },
        else => {},
    }
    
    // Initialize vibrance engine for all other commands
    var engine = initialize_vibrance_engine(allocator) catch |err| {
        switch (err) {
            drm.DrmError.RegistrationFailed => {
                print("❌ Failed to initialize display driver. Run as root or check permissions.\n");
                return CliError.PermissionDenied;
            },
            else => {
                print("❌ Hardware initialization failed: {}\n", .{err});
                return CliError.HardwareNotSupported;
            },
        }
    };
    defer engine.deinit();
    
    switch (cli_args.command) {
        .list => try handle_list_command(&engine),
        .apply => {
            const profile_name = cli_args.profile_name orelse {
                print("❌ Profile name required for apply command\n");
                return CliError.InvalidArgument;
            };
            try handle_apply_command(&engine, profile_name);
        },
        .create => try handle_create_command(&engine, cli_args),
        .delete => {
            const profile_name = cli_args.profile_name orelse {
                print("❌ Profile name required for delete command\n");
                return CliError.InvalidArgument;
            };
            if (engine.profiles.remove(profile_name)) {
                print("✅ Deleted profile '{s}'\n", .{profile_name});
            } else {
                print("❌ Profile '{s}' not found\n", .{profile_name});
            }
        },
        .auto => try handle_auto_command(&engine),
        .adjust => {
            const adjustment_str = cli_args.profile_name orelse {
                print("❌ Adjustment value required\n");
                return CliError.InvalidArgument;
            };
            try handle_adjust_command(&engine, adjustment_str);
        },
        .disable => {
            try engine.disable_vibrance();
            print("✅ Digital vibrance disabled\n");
        },
        .status => try handle_status_command(&engine),
        .monitor => try handle_monitor_command(&engine, cli_args.auto_detect),
        .help => unreachable, // Handled above
    }
}

// Simplified test runner for the CLI tool
test "cli argument parsing" {
    const allocator = std.testing.allocator;
    
    const args = [_][]const u8{ "ghostvibrance", "create", "TestProfile", "--vibrance", "50", "--saturation", "20" };
    const cli_args = try CliArgs.parse(allocator, &args);
    
    try std.testing.expect(cli_args.command == .create);
    try std.testing.expect(std.mem.eql(u8, cli_args.profile_name.?, "TestProfile"));
    try std.testing.expect(cli_args.vibrance.? == 50);
    try std.testing.expect(cli_args.saturation.? == 20);
}