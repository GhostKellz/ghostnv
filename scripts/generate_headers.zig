const std = @import("std");

const HEADER_CONTENT =
    \\#ifndef GHOSTNV_FFI_H
    \\#define GHOSTNV_FFI_H
    \\
    \\#include <stdint.h>
    \\#include <stdbool.h>
    \\
    \\#ifdef __cplusplus
    \\extern "C" {
    \\#endif
    \\
    \\// ═══════════════════════════════════════════════════════
    \\// Core Types
    \\// ═══════════════════════════════════════════════════════
    \\
    \\typedef enum {
    \\    GHOSTNV_OK = 0,
    \\    GHOSTNV_ERROR_INVALID_DEVICE = -1,
    \\    GHOSTNV_ERROR_INVALID_VALUE = -2,
    \\    GHOSTNV_ERROR_NOT_SUPPORTED = -3,
    \\    GHOSTNV_ERROR_PERMISSION_DENIED = -4,
    \\    GHOSTNV_ERROR_NOT_INITIALIZED = -5,
    \\    GHOSTNV_ERROR_MEMORY_ALLOCATION = -6,
    \\    GHOSTNV_ERROR_DEVICE_BUSY = -7,
    \\} GhostNVResult;
    \\
    \\typedef struct {
    \\    uint32_t device_id;
    \\    char name[256];
    \\    char driver_version[32];
    \\    uint32_t pci_bus;
    \\    uint32_t pci_device;
    \\    uint32_t pci_function;
    \\    bool supports_gsync;
    \\    bool supports_vrr;
    \\    bool supports_hdr;
    \\} GhostNVDevice;
    \\
    \\// ═══════════════════════════════════════════════════════
    \\// Digital Vibrance API
    \\// ═══════════════════════════════════════════════════════
    \\
    \\typedef struct {
    \\    int8_t vibrance;           // -50 to 100
    \\    int8_t saturation;         // -50 to 50  
    \\    float gamma;               // 0.8 to 3.0
    \\    int8_t brightness;         // -50 to 50
    \\    int8_t contrast;           // -50 to 50
    \\    int16_t temperature;       // -1000 to 1000 Kelvin
    \\    int8_t red_vibrance;       // -50 to 100
    \\    int8_t green_vibrance;     // -50 to 100
    \\    int8_t blue_vibrance;      // -50 to 100
    \\    bool preserve_skin_tones;
    \\    bool enhance_foliage;
    \\    bool boost_sky_colors;
    \\} GhostNVVibranceProfile;
    \\
    \\// Initialize vibrance engine
    \\GhostNVResult ghostnv_vibrance_init(void);
    \\
    \\// Apply vibrance profile by name
    \\GhostNVResult ghostnv_vibrance_apply_profile(const char* profile_name);
    \\
    \\// Create custom profile
    \\GhostNVResult ghostnv_vibrance_create_profile(const char* name, const GhostNVVibranceProfile* profile);
    \\
    \\// Real-time vibrance adjustment
    \\GhostNVResult ghostnv_vibrance_adjust(int8_t delta);
    \\
    \\// Get current vibrance settings
    \\GhostNVResult ghostnv_vibrance_get_current(GhostNVVibranceProfile* out_profile);
    \\
    \\// List available profiles (returns count, fills names array)
    \\int32_t ghostnv_vibrance_list_profiles(char names[][64], int32_t max_count);
    \\
    \\// Auto-detect game and apply profile
    \\GhostNVResult ghostnv_vibrance_auto_detect(const char* window_title);
    \\
    \\// Disable vibrance
    \\GhostNVResult ghostnv_vibrance_disable(void);
    \\
    \\// ═══════════════════════════════════════════════════════
    \\// G-SYNC / VRR API
    \\// ═══════════════════════════════════════════════════════
    \\
    \\typedef enum {
    \\    GSYNC_DISABLED = 0,
    \\    GSYNC_COMPATIBLE = 1,
    \\    GSYNC_CERTIFIED = 2,
    \\    GSYNC_ULTIMATE = 3,
    \\    GSYNC_ESPORTS = 4,
    \\} GhostNVGSyncMode;
    \\
    \\typedef enum {
    \\    GAME_COMPETITIVE_FPS = 0,
    \\    GAME_IMMERSIVE_SINGLE_PLAYER = 1,
    \\    GAME_RACING = 2,
    \\    GAME_CINEMA = 3,
    \\} GhostNVGameType;
    \\
    \\typedef struct {
    \\    GhostNVGSyncMode mode;
    \\    uint32_t min_refresh_hz;
    \\    uint32_t max_refresh_hz;
    \\    uint32_t current_refresh_hz;
    \\    bool ultra_low_latency;
    \\    bool variable_overdrive;
    \\    bool motion_blur_reduction;
    \\    bool hdr_enabled;
    \\    uint32_t peak_brightness_nits;
    \\} GhostNVGSyncStatus;
    \\
    \\// Enable G-SYNC with specified mode
    \\GhostNVResult ghostnv_gsync_enable(uint32_t device_id, GhostNVGSyncMode mode);
    \\
    \\// Disable G-SYNC
    \\GhostNVResult ghostnv_gsync_disable(uint32_t device_id);
    \\
    \\// Get current G-SYNC status
    \\GhostNVResult ghostnv_gsync_get_status(uint32_t device_id, GhostNVGSyncStatus* status);
    \\
    \\// Set refresh rate (for VRR)
    \\GhostNVResult ghostnv_gsync_set_refresh_rate(uint32_t device_id, uint32_t refresh_hz);
    \\
    \\// Optimize for specific game type
    \\GhostNVResult ghostnv_gsync_optimize_for_game(uint32_t device_id, GhostNVGameType game_type);
    \\
    \\// Enable/disable ultra low latency
    \\GhostNVResult ghostnv_gsync_set_ultra_low_latency(uint32_t device_id, bool enabled);
    \\
    \\// ═══════════════════════════════════════════════════════
    \\// Performance & System API
    \\// ═══════════════════════════════════════════════════════
    \\
    \\typedef struct {
    \\    uint32_t gpu_clock_mhz;
    \\    uint32_t memory_clock_mhz;
    \\    uint32_t temperature_c;
    \\    uint32_t fan_speed_rpm;
    \\    uint32_t power_draw_watts;
    \\    uint32_t gpu_utilization_percent;
    \\    uint32_t memory_utilization_percent;
    \\    float average_frametime_ms;
    \\    uint32_t current_fps;
    \\} GhostNVPerformanceInfo;
    \\
    \\// Get performance information
    \\GhostNVResult ghostnv_performance_get_info(uint32_t device_id, GhostNVPerformanceInfo* info);
    \\
    \\// Set performance level (0=auto, 1=power save, 2=balanced, 3=performance, 4=max)
    \\GhostNVResult ghostnv_performance_set_level(uint32_t device_id, uint32_t level);
    \\
    \\// Enable/disable frame generation
    \\GhostNVResult ghostnv_performance_set_frame_generation(uint32_t device_id, bool enabled, uint8_t max_frames);
    \\
    \\// ═══════════════════════════════════════════════════════
    \\// Device Management API
    \\// ═══════════════════════════════════════════════════════
    \\
    \\// Initialize GhostNV driver interface
    \\GhostNVResult ghostnv_init(void);
    \\
    \\// Cleanup GhostNV driver interface  
    \\void ghostnv_cleanup(void);
    \\
    \\// Get number of NVIDIA devices
    \\int32_t ghostnv_get_device_count(void);
    \\
    \\// Get device information
    \\GhostNVResult ghostnv_get_device_info(uint32_t device_id, GhostNVDevice* device);
    \\
    \\// Check if specific feature is supported
    \\bool ghostnv_supports_feature(uint32_t device_id, const char* feature_name);
    \\
    \\// Get version information
    \\const char* ghostnv_get_version(void);
    \\const char* ghostnv_get_build_info(void);
    \\
    \\#ifdef __cplusplus
    \\}
    \\#endif
    \\
    \\#endif // GHOSTNV_FFI_H
    \\
;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len < 2 or !std.mem.eql(u8, args[1], "generate-headers")) {
        std.debug.print("Usage: {s} generate-headers\n", .{args[0]});
        return;
    }
    
    // Create output directory
    try std.fs.cwd().makePath("zig-out/include");
    
    // Write header file
    try std.fs.cwd().writeFile(.{ .sub_path = "zig-out/include/ghostnv_ffi.h", .data = HEADER_CONTENT });
    
    std.debug.print("Generated C header: zig-out/include/ghostnv_ffi.h\n", .{});
    
    // Also create a pkg-config file for easier integration
    const pkg_config_content = try std.fmt.allocPrint(allocator,
        \\prefix=/usr/local
        \\exec_prefix=${{prefix}}
        \\libdir=${{exec_prefix}}/lib
        \\includedir=${{prefix}}/include
        \\
        \\Name: GhostNV
        \\Description: Pure Zig NVIDIA Driver with GPU optimizations
        \\Version: 1.0.0
        \\Libs: -L${{libdir}} -lghostnv
        \\Cflags: -I${{includedir}}
        \\
    , .{});
    defer allocator.free(pkg_config_content);
    
    try std.fs.cwd().writeFile(.{ .sub_path = "zig-out/ghostnv.pc", .data = pkg_config_content });
    
    std.debug.print("Generated pkg-config: zig-out/ghostnv.pc\n", .{});
    std.debug.print("To install:\n", .{});
    std.debug.print("  sudo cp zig-out/lib/libghostnv.so /usr/local/lib/\n", .{});
    std.debug.print("  sudo cp zig-out/include/ghostnv_ffi.h /usr/local/include/\n", .{});
    std.debug.print("  sudo cp zig-out/ghostnv.pc /usr/local/lib/pkgconfig/\n", .{});
    std.debug.print("  sudo ldconfig\n", .{});
}