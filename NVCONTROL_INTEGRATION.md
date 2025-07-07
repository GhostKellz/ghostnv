# GhostNV ↔ nvcontrol Integration Guide

**Bridging Pure Zig GPU Driver with Rust GUI Control**

---

## 🎯 Overview

This document provides the complete integration specification for connecting the **GhostNV Pure Zig NVIDIA Driver** with the existing **nvcontrol (Rust)** project. The integration enables the Rust GUI and CLI tools to control all GhostNV features including digital vibrance, G-SYNC, VRR, and performance settings.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                nvcontrol (Rust)                         │
│  ┌─────────────────┬─────────────────────────────────┐  │
│  │   GUI (egui)    │       CLI (nvctl)               │  │
│  └─────────────────┴─────────────────────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   FFI Interface                         │
│              (C ABI + cbindgen)                         │
├─────────────────────────────────────────────────────────┤
│                 GhostNV (Zig)                          │
│  ┌─────────────┬─────────────┬─────────────────────────┐ │
│  │ Vibrance    │ G-SYNC      │ Performance/VRR         │ │
│  │ Engine      │ Manager     │ Optimizer               │ │
│  └─────────────┴─────────────┴─────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                NVIDIA Hardware                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔌 FFI Interface Specification

### 1. **C-Compatible Header Generation**

**File**: `zig-nvidia/ffi/ghostnv_ffi.h` (auto-generated)

```c
#ifndef GHOSTNV_FFI_H
#define GHOSTNV_FFI_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════
// Core Types
// ═══════════════════════════════════════════════════════

typedef enum {
    GHOSTNV_OK = 0,
    GHOSTNV_ERROR_INVALID_DEVICE = -1,
    GHOSTNV_ERROR_INVALID_VALUE = -2,
    GHOSTNV_ERROR_NOT_SUPPORTED = -3,
    GHOSTNV_ERROR_PERMISSION_DENIED = -4,
    GHOSTNV_ERROR_NOT_INITIALIZED = -5,
} GhostNVResult;

typedef struct {
    uint32_t device_id;
    char name[256];
    char driver_version[32];
    uint32_t pci_bus;
    uint32_t pci_device;
    uint32_t pci_function;
    bool supports_gsync;
    bool supports_vrr;
    bool supports_hdr;
} GhostNVDevice;

// ═══════════════════════════════════════════════════════
// Digital Vibrance API
// ═══════════════════════════════════════════════════════

typedef struct {
    int8_t vibrance;           // -50 to 100
    int8_t saturation;         // -50 to 50  
    float gamma;               // 0.8 to 3.0
    int8_t brightness;         // -50 to 50
    int8_t contrast;           // -50 to 50
    int16_t temperature;       // -1000 to 1000 Kelvin
    int8_t red_vibrance;       // -50 to 100
    int8_t green_vibrance;     // -50 to 100
    int8_t blue_vibrance;      // -50 to 100
    bool preserve_skin_tones;
    bool enhance_foliage;
    bool boost_sky_colors;
} GhostNVVibranceProfile;

// Initialize vibrance engine
GhostNVResult ghostnv_vibrance_init(void);

// Apply vibrance profile by name
GhostNVResult ghostnv_vibrance_apply_profile(const char* profile_name);

// Create custom profile
GhostNVResult ghostnv_vibrance_create_profile(const char* name, const GhostNVVibranceProfile* profile);

// Real-time vibrance adjustment
GhostNVResult ghostnv_vibrance_adjust(int8_t delta);

// Get current vibrance settings
GhostNVResult ghostnv_vibrance_get_current(GhostNVVibranceProfile* out_profile);

// List available profiles (returns count, fills names array)
int32_t ghostnv_vibrance_list_profiles(char names[][64], int32_t max_count);

// Auto-detect game and apply profile
GhostNVResult ghostnv_vibrance_auto_detect(const char* window_title);

// Disable vibrance
GhostNVResult ghostnv_vibrance_disable(void);

// ═══════════════════════════════════════════════════════
// G-SYNC / VRR API
// ═══════════════════════════════════════════════════════

typedef enum {
    GSYNC_DISABLED = 0,
    GSYNC_COMPATIBLE = 1,
    GSYNC_CERTIFIED = 2,
    GSYNC_ULTIMATE = 3,
    GSYNC_ESPORTS = 4,
} GhostNVGSyncMode;

typedef enum {
    GAME_COMPETITIVE_FPS = 0,
    GAME_IMMERSIVE_SINGLE_PLAYER = 1,
    GAME_RACING = 2,
    GAME_CINEMA = 3,
} GhostNVGameType;

typedef struct {
    GhostNVGSyncMode mode;
    uint32_t min_refresh_hz;
    uint32_t max_refresh_hz;
    uint32_t current_refresh_hz;
    bool ultra_low_latency;
    bool variable_overdrive;
    bool motion_blur_reduction;
    bool hdr_enabled;
    uint32_t peak_brightness_nits;
} GhostNVGSyncStatus;

// Enable G-SYNC with specified mode
GhostNVResult ghostnv_gsync_enable(uint32_t device_id, GhostNVGSyncMode mode);

// Disable G-SYNC
GhostNVResult ghostnv_gsync_disable(uint32_t device_id);

// Get current G-SYNC status
GhostNVResult ghostnv_gsync_get_status(uint32_t device_id, GhostNVGSyncStatus* status);

// Set refresh rate (for VRR)
GhostNVResult ghostnv_gsync_set_refresh_rate(uint32_t device_id, uint32_t refresh_hz);

// Optimize for specific game type
GhostNVResult ghostnv_gsync_optimize_for_game(uint32_t device_id, GhostNVGameType game_type);

// Enable/disable ultra low latency
GhostNVResult ghostnv_gsync_set_ultra_low_latency(uint32_t device_id, bool enabled);

// ═══════════════════════════════════════════════════════
// Performance & System API
// ═══════════════════════════════════════════════════════

typedef struct {
    uint32_t gpu_clock_mhz;
    uint32_t memory_clock_mhz;
    uint32_t temperature_c;
    uint32_t fan_speed_rpm;
    uint32_t power_draw_watts;
    uint32_t gpu_utilization_percent;
    uint32_t memory_utilization_percent;
    float average_frametime_ms;
    uint32_t current_fps;
} GhostNVPerformanceInfo;

// Get performance information
GhostNVResult ghostnv_performance_get_info(uint32_t device_id, GhostNVPerformanceInfo* info);

// Set performance level (0=auto, 1=power save, 2=balanced, 3=performance, 4=max)
GhostNVResult ghostnv_performance_set_level(uint32_t device_id, uint32_t level);

// Enable/disable frame generation
GhostNVResult ghostnv_performance_set_frame_generation(uint32_t device_id, bool enabled, uint8_t max_frames);

// ═══════════════════════════════════════════════════════
// Device Management API
// ═══════════════════════════════════════════════════════

// Initialize GhostNV driver interface
GhostNVResult ghostnv_init(void);

// Cleanup GhostNV driver interface  
void ghostnv_cleanup(void);

// Get number of NVIDIA devices
int32_t ghostnv_get_device_count(void);

// Get device information
GhostNVResult ghostnv_get_device_info(uint32_t device_id, GhostNVDevice* device);

// Check if specific feature is supported
bool ghostnv_supports_feature(uint32_t device_id, const char* feature_name);

#ifdef __cplusplus
}
#endif

#endif // GHOSTNV_FFI_H
```

---

## 🦀 Rust Integration Code

### 1. **Cargo.toml Dependencies**

```toml
[dependencies]
# Existing dependencies...

# FFI and binding generation
libc = "0.2"
bindgen = "0.69"

# Optional: For runtime dynamic loading
libloading = "0.8"

[build-dependencies]
bindgen = "0.69"
```

### 2. **Build Script** (`build.rs`)

```rust
use bindgen;
use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to tell rustc to link the ghostnv library
    println!("cargo:rustc-link-lib=ghostnv");
    println!("cargo:rustc-link-search=native=/usr/lib/ghostnv");
    
    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("/usr/include/ghostnv/ghostnv_ffi.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("ghostnv_bindings.rs"))
        .expect("Couldn't write bindings!");
}
```

### 3. **Rust Wrapper Module** (`src/ghostnv.rs`)

```rust
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};

// Include generated bindings
include!(concat!(env!("OUT_DIR"), "/ghostnv_bindings.rs"));

pub struct GhostNV {
    initialized: bool,
}

#[derive(Debug, Clone)]
pub struct VibranceProfile {
    pub vibrance: i8,
    pub saturation: i8,
    pub gamma: f32,
    pub brightness: i8,
    pub contrast: i8,
    pub temperature: i16,
    pub red_vibrance: i8,
    pub green_vibrance: i8,
    pub blue_vibrance: i8,
    pub preserve_skin_tones: bool,
    pub enhance_foliage: bool,
    pub boost_sky_colors: bool,
}

#[derive(Debug, Clone)]
pub struct GSyncStatus {
    pub mode: GSyncMode,
    pub min_refresh_hz: u32,
    pub max_refresh_hz: u32,
    pub current_refresh_hz: u32,
    pub ultra_low_latency: bool,
    pub variable_overdrive: bool,
    pub motion_blur_reduction: bool,
    pub hdr_enabled: bool,
    pub peak_brightness_nits: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum GSyncMode {
    Disabled = 0,
    Compatible = 1,
    Certified = 2,
    Ultimate = 3,
    Esports = 4,
}

#[derive(Debug, Clone, Copy)]
pub enum GameType {
    CompetitiveFPS = 0,
    ImmersiveSinglePlayer = 1,
    Racing = 2,
    Cinema = 3,
}

impl GhostNV {
    pub fn new() -> Result<Self, String> {
        unsafe {
            match ghostnv_init() {
                GhostNVResult_GHOSTNV_OK => Ok(GhostNV { initialized: true }),
                error => Err(format!("Failed to initialize GhostNV: {:?}", error)),
            }
        }
    }

    pub fn get_device_count(&self) -> i32 {
        unsafe { ghostnv_get_device_count() }
    }

    // ═══════════════════════════════════════════════════════
    // Digital Vibrance Methods
    // ═══════════════════════════════════════════════════════

    pub fn vibrance_apply_profile(&self, profile_name: &str) -> Result<(), String> {
        let c_name = CString::new(profile_name).map_err(|_| "Invalid profile name")?;
        
        unsafe {
            match ghostnv_vibrance_apply_profile(c_name.as_ptr()) {
                GhostNVResult_GHOSTNV_OK => Ok(()),
                error => Err(format!("Failed to apply profile: {:?}", error)),
            }
        }
    }

    pub fn vibrance_create_profile(&self, name: &str, profile: &VibranceProfile) -> Result<(), String> {
        let c_name = CString::new(name).map_err(|_| "Invalid profile name")?;
        let c_profile = GhostNVVibranceProfile {
            vibrance: profile.vibrance,
            saturation: profile.saturation,
            gamma: profile.gamma,
            brightness: profile.brightness,
            contrast: profile.contrast,
            temperature: profile.temperature,
            red_vibrance: profile.red_vibrance,
            green_vibrance: profile.green_vibrance,
            blue_vibrance: profile.blue_vibrance,
            preserve_skin_tones: profile.preserve_skin_tones,
            enhance_foliage: profile.enhance_foliage,
            boost_sky_colors: profile.boost_sky_colors,
        };

        unsafe {
            match ghostnv_vibrance_create_profile(c_name.as_ptr(), &c_profile) {
                GhostNVResult_GHOSTNV_OK => Ok(()),
                error => Err(format!("Failed to create profile: {:?}", error)),
            }
        }
    }

    pub fn vibrance_adjust(&self, delta: i8) -> Result<(), String> {
        unsafe {
            match ghostnv_vibrance_adjust(delta) {
                GhostNVResult_GHOSTNV_OK => Ok(()),
                error => Err(format!("Failed to adjust vibrance: {:?}", error)),
            }
        }
    }

    pub fn vibrance_get_current(&self) -> Result<VibranceProfile, String> {
        let mut c_profile = GhostNVVibranceProfile {
            vibrance: 0,
            saturation: 0,
            gamma: 2.2,
            brightness: 0,
            contrast: 0,
            temperature: 0,
            red_vibrance: 0,
            green_vibrance: 0,
            blue_vibrance: 0,
            preserve_skin_tones: true,
            enhance_foliage: false,
            boost_sky_colors: false,
        };

        unsafe {
            match ghostnv_vibrance_get_current(&mut c_profile) {
                GhostNVResult_GHOSTNV_OK => Ok(VibranceProfile {
                    vibrance: c_profile.vibrance,
                    saturation: c_profile.saturation,
                    gamma: c_profile.gamma,
                    brightness: c_profile.brightness,
                    contrast: c_profile.contrast,
                    temperature: c_profile.temperature,
                    red_vibrance: c_profile.red_vibrance,
                    green_vibrance: c_profile.green_vibrance,
                    blue_vibrance: c_profile.blue_vibrance,
                    preserve_skin_tones: c_profile.preserve_skin_tones,
                    enhance_foliage: c_profile.enhance_foliage,
                    boost_sky_colors: c_profile.boost_sky_colors,
                }),
                error => Err(format!("Failed to get current profile: {:?}", error)),
            }
        }
    }

    pub fn vibrance_list_profiles(&self) -> Result<Vec<String>, String> {
        const MAX_PROFILES: usize = 32;
        let mut names: [[c_char; 64]; MAX_PROFILES] = [[0; 64]; MAX_PROFILES];
        
        unsafe {
            let count = ghostnv_vibrance_list_profiles(names.as_mut_ptr(), MAX_PROFILES as i32);
            
            if count < 0 {
                return Err("Failed to list profiles".to_string());
            }
            
            let mut result = Vec::new();
            for i in 0..(count as usize) {
                if let Ok(name) = CStr::from_ptr(names[i].as_ptr()).to_str() {
                    if !name.is_empty() {
                        result.push(name.to_string());
                    }
                }
            }
            
            Ok(result)
        }
    }

    pub fn vibrance_auto_detect(&self, window_title: &str) -> Result<(), String> {
        let c_title = CString::new(window_title).map_err(|_| "Invalid window title")?;
        
        unsafe {
            match ghostnv_vibrance_auto_detect(c_title.as_ptr()) {
                GhostNVResult_GHOSTNV_OK => Ok(()),
                error => Err(format!("Auto-detection failed: {:?}", error)),
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // G-SYNC / VRR Methods
    // ═══════════════════════════════════════════════════════

    pub fn gsync_enable(&self, device_id: u32, mode: GSyncMode) -> Result<(), String> {
        unsafe {
            match ghostnv_gsync_enable(device_id, mode as u32) {
                GhostNVResult_GHOSTNV_OK => Ok(()),
                error => Err(format!("Failed to enable G-SYNC: {:?}", error)),
            }
        }
    }

    pub fn gsync_disable(&self, device_id: u32) -> Result<(), String> {
        unsafe {
            match ghostnv_gsync_disable(device_id) {
                GhostNVResult_GHOSTNV_OK => Ok(()),
                error => Err(format!("Failed to disable G-SYNC: {:?}", error)),
            }
        }
    }

    pub fn gsync_get_status(&self, device_id: u32) -> Result<GSyncStatus, String> {
        let mut c_status = GhostNVGSyncStatus {
            mode: GhostNVGSyncMode_GSYNC_DISABLED,
            min_refresh_hz: 60,
            max_refresh_hz: 60,
            current_refresh_hz: 60,
            ultra_low_latency: false,
            variable_overdrive: false,
            motion_blur_reduction: false,
            hdr_enabled: false,
            peak_brightness_nits: 100,
        };

        unsafe {
            match ghostnv_gsync_get_status(device_id, &mut c_status) {
                GhostNVResult_GHOSTNV_OK => {
                    let mode = match c_status.mode {
                        GhostNVGSyncMode_GSYNC_DISABLED => GSyncMode::Disabled,
                        GhostNVGSyncMode_GSYNC_COMPATIBLE => GSyncMode::Compatible,
                        GhostNVGSyncMode_GSYNC_CERTIFIED => GSyncMode::Certified,
                        GhostNVGSyncMode_GSYNC_ULTIMATE => GSyncMode::Ultimate,
                        GhostNVGSyncMode_GSYNC_ESPORTS => GSyncMode::Esports,
                        _ => GSyncMode::Disabled,
                    };

                    Ok(GSyncStatus {
                        mode,
                        min_refresh_hz: c_status.min_refresh_hz,
                        max_refresh_hz: c_status.max_refresh_hz,
                        current_refresh_hz: c_status.current_refresh_hz,
                        ultra_low_latency: c_status.ultra_low_latency,
                        variable_overdrive: c_status.variable_overdrive,
                        motion_blur_reduction: c_status.motion_blur_reduction,
                        hdr_enabled: c_status.hdr_enabled,
                        peak_brightness_nits: c_status.peak_brightness_nits,
                    })
                },
                error => Err(format!("Failed to get G-SYNC status: {:?}", error)),
            }
        }
    }

    pub fn gsync_set_refresh_rate(&self, device_id: u32, refresh_hz: u32) -> Result<(), String> {
        unsafe {
            match ghostnv_gsync_set_refresh_rate(device_id, refresh_hz) {
                GhostNVResult_GHOSTNV_OK => Ok(()),
                error => Err(format!("Failed to set refresh rate: {:?}", error)),
            }
        }
    }

    pub fn gsync_optimize_for_game(&self, device_id: u32, game_type: GameType) -> Result<(), String> {
        unsafe {
            match ghostnv_gsync_optimize_for_game(device_id, game_type as u32) {
                GhostNVResult_GHOSTNV_OK => Ok(()),
                error => Err(format!("Failed to optimize for game: {:?}", error)),
            }
        }
    }

    pub fn gsync_set_ultra_low_latency(&self, device_id: u32, enabled: bool) -> Result<(), String> {
        unsafe {
            match ghostnv_gsync_set_ultra_low_latency(device_id, enabled) {
                GhostNVResult_GHOSTNV_OK => Ok(()),
                error => Err(format!("Failed to set ultra low latency: {:?}", error)),
            }
        }
    }
}

impl Drop for GhostNV {
    fn drop(&mut self) {
        if self.initialized {
            unsafe {
                ghostnv_cleanup();
            }
        }
    }
}

// ═══════════════════════════════════════════════════════
// Convenience functions for nvctl CLI
// ═══════════════════════════════════════════════════════

pub fn quick_vibrance_set(value: i8) -> Result<(), String> {
    let ghostnv = GhostNV::new()?;
    ghostnv.vibrance_adjust(value)
}

pub fn quick_gsync_toggle(device_id: u32, enabled: bool) -> Result<(), String> {
    let ghostnv = GhostNV::new()?;
    if enabled {
        ghostnv.gsync_enable(device_id, GSyncMode::Compatible)
    } else {
        ghostnv.gsync_disable(device_id)
    }
}
```

---

## 🔧 nvctl CLI Integration Examples

### Command Examples:

```bash
# Digital Vibrance
nvctl vibrance 50                    # Set vibrance to 50%
nvctl vibrance get                   # Get current vibrance
nvctl vibrance profile Gaming        # Apply gaming profile
nvctl vibrance profile create MyCS2 --vibrance 65 --saturation 30

# G-SYNC Control  
nvctl gsync enable                   # Enable G-SYNC Compatible
nvctl gsync ultimate                 # Enable G-SYNC Ultimate
nvctl gsync ull on                   # Enable Ultra Low Latency
nvctl gsync refresh 144              # Set to 144Hz
nvctl gsync optimize competitive     # Optimize for competitive gaming

# VRR Control
nvctl vrr enable 48 165              # Enable VRR 48-165Hz  
nvctl vrr status                     # Show VRR status
nvctl vrr lfc on                     # Enable Low Framerate Compensation

# Performance
nvctl perf level 3                   # Set max performance
nvctl perf framegen on 2             # Enable frame generation (max 2 frames)
nvctl perf info                      # Show performance info
```

---

## 🖥️ GUI Integration Points

### Main UI Sections:

1. **Digital Vibrance Tab**
   - Vibrance slider (-50 to 100)
   - Per-channel RGB controls
   - Game profile dropdown
   - Auto-detection toggle
   - Preview/reset buttons

2. **Display Settings Tab**
   - G-SYNC mode selection
   - VRR enable/disable
   - Refresh rate controls
   - Ultra Low Latency toggle
   - HDR settings

3. **Performance Tab**
   - Performance level slider
   - Frame generation controls
   - Real-time monitoring graphs
   - Temperature/fan controls

4. **Profiles Tab**
   - Profile management (create/edit/delete)
   - Import/export profiles
   - Game-specific associations
   - Auto-detection rules

---

## 📋 Implementation Checklist

### Phase 1: Basic Integration
- [ ] Generate C FFI headers from Zig code
- [ ] Create Rust bindgen integration
- [ ] Implement basic vibrance controls
- [ ] Add CLI command compatibility

### Phase 2: Advanced Features  
- [ ] G-SYNC/VRR integration
- [ ] Performance monitoring
- [ ] Profile management
- [ ] Auto-detection system

### Phase 3: GUI Enhancement
- [ ] Real-time sliders and controls
- [ ] Live preview functionality
- [ ] Performance graphs
- [ ] Profile import/export

### Phase 4: Polish & Testing
- [ ] Error handling and validation
- [ ] Multi-GPU support
- [ ] Configuration persistence
- [ ] Comprehensive testing

---

## 🚀 Quick Start Guide

1. **Build GhostNV with FFI support:**
   ```bash
   cd ghostnv
   zig build ffi-headers  # Generate C headers
   zig build shared-lib   # Build shared library
   ```

2. **Update nvcontrol project:**
   ```bash
   cd nvcontrol
   # Add ghostnv dependency to Cargo.toml
   # Copy build.rs and ghostnv.rs integration code
   cargo build
   ```

3. **Test integration:**
   ```bash
   nvctl vibrance 50      # Should work with GhostNV backend
   nvctl gsync status     # Should show G-SYNC status
   ```

---

This integration provides a **seamless bridge** between your existing Rust nvcontrol project and the new GhostNV Zig driver, enabling full control of digital vibrance, G-SYNC, VRR, and performance features through the familiar nvctl interface!