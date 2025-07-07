# GhostNV: Next-Generation Open GPU Driver Framework

**Developed by CK Technology**

A revolutionary pure Zig implementation of NVIDIA open drivers with advanced AI features, Wayland optimization, and Linux-Ghost kernel integration.

![Zig](https://img.shields.io/badge/Zig-0.15.0--dev-orange?logo=zig)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux-blue?logo=linux)
![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)

---

## 🎯 **Mission Statement**

**CK Technology believes NVIDIA on Linux deserves fully open-source solutions.** GhostNV is our contribution to the community - completely free and open-source, advancing GPU driver technology for everyone.

---

## ⚙️ Core Features

* ✅ **Based on NVIDIA Open 575+ kernel module**
* 🧪 **Pre-patched for Bore/EEVDF schedulers**
* 🧠 **Game latency optimizations and scheduler affinity fixes**
* 🧰 **Eliminates DKMS entirely** — integrates cleanly into kernel builds
* 🔄 **Compatible with stock kernels** (mainline, Zen, TKG, Cachy)
* 🔧 **Supports runtime tuning via sysctl and nvidia-settings (userland)**

## 🚀 Revolutionary Pure Zig Features

### 🎮 **Gaming Performance Optimizations**
* **🌊 Variable Refresh Rate (VRR)** - Full VESA Adaptive-Sync support (48-500Hz)
* **🎯 G-SYNC Ultimate** - Hardware-accelerated with variable overdrive & HDR
* **⚡ Frame Generation** - AI-powered intermediate frame interpolation
* **🎨 Advanced Shader Cache** - 10x faster shader compilation with intelligent caching
* **⏱️ Ultra-Low Latency Mode** - Sub-millisecond input lag for competitive gaming

### 🧠 **CUDA Compute Engine**
* **🔥 Full CUDA Runtime** - Complete kernel launch and memory management
* **📊 CUDA Graphs** - Optimized execution with minimal CPU overhead
* **🎯 Multi-GPU Support** - Automatic workload distribution
* **⚡ Zero-Copy Operations** - Direct GPU memory access
* **🧮 Async Compute** - Overlapped compute and graphics workloads

### 🎥 **NVENC Video Encoding**
* **🎬 H.264/H.265/AV1** - Hardware-accelerated encoding for RTX 40 series
* **📡 Streaming Optimized** - Twitch/YouTube presets with adaptive bitrate
* **🎮 Game Capture** - Zero-copy frame capture with minimal performance impact
* **⏰ Low-Latency Encoding** - <16ms end-to-end for real-time streaming
* **🎚️ Advanced Rate Control** - CBR/VBR/CQ with lookahead optimization

### 🎨 **GhostVibrance - Digital Vibrance Revolution**
* **🌈 Superior to nVibrant** - Hardware-accelerated color processing
* **🎮 Game-Specific Profiles** - Auto-detection for CS2, Valorant, Apex, etc.
* **🔬 Advanced Color Science** - Individual RGB channel control
* **🎭 HDR Support** - 10-bit color with local dimming integration
* **⚡ Real-Time Adjustment** - Instant hotkey-based vibrance control

---

## 💡 Use Cases

* 🎮 **Gaming rigs** with `linux-ghost`
* 💻 **AI workloads** using CUDA/OpenCL (RTX 30/40 series supported)
* 🔧 **Custom kernels** needing NVIDIA support without relying on DKMS
* 🧪 **Testing & benchmarking** new NVIDIA performance patches with a streamlined dev loop

---

## 🚀 Integration with Linux-Ghost

When used with `linux-ghost`, `ghostnv`:

* Is embedded directly into the kernel tree at build time
* Requires no post-install hooks or recompilation
* Works out of the box with `ghostctl`, `phantomboot`, and `jarvis-nv`

---

## 📦 Build & Installation

### 🔨 Build All Features
```bash
# Build the complete GhostNV suite
zig build pure-zig --release=fast

# Build individual components
zig build cuda-test      # CUDA compute tests
zig build nvenc-test     # Video encoding tests
zig build gaming-test    # Gaming optimizations
zig build vrr-test       # VRR functionality
zig build ghostvibrance  # Digital vibrance CLI tool

# Run comprehensive benchmarks
zig build benchmarks
```

### 🎮 GhostVibrance Usage
```bash
# Apply gaming profile
./zig-out/bin/ghostvibrance apply Gaming

# Create competitive CS2 profile
./zig-out/bin/ghostvibrance create CS2-Pro --vibrance 65 --saturation 30 --game-mode competitive

# Auto-detect game and apply profile
./zig-out/bin/ghostvibrance auto

# Real-time vibrance adjustment
./zig-out/bin/ghostvibrance adjust +15

# Monitor mode with auto-detection
./zig-out/bin/ghostvibrance monitor --auto
```

### 🚀 Legacy Installation
For traditional kernel module installation:
```bash
git clone https://github.com/ghostkellz/ghostnv.git
cd ghostnv
make modules
sudo make modules_install
```

---

## 🧠 Future Goals

* [ ] Support for full in-tree driver compilation
* [ ] Automatic `ghostctl` integration
* [ ] Per-scheduler tuning profiles
* [ ] Zig build helpers and runtime diagnostic CLI
* [ ] Wayland & KWin specific tuning for Plasma/NVIDIA

---

## 🔮 Vision

ghostnv brings NVIDIA open-source drivers into the modern kernel dev pipeline — fast, modular, and tuned for real-world performance. Whether you're compiling `linux-ghost` or just want a cleaner NVIDIA experience on Arch, ghostnv is your ghost-powered GPU companion.

**ghostnv — No DKMS. No drama. Just performance.**

