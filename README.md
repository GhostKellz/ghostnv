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

## ⚙️ Features

* ✅ **Based on NVIDIA Open 570+ kernel module**
* 🧪 **Pre-patched for Bore/EEVDF schedulers**
* 🧠 **Game latency optimizations and scheduler affinity fixes**
* 🧰 **Eliminates DKMS entirely** — integrates cleanly into kernel builds
* 🔄 **Compatible with stock kernels** (mainline, Zen, TKG, Cachy)
* 🔧 **Supports runtime tuning via sysctl and nvidia-settings (userland)**

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

## 📦 Installation Plans

> Full installation support is coming soon via `ghostctl install ghostnv`

For now, you can manually install `ghostnv` by cloning the repo and applying it to your kernel build environment (e.g., linux-ghost-tkg):

```sh
git clone https://github.com/ghostkellz/ghostnv.git
cp -r ghostnv/* /usr/src/linux-ghost/driver/nvidia/
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

