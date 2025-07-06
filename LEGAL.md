# Legal Compliance & Trademark Guidelines

## 🚨 Important Legal Notice

**GhostNV** is an **independent open-source project** developed by **CK Technology** that builds upon open-source GPU driver components. This document outlines our approach to legal compliance and trademark respect.

**CK Technology's Mission**: We believe NVIDIA on Linux deserves fully open-source solutions, and we're committed to providing this technology freely to the community.

---

## 📜 License Structure

### **MIT License (Primary)**
- **GhostNV-specific code**: All original code and innovations
- **Zig-NVIDIA driver**: Pure Zig implementation components
- **Build system**: Zig build scripts and tooling
- **Documentation**: Project-specific documentation

### **GPL v2 (Kernel Components)**
- **Kernel modules**: When compiled as Linux kernel modules
- **Derived components**: Code directly derived from GPL sources
- **Kernel interface**: Linux kernel integration components

### **Upstream License Compliance**
- **NVIDIA open sources**: Maintains original MIT/GPL v2 dual licensing
- **Attribution preserved**: Original copyright notices maintained
- **License compatibility**: Ensures proper license inheritance

---

## 🏷️ Trademark & Naming Guidelines

### **Avoid NVIDIA Trademarks**

**❌ DO NOT USE:**
- "NVIDIA" in project names, binaries, or primary branding
- "GeForce", "RTX", "GTX", "Quadro", "Tesla" in primary naming
- "CUDA" for non-compatible implementations
- NVIDIA logos, trade dress, or branding elements

**✅ SAFE USAGE:**
- Technical compatibility references: "compatible with NVIDIA hardware"
- Factual descriptions: "works with RTX graphics cards"
- Hardware identification: "supports GeForce and Quadro series"
- Clear non-affiliation statements

### **GhostNV Branding Strategy**

**Primary Names:**
- **GhostNV**: Main project name (original, non-infringing)
- **Zig-NVIDIA**: Technical component (descriptive, not trademark use)
- **ghostnv-driver**: Package name (functional description)

**Safe Descriptors:**
- "Open GPU driver for Linux"
- "Compatible with modern graphics hardware"
- "High-performance GPU driver framework"
- "Linux graphics driver with Zig implementation"

---

## 🔍 Component-Specific Compliance

### **Source Code Headers**

```zig
// GhostNV: Open GPU Driver Framework
// Copyright (c) 2025 CK Technology & GhostNV Contributors
// Licensed under MIT License (see LICENSE file)
//
// This file contains original implementations and may incorporate
// concepts from open-source GPU driver projects.
//
// CK Technology - Advancing open-source GPU drivers for Linux
```

### **Binary Naming**
- **Driver module**: `ghostnv.ko` (not `nvidia.ko`)
- **CLI tool**: `ghostnv` (not `nvidia-*`)
- **Library**: `libghost-gpu.so` (generic naming)
- **Device nodes**: `/dev/ghostnv*` (custom namespace)

### **Package Names**
- **AUR**: `ghostnv-dkms`, `ghostnv-utils`
- **Debian**: `ghostnv-driver`, `ghostnv-tools`
- **RPM**: `ghostnv-kmod`, `ghostnv-userspace`

---

## 🛡️ SDK & API Compliance

### **NVENC Integration**
```zig
// Use NVIDIA SDK through published APIs only
// No reverse engineering or proprietary code inclusion
// Respect SDK license terms and attribution requirements
```

### **CUDA Compatibility**
- **Interface only**: Standard CUDA driver API compatibility
- **No proprietary code**: Clean-room implementation
- **Clear labeling**: "CUDA-compatible" not "CUDA driver"

### **RTX Voice Features**
- **SDK-based**: Use official NVIDIA Audio Effects SDK
- **Proper attribution**: Credit NVIDIA for AI models
- **License compliance**: Respect SDK terms of use

---

## 📋 Distribution Guidelines

### **Package Descriptions**

**✅ Compliant:**
```
GhostNV - Open-source GPU driver framework
High-performance Linux driver for modern graphics hardware
Compatible with RTX, GTX, and Quadro graphics cards
```

**❌ Problematic:**
```
NVIDIA Driver Alternative
RTX Driver for Linux  
NVIDIA Open Driver Fork
```

### **Repository README**
- **Clear attribution**: Credit upstream NVIDIA open-source project
- **Non-affiliation statement**: "Not affiliated with NVIDIA Corporation"
- **Independent project**: Emphasize community-driven development

### **Documentation Standards**
- **Factual references**: Hardware compatibility information
- **Technical accuracy**: Correct API and feature descriptions
- **Clear sourcing**: Attribute information to appropriate sources

---

## 🔗 Upstream Relationship

### **NVIDIA Open GPU Kernel Modules**
- **Source**: https://github.com/NVIDIA/open-gpu-kernel-modules
- **License**: MIT/GPL v2 dual license
- **Attribution**: Maintained in all derivative works
- **Contribution**: Consider upstream contributions where appropriate

### **Collaboration Approach**
- **Respectful engagement**: Professional interaction with upstream
- **Technical focus**: Contributions based on technical merit
- **License compliance**: Strict adherence to open-source licenses
- **Innovation**: Original contributions that benefit the ecosystem

---

## ⚖️ Legal Risk Mitigation

### **Trademark Defense**
1. **Distinctive naming**: GhostNV creates unique brand identity
2. **Functional descriptions**: Use technical terms, not trademark terms
3. **Clear disclaimers**: Non-affiliation statements in prominent locations
4. **Fair use**: Technical compatibility references only

### **Copyright Compliance**
1. **Clean derivation**: Track sources of all incorporated code
2. **License preservation**: Maintain original license headers
3. **Attribution accuracy**: Credit original authors appropriately
4. **Documentation**: Clear changelog of modifications

### **Patent Considerations**
1. **Open-source focus**: Use only publicly available APIs
2. **Independent development**: Clean-room implementations
3. **Prior art research**: Document independent invention
4. **Community review**: Public development process

---

## 📞 Contact & Compliance

### **Legal Questions**
For legal or licensing questions regarding GhostNV:
- Contact CK Technology through GitHub issues
- Review this document and LICENSE file
- Consult with qualified legal counsel
- Respect all intellectual property rights

### **Trademark Concerns**
If you believe GhostNV infringes on your trademarks:
- Contact CK Technology through GitHub issues  
- Provide specific details of claimed infringement
- We will respond promptly and work toward resolution

### **CK Technology Contact**
- **Mission**: Advancing open-source GPU drivers for Linux
- **Commitment**: Fully free and open-source solutions
- **Community**: Building better NVIDIA support for Linux users

### **Contribution Guidelines**
Contributors must ensure:
- **Original work**: Only submit code you have rights to contribute
- **License compatibility**: Ensure contributions are MIT/GPL compatible
- **Clean development**: No incorporation of proprietary code
- **Attribution**: Properly credit sources and influences

---

## 🎯 Compliance Checklist

### **Before Release**
- [ ] Review all source files for proper headers
- [ ] Verify no proprietary code inclusion
- [ ] Check binary names for trademark conflicts
- [ ] Validate package descriptions and metadata
- [ ] Ensure proper upstream attribution
- [ ] Confirm license file accuracy
- [ ] Test legal disclaimer visibility

### **Ongoing Maintenance**
- [ ] Monitor upstream license changes
- [ ] Update attributions as needed
- [ ] Review contributor submissions for compliance
- [ ] Maintain clean development practices
- [ ] Document all third-party integrations
- [ ] Regular legal compliance reviews

---

**This document reflects CK Technology's commitment to developing GhostNV as a legally compliant, community-driven open-source project that respects intellectual property rights while advancing open GPU driver technology. We believe NVIDIA on Linux deserves fully open-source solutions, and we're proud to contribute this technology freely to the community.**