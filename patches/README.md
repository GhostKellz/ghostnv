# GhostNV Patch System

This directory contains modular patches for the NVIDIA open driver system, organized by version and feature.

## Directory Structure

```
patches/
├── v575/           # Patches for NVIDIA 575.x series
├── v580/           # Patches for NVIDIA 580.x series  
├── common/         # Cross-version compatibility patches
├── performance/    # Performance optimization patches
├── debug/          # Debug and development patches
└── audio/          # RTX Voice/Audio enhancement patches
```

## Patch Format

Each patch directory contains:
- `*.patch` - Standard unified diff patches
- `apply.zig` - Zig script for applying patches with validation
- `metadata.json` - Patch metadata (version, deps, checksums)

## Usage

Patches are applied automatically via the main `build.zig` orchestrator based on detected NVIDIA version and selected features.