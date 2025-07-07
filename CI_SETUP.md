# GhostNV CI/CD Setup Guide

## Overview

GhostNV uses a comprehensive CI/CD pipeline that tests across multiple NVIDIA GPU generations to ensure compatibility and performance across the ecosystem.

## Hardware Test Matrix

### Target Hardware
- **RTX 4090** (Ada Lovelace) - Primary development target
- **RTX 3070** (Ampere) - Hybrid driver testing  
- **RTX 2060** (Turing) - Legacy compatibility

### Architecture Strategy
```zig
pub const DriverSelection = enum {
    pure_zig,    // RTX 40 Series - maximum performance
    hybrid_c,    // RTX 30 Series - compatibility + performance  
    legacy_c,    // RTX 20 Series - stability first
};
```

## CI Pipeline Structure

### Main Job: `build`
- **Platform**: `self-hosted` (requires actual NVIDIA hardware)
- **Triggers**: Push to `main`, Pull Requests
- **Daily**: Automated regression testing at 2 AM UTC

### Test Phases

#### 1. Build Validation
```bash
cd zig-nvidia
zig build                    # Standard build
zig build test              # Unit tests
```

#### 2. Hardware Detection
```bash
lspci | grep -i nvidia      # PCI enumeration
nvidia-smi --query-gpu=...  # GPU information
zig run tools/gpu-test.zig  # Hardware validation
```

#### 3. Architecture-Specific Testing

**RTX 4090 (Ada Lovelace)**:
```bash
zig run tools/test-rtx.zig -- --generation=ada
zig run tools/ghostvibrance.zig -- --test-mode --gpu=0
```

**RTX 3070 (Ampere)**:
```bash
zig run tools/test-rtx.zig -- --generation=ampere
zig run tools/test-nvenc.zig -- --codec=h264 --resolution=1080p
zig run benchmarks/memory_bandwidth.zig -- --target=ampere
```

**RTX 2060 (Turing)**:
```bash
zig run tools/test-rtx.zig -- --generation=turing
# Legacy compatibility validation
```

#### 4. Performance Baseline
```bash
zig run benchmarks/main.zig -- --quick
```

#### 5. Integration Smoke Tests
- PCI enumeration validation
- Memory management testing
- Command submission pipeline
- Interrupt handling verification

## Test Tools

### Core Testing Tools

#### `tools/gpu-test.zig`
Hardware detection and validation
```bash
zig run tools/gpu-test.zig
```

#### `tools/test-rtx.zig`
RTX features testing (RT cores, Tensor cores, etc.)
```bash
zig run tools/test-rtx.zig -- --generation=ada
```

#### `tools/test-nvenc.zig`
Video encoding capabilities
```bash
zig run tools/test-nvenc.zig -- --codec=h264 --resolution=1080p
```

#### `benchmarks/main.zig`
Comprehensive performance benchmarking
```bash
zig run benchmarks/main.zig -- --quick  # CI mode
zig run benchmarks/main.zig             # Full benchmarks
```

#### `benchmarks/memory_bandwidth.zig`
Memory subsystem performance
```bash
zig run benchmarks/memory_bandwidth.zig -- --target=ampere
```

### Performance Monitoring

The CI tracks:
- **Memory bandwidth** (GB/s)
- **Compute performance** (GFLOPS)
- **Graphics throughput** (MP/s)
- **Overall performance score**

Results are saved to `benchmarks/results/` with timestamps for regression analysis.

## Build Configuration

### Standard Build
```bash
cd zig-nvidia
zig build
```

### Architecture-Specific Builds
```bash
# RTX 40 Series (Ada Lovelace)
zig build -Dgpu_generation=ada -Doptimize=ReleaseFast

# RTX 30 Series (Ampere) 
zig build -Dgpu_generation=ampere -Dhybrid_mode=true

# RTX 20 Series (Turing)
zig build -Dgpu_generation=turing -Dlegacy_mode=true
```

### Gaming Optimizations
```bash
zig build -Dgaming_optimized=true
```

## Self-Hosted Runner Setup

### Requirements
1. **NVIDIA GPU** (RTX 2060 or newer)
2. **NVIDIA drivers** (545.xx or newer)
3. **Zig compiler** (auto-installed from latest dev)
4. **Linux environment** with PCI access

### Runner Configuration
```yaml
runs-on: self-hosted
labels: [rtx-4090, ada-lovelace]  # GPU-specific labels
```

### Environment Setup
The CI automatically installs the latest Zig development build:
```bash
JSON=$(curl -sL https://ziglang.org/download/index.json)
ZIG_URL=$(echo "$JSON" | jq -r '.master."x86_64-linux".tarball')
# ... installation process
```

## Trigger Patterns

### Manual Testing
Include these strings in commit messages to trigger specific tests:

- `[test-4090]` - RTX 4090 specific tests
- `[test-3070]` - RTX 3070 specific tests  
- `[test-2060]` - RTX 2060 specific tests
- `[test-all]` - Full compatibility matrix

### Automated Testing
- **Every push** to `main` - Basic validation
- **Every PR** - Full test suite
- **Daily at 2 AM UTC** - Regression testing

## Test Results

### Success Criteria
- ✅ Build completion without errors
- ✅ All unit tests passing
- ✅ Hardware detection successful
- ✅ Architecture-specific features working
- ✅ Performance within acceptable range

### Artifacts
- Test results with hardware info
- Performance benchmarks
- Build artifacts for each architecture
- Detailed test report (markdown)

### Performance Baselines

**RTX 4090 (Ada)**: 850+ overall score
**RTX 3070 (Ampere)**: 650+ overall score  
**RTX 2060 (Turing)**: 450+ overall score

Performance below 85% of baseline triggers investigation.

## Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
lspci | grep -i nvidia  # Check PCI visibility
sudo dmesg | grep -i nvidia  # Check kernel logs
```

#### Build Failures
```bash
zig build --verbose  # Detailed build output
zig version         # Verify Zig installation
```

#### Permission Issues
```bash
sudo usermod -a -G video $(whoami)  # Add user to video group
# Logout/login required
```

### Debug Mode
Enable verbose logging:
```bash
export GHOSTNV_DEBUG=1
zig run tools/gpu-test.zig
```

## Integration with GhostShell

The CI validates that GhostShell can properly integrate with the driver:

```zig
// Example integration test
const ghostnv = @import("ghostnv");
const driver = try ghostnv.getDriver(0);
try driver.setDigitalVibrance(50);
try driver.enableVRR(48, 165);
```

## Adding New Tests

### Hardware Test
1. Create test in `tools/test-[feature].zig`
2. Add to CI workflow under appropriate GPU section
3. Include in build.zig tools list

### Performance Benchmark
1. Add function to `benchmarks/main.zig`
2. Include in benchmark suite
3. Set performance baseline

### Unit Test
1. Add test to relevant source file
2. Include in `build.zig` test list
3. Verify CI picks up new test

## Reference Implementation

The `nvidia-open/` submodule provides reference implementations for complex hardware interactions while our pure Zig driver provides the performance and safety benefits.

---

**Result**: A comprehensive CI/CD pipeline that ensures GhostNV works flawlessly across NVIDIA GPU generations with maximum performance and reliability. 🚀