const std = @import("std");
const print = std.debug.print;
const hal = @import("../src/hal/pci.zig");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    print("=== GhostNV GPU Hardware Test ===\n");

    // Test PCI enumeration
    var enumerator = hal.PciEnumerator.init(allocator);
    defer enumerator.deinit();

    try enumerator.scanPciDevices();
    print("Found {} PCI devices\n", .{enumerator.devices.items.len});

    // Find NVIDIA GPUs
    const nvidia_gpus = try enumerator.findNvidiaGpus();
    defer allocator.free(nvidia_gpus);

    if (nvidia_gpus.len == 0) {
        print("❌ No NVIDIA GPUs detected\n");
        return;
    }

    print("✅ Found {} NVIDIA GPU(s)\n", .{nvidia_gpus.len});

    for (nvidia_gpus, 0..) |gpu, i| {
        const name = try gpu.getDeviceName(allocator);
        defer allocator.free(name);

        print("\nGPU {}: {s}\n", .{ i, name });
        print("  Architecture: {s}\n", .{gpu.architecture.toString()});
        print("  Memory: {} MB\n", .{gpu.memory_size / (1024 * 1024)});
        print("  Compute: {}.{}\n", .{ gpu.compute_capability.major, gpu.compute_capability.minor });
        print("  Bus: {s}\n", .{gpu.getBusAddress()});

        // Test memory detection
        if (gpu.memory_size > 0) {
            print("  ✅ Memory detection: PASSED\n");
        } else {
            print("  ❌ Memory detection: FAILED\n");
        }

        // Test architecture detection
        if (gpu.architecture != .unknown) {
            print("  ✅ Architecture detection: PASSED\n");
        } else {
            print("  ❌ Architecture detection: FAILED\n");
        }
    }

    print("\n=== Test Summary ===\n");
    print("PCI Enumeration: ✅ PASSED\n");
    print("GPU Detection: ✅ PASSED\n");
    print("Hardware Test: ✅ COMPLETED\n");
}