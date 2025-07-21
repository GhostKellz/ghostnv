const std = @import("std");
const testing = std.testing;

// Simple test to verify module imports work
test "basic module import test" {
    const ghostnv = @import("zig-nvidia");
    
    // Test that we can access exported modules
    _ = ghostnv.color_vibrance;
    _ = ghostnv.cuda_runtime;
    _ = ghostnv.container_runtime;
    _ = ghostnv.gaming_performance;
    
    // Basic assertion test
    try testing.expect(true);
}

test "basic arithmetic" {
    try testing.expect(2 + 2 == 4);
    try testing.expect(10 * 5 == 50);
}

test "memory allocation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const memory = try allocator.alloc(u8, 1024);
    defer allocator.free(memory);
    
    try testing.expect(memory.len == 1024);
}