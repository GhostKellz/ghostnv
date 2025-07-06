const std = @import("std");
const print = std.debug.print;

pub const MemoryError = error{
    AllocationFailed,
    InvalidAddress,
    MappingFailed,
    OutOfMemory,
    PermissionDenied,
};

pub const MemoryType = enum {
    System,     // System RAM
    Device,     // Device memory (VRAM)
    Coherent,   // DMA coherent memory
    Streaming,  // DMA streaming memory
};

pub const MemoryRegion = struct {
    physical_address: usize,
    virtual_address: ?usize,
    size: usize,
    memory_type: MemoryType,
    coherent: bool,
    
    pub fn init(phys_addr: usize, size: usize, mem_type: MemoryType) MemoryRegion {
        return MemoryRegion{
            .physical_address = phys_addr,
            .virtual_address = null,
            .size = size,
            .memory_type = mem_type,
            .coherent = mem_type == .Coherent,
        };
    }
    
    pub fn map_virtual(self: *MemoryRegion) !void {
        // In real kernel module, this would use ioremap or similar
        // For now, simulate mapping
        self.virtual_address = self.physical_address;
        print("nvzig: Mapped memory region 0x{X} -> 0x{X} (size: 0x{X})\n",
              .{self.physical_address, self.virtual_address.?, self.size});
    }
    
    pub fn unmap_virtual(self: *MemoryRegion) void {
        if (self.virtual_address) |_| {
            // In real kernel module, this would use iounmap
            print("nvzig: Unmapped memory region 0x{X}\n", .{self.physical_address});
            self.virtual_address = null;
        }
    }
};

pub const DmaBuffer = struct {
    allocator: std.mem.Allocator,
    size: usize,
    physical_address: usize,
    virtual_address: ?usize,
    coherent: bool,
    
    pub fn init(allocator: std.mem.Allocator, size: usize, coherent: bool) !DmaBuffer {
        // In real kernel module, use dma_alloc_coherent or dma_alloc_attrs
        const buffer = DmaBuffer{
            .allocator = allocator,
            .size = size,
            .physical_address = 0x1000000, // Simulate physical address
            .virtual_address = null,
            .coherent = coherent,
        };
        
        print("nvzig: Allocated DMA buffer: size=0x{X}, coherent={}\n", .{size, coherent});
        return buffer;
    }
    
    pub fn deinit(self: *DmaBuffer) void {
        if (self.virtual_address) |_| {
            self.unmap();
        }
        
        // In real kernel module, use dma_free_coherent or dma_free_attrs
        print("nvzig: Freed DMA buffer: size=0x{X}\n", .{self.size});
    }
    
    pub fn map(self: *DmaBuffer) !void {
        if (self.virtual_address != null) return;
        
        // In real kernel module, this would map the DMA buffer
        self.virtual_address = self.physical_address;
        print("nvzig: Mapped DMA buffer 0x{X} -> 0x{X}\n",
              .{self.physical_address, self.virtual_address.?});
    }
    
    pub fn unmap(self: *DmaBuffer) void {
        if (self.virtual_address) |_| {
            print("nvzig: Unmapped DMA buffer 0x{X}\n", .{self.physical_address});
            self.virtual_address = null;
        }
    }
    
    pub fn sync_for_cpu(self: *DmaBuffer) void {
        if (!self.coherent) {
            // In real kernel module, use dma_sync_single_for_cpu
            print("nvzig: Syncing DMA buffer for CPU access\n");
        }
    }
    
    pub fn sync_for_device(self: *DmaBuffer) void {
        if (!self.coherent) {
            // In real kernel module, use dma_sync_single_for_device
            print("nvzig: Syncing DMA buffer for device access\n");
        }
    }
};

pub const MemoryManager = struct {
    allocator: std.mem.Allocator,
    regions: std.ArrayList(MemoryRegion),
    dma_buffers: std.ArrayList(DmaBuffer),
    total_allocated: usize,
    
    pub fn init(allocator: std.mem.Allocator) MemoryManager {
        return MemoryManager{
            .allocator = allocator,
            .regions = std.ArrayList(MemoryRegion).init(allocator),
            .dma_buffers = std.ArrayList(DmaBuffer).init(allocator),
            .total_allocated = 0,
        };
    }
    
    pub fn deinit(self: *MemoryManager) void {
        // Clean up all DMA buffers
        for (self.dma_buffers.items) |*buffer| {
            buffer.deinit();
        }
        self.dma_buffers.deinit();
        
        // Clean up all memory regions
        for (self.regions.items) |*region| {
            region.unmap_virtual();
        }
        self.regions.deinit();
        
        print("nvzig: Memory manager cleaned up (total allocated: 0x{X})\n", .{self.total_allocated});
    }
    
    pub fn allocate_dma_buffer(self: *MemoryManager, size: usize, coherent: bool) !*DmaBuffer {
        var buffer = try DmaBuffer.init(self.allocator, size, coherent);
        try self.dma_buffers.append(buffer);
        self.total_allocated += size;
        
        return &self.dma_buffers.items[self.dma_buffers.items.len - 1];
    }
    
    pub fn free_dma_buffer(self: *MemoryManager, buffer: *DmaBuffer) void {
        for (self.dma_buffers.items, 0..) |*item, i| {
            if (item == buffer) {
                self.total_allocated -= buffer.size;
                buffer.deinit();
                _ = self.dma_buffers.swapRemove(i);
                break;
            }
        }
    }
    
    pub fn map_device_memory(self: *MemoryManager, phys_addr: usize, size: usize) !*MemoryRegion {
        var region = MemoryRegion.init(phys_addr, size, .Device);
        try region.map_virtual();
        try self.regions.append(region);
        
        return &self.regions.items[self.regions.items.len - 1];
    }
    
    pub fn unmap_device_memory(self: *MemoryManager, region: *MemoryRegion) void {
        for (self.regions.items, 0..) |*item, i| {
            if (item == region) {
                region.unmap_virtual();
                _ = self.regions.swapRemove(i);
                break;
            }
        }
    }
};

pub const DeviceMemoryManager = struct {
    allocator: std.mem.Allocator,
    memory_manager: MemoryManager,
    vram_base: usize,
    vram_size: usize,
    vram_used: usize,
    
    pub fn init(allocator: std.mem.Allocator) DeviceMemoryManager {
        return DeviceMemoryManager{
            .allocator = allocator,
            .memory_manager = MemoryManager.init(allocator),
            .vram_base = 0,
            .vram_size = 0,
            .vram_used = 0,
        };
    }
    
    pub fn deinit(self: *DeviceMemoryManager) void {
        self.memory_manager.deinit();
    }
    
    pub fn setup_vram(self: *DeviceMemoryManager, base: usize, size: usize) !void {
        self.vram_base = base;
        self.vram_size = size;
        self.vram_used = 0;
        
        print("nvzig: VRAM setup - Base: 0x{X}, Size: 0x{X} ({} MB)\n",
              .{base, size, size / (1024 * 1024)});
    }
    
    pub fn allocate_vram(self: *DeviceMemoryManager, size: usize, alignment: usize) !usize {
        // Simple bump allocator for VRAM
        const aligned_used = std.mem.alignForward(usize, self.vram_used, alignment);
        
        if (aligned_used + size > self.vram_size) {
            return MemoryError.OutOfMemory;
        }
        
        const offset = aligned_used;
        self.vram_used = aligned_used + size;
        
        print("nvzig: Allocated VRAM: offset=0x{X}, size=0x{X}\n", .{offset, size});
        return self.vram_base + offset;
    }
    
    pub fn free_vram(self: *DeviceMemoryManager, address: usize, size: usize) void {
        _ = address; _ = size;
        // TODO: Implement proper VRAM free list
        print("nvzig: VRAM free (not implemented yet)\n");
    }
    
    pub fn get_vram_usage(self: *DeviceMemoryManager) struct { used: usize, total: usize } {
        return .{ .used = self.vram_used, .total = self.vram_size };
    }
};

// Page table management for GPU virtual memory
pub const PageTable = struct {
    allocator: std.mem.Allocator,
    base_address: usize,
    size: usize,
    page_size: usize,
    
    pub fn init(allocator: std.mem.Allocator, size: usize, page_size: usize) !PageTable {
        const base = try allocator.alignedAlloc(u8, page_size, size);
        @memset(base, 0);
        
        return PageTable{
            .allocator = allocator,
            .base_address = @intFromPtr(base.ptr),
            .size = size,
            .page_size = page_size,
        };
    }
    
    pub fn deinit(self: *PageTable) void {
        const ptr: [*]u8 = @ptrFromInt(self.base_address);
        self.allocator.free(ptr[0..self.size]);
    }
    
    pub fn map_page(self: *PageTable, virtual_addr: usize, physical_addr: usize) !void {
        const page_index = virtual_addr / self.page_size;
        print("nvzig: Mapping page {}: 0x{X} -> 0x{X}\n", .{page_index, virtual_addr, physical_addr});
        // TODO: Implement actual page table entry setting
    }
    
    pub fn unmap_page(self: *PageTable, virtual_addr: usize) void {
        const page_index = virtual_addr / self.page_size;
        print("nvzig: Unmapping page {}: 0x{X}\n", .{page_index, virtual_addr});
        // TODO: Implement actual page table entry clearing
    }
};

test "memory manager" {
    const allocator = std.testing.allocator;
    var mem_mgr = MemoryManager.init(allocator);
    defer mem_mgr.deinit();
    
    // Test DMA buffer allocation
    const buffer = try mem_mgr.allocate_dma_buffer(4096, true);
    try std.testing.expect(buffer.size == 4096);
    try std.testing.expect(buffer.coherent == true);
    
    // Test buffer mapping
    try buffer.map();
    try std.testing.expect(buffer.virtual_address != null);
    
    // Cleanup is handled by mem_mgr.deinit()
}

test "device memory manager" {
    const allocator = std.testing.allocator;
    var dev_mem = DeviceMemoryManager.init(allocator);
    defer dev_mem.deinit();
    
    // Setup VRAM
    try dev_mem.setup_vram(0x80000000, 1024 * 1024 * 1024); // 1GB VRAM
    
    // Test VRAM allocation
    const addr1 = try dev_mem.allocate_vram(4096, 4096);
    const addr2 = try dev_mem.allocate_vram(8192, 4096);
    
    try std.testing.expect(addr1 == 0x80000000);
    try std.testing.expect(addr2 == 0x80001000);
    
    const usage = dev_mem.get_vram_usage();
    try std.testing.expect(usage.used >= 4096 + 8192);
}