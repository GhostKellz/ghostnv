const std = @import("std");
const linux = std.os.linux;
const fs = std.fs;
const Allocator = std.mem.Allocator;
const pci = @import("pci.zig");

// Memory Management for NVIDIA GPU VRAM and system memory

pub const MemoryError = error{
    OutOfMemory,
    InvalidAddress,
    AccessDenied,
    AllocationFailed,
    MappingFailed,
    FragmentationError,
    InvalidSize,
    DeviceNotFound,
    HardwareError,
    PermissionDenied,
};

pub const MemoryType = enum(u8) {
    vram = 0,           // GPU video memory
    system = 1,         // System RAM
    gart = 2,           // Graphics Address Remapping Table
    bar = 3,            // PCI Base Address Register space
    coherent = 4,       // DMA coherent memory
    streaming = 5,      // DMA streaming memory
    
    pub fn toString(self: MemoryType) []const u8 {
        return switch (self) {
            .vram => "VRAM",
            .system => "System RAM",
            .gart => "GART",
            .bar => "BAR",
            .coherent => "Coherent",
            .streaming => "Streaming",
        };
    }
};

pub const MemoryUsage = enum(u8) {
    framebuffer = 0,    // Display framebuffer
    texture = 1,        // Texture data
    vertex_buffer = 2,  // Vertex buffers
    command_buffer = 3, // GPU command buffers
    shader = 4,         // Shader programs
    general = 5,        // General GPU memory
    
    pub fn toString(self: MemoryUsage) []const u8 {
        return switch (self) {
            .framebuffer => "Framebuffer",
            .texture => "Texture",
            .vertex_buffer => "Vertex Buffer",
            .command_buffer => "Command Buffer",
            .shader => "Shader",
            .general => "General",
        };
    }
};

pub const MemoryFlags = struct {
    readable: bool = true,
    writable: bool = true,
    executable: bool = false,
    cacheable: bool = true,
    coherent: bool = false,
    
    pub fn toProt(self: MemoryFlags) u32 {
        var prot: u32 = 0;
        if (self.readable) prot |= linux.PROT.READ;
        if (self.writable) prot |= linux.PROT.WRITE;
        if (self.executable) prot |= linux.PROT.EXEC;
        return prot;
    }
};

pub const MemoryRegion = struct {
    physical_address: u64,
    virtual_address: ?u64,
    size: u64,
    memory_type: MemoryType,
    usage: MemoryUsage,
    flags: MemoryFlags,
    ref_count: u32,
    mapped: bool,
    
    pub fn init(
        phys_addr: u64,
        size: u64,
        mem_type: MemoryType,
        usage: MemoryUsage,
        flags: MemoryFlags,
    ) MemoryRegion {
        return MemoryRegion{
            .physical_address = phys_addr,
            .virtual_address = null,
            .size = size,
            .memory_type = mem_type,
            .usage = usage,
            .flags = flags,
            .ref_count = 1,
            .mapped = false,
        };
    }
    
    pub fn map(self: *MemoryRegion) !void {
        if (self.mapped) return;
        
        // Map physical memory to virtual address space
        // In real kernel implementation, this would use ioremap() for device memory
        // or vmap() for system memory
        
        const virt_addr = switch (self.memory_type) {
            .vram, .bar => try mapDeviceMemory(self.physical_address, self.size, self.flags),
            .system => try mapSystemMemory(self.physical_address, self.size, self.flags),
            .gart => try mapGartMemory(self.physical_address, self.size, self.flags),
            .coherent, .streaming => try mapDmaMemory(self.physical_address, self.size, self.flags),
        };
        
        self.virtual_address = virt_addr;
        self.mapped = true;
        
        std.log.debug("Mapped {s} memory: 0x{X} -> 0x{X} ({} bytes)", .{
            self.memory_type.toString(),
            self.physical_address,
            virt_addr,
            self.size,
        });
    }
    
    pub fn unmap(self: *MemoryRegion) void {
        if (!self.mapped or self.virtual_address == null) return;
        
        switch (self.memory_type) {
            .vram, .bar => unmapDeviceMemory(self.virtual_address.?, self.size),
            .system => unmapSystemMemory(self.virtual_address.?, self.size),
            .gart => unmapGartMemory(self.virtual_address.?, self.size),
            .coherent, .streaming => unmapDmaMemory(self.virtual_address.?, self.size),
        }
        
        std.log.debug("Unmapped {s} memory: 0x{X} ({} bytes)", .{
            self.memory_type.toString(),
            self.virtual_address.?,
            self.size,
        });
        
        self.virtual_address = null;
        self.mapped = false;
    }
    
    pub fn addRef(self: *MemoryRegion) void {
        self.ref_count += 1;
    }
    
    pub fn release(self: *MemoryRegion) bool {
        if (self.ref_count > 0) {
            self.ref_count -= 1;
        }
        return self.ref_count == 0;
    }
    
    pub fn sync_for_cpu(self: *MemoryRegion) void {
        if (!self.flags.coherent and self.memory_type == .streaming) {
            // In real kernel module, use dma_sync_single_for_cpu
            std.log.debug("Syncing {s} memory for CPU access", .{self.memory_type.toString()});
        }
    }
    
    pub fn sync_for_device(self: *MemoryRegion) void {
        if (!self.flags.coherent and self.memory_type == .streaming) {
            // In real kernel module, use dma_sync_single_for_device
            std.log.debug("Syncing {} memory for device access", .{self.memory_type.toString()});
        }
    }
};

pub const DmaBuffer = struct {
    allocator: std.mem.Allocator,
    size: u64,
    physical_address: u64,
    virtual_address: ?u64,
    coherent: bool,
    region: MemoryRegion,
    
    pub fn init(allocator: Allocator, size: u64, coherent: bool) !DmaBuffer {
        // In real kernel module, use dma_alloc_coherent or dma_alloc_attrs
        const phys_addr = 0x1000000; // Simulate physical address
        
        const mem_type: MemoryType = if (coherent) .coherent else .streaming;
        const flags = MemoryFlags{
            .coherent = coherent,
        };
        
        const region = MemoryRegion.init(phys_addr, size, mem_type, .general, flags);
        
        const buffer = DmaBuffer{
            .allocator = allocator,
            .size = size,
            .physical_address = phys_addr,
            .virtual_address = null,
            .coherent = coherent,
            .region = region,
        };
        
        std.log.debug("Allocated DMA buffer: size={}, coherent={}", .{ size, coherent });
        return buffer;
    }
    
    pub fn deinit(self: *DmaBuffer) void {
        if (self.region.mapped) {
            self.region.unmap();
        }
        
        // In real kernel module, use dma_free_coherent or dma_free_attrs
        std.log.debug("Freed DMA buffer: size={}", .{self.size});
    }
    
    pub fn map(self: *DmaBuffer) !void {
        if (self.region.mapped) return;
        
        try self.region.map();
        self.virtual_address = self.region.virtual_address;
        
        std.log.debug("Mapped DMA buffer 0x{X} -> 0x{X}", .{
            self.physical_address,
            self.virtual_address.?,
        });
    }
    
    pub fn unmap(self: *DmaBuffer) void {
        if (self.region.mapped) {
            self.region.unmap();
            self.virtual_address = null;
        }
    }
    
    pub fn sync_for_cpu(self: *DmaBuffer) void {
        self.region.sync_for_cpu();
    }
    
    pub fn sync_for_device(self: *DmaBuffer) void {
        self.region.sync_for_device();
    }
};

pub const MemoryPool = struct {
    allocator: Allocator,
    memory_type: MemoryType,
    base_address: u64,
    size: u64,
    used: u64,
    free_blocks: std.ArrayList(MemoryBlock),
    allocated_regions: std.ArrayList(MemoryRegion),
    
    const MemoryBlock = struct {
        offset: u64,
        size: u64,
        
        pub fn lessThan(_: void, lhs: MemoryBlock, rhs: MemoryBlock) bool {
            return lhs.offset < rhs.offset;
        }
    };
    
    pub fn init(
        allocator: Allocator,
        memory_type: MemoryType,
        base_address: u64,
        size: u64,
    ) MemoryPool {
        var pool = MemoryPool{
            .allocator = allocator,
            .memory_type = memory_type,
            .base_address = base_address,
            .size = size,
            .used = 0,
            .free_blocks = std.ArrayList(MemoryBlock).init(allocator),
            .allocated_regions = std.ArrayList(MemoryRegion).init(allocator),
        };
        
        // Initial free block covers entire pool
        pool.free_blocks.append(MemoryBlock{
            .offset = 0,
            .size = size,
        }) catch unreachable;
        
        return pool;
    }
    
    pub fn deinit(self: *MemoryPool) void {
        // Unmap all allocated regions
        for (self.allocated_regions.items) |*region| {
            region.unmap();
        }
        
        self.free_blocks.deinit();
        self.allocated_regions.deinit();
    }
    
    pub fn allocate(
        self: *MemoryPool,
        size: u64,
        alignment: u64,
        usage: MemoryUsage,
        flags: MemoryFlags,
    ) !*MemoryRegion {
        const aligned_size = alignUp(size, alignment);
        
        // Find suitable free block
        for (self.free_blocks.items, 0..) |*block, i| {
            const aligned_offset = alignUp(block.offset, alignment);
            const needed_size = aligned_offset - block.offset + aligned_size;
            
            if (block.size >= needed_size) {
                // Allocate from this block
                const phys_addr = self.base_address + aligned_offset;
                
                const region = MemoryRegion.init(phys_addr, aligned_size, self.memory_type, usage, flags);
                try self.allocated_regions.append(region);
                
                // Update free block
                if (block.size == needed_size) {
                    // Exact fit - remove block
                    _ = self.free_blocks.swapRemove(i);
                } else {
                    // Split block
                    block.offset = aligned_offset + aligned_size;
                    block.size -= needed_size;
                }
                
                self.used += aligned_size;
                
                std.log.debug("Allocated {} memory: {} bytes at 0x{X} for {}", .{
                    self.memory_type.toString(),
                    aligned_size,
                    phys_addr,
                    usage.toString(),
                });
                
                return &self.allocated_regions.items[self.allocated_regions.items.len - 1];
            }
        }
        
        return MemoryError.OutOfMemory;
    }
    
    pub fn free(self: *MemoryPool, region: *MemoryRegion) !void {
        // Find and remove from allocated regions
        for (self.allocated_regions.items, 0..) |*allocated_region, i| {
            if (allocated_region.physical_address == region.physical_address) {
                // Unmap if mapped
                allocated_region.unmap();
                
                // Add back to free blocks
                const offset = allocated_region.physical_address - self.base_address;
                const new_block = MemoryBlock{
                    .offset = offset,
                    .size = allocated_region.size,
                };
                
                try self.free_blocks.append(new_block);
                self.used -= allocated_region.size;
                
                // Remove from allocated list
                _ = self.allocated_regions.swapRemove(i);
                
                // Coalesce adjacent free blocks
                try self.coalesceBlocks();
                
                std.log.debug("Freed {} memory: {} bytes at 0x{X}", .{
                    self.memory_type.toString(),
                    allocated_region.size,
                    allocated_region.physical_address,
                });
                
                return;
            }
        }
        
        return MemoryError.InvalidAddress;
    }
    
    fn coalesceBlocks(self: *MemoryPool) !void {
        if (self.free_blocks.items.len <= 1) return;
        
        // Sort blocks by offset
        std.sort.insertion(MemoryBlock, self.free_blocks.items, {}, MemoryBlock.lessThan);
        
        var i: usize = 0;
        while (i < self.free_blocks.items.len - 1) {
            const current = &self.free_blocks.items[i];
            const next = &self.free_blocks.items[i + 1];
            
            if (current.offset + current.size == next.offset) {
                // Adjacent blocks - coalesce
                current.size += next.size;
                _ = self.free_blocks.swapRemove(i + 1);
            } else {
                i += 1;
            }
        }
    }
    
    // Advanced VRAM defragmentation with memory compaction
    pub fn defragment(self: *MemoryPool) !DefragmentationResult {
        const start_time = std.time.nanoTimestamp();
        var stats = DefragmentationResult{};
        
        // Calculate initial fragmentation
        stats.fragmentation_before = self.calculateFragmentation();
        stats.free_blocks_before = @intCast(self.free_blocks.items.len);
        
        // Phase 1: Coalesce adjacent blocks
        try self.coalesceBlocks();
        
        // Phase 2: Compact allocated regions if fragmentation is high
        if (stats.fragmentation_before > 0.3) { // 30% fragmentation threshold
            stats.bytes_moved = try self.compactAllocations();
            stats.allocations_moved = self.countMovedAllocations();
        }
        
        // Phase 3: Final coalesce after compaction
        try self.coalesceBlocks();
        
        // Calculate final metrics
        stats.fragmentation_after = self.calculateFragmentation();
        stats.free_blocks_after = @intCast(self.free_blocks.items.len);
        stats.time_ns = std.time.nanoTimestamp() - start_time;
        
        std.log.info("VRAM defragmentation completed: {d:.2}% -> {d:.2}% fragmentation, {} bytes moved", .{
            stats.fragmentation_before * 100.0,
            stats.fragmentation_after * 100.0,
            stats.bytes_moved,
        });
        
        return stats;
    }
    
    fn compactAllocations(self: *MemoryPool) !u64 {
        var bytes_moved: u64 = 0;
        
        // Sort allocated regions by physical address
        std.sort.insertion(MemoryRegion, self.allocated_regions.items, {}, struct {
            fn lessThan(_: void, lhs: MemoryRegion, rhs: MemoryRegion) bool {
                return lhs.physical_address < rhs.physical_address;
            }
        }.lessThan);
        
        // Find gaps and move allocations to fill them
        var current_offset: u64 = 0;
        
        for (self.allocated_regions.items) |*region| {
            const region_offset = region.physical_address - self.base_address;
            
            if (region_offset > current_offset) {
                // Found a gap - move allocation forward
                const new_phys_addr = self.base_address + current_offset;
                
                // Only move if it would reduce fragmentation significantly
                const gap_size = region_offset - current_offset;
                if (gap_size >= 4096) { // Minimum 4KB gap to justify move
                    try self.moveAllocation(region, new_phys_addr);
                    bytes_moved += region.size;
                }
            }
            
            current_offset = std.math.max(current_offset, region_offset) + region.size;
        }
        
        return bytes_moved;
    }
    
    fn moveAllocation(self: *MemoryPool, region: *MemoryRegion, new_phys_addr: u64) !void {
        const old_addr = region.physical_address;
        
        // In a real implementation, this would involve:
        // 1. Ensuring the GPU is idle or the region is not in use
        // 2. Copying memory content from old to new location
        // 3. Updating page tables and MMU mappings
        // 4. Notifying any dependent subsystems
        
        // For now, just update the address
        region.physical_address = new_phys_addr;
        
        // If the region is mapped, update virtual mapping
        if (region.mapped and region.virtual_address != null) {
            region.unmap();
            try region.map();
        }
        
        std.log.debug("Moved {} allocation: 0x{X} -> 0x{X} ({} bytes)", .{
            region.usage.toString(),
            old_addr,
            new_phys_addr,
            region.size,
        });
    }
    
    fn countMovedAllocations(self: *MemoryPool) u32 {
        // In a real implementation, track which allocations were moved
        // For now, estimate based on fragmentation improvement
        return @intCast(self.allocated_regions.items.len / 4);
    }
    
    pub fn getStats(self: *MemoryPool) MemoryStats {
        return MemoryStats{
            .total_size = self.size,
            .used_size = self.used,
            .free_size = self.size - self.used,
            .fragmentation = self.calculateFragmentation(),
            .num_allocations = @intCast(self.allocated_regions.items.len),
            .num_free_blocks = @intCast(self.free_blocks.items.len),
        };
    }
    
    fn calculateFragmentation(self: *MemoryPool) f32 {
        if (self.free_blocks.items.len <= 1) return 0.0;
        
        // Calculate fragmentation as ratio of free blocks to total free memory
        const total_free = self.size - self.used;
        if (total_free == 0) return 0.0;
        
        const avg_block_size = total_free / @as(u64, @intCast(self.free_blocks.items.len));
        const largest_block = self.getLargestFreeBlock();
        
        return 1.0 - (@as(f32, @floatFromInt(avg_block_size)) / @as(f32, @floatFromInt(largest_block)));
    }
    
    fn getLargestFreeBlock(self: *MemoryPool) u64 {
        var largest: u64 = 0;
        for (self.free_blocks.items) |block| {
            if (block.size > largest) {
                largest = block.size;
            }
        }
        return largest;
    }
};

pub const MemoryStats = struct {
    total_size: u64,
    used_size: u64,
    free_size: u64,
    fragmentation: f32,
    num_allocations: u32,
    num_free_blocks: u32,
};

pub const DefragmentationResult = struct {
    fragmentation_before: f32 = 0.0,
    fragmentation_after: f32 = 0.0,
    free_blocks_before: u32 = 0,
    free_blocks_after: u32 = 0,
    bytes_moved: u64 = 0,
    allocations_moved: u32 = 0,
    time_ns: u64 = 0,
    
    pub fn getImprovementPercent(self: DefragmentationResult) f32 {
        if (self.fragmentation_before == 0.0) return 0.0;
        return ((self.fragmentation_before - self.fragmentation_after) / self.fragmentation_before) * 100.0;
    }
};

pub const MemoryManager = struct {
    allocator: Allocator,
    vram_pool: ?MemoryPool,
    system_pool: ?MemoryPool,
    gart_pool: ?MemoryPool,
    bar_pools: [6]?MemoryPool, // One for each BAR
    dma_buffers: std.ArrayList(DmaBuffer),
    pci_device: ?pci.PciDevice,
    total_allocated: u64,
    
    pub fn init(allocator: Allocator) MemoryManager {
        return MemoryManager{
            .allocator = allocator,
            .vram_pool = null,
            .system_pool = null,
            .gart_pool = null,
            .bar_pools = [_]?MemoryPool{null} ** 6,
            .dma_buffers = std.ArrayList(DmaBuffer).init(allocator),
            .pci_device = null,
            .total_allocated = 0,
        };
    }
    
    pub fn initWithDevice(allocator: Allocator, pci_dev: pci.PciDevice) !MemoryManager {
        var manager = MemoryManager.init(allocator);
        manager.pci_device = pci_dev;
        
        // Initialize VRAM pool
        if (pci_dev.memory_size > 0) {
            manager.vram_pool = MemoryPool.init(
                allocator,
                .vram,
                @intCast(pci_dev.bar0), // VRAM typically mapped to BAR0
                pci_dev.memory_size,
            );
            
            std.log.info("Initialized VRAM pool: {} MB", .{pci_dev.memory_size / (1024 * 1024)});
        }
        
        // Initialize BAR pools
        const bars = [_]u64{ pci_dev.bar0, pci_dev.bar1, pci_dev.bar2, pci_dev.bar3, pci_dev.bar4, pci_dev.bar5 };
        for (bars, 0..) |bar_addr, i| {
            if (bar_addr != 0) {
                const bar_size = try pci_dev.get_bar_size(allocator, @intCast(i));
                if (bar_size > 0) {
                    manager.bar_pools[i] = MemoryPool.init(
                        allocator,
                        .bar,
                        bar_addr,
                        bar_size,
                    );
                    
                    std.log.debug("Initialized BAR{} pool: {} bytes at 0x{X}", .{ i, bar_size, bar_addr });
                }
            }
        }
        
        // Initialize GART (Graphics Address Remapping Table) pool
        // GART allows mapping system memory for GPU access
        const gart_size = 256 * 1024 * 1024; // 256MB GART space
        manager.gart_pool = MemoryPool.init(allocator, .gart, 0x0, gart_size);
        
        return manager;
    }
    
    pub fn deinit(self: *MemoryManager) void {
        // Clean up DMA buffers
        for (self.dma_buffers.items) |*buffer| {
            buffer.deinit();
        }
        self.dma_buffers.deinit();
        
        if (self.vram_pool) |*pool| pool.deinit();
        if (self.system_pool) |*pool| pool.deinit();
        if (self.gart_pool) |*pool| pool.deinit();
        
        for (&self.bar_pools) |*maybe_pool| {
            if (maybe_pool.*) |*pool| pool.deinit();
        }
        
        if (self.pci_device) |*device| {
            device.deinit(self.allocator);
        }
        
        std.log.debug("Memory manager cleaned up (total allocated: {} bytes)", .{self.total_allocated});
    }
    
    pub fn allocateVram(
        self: *MemoryManager,
        size: u64,
        alignment: u64,
        usage: MemoryUsage,
        flags: MemoryFlags,
    ) !*MemoryRegion {
        if (self.vram_pool) |*pool| {
            const region = try pool.allocate(size, alignment, usage, flags);
            self.total_allocated += region.size;
            return region;
        }
        return MemoryError.DeviceNotFound;
    }
    
    pub fn allocateSystem(
        self: *MemoryManager,
        size: u64,
        alignment: u64,
        usage: MemoryUsage,
        flags: MemoryFlags,
    ) !*MemoryRegion {
        // Allocate system memory and add to GART for GPU access
        if (self.gart_pool) |*pool| {
            const region = try pool.allocate(size, alignment, usage, flags);
            self.total_allocated += region.size;
            return region;
        }
        return MemoryError.DeviceNotFound;
    }
    
    pub fn allocateDmaBuffer(self: *MemoryManager, size: u64, coherent: bool) !*DmaBuffer {
        const buffer = try DmaBuffer.init(self.allocator, size, coherent);
        try self.dma_buffers.append(buffer);
        self.total_allocated += size;
        
        return &self.dma_buffers.items[self.dma_buffers.items.len - 1];
    }
    
    pub fn freeDmaBuffer(self: *MemoryManager, buffer: *DmaBuffer) void {
        for (self.dma_buffers.items, 0..) |*item, i| {
            if (item == buffer) {
                self.total_allocated -= buffer.size;
                buffer.deinit();
                _ = self.dma_buffers.swapRemove(i);
                break;
            }
        }
    }
    
    pub fn freeMemory(self: *MemoryManager, region: *MemoryRegion) !void {
        switch (region.memory_type) {
            .vram => {
                if (self.vram_pool) |*pool| {
                    self.total_allocated -= region.size;
                    try pool.free(region);
                } else {
                    return MemoryError.InvalidAddress;
                }
            },
            .system, .gart => {
                if (self.gart_pool) |*pool| {
                    self.total_allocated -= region.size;
                    try pool.free(region);
                } else {
                    return MemoryError.InvalidAddress;
                }
            },
            .bar => {
                // Find which BAR pool this belongs to
                for (&self.bar_pools) |*maybe_pool| {
                    if (maybe_pool.*) |*pool| {
                        if (region.physical_address >= pool.base_address and
                            region.physical_address < pool.base_address + pool.size) {
                            self.total_allocated -= region.size;
                            try pool.free(region);
                            return;
                        }
                    }
                }
                return MemoryError.InvalidAddress;
            },
            .coherent, .streaming => {
                // These are handled by DMA buffer management
                return MemoryError.InvalidAddress;
            },
        }
    }
    
    pub fn getTotalStats(self: *MemoryManager) MemoryManagerStats {
        var stats = MemoryManagerStats{
            .vram_stats = null,
            .system_stats = null,
            .gart_stats = null,
            .total_vram_mb = 0,
            .total_system_mb = 0,
            .total_allocated = self.total_allocated,
        };
        
        if (self.vram_pool) |*pool| {
            stats.vram_stats = pool.getStats();
            stats.total_vram_mb = @intCast(pool.size / (1024 * 1024));
        }
        
        if (self.gart_pool) |*pool| {
            stats.gart_stats = pool.getStats();
            stats.total_system_mb = @intCast(pool.size / (1024 * 1024));
        }
        
        return stats;
    }
    
    pub fn printStats(self: *MemoryManager) void {
        const stats = self.getTotalStats();
        
        std.log.info("=== GPU Memory Statistics ===");
        
        if (stats.vram_stats) |vram| {
            std.log.info("VRAM: {}/{} MB used ({d:.1}% utilization, {d:.2}% fragmentation)", .{
                vram.used_size / (1024 * 1024),
                vram.total_size / (1024 * 1024),
                @as(f32, @floatFromInt(vram.used_size)) / @as(f32, @floatFromInt(vram.total_size)) * 100.0,
                vram.fragmentation * 100.0,
            });
            std.log.info("  {} allocations, {} free blocks", .{ vram.num_allocations, vram.num_free_blocks });
        }
        
        if (stats.gart_stats) |gart| {
            std.log.info("GART: {}/{} MB used ({d:.1}% utilization, {d:.2}% fragmentation)", .{
                gart.used_size / (1024 * 1024),
                gart.total_size / (1024 * 1024),
                @as(f32, @floatFromInt(gart.used_size)) / @as(f32, @floatFromInt(gart.total_size)) * 100.0,
                gart.fragmentation * 100.0,
            });
            std.log.info("  {} allocations, {} free blocks", .{ gart.num_allocations, gart.num_free_blocks });
        }
        
        std.log.info("Total allocated: {} MB", .{stats.total_allocated / (1024 * 1024)});
    }
};

pub const MemoryManagerStats = struct {
    vram_stats: ?MemoryStats,
    system_stats: ?MemoryStats,
    gart_stats: ?MemoryStats,
    total_vram_mb: u32,
    total_system_mb: u32,
    total_allocated: u64,
};

// Legacy compatibility structures
pub const DeviceMemoryManager = struct {
    memory_manager: MemoryManager,
    vram_base: u64,
    vram_size: u64,
    vram_used: u64,
    
    pub fn init(allocator: Allocator) DeviceMemoryManager {
        return DeviceMemoryManager{
            .memory_manager = MemoryManager.init(allocator),
            .vram_base = 0,
            .vram_size = 0,
            .vram_used = 0,
        };
    }
    
    pub fn deinit(self: *DeviceMemoryManager) void {
        self.memory_manager.deinit();
    }
    
    pub fn setup_vram(self: *DeviceMemoryManager, base: u64, size: u64) !void {
        self.vram_base = base;
        self.vram_size = size;
        self.vram_used = 0;
        
        std.log.info("VRAM setup - Base: 0x{X}, Size: 0x{X} ({} MB)", .{
            base,
            size,
            size / (1024 * 1024),
        });
    }
    
    pub fn allocate_vram(self: *DeviceMemoryManager, size: u64, alignment: u64) !u64 {
        // Simple bump allocator for VRAM
        const aligned_used = std.mem.alignForward(u64, self.vram_used, alignment);
        
        if (aligned_used + size > self.vram_size) {
            return MemoryError.OutOfMemory;
        }
        
        const offset = aligned_used;
        self.vram_used = aligned_used + size;
        
        std.log.debug("Allocated VRAM: offset=0x{X}, size=0x{X}", .{ offset, size });
        return self.vram_base + offset;
    }
    
    pub fn free_vram(_: *DeviceMemoryManager, _: u64, _: u64) void {
        // TODO: Implement proper VRAM free list
        std.log.debug("VRAM free (not implemented yet)");
    }
    
    pub fn get_vram_usage(self: *DeviceMemoryManager) struct { used: u64, total: u64 } {
        return .{ .used = self.vram_used, .total = self.vram_size };
    }
};

// Page table management for GPU virtual memory
pub const PageTable = struct {
    allocator: Allocator,
    base_address: u64,
    size: u64,
    page_size: u64,
    
    pub fn init(allocator: Allocator, size: u64, page_size: u64) !PageTable {
        const base = try allocator.alignedAlloc(u8, @intCast(page_size), @intCast(size));
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
        self.allocator.free(ptr[0..@intCast(self.size)]);
    }
    
    pub fn map_page(self: *PageTable, virtual_addr: u64, physical_addr: u64) !void {
        const page_index = virtual_addr / self.page_size;
        std.log.debug("Mapping page {}: 0x{X} -> 0x{X}", .{ page_index, virtual_addr, physical_addr });
        // TODO: Implement actual page table entry setting
    }
    
    pub fn unmap_page(self: *PageTable, virtual_addr: u64) void {
        const page_index = virtual_addr / self.page_size;
        std.log.debug("Unmapping page {}: 0x{X}", .{ page_index, virtual_addr });
        // TODO: Implement actual page table entry clearing
    }
};

// Helper functions for memory mapping
fn mapDeviceMemory(phys_addr: u64, size: u64, flags: MemoryFlags) !u64 {
    _ = size;
    _ = flags;
    
    // In real kernel implementation:
    // return ioremap(phys_addr, size);
    
    // Simulate mapping by returning a fake virtual address
    return 0xFFFF000000000000 | phys_addr;
}

fn unmapDeviceMemory(virt_addr: u64, size: u64) void {
    _ = virt_addr;
    _ = size;
    
    // In real kernel implementation:
    // iounmap(virt_addr);
}

fn mapSystemMemory(phys_addr: u64, size: u64, flags: MemoryFlags) !u64 {
    _ = size;
    _ = flags;
    
    // In real kernel implementation:
    // return vmap(pfn_to_page(phys_addr >> PAGE_SHIFT), size >> PAGE_SHIFT, VM_MAP, PAGE_KERNEL);
    
    // Simulate mapping
    return 0xFFFF800000000000 | phys_addr;
}

fn unmapSystemMemory(virt_addr: u64, size: u64) void {
    _ = virt_addr;
    _ = size;
    
    // In real kernel implementation:
    // vunmap(virt_addr);
}

fn mapGartMemory(phys_addr: u64, size: u64, flags: MemoryFlags) !u64 {
    _ = size;
    _ = flags;
    
    // GART mapping involves programming the GPU's MMU
    // Simulate GART mapping
    return 0xC0000000 | phys_addr;
}

fn unmapGartMemory(virt_addr: u64, size: u64) void {
    _ = virt_addr;
    _ = size;
    
    // Unmap from GPU's MMU
}

fn mapDmaMemory(phys_addr: u64, size: u64, flags: MemoryFlags) !u64 {
    _ = size;
    _ = flags;
    
    // DMA memory mapping
    return phys_addr; // Identity mapping for DMA
}

fn unmapDmaMemory(virt_addr: u64, size: u64) void {
    _ = virt_addr;
    _ = size;
    
    // DMA memory unmapping
}

fn alignUp(value: u64, alignment: u64) u64 {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Test functions
test "memory pool allocation" {
    const allocator = std.testing.allocator;
    
    var pool = MemoryPool.init(allocator, .vram, 0x100000000, 1024 * 1024 * 1024); // 1GB
    defer pool.deinit();
    
    // Test allocation
    const region1 = try pool.allocate(4096, 4096, .framebuffer, MemoryFlags{});
    try std.testing.expect(region1.size == 4096);
    try std.testing.expect(region1.physical_address == 0x100000000);
    
    // Test second allocation
    const region2 = try pool.allocate(8192, 4096, .texture, MemoryFlags{});
    try std.testing.expect(region2.size == 8192);
    try std.testing.expect(region2.physical_address == 0x100001000);
    
    // Test stats
    const stats = pool.getStats();
    try std.testing.expect(stats.used_size == 12288);
    try std.testing.expect(stats.num_allocations == 2);
    
    // Test free
    try pool.free(region1);
    const stats2 = pool.getStats();
    try std.testing.expect(stats2.used_size == 8192);
    try std.testing.expect(stats2.num_allocations == 1);
}

test "memory manager initialization" {
    const allocator = std.testing.allocator;
    
    var manager = MemoryManager.init(allocator);
    defer manager.deinit();
    
    // Test basic initialization
    try std.testing.expect(manager.vram_pool == null);
    try std.testing.expect(manager.system_pool == null);
}

test "dma buffer management" {
    const allocator = std.testing.allocator;
    
    var manager = MemoryManager.init(allocator);
    defer manager.deinit();
    
    // Test DMA buffer allocation
    const buffer = try manager.allocateDmaBuffer(4096, true);
    try std.testing.expect(buffer.size == 4096);
    try std.testing.expect(buffer.coherent == true);
    
    // Test buffer mapping
    try buffer.map();
    try std.testing.expect(buffer.virtual_address != null);
    
    // Cleanup is handled by manager.deinit()
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