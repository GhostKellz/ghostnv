const std = @import("std");
const runtime = @import("runtime.zig");
const memory = @import("../hal/memory.zig");
const command = @import("../hal/command.zig");

pub const MemoryTransferAccelerator = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    copy_engines: []CopyEngine,
    pinned_memory_pool: PinnedMemoryPool,
    dma_scheduler: DmaScheduler,
    compression_engine: CompressionEngine,
    
    pub const CopyEngine = struct {
        id: u32,
        is_busy: std.atomic.Value(bool),
        current_transfer: ?TransferRequest,
        throughput_gbps: f32,
        pcie_gen: u8,
        
        pub fn init(id: u32, pcie_gen: u8) CopyEngine {
            return .{
                .id = id,
                .is_busy = std.atomic.Value(bool).init(false),
                .current_transfer = null,
                .throughput_gbps = switch (pcie_gen) {
                    5 => 63.0, // PCIe 5.0 x16
                    4 => 31.5, // PCIe 4.0 x16
                    3 => 15.75, // PCIe 3.0 x16
                    else => 15.75,
                },
                .pcie_gen = pcie_gen,
            };
        }
        
        pub fn estimateTransferTime(self: *const CopyEngine, size_bytes: usize) f64 {
            const size_gb = @as(f64, @floatFromInt(size_bytes)) / (1024.0 * 1024.0 * 1024.0);
            return size_gb / self.throughput_gbps;
        }
    };
    
    pub const TransferRequest = struct {
        id: u64,
        src_type: MemoryType,
        dst_type: MemoryType,
        src_addr: u64,
        dst_addr: u64,
        size: usize,
        priority: u8,
        use_compression: bool,
        use_pinned: bool,
        callback: ?*const fn (status: TransferStatus) void,
        
        pub const MemoryType = enum {
            host,
            device,
            unified,
            peer,
        };
        
        pub const TransferStatus = enum {
            pending,
            in_progress,
            completed,
            failed,
        };
    };
    
    pub const PinnedMemoryPool = struct {
        allocator: std.mem.Allocator,
        chunks: std.ArrayList(PinnedChunk),
        total_size: usize,
        available_size: usize,
        
        pub const PinnedChunk = struct {
            host_ptr: [*]u8,
            size: usize,
            is_free: bool,
            gpu_mapped_addr: u64,
        };
        
        pub fn init(allocator: std.mem.Allocator, initial_size: usize) !PinnedMemoryPool {
            var pool = PinnedMemoryPool{
                .allocator = allocator,
                .chunks = std.ArrayList(PinnedChunk).init(allocator),
                .total_size = 0,
                .available_size = 0,
            };
            
            // Pre-allocate pinned memory chunks
            const chunk_size = 64 * 1024 * 1024; // 64MB chunks
            const num_chunks = (initial_size + chunk_size - 1) / chunk_size;
            
            for (0..num_chunks) |_| {
                try pool.allocateChunk(chunk_size);
            }
            
            return pool;
        }
        
        pub fn deinit(self: *PinnedMemoryPool) void {
            for (self.chunks.items) |chunk| {
                // Free pinned memory
                std.c.free(@ptrCast(chunk.host_ptr));
            }
            self.chunks.deinit();
        }
        
        fn allocateChunk(self: *PinnedMemoryPool, size: usize) !void {
            // Allocate pinned host memory
            const host_ptr = std.c.malloc(size) orelse return error.OutOfMemory;
            
            // Lock pages in memory
            if (std.c.mlock(host_ptr, size) != 0) {
                std.c.free(host_ptr);
                return error.MemoryLockFailed;
            }
            
            const chunk = PinnedChunk{
                .host_ptr = @ptrCast(host_ptr),
                .size = size,
                .is_free = true,
                .gpu_mapped_addr = 0, // Will be set when registered with GPU
            };
            
            try self.chunks.append(chunk);
            self.total_size += size;
            self.available_size += size;
        }
        
        pub fn acquire(self: *PinnedMemoryPool, size: usize) ?*PinnedChunk {
            // Find a free chunk of sufficient size
            for (self.chunks.items) |*chunk| {
                if (chunk.is_free and chunk.size >= size) {
                    chunk.is_free = false;
                    self.available_size -= chunk.size;
                    return chunk;
                }
            }
            
            // Try to allocate a new chunk if needed
            self.allocateChunk(size) catch return null;
            
            // Try again
            return self.acquire(size);
        }
        
        pub fn release(self: *PinnedMemoryPool, chunk: *PinnedChunk) void {
            chunk.is_free = true;
            self.available_size += chunk.size;
        }
    };
    
    pub const DmaScheduler = struct {
        allocator: std.mem.Allocator,
        pending_transfers: std.ArrayList(TransferRequest),
        active_transfers: std.ArrayList(TransferRequest),
        transfer_history: TransferHistory,
        
        pub const TransferHistory = struct {
            recent_transfers: [256]TransferStats,
            write_index: u8,
            
            pub const TransferStats = struct {
                size: usize,
                duration_ns: u64,
                throughput_gbps: f32,
            };
            
            pub fn record(self: *TransferHistory, size: usize, duration_ns: u64) void {
                const throughput_gbps = (@as(f64, @floatFromInt(size)) * 8.0) / 
                                       (@as(f64, @floatFromInt(duration_ns)));
                
                self.recent_transfers[self.write_index] = .{
                    .size = size,
                    .duration_ns = duration_ns,
                    .throughput_gbps = @floatCast(throughput_gbps),
                };
                
                self.write_index +%= 1;
            }
            
            pub fn getAverageThroughput(self: *const TransferHistory) f32 {
                var sum: f32 = 0;
                var count: u32 = 0;
                
                for (self.recent_transfers) |stats| {
                    if (stats.size > 0) {
                        sum += stats.throughput_gbps;
                        count += 1;
                    }
                }
                
                return if (count > 0) sum / @as(f32, @floatFromInt(count)) else 0;
            }
        };
        
        pub fn init(allocator: std.mem.Allocator) DmaScheduler {
            return .{
                .allocator = allocator,
                .pending_transfers = std.ArrayList(TransferRequest).init(allocator),
                .active_transfers = std.ArrayList(TransferRequest).init(allocator),
                .transfer_history = .{
                    .recent_transfers = std.mem.zeroes([256]TransferHistory.TransferStats),
                    .write_index = 0,
                },
            };
        }
        
        pub fn deinit(self: *DmaScheduler) void {
            self.pending_transfers.deinit();
            self.active_transfers.deinit();
        }
        
        pub fn scheduleTransfer(self: *DmaScheduler, request: TransferRequest) !void {
            // Insert based on priority
            var insert_index: usize = 0;
            for (self.pending_transfers.items) |pending| {
                if (request.priority > pending.priority) break;
                insert_index += 1;
            }
            
            try self.pending_transfers.insert(insert_index, request);
        }
        
        pub fn getNextTransfer(self: *DmaScheduler) ?TransferRequest {
            if (self.pending_transfers.items.len == 0) return null;
            return self.pending_transfers.orderedRemove(0);
        }
    };
    
    pub const CompressionEngine = struct {
        enabled: bool,
        compression_ratio: f32,
        min_size_threshold: usize,
        
        pub fn shouldCompress(self: *const CompressionEngine, size: usize, transfer_type: TransferRequest.MemoryType) bool {
            if (!self.enabled) return false;
            if (size < self.min_size_threshold) return false;
            
            // Only compress host-to-device transfers
            return transfer_type == .host;
        }
        
        pub fn estimateCompressedSize(self: *const CompressionEngine, original_size: usize) usize {
            return @intFromFloat(@as(f32, @floatFromInt(original_size)) / self.compression_ratio);
        }
    };
    
    pub fn init(allocator: std.mem.Allocator, num_copy_engines: u32) !Self {
        var copy_engines = try allocator.alloc(CopyEngine, num_copy_engines);
        for (copy_engines, 0..) |*engine, i| {
            engine.* = CopyEngine.init(@intCast(i), 4); // PCIe 4.0
        }
        
        return Self{
            .allocator = allocator,
            .copy_engines = copy_engines,
            .pinned_memory_pool = try PinnedMemoryPool.init(allocator, 1024 * 1024 * 1024), // 1GB
            .dma_scheduler = DmaScheduler.init(allocator),
            .compression_engine = .{
                .enabled = true,
                .compression_ratio = 2.0, // Assume 2:1 compression
                .min_size_threshold = 1024 * 1024, // 1MB
            },
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.copy_engines);
        self.pinned_memory_pool.deinit();
        self.dma_scheduler.deinit();
    }
    
    pub fn transferAsync(self: *Self, request: TransferRequest) !void {
        var optimized_request = request;
        
        // Optimization 1: Use pinned memory for large transfers
        if (request.size > 4 * 1024 * 1024 and !request.use_pinned) {
            if (self.pinned_memory_pool.acquire(request.size)) |pinned_chunk| {
                optimized_request.use_pinned = true;
                // Copy to pinned memory first for staging
                @memcpy(pinned_chunk.host_ptr[0..request.size], @as([*]u8, @ptrFromInt(request.src_addr))[0..request.size]);
                optimized_request.src_addr = @intFromPtr(pinned_chunk.host_ptr);
            }
        }
        
        // Optimization 2: Enable compression for large transfers
        if (self.compression_engine.shouldCompress(request.size, request.src_type)) {
            optimized_request.use_compression = true;
        }
        
        // Schedule the transfer
        try self.dma_scheduler.scheduleTransfer(optimized_request);
        
        // Try to dispatch immediately
        self.dispatchTransfers();
    }
    
    pub fn transferSync(self: *Self, request: TransferRequest) !void {
        try self.transferAsync(request);
        
        // Wait for completion
        while (true) {
            self.dispatchTransfers();
            
            // Check if our transfer is complete
            var found = false;
            for (self.dma_scheduler.active_transfers.items) |active| {
                if (active.id == request.id) {
                    found = true;
                    break;
                }
            }
            
            if (!found) break;
            
            std.time.sleep(1000); // 1 microsecond
        }
    }
    
    fn dispatchTransfers(self: *Self) void {
        // Find available copy engines
        for (self.copy_engines) |*engine| {
            if (!engine.is_busy.load(.acquire)) {
                if (self.dma_scheduler.getNextTransfer()) |transfer| {
                    // Mark engine as busy
                    engine.is_busy.store(true, .release);
                    engine.current_transfer = transfer;
                    
                    // Move to active transfers
                    self.dma_scheduler.active_transfers.append(transfer) catch continue;
                    
                    // Launch the transfer
                    self.launchTransfer(engine, transfer) catch {
                        engine.is_busy.store(false, .release);
                        engine.current_transfer = null;
                    };
                }
            }
        }
    }
    
    fn launchTransfer(self: *Self, engine: *CopyEngine, transfer: TransferRequest) !void {
        const start_time = std.time.nanoTimestamp();
        
        // Build DMA command
        var dma_cmd = command.DmaCommand{
            .engine_id = engine.id,
            .src_addr = transfer.src_addr,
            .dst_addr = transfer.dst_addr,
            .size = transfer.size,
            .flags = 0,
        };
        
        // Set flags based on transfer optimizations
        if (transfer.use_pinned) {
            dma_cmd.flags |= command.DMA_FLAG_PINNED;
        }
        
        if (transfer.use_compression) {
            dma_cmd.flags |= command.DMA_FLAG_COMPRESS;
            dma_cmd.size = self.compression_engine.estimateCompressedSize(transfer.size);
        }
        
        // Enable peer-to-peer for device-to-device transfers
        if (transfer.src_type == .device and transfer.dst_type == .device) {
            dma_cmd.flags |= command.DMA_FLAG_PEER_TO_PEER;
        }
        
        // Submit DMA command
        const gpu_cmd = command.GpuCommand{
            .opcode = .dma_copy,
            .data = .{ .dma_copy = dma_cmd },
        };
        
        // Execute transfer
        // ... (actual hardware submission)
        
        // Simulate transfer completion
        const duration_ns = @intCast(u64, engine.estimateTransferTime(transfer.size) * 1e9);
        std.time.sleep(duration_ns);
        
        // Record statistics
        self.dma_scheduler.transfer_history.record(transfer.size, duration_ns);
        
        // Mark as complete
        engine.is_busy.store(false, .release);
        engine.current_transfer = null;
        
        // Remove from active transfers
        for (self.dma_scheduler.active_transfers.items, 0..) |active, i| {
            if (active.id == transfer.id) {
                _ = self.dma_scheduler.active_transfers.orderedRemove(i);
                break;
            }
        }
        
        // Call callback if provided
        if (transfer.callback) |callback| {
            callback(.completed);
        }
        
        std.log.debug("DMA transfer completed: {} bytes in {} ms", .{
            transfer.size,
            @as(f64, @floatFromInt(duration_ns)) / 1e6,
        });
    }
    
    pub fn optimizeTransferPath(self: *Self, src_type: TransferRequest.MemoryType, dst_type: TransferRequest.MemoryType, size: usize) TransferOptimization {
        var optimization = TransferOptimization{
            .use_pinned = false,
            .use_compression = false,
            .use_peer_to_peer = false,
            .estimated_bandwidth_gbps = 0,
            .recommended_chunk_size = size,
        };
        
        // Determine optimal transfer strategy
        switch (src_type) {
            .host => switch (dst_type) {
                .device => {
                    // Host to Device
                    optimization.use_pinned = size > 1024 * 1024; // Use pinned for > 1MB
                    optimization.use_compression = self.compression_engine.shouldCompress(size, src_type);
                    optimization.estimated_bandwidth_gbps = 25.0; // PCIe 4.0 typical
                    optimization.recommended_chunk_size = @min(size, 32 * 1024 * 1024); // 32MB chunks
                },
                .unified => {
                    optimization.estimated_bandwidth_gbps = 30.0;
                    optimization.recommended_chunk_size = @min(size, 64 * 1024 * 1024);
                },
                else => {},
            },
            .device => switch (dst_type) {
                .device => {
                    // Device to Device
                    optimization.use_peer_to_peer = true;
                    optimization.estimated_bandwidth_gbps = 600.0; // NVLink bandwidth
                    optimization.recommended_chunk_size = @min(size, 128 * 1024 * 1024);
                },
                .host => {
                    // Device to Host
                    optimization.use_pinned = size > 1024 * 1024;
                    optimization.estimated_bandwidth_gbps = 25.0;
                    optimization.recommended_chunk_size = @min(size, 32 * 1024 * 1024);
                },
                else => {},
            },
            else => {},
        }
        
        return optimization;
    }
    
    pub const TransferOptimization = struct {
        use_pinned: bool,
        use_compression: bool,
        use_peer_to_peer: bool,
        estimated_bandwidth_gbps: f32,
        recommended_chunk_size: usize,
    };
};

// Extension to CUDA runtime for memory transfer acceleration
pub fn cudaMemcpyAsyncOptimized(
    dst: *anyopaque,
    src: *const anyopaque,
    size: usize,
    kind: runtime.CudaMemcpyKind,
    stream: *runtime.CudaStream,
    accelerator: *MemoryTransferAccelerator,
) !void {
    const request = MemoryTransferAccelerator.TransferRequest{
        .id = @intFromPtr(src) ^ @intFromPtr(dst) ^ size,
        .src_type = switch (kind) {
            .host_to_device => .host,
            .device_to_host => .device,
            .device_to_device => .device,
            .host_to_host => .host,
        },
        .dst_type = switch (kind) {
            .host_to_device => .device,
            .device_to_host => .host,
            .device_to_device => .device,
            .host_to_host => .host,
        },
        .src_addr = @intFromPtr(src),
        .dst_addr = @intFromPtr(dst),
        .size = size,
        .priority = if (stream.priority > 0) 1 else 0,
        .use_compression = false,
        .use_pinned = false,
        .callback = null,
    };
    
    try accelerator.transferAsync(request);
}

// Batch memory transfer optimization
pub fn cudaMemcpyBatchOptimized(
    transfers: []const BatchTransfer,
    accelerator: *MemoryTransferAccelerator,
) !void {
    // Sort transfers by size for better scheduling
    var sorted_transfers = try accelerator.allocator.dupe(BatchTransfer, transfers);
    defer accelerator.allocator.free(sorted_transfers);
    
    std.sort.pdq(BatchTransfer, sorted_transfers, {}, struct {
        fn lessThan(_: void, a: BatchTransfer, b: BatchTransfer) bool {
            return a.size > b.size; // Larger transfers first
        }
    }.lessThan);
    
    // Submit all transfers
    for (sorted_transfers) |transfer| {
        const request = MemoryTransferAccelerator.TransferRequest{
            .id = @intFromPtr(transfer.src) ^ @intFromPtr(transfer.dst),
            .src_type = .host,
            .dst_type = .device,
            .src_addr = @intFromPtr(transfer.src),
            .dst_addr = @intFromPtr(transfer.dst),
            .size = transfer.size,
            .priority = 0,
            .use_compression = transfer.size > 10 * 1024 * 1024,
            .use_pinned = transfer.size > 1024 * 1024,
            .callback = null,
        };
        
        try accelerator.transferAsync(request);
    }
}

pub const BatchTransfer = struct {
    src: *const anyopaque,
    dst: *anyopaque,
    size: usize,
};