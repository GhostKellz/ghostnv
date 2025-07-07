const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const memory = @import("memory.zig");
const pci = @import("pci.zig");

// Command Buffer Submission Pipeline for NVIDIA GPU

pub const CommandError = error{
    InvalidCommand,
    BufferFull,
    SubmissionFailed,
    TimeoutError,
    HardwareError,
    OutOfMemory,
    InvalidState,
    DeviceNotFound,
    FenceTimeout,
    InvalidFence,
    RingBufferFull,
    SyncTimeout,
    EngineNotReady,
    HardwareFault,
};

pub const EngineType = enum(u8) {
    graphics = 0,
    compute = 1,
    copy = 2,
    video_decode = 3,
    video_encode = 4,
    display = 5,
    
    pub fn toString(self: EngineType) []const u8 {
        return switch (self) {
            .graphics => "Graphics",
            .compute => "Compute",
            .copy => "Copy",
            .video_decode => "Video Decode",
            .video_encode => "Video Encode",
            .display => "Display",
        };
    }
};

pub const CommandType = enum(u8) {
    nop = 0x00,                    // No operation
    memory_copy = 0x01,            // Memory copy
    memory_fill = 0x02,            // Memory fill
    compute_dispatch = 0x03,       // Compute shader dispatch
    graphics_draw = 0x04,          // Graphics draw command
    barrier = 0x05,                // Memory/execution barrier
    timestamp = 0x06,              // Timestamp query
    conditional = 0x07,            // Conditional execution
    jump = 0x08,                   // Jump to address
    interrupt = 0x09,              // Generate interrupt
    fence_signal = 0x0A,           // Signal fence
    fence_wait = 0x0B,             // Wait on fence
    semaphore_acquire = 0x0C,      // Acquire semaphore
    semaphore_release = 0x0D,      // Release semaphore
    pipeline_barrier = 0x0E,       // Pipeline barrier
    video_encode = 0x0F,           // Video encode
    video_decode = 0x10,           // Video decode
    
    pub fn toString(self: CommandType) []const u8 {
        return switch (self) {
            .nop => "NOP",
            .memory_copy => "Memory Copy",
            .memory_fill => "Memory Fill",
            .compute_dispatch => "Compute Dispatch",
            .graphics_draw => "Graphics Draw",
            .barrier => "Barrier",
            .timestamp => "Timestamp",
            .conditional => "Conditional",
            .jump => "Jump",
            .interrupt => "Interrupt",
            .fence_signal => "Fence Signal",
            .fence_wait => "Fence Wait",
            .semaphore_acquire => "Semaphore Acquire",
            .semaphore_release => "Semaphore Release",
            .pipeline_barrier => "Pipeline Barrier",
            .video_encode => "Video Encode",
            .video_decode => "Video Decode",
        };
    }
};

pub const CommandPriority = enum(u8) {
    low = 0,
    normal = 1,
    high = 2,
    realtime = 3,
    
    pub fn toString(self: CommandPriority) []const u8 {
        return switch (self) {
            .low => "Low",
            .normal => "Normal", 
            .high => "High",
            .realtime => "Realtime",
        };
    }
};

pub const CommandFlags = packed struct {
    sync: bool = false,            // Synchronous execution
    interrupt: bool = false,       // Generate interrupt on completion
    fence: bool = false,           // Add fence after command
    debug: bool = false,           // Include debug information
    _reserved: u4 = 0,
};

pub const CommandHeader = packed struct {
    cmd_type: CommandType,
    engine: EngineType,
    length: u16,
    flags: CommandFlags,
    fence_id: u32,
    
    pub fn init(cmd_type: CommandType, engine: EngineType, length: u16) CommandHeader {
        return CommandHeader{
            .cmd_type = cmd_type,
            .engine = engine,
            .length = length,
            .flags = CommandFlags{},
            .fence_id = 0,
        };
    }
};

pub const Command = struct {
    header: CommandHeader,
    payload: []const u8,              // Command-specific data
    completion_fence: ?*Fence = null,
    
    pub fn init(cmd_type: CommandType, engine: EngineType, data: []const u8, flags: CommandFlags) Command {
        return Command{
            .header = CommandHeader{
                .cmd_type = cmd_type,
                .engine = engine,
                .length = @intCast(data.len),
                .flags = flags,
                .fence_id = 0,
            },
            .payload = data,
        };
    }
    
    pub fn initWithAllocator(allocator: Allocator, cmd_type: CommandType, engine: EngineType, payload_size: usize) !Command {
        const payload = try allocator.alloc(u8, payload_size);
        return Command{
            .header = CommandHeader.init(cmd_type, engine, @intCast(payload_size)),
            .payload = payload,
        };
    }
    
    pub fn deinit(self: *Command, allocator: Allocator) void {
        allocator.free(self.payload);
    }
    
    pub fn getHeaderSize() u32 {
        return @sizeOf(CommandHeader);
    }
    
    pub fn getTotalSize(self: Command) u32 {
        return Command.getHeaderSize() + @as(u32, @intCast(self.payload.len));
    }
};

pub const Fence = struct {
    fence_id: u32,
    engine: EngineType,
    value: std.atomic.Value(u64),
    signaled: std.atomic.Value(bool),
    timestamp: u64,
    
    pub fn init(fence_id: u32, engine: EngineType) Fence {
        return Fence{
            .fence_id = fence_id,
            .engine = engine,
            .value = std.atomic.Value(u64).init(0),
            .signaled = std.atomic.Value(bool).init(false),
            .timestamp = 0,
        };
    }
    
    pub fn signal(self: *Fence, value: u64) void {
        self.value.store(value, .release);
        self.signaled.store(true, .release);
        self.timestamp = std.time.nanoTimestamp();
    }
    
    pub fn wait(self: *Fence, timeout_ms: u32) !void {
        const start_time = std.time.milliTimestamp();
        
        while (!self.signaled.load(.acquire)) {
            if (std.time.milliTimestamp() - start_time > timeout_ms) {
                return CommandError.FenceTimeout;
            }
            
            std.Thread.yield() catch {};
            std.time.sleep(100000); // 0.1ms
        }
    }
    
    pub fn reset(self: *Fence) void {
        self.signaled.store(false, .release);
        self.value.store(0, .release);
        self.timestamp = 0;
    }
    
    pub fn isSignaled(self: *Fence) bool {
        return self.signaled.load(.acquire);
    }
    
    pub fn getValue(self: *Fence) u64 {
        return self.value.load(.acquire);
    }
};

pub const RingBuffer = struct {
    const RING_SIZE = 64 * 1024; // 64KB ring buffer
    const MAX_COMMANDS = 1024;
    
    allocator: Allocator,
    buffer: []u8,
    write_offset: std.atomic.Value(u32),
    read_offset: std.atomic.Value(u32),
    gpu_read_offset: std.atomic.Value(u32), // GPU's current read position
    size: u32,
    engine: EngineType,
    commands: [MAX_COMMANDS]?Command,
    command_count: std.atomic.Value(u32),
    wrapped: std.atomic.Value(bool),
    
    pub fn init(allocator: Allocator, engine: EngineType) !RingBuffer {
        const buffer = try allocator.alignedAlloc(u8, std.mem.page_size, RING_SIZE);
        @memset(buffer, 0);
        
        return RingBuffer{
            .allocator = allocator,
            .buffer = buffer,
            .write_offset = std.atomic.Value(u32).init(0),
            .read_offset = std.atomic.Value(u32).init(0),
            .gpu_read_offset = std.atomic.Value(u32).init(0),
            .size = RING_SIZE,
            .engine = engine,
            .commands = std.mem.zeroes([MAX_COMMANDS]?Command),
            .command_count = std.atomic.Value(u32).init(0),
            .wrapped = std.atomic.Value(bool).init(false),
        };
    }
    
    pub fn deinit(self: *RingBuffer) void {
        // Clean up remaining commands
        for (self.commands) |*cmd| {
            if (cmd.*) |*command| {
                command.deinit(self.allocator);
            }
        }
        self.allocator.free(self.buffer);
    }
    
    pub fn availableSpace(self: *RingBuffer) u32 {
        const write_pos = self.write_offset.load(.acquire);
        const read_pos = self.read_offset.load(.acquire);
        const wrapped = self.wrapped.load(.acquire);
        
        if (write_pos >= read_pos and !wrapped) {
            return self.size - (write_pos - read_pos);
        } else {
            return read_pos - write_pos;
        }
    }
    
    pub fn usedSpace(self: *RingBuffer) u32 {
        return self.size - self.availableSpace();
    }
    
    pub fn submit(self: *RingBuffer, command: Command) !u32 {
        const current_count = self.command_count.load(.acquire);
        if (current_count >= MAX_COMMANDS) {
            return CommandError.RingBufferFull;
        }
        
        const cmd_size = command.getTotalSize();
        if (cmd_size > self.availableSpace()) {
            return CommandError.BufferFull;
        }
        
        // Find a free command slot
        var cmd_index: u32 = 0;
        while (cmd_index < MAX_COMMANDS) : (cmd_index += 1) {
            if (self.commands[cmd_index] == null) break;
        }
        
        if (cmd_index >= MAX_COMMANDS) {
            return CommandError.RingBufferFull;
        }
        
        // Store command
        self.commands[cmd_index] = command;
        
        // Write to ring buffer
        const write_pos = self.write_offset.load(.acquire);
        try self.writeToRing(write_pos, std.mem.asBytes(&command.header));
        
        const payload_pos = (write_pos + Command.getHeaderSize()) % self.size;
        try self.writeToRing(payload_pos, command.payload);
        
        // Update write position
        const new_write_pos = (write_pos + cmd_size) % self.size;
        self.write_offset.store(new_write_pos, .release);
        
        if (new_write_pos < write_pos) {
            self.wrapped.store(true, .release);
        }
        
        self.command_count.store(current_count + 1, .release);
        
        std.log.debug("Submitted {} command to {} engine (size: {} bytes)", .{
            command.header.cmd_type.toString(),
            self.engine.toString(),
            cmd_size,
        });
        
        return cmd_index;
    }
    
    fn writeToRing(self: *RingBuffer, offset: u32, data: []const u8) !void {
        const end_space = self.size - offset;
        
        if (data.len <= end_space) {
            // Simple case: data fits without wrapping
            @memcpy(self.buffer[offset..offset + data.len], data);
        } else {
            // Complex case: data wraps around
            @memcpy(self.buffer[offset..self.size], data[0..end_space]);
            @memcpy(self.buffer[0..data.len - end_space], data[end_space..]);
        }
    }
    
    pub fn kick(self: *RingBuffer) void {
        // Notify GPU that new commands are available
        // In real implementation, this would write to GPU doorbell register
        const write_pos = self.write_offset.load(.acquire);
        
        std.log.debug("Kicking {} engine, write position: {}", .{ self.engine.toString(), write_pos });
        
        // Simulate writing to GPU doorbell register
        // In real implementation: writeRegister(gpu_base + doorbell_offset, write_pos);
    }
    
    pub fn processCompletions(self: *RingBuffer) u32 {
        // Simulate GPU progress by advancing GPU read offset
        const current_gpu_pos = self.gpu_read_offset.load(.acquire);
        const write_pos = self.write_offset.load(.acquire);
        
        // Simulate GPU processing some commands
        var new_gpu_pos = current_gpu_pos;
        if (current_gpu_pos != write_pos) {
            new_gpu_pos = (current_gpu_pos + 32) % self.size; // Simulate 32 bytes processed
            if (new_gpu_pos > write_pos and current_gpu_pos < write_pos) {
                new_gpu_pos = write_pos; // Don't go past write position
            }
        }
        
        self.gpu_read_offset.store(new_gpu_pos, .release);
        
        // Mark completed commands
        var completed: u32 = 0;
        for (self.commands, 0..) |*cmd, i| {
            if (cmd.*) |*command| {
                if (command.completion_fence) |fence| {
                    if (!fence.isSignaled()) {
                        // Simulate fence completion
                        fence.signal(1);
                        completed += 1;
                        self.commands[i] = null;
                    }
                }
            }
        }
        
        if (completed > 0) {
            const current_count = self.command_count.load(.acquire);
            self.command_count.store(current_count - completed, .release);
        }
        
        return completed;
    }
    
    pub fn isEmpty(self: *RingBuffer) bool {
        return self.command_count.load(.acquire) == 0;
    }
    
    pub fn isFull(self: *RingBuffer) bool {
        return self.command_count.load(.acquire) >= MAX_COMMANDS;
    }
    
    pub fn reset(self: *RingBuffer) void {
        // Clean up all commands
        for (self.commands) |*cmd| {
            if (cmd.*) |*command| {
                command.deinit(self.allocator);
            }
        }
        
        self.write_offset.store(0, .release);
        self.read_offset.store(0, .release);
        self.gpu_read_offset.store(0, .release);
        self.command_count.store(0, .release);
        self.wrapped.store(false, .release);
        self.commands = std.mem.zeroes([MAX_COMMANDS]?Command);
        @memset(self.buffer, 0);
        
        std.log.info("Reset {} engine ring buffer", .{self.engine.toString()});
    }
};

pub const CommandQueue = struct {
    allocator: Allocator,
    queue_id: u32,
    priority: CommandPriority,
    ring_buffer: RingBuffer,
    commands_submitted: u64,
    commands_completed: u64,
    
    pub fn init(allocator: Allocator, queue_id: u32, engine: EngineType, priority: CommandPriority) !CommandQueue {
        return CommandQueue{
            .allocator = allocator,
            .queue_id = queue_id,
            .priority = priority,
            .ring_buffer = try RingBuffer.init(allocator, engine),
            .commands_submitted = 0,
            .commands_completed = 0,
        };
    }
    
    pub fn deinit(self: *CommandQueue) void {
        self.ring_buffer.deinit();
    }
    
    pub fn submitCommand(self: *CommandQueue, command: Command) !void {
        _ = try self.ring_buffer.submit(command);
        self.commands_submitted += 1;
        self.ring_buffer.kick();
    }
    
    pub fn submitCommands(self: *CommandQueue, commands: []const Command) !void {
        for (commands) |command| {
            try self.submitCommand(command);
        }
    }
    
    pub fn waitForCompletion(self: *CommandQueue, timeout_ms: u32) !void {
        const start_time = std.time.milliTimestamp();
        
        while (self.commands_completed < self.commands_submitted) {
            const completed = self.ring_buffer.processCompletions();
            self.commands_completed += completed;
            
            if (std.time.milliTimestamp() - start_time > timeout_ms) {
                return CommandError.TimeoutError;
            }
            
            std.time.sleep(1000000); // 1ms
        }
    }
    
    pub fn getStats(self: *CommandQueue) CommandQueueStats {
        return CommandQueueStats{
            .queue_id = self.queue_id,
            .priority = self.priority,
            .engine = self.ring_buffer.engine,
            .commands_submitted = self.commands_submitted,
            .commands_completed = self.commands_completed,
            .commands_pending = self.commands_submitted - self.commands_completed,
            .buffer_used = self.ring_buffer.usedSpace(),
            .buffer_total = self.ring_buffer.size,
            .buffer_utilization = @as(f32, @floatFromInt(self.ring_buffer.usedSpace())) / 
                                 @as(f32, @floatFromInt(self.ring_buffer.size)) * 100.0,
        };
    }
    
    pub fn flush(self: *CommandQueue) !void {
        self.ring_buffer.kick();
        try self.waitForCompletion(5000); // 5 second timeout
    }
    
    pub fn reset(self: *CommandQueue) void {
        self.ring_buffer.reset();
        self.commands_submitted = 0;
        self.commands_completed = 0;
        
        std.log.info("Reset command queue {}", .{self.queue_id});
    }
};

pub const CommandQueueStats = struct {
    queue_id: u32,
    priority: CommandPriority,
    engine: EngineType,
    commands_submitted: u64,
    commands_completed: u64,
    commands_pending: u64,
    buffer_used: u32,
    buffer_total: u32,
    buffer_utilization: f32,
};

pub const CommandScheduler = struct {
    allocator: Allocator,
    queues: std.ArrayList(CommandQueue),
    fences: std.HashMap(u32, Fence, std.hash_map.AutoContext(u32), 80),
    next_fence_id: std.atomic.Value(u32),
    memory_manager: *memory.MemoryManager,
    
    pub fn init(allocator: Allocator, memory_manager: *memory.MemoryManager) CommandScheduler {
        return CommandScheduler{
            .allocator = allocator,
            .queues = std.ArrayList(CommandQueue).init(allocator),
            .fences = std.HashMap(u32, Fence, std.HashMap.DefaultContext(u32), std.HashMap.default_max_load_percentage).init(allocator),
            .next_fence_id = std.atomic.Value(u32).init(1),
            .memory_manager = memory_manager,
        };
    }
    
    pub fn deinit(self: *CommandScheduler) void {
        for (self.queues.items) |*queue| {
            queue.deinit();
        }
        self.queues.deinit();
        self.fences.deinit();
    }
    
    pub fn createQueue(self: *CommandScheduler, engine: EngineType, priority: CommandPriority) !u32 {
        const queue_id = @as(u32, @intCast(self.queues.items.len));
        const queue = try CommandQueue.init(self.allocator, queue_id, engine, priority);
        try self.queues.append(queue);
        
        std.log.info("Created {} command queue {} with priority {}", .{
            engine.toString(),
            queue_id,
            priority.toString(),
        });
        
        return queue_id;
    }
    
    pub fn getQueue(self: *CommandScheduler, queue_id: u32) !*CommandQueue {
        if (queue_id >= self.queues.items.len) {
            return CommandError.InvalidCommand;
        }
        return &self.queues.items[queue_id];
    }
    
    pub fn submitToQueue(self: *CommandScheduler, queue_id: u32, command: Command) !void {
        const queue = try self.getQueue(queue_id);
        try queue.submitCommand(command);
    }
    
    pub fn createFence(self: *CommandScheduler, engine: EngineType) !u32 {
        const fence_id = self.next_fence_id.fetchAdd(1, .acq_rel);
        try self.fences.put(fence_id, Fence.init(fence_id, engine));
        
        std.log.debug("Created fence {} for {} engine", .{ fence_id, engine.toString() });
        return fence_id;
    }
    
    pub fn submitWithFence(self: *CommandScheduler, queue_id: u32, commands: []const Command, engine: EngineType) !u32 {
        const fence_id = try self.createFence(engine);
        const fence = self.fences.getPtr(fence_id).?;
        
        const queue = try self.getQueue(queue_id);
        
        // Submit all commands
        for (commands) |command| {
            var cmd_with_fence = command;
            cmd_with_fence.header.fence_id = fence_id;
            cmd_with_fence.header.flags.fence = true;
            cmd_with_fence.completion_fence = fence;
            try queue.submitCommand(cmd_with_fence);
        }
        
        std.log.debug("Submitted {} commands with fence {}", .{ commands.len, fence_id });
        return fence_id;
    }
    
    pub fn waitForFence(self: *CommandScheduler, fence_id: u32, timeout_ms: u32) !void {
        var fence = self.fences.getPtr(fence_id) orelse return CommandError.InvalidFence;
        try fence.wait(timeout_ms);
    }
    
    pub fn signalFence(self: *CommandScheduler, fence_id: u32) !void {
        var fence = self.fences.getPtr(fence_id) orelse return CommandError.InvalidFence;
        fence.signal(1);
    }
    
    pub fn waitIdle(self: *CommandScheduler, timeout_ms: u32) !void {
        const start_time = std.time.milliTimestamp();
        
        while (true) {
            var all_idle = true;
            
            for (self.queues.items) |*queue| {
                if (queue.commands_submitted > queue.commands_completed) {
                    queue.commands_completed += queue.ring_buffer.processCompletions();
                    
                    if (queue.commands_submitted > queue.commands_completed) {
                        all_idle = false;
                    }
                }
            }
            
            if (all_idle) break;
            
            if (std.time.milliTimestamp() - start_time > timeout_ms) {
                return CommandError.SyncTimeout;
            }
            
            std.time.sleep(1000000); // 1ms
        }
    }
    
    pub fn flushAllQueues(self: *CommandScheduler) !void {
        for (self.queues.items) |*queue| {
            try queue.flush();
        }
    }
    
    pub fn getAllQueueStats(self: *CommandScheduler) ![]CommandQueueStats {
        var stats = try self.allocator.alloc(CommandQueueStats, self.queues.items.len);
        
        for (self.queues.items, 0..) |*queue, i| {
            stats[i] = queue.getStats();
        }
        
        return stats;
    }
    
    pub fn printStats(self: *CommandScheduler) !void {
        const stats = try self.getAllQueueStats();
        defer self.allocator.free(stats);
        
        std.log.info("=== Command Scheduler Statistics ===");
        
        for (stats) |queue_stats| {
            std.log.info("Queue {}: {} {} priority", .{
                queue_stats.queue_id,
                queue_stats.engine.toString(),
                queue_stats.priority.toString(),
            });
            std.log.info("  Commands: {}/{} submitted/completed ({} pending)", .{
                queue_stats.commands_submitted,
                queue_stats.commands_completed,
                queue_stats.commands_pending,
            });
            std.log.info("  Buffer: {}/{} bytes ({d:.1}% utilization)", .{
                queue_stats.buffer_used,
                queue_stats.buffer_total,
                queue_stats.buffer_utilization,
            });
        }
        
        std.log.info("Total fences: {}", .{self.fences.count()});
    }
    
    pub fn resetAllQueues(self: *CommandScheduler) void {
        for (self.queues.items) |*queue| {
            queue.reset();
        }
        
        self.fences.clearAndFree();
        self.next_fence_id.store(1, .release);
        
        std.log.info("Reset all command queues and fences");
    }
};

// High-level command builders
pub const CommandBuilder = struct {
    scheduler: *CommandScheduler,
    allocator: Allocator,
    
    pub fn init(scheduler: *CommandScheduler, allocator: Allocator) CommandBuilder {
        return CommandBuilder{
            .scheduler = scheduler,
            .allocator = allocator,
        };
    }
    
    pub fn createMemoryCopyCommand(
        self: *CommandBuilder,
        src_addr: u64,
        dst_addr: u64,
        size: u64,
    ) !Command {
        _ = self;
        
        var data: [24]u8 = undefined;
        var stream = std.io.fixedBufferStream(&data);
        var writer = stream.writer();
        
        try writer.writeInt(u64, src_addr, .little);
        try writer.writeInt(u64, dst_addr, .little);
        try writer.writeInt(u64, size, .little);
        
        return Command.init(.memory_copy, .copy, data[0..stream.getWritten().len], CommandFlags{});
    }
    
    pub fn createMemoryFillCommand(
        self: *CommandBuilder,
        dst_addr: u64,
        value: u32,
        size: u64,
    ) !Command {
        _ = self;
        
        var data: [20]u8 = undefined;
        var stream = std.io.fixedBufferStream(&data);
        var writer = stream.writer();
        
        try writer.writeInt(u64, dst_addr, .little);
        try writer.writeInt(u32, value, .little);
        try writer.writeInt(u64, size, .little);
        
        return Command.init(.memory_fill, .copy, data[0..stream.getWritten().len], CommandFlags{});
    }
    
    pub fn createComputeDispatchCommand(
        self: *CommandBuilder,
        shader_addr: u64,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) !Command {
        _ = self;
        
        var data: [20]u8 = undefined;
        var stream = std.io.fixedBufferStream(&data);
        var writer = stream.writer();
        
        try writer.writeInt(u64, shader_addr, .little);
        try writer.writeInt(u32, group_count_x, .little);
        try writer.writeInt(u32, group_count_y, .little);
        try writer.writeInt(u32, group_count_z, .little);
        
        return Command.init(.compute_dispatch, .compute, data[0..stream.getWritten().len], CommandFlags{});
    }
    
    pub fn createGraphicsDrawCommand(
        self: *CommandBuilder,
        vertex_buffer: u64,
        index_buffer: u64,
        vertex_count: u32,
        index_count: u32,
    ) !Command {
        _ = self;
        
        var data: [24]u8 = undefined;
        var stream = std.io.fixedBufferStream(&data);
        var writer = stream.writer();
        
        try writer.writeInt(u64, vertex_buffer, .little);
        try writer.writeInt(u64, index_buffer, .little);
        try writer.writeInt(u32, vertex_count, .little);
        try writer.writeInt(u32, index_count, .little);
        
        return Command.init(.graphics_draw, .graphics, data[0..stream.getWritten().len], CommandFlags{});
    }
    
    pub fn createVideoEncodeCommand(
        self: *CommandBuilder,
        input_addr: u64,
        output_addr: u64,
        width: u32,
        height: u32,
        format: u32,
    ) !Command {
        _ = self;
        
        var data: [28]u8 = undefined;
        var stream = std.io.fixedBufferStream(&data);
        var writer = stream.writer();
        
        try writer.writeInt(u64, input_addr, .little);
        try writer.writeInt(u64, output_addr, .little);
        try writer.writeInt(u32, width, .little);
        try writer.writeInt(u32, height, .little);
        try writer.writeInt(u32, format, .little);
        
        return Command.init(.video_encode, .video_encode, data[0..stream.getWritten().len], CommandFlags{});
    }
    
    pub fn submitMemoryCopy(self: *CommandBuilder, queue_id: u32, src: u64, dst: u64, size: u64) !void {
        const cmd = try self.createMemoryCopyCommand(src, dst, size);
        try self.scheduler.submitToQueue(queue_id, cmd);
    }
    
    pub fn submitComputeDispatch(
        self: *CommandBuilder,
        queue_id: u32,
        shader_addr: u64,
        group_x: u32,
        group_y: u32,
        group_z: u32,
    ) !void {
        const cmd = try self.createComputeDispatchCommand(shader_addr, group_x, group_y, group_z);
        try self.scheduler.submitToQueue(queue_id, cmd);
    }
    
    pub fn submitGraphicsDraw(
        self: *CommandBuilder,
        queue_id: u32,
        vertex_buffer: u64,
        index_buffer: u64,
        vertex_count: u32,
        index_count: u32,
    ) !void {
        const cmd = try self.createGraphicsDrawCommand(vertex_buffer, index_buffer, vertex_count, index_count);
        try self.scheduler.submitToQueue(queue_id, cmd);
    }
};

// Helper functions for common command patterns
pub fn createNopCommand() Command {
    return Command.init(.nop, .graphics, &[_]u8{}, CommandFlags{});
}

pub fn createBarrierCommand(engine: EngineType) Command {
    return Command.init(.barrier, engine, &[_]u8{}, CommandFlags{});
}

pub fn createInterruptCommand(engine: EngineType) Command {
    return Command.init(.interrupt, engine, &[_]u8{}, CommandFlags{ .interrupt = true });
}

// Test functions
test "ring buffer operations" {
    const allocator = std.testing.allocator;
    
    var ring = try RingBuffer.init(allocator, .graphics);
    defer ring.deinit();
    
    // Test command submission
    const nop_cmd = createNopCommand();
    const cmd_id = try ring.submit(nop_cmd);
    
    try std.testing.expect(cmd_id == 0);
    try std.testing.expect(ring.command_count.load(.acquire) == 1);
    try std.testing.expect(!ring.isEmpty());
}

test "command queue operations" {
    const allocator = std.testing.allocator;
    
    var queue = try CommandQueue.init(allocator, 0, .graphics, .normal);
    defer queue.deinit();
    
    // Test command submission
    const nop_cmd = createNopCommand();
    try queue.submitCommand(nop_cmd);
    
    try std.testing.expect(queue.commands_submitted == 1);
    try std.testing.expect(queue.commands_completed == 0);
    
    // Test completion waiting
    try queue.waitForCompletion(1000);
    
    try std.testing.expect(queue.commands_completed >= 1);
}

test "command scheduler" {
    const allocator = std.testing.allocator;
    
    var memory_manager = memory.MemoryManager.init(allocator);
    defer memory_manager.deinit();
    
    var scheduler = CommandScheduler.init(allocator, &memory_manager);
    defer scheduler.deinit();
    
    // Create queues
    const graphics_queue = try scheduler.createQueue(.graphics, .normal);
    const compute_queue = try scheduler.createQueue(.compute, .high);
    
    try std.testing.expect(graphics_queue == 0);
    try std.testing.expect(compute_queue == 1);
    
    // Test command submission
    const nop_cmd = createNopCommand();
    try scheduler.submitToQueue(graphics_queue, nop_cmd);
    
    // Test fence creation and waiting
    const fence_id = try scheduler.createFence(.graphics);
    try std.testing.expect(fence_id == 1);
    
    try scheduler.signalFence(fence_id);
    try scheduler.waitForFence(fence_id, 1000);
}

test "command builder" {
    const allocator = std.testing.allocator;
    
    var memory_manager = memory.MemoryManager.init(allocator);
    defer memory_manager.deinit();
    
    var scheduler = CommandScheduler.init(allocator, &memory_manager);
    defer scheduler.deinit();
    
    var builder = CommandBuilder.init(&scheduler, allocator);
    
    // Test command creation
    const copy_cmd = try builder.createMemoryCopyCommand(0x1000, 0x2000, 1024);
    try std.testing.expect(copy_cmd.header.cmd_type == .memory_copy);
    try std.testing.expect(copy_cmd.header.engine == .copy);
    try std.testing.expect(copy_cmd.header.length == 24); // 3 * 8 bytes
    
    const compute_cmd = try builder.createComputeDispatchCommand(0x3000, 32, 32, 1);
    try std.testing.expect(compute_cmd.header.cmd_type == .compute_dispatch);
    try std.testing.expect(compute_cmd.header.engine == .compute);
    try std.testing.expect(compute_cmd.header.length == 20); // 1 * 8 + 3 * 4 bytes
}

test "fence synchronization" {
    const allocator = std.testing.allocator;
    
    var memory_manager = memory.MemoryManager.init(allocator);
    defer memory_manager.deinit();
    
    var scheduler = CommandScheduler.init(allocator, &memory_manager);
    defer scheduler.deinit();
    
    // Create fence
    const fence_id = try scheduler.createFence(.compute);
    
    // Test fence waiting (should timeout since we don't signal it)
    const result = scheduler.waitForFence(fence_id, 10); // 10ms timeout
    try std.testing.expectError(CommandError.FenceTimeout, result);
    
    // Signal fence
    try scheduler.signalFence(fence_id);
    
    // Now wait should succeed
    try scheduler.waitForFence(fence_id, 1000);
}