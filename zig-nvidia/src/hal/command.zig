const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const memory = @import("memory.zig");
const device = @import("../device/state.zig");

pub const CommandError = error{
    RingBufferFull,
    InvalidCommand,
    SubmissionFailed,
    SyncTimeout,
    EngineNotReady,
    OutOfMemory,
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
    nop = 0x00,
    mem_copy = 0x01,
    mem_fill = 0x02,
    compute_dispatch = 0x03,
    graphics_draw = 0x04,
    video_encode = 0x05,
    video_decode = 0x06,
    fence_signal = 0x07,
    fence_wait = 0x08,
    semaphore_acquire = 0x09,
    semaphore_release = 0x0A,
    pipeline_barrier = 0x0B,
    timestamp = 0x0C,
};

pub const CommandHeader = packed struct {
    cmd_type: CommandType,
    engine: EngineType,
    length: u16,
    flags: u32,
    
    pub fn init(cmd_type: CommandType, engine: EngineType, length: u16) CommandHeader {
        return CommandHeader{
            .cmd_type = cmd_type,
            .engine = engine,
            .length = length,
            .flags = 0,
        };
    }
};

pub const Command = struct {
    header: CommandHeader,
    payload: []u8,
    completion_fence: ?*Fence = null,
    
    pub fn init(allocator: Allocator, cmd_type: CommandType, engine: EngineType, payload_size: usize) !Command {
        const payload = try allocator.alloc(u8, payload_size);
        return Command{
            .header = CommandHeader.init(cmd_type, engine, @intCast(payload_size)),
            .payload = payload,
        };
    }
    
    pub fn deinit(self: *Command, allocator: Allocator) void {
        allocator.free(self.payload);
    }
};

pub const Fence = struct {
    id: u64,
    value: std.atomic.Value(u64),
    engine: EngineType,
    signaled: std.atomic.Value(bool),
    
    pub fn init(id: u64, engine: EngineType) Fence {
        return Fence{
            .id = id,
            .value = std.atomic.Value(u64).init(0),
            .engine = engine,
            .signaled = std.atomic.Value(bool).init(false),
        };
    }
    
    pub fn wait(self: *Fence, timeout_ns: u64) !void {
        const start_time = std.time.nanoTimestamp();
        while (!self.signaled.load(.acquire)) {
            if (std.time.nanoTimestamp() - start_time > timeout_ns) {
                return CommandError.SyncTimeout;
            }
            std.Thread.yield() catch {};
        }
    }
    
    pub fn signal(self: *Fence, value: u64) void {
        self.value.store(value, .release);
        self.signaled.store(true, .release);
    }
};

pub const RingBuffer = struct {
    const RING_SIZE = 64 * 1024; // 64KB ring buffer
    const MAX_COMMANDS = 1024;
    
    buffer: []u8,
    head: std.atomic.Value(u32),
    tail: std.atomic.Value(u32),
    gpu_head: std.atomic.Value(u32), // GPU's current position
    size: u32,
    engine: EngineType,
    commands: [MAX_COMMANDS]?Command,
    command_count: std.atomic.Value(u32),
    
    pub fn init(allocator: Allocator, engine: EngineType) !RingBuffer {
        const buffer = try allocator.alignedAlloc(u8, std.mem.page_size, RING_SIZE);
        
        return RingBuffer{
            .buffer = buffer,
            .head = std.atomic.Value(u32).init(0),
            .tail = std.atomic.Value(u32).init(0),
            .gpu_head = std.atomic.Value(u32).init(0),
            .size = RING_SIZE,
            .engine = engine,
            .commands = std.mem.zeroes([MAX_COMMANDS]?Command),
            .command_count = std.atomic.Value(u32).init(0),
        };
    }
    
    pub fn deinit(self: *RingBuffer, allocator: Allocator) void {
        // Clean up remaining commands
        for (self.commands) |*cmd| {
            if (cmd.*) |*command| {
                command.deinit(allocator);
            }
        }
        allocator.free(self.buffer);
    }
    
    pub fn submit(self: *RingBuffer, command: Command) !u32 {
        const current_count = self.command_count.load(.acquire);
        if (current_count >= MAX_COMMANDS) {
            return CommandError.RingBufferFull;
        }
        
        const cmd_index = current_count;
        self.commands[cmd_index] = command;
        
        // Write command to ring buffer
        const head = self.head.load(.acquire);
        const tail = self.tail.load(.acquire);
        const available = if (tail >= head) self.size - (tail - head) else head - tail;
        
        const cmd_size = @sizeOf(CommandHeader) + command.payload.len;
        if (available < cmd_size) {
            return CommandError.RingBufferFull;
        }
        
        // Write command header
        const header_bytes = std.mem.asBytes(&command.header);
        const wrap_point = (tail + header_bytes.len) % self.size;
        
        if (wrap_point < tail) {
            // Command wraps around
            const first_part = self.size - tail;
            @memcpy(self.buffer[tail..], header_bytes[0..first_part]);
            @memcpy(self.buffer[0..wrap_point], header_bytes[first_part..]);
        } else {
            @memcpy(self.buffer[tail..tail + header_bytes.len], header_bytes);
        }
        
        // Write payload
        var payload_start = (tail + header_bytes.len) % self.size;
        const payload_wrap = (payload_start + command.payload.len) % self.size;
        
        if (payload_wrap < payload_start) {
            const first_part = self.size - payload_start;
            @memcpy(self.buffer[payload_start..], command.payload[0..first_part]);
            @memcpy(self.buffer[0..payload_wrap], command.payload[first_part..]);
        } else {
            @memcpy(self.buffer[payload_start..payload_start + command.payload.len], command.payload);
        }
        
        // Update tail and command count
        self.tail.store((tail + cmd_size) % self.size, .release);
        self.command_count.store(current_count + 1, .release);
        
        return cmd_index;
    }
    
    pub fn kick(self: *RingBuffer) void {
        // Notify GPU that new commands are available
        // In real implementation, this would write to GPU doorbell register
        const tail = self.tail.load(.acquire);
        std.log.debug("Kicking {} engine, tail at {}", .{ self.engine, tail });
        
        // TODO: Write to actual GPU doorbell register
        // writel(tail, gpu_mmio + doorbell_offset);
    }
    
    pub fn process_completions(self: *RingBuffer) u32 {
        // Read GPU's current head position
        // In real implementation, this would read from GPU status register
        const new_gpu_head = self.gpu_head.load(.acquire);
        var completed = 0;
        
        // Process completed commands
        for (self.commands, 0..) |*cmd, i| {
            if (cmd.*) |*command| {
                if (command.completion_fence) |fence| {
                    if (fence.signaled.load(.acquire)) {
                        completed += 1;
                        self.commands[i] = null;
                    }
                }
            }
        }
        
        return @intCast(completed);
    }
};

pub const CommandScheduler = struct {
    rings: [6]RingBuffer, // One for each engine type
    allocator: Allocator,
    fence_counter: std.atomic.Value(u64),
    active_fences: std.ArrayList(*Fence),
    
    pub fn init(allocator: Allocator) !CommandScheduler {
        var rings: [6]RingBuffer = undefined;
        
        const engine_types = [_]EngineType{
            .graphics, .compute, .copy, .video_decode, .video_encode, .display
        };
        
        for (engine_types, 0..) |engine, i| {
            rings[i] = try RingBuffer.init(allocator, engine);
        }
        
        return CommandScheduler{
            .rings = rings,
            .allocator = allocator,
            .fence_counter = std.atomic.Value(u64).init(1),
            .active_fences = std.ArrayList(*Fence).init(allocator),
        };
    }
    
    pub fn deinit(self: *CommandScheduler) void {
        for (self.rings) |*ring| {
            ring.deinit(self.allocator);
        }
        
        // Clean up active fences
        for (self.active_fences.items) |fence| {
            self.allocator.destroy(fence);
        }
        self.active_fences.deinit();
    }
    
    pub fn submit_command(self: *CommandScheduler, command: Command) !u32 {
        const engine_index = @intFromEnum(command.header.engine);
        return try self.rings[engine_index].submit(command);
    }
    
    pub fn create_fence(self: *CommandScheduler, engine: EngineType) !*Fence {
        const fence_id = self.fence_counter.fetchAdd(1, .acq_rel);
        const fence = try self.allocator.create(Fence);
        fence.* = Fence.init(fence_id, engine);
        
        try self.active_fences.append(fence);
        return fence;
    }
    
    pub fn kick_engine(self: *CommandScheduler, engine: EngineType) void {
        const engine_index = @intFromEnum(engine);
        self.rings[engine_index].kick();
    }
    
    pub fn kick_all(self: *CommandScheduler) void {
        for (self.rings) |*ring| {
            ring.kick();
        }
    }
    
    pub fn process_completions(self: *CommandScheduler) u32 {
        var total_completed: u32 = 0;
        
        for (self.rings) |*ring| {
            total_completed += ring.process_completions();
        }
        
        return total_completed;
    }
    
    pub fn wait_idle(self: *CommandScheduler, timeout_ns: u64) !void {
        const start_time = std.time.nanoTimestamp();
        
        while (true) {
            var all_idle = true;
            
            for (self.rings) |*ring| {
                if (ring.command_count.load(.acquire) > 0) {
                    all_idle = false;
                    break;
                }
            }
            
            if (all_idle) break;
            
            if (std.time.nanoTimestamp() - start_time > timeout_ns) {
                return CommandError.SyncTimeout;
            }
            
            std.Thread.yield() catch {};
        }
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
    
    pub fn memory_copy(self: *CommandBuilder, src: u64, dst: u64, size: u64) !u32 {
        var command = try Command.init(self.allocator, .mem_copy, .copy, 24);
        
        // Pack copy parameters
        std.mem.writeInt(u64, command.payload[0..8], src, .little);
        std.mem.writeInt(u64, command.payload[8..16], dst, .little);
        std.mem.writeInt(u64, command.payload[16..24], size, .little);
        
        const cmd_id = try self.scheduler.submit_command(command);
        self.scheduler.kick_engine(.copy);
        
        return cmd_id;
    }
    
    pub fn compute_dispatch(self: *CommandBuilder, kernel_addr: u64, grid_x: u32, grid_y: u32, grid_z: u32, block_x: u32, block_y: u32, block_z: u32) !u32 {
        var command = try Command.init(self.allocator, .compute_dispatch, .compute, 32);
        
        // Pack compute parameters
        std.mem.writeInt(u64, command.payload[0..8], kernel_addr, .little);
        std.mem.writeInt(u32, command.payload[8..12], grid_x, .little);
        std.mem.writeInt(u32, command.payload[12..16], grid_y, .little);
        std.mem.writeInt(u32, command.payload[16..20], grid_z, .little);
        std.mem.writeInt(u32, command.payload[20..24], block_x, .little);
        std.mem.writeInt(u32, command.payload[24..28], block_y, .little);
        std.mem.writeInt(u32, command.payload[28..32], block_z, .little);
        
        const cmd_id = try self.scheduler.submit_command(command);
        self.scheduler.kick_engine(.compute);
        
        return cmd_id;
    }
    
    pub fn video_encode(self: *CommandBuilder, input_addr: u64, output_addr: u64, width: u32, height: u32, format: u32) !u32 {
        var command = try Command.init(self.allocator, .video_encode, .video_encode, 28);
        
        // Pack video encode parameters
        std.mem.writeInt(u64, command.payload[0..8], input_addr, .little);
        std.mem.writeInt(u64, command.payload[8..16], output_addr, .little);
        std.mem.writeInt(u32, command.payload[16..20], width, .little);
        std.mem.writeInt(u32, command.payload[20..24], height, .little);
        std.mem.writeInt(u32, command.payload[24..28], format, .little);
        
        const cmd_id = try self.scheduler.submit_command(command);
        self.scheduler.kick_engine(.video_encode);
        
        return cmd_id;
    }
    
    pub fn graphics_draw(self: *CommandBuilder, vertex_buffer: u64, index_buffer: u64, vertex_count: u32, index_count: u32) !u32 {
        var command = try Command.init(self.allocator, .graphics_draw, .graphics, 24);
        
        // Pack draw parameters
        std.mem.writeInt(u64, command.payload[0..8], vertex_buffer, .little);
        std.mem.writeInt(u64, command.payload[8..16], index_buffer, .little);
        std.mem.writeInt(u32, command.payload[16..20], vertex_count, .little);
        std.mem.writeInt(u32, command.payload[20..24], index_count, .little);
        
        const cmd_id = try self.scheduler.submit_command(command);
        self.scheduler.kick_engine(.graphics);
        
        return cmd_id;
    }
    
    pub fn insert_fence(self: *CommandBuilder, engine: EngineType) !*Fence {
        const fence = try self.scheduler.create_fence(engine);
        
        var command = try Command.init(self.allocator, .fence_signal, engine, 8);
        std.mem.writeInt(u64, command.payload[0..8], fence.id, .little);
        command.completion_fence = fence;
        
        _ = try self.scheduler.submit_command(command);
        self.scheduler.kick_engine(engine);
        
        return fence;
    }
    
    pub fn wait_fence(self: *CommandBuilder, fence: *Fence, engine: EngineType) !u32 {
        var command = try Command.init(self.allocator, .fence_wait, engine, 8);
        std.mem.writeInt(u64, command.payload[0..8], fence.id, .little);
        
        const cmd_id = try self.scheduler.submit_command(command);
        self.scheduler.kick_engine(engine);
        
        return cmd_id;
    }
};

// Test functions
test "command submission" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var scheduler = try CommandScheduler.init(allocator);
    defer scheduler.deinit();
    
    var builder = CommandBuilder.init(&scheduler, allocator);
    
    // Test memory copy
    const copy_id = try builder.memory_copy(0x1000, 0x2000, 1024);
    try std.testing.expect(copy_id == 0);
    
    // Test compute dispatch
    const compute_id = try builder.compute_dispatch(0x3000, 32, 32, 1, 256, 1, 1);
    try std.testing.expect(compute_id == 0);
}

test "fence synchronization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var scheduler = try CommandScheduler.init(allocator);
    defer scheduler.deinit();
    
    var builder = CommandBuilder.init(&scheduler, allocator);
    
    // Create fence
    const fence = try builder.insert_fence(.compute);
    
    // Test fence waiting (should timeout since we don't signal it)
    const result = fence.wait(1000000); // 1ms timeout
    try std.testing.expectError(CommandError.SyncTimeout, result);
    
    // Signal fence
    fence.signal(1);
    
    // Now wait should succeed
    try fence.wait(1000000);
}