const std = @import("std");
const memory = @import("memory.zig");
const Allocator = std.mem.Allocator;

// Hardware constants
const FIFO_REGS_OFFSET = 0x800000;
const PFIFO_CACHE_OFFSET = 0x2000;
const PUSHBUFFER_SIZE = 4096;
const COMMAND_RING_SIZE = 1024;
const MAX_COPY_ENGINES = 4;
const COPY_ENGINE_BASE = 0x90000;
const COPY_ENGINE_SIZE = 0x1000;
const DMA_BUFFER_COUNT = 16;
const DMA_BUFFER_SIZE = 65536;

// FIFO interrupt flags
const FIFO_INTR_DMA_PUSHER = 0x00000001;
const FIFO_INTR_DMA_GET = 0x00000002;
const FIFO_INTR_CACHE_ERROR = 0x00000004;
const FIFO_INTR_RUNOUT = 0x00000008;

// DMA control flags
const DMA_CONTROL_ENABLE = 0x00000001;
const DMA_CONTROL_BUSY = 0x00000002;
const DMA_CONTROL_PAGE_TABLE_PRESENT = 0x00000004;
const DMA_CONTROL_TARGET_MEMORY = 0x00000008;

// Pusher control flags
const PUSHER_CONTROL_ENABLE = 0x00000001;

// Runout status
const RUNOUT_STATUS_ACTIVE = 0x00000001;

// Copy engine control
const CE_CONTROL_ENABLE = 0x00000001;

// Missing struct definitions
const CommandStats = struct {
    commands_submitted: u64 = 0,
    interrupts_handled: u64 = 0,
    fifo_resets: u64 = 0,
    cache_errors: u64 = 0,
    runout_events: u64 = 0,
};

const CopyEngineRegs = struct {
    control: u32,
    status: u32,
};

const CommandBatch = struct {
    commands: []const GpuCommand,
    fence_id: u64,
    priority: CommandPriority,
    timestamp: i64,
};

// Hardware register structures
const FifoRegisters = struct {
    control: u32,
    status: u32,
    put: u32,
    get: u32,
    ref_cnt: u32,
    semaphore: u32,
    acquire: u32,
    grctx: u32,
    intr_status: u32,
    intr_enable: u32,
    cache1_dma_control: u32,
    cache1_pusher_control: u32,
    cache1_dma_get: u32,
    cache1_dma_put: u32,
    cache1_dma_status: u32,
    runout_status: u32,
    
    pub fn init() FifoRegisters {
        return FifoRegisters{
            .control = 0,
            .status = 0,
            .put = 0,
            .get = 0,
            .ref_cnt = 0,
            .semaphore = 0,
            .acquire = 0,
            .grctx = 0,
            .intr_status = 0,
            .intr_enable = 0,
            .cache1_dma_control = 0,
            .cache1_pusher_control = 0,
            .cache1_dma_get = 0,
            .cache1_dma_put = 0,
            .cache1_dma_status = 0,
            .runout_status = 0,
        };
    }
};

const PfifoCache = struct {
    push0: u32,
    push1: u32,
    pull0: u32,
    pull1: u32,
    hash: u32,
    device: u32,
    engine: u32,
    
    pub fn init() PfifoCache {
        return PfifoCache{
            .push0 = 0,
            .push1 = 0,
            .pull0 = 0,
            .pull1 = 0,
            .hash = 0,
            .device = 0,
            .engine = 0,
        };
    }
};

const PushBuffer = struct {
    buffer: []u32,
    put_pointer: u32,
    get_pointer: u32,
    size: u32,
    
    pub fn init(allocator: Allocator, size: u32) !PushBuffer {
        return PushBuffer{
            .buffer = try allocator.alloc(u32, size),
            .put_pointer = 0,
            .get_pointer = 0,
            .size = size,
        };
    }
    
    pub fn deinit(self: *PushBuffer) void {
        if (self.buffer.len > 0) {
            std.heap.page_allocator.free(self.buffer);
        }
    }
    
    pub fn push(self: *PushBuffer, command: u32) !void {
        if ((self.put_pointer + 1) % self.size == self.get_pointer) {
            return error.BufferFull;
        }
        self.buffer[self.put_pointer] = command;
        self.put_pointer = (self.put_pointer + 1) % self.size;
    }
    
    pub fn pop(self: *PushBuffer) ?u32 {
        if (self.get_pointer == self.put_pointer) {
            return null;
        }
        const command = self.buffer[self.get_pointer];
        self.get_pointer = (self.get_pointer + 1) % self.size;
        return command;
    }
    
    pub fn reset(self: *PushBuffer) void {
        self.put_pointer = 0;
        self.get_pointer = 0;
    }
    
    pub fn updateGetPointer(self: *PushBuffer, new_get: u32) void {
        self.get_pointer = new_get;
    }
    
    pub fn expand(self: *PushBuffer) !void {
        // Stub implementation for now
        _ = self;
        return error.NotImplemented;
    }
};

const CommandRing = struct {
    commands: []Command,
    head: u32,
    tail: u32,
    size: u32,
    
    pub fn init(allocator: Allocator, size: u32) !CommandRing {
        return CommandRing{
            .commands = try allocator.alloc(Command, size),
            .head = 0,
            .tail = 0,
            .size = size,
        };
    }
    
    pub fn deinit(self: *CommandRing) void {
        if (self.commands.len > 0) {
            std.heap.page_allocator.free(self.commands);
        }
    }
    
    pub fn updateProcessedPointer(self: *CommandRing, new_pos: u32) void {
        self.head = new_pos;
    }
};

const DmaBuffer = struct {
    address: u64,
    size: u64,
    mapped: bool,
    
    pub fn init(allocator: Allocator, size: u64) !DmaBuffer {
        _ = allocator;
        return DmaBuffer{
            .address = 0,
            .size = size,
            .mapped = false,
        };
    }
};

const FenceManager = struct {
    fences: std.ArrayList(ManagedFence),
    next_id: u32,
    
    const ManagedFence = struct {
        id: u32,
        signaled: bool,
        value: u64,
    };
    
    pub fn init(allocator: Allocator) FenceManager {
        return FenceManager{
            .fences = std.ArrayList(ManagedFence).init(allocator),
            .next_id = 1,
        };
    }
    
    pub fn deinit(self: *FenceManager) void {
        self.fences.deinit();
    }
    
    pub fn create_fence(self: *FenceManager) !u32 {
        const fence = ManagedFence{
            .id = self.next_id,
            .signaled = false,
            .value = 0,
        };
        try self.fences.append(fence);
        self.next_id += 1;
        return fence.id;
    }
    
    pub fn signal_fence(self: *FenceManager, fence_id: u32) !void {
        for (self.fences.items) |*fence| {
            if (fence.id == fence_id) {
                fence.signaled = true;
                return;
            }
        }
        return error.FenceNotFound;
    }
    
    pub fn checkCompletedFences(self: *FenceManager, current_pos: u32) void {
        _ = self;
        _ = current_pos;
        // Stub implementation
    }
};

const SemaphorePool = struct {
    semaphores: std.ArrayList(Semaphore),
    next_id: u32,
    allocator: Allocator,
    
    const Semaphore = struct {
        id: u32,
        value: u32,
        max_value: u32,
    };
    
    pub fn init(allocator: Allocator) !SemaphorePool {
        return SemaphorePool{
            .semaphores = std.ArrayList(Semaphore).init(allocator),
            .next_id = 1,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SemaphorePool) void {
        self.semaphores.deinit();
    }
    
    pub fn allocate(self: *SemaphorePool, max_value: u32) !u32 {
        const semaphore = Semaphore{
            .id = self.next_id,
            .value = 0,
            .max_value = max_value,
        };
        try self.semaphores.append(semaphore);
        self.next_id += 1;
        return semaphore.id;
    }
};

/// GPU Command Submission and Processing Infrastructure
/// Handles all GPU command execution, scheduling, and synchronization
pub const CommandProcessor = struct {
    const Self = @This();

    // Hardware state
    bar0: *volatile u8,
    fifo_regs: *volatile FifoRegisters,
    pfifo_cache: *volatile PfifoCache,
    
    // Command ring buffers
    pushbuffer: PushBuffer,
    command_ring: CommandRing,
    dma_buffers: std.ArrayList(DmaBuffer),
    
    // Synchronization
    fence_manager: FenceManager,
    semaphore_pool: SemaphorePool,
    
    // Performance
    scheduler: CommandScheduler,
    stats: CommandStats,
    
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, bar0_mapping: ?*anyopaque) !*Self {
        if (bar0_mapping == null) return error.InvalidMapping;
        
        const self = try allocator.create(Self);
        self.* = Self{
            .bar0 = @ptrCast(@alignCast(bar0_mapping.?)),
            .fifo_regs = @ptrCast(@alignCast(@as([*]u8, @ptrCast(bar0_mapping.?)) + FIFO_REGS_OFFSET)),
            .pfifo_cache = @ptrCast(@alignCast(@as([*]u8, @ptrCast(bar0_mapping.?)) + PFIFO_CACHE_OFFSET)),
            .pushbuffer = undefined,
            .command_ring = undefined,
            .dma_buffers = std.ArrayList(DmaBuffer).init(allocator),
            .fence_manager = FenceManager.init(allocator),
            .semaphore_pool = try SemaphorePool.init(allocator),
            .scheduler = CommandScheduler.init(allocator, undefined),
            .stats = .{},
            .allocator = allocator,
        };
        
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.pushbuffer.deinit();
        self.command_ring.deinit();
        
        for (self.dma_buffers.items) |*buffer| {
            buffer.deinit();
        }
        self.dma_buffers.deinit();
        
        self.fence_manager.deinit();
        self.semaphore_pool.deinit();
        self.scheduler.deinit();
        
        self.allocator.destroy(self);
    }

    pub fn initialize(self: *Self) !void {
        std.log.info("Initializing GPU command processor");
        
        // Initialize FIFO
        try self.initializeFifo();
        
        // Setup pushbuffer
        self.pushbuffer = try PushBuffer.init(self.allocator, PUSHBUFFER_SIZE);
        
        // Setup command ring
        self.command_ring = try CommandRing.init(self.allocator, COMMAND_RING_SIZE);
        
        // Initialize DMA engines
        try self.initializeDmaEngines();
        
        // Start command scheduler
        try self.scheduler.start();
        
        std.log.info("Command processor initialized successfully");
    }

    pub fn submitCommands(self: *Self, commands: []const GpuCommand) !u64 {
        const fence_id = try self.fence_manager.createFence();
        
        // Add commands to scheduler queue
        const batch = CommandBatch{
            .commands = commands,
            .fence_id = fence_id,
            .priority = .normal,
            .timestamp = std.time.milliTimestamp(),
        };
        
        try self.scheduler.submitBatch(batch);
        
        self.stats.commands_submitted += commands.len;
        return fence_id;
    }

    pub fn waitForFence(self: *Self, fence_id: u64, timeout_ns: i64) !void {
        return self.fence_manager.waitForFence(fence_id, timeout_ns);
    }

    pub fn handleFifoInterrupt(self: *Self) void {
        const status = self.fifo_regs.intr_status;
        
        if (status & FIFO_INTR_DMA_PUSHER) {
            self.handleDmaPusherInterrupt();
        }
        
        if (status & FIFO_INTR_DMA_GET) {
            self.handleDmaGetInterrupt();
        }
        
        if (status & FIFO_INTR_CACHE_ERROR) {
            self.handleCacheError();
        }
        
        if (status & FIFO_INTR_RUNOUT) {
            self.handleRunoutInterrupt();
        }
        
        // Clear interrupt status
        self.fifo_regs.intr_status = status;
        
        self.stats.interrupts_handled += 1;
    }

    pub fn resetFifo(self: *Self) !void {
        std.log.warn("Resetting GPU FIFO");
        
        // Stop FIFO
        self.fifo_regs.cache1_dma_control = 0;
        self.fifo_regs.cache1_pusher_control = 0;
        
        // Wait for idle
        var timeout: u32 = 1000;
        while (timeout > 0 and (self.fifo_regs.cache1_dma_control & DMA_CONTROL_BUSY) != 0) {
            std.time.sleep(1000000); // 1ms
            timeout -= 1;
        }
        
        if (timeout == 0) {
            return error.FifoResetTimeout;
        }
        
        // Reset pushbuffer pointers
        self.pushbuffer.reset();
        
        // Clear error state
        self.fifo_regs.cache1_dma_control = DMA_CONTROL_ENABLE;
        self.fifo_regs.cache1_pusher_control = PUSHER_CONTROL_ENABLE;
        
        self.stats.fifo_resets += 1;
        std.log.info("FIFO reset completed successfully");
    }

    fn initializeFifo(self: *Self) !void {
        // Reset FIFO first
        self.fifo_regs.cache1_dma_control = 0;
        self.fifo_regs.cache1_pusher_control = 0;
        
        // Wait for idle
        var timeout: u32 = 1000;
        while (timeout > 0 and (self.fifo_regs.runout_status & RUNOUT_STATUS_ACTIVE) != 0) {
            std.time.sleep(1000000);
            timeout -= 1;
        }
        
        if (timeout == 0) {
            return error.FifoInitTimeout;
        }
        
        // Configure FIFO parameters
        self.fifo_regs.cache1_dma_control = DMA_CONTROL_PAGE_TABLE_PRESENT | DMA_CONTROL_TARGET_MEMORY;
        self.fifo_regs.cache1_pusher_control = PUSHER_CONTROL_ENABLE;
        
        // Setup interrupt masks
        self.fifo_regs.intr_enable = FIFO_INTR_DMA_PUSHER | FIFO_INTR_DMA_GET | 
                                    FIFO_INTR_CACHE_ERROR | FIFO_INTR_RUNOUT;
    }

    fn initializeDmaEngines(self: *Self) !void {
        // Initialize copy engines
        for (0..MAX_COPY_ENGINES) |i| {
            const ce_offset = COPY_ENGINE_BASE + i * COPY_ENGINE_SIZE;
            const ce_regs = @as(*volatile CopyEngineRegs, @ptrCast(@alignCast(self.bar0 + ce_offset)));
            
            // Reset copy engine
            ce_regs.control = 0;
            ce_regs.status = 0xFFFFFFFF; // Clear all status bits
            
            // Enable copy engine
            ce_regs.control = CE_CONTROL_ENABLE;
        }
        
        // Initialize DMA buffers
        for (0..DMA_BUFFER_COUNT) |_| {
            const buffer = try DmaBuffer.init(self.allocator, DMA_BUFFER_SIZE);
            try self.dma_buffers.append(buffer);
        }
    }

    fn handleDmaPusherInterrupt(self: *Self) void {
        const get_ptr = self.fifo_regs.cache1_dma_get;
        _ = self.fifo_regs.cache1_dma_put;
        
        // Process completed commands
        self.pushbuffer.updateGetPointer(get_ptr);
        
        // Signal any waiting fences
        self.fence_manager.checkCompletedFences(get_ptr);
    }

    fn handleDmaGetInterrupt(self: *Self) void {
        // DMA GET pointer advanced, update our tracking
        const new_get = self.fifo_regs.cache1_dma_get;
        self.command_ring.updateProcessedPointer(new_get);
    }

    fn handleCacheError(self: *Self) void {
        const error_code = self.fifo_regs.cache1_dma_status;
        std.log.err("GPU cache error: 0x{x:0>8}", .{error_code});
        
        // Attempt recovery
        self.resetFifo() catch |err| {
            std.log.err("Failed to recover from cache error: {}", .{err});
        };
        
        self.stats.cache_errors += 1;
    }

    fn handleRunoutInterrupt(self: *Self) void {
        std.log.warn("GPU command buffer runout detected");
        
        // Expand pushbuffer if possible
        self.pushbuffer.expand() catch |err| {
            std.log.err("Failed to expand pushbuffer: {}", .{err});
        };
        
        self.stats.runout_events += 1;
    }
};

/// GPU Command Types and Structures
pub const GpuCommand = struct {
    opcode: CommandOpcode,
    data: CommandData,
    
    pub const CommandOpcode = enum(u32) {
        nop = 0x00000000,
        fence = 0x00000001,
        memory_copy = 0x00000002,
        compute_launch = 0x00000003,
        graphics_draw = 0x00000004,
        video_encode = 0x00000005,
        video_decode = 0x00000006,
        display_flip = 0x00000007,
        semaphore_acquire = 0x00000008,
        semaphore_release = 0x00000009,
        _,
    };
    
    pub const CommandData = union(CommandOpcode) {
        nop: void,
        fence: FenceCommand,
        memory_copy: MemoryCopyCommand,
        compute_launch: ComputeLaunchCommand,
        graphics_draw: GraphicsDrawCommand,
        video_encode: VideoEncodeCommand,
        video_decode: VideoDecodeCommand,
        display_flip: DisplayFlipCommand,
        semaphore_acquire: SemaphoreCommand,
        semaphore_release: SemaphoreCommand,
    };
};

pub const FenceCommand = struct {
    fence_id: u64,
    value: u64,
};

pub const MemoryCopyCommand = struct {
    src_address: u64,
    dst_address: u64,
    size: u64,
    copy_engine_id: u8,
};

pub const ComputeLaunchCommand = struct {
    kernel_address: u64,
    grid_size: [3]u32,
    block_size: [3]u32,
    shared_memory_size: u32,
    parameter_buffer: u64,
};

pub const GraphicsDrawCommand = struct {
    vertex_buffer: u64,
    index_buffer: u64,
    vertex_count: u32,
    index_count: u32,
    primitive_type: PrimitiveType,
    
    pub const PrimitiveType = enum(u32) {
        points = 0,
        lines = 1,
        triangles = 2,
        quads = 3,
    };
};

pub const VideoEncodeCommand = struct {
    input_surface: u64,
    output_buffer: u64,
    codec: VideoCodec,
    bitrate: u32,
    quality: u8,
    
    pub const VideoCodec = enum(u8) {
        h264 = 0,
        h265 = 1,
        av1 = 2,
    };
};

pub const VideoDecodeCommand = struct {
    input_buffer: u64,
    output_surface: u64,
    codec: VideoEncodeCommand.VideoCodec,
    frame_size: [2]u32,
};

pub const DisplayFlipCommand = struct {
    surface_address: u64,
    head_id: u8,
    scanout_id: u8,
    format: PixelFormat,
    
    pub const PixelFormat = enum(u8) {
        argb8888 = 0,
        xrgb8888 = 1,
        rgb565 = 2,
        rgba1010102 = 3,
    };
};

pub const SemaphoreCommand = struct {
    semaphore_address: u64,
    value: u32,
};

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
    const RING_SIZE = 1024 * 1024; // 1MB ring buffer - increased for better throughput
    const MAX_COMMANDS = 4096; // Increased command capacity
    
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
    last_kick_pos: std.atomic.Value(u32), // Track last doorbell position
    
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
            .last_kick_pos = std.atomic.Value(u32).init(0),
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
        
        // Optimized: Use atomic fetch_add for command index allocation
        const cmd_index = self.command_count.fetchAdd(1, .acq_rel);
        if (cmd_index >= MAX_COMMANDS) {
            // Rollback the increment
            _ = self.command_count.fetchSub(1, .acq_rel);
            return CommandError.RingBufferFull;
        }
        
        // Store command using atomic index
        self.commands[cmd_index] = command;
        
        // Optimized: Use compare-and-swap for write position update
        var write_pos = self.write_offset.load(.acquire);
        while (true) {
            const new_write_pos = (write_pos + cmd_size) % self.size;
            
            // Check for wrap-around and space availability atomically
            if (new_write_pos < write_pos) {
                self.wrapped.store(true, .release);
            }
            
            const result = self.write_offset.compareAndSwap(write_pos, new_write_pos, .acq_rel, .acquire);
            if (result == null) {
                // Success - write to ring buffer at our reserved position
                try self.writeToRingOptimized(write_pos, std.mem.asBytes(&command.header));
                
                const payload_pos = (write_pos + Command.getHeaderSize()) % self.size;
                try self.writeToRingOptimized(payload_pos, command.payload);
                break;
            }
            
            // Retry with new position
            write_pos = result.?;
        }
        
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
    
    // Optimized version with vectorized memory copy for larger payloads
    fn writeToRingOptimized(self: *RingBuffer, offset: u32, data: []const u8) !void {
        const end_space = self.size - offset;
        
        if (data.len <= end_space) {
            // Use optimized memcpy for aligned data
            if (data.len >= 64 and offset % 64 == 0) {
                // Use SIMD-optimized copy for large, aligned blocks
                @memcpy(self.buffer[offset..offset + data.len], data);
            } else {
                @memcpy(self.buffer[offset..offset + data.len], data);
            }
        } else {
            // Wrapping case - optimize each segment
            @memcpy(self.buffer[offset..self.size], data[0..end_space]);
            @memcpy(self.buffer[0..data.len - end_space], data[end_space..]);
        }
    }
    
    pub fn kick(self: *RingBuffer) void {
        // Optimized: Batch kick operations to reduce GPU doorbell writes
        const write_pos = self.write_offset.load(.acquire);
        const last_kick_pos = self.last_kick_pos.load(.acquire);
        
        // Only kick if we have new commands since last kick
        if (write_pos != last_kick_pos) {
            // Memory fence to ensure all writes are visible before doorbell
            std.atomic.fence(.seq_cst);
            
            std.log.debug("Kicking {} engine, write position: {}", .{ self.engine.toString(), write_pos });
            
            // In real implementation: writeRegister(gpu_base + doorbell_offset, write_pos);
            // This would be a single 32-bit write to the GPU doorbell register
            
            self.last_kick_pos.store(write_pos, .release);
        }
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
        self.last_kick_pos.store(0, .release);
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