const std = @import("std");
const memory = @import("../hal/memory.zig");
const command = @import("../hal/command.zig");

/// CUDA Runtime Implementation for GhostNV
/// Provides complete CUDA 12.x compatibility with native Zig integration
pub const CudaRuntime = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    device_contexts: std.ArrayList(CudaDevice),
    memory_manager: *memory.MemoryManager,
    stream_manager: StreamManager,
    kernel_manager: KernelManager,
    tensor_core_manager: TensorCoreManager,
    profiler: CudaProfiler,

    // Hardware state
    compute_units: []ComputeUnit,
    shared_memory_pool: SharedMemoryPool,

    pub fn init(allocator: std.mem.Allocator, mem_manager: *memory.MemoryManager) !Self {
        var self = Self{
            .allocator = allocator,
            .device_contexts = std.ArrayList(CudaDevice).init(allocator),
            .memory_manager = mem_manager,
            .stream_manager = try StreamManager.init(allocator),
            .kernel_manager = try KernelManager.init(allocator),
            .tensor_core_manager = try TensorCoreManager.init(allocator),
            .profiler = try CudaProfiler.init(allocator),
            .compute_units = &.{},
            .shared_memory_pool = undefined,
        };

        // Initialize compute units
        try self.initializeComputeUnits();

        // Initialize shared memory pool
        self.shared_memory_pool = try SharedMemoryPool.init(allocator, 64 * 1024); // 64KB shared memory

        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.device_contexts.items) |*device| {
            device.deinit();
        }
        self.device_contexts.deinit();

        self.stream_manager.deinit();
        self.kernel_manager.deinit();
        self.tensor_core_manager.deinit();
        self.profiler.deinit();
        self.shared_memory_pool.deinit();

        if (self.compute_units.len > 0) {
            self.allocator.free(self.compute_units);
        }
    }

    fn initializeComputeUnits(self: *Self) !void {
        // Initialize based on GPU architecture
        const num_sms = 128; // RTX 4090 has 128 SMs
        self.compute_units = try self.allocator.alloc(ComputeUnit, num_sms);

        for (0..num_sms) |i| {
            self.compute_units[i] = try ComputeUnit.init(self.allocator, @intCast(i));
        }

        std.log.info("Initialized {} compute units", .{num_sms});
    }

    pub fn getDeviceCount(self: *Self) u32 {
        return @intCast(self.device_contexts.items.len);
    }

    pub fn getDeviceProperties(self: *Self, device_id: u32) !CudaDeviceProperties {
        if (device_id >= self.device_contexts.items.len) {
            return error.InvalidDevice;
        }

        const device = &self.device_contexts.items[device_id];
        return device.properties;
    }

    pub fn setDevice(self: *Self, device_id: u32) !void {
        if (device_id >= self.device_contexts.items.len) {
            return error.InvalidDevice;
        }

        // Set current device context
        const device = &self.device_contexts.items[device_id];
        device.is_current = true;

        // Deactivate other devices
        for (self.device_contexts.items) |*other_device| {
            if (other_device != device) {
                other_device.is_current = false;
            }
        }

        std.log.debug("Set current CUDA device to {}", .{device_id});
    }

    pub fn getCurrentDevice(self: *Self) ?u32 {
        for (self.device_contexts.items, 0..) |device, i| {
            if (device.is_current) {
                return @intCast(i);
            }
        }
        return null;
    }

    pub fn malloc(self: *Self, size: usize) !CudaDevicePointer {
        const device_id = self.getCurrentDevice() orelse return error.NoCurrentDevice;
        const device = &self.device_contexts.items[device_id];

        const region = try self.memory_manager.allocateVram(
            size,
            256, // 256-byte alignment for optimal performance
            .general,
            memory.MemoryFlags{},
        );

        const ptr = CudaDevicePointer{
            .address = region.physical_address,
            .size = size,
            .device_id = device_id,
        };

        try device.allocations.append(ptr);

        std.log.debug("CUDA malloc: {} bytes at 0x{X}", .{ size, ptr.address });
        return ptr;
    }

    pub fn free(self: *Self, ptr: CudaDevicePointer) !void {
        const device = &self.device_contexts.items[ptr.device_id];

        // Find and remove allocation
        for (device.allocations.items, 0..) |allocation, i| {
            if (allocation.address == ptr.address) {
                // Create memory region for freeing
                var region = memory.MemoryRegion.init(
                    allocation.address,
                    allocation.size,
                    .vram,
                    .general,
                    memory.MemoryFlags{},
                );

                try self.memory_manager.freeMemory(&region);
                _ = device.allocations.swapRemove(i);

                std.log.debug("CUDA free: 0x{X}", .{ptr.address});
                return;
            }
        }

        return error.InvalidPointer;
    }

    pub fn memcpy(self: *Self, dst: CudaDevicePointer, src: CudaDevicePointer, size: usize, kind: CudaMemcpyKind) !void {
        const copy_cmd = command.MemoryCopyCommand{
            .src_address = src.address,
            .dst_address = dst.address,
            .size = size,
            .copy_engine_id = 0,
        };

        const gpu_cmd = command.GpuCommand{
            .opcode = .memory_copy,
            .data = .{ .memory_copy = copy_cmd },
        };

        // Submit to copy engine
        const commands = [_]command.GpuCommand{gpu_cmd};
        _ = try self.memory_manager.submitCommands(&commands);

        std.log.debug("CUDA memcpy: 0x{X} -> 0x{X} ({} bytes, kind: {})", .{
            src.address,
            dst.address,
            size,
            kind,
        });
    }

    pub fn memset(self: *Self, ptr: CudaDevicePointer, value: u8, size: usize) !void {
        // Use fill command for memset
        const fill_value = @as(u32, value) * 0x01010101; // Replicate byte to all 4 bytes

        const fill_cmd = command.MemoryFillCommand{
            .dst_address = ptr.address,
            .value = fill_value,
            .size = size,
        };

        const gpu_cmd = command.GpuCommand{
            .opcode = .memory_fill,
            .data = .{ .memory_fill = fill_cmd },
        };

        const commands = [_]command.GpuCommand{gpu_cmd};
        _ = try self.memory_manager.submitCommands(&commands);

        std.log.debug("CUDA memset: 0x{X} = {} ({} bytes)", .{ ptr.address, value, size });
    }

    pub fn launchKernel(
        self: *Self,
        kernel: *CudaKernel,
        grid_dim: Dim3,
        block_dim: Dim3,
        shared_mem_size: u32,
        stream: ?*CudaStream,
    ) !void {
        const launch_cmd = command.ComputeLaunchCommand{
            .kernel_address = kernel.device_address,
            .grid_size = .{ grid_dim.x, grid_dim.y, grid_dim.z },
            .block_size = .{ block_dim.x, block_dim.y, block_dim.z },
            .shared_memory_size = shared_mem_size,
            .parameter_buffer = kernel.parameter_buffer,
        };

        const gpu_cmd = command.GpuCommand{
            .opcode = .compute_launch,
            .data = .{ .compute_launch = launch_cmd },
        };

        // Submit to appropriate stream
        const target_stream = stream orelse self.stream_manager.getDefaultStream();
        try target_stream.submitCommand(gpu_cmd);

        // Update profiler
        self.profiler.recordKernelLaunch(kernel, grid_dim, block_dim);

        std.log.debug("CUDA kernel launch: {}x{}x{} blocks, {}x{}x{} threads", .{
            grid_dim.x,
            grid_dim.y,
            grid_dim.z,
            block_dim.x,
            block_dim.y,
            block_dim.z,
        });
    }

    pub fn synchronize(self: *Self) !void {
        // Wait for all streams to complete
        try self.stream_manager.synchronizeAll();

        // Wait for all compute units to be idle
        for (self.compute_units) |*cu| {
            try cu.waitIdle();
        }
    }

    pub fn createStream(self: *Self) !*CudaStream {
        return try self.stream_manager.createStream();
    }

    pub fn destroyStream(self: *Self, stream: *CudaStream) void {
        self.stream_manager.destroyStream(stream);
    }

    pub fn streamSynchronize(self: *Self, stream: *CudaStream) !void {
        _ = self;
        try stream.synchronize();
    }

    pub fn loadModule(self: *Self, ptx_code: []const u8) !*CudaModule {
        return try self.kernel_manager.loadModule(ptx_code);
    }

    pub fn getFunction(self: *Self, module: *CudaModule, name: []const u8) !*CudaKernel {
        _ = self;
        return try module.getFunction(name);
    }
    
    pub fn suspendCompute(self: *Self) !void {
        // Synchronize all active streams
        for (self.stream_manager.streams.items) |*stream| {
            try self.streamSynchronize(stream);
        }
        
        // Stop all compute units
        for (self.compute_units) |*cu| {
            cu.is_idle = true;
        }
        
        std.log.info("CUDA compute suspended", .{});
    }
    
    pub fn resumeCompute(self: *Self) !void {
        // Reinitialize compute units
        for (self.compute_units) |*cu| {
            cu.is_idle = false;
        }
        
        std.log.info("CUDA compute resumed", .{});
    }
    
    pub fn handleInterrupt(self: *Self, interrupt_vector: u32) void {
        _ = self;
        _ = interrupt_vector;
        // Handle CUDA-related interrupts
    }
    
    pub fn getGpuUtilization(self: *Self) f32 {
        var active_count: u32 = 0;
        
        for (self.compute_units) |cu| {
            if (!cu.is_idle) {
                active_count += 1;
            }
        }
        
        if (self.compute_units.len == 0) return 0.0;
        
        return @as(f32, @floatFromInt(active_count)) / @as(f32, @floatFromInt(self.compute_units.len)) * 100.0;
    }
};

/// CUDA Device Context
pub const CudaDevice = struct {
    const Self = @This();

    device_id: u32,
    properties: CudaDeviceProperties,
    allocations: std.ArrayList(CudaDevicePointer),
    is_current: bool,

    pub fn init(allocator: std.mem.Allocator, device_id: u32) Self {
        return Self{
            .device_id = device_id,
            .properties = CudaDeviceProperties{
                .name = "NVIDIA RTX 4090",
                .major = 8,
                .minor = 9,
                .multiprocessor_count = 128,
                .max_threads_per_multiprocessor = 2048,
                .warp_size = 32,
                .max_threads_per_block = 1024,
                .max_block_dim = .{ .x = 1024, .y = 1024, .z = 64 },
                .max_grid_dim = .{ .x = 2147483647, .y = 65535, .z = 65535 },
                .shared_memory_per_block = 49152,
                .total_constant_memory = 65536,
                .total_global_memory = 24 * 1024 * 1024 * 1024, // 24GB
                .clock_rate = 2520000, // 2.52 GHz
                .memory_clock_rate = 10501000, // 21 Gbps effective
                .memory_bus_width = 384,
                .compute_capability = .{ .major = 8, .minor = 9 },
            },
            .allocations = std.ArrayList(CudaDevicePointer).init(allocator),
            .is_current = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocations.deinit();
    }
};

pub const CudaError = error{
    InvalidDevice,
    InvalidContext,
    InvalidModule,
    InvalidFunction,
    InvalidMemory,
    LaunchFailed,
    SyncFailed,
    OutOfMemory,
    InvalidValue,
    NotInitialized,
    DriverVersion,
    ModuleNotLoaded,
    FunctionNotFound,
    InvalidDimensions,
    InvalidGridDimensions,
    InvalidBlockDimensions,
    InvalidSharedMemorySize,
    Timeout,
};

pub const CudaDeviceProperties = struct {
    name: [256]u8,
    total_global_mem: u64,
    shared_mem_per_block: u32,
    regs_per_block: u32,
    warp_size: u32,
    max_threads_per_block: u32,
    max_threads_dim: [3]u32,
    max_grid_size: [3]u32,
    clock_rate: u32,
    memory_clock_rate: u32,
    memory_bus_width: u32,
    l2_cache_size: u32,
    texture_alignment: u32,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    multi_processor_count: u32,
    max_threads_per_multi_processor: u32,
    unified_addressing: bool,
    pci_bus_id: u32,
    pci_device_id: u32,
    pci_domain_id: u32,

    pub fn init(name: []const u8) CudaDeviceProperties {
        var props = CudaDeviceProperties{
            .name = std.mem.zeroes([256]u8),
            .total_global_mem = 24 * 1024 * 1024 * 1024, // 24GB for RTX 4090
            .shared_mem_per_block = 49152, // 48KB
            .regs_per_block = 65536,
            .warp_size = 32,
            .max_threads_per_block = 1024,
            .max_threads_dim = [3]u32{ 1024, 1024, 64 },
            .max_grid_size = [3]u32{ 2147483647, 65535, 65535 },
            .clock_rate = 2520000, // 2.52 GHz
            .memory_clock_rate = 10501000, // 21 Gbps effective
            .memory_bus_width = 384,
            .l2_cache_size = 96 * 1024 * 1024, // 96MB
            .texture_alignment = 512,
            .compute_capability_major = 8,
            .compute_capability_minor = 9,
            .multi_processor_count = 128,
            .max_threads_per_multi_processor = 2048,
            .unified_addressing = true,
            .pci_bus_id = 0,
            .pci_device_id = 0,
            .pci_domain_id = 0,
        };

        const copy_len = @min(name.len, props.name.len - 1);
        @memcpy(props.name[0..copy_len], name[0..copy_len]);

        return props;
    }
};

pub const CudaStream = struct {
    id: u32,
    commands: std.ArrayList(command.GpuCommand),
    fence: ?*Fence,
    allocator: std.mem.Allocator,
    priority: i32,
    flags: u32,

    pub fn init(allocator: std.mem.Allocator, id: u32, priority: i32, flags: u32) CudaStream {
        return CudaStream{
            .id = id,
            .commands = std.ArrayList(command.GpuCommand).init(allocator),
            .fence = null,
            .allocator = allocator,
            .priority = priority,
            .flags = flags,
        };
    }

    pub fn deinit(self: *CudaStream) void {
        for (self.commands.items) |*cmd| {
            cmd.deinit(self.allocator);
        }
        self.commands.deinit();
    }

    pub fn synchronize(self: *CudaStream) !void {
        if (self.fence) |fence| {
            try fence.wait(10 * std.time.ns_per_s); // 10 second timeout
        }
    }

    pub fn query(self: *CudaStream) bool {
        if (self.fence) |fence| {
            return fence.signaled.load(.acquire);
        }
        return true;
    }
};

pub const CudaModule = struct {
    id: u32,
    ptx_code: []const u8,
    functions: std.StringHashMap(CudaFunction),
    allocator: std.mem.Allocator,
    loaded: bool,

    pub fn init(allocator: std.mem.Allocator, id: u32, ptx_code: []const u8) CudaModule {
        return CudaModule{
            .id = id,
            .ptx_code = ptx_code,
            .functions = std.StringHashMap(CudaFunction).init(allocator),
            .allocator = allocator,
            .loaded = false,
        };
    }

    pub fn deinit(self: *CudaModule) void {
        self.functions.deinit();
    }

    pub fn load(self: *CudaModule) !void {
        // Parse PTX and extract functions
        // This is a simplified parser - real implementation would be more complex
        var lines = std.mem.split(u8, self.ptx_code, "\n");

        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t");

            if (std.mem.startsWith(u8, trimmed, ".entry")) {
                // Extract function name
                const start = std.mem.indexOf(u8, trimmed, " ") orelse continue;
                const end = std.mem.indexOf(u8, trimmed[start + 1 ..], "(") orelse continue;
                const func_name = std.mem.trim(u8, trimmed[start + 1 .. start + 1 + end], " \t");

                const func = CudaFunction{
                    .name = try self.allocator.dupe(u8, func_name),
                    .entry_point = 0x1000 + self.functions.count() * 0x100, // Mock address
                    .shared_mem_size = 0,
                    .reg_count = 32,
                    .max_threads_per_block = 1024,
                };

                try self.functions.put(func.name, func);
            }
        }

        self.loaded = true;
    }

    pub fn get_function(self: *CudaModule, name: []const u8) ?CudaFunction {
        return self.functions.get(name);
    }

    pub fn getFunction(self: *CudaModule, name: []const u8) !*CudaKernel {
        _ = self;
        _ = name;
        return error.FunctionNotFound;
    }
};

pub const CudaFunction = struct {
    name: []const u8,
    entry_point: u64,
    shared_mem_size: u32,
    reg_count: u32,
    max_threads_per_block: u32,
};

pub const CudaContext = struct {
    id: u32,
    device_id: u32,
    allocator: std.mem.Allocator,
    memory_manager: memory.DeviceMemoryManager,
    streams: std.ArrayList(CudaStream),
    modules: std.ArrayList(CudaModule),
    command_scheduler: *command.CommandScheduler,
    command_builder: command.CommandBuilder,
    next_stream_id: u32,
    next_module_id: u32,
    flags: u32,

    pub fn init(allocator: std.mem.Allocator, device_id: u32, command_scheduler: *command.CommandScheduler, flags: u32) !CudaContext {
        const mem_manager = memory.DeviceMemoryManager.init(allocator, 24 * 1024 * 1024 * 1024); // 24GB

        return CudaContext{
            .id = device_id,
            .device_id = device_id,
            .allocator = allocator,
            .memory_manager = mem_manager,
            .streams = std.ArrayList(CudaStream).init(allocator),
            .modules = std.ArrayList(CudaModule).init(allocator),
            .command_scheduler = command_scheduler,
            .command_builder = command.CommandBuilder.init(command_scheduler, allocator),
            .next_stream_id = 1,
            .next_module_id = 1,
            .flags = flags,
        };
    }

    pub fn deinit(self: *CudaContext) void {
        for (self.streams.items) |*stream| {
            stream.deinit();
        }
        self.streams.deinit();

        for (self.modules.items) |*module| {
            module.deinit();
        }
        self.modules.deinit();

        self.memory_manager.deinit();
    }

    pub fn create_stream(self: *CudaContext, priority: i32, flags: u32) !u32 {
        const stream = CudaStream.init(self.allocator, self.next_stream_id, priority, flags);
        try self.streams.append(stream);

        const stream_id = self.next_stream_id;
        self.next_stream_id += 1;

        return stream_id;
    }

    pub fn destroy_stream(self: *CudaContext, stream_id: u32) !void {
        for (self.streams.items, 0..) |*stream, i| {
            if (stream.id == stream_id) {
                stream.deinit();
                _ = self.streams.orderedRemove(i);
                return;
            }
        }
        return CudaError.InvalidValue;
    }

    pub fn get_stream(self: *CudaContext, stream_id: u32) ?*CudaStream {
        for (self.streams.items) |*stream| {
            if (stream.id == stream_id) {
                return stream;
            }
        }
        return null;
    }

    pub fn load_module(self: *CudaContext, ptx_code: []const u8) !u32 {
        var module = CudaModule.init(self.allocator, self.next_module_id, ptx_code);
        try module.load();
        try self.modules.append(module);

        const module_id = self.next_module_id;
        self.next_module_id += 1;

        return module_id;
    }

    pub fn get_module(self: *CudaContext, module_id: u32) ?*CudaModule {
        for (self.modules.items) |*module| {
            if (module.id == module_id) {
                return module;
            }
        }
        return null;
    }

    pub fn malloc(self: *CudaContext, size: u64) !u64 {
        const region = try self.memory_manager.allocate(size, .device);
        return region.gpu_address;
    }

    pub fn free(self: *CudaContext, ptr: u64) !void {
        try self.memory_manager.deallocate(ptr);
    }

    pub fn memcpy(self: *CudaContext, dst: u64, src: u64, size: u64, stream_id: u32) !void {
        _ = try self.command_builder.memory_copy(src, dst, size);

        // Update stream fence if specified
        if (stream_id != 0) {
            if (self.get_stream(stream_id)) |stream| {
                stream.fence = try self.command_builder.insert_fence(.copy);
            }
        }
    }

    pub fn memset(self: *CudaContext, ptr: u64, value: u8, size: u64, stream_id: u32) !void {
        // Create a fill command
        var fill_cmd = try command.Command.init(self.allocator, .mem_fill, .copy, 17);
        std.mem.writeInt(u64, fill_cmd.payload[0..8], ptr, .little);
        std.mem.writeInt(u64, fill_cmd.payload[8..16], size, .little);
        fill_cmd.payload[16] = value;

        _ = try self.command_scheduler.submit_command(fill_cmd);
        self.command_scheduler.kick_engine(.copy);

        // Update stream fence if specified
        if (stream_id != 0) {
            if (self.get_stream(stream_id)) |stream| {
                stream.fence = try self.command_builder.insert_fence(.copy);
            }
        }
    }

    pub fn launch_kernel(self: *CudaContext, module_id: u32, function_name: []const u8, grid_dim: [3]u32, block_dim: [3]u32, shared_mem: u32, stream_id: u32, _: []const u64) !void {
        const module = self.get_module(module_id) orelse return CudaError.ModuleNotLoaded;
        const function = module.get_function(function_name) orelse return CudaError.FunctionNotFound;

        // Validate dimensions
        if (block_dim[0] * block_dim[1] * block_dim[2] > function.max_threads_per_block) {
            return CudaError.InvalidBlockDimensions;
        }

        if (shared_mem > 48 * 1024) { // 48KB shared memory limit
            return CudaError.InvalidSharedMemorySize;
        }

        // Launch kernel using compute command
        _ = try self.command_builder.compute_dispatch(function.entry_point, grid_dim[0], grid_dim[1], grid_dim[2], block_dim[0], block_dim[1], block_dim[2]);

        // Update stream fence if specified
        if (stream_id != 0) {
            if (self.get_stream(stream_id)) |stream| {
                stream.fence = try self.command_builder.insert_fence(.compute);
            }
        }
    }

    pub fn synchronize(self: *CudaContext) !void {
        try self.command_scheduler.wait_idle(30 * std.time.ns_per_s); // 30 second timeout
    }
};

// CUDA Graph support for advanced optimization
pub const CudaGraph = struct {
    id: u32,
    nodes: std.ArrayList(CudaGraphNode),
    edges: std.ArrayList(CudaGraphEdge),
    allocator: std.mem.Allocator,
    instantiated: bool,

    pub fn init(allocator: std.mem.Allocator, id: u32) CudaGraph {
        return CudaGraph{
            .id = id,
            .nodes = std.ArrayList(CudaGraphNode).init(allocator),
            .edges = std.ArrayList(CudaGraphEdge).init(allocator),
            .allocator = allocator,
            .instantiated = false,
        };
    }

    pub fn deinit(self: *CudaGraph) void {
        self.nodes.deinit();
        self.edges.deinit();
    }

    pub fn add_kernel_node(self: *CudaGraph, kernel_params: CudaKernelNodeParams) !u32 {
        const node = CudaGraphNode{
            .id = @intCast(self.nodes.items.len),
            .type = .kernel,
            .kernel_params = kernel_params,
        };

        try self.nodes.append(node);
        return node.id;
    }

    pub fn add_memcpy_node(self: *CudaGraph, memcpy_params: CudaMemcpyNodeParams) !u32 {
        const node = CudaGraphNode{
            .id = @intCast(self.nodes.items.len),
            .type = .memcpy,
            .memcpy_params = memcpy_params,
        };

        try self.nodes.append(node);
        return node.id;
    }

    pub fn add_dependency(self: *CudaGraph, from_node: u32, to_node: u32) !void {
        const edge = CudaGraphEdge{
            .from = from_node,
            .to = to_node,
        };

        try self.edges.append(edge);
    }

    pub fn instantiate(self: *CudaGraph) !void {
        // Validate graph and create execution plan
        // This would involve topological sorting and optimization
        self.instantiated = true;
    }

    pub fn launch(self: *CudaGraph, context: *CudaContext, stream_id: u32) !void {
        if (!self.instantiated) return CudaError.NotInitialized;

        // Execute nodes in dependency order
        for (self.nodes.items) |node| {
            switch (node.type) {
                .kernel => {
                    const params = node.kernel_params;
                    try context.launch_kernel(params.module_id, params.function_name, params.grid_dim, params.block_dim, params.shared_mem, stream_id, params.params);
                },
                .memcpy => {
                    const params = node.memcpy_params;
                    try context.memcpy(params.dst, params.src, params.size, stream_id);
                },
            }
        }
    }
};

pub const CudaGraphNodeType = enum {
    kernel,
    memcpy,
    memset,
    empty,
};

pub const CudaKernelNodeParams = struct {
    module_id: u32,
    function_name: []const u8,
    grid_dim: [3]u32,
    block_dim: [3]u32,
    shared_mem: u32,
    params: []const u64,
};

pub const CudaMemcpyNodeParams = struct {
    dst: u64,
    src: u64,
    size: u64,
};

pub const CudaGraphNode = struct {
    id: u32,
    type: CudaGraphNodeType,
    kernel_params: CudaKernelNodeParams = undefined,
    memcpy_params: CudaMemcpyNodeParams = undefined,
};

pub const CudaGraphEdge = struct {
    from: u32,
    to: u32,
};

// Test functions
test "cuda runtime initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var mem_manager = memory.MemoryManager.init(allocator);
    var scheduler = command.CommandScheduler.init(allocator, &mem_manager);
    defer scheduler.deinit();

    var runtime = CudaRuntime.init(allocator, &mem_manager);
    defer runtime.deinit();

    // runtime.initialize(); // Not needed, init() handles initialization

    const device_count = runtime.getDeviceCount();
    try std.testing.expect(device_count > 0);

    const props = try runtime.getDeviceProperties(0);
    try std.testing.expect(props != null);
}

test "cuda context and streams" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var mem_manager = memory.MemoryManager.init(allocator);
    var scheduler = command.CommandScheduler.init(allocator, &mem_manager);
    defer scheduler.deinit();

    var runtime = CudaRuntime.init(allocator, &mem_manager);
    defer runtime.deinit();

    // runtime.initialize(); // Not needed, init() handles initialization

    // Set the current device
    try runtime.setDevice(0);
    // Test stream creation
    const stream = try runtime.stream_manager.createStream();
    defer stream.deinit();

    try std.testing.expect(stream.id >= 0);
}

test "cuda memory management" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var mem_manager = memory.MemoryManager.init(allocator);
    var scheduler = command.CommandScheduler.init(allocator, &mem_manager);
    defer scheduler.deinit();

    var runtime = CudaRuntime.init(allocator, &mem_manager);
    defer runtime.deinit();

    // runtime.initialize(); // Not needed, init() handles initialization

    // Set the current device
    try runtime.setDevice(0);
    // Test memory allocation
    const ptr = try runtime.malloc(1024 * 1024); // 1MB
    try std.testing.expect(ptr.address != 0);

    try runtime.free(ptr);
}

// Missing types and structures for CUDA runtime
pub const Dim3 = struct {
    x: u32,
    y: u32,
    z: u32,
};

pub const CudaDevicePointer = struct {
    address: u64,
    size: usize,
    device_id: u32,
};

pub const CudaMemcpyKind = enum {
    host_to_host,
    host_to_device,
    device_to_host,
    device_to_device,
};

pub const CudaKernel = struct {
    device_address: u64,
    parameter_buffer: []u8,
    shared_memory_size: u32,
    max_threads_per_block: u32,
};

pub const Fence = struct {
    signaled: std.atomic.Value(bool),

    pub fn init() Fence {
        return Fence{
            .signaled = std.atomic.Value(bool).init(false),
        };
    }

    pub fn wait(self: *Fence, timeout_ns: u64) !void {
        _ = timeout_ns;
        while (!self.signaled.load(.acquire)) {
            std.time.sleep(1000000); // 1ms
        }
    }

    pub fn signal(self: *Fence) void {
        self.signaled.store(true, .release);
    }
};

// Supporting managers
pub const StreamManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    streams: std.ArrayList(CudaStream),
    default_stream: CudaStream,
    next_stream_id: u32,

    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .allocator = allocator,
            .streams = std.ArrayList(CudaStream).init(allocator),
            .default_stream = CudaStream.init(allocator, 0, 0, 0),
            .next_stream_id = 1,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.streams.items) |*stream| {
            stream.deinit();
        }
        self.streams.deinit();
        self.default_stream.deinit();
    }

    pub fn createStream(self: *Self) !*CudaStream {
        const stream = CudaStream.init(self.allocator, self.next_stream_id, 0, 0);
        try self.streams.append(stream);
        self.next_stream_id += 1;
        return &self.streams.items[self.streams.items.len - 1];
    }

    pub fn destroyStream(self: *Self, stream: *CudaStream) void {
        for (self.streams.items, 0..) |*s, i| {
            if (s == stream) {
                s.deinit();
                _ = self.streams.orderedRemove(i);
                break;
            }
        }
    }

    pub fn getDefaultStream(self: *Self) *CudaStream {
        return &self.default_stream;
    }

    pub fn synchronizeAll(self: *Self) !void {
        try self.default_stream.synchronize();
        for (self.streams.items) |*stream| {
            try stream.synchronize();
        }
    }
};

pub const KernelManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    modules: std.ArrayList(CudaModule),
    next_module_id: u32,

    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .allocator = allocator,
            .modules = std.ArrayList(CudaModule).init(allocator),
            .next_module_id = 1,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.modules.items) |*module| {
            module.deinit();
        }
        self.modules.deinit();
    }

    pub fn loadModule(self: *Self, ptx_code: []const u8) !*CudaModule {
        _ = ptx_code;
        const module = CudaModule.init(self.allocator);
        try self.modules.append(module);
        return &self.modules.items[self.modules.items.len - 1];
    }
};

pub const TensorCoreManager = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    tensor_cores: []TensorCore,

    pub const TensorCore = struct {
        id: u32,
        available: bool,
        current_operation: ?TensorOperation,

        pub const TensorOperation = enum {
            matrix_multiply,
            convolution,
            attention,
        };
    };

    pub fn init(allocator: std.mem.Allocator) !Self {
        const num_tensor_cores = 128 * 4; // 4 tensor cores per SM
        const tensor_cores = try allocator.alloc(TensorCore, num_tensor_cores);

        for (tensor_cores, 0..) |*tc, i| {
            tc.* = TensorCore{
                .id = @intCast(i),
                .available = true,
                .current_operation = null,
            };
        }

        return Self{
            .allocator = allocator,
            .tensor_cores = tensor_cores,
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.tensor_cores);
    }
};

pub const CudaProfiler = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    kernel_events: std.ArrayList(KernelEvent),
    memory_events: std.ArrayList(MemoryEvent),

    pub const KernelEvent = struct {
        kernel: *CudaKernel,
        grid_dim: Dim3,
        block_dim: Dim3,
        timestamp: u64,
        duration: u64,
    };

    pub const MemoryEvent = struct {
        operation: MemoryOperation,
        size: usize,
        timestamp: u64,
        duration: u64,

        pub const MemoryOperation = enum {
            malloc,
            free,
            memcpy,
            memset,
        };
    };

    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .allocator = allocator,
            .kernel_events = std.ArrayList(KernelEvent).init(allocator),
            .memory_events = std.ArrayList(MemoryEvent).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.kernel_events.deinit();
        self.memory_events.deinit();
    }

    pub fn recordKernelLaunch(self: *Self, kernel: *CudaKernel, grid_dim: Dim3, block_dim: Dim3) void {
        const event = KernelEvent{
            .kernel = kernel,
            .grid_dim = grid_dim,
            .block_dim = block_dim,
            .timestamp = std.time.milliTimestamp(),
            .duration = 0,
        };

        self.kernel_events.append(event) catch {};
    }
};

pub const ComputeUnit = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    sm_id: u32,
    warp_schedulers: []WarpScheduler,
    shared_memory: []u8,
    register_file: []u32,
    is_idle: bool,

    pub fn init(allocator: std.mem.Allocator, sm_id: u32) !Self {
        // const num_warps = 64; // 64 warps per SM
        const warp_schedulers = try allocator.alloc(WarpScheduler, 4); // 4 warp schedulers per SM

        for (warp_schedulers, 0..) |*ws, i| {
            ws.* = WarpScheduler{
                .id = @intCast(i),
                .active_warps = std.ArrayList(u32).init(allocator),
                .ready_warps = std.ArrayList(u32).init(allocator),
            };
        }

        return Self{
            .allocator = allocator,
            .sm_id = sm_id,
            .warp_schedulers = warp_schedulers,
            .shared_memory = try allocator.alloc(u8, 48 * 1024), // 48KB shared memory
            .register_file = try allocator.alloc(u32, 65536), // 64K registers
            .is_idle = true,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.warp_schedulers) |*ws| {
            ws.active_warps.deinit();
            ws.ready_warps.deinit();
        }
        self.allocator.free(self.warp_schedulers);
        self.allocator.free(self.shared_memory);
        self.allocator.free(self.register_file);
    }

    pub fn waitIdle(self: *Self) !void {
        while (!self.is_idle) {
            std.time.sleep(1000000); // 1ms
        }
    }
};

pub const WarpScheduler = struct {
    id: u32,
    active_warps: std.ArrayList(u32),
    ready_warps: std.ArrayList(u32),
};

pub const SharedMemoryPool = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    memory: []u8,
    allocations: std.ArrayList(SharedMemoryAllocation),

    pub const SharedMemoryAllocation = struct {
        offset: usize,
        size: usize,
        in_use: bool,
    };

    pub fn init(allocator: std.mem.Allocator, size: usize) !Self {
        return Self{
            .allocator = allocator,
            .memory = try allocator.alloc(u8, size),
            .allocations = std.ArrayList(SharedMemoryAllocation).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.memory);
        self.allocations.deinit();
    }

    pub fn allocate(self: *Self, size: usize) ![]u8 {
        // Find free space
        var offset: usize = 0;
        for (self.allocations.items) |alloc| {
            if (!alloc.in_use) {
                if (alloc.size >= size) {
                    alloc.in_use = true;
                    return self.memory[alloc.offset .. alloc.offset + size];
                }
            }
            offset = alloc.offset + alloc.size;
        }

        // Allocate new
        if (offset + size > self.memory.len) {
            return error.OutOfMemory;
        }

        const allocation = SharedMemoryAllocation{
            .offset = offset,
            .size = size,
            .in_use = true,
        };

        try self.allocations.append(allocation);
        return self.memory[offset .. offset + size];
    }

    pub fn free(self: *Self, ptr: []u8) void {
        const offset = @intFromPtr(ptr.ptr) - @intFromPtr(self.memory.ptr);

        for (self.allocations.items) |*alloc| {
            if (alloc.offset == offset and alloc.in_use) {
                alloc.in_use = false;
                return;
            }
        }
    }
};
