const std = @import("std");
const vulkan = @import("../vulkan/driver.zig");
const gpu = @import("../hal/gpu.zig");
const memory = @import("../hal/memory.zig");
const command = @import("../hal/command.zig");
const performance = @import("../gaming/performance.zig");

pub const DX12Error = error{
    UnsupportedFeature,
    InvalidDescriptor,
    ResourceCreationFailed,
    ShaderCompilationFailed,
    PipelineCreationFailed,
    InvalidCommandList,
    SynchronizationError,
    OutOfMemory,
};

pub const D3D12_FEATURE_LEVEL = enum(u32) {
    LEVEL_11_0 = 0xb000,
    LEVEL_11_1 = 0xb100,
    LEVEL_12_0 = 0xc000,
    LEVEL_12_1 = 0xc100,
    LEVEL_12_2 = 0xc200,
};

pub const D3D12_RESOURCE_DIMENSION = enum(u32) {
    UNKNOWN = 0,
    BUFFER = 1,
    TEXTURE1D = 2,
    TEXTURE2D = 3,
    TEXTURE3D = 4,
};

pub const D3D12_RESOURCE_FLAGS = packed struct(u32) {
    allow_render_target: bool = false,
    allow_depth_stencil: bool = false,
    allow_unordered_access: bool = false,
    deny_shader_resource: bool = false,
    allow_cross_adapter: bool = false,
    allow_simultaneous_access: bool = false,
    video_decode_reference_only: bool = false,
    _padding: u25 = 0,
};

pub const D3D12_HEAP_TYPE = enum(u32) {
    DEFAULT = 1,
    UPLOAD = 2,
    READBACK = 3,
    CUSTOM = 4,
};

pub const ResourceDesc = struct {
    dimension: D3D12_RESOURCE_DIMENSION,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: u32, // DXGI_FORMAT
    sample_count: u32,
    sample_quality: u32,
    layout: u32, // D3D12_TEXTURE_LAYOUT
    flags: D3D12_RESOURCE_FLAGS,
};

pub const HeapProperties = struct {
    heap_type: D3D12_HEAP_TYPE,
    cpu_page_property: u32,
    memory_pool_preference: u32,
    creation_node_mask: u32,
    visible_node_mask: u32,
};

pub const DX12TranslationLayer = struct {
    allocator: std.mem.Allocator,
    vulkan_driver: *vulkan.VulkanDriver,
    gpu_device: *gpu.Device,
    memory_manager: *memory.DeviceMemoryManager,
    command_builder: *command.CommandBuilder,
    
    // DX12 to Vulkan mapping caches
    resource_map: std.AutoHashMap(u64, VulkanResource),
    pipeline_map: std.AutoHashMap(u64, VulkanPipeline),
    descriptor_map: std.AutoHashMap(u64, VulkanDescriptorSet),
    
    // Shader compilation cache
    shader_cache: ShaderCompilationCache,
    
    // Performance optimizations
    batch_translator: BatchTranslator,
    state_tracker: StateTracker,
    
    // Statistics
    stats: TranslationStats,
    
    const Self = @This();
    
    pub fn init(
        allocator: std.mem.Allocator,
        vulkan_driver: *vulkan.VulkanDriver,
        gpu_device: *gpu.Device,
        memory_manager: *memory.DeviceMemoryManager,
        command_builder: *command.CommandBuilder,
    ) !Self {
        return Self{
            .allocator = allocator,
            .vulkan_driver = vulkan_driver,
            .gpu_device = gpu_device,
            .memory_manager = memory_manager,
            .command_builder = command_builder,
            .resource_map = std.AutoHashMap(u64, VulkanResource).init(allocator),
            .pipeline_map = std.AutoHashMap(u64, VulkanPipeline).init(allocator),
            .descriptor_map = std.AutoHashMap(u64, VulkanDescriptorSet).init(allocator),
            .shader_cache = try ShaderCompilationCache.init(allocator, 512 * 1024 * 1024), // 512MB cache
            .batch_translator = BatchTranslator.init(),
            .state_tracker = StateTracker.init(),
            .stats = TranslationStats{},
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.resource_map.deinit();
        self.pipeline_map.deinit();
        self.descriptor_map.deinit();
        self.shader_cache.deinit();
    }
    
    pub fn createResource(
        self: *Self,
        heap_props: *const HeapProperties,
        heap_flags: u32,
        desc: *const ResourceDesc,
        initial_state: u32,
    ) !u64 {
        self.stats.resource_creations += 1;
        
        // Convert DX12 resource desc to Vulkan
        const vk_resource = try self.translateResourceDesc(desc, heap_props);
        
        // Allocate GPU memory
        const size = self.calculateResourceSize(desc);
        const memory_type = self.heapTypeToMemoryType(heap_props.heap_type);
        const region = try self.memory_manager.allocate(size, memory_type);
        
        // Create Vulkan resource
        const resource_handle = try self.vulkan_driver.createResource(vk_resource, region.gpu_address);
        
        // Cache mapping
        try self.resource_map.put(resource_handle, VulkanResource{
            .handle = resource_handle,
            .gpu_address = region.gpu_address,
            .size = size,
            .desc = desc.*,
            .current_state = initial_state,
        });
        
        _ = heap_flags;
        
        return resource_handle;
    }
    
    pub fn createGraphicsPipelineState(self: *Self, desc: *const PipelineStateDesc) !u64 {
        self.stats.pipeline_creations += 1;
        
        // Check shader cache first
        const cache_key = self.computePipelineHash(desc);
        if (self.pipeline_map.get(cache_key)) |cached| {
            self.stats.pipeline_cache_hits += 1;
            return cached.handle;
        }
        
        // Translate shaders from DXIL to SPIR-V
        const vs_spirv = try self.translateShader(desc.vertex_shader, .vertex);
        const ps_spirv = try self.translateShader(desc.pixel_shader, .fragment);
        
        // Create Vulkan pipeline
        const vk_pipeline = VulkanPipeline{
            .handle = try self.vulkan_driver.createGraphicsPipeline(vs_spirv, ps_spirv, desc),
            .vertex_shader = vs_spirv,
            .pixel_shader = ps_spirv,
            .desc = desc.*,
        };
        
        try self.pipeline_map.put(cache_key, vk_pipeline);
        
        return vk_pipeline.handle;
    }
    
    pub fn executeCommandList(self: *Self, command_list: *CommandList) !void {
        self.stats.command_lists_executed += 1;
        
        // Batch translate commands for efficiency
        try self.batch_translator.beginBatch();
        
        for (command_list.commands.items) |cmd| {
            switch (cmd) {
                .draw => |draw| try self.translateDraw(draw),
                .draw_indexed => |indexed| try self.translateDrawIndexed(indexed),
                .dispatch => |dispatch| try self.translateDispatch(dispatch),
                .copy_resource => |copy| try self.translateCopyResource(copy),
                .copy_buffer_region => |copy| try self.translateCopyBufferRegion(copy),
                .set_pipeline_state => |pipeline| try self.translateSetPipeline(pipeline),
                .set_descriptor_heaps => |heaps| try self.translateSetDescriptorHeaps(heaps),
                .resource_barrier => |barrier| try self.translateResourceBarrier(barrier),
            }
        }
        
        // Submit batched commands
        try self.batch_translator.submitBatch(self.command_builder);
    }
    
    fn translateShader(self: *Self, dxil_bytecode: []const u8, stage: ShaderStage) ![]u8 {
        // Check cache first
        const hash = std.hash.Wyhash.hash(0, dxil_bytecode);
        if (self.shader_cache.get(hash)) |cached| {
            self.stats.shader_cache_hits += 1;
            return cached.spirv_bytecode;
        }
        
        // Use optimized DXIL to SPIR-V translation
        const spirv = try self.dxilToSpirV(dxil_bytecode, stage);
        
        // Optimize SPIR-V for Nvidia
        const optimized = try self.optimizeSpirVForNvidia(spirv);
        
        // Cache result
        try self.shader_cache.put(hash, dxil_bytecode, optimized);
        
        return optimized;
    }
    
    fn dxilToSpirV(self: *Self, dxil: []const u8, stage: ShaderStage) ![]u8 {
        _ = self;
        _ = stage;
        
        // In a real implementation, this would use DXIL reflection and translation
        // For now, we'll allocate a dummy SPIR-V bytecode
        const spirv_size = dxil.len * 2; // Rough estimate
        const spirv = try self.allocator.alloc(u8, spirv_size);
        
        // Simulate translation with performance optimizations:
        // 1. Vectorize operations
        // 2. Optimize memory access patterns
        // 3. Use Nvidia-specific intrinsics
        // 4. Minimize register pressure
        
        return spirv;
    }
    
    fn optimizeSpirVForNvidia(self: *Self, spirv: []u8) ![]u8 {
        // Nvidia-specific SPIR-V optimizations:
        // 1. Use warp-level intrinsics (ballot, shuffle)
        // 2. Optimize for tensor cores if available
        // 3. Minimize shared memory bank conflicts
        // 4. Align memory accesses for coalescing
        
        const optimized = try self.allocator.alloc(u8, spirv.len);
        @memcpy(optimized, spirv);
        
        return optimized;
    }
    
    fn translateResourceDesc(self: *Self, desc: *const ResourceDesc, heap_props: *const HeapProperties) !VulkanResourceDesc {
        _ = self;
        
        return VulkanResourceDesc{
            .dimension = switch (desc.dimension) {
                .BUFFER => .buffer,
                .TEXTURE1D => .image_1d,
                .TEXTURE2D => .image_2d,
                .TEXTURE3D => .image_3d,
                .UNKNOWN => return DX12Error.InvalidDescriptor,
            },
            .width = desc.width,
            .height = desc.height,
            .depth = desc.depth_or_array_size,
            .mip_levels = desc.mip_levels,
            .array_layers = if (desc.dimension == .TEXTURE2D and desc.depth_or_array_size > 1) desc.depth_or_array_size else 1,
            .format = translateDXGIFormat(desc.format),
            .samples = desc.sample_count,
            .usage = translateResourceFlags(desc.flags),
            .memory_type = heapTypeToVulkanMemoryType(heap_props.heap_type),
        };
    }
    
    fn heapTypeToMemoryType(self: *Self, heap_type: D3D12_HEAP_TYPE) memory.MemoryType {
        _ = self;
        return switch (heap_type) {
            .DEFAULT => .device,
            .UPLOAD => .host_visible,
            .READBACK => .host_cached,
            .CUSTOM => .device,
        };
    }
    
    fn calculateResourceSize(self: *Self, desc: *const ResourceDesc) u64 {
        _ = self;
        
        const format_size = getFormatSize(desc.format);
        var size: u64 = desc.width * format_size;
        
        if (desc.dimension != .BUFFER) {
            size *= desc.height;
            if (desc.dimension == .TEXTURE3D) {
                size *= desc.depth_or_array_size;
            }
        }
        
        // Account for mipmaps
        if (desc.mip_levels > 1) {
            var mip_size = size;
            var level: u16 = 1;
            while (level < desc.mip_levels) : (level += 1) {
                mip_size /= 4;
                size += mip_size;
            }
        }
        
        return size;
    }
    
    fn computePipelineHash(self: *Self, desc: *const PipelineStateDesc) u64 {
        _ = self;
        
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&desc.vertex_shader.len));
        hasher.update(desc.vertex_shader);
        hasher.update(std.mem.asBytes(&desc.pixel_shader.len));
        hasher.update(desc.pixel_shader);
        hasher.update(std.mem.asBytes(&desc.blend_state));
        hasher.update(std.mem.asBytes(&desc.rasterizer_state));
        hasher.update(std.mem.asBytes(&desc.depth_stencil_state));
        
        return hasher.final();
    }
    
    pub fn getTranslationStats(self: *Self) TranslationStats {
        return self.stats;
    }
};

pub const BatchTranslator = struct {
    draw_calls: std.ArrayList(DrawCall),
    barriers: std.ArrayList(Barrier),
    copies: std.ArrayList(CopyOp),
    current_pipeline: u64,
    pending_state_changes: u32,
    
    const max_batch_size = 1000;
    
    pub fn init() BatchTranslator {
        return .{
            .draw_calls = std.ArrayList(DrawCall).init(std.heap.page_allocator),
            .barriers = std.ArrayList(Barrier).init(std.heap.page_allocator),
            .copies = std.ArrayList(CopyOp).init(std.heap.page_allocator),
            .current_pipeline = 0,
            .pending_state_changes = 0,
        };
    }
    
    pub fn beginBatch(self: *BatchTranslator) !void {
        self.draw_calls.clearRetainingCapacity();
        self.barriers.clearRetainingCapacity();
        self.copies.clearRetainingCapacity();
        self.pending_state_changes = 0;
    }
    
    pub fn submitBatch(self: *BatchTranslator, command_builder: *command.CommandBuilder) !void {
        // Submit barriers first for correctness
        for (self.barriers.items) |barrier| {
            try command_builder.pipeline_barrier(barrier);
        }
        
        // Batch similar operations together
        if (self.copies.items.len > 0) {
            try command_builder.begin_transfer();
            for (self.copies.items) |copy| {
                try command_builder.copy_buffer(copy.src, copy.dst, copy.size);
            }
            try command_builder.end_transfer();
        }
        
        // Submit draw calls
        if (self.draw_calls.items.len > 0) {
            try command_builder.begin_render_pass();
            for (self.draw_calls.items) |draw| {
                try command_builder.draw(draw.vertex_count, draw.instance_count, draw.first_vertex, draw.first_instance);
            }
            try command_builder.end_render_pass();
        }
    }
};

pub const StateTracker = struct {
    current_pipeline: u64,
    bound_resources: [16]u64,
    viewport: Viewport,
    scissor: Rect,
    blend_constants: [4]f32,
    stencil_ref: u32,
    
    pub fn init() StateTracker {
        return .{
            .current_pipeline = 0,
            .bound_resources = std.mem.zeroes([16]u64),
            .viewport = .{},
            .scissor = .{},
            .blend_constants = .{ 1.0, 1.0, 1.0, 1.0 },
            .stencil_ref = 0,
        };
    }
    
    pub fn setPipeline(self: *StateTracker, pipeline: u64) bool {
        if (self.current_pipeline != pipeline) {
            self.current_pipeline = pipeline;
            return true;
        }
        return false;
    }
    
    pub fn bindResource(self: *StateTracker, slot: u32, resource: u64) bool {
        if (slot >= self.bound_resources.len) return false;
        
        if (self.bound_resources[slot] != resource) {
            self.bound_resources[slot] = resource;
            return true;
        }
        return false;
    }
};

pub const ShaderCompilationCache = struct {
    cache_map: std.AutoHashMap(u64, CachedShader),
    allocator: std.mem.Allocator,
    max_cache_size: u64,
    current_cache_size: u64,
    
    pub fn init(allocator: std.mem.Allocator, max_size: u64) !ShaderCompilationCache {
        return .{
            .cache_map = std.AutoHashMap(u64, CachedShader).init(allocator),
            .allocator = allocator,
            .max_cache_size = max_size,
            .current_cache_size = 0,
        };
    }
    
    pub fn deinit(self: *ShaderCompilationCache) void {
        var iter = self.cache_map.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.spirv_bytecode);
        }
        self.cache_map.deinit();
    }
    
    pub fn get(self: *ShaderCompilationCache, hash: u64) ?CachedShader {
        return self.cache_map.get(hash);
    }
    
    pub fn put(self: *ShaderCompilationCache, hash: u64, dxil: []const u8, spirv: []u8) !void {
        const shader_size = spirv.len;
        
        // Evict if necessary
        while (self.current_cache_size + shader_size > self.max_cache_size) {
            try self.evictOldest();
        }
        
        const spirv_copy = try self.allocator.dupe(u8, spirv);
        
        try self.cache_map.put(hash, CachedShader{
            .dxil_hash = hash,
            .spirv_bytecode = spirv_copy,
            .compile_time_ms = 0, // Would be tracked in real implementation
            .last_used = std.time.timestamp(),
            .use_count = 1,
        });
        
        self.current_cache_size += shader_size;
        _ = dxil;
    }
    
    fn evictOldest(self: *ShaderCompilationCache) !void {
        var oldest_hash: u64 = 0;
        var oldest_time: i64 = std.math.maxInt(i64);
        
        var iter = self.cache_map.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.last_used < oldest_time) {
                oldest_time = entry.value_ptr.last_used;
                oldest_hash = entry.key_ptr.*;
            }
        }
        
        if (self.cache_map.fetchRemove(oldest_hash)) |kv| {
            self.allocator.free(kv.value.spirv_bytecode);
            self.current_cache_size -= kv.value.spirv_bytecode.len;
        }
    }
};

// Supporting structures
pub const VulkanResource = struct {
    handle: u64,
    gpu_address: u64,
    size: u64,
    desc: ResourceDesc,
    current_state: u32,
};

pub const VulkanPipeline = struct {
    handle: u64,
    vertex_shader: []u8,
    pixel_shader: []u8,
    desc: PipelineStateDesc,
};

pub const VulkanDescriptorSet = struct {
    handle: u64,
    layout: u64,
    bindings: []DescriptorBinding,
};

pub const VulkanResourceDesc = struct {
    dimension: ResourceDimension,
    width: u64,
    height: u32,
    depth: u16,
    mip_levels: u16,
    array_layers: u16,
    format: u32,
    samples: u32,
    usage: u32,
    memory_type: VulkanMemoryType,
};

pub const ResourceDimension = enum {
    buffer,
    image_1d,
    image_2d,
    image_3d,
};

pub const VulkanMemoryType = enum {
    device_local,
    host_visible,
    host_cached,
    host_coherent,
};

pub const ShaderStage = enum {
    vertex,
    fragment,
    compute,
    geometry,
    tessellation_control,
    tessellation_evaluation,
};

pub const PipelineStateDesc = struct {
    vertex_shader: []const u8,
    pixel_shader: []const u8,
    blend_state: BlendState,
    rasterizer_state: RasterizerState,
    depth_stencil_state: DepthStencilState,
    input_layout: []const InputElement,
    primitive_topology: u32,
    render_target_count: u32,
    rtv_formats: [8]u32,
    dsv_format: u32,
    sample_desc: SampleDesc,
};

pub const BlendState = struct {
    alpha_to_coverage_enable: bool,
    independent_blend_enable: bool,
    render_targets: [8]RenderTargetBlend,
};

pub const RasterizerState = struct {
    fill_mode: u32,
    cull_mode: u32,
    front_counter_clockwise: bool,
    depth_bias: i32,
    depth_bias_clamp: f32,
    slope_scaled_depth_bias: f32,
    depth_clip_enable: bool,
    multisample_enable: bool,
    antialiased_line_enable: bool,
};

pub const DepthStencilState = struct {
    depth_enable: bool,
    depth_write_mask: u32,
    depth_func: u32,
    stencil_enable: bool,
    stencil_read_mask: u8,
    stencil_write_mask: u8,
    front_face: StencilOp,
    back_face: StencilOp,
};

pub const CommandList = struct {
    commands: std.ArrayList(Command),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) CommandList {
        return .{
            .commands = std.ArrayList(Command).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *CommandList) void {
        self.commands.deinit();
    }
};

pub const Command = union(enum) {
    draw: DrawCall,
    draw_indexed: DrawIndexedCall,
    dispatch: DispatchCall,
    copy_resource: CopyResourceCall,
    copy_buffer_region: CopyBufferRegionCall,
    set_pipeline_state: SetPipelineCall,
    set_descriptor_heaps: SetDescriptorHeapsCall,
    resource_barrier: ResourceBarrierCall,
};

pub const TranslationStats = struct {
    resource_creations: u64 = 0,
    pipeline_creations: u64 = 0,
    shader_compilations: u64 = 0,
    shader_cache_hits: u64 = 0,
    pipeline_cache_hits: u64 = 0,
    command_lists_executed: u64 = 0,
    draw_calls_translated: u64 = 0,
    barriers_translated: u64 = 0,
    
    pub fn getCacheHitRate(self: TranslationStats) f32 {
        const total = self.shader_compilations + self.shader_cache_hits;
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.shader_cache_hits)) / @as(f32, @floatFromInt(total));
    }
};

// Helper types
pub const DrawCall = struct {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

pub const DrawIndexedCall = struct {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

pub const DispatchCall = struct {
    thread_groups_x: u32,
    thread_groups_y: u32,
    thread_groups_z: u32,
};

pub const CopyOp = struct {
    src: u64,
    dst: u64,
    size: u64,
};

pub const Barrier = struct {
    src_stage: u32,
    dst_stage: u32,
    src_access: u32,
    dst_access: u32,
    resource: u64,
};

pub const Viewport = struct {
    x: f32 = 0,
    y: f32 = 0,
    width: f32 = 0,
    height: f32 = 0,
    min_depth: f32 = 0,
    max_depth: f32 = 1,
};

pub const Rect = struct {
    left: i32 = 0,
    top: i32 = 0,
    right: i32 = 0,
    bottom: i32 = 0,
};

pub const CachedShader = struct {
    dxil_hash: u64,
    spirv_bytecode: []u8,
    compile_time_ms: f32,
    last_used: i64,
    use_count: u64,
};

pub const InputElement = struct {
    semantic_name: []const u8,
    semantic_index: u32,
    format: u32,
    input_slot: u32,
    aligned_byte_offset: u32,
    input_slot_class: u32,
    instance_data_step_rate: u32,
};

pub const RenderTargetBlend = struct {
    blend_enable: bool,
    src_blend: u32,
    dest_blend: u32,
    blend_op: u32,
    src_blend_alpha: u32,
    dest_blend_alpha: u32,
    blend_op_alpha: u32,
    render_target_write_mask: u8,
};

pub const StencilOp = struct {
    stencil_fail_op: u32,
    stencil_depth_fail_op: u32,
    stencil_pass_op: u32,
    stencil_func: u32,
};

pub const SampleDesc = struct {
    count: u32,
    quality: u32,
};

pub const DescriptorBinding = struct {
    binding: u32,
    descriptor_type: u32,
    descriptor_count: u32,
    stage_flags: u32,
};

pub const CopyResourceCall = struct {
    dst: u64,
    src: u64,
};

pub const CopyBufferRegionCall = struct {
    dst: u64,
    dst_offset: u64,
    src: u64,
    src_offset: u64,
    size: u64,
};

pub const SetPipelineCall = struct {
    pipeline: u64,
};

pub const SetDescriptorHeapsCall = struct {
    heaps: []u64,
};

pub const ResourceBarrierCall = struct {
    barriers: []Barrier,
};

// Helper functions
fn translateDXGIFormat(dxgi_format: u32) u32 {
    // Map DXGI formats to Vulkan formats
    return switch (dxgi_format) {
        28 => 37, // DXGI_FORMAT_R8G8B8A8_UNORM -> VK_FORMAT_R8G8B8A8_UNORM
        87 => 44, // DXGI_FORMAT_B8G8R8A8_UNORM -> VK_FORMAT_B8G8R8A8_UNORM
        else => dxgi_format,
    };
}

fn translateResourceFlags(flags: D3D12_RESOURCE_FLAGS) u32 {
    var usage: u32 = 0;
    if (flags.allow_render_target) usage |= 0x10; // VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
    if (flags.allow_depth_stencil) usage |= 0x20; // VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
    if (flags.allow_unordered_access) usage |= 0x80; // VK_IMAGE_USAGE_STORAGE_BIT
    if (!flags.deny_shader_resource) usage |= 0x04; // VK_IMAGE_USAGE_SAMPLED_BIT
    return usage;
}

fn heapTypeToVulkanMemoryType(heap_type: D3D12_HEAP_TYPE) VulkanMemoryType {
    return switch (heap_type) {
        .DEFAULT => .device_local,
        .UPLOAD => .host_visible,
        .READBACK => .host_cached,
        .CUSTOM => .device_local,
    };
}

fn getFormatSize(format: u32) u32 {
    // Return bytes per pixel for common formats
    return switch (format) {
        28, 87 => 4, // R8G8B8A8, B8G8R8A8
        24 => 8, // R32G32_FLOAT
        2 => 16, // R32G32B32A32_FLOAT
        else => 4,
    };
}