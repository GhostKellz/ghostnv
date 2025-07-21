const std = @import("std");
const hal = @import("../hal/command.zig");
const memory = @import("../hal/memory.zig");

pub const VulkanPipelineCache = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    pipeline_cache: PipelineCache,
    descriptor_cache: DescriptorCache,
    render_pass_cache: RenderPassCache,
    shader_cache: ShaderCache,
    state_optimizer: StateOptimizer,
    
    pub const PipelineCache = struct {
        allocator: std.mem.Allocator,
        graphics_pipelines: std.HashMap(u64, GraphicsPipeline, std.hash_map.AutoContext(u64), 80),
        compute_pipelines: std.HashMap(u64, ComputePipeline, std.hash_map.AutoContext(u64), 80),
        cache_hits: u64,
        cache_misses: u64,
        compilation_time_saved_ms: f64,
        
        pub const GraphicsPipeline = struct {
            pipeline_id: u64,
            hash: u64,
            vertex_shader: ShaderModule,
            fragment_shader: ShaderModule,
            geometry_shader: ?ShaderModule,
            tessellation_control_shader: ?ShaderModule,
            tessellation_evaluation_shader: ?ShaderModule,
            vertex_input_state: VertexInputState,
            input_assembly_state: InputAssemblyState,
            viewport_state: ViewportState,
            rasterization_state: RasterizationState,
            multisample_state: MultisampleState,
            depth_stencil_state: DepthStencilState,
            color_blend_state: ColorBlendState,
            dynamic_state: DynamicState,
            render_pass: u64,
            subpass: u32,
            binary_data: []u8,
            creation_time: i64,
            last_used: i64,
            use_count: u64,
            compilation_time_ms: f32,
            
            pub const VertexInputState = struct {
                vertex_binding_descriptions: []VertexBindingDescription,
                vertex_attribute_descriptions: []VertexAttributeDescription,
                
                pub const VertexBindingDescription = struct {
                    binding: u32,
                    stride: u32,
                    input_rate: InputRate,
                    
                    pub const InputRate = enum {
                        vertex,
                        instance,
                    };
                };
                
                pub const VertexAttributeDescription = struct {
                    location: u32,
                    binding: u32,
                    format: Format,
                    offset: u32,
                    
                    pub const Format = enum {
                        r32_sfloat,
                        r32g32_sfloat,
                        r32g32b32_sfloat,
                        r32g32b32a32_sfloat,
                        r8g8b8a8_unorm,
                        r16g16_sfloat,
                        r16g16b16a16_sfloat,
                    };
                };
            };
            
            pub const InputAssemblyState = struct {
                topology: PrimitiveTopology,
                primitive_restart_enable: bool,
                
                pub const PrimitiveTopology = enum {
                    point_list,
                    line_list,
                    line_strip,
                    triangle_list,
                    triangle_strip,
                    triangle_fan,
                };
            };
            
            pub const ViewportState = struct {
                viewport_count: u32,
                scissor_count: u32,
                viewports: []Viewport,
                scissors: []Scissor,
                
                pub const Viewport = struct {
                    x: f32,
                    y: f32,
                    width: f32,
                    height: f32,
                    min_depth: f32,
                    max_depth: f32,
                };
                
                pub const Scissor = struct {
                    offset: Offset2D,
                    extent: Extent2D,
                    
                    pub const Offset2D = struct {
                        x: i32,
                        y: i32,
                    };
                    
                    pub const Extent2D = struct {
                        width: u32,
                        height: u32,
                    };
                };
            };
            
            pub const RasterizationState = struct {
                depth_clamp_enable: bool,
                rasterizer_discard_enable: bool,
                polygon_mode: PolygonMode,
                cull_mode: CullMode,
                front_face: FrontFace,
                depth_bias_enable: bool,
                depth_bias_constant_factor: f32,
                depth_bias_clamp: f32,
                depth_bias_slope_factor: f32,
                line_width: f32,
                
                pub const PolygonMode = enum {
                    fill,
                    line,
                    point,
                };
                
                pub const CullMode = enum {
                    none,
                    front,
                    back,
                    front_and_back,
                };
                
                pub const FrontFace = enum {
                    counter_clockwise,
                    clockwise,
                };
            };
            
            pub const MultisampleState = struct {
                rasterization_samples: SampleCount,
                sample_shading_enable: bool,
                min_sample_shading: f32,
                sample_mask: ?[]u32,
                alpha_to_coverage_enable: bool,
                alpha_to_one_enable: bool,
                
                pub const SampleCount = enum {
                    @"1",
                    @"2",
                    @"4",
                    @"8",
                    @"16",
                    @"32",
                    @"64",
                };
            };
            
            pub const DepthStencilState = struct {
                depth_test_enable: bool,
                depth_write_enable: bool,
                depth_compare_op: CompareOp,
                depth_bounds_test_enable: bool,
                stencil_test_enable: bool,
                front: StencilOpState,
                back: StencilOpState,
                min_depth_bounds: f32,
                max_depth_bounds: f32,
                
                pub const CompareOp = enum {
                    never,
                    less,
                    equal,
                    less_or_equal,
                    greater,
                    not_equal,
                    greater_or_equal,
                    always,
                };
                
                pub const StencilOpState = struct {
                    fail_op: StencilOp,
                    pass_op: StencilOp,
                    depth_fail_op: StencilOp,
                    compare_op: CompareOp,
                    compare_mask: u32,
                    write_mask: u32,
                    reference: u32,
                    
                    pub const StencilOp = enum {
                        keep,
                        zero,
                        replace,
                        increment_and_clamp,
                        decrement_and_clamp,
                        invert,
                        increment_and_wrap,
                        decrement_and_wrap,
                    };
                };
            };
            
            pub const ColorBlendState = struct {
                logic_op_enable: bool,
                logic_op: LogicOp,
                attachments: []ColorBlendAttachmentState,
                blend_constants: [4]f32,
                
                pub const LogicOp = enum {
                    clear,
                    @"and",
                    and_reverse,
                    copy,
                    and_inverted,
                    no_op,
                    @"xor",
                    @"or",
                    nor,
                    equivalent,
                    invert,
                    or_reverse,
                    copy_inverted,
                    or_inverted,
                    nand,
                    set,
                };
                
                pub const ColorBlendAttachmentState = struct {
                    color_write_mask: ColorComponentFlags,
                    blend_enable: bool,
                    src_color_blend_factor: BlendFactor,
                    dst_color_blend_factor: BlendFactor,
                    color_blend_op: BlendOp,
                    src_alpha_blend_factor: BlendFactor,
                    dst_alpha_blend_factor: BlendFactor,
                    alpha_blend_op: BlendOp,
                    
                    pub const ColorComponentFlags = packed struct {
                        r: bool = false,
                        g: bool = false,
                        b: bool = false,
                        a: bool = false,
                        _: u28 = 0,
                    };
                    
                    pub const BlendFactor = enum {
                        zero,
                        one,
                        src_color,
                        one_minus_src_color,
                        dst_color,
                        one_minus_dst_color,
                        src_alpha,
                        one_minus_src_alpha,
                        dst_alpha,
                        one_minus_dst_alpha,
                        constant_color,
                        one_minus_constant_color,
                        constant_alpha,
                        one_minus_constant_alpha,
                        src_alpha_saturate,
                    };
                    
                    pub const BlendOp = enum {
                        add,
                        subtract,
                        reverse_subtract,
                        min,
                        max,
                    };
                };
            };
            
            pub const DynamicState = struct {
                dynamic_states: []DynamicStateType,
                
                pub const DynamicStateType = enum {
                    viewport,
                    scissor,
                    line_width,
                    depth_bias,
                    blend_constants,
                    depth_bounds,
                    stencil_compare_mask,
                    stencil_write_mask,
                    stencil_reference,
                };
            };
            
            pub fn calculateHash(self: *const GraphicsPipeline) u64 {
                var hasher = std.hash.Wyhash.init(0);
                hasher.update(std.mem.asBytes(&self.vertex_shader.hash));
                hasher.update(std.mem.asBytes(&self.fragment_shader.hash));
                hasher.update(std.mem.asBytes(&self.vertex_input_state));
                hasher.update(std.mem.asBytes(&self.input_assembly_state));
                hasher.update(std.mem.asBytes(&self.rasterization_state));
                hasher.update(std.mem.asBytes(&self.multisample_state));
                hasher.update(std.mem.asBytes(&self.depth_stencil_state));
                hasher.update(std.mem.asBytes(&self.render_pass));
                hasher.update(std.mem.asBytes(&self.subpass));
                return hasher.final();
            }
        };
        
        pub const ComputePipeline = struct {
            pipeline_id: u64,
            hash: u64,
            compute_shader: ShaderModule,
            binary_data: []u8,
            creation_time: i64,
            last_used: i64,
            use_count: u64,
            compilation_time_ms: f32,
            
            pub fn calculateHash(self: *const ComputePipeline) u64 {
                return std.hash.Wyhash.hash(0, std.mem.asBytes(&self.compute_shader.hash));
            }
        };
        
        pub fn init(allocator: std.mem.Allocator) PipelineCache {
            return .{
                .allocator = allocator,
                .graphics_pipelines = std.HashMap(u64, GraphicsPipeline, std.hash_map.AutoContext(u64), 80).init(allocator),
                .compute_pipelines = std.HashMap(u64, ComputePipeline, std.hash_map.AutoContext(u64), 80).init(allocator),
                .cache_hits = 0,
                .cache_misses = 0,
                .compilation_time_saved_ms = 0,
            };
        }
        
        pub fn deinit(self: *PipelineCache) void {
            var graphics_iter = self.graphics_pipelines.iterator();
            while (graphics_iter.next()) |entry| {
                self.allocator.free(entry.value_ptr.binary_data);
            }
            self.graphics_pipelines.deinit();
            
            var compute_iter = self.compute_pipelines.iterator();
            while (compute_iter.next()) |entry| {
                self.allocator.free(entry.value_ptr.binary_data);
            }
            self.compute_pipelines.deinit();
        }
        
        pub fn getOrCreateGraphicsPipeline(self: *PipelineCache, desc: GraphicsPipelineDesc) !*GraphicsPipeline {
            const hash = desc.calculateHash();
            
            if (self.graphics_pipelines.getPtr(hash)) |pipeline| {
                pipeline.last_used = std.time.milliTimestamp();
                pipeline.use_count += 1;
                self.cache_hits += 1;
                self.compilation_time_saved_ms += pipeline.compilation_time_ms;
                return pipeline;
            }
            
            // Cache miss - create new pipeline
            self.cache_misses += 1;
            const start_time = std.time.milliTimestamp();
            
            const pipeline = try self.compilePipeline(desc);
            const compilation_time = @as(f32, @floatFromInt(std.time.milliTimestamp() - start_time));
            
            var new_pipeline = GraphicsPipeline{
                .pipeline_id = hash,
                .hash = hash,
                .vertex_shader = desc.vertex_shader,
                .fragment_shader = desc.fragment_shader,
                .geometry_shader = desc.geometry_shader,
                .tessellation_control_shader = desc.tessellation_control_shader,
                .tessellation_evaluation_shader = desc.tessellation_evaluation_shader,
                .vertex_input_state = desc.vertex_input_state,
                .input_assembly_state = desc.input_assembly_state,
                .viewport_state = desc.viewport_state,
                .rasterization_state = desc.rasterization_state,
                .multisample_state = desc.multisample_state,
                .depth_stencil_state = desc.depth_stencil_state,
                .color_blend_state = desc.color_blend_state,
                .dynamic_state = desc.dynamic_state,
                .render_pass = desc.render_pass,
                .subpass = desc.subpass,
                .binary_data = pipeline,
                .creation_time = start_time,
                .last_used = start_time,
                .use_count = 1,
                .compilation_time_ms = compilation_time,
            };
            
            try self.graphics_pipelines.put(hash, new_pipeline);
            return self.graphics_pipelines.getPtr(hash).?;
        }
        
        pub fn getOrCreateComputePipeline(self: *PipelineCache, desc: ComputePipelineDesc) !*ComputePipeline {
            const hash = desc.calculateHash();
            
            if (self.compute_pipelines.getPtr(hash)) |pipeline| {
                pipeline.last_used = std.time.milliTimestamp();
                pipeline.use_count += 1;
                self.cache_hits += 1;
                self.compilation_time_saved_ms += pipeline.compilation_time_ms;
                return pipeline;
            }
            
            // Cache miss - create new pipeline
            self.cache_misses += 1;
            const start_time = std.time.milliTimestamp();
            
            const pipeline = try self.compileComputePipeline(desc);
            const compilation_time = @as(f32, @floatFromInt(std.time.milliTimestamp() - start_time));
            
            var new_pipeline = ComputePipeline{
                .pipeline_id = hash,
                .hash = hash,
                .compute_shader = desc.compute_shader,
                .binary_data = pipeline,
                .creation_time = start_time,
                .last_used = start_time,
                .use_count = 1,
                .compilation_time_ms = compilation_time,
            };
            
            try self.compute_pipelines.put(hash, new_pipeline);
            return self.compute_pipelines.getPtr(hash).?;
        }
        
        fn compilePipeline(self: *PipelineCache, desc: GraphicsPipelineDesc) ![]u8 {
            _ = desc;
            // Mock pipeline compilation
            const binary = try self.allocator.alloc(u8, 4096);
            @memset(binary, 0xCC); // Mock binary data
            return binary;
        }
        
        fn compileComputePipeline(self: *PipelineCache, desc: ComputePipelineDesc) ![]u8 {
            _ = desc;
            // Mock compute pipeline compilation
            const binary = try self.allocator.alloc(u8, 2048);
            @memset(binary, 0xDD); // Mock binary data
            return binary;
        }
        
        pub fn getCacheStats(self: *const PipelineCache) CacheStats {
            return .{
                .cache_hits = self.cache_hits,
                .cache_misses = self.cache_misses,
                .hit_ratio = if (self.cache_hits + self.cache_misses > 0)
                    @as(f32, @floatFromInt(self.cache_hits)) / @as(f32, @floatFromInt(self.cache_hits + self.cache_misses))
                else
                    0.0,
                .compilation_time_saved_ms = self.compilation_time_saved_ms,
                .graphics_pipelines_cached = @intCast(self.graphics_pipelines.count()),
                .compute_pipelines_cached = @intCast(self.compute_pipelines.count()),
            };
        }
        
        pub const CacheStats = struct {
            cache_hits: u64,
            cache_misses: u64,
            hit_ratio: f32,
            compilation_time_saved_ms: f64,
            graphics_pipelines_cached: u32,
            compute_pipelines_cached: u32,
        };
    };
    
    pub const DescriptorCache = struct {
        allocator: std.mem.Allocator,
        descriptor_sets: std.HashMap(u64, DescriptorSet, std.hash_map.AutoContext(u64), 80),
        descriptor_pools: std.ArrayList(DescriptorPool),
        
        pub const DescriptorSet = struct {
            hash: u64,
            layout: DescriptorSetLayout,
            bindings: []DescriptorBinding,
            pool: *DescriptorPool,
            last_used: i64,
            
            pub const DescriptorSetLayout = struct {
                bindings: []DescriptorSetLayoutBinding,
                
                pub const DescriptorSetLayoutBinding = struct {
                    binding: u32,
                    descriptor_type: DescriptorType,
                    descriptor_count: u32,
                    stage_flags: ShaderStageFlags,
                    
                    pub const DescriptorType = enum {
                        sampler,
                        combined_image_sampler,
                        sampled_image,
                        storage_image,
                        uniform_texel_buffer,
                        storage_texel_buffer,
                        uniform_buffer,
                        storage_buffer,
                        uniform_buffer_dynamic,
                        storage_buffer_dynamic,
                        input_attachment,
                    };
                    
                    pub const ShaderStageFlags = packed struct {
                        vertex: bool = false,
                        tessellation_control: bool = false,
                        tessellation_evaluation: bool = false,
                        geometry: bool = false,
                        fragment: bool = false,
                        compute: bool = false,
                        _: u26 = 0,
                    };
                };
            };
            
            pub const DescriptorBinding = struct {
                binding: u32,
                resource: DescriptorResource,
                
                pub const DescriptorResource = union(enum) {
                    buffer: BufferDescriptor,
                    image: ImageDescriptor,
                    sampler: SamplerDescriptor,
                    
                    pub const BufferDescriptor = struct {
                        buffer: u64,
                        offset: u64,
                        range: u64,
                    };
                    
                    pub const ImageDescriptor = struct {
                        image: u64,
                        image_view: u64,
                        layout: ImageLayout,
                        
                        pub const ImageLayout = enum {
                            undefined,
                            general,
                            color_attachment_optimal,
                            depth_stencil_attachment_optimal,
                            depth_stencil_read_only_optimal,
                            shader_read_only_optimal,
                            transfer_src_optimal,
                            transfer_dst_optimal,
                            preinitialized,
                        };
                    };
                    
                    pub const SamplerDescriptor = struct {
                        sampler: u64,
                    };
                };
            };
        };
        
        pub const DescriptorPool = struct {
            pool_id: u64,
            max_sets: u32,
            allocated_sets: u32,
            pool_sizes: []PoolSize,
            
            pub const PoolSize = struct {
                type: DescriptorSet.DescriptorSetLayout.DescriptorSetLayoutBinding.DescriptorType,
                descriptor_count: u32,
            };
        };
        
        pub fn init(allocator: std.mem.Allocator) DescriptorCache {
            return .{
                .allocator = allocator,
                .descriptor_sets = std.HashMap(u64, DescriptorSet, std.hash_map.AutoContext(u64), 80).init(allocator),
                .descriptor_pools = std.ArrayList(DescriptorPool).init(allocator),
            };
        }
        
        pub fn deinit(self: *DescriptorCache) void {
            self.descriptor_sets.deinit();
            self.descriptor_pools.deinit();
        }
    };
    
    pub const RenderPassCache = struct {
        allocator: std.mem.Allocator,
        render_passes: std.HashMap(u64, RenderPass, std.hash_map.AutoContext(u64), 80),
        framebuffers: std.HashMap(u64, Framebuffer, std.hash_map.AutoContext(u64), 80),
        
        pub const RenderPass = struct {
            hash: u64,
            attachments: []AttachmentDescription,
            subpasses: []SubpassDescription,
            dependencies: []SubpassDependency,
            
            pub const AttachmentDescription = struct {
                format: Format,
                samples: SampleCount,
                load_op: AttachmentLoadOp,
                store_op: AttachmentStoreOp,
                stencil_load_op: AttachmentLoadOp,
                stencil_store_op: AttachmentStoreOp,
                initial_layout: ImageLayout,
                final_layout: ImageLayout,
                
                pub const Format = enum {
                    r8g8b8a8_unorm,
                    b8g8r8a8_unorm,
                    d32_sfloat,
                    d24_unorm_s8_uint,
                    r16g16b16a16_sfloat,
                };
                
                pub const SampleCount = enum {
                    @"1",
                    @"2",
                    @"4",
                    @"8",
                    @"16",
                    @"32",
                    @"64",
                };
                
                pub const AttachmentLoadOp = enum {
                    load,
                    clear,
                    dont_care,
                };
                
                pub const AttachmentStoreOp = enum {
                    store,
                    dont_care,
                };
                
                pub const ImageLayout = enum {
                    undefined,
                    general,
                    color_attachment_optimal,
                    depth_stencil_attachment_optimal,
                    depth_stencil_read_only_optimal,
                    shader_read_only_optimal,
                    transfer_src_optimal,
                    transfer_dst_optimal,
                    preinitialized,
                    present_src,
                };
            };
            
            pub const SubpassDescription = struct {
                pipeline_bind_point: PipelineBindPoint,
                input_attachments: []AttachmentReference,
                color_attachments: []AttachmentReference,
                resolve_attachments: []AttachmentReference,
                depth_stencil_attachment: ?AttachmentReference,
                preserve_attachments: []u32,
                
                pub const PipelineBindPoint = enum {
                    graphics,
                    compute,
                };
                
                pub const AttachmentReference = struct {
                    attachment: u32,
                    layout: AttachmentDescription.ImageLayout,
                };
            };
            
            pub const SubpassDependency = struct {
                src_subpass: u32,
                dst_subpass: u32,
                src_stage_mask: PipelineStageFlags,
                dst_stage_mask: PipelineStageFlags,
                src_access_mask: AccessFlags,
                dst_access_mask: AccessFlags,
                dependency_flags: DependencyFlags,
                
                pub const PipelineStageFlags = packed struct {
                    top_of_pipe: bool = false,
                    draw_indirect: bool = false,
                    vertex_input: bool = false,
                    vertex_shader: bool = false,
                    tessellation_control_shader: bool = false,
                    tessellation_evaluation_shader: bool = false,
                    geometry_shader: bool = false,
                    fragment_shader: bool = false,
                    early_fragment_tests: bool = false,
                    late_fragment_tests: bool = false,
                    color_attachment_output: bool = false,
                    compute_shader: bool = false,
                    transfer: bool = false,
                    bottom_of_pipe: bool = false,
                    host: bool = false,
                    all_graphics: bool = false,
                    all_commands: bool = false,
                    _: u15 = 0,
                };
                
                pub const AccessFlags = packed struct {
                    indirect_command_read: bool = false,
                    index_read: bool = false,
                    vertex_attribute_read: bool = false,
                    uniform_read: bool = false,
                    input_attachment_read: bool = false,
                    shader_read: bool = false,
                    shader_write: bool = false,
                    color_attachment_read: bool = false,
                    color_attachment_write: bool = false,
                    depth_stencil_attachment_read: bool = false,
                    depth_stencil_attachment_write: bool = false,
                    transfer_read: bool = false,
                    transfer_write: bool = false,
                    host_read: bool = false,
                    host_write: bool = false,
                    memory_read: bool = false,
                    memory_write: bool = false,
                    _: u15 = 0,
                };
                
                pub const DependencyFlags = packed struct {
                    by_region: bool = false,
                    device_group: bool = false,
                    view_local: bool = false,
                    _: u29 = 0,
                };
            };
        };
        
        pub const Framebuffer = struct {
            hash: u64,
            render_pass: u64,
            attachments: []u64,
            width: u32,
            height: u32,
            layers: u32,
        };
        
        pub fn init(allocator: std.mem.Allocator) RenderPassCache {
            return .{
                .allocator = allocator,
                .render_passes = std.HashMap(u64, RenderPass, std.hash_map.AutoContext(u64), 80).init(allocator),
                .framebuffers = std.HashMap(u64, Framebuffer, std.hash_map.AutoContext(u64), 80).init(allocator),
            };
        }
        
        pub fn deinit(self: *RenderPassCache) void {
            self.render_passes.deinit();
            self.framebuffers.deinit();
        }
    };
    
    pub const ShaderCache = struct {
        allocator: std.mem.Allocator,
        shader_modules: std.HashMap(u64, ShaderModule, std.hash_map.AutoContext(u64), 80),
        compiled_shaders: std.HashMap(u64, CompiledShader, std.hash_map.AutoContext(u64), 80),
        
        pub const CompiledShader = struct {
            hash: u64,
            binary_data: []u8,
            compilation_time_ms: f32,
            optimization_level: u8,
            target_arch: u32,
        };
        
        pub fn init(allocator: std.mem.Allocator) ShaderCache {
            return .{
                .allocator = allocator,
                .shader_modules = std.HashMap(u64, ShaderModule, std.hash_map.AutoContext(u64), 80).init(allocator),
                .compiled_shaders = std.HashMap(u64, CompiledShader, std.hash_map.AutoContext(u64), 80).init(allocator),
            };
        }
        
        pub fn deinit(self: *ShaderCache) void {
            var shader_iter = self.shader_modules.iterator();
            while (shader_iter.next()) |entry| {
                self.allocator.free(entry.value_ptr.code);
            }
            self.shader_modules.deinit();
            
            var compiled_iter = self.compiled_shaders.iterator();
            while (compiled_iter.next()) |entry| {
                self.allocator.free(entry.value_ptr.binary_data);
            }
            self.compiled_shaders.deinit();
        }
    };
    
    pub const StateOptimizer = struct {
        enable_state_merging: bool,
        enable_redundant_state_elimination: bool,
        enable_pipeline_derivatives: bool,
        
        pub fn optimizePipelineCreation(self: *const StateOptimizer, desc: *GraphicsPipelineDesc) void {
            if (self.enable_redundant_state_elimination) {
                self.eliminateRedundantState(desc);
            }
            
            if (self.enable_state_merging) {
                self.mergeCompatibleStates(desc);
            }
        }
        
        fn eliminateRedundantState(self: *const StateOptimizer, desc: *GraphicsPipelineDesc) void {
            _ = self;
            _ = desc;
            // Remove redundant or default states to reduce hash computation
        }
        
        fn mergeCompatibleStates(self: *const StateOptimizer, desc: *GraphicsPipelineDesc) void {
            _ = self;
            _ = desc;
            // Merge compatible dynamic states
        }
    };
    
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .pipeline_cache = PipelineCache.init(allocator),
            .descriptor_cache = DescriptorCache.init(allocator),
            .render_pass_cache = RenderPassCache.init(allocator),
            .shader_cache = ShaderCache.init(allocator),
            .state_optimizer = .{
                .enable_state_merging = true,
                .enable_redundant_state_elimination = true,
                .enable_pipeline_derivatives = true,
            },
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.pipeline_cache.deinit();
        self.descriptor_cache.deinit();
        self.render_pass_cache.deinit();
        self.shader_cache.deinit();
    }
    
    pub fn getOrCreateGraphicsPipeline(self: *Self, desc: GraphicsPipelineDesc) !*PipelineCache.GraphicsPipeline {
        var optimized_desc = desc;
        self.state_optimizer.optimizePipelineCreation(&optimized_desc);
        return try self.pipeline_cache.getOrCreateGraphicsPipeline(optimized_desc);
    }
    
    pub fn getOrCreateComputePipeline(self: *Self, desc: ComputePipelineDesc) !*PipelineCache.ComputePipeline {
        return try self.pipeline_cache.getOrCreateComputePipeline(desc);
    }
    
    pub fn getCacheStatistics(self: *const Self) CacheStatistics {
        const pipeline_stats = self.pipeline_cache.getCacheStats();
        
        return .{
            .pipeline_cache_hits = pipeline_stats.cache_hits,
            .pipeline_cache_misses = pipeline_stats.cache_misses,
            .pipeline_hit_ratio = pipeline_stats.hit_ratio,
            .compilation_time_saved_ms = pipeline_stats.compilation_time_saved_ms,
            .total_graphics_pipelines = pipeline_stats.graphics_pipelines_cached,
            .total_compute_pipelines = pipeline_stats.compute_pipelines_cached,
            .total_shader_modules = @intCast(self.shader_cache.shader_modules.count()),
            .total_render_passes = @intCast(self.render_pass_cache.render_passes.count()),
            .total_descriptor_sets = @intCast(self.descriptor_cache.descriptor_sets.count()),
        };
    }
    
    pub const CacheStatistics = struct {
        pipeline_cache_hits: u64,
        pipeline_cache_misses: u64,
        pipeline_hit_ratio: f32,
        compilation_time_saved_ms: f64,
        total_graphics_pipelines: u32,
        total_compute_pipelines: u32,
        total_shader_modules: u32,
        total_render_passes: u32,
        total_descriptor_sets: u32,
    };
};

pub const ShaderModule = struct {
    hash: u64,
    stage: ShaderStage,
    code: []const u8,
    entry_point: []const u8,
    
    pub const ShaderStage = enum {
        vertex,
        tessellation_control,
        tessellation_evaluation,
        geometry,
        fragment,
        compute,
    };
};

pub const GraphicsPipelineDesc = struct {
    vertex_shader: ShaderModule,
    fragment_shader: ShaderModule,
    geometry_shader: ?ShaderModule = null,
    tessellation_control_shader: ?ShaderModule = null,
    tessellation_evaluation_shader: ?ShaderModule = null,
    vertex_input_state: VulkanPipelineCache.PipelineCache.GraphicsPipeline.VertexInputState,
    input_assembly_state: VulkanPipelineCache.PipelineCache.GraphicsPipeline.InputAssemblyState,
    viewport_state: VulkanPipelineCache.PipelineCache.GraphicsPipeline.ViewportState,
    rasterization_state: VulkanPipelineCache.PipelineCache.GraphicsPipeline.RasterizationState,
    multisample_state: VulkanPipelineCache.PipelineCache.GraphicsPipeline.MultisampleState,
    depth_stencil_state: VulkanPipelineCache.PipelineCache.GraphicsPipeline.DepthStencilState,
    color_blend_state: VulkanPipelineCache.PipelineCache.GraphicsPipeline.ColorBlendState,
    dynamic_state: VulkanPipelineCache.PipelineCache.GraphicsPipeline.DynamicState,
    render_pass: u64,
    subpass: u32,
    
    pub fn calculateHash(self: *const GraphicsPipelineDesc) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&self.vertex_shader.hash));
        hasher.update(std.mem.asBytes(&self.fragment_shader.hash));
        hasher.update(std.mem.asBytes(&self.vertex_input_state));
        hasher.update(std.mem.asBytes(&self.input_assembly_state));
        hasher.update(std.mem.asBytes(&self.rasterization_state));
        hasher.update(std.mem.asBytes(&self.multisample_state));
        hasher.update(std.mem.asBytes(&self.depth_stencil_state));
        hasher.update(std.mem.asBytes(&self.render_pass));
        hasher.update(std.mem.asBytes(&self.subpass));
        return hasher.final();
    }
};

pub const ComputePipelineDesc = struct {
    compute_shader: ShaderModule,
    
    pub fn calculateHash(self: *const ComputePipelineDesc) u64 {
        return std.hash.Wyhash.hash(0, std.mem.asBytes(&self.compute_shader.hash));
    }
};