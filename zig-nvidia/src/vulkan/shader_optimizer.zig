const std = @import("std");
const pipeline_cache = @import("pipeline_cache.zig");
const hal = @import("../hal/command.zig");

pub const ShaderOptimizer = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    shader_cache: ShaderCache,
    compiler: ShaderCompiler,
    optimizer_engine: OptimizerEngine,
    binary_cache: BinaryCache,
    
    pub const ShaderCache = struct {
        allocator: std.mem.Allocator,
        spv_cache: std.HashMap(u64, SpvShader, std.hash_map.AutoContext(u64), 80),
        sass_cache: std.HashMap(u64, SassShader, std.hash_map.AutoContext(u64), 80),
        optimization_cache: std.HashMap(u64, OptimizedShader, std.hash_map.AutoContext(u64), 80),
        persistent_cache: PersistentCache,
        
        pub const SpvShader = struct {
            hash: u64,
            stage: pipeline_cache.ShaderModule.ShaderStage,
            spirv_code: []const u32,
            metadata: ShaderMetadata,
            compilation_time_ms: f32,
            optimization_level: u8,
            
            pub const ShaderMetadata = struct {
                entry_points: []EntryPoint,
                resources: ResourceInfo,
                capabilities: []SpvCapability,
                extensions: [][]const u8,
                
                pub const EntryPoint = struct {
                    name: []const u8,
                    execution_model: ExecutionModel,
                    interface_variables: []u32,
                    
                    pub const ExecutionModel = enum {
                        vertex,
                        tessellation_control,
                        tessellation_evaluation,
                        geometry,
                        fragment,
                        gl_compute,
                        kernel,
                    };
                };
                
                pub const ResourceInfo = struct {
                    uniform_buffers: []UniformBuffer,
                    storage_buffers: []StorageBuffer,
                    samplers: []Sampler,
                    images: []Image,
                    input_variables: []InputVariable,
                    output_variables: []OutputVariable,
                    
                    pub const UniformBuffer = struct {
                        binding: u32,
                        set: u32,
                        size: u32,
                        member_count: u32,
                    };
                    
                    pub const StorageBuffer = struct {
                        binding: u32,
                        set: u32,
                        readonly: bool,
                        size: u32,
                    };
                    
                    pub const Sampler = struct {
                        binding: u32,
                        set: u32,
                        dimension: ImageDimension,
                        sampled: bool,
                        
                        pub const ImageDimension = enum {
                            @"1d",
                            @"2d",
                            @"3d",
                            cube,
                            rect,
                            buffer,
                            subpass_data,
                        };
                    };
                    
                    pub const Image = struct {
                        binding: u32,
                        set: u32,
                        dimension: Sampler.ImageDimension,
                        format: ImageFormat,
                        access: ImageAccess,
                        
                        pub const ImageFormat = enum {
                            unknown,
                            rgba32f,
                            rgba16f,
                            r32f,
                            rgba8,
                            rg16f,
                            r16f,
                            rgba32i,
                            rgba16i,
                            rgba8i,
                            r32i,
                        };
                        
                        pub const ImageAccess = enum {
                            read_only,
                            write_only,
                            read_write,
                        };
                    };
                    
                    pub const InputVariable = struct {
                        location: u32,
                        component: u32,
                        format: VertexFormat,
                        
                        pub const VertexFormat = enum {
                            float,
                            vec2,
                            vec3,
                            vec4,
                            int,
                            ivec2,
                            ivec3,
                            ivec4,
                            uint,
                            uvec2,
                            uvec3,
                            uvec4,
                        };
                    };
                    
                    pub const OutputVariable = struct {
                        location: u32,
                        component: u32,
                        format: InputVariable.VertexFormat,
                    };
                };
                
                pub const SpvCapability = enum {
                    matrix,
                    shader,
                    geometry,
                    tessellation,
                    addresses,
                    linkage,
                    kernel,
                    vector16,
                    float16_buffer,
                    float16,
                    float64,
                    int64,
                    int64_atomics,
                    image_basic,
                    image_read_write,
                    image_mipmap,
                    pipes,
                    groups,
                    device_enqueue,
                    literal_sampler,
                    atomic_storage,
                    int16,
                    tessellation_point_size,
                    geometry_point_size,
                    image_gather_extended,
                    storage_image_multisample,
                    uniform_buffer_array_dynamic_indexing,
                    sampled_image_array_dynamic_indexing,
                    storage_buffer_array_dynamic_indexing,
                    storage_image_array_dynamic_indexing,
                    clip_distance,
                    cull_distance,
                    image_cube_array,
                    sample_rate_shading,
                    image_rect,
                    sampled_rect,
                    generic_pointer,
                    int8,
                    input_attachment,
                    sparse_residency,
                    min_lod,
                    sampled1d,
                    image1d,
                    sampled_cube_array,
                    sampled_buffer,
                    image_buffer,
                    image_ms_array,
                    storage_image_extended_formats,
                    image_query,
                    derivative_group,
                    interpolation_function,
                    transform_feedback,
                    geometry_streams,
                    storage_image_read_without_format,
                    storage_image_write_without_format,
                    multiviewport,
                };
            };
        };
        
        pub const SassShader = struct {
            hash: u64,
            sass_code: []const u8,
            register_count: u32,
            shared_memory_size: u32,
            occupancy_info: OccupancyInfo,
            instruction_count: u32,
            
            pub const OccupancyInfo = struct {
                max_warps_per_sm: u32,
                max_blocks_per_sm: u32,
                limiting_factor: LimitingFactor,
                
                pub const LimitingFactor = enum {
                    registers,
                    shared_memory,
                    thread_count,
                    block_count,
                };
            };
        };
        
        pub const OptimizedShader = struct {
            original_hash: u64,
            optimized_hash: u64,
            optimization_passes: []OptimizationPass,
            performance_improvement: f32,
            size_reduction: f32,
            
            pub const OptimizationPass = struct {
                pass_name: []const u8,
                applied: bool,
                improvement_factor: f32,
            };
        };
        
        pub const PersistentCache = struct {
            cache_file_path: []const u8,
            cache_version: u32,
            is_dirty: bool,
            
            pub fn load(self: *PersistentCache, allocator: std.mem.Allocator, cache: *ShaderCache) !void {
                _ = self;
                _ = allocator;
                _ = cache;
                // Load cache from disk
            }
            
            pub fn save(self: *PersistentCache, cache: *const ShaderCache) !void {
                _ = self;
                _ = cache;
                // Save cache to disk
            }
        };
        
        pub fn init(allocator: std.mem.Allocator, cache_dir: []const u8) !ShaderCache {
            var cache_path = try std.fmt.allocPrint(allocator, "{s}/shader_cache.bin", .{cache_dir});
            
            return ShaderCache{
                .allocator = allocator,
                .spv_cache = std.HashMap(u64, SpvShader, std.hash_map.AutoContext(u64), 80).init(allocator),
                .sass_cache = std.HashMap(u64, SassShader, std.hash_map.AutoContext(u64), 80).init(allocator),
                .optimization_cache = std.HashMap(u64, OptimizedShader, std.hash_map.AutoContext(u64), 80).init(allocator),
                .persistent_cache = .{
                    .cache_file_path = cache_path,
                    .cache_version = 1,
                    .is_dirty = false,
                },
            };
        }
        
        pub fn deinit(self: *ShaderCache) void {
            self.allocator.free(self.persistent_cache.cache_file_path);
            
            var spv_iter = self.spv_cache.iterator();
            while (spv_iter.next()) |entry| {
                self.allocator.free(entry.value_ptr.spirv_code);
            }
            self.spv_cache.deinit();
            
            var sass_iter = self.sass_cache.iterator();
            while (sass_iter.next()) |entry| {
                self.allocator.free(entry.value_ptr.sass_code);
            }
            self.sass_cache.deinit();
            
            self.optimization_cache.deinit();
        }
        
        pub fn getOrCompileSpv(self: *ShaderCache, source_hash: u64, compiler: *ShaderCompiler) !*SpvShader {
            if (self.spv_cache.getPtr(source_hash)) |shader| {
                return shader;
            }
            
            // Compile new shader
            const compiled = try compiler.compileToSpv(source_hash);
            try self.spv_cache.put(source_hash, compiled);
            self.persistent_cache.is_dirty = true;
            
            return self.spv_cache.getPtr(source_hash).?;
        }
        
        pub fn getOrCompileSass(self: *ShaderCache, spv_hash: u64, compiler: *ShaderCompiler) !*SassShader {
            if (self.sass_cache.getPtr(spv_hash)) |shader| {
                return shader;
            }
            
            // Compile SPIR-V to SASS
            const compiled = try compiler.compileToSass(spv_hash);
            try self.sass_cache.put(spv_hash, compiled);
            self.persistent_cache.is_dirty = true;
            
            return self.sass_cache.getPtr(spv_hash).?;
        }
    };
    
    pub const ShaderCompiler = struct {
        allocator: std.mem.Allocator,
        glslang_compiler: GlslangCompiler,
        spirv_optimizer: SpirvOptimizer,
        sass_compiler: SassCompiler,
        
        pub const GlslangCompiler = struct {
            include_directories: [][]const u8,
            defines: std.StringHashMap([]const u8),
            optimization_level: u8,
            
            pub fn compileGlslToSpv(
                self: *GlslangCompiler,
                source: []const u8,
                stage: pipeline_cache.ShaderModule.ShaderStage,
                entry_point: []const u8,
            ) ![]u32 {
                _ = self;
                _ = source;
                _ = stage;
                _ = entry_point;
                
                // Mock GLSL to SPIR-V compilation
                var spirv_code = try self.allocator.alloc(u32, 1024);
                spirv_code[0] = 0x07230203; // SPIR-V magic number
                spirv_code[1] = 0x00010000; // Version 1.0
                spirv_code[2] = 0x00000000; // Generator
                spirv_code[3] = 1000; // Bound
                spirv_code[4] = 0x00000000; // Schema
                
                // Fill with mock instructions
                for (spirv_code[5..]) |*word| {
                    word.* = 0x12345678;
                }
                
                return spirv_code;
            }
        };
        
        pub const SpirvOptimizer = struct {
            optimization_passes: []OptimizationPass,
            target_env: TargetEnvironment,
            
            pub const OptimizationPass = enum {
                eliminate_dead_functions,
                eliminate_dead_variables,
                fold_spec_const_op_and_composite,
                unify_const,
                eliminate_dead_const,
                strength_reduction,
                simplify_instructions,
                redundancy_elimination,
                merge_return,
                inline_entry_points_exhaustive,
                eliminate_dead_code_aggressive,
                scalar_replacement,
                local_access_chain_convert,
                local_single_block_load_store_elim,
                local_single_store_elim,
                local_multi_store_elim,
                local_ssa_rewrite,
                local_redundancy_elimination,
                loop_unroll,
                merge_blocks,
                reduce_load_size,
                if_conversion,
                workaround_1209,
                convert_local_access_chains,
                vector_dce,
                loop_fission,
                loop_fusion,
                copy_propagate_arrays,
                eliminate_local_single_block,
                eliminate_local_single_store,
            };
            
            pub const TargetEnvironment = enum {
                universal_1_0,
                universal_1_1,
                universal_1_2,
                universal_1_3,
                universal_1_4,
                universal_1_5,
                universal_1_6,
                vulkan_1_0,
                vulkan_1_1,
                vulkan_1_2,
                vulkan_1_3,
                opengl_4_0,
                opengl_4_1,
                opengl_4_2,
                opengl_4_3,
                opengl_4_5,
                opencl_2_1,
                opencl_2_2,
                webgpu_0,
            };
            
            pub fn optimize(self: *SpirvOptimizer, spirv_code: []const u32) ![]u32 {
                // Apply optimization passes
                var optimized = try self.allocator.dupe(u32, spirv_code);
                
                for (self.optimization_passes) |pass| {
                    optimized = try self.applyOptimizationPass(optimized, pass);
                }
                
                return optimized;
            }
            
            fn applyOptimizationPass(self: *SpirvOptimizer, code: []u32, pass: OptimizationPass) ![]u32 {
                _ = pass;
                // Mock optimization - just return the input
                return try self.allocator.dupe(u32, code);
            }
        };
        
        pub const SassCompiler = struct {
            target_architecture: TargetArchitecture,
            optimization_level: u8,
            debug_info: bool,
            
            pub const TargetArchitecture = enum {
                sm_50,
                sm_52,
                sm_53,
                sm_60,
                sm_61,
                sm_62,
                sm_70,
                sm_72,
                sm_75,
                sm_80,
                sm_86,
                sm_87,
                sm_89,
                sm_90,
            };
            
            pub fn compileSpvToSass(self: *SassCompiler, spirv_code: []const u32) ![]u8 {
                _ = spirv_code;
                
                // Mock SPIR-V to SASS compilation
                var sass_code = try self.allocator.alloc(u8, 2048);
                
                // Fill with mock SASS instructions
                for (sass_code, 0..) |*byte, i| {
                    byte.* = @intCast(i % 256);
                }
                
                return sass_code;
            }
        };
        
        pub fn init(allocator: std.mem.Allocator) ShaderCompiler {
            return .{
                .allocator = allocator,
                .glslang_compiler = .{
                    .include_directories = &.{},
                    .defines = std.StringHashMap([]const u8).init(allocator),
                    .optimization_level = 3,
                },
                .spirv_optimizer = .{
                    .optimization_passes = &.{
                        .eliminate_dead_functions,
                        .eliminate_dead_variables,
                        .fold_spec_const_op_and_composite,
                        .unify_const,
                        .eliminate_dead_const,
                        .strength_reduction,
                        .simplify_instructions,
                        .redundancy_elimination,
                        .local_single_block_load_store_elim,
                        .local_single_store_elim,
                        .local_multi_store_elim,
                        .local_ssa_rewrite,
                        .scalar_replacement,
                        .vector_dce,
                        .eliminate_dead_code_aggressive,
                    },
                    .target_env = .vulkan_1_3,
                },
                .sass_compiler = .{
                    .target_architecture = .sm_89, // RTX 4090
                    .optimization_level = 3,
                    .debug_info = false,
                },
            };
        }
        
        pub fn deinit(self: *ShaderCompiler) void {
            self.glslang_compiler.defines.deinit();
        }
        
        pub fn compileToSpv(self: *ShaderCompiler, source_hash: u64) !ShaderCache.SpvShader {
            _ = source_hash;
            
            // Mock compilation
            const spirv_code = try self.allocator.alloc(u32, 1024);
            spirv_code[0] = 0x07230203; // SPIR-V magic
            
            return ShaderCache.SpvShader{
                .hash = source_hash,
                .stage = .fragment,
                .spirv_code = spirv_code,
                .metadata = std.mem.zeroes(ShaderCache.SpvShader.ShaderMetadata),
                .compilation_time_ms = 10.5,
                .optimization_level = 3,
            };
        }
        
        pub fn compileToSass(self: *ShaderCompiler, spv_hash: u64) !ShaderCache.SassShader {
            const sass_code = try self.sass_compiler.compileSpvToSass(&.{0x07230203});
            
            return ShaderCache.SassShader{
                .hash = spv_hash,
                .sass_code = sass_code,
                .register_count = 32,
                .shared_memory_size = 0,
                .occupancy_info = .{
                    .max_warps_per_sm = 64,
                    .max_blocks_per_sm = 32,
                    .limiting_factor = .registers,
                },
                .instruction_count = 128,
            };
        }
    };
    
    pub const OptimizerEngine = struct {
        allocator: std.mem.Allocator,
        optimization_strategies: []OptimizationStrategy,
        profiling_data: ProfilingData,
        
        pub const OptimizationStrategy = struct {
            strategy_type: StrategyType,
            enabled: bool,
            aggressiveness: u8,
            
            pub const StrategyType = enum {
                dead_code_elimination,
                constant_folding,
                loop_unrolling,
                instruction_scheduling,
                register_allocation,
                memory_coalescing,
                branch_optimization,
                vectorization,
                texture_cache_optimization,
                shared_memory_banking,
            };
        };
        
        pub const ProfilingData = struct {
            shader_execution_times: std.HashMap(u64, f32, std.hash_map.AutoContext(u64), 80),
            register_pressure_data: std.HashMap(u64, u32, std.hash_map.AutoContext(u64), 80),
            memory_access_patterns: std.HashMap(u64, MemoryAccessPattern, std.hash_map.AutoContext(u64), 80),
            
            pub const MemoryAccessPattern = struct {
                coalesced_accesses: u32,
                uncoalesced_accesses: u32,
                cache_hit_rate: f32,
                bank_conflicts: u32,
            };
        };
        
        pub fn init(allocator: std.mem.Allocator) OptimizerEngine {
            return .{
                .allocator = allocator,
                .optimization_strategies = &.{
                    .{ .strategy_type = .dead_code_elimination, .enabled = true, .aggressiveness = 3 },
                    .{ .strategy_type = .constant_folding, .enabled = true, .aggressiveness = 3 },
                    .{ .strategy_type = .loop_unrolling, .enabled = true, .aggressiveness = 2 },
                    .{ .strategy_type = .instruction_scheduling, .enabled = true, .aggressiveness = 3 },
                    .{ .strategy_type = .register_allocation, .enabled = true, .aggressiveness = 3 },
                    .{ .strategy_type = .memory_coalescing, .enabled = true, .aggressiveness = 3 },
                    .{ .strategy_type = .branch_optimization, .enabled = true, .aggressiveness = 2 },
                    .{ .strategy_type = .vectorization, .enabled = true, .aggressiveness = 2 },
                    .{ .strategy_type = .texture_cache_optimization, .enabled = true, .aggressiveness = 3 },
                    .{ .strategy_type = .shared_memory_banking, .enabled = true, .aggressiveness = 3 },
                },
                .profiling_data = .{
                    .shader_execution_times = std.HashMap(u64, f32, std.hash_map.AutoContext(u64), 80).init(allocator),
                    .register_pressure_data = std.HashMap(u64, u32, std.hash_map.AutoContext(u64), 80).init(allocator),
                    .memory_access_patterns = std.HashMap(u64, ProfilingData.MemoryAccessPattern, std.hash_map.AutoContext(u64), 80).init(allocator),
                },
            };
        }
        
        pub fn deinit(self: *OptimizerEngine) void {
            self.profiling_data.shader_execution_times.deinit();
            self.profiling_data.register_pressure_data.deinit();
            self.profiling_data.memory_access_patterns.deinit();
        }
        
        pub fn optimizeShader(
            self: *OptimizerEngine,
            shader: *const ShaderCache.SpvShader,
        ) !ShaderCache.OptimizedShader {
            var optimization_passes = std.ArrayList(ShaderCache.OptimizedShader.OptimizationPass).init(self.allocator);
            defer optimization_passes.deinit();
            
            var performance_improvement: f32 = 1.0;
            var size_reduction: f32 = 1.0;
            
            for (self.optimization_strategies) |strategy| {
                if (!strategy.enabled) continue;
                
                const pass_result = try self.applyOptimizationStrategy(shader, strategy);
                try optimization_passes.append(pass_result.pass);
                
                performance_improvement *= pass_result.performance_factor;
                size_reduction *= pass_result.size_factor;
            }
            
            return ShaderCache.OptimizedShader{
                .original_hash = shader.hash,
                .optimized_hash = shader.hash ^ 0xDEADBEEF, // Mock optimized hash
                .optimization_passes = try optimization_passes.toOwnedSlice(),
                .performance_improvement = performance_improvement,
                .size_reduction = size_reduction,
            };
        }
        
        fn applyOptimizationStrategy(
            self: *OptimizerEngine,
            shader: *const ShaderCache.SpvShader,
            strategy: OptimizationStrategy,
        ) !OptimizationResult {
            _ = self;
            _ = shader;
            
            const improvement = switch (strategy.strategy_type) {
                .dead_code_elimination => 1.05,
                .constant_folding => 1.03,
                .loop_unrolling => 1.15,
                .instruction_scheduling => 1.08,
                .register_allocation => 1.12,
                .memory_coalescing => 1.25,
                .branch_optimization => 1.10,
                .vectorization => 1.20,
                .texture_cache_optimization => 1.07,
                .shared_memory_banking => 1.18,
            };
            
            const size_factor = switch (strategy.strategy_type) {
                .dead_code_elimination => 0.95,
                .constant_folding => 0.98,
                .loop_unrolling => 1.20, // Increases size
                else => 1.0,
            };
            
            return OptimizationResult{
                .pass = .{
                    .pass_name = @tagName(strategy.strategy_type),
                    .applied = true,
                    .improvement_factor = improvement,
                },
                .performance_factor = improvement,
                .size_factor = size_factor,
            };
        }
        
        const OptimizationResult = struct {
            pass: ShaderCache.OptimizedShader.OptimizationPass,
            performance_factor: f32,
            size_factor: f32,
        };
    };
    
    pub const BinaryCache = struct {
        allocator: std.mem.Allocator,
        cache_directory: []const u8,
        memory_cache: std.HashMap(u64, CachedBinary, std.hash_map.AutoContext(u64), 80),
        
        pub const CachedBinary = struct {
            hash: u64,
            binary_data: []u8,
            metadata: BinaryMetadata,
            last_accessed: i64,
            access_count: u64,
            
            pub const BinaryMetadata = struct {
                architecture: ShaderCompiler.SassCompiler.TargetArchitecture,
                optimization_level: u8,
                compilation_flags: u32,
                driver_version: u32,
            };
        };
        
        pub fn init(allocator: std.mem.Allocator, cache_dir: []const u8) BinaryCache {
            return .{
                .allocator = allocator,
                .cache_directory = cache_dir,
                .memory_cache = std.HashMap(u64, CachedBinary, std.hash_map.AutoContext(u64), 80).init(allocator),
            };
        }
        
        pub fn deinit(self: *BinaryCache) void {
            var iter = self.memory_cache.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.value_ptr.binary_data);
            }
            self.memory_cache.deinit();
        }
        
        pub fn getOrStore(self: *BinaryCache, hash: u64, binary_data: []const u8) ![]u8 {
            if (self.memory_cache.getPtr(hash)) |cached| {
                cached.last_accessed = std.time.milliTimestamp();
                cached.access_count += 1;
                return cached.binary_data;
            }
            
            // Store in cache
            const owned_data = try self.allocator.dupe(u8, binary_data);
            const cached = CachedBinary{
                .hash = hash,
                .binary_data = owned_data,
                .metadata = .{
                    .architecture = .sm_89,
                    .optimization_level = 3,
                    .compilation_flags = 0,
                    .driver_version = 535,
                },
                .last_accessed = std.time.milliTimestamp(),
                .access_count = 1,
            };
            
            try self.memory_cache.put(hash, cached);
            
            // Also save to disk
            try self.saveToDisk(hash, binary_data);
            
            return owned_data;
        }
        
        fn saveToDisk(self: *BinaryCache, hash: u64, data: []const u8) !void {
            var path_buffer: [512]u8 = undefined;
            const path = try std.fmt.bufPrint(path_buffer[0..], "{s}/shader_{x}.bin", .{ self.cache_directory, hash });
            
            const file = std.fs.cwd().createFile(path, .{}) catch |err| switch (err) {
                error.FileNotFound => blk: {
                    try std.fs.cwd().makePath(self.cache_directory);
                    break :blk try std.fs.cwd().createFile(path, .{});
                },
                else => return err,
            };
            defer file.close();
            
            try file.writeAll(data);
        }
    };
    
    pub fn init(allocator: std.mem.Allocator, cache_dir: []const u8) !Self {
        return Self{
            .allocator = allocator,
            .shader_cache = try ShaderCache.init(allocator, cache_dir),
            .compiler = ShaderCompiler.init(allocator),
            .optimizer_engine = OptimizerEngine.init(allocator),
            .binary_cache = BinaryCache.init(allocator, cache_dir),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.shader_cache.deinit();
        self.compiler.deinit();
        self.optimizer_engine.deinit();
        self.binary_cache.deinit();
    }
    
    pub fn compileAndOptimizeShader(
        self: *Self,
        source_code: []const u8,
        stage: pipeline_cache.ShaderModule.ShaderStage,
        entry_point: []const u8,
    ) !OptimizedShaderResult {
        const source_hash = std.hash.Wyhash.hash(0, source_code);
        
        // Get or compile SPIR-V
        const spv_shader = try self.shader_cache.getOrCompileSpv(source_hash, &self.compiler);
        
        // Optimize SPIR-V
        const optimized = try self.optimizer_engine.optimizeShader(spv_shader);
        
        // Compile to SASS
        const sass_shader = try self.shader_cache.getOrCompileSass(optimized.optimized_hash, &self.compiler);
        
        // Cache the final binary
        const cached_binary = try self.binary_cache.getOrStore(sass_shader.hash, sass_shader.sass_code);
        
        return OptimizedShaderResult{
            .spv_shader = spv_shader,
            .sass_shader = sass_shader,
            .optimized_info = optimized,
            .final_binary = cached_binary,
            .performance_improvement = optimized.performance_improvement,
            .compilation_time_saved = if (self.shader_cache.spv_cache.contains(source_hash)) spv_shader.compilation_time_ms else 0,
        };
    }
    
    pub fn getOptimizationStatistics(self: *const Self) OptimizationStatistics {
        return .{
            .total_shaders_cached = @intCast(self.shader_cache.spv_cache.count()),
            .total_sass_binaries = @intCast(self.shader_cache.sass_cache.count()),
            .cache_hit_ratio = if (self.shader_cache.spv_cache.count() > 0)
                0.75 // Mock ratio
            else
                0.0,
            .average_compilation_time_ms = 12.5, // Mock average
            .total_compilation_time_saved_ms = 45000.0, // Mock saved time
            .average_performance_improvement = 1.15, // 15% improvement
            .average_size_reduction = 0.92, // 8% size reduction
        };
    }
    
    pub const OptimizedShaderResult = struct {
        spv_shader: *const ShaderCache.SpvShader,
        sass_shader: *const ShaderCache.SassShader,
        optimized_info: ShaderCache.OptimizedShader,
        final_binary: []u8,
        performance_improvement: f32,
        compilation_time_saved: f32,
    };
    
    pub const OptimizationStatistics = struct {
        total_shaders_cached: u32,
        total_sass_binaries: u32,
        cache_hit_ratio: f32,
        average_compilation_time_ms: f32,
        total_compilation_time_saved_ms: f64,
        average_performance_improvement: f32,
        average_size_reduction: f32,
    };
};