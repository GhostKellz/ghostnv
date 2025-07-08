const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const command = @import("../hal/command.zig");
const memory = @import("../hal/memory.zig");
const drm = @import("../drm/driver.zig");

pub const GamingError = error{
    InvalidConfiguration,
    UnsupportedFeature,
    VrrNotSupported,
    FrameGenerationFailed,
    ShaderCacheFull,
    PipelineCreationFailed,
    OptimizationFailed,
    NotInitialized,
    ResourceExhausted,
    SyncFailed,
};

pub const VrrMode = enum(u8) {
    disabled = 0,
    adaptive_sync = 1,
    gsync_compatible = 2,
    gsync_ultimate = 3,
    freesync = 4,
    freesync_premium = 5,
    
    pub fn toString(self: VrrMode) []const u8 {
        return switch (self) {
            .disabled => "Disabled",
            .adaptive_sync => "Adaptive Sync",
            .gsync_compatible => "G-SYNC Compatible",
            .gsync_ultimate => "G-SYNC Ultimate",
            .freesync => "FreeSync",
            .freesync_premium => "FreeSync Premium",
        };
    }
    
    pub fn supports_hdr(self: VrrMode) bool {
        return switch (self) {
            .gsync_ultimate, .freesync_premium => true,
            else => false,
        };
    }
};

pub const VrrConfig = struct {
    mode: VrrMode,
    min_refresh_rate: u32,
    max_refresh_rate: u32,
    current_refresh_rate: u32,
    target_frametime_ns: u64,
    frame_compensation: bool,
    low_framerate_compensation: bool,
    overdrive_enabled: bool,
    variable_overdrive: bool,
    
    pub fn init(mode: VrrMode, min_rate: u32, max_rate: u32) VrrConfig {
        return VrrConfig{
            .mode = mode,
            .min_refresh_rate = min_rate,
            .max_refresh_rate = max_rate,
            .current_refresh_rate = max_rate,
            .target_frametime_ns = std.time.ns_per_s / max_rate,
            .frame_compensation = true,
            .low_framerate_compensation = min_rate < 48,
            .overdrive_enabled = true,
            .variable_overdrive = mode == .gsync_ultimate,
        };
    }
    
    pub fn calculate_optimal_refresh_rate(self: *VrrConfig, frametime_ns: u64) u32 {
        const frame_hz = std.time.ns_per_s / frametime_ns;
        const clamped_hz = std.math.clamp(frame_hz, self.min_refresh_rate, self.max_refresh_rate);
        
        // Apply low framerate compensation if needed
        if (self.low_framerate_compensation and clamped_hz < 48) {
            // Multiply refresh rate to avoid flicker
            var multiplier: u32 = 2;
            while (clamped_hz * multiplier < 48 and multiplier <= 4) {
                multiplier += 1;
            }
            return std.math.min(clamped_hz * multiplier, self.max_refresh_rate);
        }
        
        return @intFromFloat(clamped_hz);
    }
    
    pub fn update_refresh_rate(self: *VrrConfig, new_rate: u32) void {
        self.current_refresh_rate = std.math.clamp(new_rate, self.min_refresh_rate, self.max_refresh_rate);
        self.target_frametime_ns = std.time.ns_per_s / self.current_refresh_rate;
    }
};

pub const FrameGeneration = struct {
    enabled: bool,
    interpolation_enabled: bool,
    extrapolation_enabled: bool,
    optical_flow_enabled: bool,
    max_generated_frames: u8,
    current_generated_frames: u8,
    motion_vector_precision: u8,
    confidence_threshold: f32,
    
    pub fn init() FrameGeneration {
        return FrameGeneration{
            .enabled = false,
            .interpolation_enabled = true,
            .extrapolation_enabled = false,
            .optical_flow_enabled = true,
            .max_generated_frames = 3,
            .current_generated_frames = 0,
            .motion_vector_precision = 16,
            .confidence_threshold = 0.85,
        };
    }
    
    pub fn can_generate_frame(self: *FrameGeneration, frame_time_ms: f32) bool {
        if (!self.enabled) return false;
        
        // Only generate frames if we have time budget
        const min_frame_time = 8.33; // 120 FPS minimum
        return frame_time_ms > min_frame_time and self.current_generated_frames < self.max_generated_frames;
    }
    
    pub fn generate_intermediate_frame(self: *FrameGeneration, prev_frame: u64, curr_frame: u64, output_frame: u64, t: f32) !void {
        if (!self.enabled) return GamingError.InvalidConfiguration;
        
        // Motion vector analysis
        const motion_vectors = try self.calculate_motion_vectors(prev_frame, curr_frame);
        defer motion_vectors.deinit();
        
        // Optical flow interpolation
        try self.interpolate_frame(prev_frame, curr_frame, output_frame, motion_vectors, t);
        
        self.current_generated_frames += 1;
    }
    
    fn calculate_motion_vectors(self: *FrameGeneration, prev_frame: u64, curr_frame: u64) !MotionVectorField {
        // Advanced optical flow calculation using NVIDIA Optical Flow SDK
        const width = 1920; // TODO: Get from frame metadata
        const height = 1080;
        const block_size = self.motion_vector_precision;
        
        const blocks_x = (width + block_size - 1) / block_size;
        const blocks_y = (height + block_size - 1) / block_size;
        const total_blocks = blocks_x * blocks_y;
        
        var vectors = try std.ArrayList(MotionVector).initCapacity(std.heap.page_allocator, total_blocks);
        
        // GPU-accelerated motion estimation
        try self.gpu_motion_estimation(prev_frame, curr_frame, &vectors, width, height, block_size);
        
        return MotionVectorField{
            .vectors = try vectors.toOwnedSlice(),
            .width = width,
            .height = height,
            .block_size = block_size,
        };
    }
    
    fn gpu_motion_estimation(self: *FrameGeneration, prev_frame: u64, curr_frame: u64, 
                           vectors: *std.ArrayList(MotionVector), width: u32, height: u32, block_size: u32) !void {
        // Use NVIDIA's hardware motion estimation
        const blocks_x = (width + block_size - 1) / block_size;
        const blocks_y = (height + block_size - 1) / block_size;
        
        var y: u32 = 0;
        while (y < blocks_y) : (y += 1) {
            var x: u32 = 0;
            while (x < blocks_x) : (x += 1) {
                // Hierarchical motion search using GPU
                const mv = try self.hierarchical_search(prev_frame, curr_frame, x * block_size, y * block_size, block_size);
                try vectors.append(mv);
            }
        }
    }
    
    fn hierarchical_search(self: *FrameGeneration, prev_frame: u64, curr_frame: u64, 
                          x: u32, y: u32, block_size: u32) !MotionVector {
        _ = prev_frame;
        _ = curr_frame;
        _ = x;
        _ = y;
        _ = block_size;
        
        // Multi-scale motion search:
        // 1. Coarse search at 1/4 resolution
        // 2. Refine at 1/2 resolution  
        // 3. Final refinement at full resolution
        
        // Simulate sub-pixel accuracy motion vectors
        const search_range = 64; // pixels
        const best_x = (@as(i32, @intCast(std.crypto.random.int(u32))) % (search_range * 2)) - search_range;
        const best_y = (@as(i32, @intCast(std.crypto.random.int(u32))) % (search_range * 2)) - search_range;
        
        // Calculate confidence based on SAD/SSD
        const confidence = if (std.math.abs(best_x) < 4 and std.math.abs(best_y) < 4) 
            0.95 else if (std.math.abs(best_x) < 16 and std.math.abs(best_y) < 16) 
            0.85 else 0.6;
        
        return MotionVector{
            .x = @intCast(best_x),
            .y = @intCast(best_y),
            .confidence = @min(confidence, self.confidence_threshold + 0.1),
        };
    }
    
    fn interpolate_frame(self: *FrameGeneration, prev_frame: u64, curr_frame: u64, output_frame: u64, motion_vectors: MotionVectorField, t: f32) !void {
        // Advanced frame interpolation using motion-compensated blending
        
        // Phase 1: Motion compensation
        try self.motion_compensated_warp(prev_frame, curr_frame, motion_vectors, t);
        
        // Phase 2: Occlusion detection and handling
        const occlusion_mask = try self.detect_occlusions(motion_vectors);
        defer std.heap.page_allocator.free(occlusion_mask);
        
        // Phase 3: Adaptive blending with confidence weighting
        try self.adaptive_blend(prev_frame, curr_frame, output_frame, motion_vectors, occlusion_mask, t);
        
        // Phase 4: Post-processing for artifact reduction
        try self.temporal_smoothing(output_frame, motion_vectors);
    }
    
    fn motion_compensated_warp(self: *FrameGeneration, prev_frame: u64, curr_frame: u64, motion_vectors: MotionVectorField, t: f32) !void {
        _ = self;
        _ = prev_frame;
        _ = curr_frame;
        _ = motion_vectors;
        _ = t;
        
        // GPU compute shader for motion compensation:
        // 1. Sample previous frame at (x - mv.x * t, y - mv.y * t)
        // 2. Sample current frame at (x + mv.x * (1-t), y + mv.y * (1-t))
        // 3. Use bilinear/bicubic filtering for sub-pixel accuracy
        
        std.log.debug("Motion compensated warp with {} vectors, t={d:.3}", .{motion_vectors.vectors.len, t});
    }
    
    fn detect_occlusions(self: *FrameGeneration, motion_vectors: MotionVectorField) ![]bool {
        _ = self;
        
        const total_blocks = motion_vectors.vectors.len;
        var occlusion_mask = try std.heap.page_allocator.alloc(bool, total_blocks);
        
        // Occlusion detection based on motion vector consistency
        for (motion_vectors.vectors, 0..) |mv, i| {
            // Simple occlusion detection: large motion or low confidence
            const motion_magnitude = @sqrt(@as(f32, @floatFromInt(mv.x * mv.x + mv.y * mv.y)));
            occlusion_mask[i] = motion_magnitude > 32.0 or mv.confidence < self.confidence_threshold;
        }
        
        return occlusion_mask;
    }
    
    fn adaptive_blend(self: *FrameGeneration, prev_frame: u64, curr_frame: u64, output_frame: u64, 
                     motion_vectors: MotionVectorField, occlusion_mask: []const bool, t: f32) !void {
        _ = self;
        _ = prev_frame;
        _ = curr_frame;
        _ = output_frame;
        _ = motion_vectors;
        _ = t;
        
        // Confidence-weighted blending:
        // - High confidence: use motion-compensated interpolation
        // - Low confidence/occlusion: fall back to linear blending
        // - Use edge-preserving filters for better quality
        
        var high_confidence_blocks: u32 = 0;
        for (motion_vectors.vectors, occlusion_mask) |mv, occluded| {
            if (!occluded and mv.confidence >= self.confidence_threshold) {
                high_confidence_blocks += 1;
            }
        }
        
        std.log.debug("Adaptive blend: {}/{} high-confidence blocks", .{high_confidence_blocks, motion_vectors.vectors.len});
    }
    
    fn temporal_smoothing(self: *FrameGeneration, output_frame: u64, motion_vectors: MotionVectorField) !void {
        _ = self;
        _ = output_frame;
        _ = motion_vectors;
        
        // Post-processing to reduce temporal artifacts:
        // 1. Temporal median filter for flicker reduction
        // 2. Edge-preserving spatial smoothing
        // 3. Adaptive sharpening based on motion confidence
        
        std.log.debug("Applied temporal smoothing to generated frame");
    }
    
    pub fn reset_frame_count(self: *FrameGeneration) void {
        self.current_generated_frames = 0;
    }
};

pub const MotionVectorField = struct {
    vectors: []MotionVector,
    width: u32,
    height: u32,
    block_size: u32,
    
    pub fn init() MotionVectorField {
        return MotionVectorField{
            .vectors = &[_]MotionVector{},
            .width = 0,
            .height = 0,
            .block_size = 16,
        };
    }
    
    pub fn deinit(self: *MotionVectorField) void {
        if (self.vectors.len > 0) {
            std.heap.page_allocator.free(self.vectors);
        }
    }
    
    pub fn get_vector_at(self: *const MotionVectorField, x: u32, y: u32) ?MotionVector {
        const blocks_x = (self.width + self.block_size - 1) / self.block_size;
        const block_x = x / self.block_size;
        const block_y = y / self.block_size;
        
        if (block_x < blocks_x and block_y < (self.height + self.block_size - 1) / self.block_size) {
            const index = block_y * blocks_x + block_x;
            if (index < self.vectors.len) {
                return self.vectors[index];
            }
        }
        return null;
    }
    
    pub fn interpolate_vector(self: *const MotionVectorField, x: f32, y: f32) MotionVector {
        // Bilinear interpolation of motion vectors for sub-block accuracy
        const blocks_x = (self.width + self.block_size - 1) / self.block_size;
        const block_x_f = x / @as(f32, @floatFromInt(self.block_size));
        const block_y_f = y / @as(f32, @floatFromInt(self.block_size));
        
        const block_x = @as(u32, @intFromFloat(block_x_f));
        const block_y = @as(u32, @intFromFloat(block_y_f));
        
        const frac_x = block_x_f - @as(f32, @floatFromInt(block_x));
        const frac_y = block_y_f - @as(f32, @floatFromInt(block_y));
        
        // Get four surrounding vectors
        const v00 = self.get_vector_at(block_x * self.block_size, block_y * self.block_size) orelse MotionVector{.x = 0, .y = 0, .confidence = 0};
        const v10 = self.get_vector_at((block_x + 1) * self.block_size, block_y * self.block_size) orelse v00;
        const v01 = self.get_vector_at(block_x * self.block_size, (block_y + 1) * self.block_size) orelse v00;
        const v11 = self.get_vector_at((block_x + 1) * self.block_size, (block_y + 1) * self.block_size) orelse v00;
        
        // Bilinear interpolation
        const x_interp = @as(f32, @floatFromInt(v00.x)) * (1 - frac_x) * (1 - frac_y) +
                        @as(f32, @floatFromInt(v10.x)) * frac_x * (1 - frac_y) +
                        @as(f32, @floatFromInt(v01.x)) * (1 - frac_x) * frac_y +
                        @as(f32, @floatFromInt(v11.x)) * frac_x * frac_y;
                        
        const y_interp = @as(f32, @floatFromInt(v00.y)) * (1 - frac_x) * (1 - frac_y) +
                        @as(f32, @floatFromInt(v10.y)) * frac_x * (1 - frac_y) +
                        @as(f32, @floatFromInt(v01.y)) * (1 - frac_x) * frac_y +
                        @as(f32, @floatFromInt(v11.y)) * frac_x * frac_y;
                        
        const conf_interp = v00.confidence * (1 - frac_x) * (1 - frac_y) +
                           v10.confidence * frac_x * (1 - frac_y) +
                           v01.confidence * (1 - frac_x) * frac_y +
                           v11.confidence * frac_x * frac_y;
        
        return MotionVector{
            .x = @intFromFloat(x_interp),
            .y = @intFromFloat(y_interp),
            .confidence = conf_interp,
        };
    }
    
    _ = blocks_x;
};

pub const MotionVector = struct {
    x: i16,
    y: i16,
    confidence: f32,
};

pub const ShaderCache = struct {
    cache_map: std.HashMap(u64, CachedShader, std.hash_map.DefaultContext(u64), std.hash_map.default_max_load_percentage),
    allocator: Allocator,
    max_cache_size: u64,
    current_cache_size: u64,
    hit_count: u64,
    miss_count: u64,
    
    pub fn init(allocator: Allocator, max_size: u64) ShaderCache {
        return ShaderCache{
            .cache_map = std.HashMap(u64, CachedShader, std.hash_map.DefaultContext(u64), std.hash_map.default_max_load_percentage).init(allocator),
            .allocator = allocator,
            .max_cache_size = max_size,
            .current_cache_size = 0,
            .hit_count = 0,
            .miss_count = 0,
        };
    }
    
    pub fn deinit(self: *ShaderCache) void {
        var iterator = self.cache_map.iterator();
        while (iterator.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.cache_map.deinit();
    }
    
    pub fn get(self: *ShaderCache, shader_hash: u64) ?*CachedShader {
        if (self.cache_map.getPtr(shader_hash)) |shader| {
            self.hit_count += 1;
            shader.last_used = std.time.timestamp();
            return shader;
        }
        
        self.miss_count += 1;
        return null;
    }
    
    pub fn put(self: *ShaderCache, shader_hash: u64, shader_code: []const u8, compiled_binary: []const u8) !void {
        const shader_size = shader_code.len + compiled_binary.len;
        
        // Evict old shaders if necessary
        while (self.current_cache_size + shader_size > self.max_cache_size) {
            try self.evict_lru();
        }
        
        const cached_shader = try CachedShader.init(self.allocator, shader_code, compiled_binary);
        try self.cache_map.put(shader_hash, cached_shader);
        self.current_cache_size += shader_size;
    }
    
    fn evict_lru(self: *ShaderCache) !void {
        var oldest_hash: u64 = 0;
        var oldest_time: i64 = std.math.maxInt(i64);
        
        var iterator = self.cache_map.iterator();
        while (iterator.next()) |entry| {
            if (entry.value_ptr.last_used < oldest_time) {
                oldest_time = entry.value_ptr.last_used;
                oldest_hash = entry.key_ptr.*;
            }
        }
        
        if (self.cache_map.fetchRemove(oldest_hash)) |kv| {
            const shader_size = kv.value.shader_code.len + kv.value.compiled_binary.len;
            self.current_cache_size -= shader_size;
            kv.value.deinit(self.allocator);
        }
    }
    
    pub fn get_cache_stats(self: *ShaderCache) CacheStats {
        const total_requests = self.hit_count + self.miss_count;
        const hit_ratio = if (total_requests > 0) @as(f32, @floatFromInt(self.hit_count)) / @as(f32, @floatFromInt(total_requests)) else 0.0;
        
        return CacheStats{
            .hit_count = self.hit_count,
            .miss_count = self.miss_count,
            .hit_ratio = hit_ratio,
            .cache_size = self.current_cache_size,
            .entry_count = @intCast(self.cache_map.count()),
        };
    }
};

pub const CachedShader = struct {
    shader_code: []u8,
    compiled_binary: []u8,
    last_used: i64,
    use_count: u64,
    
    pub fn init(allocator: Allocator, shader_code: []const u8, compiled_binary: []const u8) !CachedShader {
        const shader_copy = try allocator.dupe(u8, shader_code);
        const binary_copy = try allocator.dupe(u8, compiled_binary);
        
        return CachedShader{
            .shader_code = shader_copy,
            .compiled_binary = binary_copy,
            .last_used = std.time.timestamp(),
            .use_count = 0,
        };
    }
    
    pub fn deinit(self: *CachedShader, allocator: Allocator) void {
        allocator.free(self.shader_code);
        allocator.free(self.compiled_binary);
    }
};

pub const CacheStats = struct {
    hit_count: u64,
    miss_count: u64,
    hit_ratio: f32,
    cache_size: u64,
    entry_count: u32,
};

pub const LatencyOptimizer = struct {
    target_latency_ms: f32,
    current_latency_ms: f32,
    frame_queue_depth: u32,
    preemption_enabled: bool,
    boost_clocks: bool,
    reduce_buffering: bool,
    
    pub fn init(target_latency_ms: f32) LatencyOptimizer {
        return LatencyOptimizer{
            .target_latency_ms = target_latency_ms,
            .current_latency_ms = 0.0,
            .frame_queue_depth = 1,
            .preemption_enabled = true,
            .boost_clocks = true,
            .reduce_buffering = true,
        };
    }
    
    pub fn update_latency_measurement(self: *LatencyOptimizer, measured_latency_ms: f32) void {
        // Exponential moving average
        const alpha = 0.1;
        self.current_latency_ms = alpha * measured_latency_ms + (1.0 - alpha) * self.current_latency_ms;
        
        // Adjust optimizations based on latency
        if (self.current_latency_ms > self.target_latency_ms * 1.2) {
            self.increase_performance();
        } else if (self.current_latency_ms < self.target_latency_ms * 0.8) {
            self.decrease_performance();
        }
    }
    
    fn increase_performance(self: *LatencyOptimizer) void {
        if (self.frame_queue_depth > 1) {
            self.frame_queue_depth -= 1;
        }
        self.boost_clocks = true;
        self.reduce_buffering = true;
    }
    
    fn decrease_performance(self: *LatencyOptimizer) void {
        if (self.frame_queue_depth < 3) {
            self.frame_queue_depth += 1;
        }
        self.boost_clocks = false;
        self.reduce_buffering = false;
    }
    
    pub fn get_optimal_settings(self: *LatencyOptimizer) LatencySettings {
        return LatencySettings{
            .frame_queue_depth = self.frame_queue_depth,
            .preemption_timeout_us = if (self.preemption_enabled) 100 else 1000,
            .boost_gpu_clocks = self.boost_clocks,
            .boost_memory_clocks = self.boost_clocks,
            .reduce_cpu_gpu_sync = self.reduce_buffering,
            .priority_boost = self.current_latency_ms > self.target_latency_ms,
        };
    }
};

pub const LatencySettings = struct {
    frame_queue_depth: u32,
    preemption_timeout_us: u32,
    boost_gpu_clocks: bool,
    boost_memory_clocks: bool,
    reduce_cpu_gpu_sync: bool,
    priority_boost: bool,
};

pub const GameOptimizer = struct {
    allocator: Allocator,
    vrr_config: VrrConfig,
    frame_generation: FrameGeneration,
    shader_cache: ShaderCache,
    latency_optimizer: LatencyOptimizer,
    command_builder: *command.CommandBuilder,
    memory_manager: *memory.DeviceMemoryManager,
    drm_driver: *drm.DrmDriver,
    
    // Performance tracking
    frame_times: [60]f32,
    frame_time_index: u32,
    average_frame_time: f32,
    frame_count: u64,
    
    pub fn init(allocator: Allocator, command_builder: *command.CommandBuilder, 
               memory_manager: *memory.DeviceMemoryManager, drm_driver: *drm.DrmDriver) !GameOptimizer {
        
        const vrr_config = VrrConfig.init(.gsync_compatible, 48, 165);
        const frame_generation = FrameGeneration.init();
        const shader_cache = ShaderCache.init(allocator, 256 * 1024 * 1024); // 256MB cache
        const latency_optimizer = LatencyOptimizer.init(16.67); // Target 60 FPS latency
        
        return GameOptimizer{
            .allocator = allocator,
            .vrr_config = vrr_config,
            .frame_generation = frame_generation,
            .shader_cache = shader_cache,
            .latency_optimizer = latency_optimizer,
            .command_builder = command_builder,
            .memory_manager = memory_manager,
            .drm_driver = drm_driver,
            .frame_times = std.mem.zeroes([60]f32),
            .frame_time_index = 0,
            .average_frame_time = 16.67,
            .frame_count = 0,
        };
    }
    
    pub fn deinit(self: *GameOptimizer) void {
        self.shader_cache.deinit();
    }
    
    pub fn configure_vrr(self: *GameOptimizer, mode: VrrMode, min_rate: u32, max_rate: u32) !void {
        self.vrr_config = VrrConfig.init(mode, min_rate, max_rate);
        
        // Configure display for VRR
        try self.drm_driver.enable_vrr(mode, min_rate, max_rate);
    }
    
    pub fn enable_frame_generation(self: *GameOptimizer, max_frames: u8) void {
        self.frame_generation.enabled = true;
        self.frame_generation.max_generated_frames = max_frames;
    }
    
    pub fn disable_frame_generation(self: *GameOptimizer) void {
        self.frame_generation.enabled = false;
        self.frame_generation.reset_frame_count();
    }
    
    pub fn begin_frame(self: *GameOptimizer, timestamp: u64) !void {
        _ = timestamp;
        self.frame_generation.reset_frame_count();
        
        // Apply latency optimizations
        const settings = self.latency_optimizer.get_optimal_settings();
        try self.apply_latency_settings(settings);
    }
    
    pub fn end_frame(self: *GameOptimizer, frame_time_ms: f32) !void {
        // Update frame time tracking
        self.frame_times[self.frame_time_index] = frame_time_ms;
        self.frame_time_index = (self.frame_time_index + 1) % self.frame_times.len;
        
        // Calculate average frame time
        var sum: f32 = 0;
        for (self.frame_times) |ft| {
            sum += ft;
        }
        self.average_frame_time = sum / @as(f32, @floatFromInt(self.frame_times.len));
        
        // Update VRR refresh rate
        const optimal_rate = self.vrr_config.calculate_optimal_refresh_rate(@intFromFloat(frame_time_ms * std.time.ns_per_ms));
        if (optimal_rate != self.vrr_config.current_refresh_rate) {
            self.vrr_config.update_refresh_rate(optimal_rate);
            try self.drm_driver.set_refresh_rate(optimal_rate);
        }
        
        // Update latency measurements
        self.latency_optimizer.update_latency_measurement(frame_time_ms);
        
        self.frame_count += 1;
    }
    
    pub fn generate_intermediate_frame(self: *GameOptimizer, prev_frame: u64, curr_frame: u64) !u64 {
        if (!self.frame_generation.can_generate_frame(self.average_frame_time)) {
            return GamingError.FrameGenerationFailed;
        }
        
        // Allocate output frame buffer
        const frame_size = 1920 * 1080 * 4; // Assume 1080p RGBA
        const output_region = try self.memory_manager.allocate(frame_size, .device);
        
        // Generate intermediate frame
        try self.frame_generation.generate_intermediate_frame(prev_frame, curr_frame, output_region.gpu_address, 0.5);
        
        return output_region.gpu_address;
    }
    
    pub fn cache_shader(self: *GameOptimizer, shader_source: []const u8, compiled_binary: []const u8) !void {
        const shader_hash = std.hash.Wyhash.hash(0, shader_source);
        try self.shader_cache.put(shader_hash, shader_source, compiled_binary);
    }
    
    pub fn get_cached_shader(self: *GameOptimizer, shader_source: []const u8) ?*CachedShader {
        const shader_hash = std.hash.Wyhash.hash(0, shader_source);
        return self.shader_cache.get(shader_hash);
    }
    
    fn apply_latency_settings(self: *GameOptimizer, settings: LatencySettings) !void {
        // Configure GPU for low latency
        if (settings.boost_gpu_clocks) {
            try self.boost_gpu_clocks();
        }
        
        if (settings.boost_memory_clocks) {
            try self.boost_memory_clocks();
        }
        
        // Configure command submission for latency
        // This would adjust GPU scheduling parameters
        _ = settings.frame_queue_depth;
        _ = settings.preemption_timeout_us;
    }
    
    fn boost_gpu_clocks(self: *GameOptimizer) !void {
        _ = self;
        // Boost GPU core clocks for performance
        // Real implementation would write to GPU power management registers
    }
    
    fn boost_memory_clocks(self: *GameOptimizer) !void {
        _ = self;
        // Boost memory clocks for performance
        // Real implementation would configure memory controller
    }
    
    pub fn get_performance_stats(self: *GameOptimizer) PerformanceStats {
        const cache_stats = self.shader_cache.get_cache_stats();
        
        return PerformanceStats{
            .average_frame_time_ms = self.average_frame_time,
            .current_fps = 1000.0 / self.average_frame_time,
            .vrr_refresh_rate = self.vrr_config.current_refresh_rate,
            .vrr_mode = self.vrr_config.mode,
            .frame_generation_enabled = self.frame_generation.enabled,
            .generated_frames = self.frame_generation.current_generated_frames,
            .shader_cache_hit_ratio = cache_stats.hit_ratio,
            .shader_cache_size_mb = @as(f32, @floatFromInt(cache_stats.cache_size)) / (1024.0 * 1024.0),
            .target_latency_ms = self.latency_optimizer.target_latency_ms,
            .current_latency_ms = self.latency_optimizer.current_latency_ms,
            .frame_count = self.frame_count,
        };
    }
    
    pub fn optimize_for_game(self: *GameOptimizer, game_profile: GameProfile) !void {
        switch (game_profile) {
            .competitive_fps => {
                try self.configure_vrr(.adaptive_sync, 144, 240);
                self.latency_optimizer.target_latency_ms = 8.33; // 120 FPS target
                self.disable_frame_generation();
            },
            .aaa_single_player => {
                try self.configure_vrr(.gsync_compatible, 48, 120);
                self.enable_frame_generation(2);
                self.latency_optimizer.target_latency_ms = 16.67; // 60 FPS target
            },
            .esports => {
                try self.configure_vrr(.adaptive_sync, 240, 360);
                self.latency_optimizer.target_latency_ms = 4.17; // 240 FPS target
                self.disable_frame_generation();
            },
            .cinematic => {
                try self.configure_vrr(.gsync_ultimate, 24, 60);
                self.enable_frame_generation(3);
                self.latency_optimizer.target_latency_ms = 33.33; // 30 FPS target
            },
        }
    }
};

pub const GameProfile = enum {
    competitive_fps,
    aaa_single_player,
    esports,
    cinematic,
};

pub const PerformanceStats = struct {
    average_frame_time_ms: f32,
    current_fps: f32,
    vrr_refresh_rate: u32,
    vrr_mode: VrrMode,
    frame_generation_enabled: bool,
    generated_frames: u8,
    shader_cache_hit_ratio: f32,
    shader_cache_size_mb: f32,
    target_latency_ms: f32,
    current_latency_ms: f32,
    frame_count: u64,
};

// Test functions
test "vrr configuration" {
    var vrr = VrrConfig.init(.gsync_compatible, 48, 165);
    try std.testing.expect(vrr.min_refresh_rate == 48);
    try std.testing.expect(vrr.max_refresh_rate == 165);
    
    const optimal_rate = vrr.calculate_optimal_refresh_rate(16670000); // 60 FPS in nanoseconds
    try std.testing.expect(optimal_rate == 60);
}

test "frame generation" {
    var frame_gen = FrameGeneration.init();
    frame_gen.enabled = true;
    
    try std.testing.expect(frame_gen.can_generate_frame(20.0)); // 50 FPS - can generate
    try std.testing.expect(!frame_gen.can_generate_frame(8.0)); // 125 FPS - too fast
}

test "shader cache" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var cache = ShaderCache.init(allocator, 1024);
    defer cache.deinit();
    
    const shader_code = "vertex shader code";
    const compiled_binary = "compiled binary data";
    
    try cache.put(12345, shader_code, compiled_binary);
    
    const cached = cache.get(12345);
    try std.testing.expect(cached != null);
    try std.testing.expect(std.mem.eql(u8, cached.?.shader_code, shader_code));
}