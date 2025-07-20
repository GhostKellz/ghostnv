const std = @import("std");
const gpu = @import("../hal/gpu.zig");
const memory = @import("../hal/memory.zig");
const command = @import("../hal/command.zig");
const performance = @import("performance.zig");

pub const UpscalingError = error{
    InvalidResolution,
    UnsupportedMode,
    TemporalDataMissing,
    MotionVectorError,
    ShaderCompilationFailed,
    OutOfMemory,
};

pub const UpscalingMode = enum(u8) {
    ultra_performance = 0, // 3x scale (1080p -> 4K = 720p internal)
    performance = 1,       // 2x scale (1080p -> 4K = 1080p internal)  
    balanced = 2,          // 1.7x scale (1080p -> 4K = ~1270p internal)
    quality = 3,           // 1.5x scale (1080p -> 4K = 1440p internal)
    ultra_quality = 4,     // 1.3x scale (1080p -> 4K = ~1662p internal)
    native = 5,            // 1x scale (no upscaling)
    
    pub fn getScaleFactor(self: UpscalingMode) f32 {
        return switch (self) {
            .ultra_performance => 3.0,
            .performance => 2.0,
            .balanced => 1.7,
            .quality => 1.5,
            .ultra_quality => 1.3,
            .native => 1.0,
        };
    }
    
    pub fn getRenderScale(self: UpscalingMode) f32 {
        return 1.0 / self.getScaleFactor();
    }
    
    pub fn getInternalResolution(self: UpscalingMode, target_width: u32, target_height: u32) struct { width: u32, height: u32 } {
        const scale = self.getRenderScale();
        return .{
            .width = @intFromFloat(@as(f32, @floatFromInt(target_width)) * scale),
            .height = @intFromFloat(@as(f32, @floatFromInt(target_height)) * scale),
        };
    }
    
    pub fn getName(self: UpscalingMode) []const u8 {
        return switch (self) {
            .ultra_performance => "Ultra Performance",
            .performance => "Performance", 
            .balanced => "Balanced",
            .quality => "Quality",
            .ultra_quality => "Ultra Quality",
            .native => "Native",
        };
    }
    
    pub fn getSharpness(self: UpscalingMode) f32 {
        return switch (self) {
            .ultra_performance => 0.8, // More sharpening for lower res
            .performance => 0.7,
            .balanced => 0.6,
            .quality => 0.5,
            .ultra_quality => 0.4,
            .native => 0.0,
        };
    }
};

pub const TemporalData = struct {
    previous_frame: u64,
    motion_vectors: *performance.MotionVectorField,
    jitter_offset_x: f32,
    jitter_offset_y: f32,
    frame_index: u32,
    history_length: u32,
    
    pub fn init() TemporalData {
        return .{
            .previous_frame = 0,
            .motion_vectors = undefined,
            .jitter_offset_x = 0,
            .jitter_offset_y = 0,
            .frame_index = 0,
            .history_length = 0,
        };
    }
    
    pub fn update(self: *TemporalData, frame: u64, mv_field: *performance.MotionVectorField) void {
        self.previous_frame = frame;
        self.motion_vectors = mv_field;
        self.frame_index += 1;
        self.history_length = @min(self.history_length + 1, 16);
        
        // Halton sequence for temporal jittering
        self.jitter_offset_x = halton(self.frame_index, 2) - 0.5;
        self.jitter_offset_y = halton(self.frame_index, 3) - 0.5;
    }
    
    fn halton(index: u32, base: u32) f32 {
        var result: f32 = 0;
        var f: f32 = 1;
        var i = index;
        
        while (i > 0) {
            f = f / @as(f32, @floatFromInt(base));
            result = result + f * @as(f32, @floatFromInt(i % base));
            i = i / base;
        }
        
        return result;
    }
};

pub const UpscalingPipeline = struct {
    allocator: std.mem.Allocator,
    gpu_device: *gpu.Device,
    memory_manager: *memory.DeviceMemoryManager,
    command_builder: *command.CommandBuilder,
    
    current_mode: UpscalingMode,
    target_width: u32,
    target_height: u32,
    internal_width: u32,
    internal_height: u32,
    
    temporal_data: TemporalData,
    accumulation_buffer: u64,
    depth_buffer: u64,
    motion_vector_buffer: u64,
    
    upscale_shader: u64,
    temporal_shader: u64,
    sharpen_shader: u64,
    
    const Self = @This();
    
    pub fn init(
        allocator: std.mem.Allocator,
        gpu_device: *gpu.Device,
        memory_manager: *memory.DeviceMemoryManager,
        command_builder: *command.CommandBuilder,
        target_width: u32,
        target_height: u32,
        mode: UpscalingMode,
    ) !Self {
        const internal_res = mode.getInternalResolution(target_width, target_height);
        
        var pipeline = Self{
            .allocator = allocator,
            .gpu_device = gpu_device,
            .memory_manager = memory_manager,
            .command_builder = command_builder,
            .current_mode = mode,
            .target_width = target_width,
            .target_height = target_height,
            .internal_width = internal_res.width,
            .internal_height = internal_res.height,
            .temporal_data = TemporalData.init(),
            .accumulation_buffer = 0,
            .depth_buffer = 0,
            .motion_vector_buffer = 0,
            .upscale_shader = 0,
            .temporal_shader = 0,
            .sharpen_shader = 0,
        };
        
        try pipeline.allocateBuffers();
        try pipeline.compileShaders();
        
        return pipeline;
    }
    
    pub fn deinit(self: *Self) void {
        if (self.accumulation_buffer != 0) {
            self.memory_manager.free(self.accumulation_buffer) catch {};
        }
        if (self.depth_buffer != 0) {
            self.memory_manager.free(self.depth_buffer) catch {};
        }
        if (self.motion_vector_buffer != 0) {
            self.memory_manager.free(self.motion_vector_buffer) catch {};
        }
    }
    
    pub fn setMode(self: *Self, mode: UpscalingMode) !void {
        if (mode == self.current_mode) return;
        
        self.current_mode = mode;
        const internal_res = mode.getInternalResolution(self.target_width, self.target_height);
        
        if (internal_res.width != self.internal_width or internal_res.height != self.internal_height) {
            self.internal_width = internal_res.width;
            self.internal_height = internal_res.height;
            
            // Reallocate buffers for new resolution
            self.deinit();
            try self.allocateBuffers();
        }
    }
    
    fn allocateBuffers(self: *Self) !void {
        const pixel_size = 4; // RGBA8
        const mv_size = @sizeOf(performance.MotionVector);
        
        // Accumulation buffer for temporal stability
        const accum_size = self.target_width * self.target_height * pixel_size;
        const accum_region = try self.memory_manager.allocate(accum_size, .device);
        self.accumulation_buffer = accum_region.gpu_address;
        
        // Depth buffer for depth-aware upscaling
        const depth_size = self.internal_width * self.internal_height * 4; // 32-bit depth
        const depth_region = try self.memory_manager.allocate(depth_size, .device);
        self.depth_buffer = depth_region.gpu_address;
        
        // Motion vector buffer
        const mv_blocks_x = (self.internal_width + 15) / 16;
        const mv_blocks_y = (self.internal_height + 15) / 16;
        const mv_buffer_size = mv_blocks_x * mv_blocks_y * mv_size;
        const mv_region = try self.memory_manager.allocate(mv_buffer_size, .device);
        self.motion_vector_buffer = mv_region.gpu_address;
    }
    
    fn compileShaders(self: *Self) !void {
        // In real implementation, these would compile actual GPU shaders
        // For now, we'll use placeholder addresses
        self.upscale_shader = 0x1000;
        self.temporal_shader = 0x2000;
        self.sharpen_shader = 0x3000;
    }
    
    pub fn upscale(
        self: *Self,
        input_frame: u64,
        output_frame: u64,
        depth_buffer: u64,
        motion_vectors: *performance.MotionVectorField,
    ) !void {
        if (self.current_mode == .native) {
            // Direct copy for native resolution
            try self.command_builder.copy_buffer(input_frame, output_frame, self.target_width * self.target_height * 4);
            return;
        }
        
        // Update temporal data
        self.temporal_data.update(input_frame, motion_vectors);
        
        // Phase 1: Spatial upscaling with edge-aware filtering
        try self.spatialUpscale(input_frame, output_frame, depth_buffer);
        
        // Phase 2: Temporal accumulation for stability
        if (self.temporal_data.history_length > 0) {
            try self.temporalAccumulation(output_frame);
        }
        
        // Phase 3: Sharpening pass based on mode
        const sharpness = self.current_mode.getSharpness();
        if (sharpness > 0) {
            try self.sharpenOutput(output_frame, sharpness);
        }
    }
    
    fn spatialUpscale(self: *Self, input: u64, output: u64, depth: u64) !void {
        // Configure upscaling shader
        const params = UpscaleParams{
            .input_texture = input,
            .depth_texture = depth,
            .output_texture = output,
            .input_width = self.internal_width,
            .input_height = self.internal_height,
            .output_width = self.target_width,
            .output_height = self.target_height,
            .scale_factor = self.current_mode.getScaleFactor(),
            .jitter_x = self.temporal_data.jitter_offset_x,
            .jitter_y = self.temporal_data.jitter_offset_y,
        };
        
        // Dispatch compute shader
        const threads_x = (self.target_width + 15) / 16;
        const threads_y = (self.target_height + 15) / 16;
        
        try self.command_builder.begin_compute();
        try self.command_builder.bind_compute_shader(self.upscale_shader);
        try self.command_builder.set_compute_params(&params, @sizeOf(UpscaleParams));
        try self.command_builder.dispatch_compute(threads_x, threads_y, 1);
        try self.command_builder.end_compute();
    }
    
    fn temporalAccumulation(self: *Self, frame: u64) !void {
        const params = TemporalParams{
            .current_frame = frame,
            .history_buffer = self.accumulation_buffer,
            .motion_vectors = self.motion_vector_buffer,
            .width = self.target_width,
            .height = self.target_height,
            .blend_factor = @min(0.9, @as(f32, @floatFromInt(self.temporal_data.history_length)) / 8.0),
            .motion_scale = 1.0 / self.current_mode.getScaleFactor(),
        };
        
        const threads_x = (self.target_width + 15) / 16;
        const threads_y = (self.target_height + 15) / 16;
        
        try self.command_builder.begin_compute();
        try self.command_builder.bind_compute_shader(self.temporal_shader);
        try self.command_builder.set_compute_params(&params, @sizeOf(TemporalParams));
        try self.command_builder.dispatch_compute(threads_x, threads_y, 1);
        try self.command_builder.end_compute();
        
        // Copy result back to accumulation buffer
        try self.command_builder.copy_buffer(frame, self.accumulation_buffer, self.target_width * self.target_height * 4);
    }
    
    fn sharpenOutput(self: *Self, frame: u64, sharpness: f32) !void {
        const params = SharpenParams{
            .input_output = frame,
            .width = self.target_width,
            .height = self.target_height,
            .sharpness = sharpness,
            .threshold = 0.05, // Edge detection threshold
        };
        
        const threads_x = (self.target_width + 15) / 16;
        const threads_y = (self.target_height + 15) / 16;
        
        try self.command_builder.begin_compute();
        try self.command_builder.bind_compute_shader(self.sharpen_shader);
        try self.command_builder.set_compute_params(&params, @sizeOf(SharpenParams));
        try self.command_builder.dispatch_compute(threads_x, threads_y, 1);
        try self.command_builder.end_compute();
    }
    
    pub fn getStatistics(self: *Self) UpscalingStats {
        const scale = self.current_mode.getScaleFactor();
        const pixels_rendered = self.internal_width * self.internal_height;
        const pixels_output = self.target_width * self.target_height;
        const pixel_savings = pixels_output - pixels_rendered;
        
        return .{
            .mode = self.current_mode,
            .render_resolution = .{ .width = self.internal_width, .height = self.internal_height },
            .output_resolution = .{ .width = self.target_width, .height = self.target_height },
            .scale_factor = scale,
            .pixels_saved_percent = @as(f32, @floatFromInt(pixel_savings)) / @as(f32, @floatFromInt(pixels_output)) * 100.0,
            .temporal_samples = self.temporal_data.history_length,
            .estimated_performance_gain = (scale * scale - 1.0) / (scale * scale) * 100.0,
        };
    }
};

const UpscaleParams = extern struct {
    input_texture: u64,
    depth_texture: u64,
    output_texture: u64,
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    scale_factor: f32,
    jitter_x: f32,
    jitter_y: f32,
    _padding: [8]u8 = undefined,
};

const TemporalParams = extern struct {
    current_frame: u64,
    history_buffer: u64,
    motion_vectors: u64,
    width: u32,
    height: u32,
    blend_factor: f32,
    motion_scale: f32,
};

const SharpenParams = extern struct {
    input_output: u64,
    width: u32,
    height: u32,
    sharpness: f32,
    threshold: f32,
};

pub const UpscalingStats = struct {
    mode: UpscalingMode,
    render_resolution: struct { width: u32, height: u32 },
    output_resolution: struct { width: u32, height: u32 },
    scale_factor: f32,
    pixels_saved_percent: f32,
    temporal_samples: u32,
    estimated_performance_gain: f32,
};

// Tests
test "upscaling modes" {
    const testing = std.testing;
    
    // Test scale factors
    try testing.expectEqual(@as(f32, 3.0), UpscalingMode.ultra_performance.getScaleFactor());
    try testing.expectEqual(@as(f32, 2.0), UpscalingMode.performance.getScaleFactor());
    try testing.expectEqual(@as(f32, 1.7), UpscalingMode.balanced.getScaleFactor());
    try testing.expectEqual(@as(f32, 1.5), UpscalingMode.quality.getScaleFactor());
    try testing.expectEqual(@as(f32, 1.3), UpscalingMode.ultra_quality.getScaleFactor());
    
    // Test internal resolution calculation for 4K target
    const target_4k_w = 3840;
    const target_4k_h = 2160;
    
    const perf_res = UpscalingMode.performance.getInternalResolution(target_4k_w, target_4k_h);
    try testing.expectEqual(@as(u32, 1920), perf_res.width);
    try testing.expectEqual(@as(u32, 1080), perf_res.height);
    
    const balanced_res = UpscalingMode.balanced.getInternalResolution(target_4k_w, target_4k_h);
    try testing.expectEqual(@as(u32, 2258), balanced_res.width); // ~1270p
    try testing.expectEqual(@as(u32, 1270), balanced_res.height);
    
    const quality_res = UpscalingMode.quality.getInternalResolution(target_4k_w, target_4k_h);
    try testing.expectEqual(@as(u32, 2560), quality_res.width); // 1440p
    try testing.expectEqual(@as(u32, 1440), quality_res.height);
}

test "temporal jittering" {
    const testing = std.testing;
    
    var temporal = TemporalData.init();
    
    // Test Halton sequence generates values in [-0.5, 0.5]
    var i: u32 = 0;
    while (i < 16) : (i += 1) {
        var mv_field = performance.MotionVectorField.init();
        temporal.update(i, &mv_field);
        
        try testing.expect(temporal.jitter_offset_x >= -0.5);
        try testing.expect(temporal.jitter_offset_x <= 0.5);
        try testing.expect(temporal.jitter_offset_y >= -0.5);
        try testing.expect(temporal.jitter_offset_y <= 0.5);
    }
}