const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const command = @import("../hal/command.zig");
const memory = @import("../hal/memory.zig");

pub const NvencError = error{
    InvalidEncoder,
    InvalidSession,
    InvalidParam,
    OutOfMemory,
    UnsupportedCodec,
    UnsupportedPreset,
    UnsupportedProfile,
    EncodeFailed,
    BitstreamFull,
    InvalidInput,
    DeviceNotSupported,
    NotInitialized,
    ResourceInUse,
    MapFailed,
    UnmapFailed,
    NeedMoreInput,
    EncoderBusy,
    EventNotSet,
    InvalidCall,
    InvalidDevice,
    InvalidFormat,
};

pub const NvencCodec = enum(u8) {
    h264 = 0,
    hevc = 1,
    av1 = 2,
    
    pub fn toString(self: NvencCodec) []const u8 {
        return switch (self) {
            .h264 => "H.264",
            .hevc => "H.265/HEVC",
            .av1 => "AV1",
        };
    }
    
    pub fn isSupported(self: NvencCodec, generation: u8) bool {
        return switch (self) {
            .h264 => generation >= 1, // Maxwell+
            .hevc => generation >= 2, // Pascal+
            .av1 => generation >= 5,  // Ada Lovelace+
        };
    }
};

pub const NvencProfile = enum(u8) {
    baseline = 0,
    main = 1,
    high = 2,
    high444 = 3,
    main10 = 4,
    
    pub fn isCompatible(self: NvencProfile, codec: NvencCodec) bool {
        return switch (codec) {
            .h264 => switch (self) {
                .baseline, .main, .high, .high444 => true,
                else => false,
            },
            .hevc => switch (self) {
                .main, .main10 => true,
                else => false,
            },
            .av1 => switch (self) {
                .main => true,
                else => false,
            },
        };
    }
};

pub const NvencPreset = enum(u8) {
    p1 = 1,   // Fastest
    p2 = 2,
    p3 = 3,
    p4 = 4,   // Default
    p5 = 5,
    p6 = 6,
    p7 = 7,   // Highest Quality
    
    pub fn getQualityLevel(self: NvencPreset) u8 {
        return @intFromEnum(self);
    }
    
    pub fn getPerformanceLevel(self: NvencPreset) u8 {
        return 8 - @intFromEnum(self);
    }
};

pub const NvencRateControl = enum(u8) {
    constqp = 0,     // Constant QP
    vbr = 1,         // Variable Bitrate
    cbr = 2,         // Constant Bitrate
    vbr_minqp = 3,   // VBR with minimum QP
    ll_hq = 4,       // Low-latency high quality
    ll_hp = 5,       // Low-latency high performance
    vbr_hq = 6,      // VBR high quality
};

pub const NvencEncodeConfig = struct {
    codec: NvencCodec,
    profile: NvencProfile,
    preset: NvencPreset,
    rate_control: NvencRateControl,
    width: u32,
    height: u32,
    framerate_num: u32,
    framerate_den: u32,
    bitrate: u32,
    max_bitrate: u32,
    qp_init: u8,
    qp_min: u8,
    qp_max: u8,
    gop_length: u32,
    b_frames: u8,
    lookahead_depth: u8,
    aq_enabled: bool,
    temporal_aq: bool,
    weighted_prediction: bool,
    strict_gop: bool,
    low_delay_key_frame_scale: u8,
    
    pub fn init(codec: NvencCodec, width: u32, height: u32) NvencEncodeConfig {
        return NvencEncodeConfig{
            .codec = codec,
            .profile = switch (codec) {
                .h264 => .high,
                .hevc => .main,
                .av1 => .main,
            },
            .preset = .p4,
            .rate_control = .vbr,
            .width = width,
            .height = height,
            .framerate_num = 60,
            .framerate_den = 1,
            .bitrate = calculateDefaultBitrate(width, height, 60),
            .max_bitrate = calculateDefaultBitrate(width, height, 60) * 2,
            .qp_init = 25,
            .qp_min = 18,
            .qp_max = 51,
            .gop_length = 60,
            .b_frames = 3,
            .lookahead_depth = 16,
            .aq_enabled = true,
            .temporal_aq = true,
            .weighted_prediction = true,
            .strict_gop = false,
            .low_delay_key_frame_scale = 1,
        };
    }
    
    fn calculateDefaultBitrate(width: u32, height: u32, fps: u32) u32 {
        const pixels_per_second = width * height * fps;
        const bits_per_pixel: f32 = switch (width * height) {
            0...921600 => 0.1,    // 720p and below
            921601...2073600 => 0.08, // 1080p
            2073601...8294400 => 0.06, // 4K
            else => 0.04,             // 8K+
        };
        
        return @intFromFloat(@as(f32, @floatFromInt(pixels_per_second)) * bits_per_pixel);
    }
    
    pub fn optimize_for_streaming(self: *NvencEncodeConfig) void {
        self.preset = .p6;
        self.rate_control = .cbr;
        self.b_frames = 0;
        self.strict_gop = true;
        self.low_delay_key_frame_scale = 1;
        self.lookahead_depth = 0;
    }
    
    pub fn optimize_for_recording(self: *NvencEncodeConfig) void {
        self.preset = .p7;
        self.rate_control = .vbr_hq;
        self.b_frames = 3;
        self.lookahead_depth = 32;
        self.aq_enabled = true;
        self.temporal_aq = true;
    }
    
    pub fn optimize_for_low_latency(self: *NvencEncodeConfig) void {
        self.preset = .p1;
        self.rate_control = .ll_hp;
        self.b_frames = 0;
        self.gop_length = 30;
        self.lookahead_depth = 0;
        self.strict_gop = true;
    }
};

pub const NvencInputBuffer = struct {
    id: u32,
    gpu_address: u64,
    cpu_address: ?[*]u8,
    size: u64,
    width: u32,
    height: u32,
    pitch: u32,
    format: NvencInputFormat,
    timestamp: u64,
    locked: bool,
    
    pub fn init(id: u32, gpu_addr: u64, width: u32, height: u32, format: NvencInputFormat) NvencInputBuffer {
        const pitch = calculatePitch(width, format);
        const size = calculateBufferSize(width, height, format);
        
        return NvencInputBuffer{
            .id = id,
            .gpu_address = gpu_addr,
            .cpu_address = null,
            .size = size,
            .width = width,
            .height = height,
            .pitch = pitch,
            .format = format,
            .timestamp = 0,
            .locked = false,
        };
    }
    
    fn calculatePitch(width: u32, format: NvencInputFormat) u32 {
        const bytes_per_pixel = switch (format) {
            .nv12 => 1,
            .yuv420 => 1,
            .yuv444 => 3,
            .argb => 4,
            .abgr => 4,
        };
        
        // Align to 256 bytes for optimal memory access
        return ((width * bytes_per_pixel + 255) / 256) * 256;
    }
    
    fn calculateBufferSize(width: u32, height: u32, format: NvencInputFormat) u64 {
        const pitch = calculatePitch(width, format);
        return switch (format) {
            .nv12 => pitch * height * 3 / 2, // Y + UV/2
            .yuv420 => pitch * height * 3 / 2, // Y + U/4 + V/4
            .yuv444 => pitch * height * 3, // Y + U + V
            .argb, .abgr => pitch * height, // RGBA
        };
    }
};

pub const NvencOutputBuffer = struct {
    id: u32,
    gpu_address: u64,
    cpu_address: ?[*]u8,
    size: u64,
    capacity: u64,
    bitstream_data: []u8,
    timestamp: u64,
    frame_type: NvencFrameType,
    locked: bool,
    
    pub fn init(allocator: Allocator, id: u32, gpu_addr: u64, capacity: u64) !NvencOutputBuffer {
        const bitstream_data = try allocator.alloc(u8, capacity);
        
        return NvencOutputBuffer{
            .id = id,
            .gpu_address = gpu_addr,
            .cpu_address = null,
            .size = 0,
            .capacity = capacity,
            .bitstream_data = bitstream_data,
            .timestamp = 0,
            .frame_type = .unknown,
            .locked = false,
        };
    }
    
    pub fn deinit(self: *NvencOutputBuffer, allocator: Allocator) void {
        allocator.free(self.bitstream_data);
    }
};

pub const NvencInputFormat = enum(u8) {
    nv12 = 0,
    yuv420 = 1,
    yuv444 = 2,
    argb = 3,
    abgr = 4,
};

pub const NvencFrameType = enum(u8) {
    unknown = 0,
    i_frame = 1,
    p_frame = 2,
    b_frame = 3,
    idr_frame = 4,
};

pub const NvencSession = struct {
    id: u32,
    config: NvencEncodeConfig,
    input_buffers: std.ArrayList(NvencInputBuffer),
    output_buffers: std.ArrayList(NvencOutputBuffer),
    allocator: Allocator,
    memory_manager: *memory.DeviceMemoryManager,
    command_builder: *command.CommandBuilder,
    initialized: bool,
    frame_count: u64,
    
    const MAX_INPUT_BUFFERS = 8;
    const MAX_OUTPUT_BUFFERS = 8;
    const OUTPUT_BUFFER_SIZE = 2 * 1024 * 1024; // 2MB per output buffer
    
    pub fn init(allocator: Allocator, id: u32, config: NvencEncodeConfig, 
               memory_manager: *memory.DeviceMemoryManager, 
               command_builder: *command.CommandBuilder) !NvencSession {
        
        var session = NvencSession{
            .id = id,
            .config = config,
            .input_buffers = std.ArrayList(NvencInputBuffer).init(allocator),
            .output_buffers = std.ArrayList(NvencOutputBuffer).init(allocator),
            .allocator = allocator,
            .memory_manager = memory_manager,
            .command_builder = command_builder,
            .initialized = false,
            .frame_count = 0,
        };
        
        // Create input buffers
        for (0..MAX_INPUT_BUFFERS) |i| {
            const buffer_size = NvencInputBuffer.calculateBufferSize(config.width, config.height, .nv12);
            const region = try memory_manager.allocate(buffer_size, .device);
            
            const input_buffer = NvencInputBuffer.init(
                @intCast(i), 
                region.gpu_address, 
                config.width, 
                config.height, 
                .nv12
            );
            
            try session.input_buffers.append(input_buffer);
        }
        
        // Create output buffers
        for (0..MAX_OUTPUT_BUFFERS) |i| {
            const region = try memory_manager.allocate(OUTPUT_BUFFER_SIZE, .device);
            
            const output_buffer = try NvencOutputBuffer.init(
                allocator,
                @intCast(i),
                region.gpu_address,
                OUTPUT_BUFFER_SIZE
            );
            
            try session.output_buffers.append(output_buffer);
        }
        
        session.initialized = true;
        return session;
    }
    
    pub fn deinit(self: *NvencSession) void {
        for (self.output_buffers.items) |*buffer| {
            buffer.deinit(self.allocator);
        }
        self.input_buffers.deinit();
        self.output_buffers.deinit();
    }
    
    pub fn get_available_input_buffer(self: *NvencSession) ?*NvencInputBuffer {
        for (self.input_buffers.items) |*buffer| {
            if (!buffer.locked) {
                return buffer;
            }
        }
        return null;
    }
    
    pub fn get_available_output_buffer(self: *NvencSession) ?*NvencOutputBuffer {
        for (self.output_buffers.items) |*buffer| {
            if (!buffer.locked) {
                return buffer;
            }
        }
        return null;
    }
    
    pub fn encode_frame(self: *NvencSession, input_buffer: *NvencInputBuffer, output_buffer: *NvencOutputBuffer) !void {
        if (!self.initialized) return NvencError.NotInitialized;
        
        input_buffer.locked = true;
        output_buffer.locked = true;
        input_buffer.timestamp = std.time.nanoTimestamp();
        
        // Advanced encode command with optimizations
        const encode_cmd = try self.command_builder.createAdvancedVideoEncodeCommand(
            input_buffer.gpu_address,
            output_buffer.gpu_address,
            self.config.width,
            self.config.height,
            @intFromEnum(self.config.codec),
            .{
                .preset = @intFromEnum(self.config.preset),
                .rate_control = @intFromEnum(self.config.rate_control),
                .bitrate = self.config.bitrate,
                .qp_init = self.config.qp_init,
                .gop_length = self.config.gop_length,
                .b_frames = self.config.b_frames,
                .lookahead_depth = self.config.lookahead_depth,
                .aq_enabled = self.config.aq_enabled,
                .temporal_aq = self.config.temporal_aq,
            }
        );
        
        try self.command_builder.scheduler.submitToQueue(0, encode_cmd);
        
        self.frame_count += 1;
        
        // Improved frame type determination with B-frame optimization
        output_buffer.frame_type = self.determineFrameType();
        output_buffer.timestamp = input_buffer.timestamp;
        
        // Performance optimization: kick immediately for low-latency
        if (self.config.rate_control == .ll_hp or self.config.rate_control == .ll_hq) {
            self.command_builder.scheduler.flushAllQueues() catch {};
        }
    }
    
    fn determineFrameType(self: *NvencSession) NvencFrameType {
        // IDR frame at start and scene changes
        if (self.frame_count == 1 or (self.frame_count % (self.config.gop_length * 8) == 1)) {
            return .idr_frame;
        }
        
        // I-frame at GOP boundaries
        if (self.frame_count % self.config.gop_length == 1) {
            return .i_frame;
        }
        
        // B-frame logic for better compression
        if (self.config.b_frames > 0) {
            const pos_in_gop = self.frame_count % self.config.gop_length;
            const b_pattern = pos_in_gop % (self.config.b_frames + 1);
            
            if (b_pattern != 0 and b_pattern <= self.config.b_frames) {
                return .b_frame;
            }
        }
        
        return .p_frame;
    }
    
    pub fn flush(self: *NvencSession) !void {
        // Insert fence to wait for all pending encodes
        const fence = try self.command_builder.insert_fence(.video_encode);
        try fence.wait(5 * std.time.ns_per_s); // 5 second timeout
        
        // Unlock all buffers
        for (self.input_buffers.items) |*buffer| {
            buffer.locked = false;
        }
        
        for (self.output_buffers.items) |*buffer| {
            buffer.locked = false;
        }
    }
    
    pub fn update_config(self: *NvencSession, new_config: NvencEncodeConfig) !void {
        // Validate new configuration
        if (new_config.width != self.config.width or new_config.height != self.config.height) {
            return NvencError.InvalidParam; // Resolution changes require session recreation
        }
        
        self.config = new_config;
        // In real implementation, would update hardware encoder settings
    }
};

pub const NvencEncoder = struct {
    allocator: Allocator,
    sessions: std.ArrayList(NvencSession),
    memory_manager: *memory.DeviceMemoryManager,
    command_builder: *command.CommandBuilder,
    next_session_id: u32,
    device_generation: u8,
    max_sessions: u32,
    performance_monitor: PerformanceMonitor,
    
    const PerformanceMonitor = struct {
        frames_encoded: u64,
        total_encoding_time: u64,
        avg_bitrate: f32,
        encoder_utilization: f32,
        
        pub fn init() PerformanceMonitor {
            return PerformanceMonitor{
                .frames_encoded = 0,
                .total_encoding_time = 0,
                .avg_bitrate = 0.0,
                .encoder_utilization = 0.0,
            };
        }
        
        pub fn update(self: *PerformanceMonitor, encoding_time: u64, bitrate: u32) void {
            self.frames_encoded += 1;
            self.total_encoding_time += encoding_time;
            
            // Exponential moving average for bitrate
            const alpha: f32 = 0.1;
            self.avg_bitrate = self.avg_bitrate * (1.0 - alpha) + @as(f32, @floatFromInt(bitrate)) * alpha;
            
            // Calculate utilization (simplified)
            const avg_frame_time = self.total_encoding_time / self.frames_encoded;
            self.encoder_utilization = @min(1.0, @as(f32, @floatFromInt(avg_frame_time)) / (std.time.ns_per_s / 60));
        }
    };
    
    pub fn init(allocator: Allocator, memory_manager: *memory.DeviceMemoryManager, 
               command_builder: *command.CommandBuilder, device_generation: u8) NvencEncoder {
        
        const max_sessions = switch (device_generation) {
            1, 2 => 2,   // Maxwell, Pascal: 2 sessions
            3, 4 => 3,   // Turing, Ampere: 3 sessions
            5 => 8,      // Ada Lovelace: 8 sessions
            else => 1,
        };
        
        return NvencEncoder{
            .allocator = allocator,
            .sessions = std.ArrayList(NvencSession).init(allocator),
            .memory_manager = memory_manager,
            .command_builder = command_builder,
            .next_session_id = 1,
            .device_generation = device_generation,
            .max_sessions = max_sessions,
            .performance_monitor = PerformanceMonitor.init(),
        };
    }
    
    pub fn deinit(self: *NvencEncoder) void {
        for (self.sessions.items) |*session| {
            session.deinit();
        }
        self.sessions.deinit();
    }
    
    pub fn create_session(self: *NvencEncoder, config: NvencEncodeConfig) !u32 {
        if (self.sessions.items.len >= self.max_sessions) {
            return NvencError.ResourceInUse;
        }
        
        // Validate codec support
        if (!config.codec.isSupported(self.device_generation)) {
            return NvencError.UnsupportedCodec;
        }
        
        // Validate profile compatibility
        if (!config.profile.isCompatible(config.codec)) {
            return NvencError.UnsupportedProfile;
        }
        
        const session = try NvencSession.init(
            self.allocator,
            self.next_session_id,
            config,
            self.memory_manager,
            self.command_builder
        );
        
        try self.sessions.append(session);
        
        const session_id = self.next_session_id;
        self.next_session_id += 1;
        
        return session_id;
    }
    
    pub fn destroy_session(self: *NvencEncoder, session_id: u32) !void {
        for (self.sessions.items, 0..) |*session, i| {
            if (session.id == session_id) {
                try session.flush();
                session.deinit();
                _ = self.sessions.orderedRemove(i);
                return;
            }
        }
        return NvencError.InvalidSession;
    }
    
    pub fn get_session(self: *NvencEncoder, session_id: u32) ?*NvencSession {
        for (self.sessions.items) |*session| {
            if (session.id == session_id) {
                return session;
            }
        }
        return null;
    }
    
    pub fn get_caps(_: *NvencEncoder, codec: NvencCodec) NvencCaps {
        return NvencCaps{
            .codec = codec,
            .max_width = 8192,
            .max_height = 8192,
            .max_framerate = 240,
            .supports_bframes = codec != .av1,
            .supports_lookahead = true,
            .supports_aq = true,
            .supports_temporal_aq = true,
            .supports_weighted_prediction = codec == .h264 or codec == .hevc,
            .min_bitrate = 64000,    // 64 Kbps
            .max_bitrate = 800000000, // 800 Mbps
            .level_max = switch (codec) {
                .h264 => 62,   // Level 6.2
                .hevc => 186,  // Level 6.2
                .av1 => 23,    // Level 7.3
            },
        };
    }
    
    pub fn encode_frame_async(self: *NvencEncoder, session_id: u32, input_data: []const u8, timestamp: u64) ![]u8 {
        const session = self.get_session(session_id) orelse return NvencError.InvalidSession;
        
        const input_buffer = session.get_available_input_buffer() orelse return NvencError.EncoderBusy;
        const output_buffer = session.get_available_output_buffer() orelse return NvencError.EncoderBusy;
        
        // Copy input data to GPU buffer (simplified)
        if (input_data.len > input_buffer.size) {
            return NvencError.InvalidInput;
        }
        
        // In real implementation, would DMA transfer to GPU
        // For now, just set timestamp
        input_buffer.timestamp = timestamp;
        
        try session.encode_frame(input_buffer, output_buffer);
        
        return output_buffer.bitstream_data[0..output_buffer.size];
    }
    
    pub fn flush_all_sessions(self: *NvencEncoder) !void {
        for (self.sessions.items) |*session| {
            try session.flush();
        }
    }
};

pub const NvencCaps = struct {
    codec: NvencCodec,
    max_width: u32,
    max_height: u32,
    max_framerate: u32,
    supports_bframes: bool,
    supports_lookahead: bool,
    supports_aq: bool,
    supports_temporal_aq: bool,
    supports_weighted_prediction: bool,
    min_bitrate: u32,
    max_bitrate: u32,
    level_max: u8,
};

// Streaming optimization utilities
pub const StreamingOptimizer = struct {
    target_bitrate: u32,
    current_bitrate: u32,
    frame_drops: u32,
    network_rtt: u32,
    adaptive_bitrate: bool,
    quality_metric: f32,
    encoding_time_avg: f32,
    target_frametime_ns: u64,
    
    pub fn init(target_bitrate: u32) StreamingOptimizer {
        return StreamingOptimizer{
            .target_bitrate = target_bitrate,
            .current_bitrate = target_bitrate,
            .frame_drops = 0,
            .network_rtt = 0,
            .adaptive_bitrate = true,
            .quality_metric = 1.0,
            .encoding_time_avg = 0.0,
            .target_frametime_ns = std.time.ns_per_s / 60, // 60 FPS default
        };
    }
    
    pub fn update_network_stats(self: *StreamingOptimizer, rtt: u32, packet_loss: f32) void {
        self.network_rtt = rtt;
        
        // Adjust bitrate based on network conditions
        if (packet_loss > 0.05) { // 5% packet loss
            self.current_bitrate = @max(self.current_bitrate * 80 / 100, self.target_bitrate / 2);
        } else if (packet_loss < 0.01 and rtt < 50) { // Good network
            self.current_bitrate = @min(self.current_bitrate * 105 / 100, self.target_bitrate);
        }
    }
    
    pub fn get_optimal_config(self: *StreamingOptimizer, base_config: NvencEncodeConfig) NvencEncodeConfig {
        var config = base_config;
        config.bitrate = self.current_bitrate;
        
        // Dynamic optimization based on conditions
        if (self.network_rtt > 100 or self.encoding_time_avg > @as(f32, @floatFromInt(self.target_frametime_ns)) * 0.8) {
            config.optimize_for_low_latency();
        } else if (self.quality_metric < 0.8) {
            config.optimize_for_recording(); // Better quality when network allows
        } else {
            config.optimize_for_streaming();
        }
        
        // Adaptive preset selection based on performance
        if (self.encoding_time_avg > @as(f32, @floatFromInt(self.target_frametime_ns)) * 0.9) {
            config.preset = .p1; // Fastest
        } else if (self.encoding_time_avg < @as(f32, @floatFromInt(self.target_frametime_ns)) * 0.5) {
            config.preset = .p6; // Higher quality when time allows
        }
        
        return config;
    }
    
    pub fn update_encoding_stats(self: *StreamingOptimizer, encoding_time_ns: u64, output_size: u64) void {
        const encoding_time_f = @as(f32, @floatFromInt(encoding_time_ns));
        
        // Exponential moving average for encoding time
        self.encoding_time_avg = self.encoding_time_avg * 0.9 + encoding_time_f * 0.1;
        
        // Quality metric based on compression efficiency
        const target_size = self.current_bitrate / 8; // Convert to bytes per second
        self.quality_metric = @min(1.0, @as(f32, @floatFromInt(target_size)) / @as(f32, @floatFromInt(output_size)));
    }
};

// Test functions
test "nvenc configuration" {
    const config = NvencEncodeConfig.init(.h264, 1920, 1080);
    try std.testing.expect(config.width == 1920);
    try std.testing.expect(config.height == 1080);
    try std.testing.expect(config.codec == .h264);
}

test "nvenc encoder initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var command_scheduler = try command.CommandScheduler.init(allocator);
    defer command_scheduler.deinit();
    
    var command_builder = command.CommandBuilder.init(&command_scheduler, allocator);
    var memory_manager = memory.DeviceMemoryManager.init(allocator, 8 * 1024 * 1024 * 1024);
    defer memory_manager.deinit();
    
    var encoder = NvencEncoder.init(allocator, &memory_manager, &command_builder, 5);
    defer encoder.deinit();
    
    const config = NvencEncodeConfig.init(.h264, 1920, 1080);
    const session_id = try encoder.create_session(config);
    
    try std.testing.expect(session_id == 1);
    
    const caps = encoder.get_caps(.h264);
    try std.testing.expect(caps.max_width >= 1920);
    try std.testing.expect(caps.max_height >= 1080);
}