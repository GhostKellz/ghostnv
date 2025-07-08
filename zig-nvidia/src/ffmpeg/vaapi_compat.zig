const std = @import("std");
const nvenc = @import("../nvenc/encoder.zig");
const memory = @import("../hal/memory.zig");
const command = @import("../hal/command.zig");

// VAAPI compatibility layer for FFmpeg integration
// Provides AMD-like VAAPI interface while using NVIDIA hardware

pub const VAAPIError = error{
    NotSupported,
    InitializationFailed,
    InvalidContext,
    InvalidSurface,
    InvalidBuffer,
    EncodingFailed,
    DecodingFailed,
    ResourceExhausted,
};

pub const VAProfile = enum(u32) {
    h264_baseline = 0,
    h264_main = 1,
    h264_high = 2,
    hevc_main = 3,
    hevc_main10 = 4,
    av1_main = 5,
    
    pub fn to_nvenc_codec(self: VAProfile) nvenc.NvencCodec {
        return switch (self) {
            .h264_baseline, .h264_main, .h264_high => .h264,
            .hevc_main, .hevc_main10 => .hevc,
            .av1_main => .av1,
        };
    }
    
    pub fn to_nvenc_profile(self: VAProfile) nvenc.NvencProfile {
        return switch (self) {
            .h264_baseline => .baseline,
            .h264_main => .main,
            .h264_high => .high,
            .hevc_main => .main,
            .hevc_main10 => .main10,
            .av1_main => .main,
        };
    }
};

pub const VAEntrypoint = enum(u32) {
    encode = 1,
    decode = 2,
    video_proc = 3,
};

pub const VASurfaceFormat = enum(u32) {
    nv12 = 0x3231564E,
    yv12 = 0x32315659,
    i420 = 0x30323449,
    rgba = 0x41424752,
    rgbx = 0x58424752,
    
    pub fn to_nvenc_format(self: VASurfaceFormat) nvenc.NvencInputFormat {
        return switch (self) {
            .nv12 => .nv12,
            .yv12, .i420 => .yuv420,
            .rgba => .argb,
            .rgbx => .abgr,
        };
    }
};

pub const VAConfigAttrib = struct {
    type: VAConfigAttribType,
    value: u32,
};

pub const VAConfigAttribType = enum(u32) {
    rt_format = 0,
    max_picture_width = 1,
    max_picture_height = 2,
    min_picture_width = 3,
    min_picture_height = 4,
    rate_control = 5,
    quality = 6,
};

pub const VASurface = struct {
    id: u32,
    width: u32,
    height: u32,
    format: VASurfaceFormat,
    gpu_address: u64,
    size: u64,
    nvenc_buffer: ?*nvenc.NvencInputBuffer,
    
    pub fn init(id: u32, width: u32, height: u32, format: VASurfaceFormat) VASurface {
        const size = calculateSurfaceSize(width, height, format);
        return VASurface{
            .id = id,
            .width = width,
            .height = height,
            .format = format,
            .gpu_address = 0,
            .size = size,
            .nvenc_buffer = null,
        };
    }
    
    fn calculateSurfaceSize(width: u32, height: u32, format: VASurfaceFormat) u64 {
        return switch (format) {
            .nv12 => width * height * 3 / 2,
            .yv12, .i420 => width * height * 3 / 2,
            .rgba, .rgbx => width * height * 4,
        };
    }
};

pub const VAContext = struct {
    id: u32,
    profile: VAProfile,
    entrypoint: VAEntrypoint,
    width: u32,
    height: u32,
    flags: u32,
    nvenc_session: ?u32, // NVENC session ID
    surfaces: std.ArrayList(VASurface),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, id: u32, profile: VAProfile, entrypoint: VAEntrypoint) VAContext {
        return VAContext{
            .id = id,
            .profile = profile,
            .entrypoint = entrypoint,
            .width = 0,
            .height = 0,
            .flags = 0,
            .nvenc_session = null,
            .surfaces = std.ArrayList(VASurface).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *VAContext) void {
        self.surfaces.deinit();
    }
};

pub const VAAPIDisplay = struct {
    allocator: std.mem.Allocator,
    contexts: std.HashMap(u32, VAContext, std.hash_map.AutoContext(u32), 80),
    surfaces: std.HashMap(u32, VASurface, std.hash_map.AutoContext(u32), 80),
    next_id: u32,
    nvenc_encoder: *nvenc.NvencEncoder,
    memory_manager: *memory.MemoryManager,
    
    pub fn init(allocator: std.mem.Allocator, nvenc_encoder: *nvenc.NvencEncoder, mem_manager: *memory.MemoryManager) VAAPIDisplay {
        return VAAPIDisplay{
            .allocator = allocator,
            .contexts = std.HashMap(u32, VAContext, std.HashMap.DefaultContext(u32), std.HashMap.default_max_load_percentage).init(allocator),
            .surfaces = std.HashMap(u32, VASurface, std.HashMap.DefaultContext(u32), std.HashMap.default_max_load_percentage).init(allocator),
            .next_id = 1,
            .nvenc_encoder = nvenc_encoder,
            .memory_manager = mem_manager,
        };
    }
    
    pub fn deinit(self: *VAAPIDisplay) void {
        var ctx_iter = self.contexts.iterator();
        while (ctx_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.contexts.deinit();
        self.surfaces.deinit();
    }
    
    // VAAPI-compatible API functions
    pub fn vaCreateSurfaces(self: *VAAPIDisplay, width: u32, height: u32, format: VASurfaceFormat, num_surfaces: u32) ![]u32 {
        var surface_ids = try self.allocator.alloc(u32, num_surfaces);
        
        for (0..num_surfaces) |i| {
            const id = self.next_id;
            self.next_id += 1;
            
            var surface = VASurface.init(id, width, height, format);
            
            // Allocate GPU memory for surface
            const region = try self.memory_manager.allocateVram(
                surface.size,
                4096, // Page alignment
                .texture,
                memory.MemoryFlags{ .readable = true, .writable = true }
            );
            
            surface.gpu_address = region.physical_address;
            try self.surfaces.put(id, surface);
            surface_ids[i] = id;
            
            std.log.info("VAAPI: Created surface {} ({}x{}, format: {}, VRAM: 0x{X})", .{
                id, width, height, @intFromEnum(format), surface.gpu_address
            });
        }
        
        return surface_ids;
    }
    
    pub fn vaDestroySurfaces(self: *VAAPIDisplay, surface_ids: []const u32) !void {
        for (surface_ids) |id| {
            if (self.surfaces.remove(id)) {
                std.log.info("VAAPI: Destroyed surface {}", .{id});
            }
        }
    }
    
    pub fn vaCreateContext(self: *VAAPIDisplay, profile: VAProfile, entrypoint: VAEntrypoint, 
                          width: u32, height: u32, flags: u32, surface_ids: []const u32) !u32 {
        const id = self.next_id;
        self.next_id += 1;
        
        var context = VAContext.init(self.allocator, id, profile, entrypoint);
        context.width = width;
        context.height = height;
        context.flags = flags;
        
        // Create NVENC session for encoding contexts
        if (entrypoint == .encode) {
            const nvenc_config = nvenc.NvencEncodeConfig.init(profile.to_nvenc_codec(), width, height);
            context.nvenc_session = try self.nvenc_encoder.create_session(nvenc_config);
        }
        
        // Add surfaces to context
        for (surface_ids) |surface_id| {
            if (self.surfaces.get(surface_id)) |surface| {
                try context.surfaces.append(surface);
            }
        }
        
        try self.contexts.put(id, context);
        
        std.log.info("VAAPI: Created context {} ({}x{}, profile: {}, entrypoint: {})", .{
            id, width, height, @intFromEnum(profile), @intFromEnum(entrypoint)
        });
        
        return id;
    }
    
    pub fn vaDestroyContext(self: *VAAPIDisplay, context_id: u32) !void {
        if (self.contexts.getPtr(context_id)) |context| {
            // Cleanup NVENC session
            if (context.nvenc_session) |session_id| {
                try self.nvenc_encoder.destroy_session(session_id);
            }
            
            context.deinit();
            _ = self.contexts.remove(context_id);
            
            std.log.info("VAAPI: Destroyed context {}", .{context_id});
        }
    }
    
    pub fn vaBeginPicture(self: *VAAPIDisplay, context_id: u32, surface_id: u32) !void {
        const context = self.contexts.getPtr(context_id) orelse return VAAPIError.InvalidContext;
        const surface = self.surfaces.getPtr(surface_id) orelse return VAAPIError.InvalidSurface;
        
        std.log.debug("VAAPI: Begin picture - context: {}, surface: {}", .{context_id, surface_id});
        
        // Prepare surface for encoding/decoding
        if (context.entrypoint == .encode and context.nvenc_session != null) {
            const session = self.nvenc_encoder.get_session(context.nvenc_session.?) orelse return VAAPIError.InvalidContext;
            
            // Get available input buffer and link to surface
            if (session.get_available_input_buffer()) |nvenc_buffer| {
                // Link VAAPI surface to NVENC buffer
                nvenc_buffer.gpu_address = surface.gpu_address;
                surface.nvenc_buffer = nvenc_buffer;
            }
        }
        
        _ = surface;
    }
    
    pub fn vaEndPicture(self: *VAAPIDisplay, context_id: u32) !void {
        const context = self.contexts.getPtr(context_id) orelse return VAAPIError.InvalidContext;
        
        std.log.debug("VAAPI: End picture - context: {}", .{context_id});
        
        // Finalize picture processing
        if (context.entrypoint == .encode and context.nvenc_session != null) {
            // Trigger NVENC encoding
            const session = self.nvenc_encoder.get_session(context.nvenc_session.?) orelse return VAAPIError.InvalidContext;
            
            // Find surface with linked buffer
            for (context.surfaces.items) |*surface| {
                if (surface.nvenc_buffer) |input_buffer| {
                    if (session.get_available_output_buffer()) |output_buffer| {
                        try session.encode_frame(input_buffer, output_buffer);
                        surface.nvenc_buffer = null; // Unlink
                        break;
                    }
                }
            }
        }
    }
    
    pub fn vaMapBuffer(self: *VAAPIDisplay, surface_id: u32) !*anyopaque {
        const surface = self.surfaces.getPtr(surface_id) orelse return VAAPIError.InvalidSurface;
        
        // Map GPU memory for CPU access
        const region = try self.memory_manager.getRegionByAddress(surface.gpu_address);
        try region.map();
        
        std.log.debug("VAAPI: Mapped buffer for surface {} at 0x{X}", .{surface_id, region.virtual_address.?});
        
        return @ptrFromInt(region.virtual_address.?);
    }
    
    pub fn vaUnmapBuffer(self: *VAAPIDisplay, surface_id: u32) !void {
        const surface = self.surfaces.getPtr(surface_id) orelse return VAAPIError.InvalidSurface;
        
        const region = try self.memory_manager.getRegionByAddress(surface.gpu_address);
        region.unmap();
        
        std.log.debug("VAAPI: Unmapped buffer for surface {}", .{surface_id});
        
        _ = surface;
    }
    
    // FFmpeg integration helpers
    pub fn exportToFFmpeg(self: *VAAPIDisplay, surface_id: u32) !FFmpegSurface {
        const surface = self.surfaces.getPtr(surface_id) orelse return VAAPIError.InvalidSurface;
        
        return FFmpegSurface{
            .width = surface.width,
            .height = surface.height,
            .format = surface.format,
            .gpu_address = surface.gpu_address,
            .dmabuf_fd = try self.createDmaBufFd(surface.gpu_address),
        };
    }
    
    fn createDmaBufFd(self: *VAAPIDisplay, gpu_address: u64) !i32 {
        // Create DMA-BUF file descriptor for sharing with FFmpeg
        _ = self;
        _ = gpu_address;
        
        // In real implementation, would create actual DMA-BUF FD
        return 42; // Dummy FD
    }
    
    // Capability querying (VAAPI compatibility)
    pub fn vaQueryConfigAttributes(self: *VAAPIDisplay, profile: VAProfile, entrypoint: VAEntrypoint) ![]VAConfigAttrib {
        _ = self;
        
        var attributes = std.ArrayList(VAConfigAttrib).init(self.allocator);
        
        // Report NVIDIA capabilities as VAAPI attributes
        try attributes.append(VAConfigAttrib{ .type = .rt_format, .value = @intFromEnum(VASurfaceFormat.nv12) });
        try attributes.append(VAConfigAttrib{ .type = .max_picture_width, .value = 8192 });
        try attributes.append(VAConfigAttrib{ .type = .max_picture_height, .value = 8192 });
        try attributes.append(VAConfigAttrib{ .type = .min_picture_width, .value = 128 });
        try attributes.append(VAConfigAttrib{ .type = .min_picture_height, .value = 128 });
        
        if (entrypoint == .encode) {
            try attributes.append(VAConfigAttrib{ .type = .rate_control, .value = 0x7 }); // CBR | VBR | CQP
            try attributes.append(VAConfigAttrib{ .type = .quality, .value = 7 }); // P1-P7 presets
        }
        
        std.log.info("VAAPI: Queried {} attributes for profile: {}, entrypoint: {}", .{
            attributes.items.len, @intFromEnum(profile), @intFromEnum(entrypoint)
        });
        
        return attributes.toOwnedSlice();
    }
};

pub const FFmpegSurface = struct {
    width: u32,
    height: u32,
    format: VASurfaceFormat,
    gpu_address: u64,
    dmabuf_fd: i32,
};

// FFmpeg codec integration
pub const ZigNvidiaFFmpegCodec = struct {
    vaapi_display: *VAAPIDisplay,
    context_id: u32,
    
    pub fn init(vaapi_display: *VAAPIDisplay) !ZigNvidiaFFmpegCodec {
        return ZigNvidiaFFmpegCodec{
            .vaapi_display = vaapi_display,
            .context_id = 0,
        };
    }
    
    pub fn create_encoder_context(self: *ZigNvidiaFFmpegCodec, profile: VAProfile, width: u32, height: u32) !void {
        // Create surfaces for encoding
        const surface_ids = try self.vaapi_display.vaCreateSurfaces(width, height, .nv12, 8);
        defer self.vaapi_display.allocator.free(surface_ids);
        
        // Create encoding context
        self.context_id = try self.vaapi_display.vaCreateContext(
            profile, .encode, width, height, 0, surface_ids
        );
        
        std.log.info("ZigNvidia-FFmpeg: Created encoder context {} ({}x{})", .{self.context_id, width, height});
    }
    
    pub fn encode_frame(self: *ZigNvidiaFFmpegCodec, input_data: []const u8, surface_id: u32) ![]u8 {
        // Begin encoding
        try self.vaapi_display.vaBeginPicture(self.context_id, surface_id);
        
        // Map surface and copy input data
        const mapped_ptr = try self.vaapi_display.vaMapBuffer(surface_id);
        const mapped_data: [*]u8 = @ptrCast(mapped_ptr);
        @memcpy(mapped_data[0..input_data.len], input_data);
        
        try self.vaapi_display.vaUnmapBuffer(surface_id);
        
        // End encoding (triggers NVENC)
        try self.vaapi_display.vaEndPicture(self.context_id);
        
        // Get encoded output (simplified)
        std.log.info("ZigNvidia-FFmpeg: Encoded frame using NVIDIA hardware via VAAPI");
        
        // Return dummy encoded data
        return try self.vaapi_display.allocator.dupe(u8, "NVENC_ENCODED_FRAME");
    }
    
    pub fn deinit(self: *ZigNvidiaFFmpegCodec) !void {
        if (self.context_id != 0) {
            try self.vaapi_display.vaDestroyContext(self.context_id);
        }
    }
};

// Test functions
test "vaapi compatibility" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var mem_manager = memory.MemoryManager.init(allocator);
    defer mem_manager.deinit();
    
    var command_scheduler = command.CommandScheduler.init(allocator, &mem_manager);
    defer command_scheduler.deinit();
    
    var command_builder = command.CommandBuilder.init(&command_scheduler, allocator);
    var nvenc_encoder = nvenc.NvencEncoder.init(allocator, &mem_manager, &command_builder, 5);
    defer nvenc_encoder.deinit();
    
    var vaapi_display = VAAPIDisplay.init(allocator, &nvenc_encoder, &mem_manager);
    defer vaapi_display.deinit();
    
    // Test surface creation
    const surface_ids = try vaapi_display.vaCreateSurfaces(1920, 1080, .nv12, 2);
    defer allocator.free(surface_ids);
    
    try std.testing.expect(surface_ids.len == 2);
    
    // Test context creation
    const context_id = try vaapi_display.vaCreateContext(.h264_high, .encode, 1920, 1080, 0, surface_ids);
    try std.testing.expect(context_id > 0);
    
    // Test encoding workflow
    try vaapi_display.vaBeginPicture(context_id, surface_ids[0]);
    try vaapi_display.vaEndPicture(context_id);
}