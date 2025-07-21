const std = @import("std");
const memory = @import("../hal/memory.zig");
const command = @import("../hal/command.zig");
const bezel = @import("bezel_correction.zig");
const spanning = @import("spanning_manager.zig");
const timing = @import("timing_manager.zig");

/// NVIDIA Display Engine Implementation
/// Handles all display output functionality including multi-monitor, VRR, HDR
pub const DisplayEngine = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    heads: []DisplayHead,
    memory_manager: *memory.MemoryManager,
    
    // Advanced display features
    bezel_corrector: bezel.BezelCorrector,
    spanning_manager: spanning.SpanningManager,
    timing_manager: timing.TimingManager,
    
    // Hardware state
    bar0: *volatile u8,
    display_regs: *volatile DisplayRegisters,
    
    // Performance tracking
    frame_stats: FrameStats,
    
    pub fn init(allocator: std.mem.Allocator, device: *anyopaque, mem_manager: *memory.MemoryManager) !*Self {
        var self = try allocator.create(Self);
        
        // Initialize display heads (up to 4 for modern GPUs)
        const num_heads = 4;
        self.heads = try allocator.alloc(DisplayHead, num_heads);
        for (0..num_heads) |i| {
            self.heads[i] = try DisplayHead.init(allocator, @intCast(i));
        }
        
        self.* = Self{
            .allocator = allocator,
            .heads = self.heads,
            .memory_manager = mem_manager,
            .bezel_corrector = bezel.BezelCorrector.init(allocator),
            .spanning_manager = try spanning.SpanningManager.init(allocator),
            .timing_manager = timing.TimingManager.init(allocator),
            .bar0 = @ptrCast(@alignCast(device)),
            .display_regs = @ptrCast(@alignCast(@as([*]u8, @ptrCast(device)) + DISPLAY_REGS_OFFSET)),
            .frame_stats = .{},
        };
        
        // Initialize hardware
        try self.initializeHardware();
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        for (self.heads) |*head| {
            head.deinit();
        }
        self.allocator.free(self.heads);
        
        // Clean up advanced display features
        self.bezel_corrector.deinit();
        self.spanning_manager.deinit();
        self.timing_manager.deinit();
        
        self.allocator.destroy(self);
    }
    
    fn initializeHardware(self: *Self) !void {
        // Initialize display controller
        self.display_regs.control = DISPLAY_CTRL_ENABLE | DISPLAY_CTRL_RESET;
        
        // Wait for reset completion
        var timeout: u32 = 1000;
        while (timeout > 0 and (self.display_regs.status & DISPLAY_STATUS_READY) == 0) {
            std.time.sleep(1000000); // 1ms
            timeout -= 1;
        }
        
        if (timeout == 0) {
            return error.DisplayInitTimeout;
        }
        
        // Initialize each display head
        for (self.heads) |*head| {
            try head.initializeHardware(self.display_regs);
        }
        
        std.log.info("Display engine initialized with {} heads", .{self.heads.len});
    }
    
    pub fn handleInterrupt(self: *Self, status: u32) void {
        // Handle display interrupts
        if (status & DISPLAY_INTR_VBLANK) {
            self.handleVBlankInterrupt();
        }
        
        if (status & DISPLAY_INTR_FLIP_COMPLETE) {
            self.handleFlipComplete();
        }
        
        if (status & DISPLAY_INTR_HOTPLUG) {
            self.handleHotplugInterrupt();
        }
    }
    
    fn handleVBlankInterrupt(self: *Self) void {
        // Update frame statistics
        self.frame_stats.vblank_count += 1;
        
        // Process any pending flips
        for (self.heads) |*head| {
            head.processVBlank();
        }
    }
    
    fn handleFlipComplete(self: *Self) void {
        self.frame_stats.flip_count += 1;
        
        // Signal completion to waiting clients
        for (self.heads) |*head| {
            head.signalFlipComplete();
        }
    }
    
    fn handleHotplugInterrupt(self: *Self) void {
        // Detect display changes
        for (self.heads) |*head| {
            head.detectDisplays();
        }
    }
    
    pub fn setMode(self: *Self, head_id: u8, mode: DisplayMode) !void {
        if (head_id >= self.heads.len) return error.InvalidHead;
        
        const head = &self.heads[head_id];
        try head.setMode(mode);
        
        std.log.info("Set display mode on head {}: {}x{}@{}Hz", .{
            head_id,
            mode.width,
            mode.height,
            mode.refresh_rate,
        });
    }
    
    pub fn flip(self: *Self, head_id: u8, surface: *DisplaySurface) !void {
        if (head_id >= self.heads.len) return error.InvalidHead;
        
        const head = &self.heads[head_id];
        try head.flip(surface);
        
        self.frame_stats.flip_requests += 1;
    }
    
    pub fn enableVRR(self: *Self, head_id: u8, min_hz: u32, max_hz: u32) !void {
        if (head_id >= self.heads.len) return error.InvalidHead;
        
        const head = &self.heads[head_id];
        try head.enableVRR(min_hz, max_hz);
        
        std.log.info("Enabled VRR on head {}: {}Hz - {}Hz", .{ head_id, min_hz, max_hz });
    }
    
    pub fn setHDRMode(self: *Self, head_id: u8, hdr_mode: HDRMode) !void {
        if (head_id >= self.heads.len) return error.InvalidHead;
        
        const head = &self.heads[head_id];
        try head.setHDRMode(hdr_mode);
        
        std.log.info("Set HDR mode on head {}: {}", .{ head_id, hdr_mode });
    }
    
    pub fn suspendDisplay(self: *Self) !void {
        // Save current display state
        for (self.heads, 0..) |*head, i| {
            // Power down display head
            head.is_active = false;
            std.log.info("Suspended display head {}", .{i});
        }
        
        std.log.info("Display engine suspended", .{});
    }
    
    pub fn resumeDisplay(self: *Self) !void {
        // Restore display state
        for (self.heads, 0..) |*head, i| {
            // Restore display head state
            if (head.current_mode) |_| {
                head.is_active = true;
                std.log.info("Resumed display head {}", .{i});
            }
        }
        
        std.log.info("Display engine resumed", .{});
    }
};

/// Individual Display Head (CRTC)
pub const DisplayHead = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    head_id: u8,
    current_mode: ?DisplayMode,
    connected_displays: std.ArrayList(ConnectedDisplay),
    framebuffer: ?*DisplaySurface,
    
    // VRR state
    vrr_enabled: bool,
    vrr_min_hz: u32,
    vrr_max_hz: u32,
    
    // HDR state
    hdr_mode: HDRMode,
    color_space: ColorSpace,
    
    // Display state
    is_active: bool,
    
    // Hardware registers
    crtc_regs: *volatile CRTCRegisters,
    
    pub fn init(allocator: std.mem.Allocator, head_id: u8) !Self {
        return Self{
            .allocator = allocator,
            .head_id = head_id,
            .current_mode = null,
            .connected_displays = std.ArrayList(ConnectedDisplay).init(allocator),
            .framebuffer = null,
            .vrr_enabled = false,
            .vrr_min_hz = 0,
            .vrr_max_hz = 0,
            .hdr_mode = .sdr,
            .color_space = .srgb,
            .is_active = false,
            .crtc_regs = undefined,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.connected_displays.deinit();
    }
    
    pub fn initializeHardware(self: *Self, display_regs: *volatile DisplayRegisters) !void {
        const head_offset = CRTC_REGS_BASE + self.head_id * CRTC_REGS_SIZE;
        self.crtc_regs = @ptrCast(@alignCast(@as([*]u8, @ptrCast(display_regs)) + head_offset));
        
        // Reset CRTC
        self.crtc_regs.control = CRTC_CTRL_RESET;
        std.time.sleep(1000000); // 1ms
        self.crtc_regs.control = 0;
        
        // Detect connected displays
        self.detectDisplays();
    }
    
    pub fn detectDisplays(self: *Self) void {
        // Clear existing displays
        self.connected_displays.clearRetainingCapacity();
        
        // Check each output port
        for (0..4) |port| {
            if (self.isDisplayConnected(@intCast(port))) {
                const display = self.probeDisplay(@intCast(port)) catch continue;
                self.connected_displays.append(display) catch continue;
            }
        }
        
        std.log.info("Head {}: Found {} connected displays", .{ self.head_id, self.connected_displays.items.len });
    }
    
    fn isDisplayConnected(self: *Self, port: u8) bool {
        // Check hotplug detect pins
        const hotplug_status = self.crtc_regs.hotplug_status;
        const port_mask = @as(u32, 1) << port;
        return (hotplug_status & port_mask) != 0;
    }
    
    fn probeDisplay(self: *Self, port: u8) !ConnectedDisplay {
        // Read EDID
        const edid_data = try self.readEDID(port);
        const edid = try EDID.parse(edid_data);
        
        // Determine display capabilities
        const caps = DisplayCapabilities{
            .max_width = edid.max_horizontal_size,
            .max_height = edid.max_vertical_size,
            .supported_formats = &.{ .argb8888, .xrgb8888, .rgb565 },
            .vrr_capable = edid.vrr_capable,
            .vrr_min_hz = edid.vrr_min_refresh,
            .vrr_max_hz = edid.vrr_max_refresh,
            .hdr_capable = edid.hdr_capable,
            .max_luminance = edid.max_luminance,
            .connector_type = switch (port) {
                0 => .displayport,
                1 => .displayport,
                2 => .hdmi,
                3 => .hdmi,
                else => .unknown,
            },
        };
        
        return ConnectedDisplay{
            .port = port,
            .edid = edid,
            .capabilities = caps,
            .current_mode = null,
        };
    }
    
    fn readEDID(self: *Self, port: u8) ![]u8 {
        _ = self;
        _ = port;
        
        // In real implementation, read EDID via I2C/DDC
        // For now, return simulated EDID
        const edid_data = try std.testing.allocator.alloc(u8, 256);
        @memset(edid_data, 0);
        
        // EDID header
        const header = [_]u8{ 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00 };
        @memcpy(edid_data[0..8], &header);
        
        return edid_data;
    }
    
    pub fn setMode(self: *Self, mode: DisplayMode) !void {
        // Program CRTC timings
        self.crtc_regs.h_total = mode.htotal;
        self.crtc_regs.h_display = mode.width;
        self.crtc_regs.h_sync_start = mode.hsync_start;
        self.crtc_regs.h_sync_end = mode.hsync_end;
        
        self.crtc_regs.v_total = mode.vtotal;
        self.crtc_regs.v_display = mode.height;
        self.crtc_regs.v_sync_start = mode.vsync_start;
        self.crtc_regs.v_sync_end = mode.vsync_end;
        
        // Set pixel clock
        self.crtc_regs.pixel_clock = mode.pixel_clock;
        
        // Configure sync polarities
        var control = self.crtc_regs.control;
        if (mode.hsync_positive) control |= CRTC_CTRL_HSYNC_POS;
        if (mode.vsync_positive) control |= CRTC_CTRL_VSYNC_POS;
        
        self.crtc_regs.control = control | CRTC_CTRL_ENABLE;
        
        self.current_mode = mode;
    }
    
    pub fn flip(self: *Self, surface: *DisplaySurface) !void {
        // Program surface address
        self.crtc_regs.surface_address = surface.address;
        self.crtc_regs.surface_stride = surface.stride;
        self.crtc_regs.surface_format = @intFromEnum(surface.format);
        
        // Trigger flip on next vblank
        self.crtc_regs.control |= CRTC_CTRL_FLIP_PENDING;
        
        self.framebuffer = surface;
    }
    
    pub fn enableVRR(self: *Self, min_hz: u32, max_hz: u32) !void {
        if (self.connected_displays.items.len == 0) return error.NoDisplay;
        
        // Check if display supports VRR
        const display = &self.connected_displays.items[0];
        if (!display.capabilities.vrr_capable) return error.VRRNotSupported;
        
        // Program VRR parameters
        self.crtc_regs.vrr_min_vtotal = self.calculateVTotal(min_hz);
        self.crtc_regs.vrr_max_vtotal = self.calculateVTotal(max_hz);
        self.crtc_regs.vrr_control = VRR_CTRL_ENABLE | VRR_CTRL_ADAPTIVE;
        
        self.vrr_enabled = true;
        self.vrr_min_hz = min_hz;
        self.vrr_max_hz = max_hz;
    }
    
    pub fn setHDRMode(self: *Self, hdr_mode: HDRMode) !void {
        if (self.connected_displays.items.len == 0) return error.NoDisplay;
        
        const display = &self.connected_displays.items[0];
        if (hdr_mode != .sdr and !display.capabilities.hdr_capable) {
            return error.HDRNotSupported;
        }
        
        // Configure HDR parameters
        switch (hdr_mode) {
            .sdr => {
                self.crtc_regs.hdr_control = 0;
                self.color_space = .srgb;
            },
            .hdr10 => {
                self.crtc_regs.hdr_control = HDR_CTRL_ENABLE | HDR_CTRL_HDR10;
                self.crtc_regs.hdr_max_luminance = display.capabilities.max_luminance;
                self.color_space = .rec2020;
            },
            .dolby_vision => {
                self.crtc_regs.hdr_control = HDR_CTRL_ENABLE | HDR_CTRL_DOLBY_VISION;
                self.color_space = .rec2020;
            },
        }
        
        self.hdr_mode = hdr_mode;
    }
    
    pub fn processVBlank(self: *Self) void {
        if (self.crtc_regs.control & CRTC_CTRL_FLIP_PENDING) {
            // Complete the flip
            self.crtc_regs.control &= ~CRTC_CTRL_FLIP_PENDING;
            self.crtc_regs.control |= CRTC_CTRL_FLIP_COMPLETE;
        }
    }
    
    pub fn signalFlipComplete(self: *Self) void {
        _ = self;
        // Signal completion to waiting clients
        // In real implementation, wake up waiting processes
    }
    
    fn calculateVTotal(self: *Self, refresh_hz: u32) u32 {
        if (self.current_mode) |mode| {
            const pixel_clock_hz = mode.pixel_clock * 1000;
            const htotal = mode.htotal;
            return pixel_clock_hz / (htotal * refresh_hz);
        }
        return 0;
    }
};

/// Wayland Compositor Integration
pub const WaylandCompositor = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    surfaces: std.ArrayList(WaylandSurface),
    frame_callbacks: std.ArrayList(FrameCallback),
    
    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .allocator = allocator,
            .surfaces = std.ArrayList(WaylandSurface).init(allocator),
            .frame_callbacks = std.ArrayList(FrameCallback).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.surfaces.deinit();
        self.frame_callbacks.deinit();
    }
    
    pub fn handleVBlank(self: *Self) void {
        // Execute frame callbacks
        for (self.frame_callbacks.items) |callback| {
            callback.execute();
        }
        self.frame_callbacks.clearRetainingCapacity();
    }
    
    pub fn createSurface(self: *Self, width: u32, height: u32, format: PixelFormat) !*WaylandSurface {
        const surface = WaylandSurface{
            .id = @intCast(self.surfaces.items.len),
            .width = width,
            .height = height,
            .format = format,
            .buffer = null,
            .damage_regions = std.ArrayList(DamageRegion).init(self.allocator),
        };
        
        try self.surfaces.append(surface);
        return &self.surfaces.items[self.surfaces.items.len - 1];
    }
    
    pub fn commitSurface(self: *Self, surface: *WaylandSurface) !void {
        // Apply pending state
        if (surface.buffer) |buffer| {
            // Upload buffer to GPU
            try self.uploadSurfaceBuffer(surface, buffer);
        }
        
        // Clear damage regions
        surface.damage_regions.clearRetainingCapacity();
    }
    
    fn uploadSurfaceBuffer(self: *Self, surface: *WaylandSurface, buffer: *WaylandBuffer) !void {
        _ = self;
        _ = surface;
        _ = buffer;
        
        // In real implementation:
        // 1. Map buffer memory
        // 2. Copy to GPU surface
        // 3. Set up texture sampler
        // 4. Update surface descriptor
    }
};

/// Shader Compiler for 3D Graphics Pipeline
pub const ShaderCompiler = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    compiled_shaders: std.AutoHashMap(u64, CompiledShader),
    
    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .allocator = allocator,
            .compiled_shaders = std.AutoHashMap(u64, CompiledShader).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        var iterator = self.compiled_shaders.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.value_ptr.bytecode);
        }
        self.compiled_shaders.deinit();
    }
    
    pub fn compileShader(self: *Self, source: []const u8, stage: ShaderStage) !u64 {
        const hash = std.hash.CityHash64.hash(source);
        
        if (self.compiled_shaders.contains(hash)) {
            return hash;
        }
        
        // Compile shader
        const bytecode = try self.performCompilation(source, stage);
        
        const compiled = CompiledShader{
            .stage = stage,
            .bytecode = bytecode,
            .uniforms = std.ArrayList(UniformBinding).init(self.allocator),
        };
        
        try self.compiled_shaders.put(hash, compiled);
        return hash;
    }
    
    fn performCompilation(self: *Self, source: []const u8, stage: ShaderStage) ![]u8 {
        // Simplified shader compilation
        // In real implementation, use SPIR-V compiler
        
        const bytecode = try self.allocator.alloc(u8, source.len);
        @memcpy(bytecode, source);
        
        std.log.debug("Compiled {} shader: {} bytes", .{ stage, bytecode.len });
        return bytecode;
    }
    
    pub const ShaderStage = enum {
        vertex,
        fragment,
        compute,
        geometry,
        tessellation_control,
        tessellation_evaluation,
    };
    
    pub const CompiledShader = struct {
        stage: ShaderStage,
        bytecode: []u8,
        uniforms: std.ArrayList(UniformBinding),
    };
    
    pub const UniformBinding = struct {
        name: []const u8,
        location: u32,
        size: u32,
    };
};

/// Pipeline State Caching
pub const PipelineCache = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    cached_pipelines: std.AutoHashMap(u64, Pipeline),
    
    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .allocator = allocator,
            .cached_pipelines = std.AutoHashMap(u64, Pipeline).init(allocator),
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.cached_pipelines.deinit();
    }
    
    pub fn getPipeline(self: *Self, desc: PipelineDescriptor) !*Pipeline {
        const hash = desc.hash();
        
        if (self.cached_pipelines.getPtr(hash)) |pipeline| {
            return pipeline;
        }
        
        // Create new pipeline
        const pipeline = try self.createPipeline(desc);
        try self.cached_pipelines.put(hash, pipeline);
        
        return self.cached_pipelines.getPtr(hash).?;
    }
    
    fn createPipeline(self: *Self, desc: PipelineDescriptor) !Pipeline {
        _ = self;
        _ = desc;
        
        // Create graphics pipeline
        return Pipeline{
            .vertex_shader = 0,
            .fragment_shader = 0,
            .blend_state = BlendState{},
            .rasterizer_state = RasterizerState{},
            .depth_stencil_state = DepthStencilState{},
        };
    }
};

/// Display Types and Structures
pub const DisplayMode = struct {
    width: u32,
    height: u32,
    refresh_rate: u32,
    pixel_clock: u32,
    htotal: u32,
    hsync_start: u32,
    hsync_end: u32,
    vtotal: u32,
    vsync_start: u32,
    vsync_end: u32,
    hsync_positive: bool,
    vsync_positive: bool,
    interlaced: bool,
};

pub const DisplaySurface = struct {
    address: u64,
    stride: u32,
    width: u32,
    height: u32,
    format: PixelFormat,
};

pub const PixelFormat = enum(u32) {
    argb8888 = 0,
    xrgb8888 = 1,
    rgb565 = 2,
    rgba1010102 = 3,
    rgba16161616f = 4,
};

pub const HDRMode = enum {
    sdr,
    hdr10,
    dolby_vision,
};

pub const ColorSpace = enum {
    srgb,
    rec709,
    rec2020,
    dci_p3,
};

pub const ConnectedDisplay = struct {
    port: u8,
    edid: EDID,
    capabilities: DisplayCapabilities,
    current_mode: ?DisplayMode,
};

pub const DisplayCapabilities = struct {
    max_width: u32,
    max_height: u32,
    supported_formats: []const PixelFormat,
    vrr_capable: bool,
    vrr_min_hz: u32,
    vrr_max_hz: u32,
    hdr_capable: bool,
    max_luminance: u32,
    connector_type: ConnectorType,
    
    pub const ConnectorType = enum {
        unknown,
        displayport,
        hdmi,
        dvi,
        vga,
        usb_c,
    };
};

pub const EDID = struct {
    manufacturer: [4]u8,
    product_code: u16,
    serial_number: u32,
    max_horizontal_size: u32,
    max_vertical_size: u32,
    vrr_capable: bool,
    vrr_min_refresh: u32,
    vrr_max_refresh: u32,
    hdr_capable: bool,
    max_luminance: u32,
    
    pub fn parse(data: []const u8) !EDID {
        if (data.len < 128) return error.InvalidEDID;
        
        // Simplified EDID parsing
        return EDID{
            .manufacturer = .{ 'T', 'E', 'S', 'T' },
            .product_code = 0x1234,
            .serial_number = 0,
            .max_horizontal_size = 1920,
            .max_vertical_size = 1080,
            .vrr_capable = true,
            .vrr_min_refresh = 48,
            .vrr_max_refresh = 144,
            .hdr_capable = true,
            .max_luminance = 400,
        };
    }
};

pub const WaylandSurface = struct {
    id: u32,
    width: u32,
    height: u32,
    format: PixelFormat,
    buffer: ?*WaylandBuffer,
    damage_regions: std.ArrayList(DamageRegion),
};

pub const WaylandBuffer = struct {
    width: u32,
    height: u32,
    stride: u32,
    format: PixelFormat,
    data: []u8,
};

pub const DamageRegion = struct {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
};

pub const FrameCallback = struct {
    callback: *const fn () void,
    
    pub fn execute(self: FrameCallback) void {
        self.callback();
    }
};

pub const Pipeline = struct {
    vertex_shader: u64,
    fragment_shader: u64,
    blend_state: BlendState,
    rasterizer_state: RasterizerState,
    depth_stencil_state: DepthStencilState,
};

pub const PipelineDescriptor = struct {
    vertex_shader: u64,
    fragment_shader: u64,
    blend_enabled: bool,
    depth_test_enabled: bool,
    
    pub fn hash(self: PipelineDescriptor) u64 {
        var hasher = std.hash.CityHash64.init();
        hasher.update(std.mem.asBytes(&self.vertex_shader));
        hasher.update(std.mem.asBytes(&self.fragment_shader));
        hasher.update(std.mem.asBytes(&self.blend_enabled));
        hasher.update(std.mem.asBytes(&self.depth_test_enabled));
        return hasher.final();
    }
};

pub const BlendState = struct {
    enabled: bool = false,
    src_factor: BlendFactor = .one,
    dst_factor: BlendFactor = .zero,
    
    pub const BlendFactor = enum {
        zero,
        one,
        src_alpha,
        inv_src_alpha,
        dst_alpha,
        inv_dst_alpha,
    };
};

pub const RasterizerState = struct {
    cull_mode: CullMode = .back,
    fill_mode: FillMode = .solid,
    
    pub const CullMode = enum {
        none,
        front,
        back,
    };
    
    pub const FillMode = enum {
        solid,
        wireframe,
    };
};

pub const DepthStencilState = struct {
    depth_test_enabled: bool = true,
    depth_write_enabled: bool = true,
    depth_func: CompareFunc = .less,
    
    pub const CompareFunc = enum {
        never,
        less,
        equal,
        less_equal,
        greater,
        not_equal,
        greater_equal,
        always,
    };
};

pub const FrameStats = struct {
    vblank_count: u64 = 0,
    flip_count: u64 = 0,
    flip_requests: u64 = 0,
};

/// Hardware Register Definitions
pub const DisplayRegisters = extern struct {
    control: u32,
    status: u32,
    interrupt_enable: u32,
    interrupt_status: u32,
};

pub const CRTCRegisters = extern struct {
    control: u32,
    status: u32,
    h_total: u32,
    h_display: u32,
    h_sync_start: u32,
    h_sync_end: u32,
    v_total: u32,
    v_display: u32,
    v_sync_start: u32,
    v_sync_end: u32,
    pixel_clock: u32,
    surface_address: u64,
    surface_stride: u32,
    surface_format: u32,
    hotplug_status: u32,
    vrr_control: u32,
    vrr_min_vtotal: u32,
    vrr_max_vtotal: u32,
    hdr_control: u32,
    hdr_max_luminance: u32,
};

/// Hardware Constants
const DISPLAY_REGS_OFFSET = 0x610000;
const CRTC_REGS_BASE = 0x620000;
const CRTC_REGS_SIZE = 0x1000;

// Display control bits
const DISPLAY_CTRL_ENABLE = 0x00000001;
const DISPLAY_CTRL_RESET = 0x00000002;

// Display status bits
const DISPLAY_STATUS_READY = 0x00000001;

// Display interrupt bits
const DISPLAY_INTR_VBLANK = 0x00000001;
const DISPLAY_INTR_FLIP_COMPLETE = 0x00000002;
const DISPLAY_INTR_HOTPLUG = 0x00000004;

// CRTC control bits
const CRTC_CTRL_ENABLE = 0x00000001;
const CRTC_CTRL_RESET = 0x00000002;
const CRTC_CTRL_HSYNC_POS = 0x00000004;
const CRTC_CTRL_VSYNC_POS = 0x00000008;
const CRTC_CTRL_FLIP_PENDING = 0x00000010;
const CRTC_CTRL_FLIP_COMPLETE = 0x00000020;

// VRR control bits
const VRR_CTRL_ENABLE = 0x00000001;
const VRR_CTRL_ADAPTIVE = 0x00000002;

// HDR control bits
const HDR_CTRL_ENABLE = 0x00000001;
const HDR_CTRL_HDR10 = 0x00000002;
const HDR_CTRL_DOLBY_VISION = 0x00000004;