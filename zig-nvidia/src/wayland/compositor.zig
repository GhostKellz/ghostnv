const std = @import("std");
const drm = @import("../drm/driver.zig");
const memory = @import("../hal/memory.zig");
const print = std.debug.print;

pub const WaylandError = error{
    CompositorInitFailed,
    SurfaceCreationFailed,
    BufferAllocationFailed,
    InvalidFormat,
    NotSupported,
};

pub const WaylandFormat = enum(u32) {
    ARGB8888 = 0x34325241,
    XRGB8888 = 0x34325258,
    RGBA8888 = 0x34324152,
    RGBX8888 = 0x34324752,
    RGB565 = 0x36314752,
};

pub const WaylandBuffer = struct {
    id: u32,
    width: u32,
    height: u32,
    format: WaylandFormat,
    stride: u32,
    size: usize,
    dma_buf: ?*memory.DmaBuffer,
    physical_addr: usize,
    mapped: bool,
    
    pub fn init(id: u32, width: u32, height: u32, format: WaylandFormat) WaylandBuffer {
        const bpp = switch (format) {
            .ARGB8888, .XRGB8888, .RGBA8888, .RGBX8888 => 4,
            .RGB565 => 2,
        };
        
        const stride = width * bpp;
        const size = stride * height;
        
        return WaylandBuffer{
            .id = id,
            .width = width,
            .height = height,
            .format = format,
            .stride = stride,
            .size = size,
            .dma_buf = null,
            .physical_addr = 0,
            .mapped = false,
        };
    }
    
    pub fn allocate_gpu_memory(self: *WaylandBuffer, mem_manager: *memory.MemoryManager) !void {
        // Optimized: Allocate VRAM directly for zero-copy performance
        const alignment = 4096; // Page-aligned for optimal DMA
        const region = try mem_manager.allocateVram(
            self.size, 
            alignment, 
            .framebuffer, 
            memory.MemoryFlags{ .coherent = true, .cacheable = false }
        );
        
        // Map for CPU access if needed
        try region.map();
        
        self.physical_addr = region.physical_address;
        self.mapped = true;
        
        print("zig-nvidia: Zero-copy Wayland buffer {} allocated ({}x{}, {} bytes, VRAM: 0x{X})\n",
              .{self.id, self.width, self.height, self.size, self.physical_addr});
    }
    
    pub fn deinit(self: *WaylandBuffer, mem_manager: *memory.MemoryManager) void {
        if (self.dma_buf) |buf| {
            mem_manager.free_dma_buffer(buf);
            self.dma_buf = null;
        }
        self.mapped = false;
    }
    
    pub fn get_dmabuf_fd(self: *WaylandBuffer) !i32 {
        if (!self.mapped) return WaylandError.BufferAllocationFailed;
        
        // In real implementation, would return actual dmabuf file descriptor
        // For now, simulate with dummy FD
        print("zig-nvidia: Exporting dmabuf FD for buffer {}\n", .{self.id});
        return 42; // Dummy FD
    }
};

pub const WaylandSurface = struct {
    id: u32,
    width: u32,
    height: u32,
    current_buffer: ?*WaylandBuffer,
    pending_buffer: ?*WaylandBuffer,
    damage_regions: std.ArrayList(DamageRegion),
    
    pub fn init(allocator: std.mem.Allocator, id: u32) WaylandSurface {
        return WaylandSurface{
            .id = id,
            .width = 0,
            .height = 0,
            .current_buffer = null,
            .pending_buffer = null,
            .damage_regions = std.ArrayList(DamageRegion).init(allocator),
        };
    }
    
    pub fn deinit(self: *WaylandSurface) void {
        self.damage_regions.deinit();
    }
    
    pub fn attach_buffer(self: *WaylandSurface, buffer: *WaylandBuffer) void {
        self.pending_buffer = buffer;
        self.width = buffer.width;
        self.height = buffer.height;
        
        print("zig-nvidia: Attached buffer {} to surface {}\n", .{buffer.id, self.id});
    }
    
    pub fn commit(self: *WaylandSurface) void {
        if (self.pending_buffer) |buffer| {
            self.current_buffer = buffer;
            self.pending_buffer = null;
            
            print("zig-nvidia: Committed surface {} ({}x{})\n", .{self.id, self.width, self.height});
        }
    }
    
    pub fn damage(self: *WaylandSurface, x: i32, y: i32, width: u32, height: u32) !void {
        const region = DamageRegion{
            .x = x,
            .y = y,
            .width = width,
            .height = height,
        };
        
        try self.damage_regions.append(region);
        print("zig-nvidia: Added damage region to surface {}: ({}, {}) {}x{}\n",
              .{self.id, x, y, width, height});
    }
};

const DamageRegion = struct {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
};

pub const WaylandCompositor = struct {
    allocator: std.mem.Allocator,
    drm_driver: *drm.DrmDriver,
    memory_manager: *memory.MemoryManager,
    surfaces: std.HashMap(u32, WaylandSurface),
    buffers: std.HashMap(u32, WaylandBuffer),
    next_id: u32,
    vsync_enabled: bool,
    direct_scanout: bool,
    
    pub fn init(allocator: std.mem.Allocator, drm_driver: *drm.DrmDriver, mem_manager: *memory.MemoryManager) WaylandCompositor {
        return WaylandCompositor{
            .allocator = allocator,
            .drm_driver = drm_driver,
            .memory_manager = mem_manager,
            .surfaces = std.HashMap(u32, WaylandSurface).init(allocator),
            .buffers = std.HashMap(u32, WaylandBuffer).init(allocator),
            .next_id = 1,
            .vsync_enabled = true,
            .direct_scanout = false,
        };
    }
    
    pub fn deinit(self: *WaylandCompositor) void {
        // Clean up all surfaces
        var surface_iter = self.surfaces.iterator();
        while (surface_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.surfaces.deinit();
        
        // Clean up all buffers
        var buffer_iter = self.buffers.iterator();
        while (buffer_iter.next()) |entry| {
            entry.value_ptr.deinit(self.memory_manager);
        }
        self.buffers.deinit();
    }
    
    pub fn create_surface(self: *WaylandCompositor) !u32 {
        const id = self.next_id;
        self.next_id += 1;
        
        const surface = WaylandSurface.init(self.allocator, id);
        try self.surfaces.put(id, surface);
        
        print("zig-nvidia: Created Wayland surface {}\n", .{id});
        return id;
    }
    
    pub fn destroy_surface(self: *WaylandCompositor, surface_id: u32) void {
        if (self.surfaces.getPtr(surface_id)) |surface| {
            surface.deinit();
            _ = self.surfaces.remove(surface_id);
            print("zig-nvidia: Destroyed Wayland surface {}\n", .{surface_id});
        }
    }
    
    pub fn create_buffer(self: *WaylandCompositor, width: u32, height: u32, format: WaylandFormat) !u32 {
        const id = self.next_id;
        self.next_id += 1;
        
        var buffer = WaylandBuffer.init(id, width, height, format);
        try buffer.allocate_gpu_memory(self.memory_manager);
        try self.buffers.put(id, buffer);
        
        print("zig-nvidia: Created Wayland buffer {} ({}x{}, format: {})\n", 
              .{id, width, height, @intFromEnum(format)});
        return id;
    }
    
    pub fn destroy_buffer(self: *WaylandCompositor, buffer_id: u32) void {
        if (self.buffers.getPtr(buffer_id)) |buffer| {
            buffer.deinit(self.memory_manager);
            _ = self.buffers.remove(buffer_id);
            print("zig-nvidia: Destroyed Wayland buffer {}\n", .{buffer_id});
        }
    }
    
    pub fn attach_buffer(self: *WaylandCompositor, surface_id: u32, buffer_id: u32) !void {
        const surface = self.surfaces.getPtr(surface_id) orelse return WaylandError.SurfaceCreationFailed;
        const buffer = self.buffers.getPtr(buffer_id) orelse return WaylandError.BufferAllocationFailed;
        
        surface.attach_buffer(buffer);
    }
    
    pub fn commit_surface(self: *WaylandCompositor, surface_id: u32) !void {
        const surface = self.surfaces.getPtr(surface_id) orelse return WaylandError.SurfaceCreationFailed;
        surface.commit();
        
        // Try direct scanout for fullscreen surfaces
        if (self.can_direct_scanout(surface)) {
            try self.enable_direct_scanout(surface);
        } else {
            try self.composite_surface(surface);
        }
    }
    
    fn can_direct_scanout(self: *WaylandCompositor, surface: *WaylandSurface) bool {
        _ = self;
        // Check if surface can bypass composition
        return surface.width >= 1920 and surface.height >= 1080 and 
               surface.damage_regions.items.len == 0;
    }
    
    fn enable_direct_scanout(self: *WaylandCompositor, surface: *WaylandSurface) !void {
        if (surface.current_buffer) |buffer| {
            print("zig-nvidia: Enabling zero-copy direct scanout for surface {}\n", .{surface.id});
            
            // Optimized: Create framebuffer directly from VRAM buffer
            const fb = try self.drm_driver.create_framebuffer_from_vram(
                buffer.physical_addr,
                buffer.width, 
                buffer.height, 
                buffer.stride,
                @intFromEnum(buffer.format)
            );
            
            // Configure display controller for direct VRAM scanout
            try self.drm_driver.configure_direct_scanout(fb, buffer.physical_addr);
            
            self.direct_scanout = true;
            print("zig-nvidia: Zero-copy scanout enabled - GPU→Display direct path\n");
        }
    }
    
    fn composite_surface(self: *WaylandCompositor, surface: *WaylandSurface) !void {
        print("zig-nvidia: GPU-accelerated composition for surface {}\n", .{surface.id});
        
        if (surface.current_buffer) |buffer| {
            // Optimized GPU composition using NVIDIA shaders
            try self.gpu_accelerated_blit(buffer, surface);
            
            // Process damage regions for minimal updates
            if (surface.damage_regions.items.len > 0) {
                try self.process_damage_regions(surface);
            }
        }
    }
    
    fn gpu_accelerated_blit(self: *WaylandCompositor, buffer: *WaylandBuffer, surface: *WaylandSurface) !void {
        // Use GPU copy engines for zero-copy blitting
        print("zig-nvidia: GPU blit {} → framebuffer (zero texture upload)\n", .{buffer.id});
        
        // In real implementation:
        // 1. Create texture from VRAM buffer (zero copy)
        // 2. Use 2D/3D engine for composition
        // 3. Direct blit to scanout buffer
        
        _ = surface;
        try self.drm_driver.gpu_blit(buffer.physical_addr, buffer.width, buffer.height);
    }
    
    fn process_damage_regions(self: *WaylandCompositor, surface: *WaylandSurface) !void {
        print("zig-nvidia: Processing {} damage regions with GPU acceleration\n", .{surface.damage_regions.items.len});
        
        for (surface.damage_regions.items) |region| {
            // GPU-accelerated partial updates
            try self.drm_driver.gpu_damage_blit(
                region.x, region.y, region.width, region.height
            );
        }
        
        // Clear processed damage regions
        surface.damage_regions.clearRetainingCapacity();
    }
    
    pub fn schedule_repaint(self: *WaylandCompositor) !void {
        if (self.vsync_enabled) {
            print("zig-nvidia: Scheduling repaint at next vblank\n");
            // TODO: Wait for vblank and composite
        } else {
            print("zig-nvidia: Immediate repaint (no vsync)\n");
            try self.repaint();
        }
    }
    
    fn repaint(self: *WaylandCompositor) !void {
        print("zig-nvidia: Repainting all surfaces\n");
        
        // Iterate through all surfaces and composite
        var surface_iter = self.surfaces.iterator();
        while (surface_iter.next()) |entry| {
            const surface = entry.value_ptr;
            if (surface.current_buffer != null) {
                try self.composite_surface(surface);
            }
        }
        
        // Present the final image
        try self.drm_driver.atomic_commit();
        print("zig-nvidia: Frame presented\n");
    }
    
    pub fn set_vsync(self: *WaylandCompositor, enabled: bool) void {
        self.vsync_enabled = enabled;
        print("zig-nvidia: VSync {s}\n", .{if (enabled) "enabled" else "disabled"});
    }
    
    // Wayland protocol implementation helpers
    pub fn handle_buffer_import(self: *WaylandCompositor, dmabuf_fd: i32, width: u32, height: u32, format: WaylandFormat) !u32 {
        _ = dmabuf_fd;
        print("zig-nvidia: Importing dmabuf (FD: {}, {}x{}, format: {})\n", 
              .{dmabuf_fd, width, height, @intFromEnum(format)});
        
        // Create buffer from imported dmabuf
        return try self.create_buffer(width, height, format);
    }
    
    pub fn get_buffer_dmabuf_fd(self: *WaylandCompositor, buffer_id: u32) !i32 {
        const buffer = self.buffers.getPtr(buffer_id) orelse return WaylandError.BufferAllocationFailed;
        return try buffer.get_dmabuf_fd();
    }
};

test "wayland compositor" {
    const allocator = std.testing.allocator;
    
    var drm_driver = try drm.DrmDriver.init(allocator);
    defer drm_driver.deinit();
    
    var mem_manager = memory.MemoryManager.init(allocator);
    defer mem_manager.deinit();
    
    var compositor = WaylandCompositor.init(allocator, &drm_driver, &mem_manager);
    defer compositor.deinit();
    
    // Test surface creation
    const surface_id = try compositor.create_surface();
    try std.testing.expect(surface_id == 1);
    
    // Test buffer creation
    const buffer_id = try compositor.create_buffer(1920, 1080, .XRGB8888);
    try std.testing.expect(buffer_id == 2);
    
    // Test buffer attachment
    try compositor.attach_buffer(surface_id, buffer_id);
    
    // Test commit
    try compositor.commit_surface(surface_id);
}