const std = @import("std");
const display = @import("engine.zig");
const bezel = @import("bezel_correction.zig");
const drm = @import("../drm/driver.zig");
const memory = @import("../hal/memory.zig");

/// Multi-Monitor Desktop Spanning Manager
/// Provides seamless window movement, taskbar spanning, and desktop management

pub const SpanningError = error{
    InvalidDesktopConfiguration,
    WindowManagerNotFound,
    TaskbarConfigurationFailed,
    HotPlugHandlingFailed,
    DisplayPowerManagementFailed,
    OutOfMemory,
};

pub const WindowSpanBehavior = enum(u8) {
    snap_to_edges = 0,      // Windows snap to display edges
    seamless_movement = 1,   // Windows move smoothly across displays  
    bezel_awareness = 2,     // Windows avoid crossing bezel gaps
    magnetic_edges = 3,      // Window edges attract to display boundaries
    
    pub fn allowsCrossDisplay(self: WindowSpanBehavior) bool {
        return self == .seamless_movement or self == .bezel_awareness;
    }
};

pub const TaskbarMode = enum(u8) {
    primary_only = 0,        // Taskbar only on primary display
    replicated = 1,          // Same taskbar on all displays
    extended = 2,            // Taskbar spans across all displays
    per_display = 3,         // Different taskbar content per display
    
    pub fn requiresMultipleTaskbars(self: TaskbarMode) bool {
        return self == .replicated or self == .per_display;
    }
};

pub const DisplayRole = enum(u8) {
    primary = 0,       // Main display (login screen, primary taskbar)
    secondary = 1,     // Extended desktop
    gaming = 2,        // Optimized for gaming (high refresh, low latency)
    portrait = 3,      // Rotated for documents/code
    ambient = 4,       // Status/monitoring display
};

pub const WindowState = struct {
    window_id: u64,
    title: []const u8,
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    display_id: u32,
    is_maximized: bool,
    is_fullscreen: bool,
    is_always_on_top: bool,
    span_displays: bool,
    
    pub fn getBounds(self: WindowState) struct { left: i32, top: i32, right: i32, bottom: i32 } {
        return .{
            .left = self.x,
            .top = self.y,
            .right = self.x + @as(i32, @intCast(self.width)),
            .bottom = self.y + @as(i32, @intCast(self.height)),
        };
    }
    
    pub fn intersectsDisplay(self: WindowState, display_bounds: struct { x: i32, y: i32, width: u32, height: u32 }) bool {
        const window_bounds = self.getBounds();
        const display_right = display_bounds.x + @as(i32, @intCast(display_bounds.width));
        const display_bottom = display_bounds.y + @as(i32, @intCast(display_bounds.height));
        
        return !(window_bounds.right <= display_bounds.x or 
                 window_bounds.left >= display_right or
                 window_bounds.bottom <= display_bounds.y or
                 window_bounds.top >= display_bottom);
    }
};

pub const VirtualDesktop = struct {
    id: u32,
    name: []const u8,
    displays: []u32, // Display IDs that belong to this virtual desktop
    wallpaper_path: ?[]const u8,
    window_layout: std.ArrayList(WindowState),
    active: bool,
    
    pub fn init(allocator: std.mem.Allocator, id: u32, name: []const u8) VirtualDesktop {
        return VirtualDesktop{
            .id = id,
            .name = name,
            .displays = &.{},
            .wallpaper_path = null,
            .window_layout = std.ArrayList(WindowState).init(allocator),
            .active = false,
        };
    }
    
    pub fn deinit(self: *VirtualDesktop) void {
        self.window_layout.deinit();
    }
    
    pub fn containsDisplay(self: VirtualDesktop, display_id: u32) bool {
        for (self.displays) |id| {
            if (id == display_id) return true;
        }
        return false;
    }
};

pub const SpanningManager = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    bezel_corrector: bezel.BezelCorrector,
    displays: std.AutoHashMap(u32, DisplayInfo),
    virtual_desktops: std.ArrayList(VirtualDesktop),
    active_desktop_id: u32,
    window_span_behavior: WindowSpanBehavior,
    taskbar_mode: TaskbarMode,
    
    // Window management
    windows: std.AutoHashMap(u64, WindowState),
    window_focus_history: std.ArrayList(u64),
    snap_threshold_pixels: u16,
    
    // Display hotplug management
    hotplug_callback: ?fn(display_id: u32, connected: bool) void,
    display_power_states: std.AutoHashMap(u32, DisplayPowerState),
    
    const DisplayInfo = struct {
        display: bezel.PhysicalDisplay,
        role: DisplayRole,
        active: bool,
        brightness_percent: u8,
        color_profile: ?[]const u8,
        refresh_rate: u16,
        vrr_enabled: bool,
    };
    
    const DisplayPowerState = enum(u8) {
        on = 0,
        standby = 1,
        suspended = 2,
        off = 3,
    };
    
    pub fn init(allocator: std.mem.Allocator) !Self {
        var self = Self{
            .allocator = allocator,
            .bezel_corrector = bezel.BezelCorrector.init(allocator),
            .displays = std.AutoHashMap(u32, DisplayInfo).init(allocator),
            .virtual_desktops = std.ArrayList(VirtualDesktop).init(allocator),
            .active_desktop_id = 0,
            .window_span_behavior = .bezel_awareness,
            .taskbar_mode = .primary_only,
            .windows = std.AutoHashMap(u64, WindowState).init(allocator),
            .window_focus_history = std.ArrayList(u64).init(allocator),
            .snap_threshold_pixels = 20,
            .hotplug_callback = null,
            .display_power_states = std.AutoHashMap(u32, DisplayPowerState).init(allocator),
        };
        
        // Create default virtual desktop
        var default_desktop = VirtualDesktop.init(allocator, 0, "Desktop 1");
        default_desktop.active = true;
        try self.virtual_desktops.append(default_desktop);
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        self.bezel_corrector.deinit();
        self.displays.deinit();
        
        for (self.virtual_desktops.items) |*desktop| {
            desktop.deinit();
        }
        self.virtual_desktops.deinit();
        
        self.windows.deinit();
        self.window_focus_history.deinit();
        self.display_power_states.deinit();
    }
    
    pub fn addDisplay(self: *Self, phys_display: bezel.PhysicalDisplay, role: DisplayRole) !void {
        const display_info = DisplayInfo{
            .display = phys_display,
            .role = role,
            .active = true,
            .brightness_percent = 100,
            .color_profile = null,
            .refresh_rate = phys_display.refresh_rate,
            .vrr_enabled = false,
        };
        
        try self.displays.put(phys_display.display_id, display_info);
        try self.bezel_corrector.addDisplay(phys_display);
        try self.display_power_states.put(phys_display.display_id, .on);
        
        // Add display to current virtual desktop
        if (self.getActiveDesktop()) |desktop| {
            var new_displays = try self.allocator.alloc(u32, desktop.displays.len + 1);
            @memcpy(new_displays[0..desktop.displays.len], desktop.displays);
            new_displays[desktop.displays.len] = phys_display.display_id;
            
            if (desktop.displays.len > 0) {
                self.allocator.free(desktop.displays);
            }
            desktop.displays = new_displays;
        }
        
        // Recalculate spanning configuration
        try self.updateSpanningConfiguration();
    }
    
    pub fn removeDisplay(self: *Self, display_id: u32) !void {
        if (self.displays.remove(display_id)) {
            self.bezel_corrector.removeDisplay(display_id);
            _ = self.display_power_states.remove(display_id);
            
            // Remove from virtual desktops
            for (self.virtual_desktops.items) |*desktop| {
                var new_displays = std.ArrayList(u32).init(self.allocator);
                defer {
                    if (desktop.displays.len > 0) {
                        self.allocator.free(desktop.displays);
                    }
                    desktop.displays = new_displays.toOwnedSlice() catch &.{};
                }
                
                for (desktop.displays) |id| {
                    if (id != display_id) {
                        try new_displays.append(id);
                    }
                }
            }
            
            // Move windows from removed display to primary display
            try self.relocateWindowsFromDisplay(display_id);
            
            try self.updateSpanningConfiguration();
        }
    }
    
    fn updateSpanningConfiguration(self: *Self) !void {
        // Update bezel correction based on current display configuration
        const spanning_mode: bezel.DesktopSpanningMode = switch (self.window_span_behavior) {
            .snap_to_edges => .extend,
            .seamless_movement, .bezel_awareness => .spanning,
            .magnetic_edges => .spanning,
        };
        
        try self.bezel_corrector.setSpanningMode(spanning_mode);
    }
    
    pub fn setWindowSpanBehavior(self: *Self, behavior: WindowSpanBehavior) !void {
        self.window_span_behavior = behavior;
        try self.updateSpanningConfiguration();
    }
    
    pub fn setTaskbarMode(self: *Self, mode: TaskbarMode) void {
        self.taskbar_mode = mode;
        // Taskbar reconfiguration would be handled by window manager integration
    }
    
    pub fn addWindow(self: *Self, window: WindowState) !void {
        try self.windows.put(window.window_id, window);
        try self.window_focus_history.append(window.window_id);
        
        // Ensure window is within valid display bounds
        try self.validateWindowPlacement(window.window_id);
    }
    
    pub fn moveWindow(self: *Self, window_id: u64, new_x: i32, new_y: i32) !void {
        if (self.windows.getPtr(window_id)) |window| {
            const old_x = window.x;
            const old_y = window.y;
            
            window.x = new_x;
            window.y = new_y;
            
            // Apply spanning behavior
            switch (self.window_span_behavior) {
                .snap_to_edges => try self.applyEdgeSnapping(window),
                .seamless_movement => {}, // Allow free movement
                .bezel_awareness => try self.applyBezelAwareness(window),
                .magnetic_edges => try self.applyMagneticEdges(window, old_x, old_y),
            }
            
            // Update display association
            try self.updateWindowDisplayAssociation(window_id);
        }
    }
    
    fn applyEdgeSnapping(self: *Self, window: *WindowState) !void {
        var display_iter = self.displays.iterator();
        while (display_iter.next()) |entry| {
            const display_info = entry.value_ptr;
            const disp = display_info.display;
            const res = disp.getEffectiveResolution();
            
            // Check if window is near display edges
            const display_bounds = struct {
                x: i32 = disp.position_x,
                y: i32 = disp.position_y,
                width: u32 = res.width,
                height: u32 = res.height,
            };
            
            if (window.intersectsDisplay(display_bounds)) {
                // Snap to edges if within threshold
                if (@abs(window.x - disp.position_x) <= self.snap_threshold_pixels) {
                    window.x = disp.position_x;
                }
                
                if (@abs(window.y - disp.position_y) <= self.snap_threshold_pixels) {
                    window.y = disp.position_y;
                }
                
                const right_edge = disp.position_x + @as(i32, @intCast(res.width));
                const bottom_edge = disp.position_y + @as(i32, @intCast(res.height));
                
                if (@abs((window.x + @as(i32, @intCast(window.width))) - right_edge) <= self.snap_threshold_pixels) {
                    window.x = right_edge - @as(i32, @intCast(window.width));
                }
                
                if (@abs((window.y + @as(i32, @intCast(window.height))) - bottom_edge) <= self.snap_threshold_pixels) {
                    window.y = bottom_edge - @as(i32, @intCast(window.height));
                }
                
                break;
            }
        }
    }
    
    fn applyBezelAwareness(self: *Self, window: *WindowState) !void {
        // Prevent windows from being positioned across bezel gaps
        _ = self.bezel_corrector.getVirtualDesktopSize();
        
        // Check if window crosses display boundaries with significant bezel gaps
        var display_iter = self.displays.iterator();
        while (display_iter.next()) |entry| {
            const display_info = entry.value_ptr;
            const disp = display_info.display;
            const res = disp.getEffectiveResolution();
            const bezel_offsets = disp.getBezelOffsets();
            
            // If bezel gap is significant (>10 pixels equivalent), prevent crossing
            if (bezel_offsets.right > 10 or bezel_offsets.left > 10) {
                const display_right = disp.position_x + @as(i32, @intCast(res.width));
                const window_right = window.x + @as(i32, @intCast(window.width));
                
                // Check if window would span across bezel
                if (window.x < display_right and window_right > display_right) {
                    // Move window to avoid spanning the bezel
                    if (@abs(window.x - display_right) < @abs(window_right - display_right)) {
                        window.x = display_right; // Move to right display
                    } else {
                        window.x = display_right - @as(i32, @intCast(window.width)); // Keep on left display
                    }
                }
            }
        }
    }
    
    fn applyMagneticEdges(self: *Self, window: *WindowState, old_x: i32, old_y: i32) !void {
        const magnetic_threshold = self.snap_threshold_pixels * 2;
        
        var display_iter = self.displays.iterator();
        while (display_iter.next()) |entry| {
            const display_info = entry.value_ptr;
            const disp = display_info.display;
            const res = disp.getEffectiveResolution();
            
            // Calculate attraction force to display edges
            const display_left = disp.position_x;
            _ = disp.position_x + @as(i32, @intCast(res.width));
            const display_top = disp.position_y;
            _ = disp.position_y + @as(i32, @intCast(res.height));
            
            // Apply magnetic attraction
            if (@abs(window.x - display_left) <= magnetic_threshold) {
                const attraction = magnetic_threshold - @abs(window.x - display_left);
                window.x = display_left + @as(i32, @intFromFloat(@as(f32, @floatFromInt(attraction)) * 0.3));
            }
            
            if (@abs(window.y - display_top) <= magnetic_threshold) {
                const attraction = magnetic_threshold - @abs(window.y - display_top);
                window.y = display_top + @as(i32, @intFromFloat(@as(f32, @floatFromInt(attraction)) * 0.3));
            }
        }
        
        _ = old_x;
        _ = old_y;
    }
    
    fn updateWindowDisplayAssociation(self: *Self, window_id: u64) !void {
        if (self.windows.getPtr(window_id)) |window| {
            // Find which display contains the majority of the window
            var best_display_id: u32 = 0;
            var best_overlap_area: u32 = 0;
            
            var display_iter = self.displays.iterator();
            while (display_iter.next()) |entry| {
                const display_info = entry.value_ptr;
                const disp = display_info.display;
                const res = disp.getEffectiveResolution();
                
                const display_bounds = struct {
                    x: i32 = disp.position_x,
                    y: i32 = disp.position_y,
                    width: u32 = res.width,
                    height: u32 = res.height,
                };
                
                if (window.intersectsDisplay(display_bounds)) {
                    // Calculate overlap area
                    const window_bounds = window.getBounds();
                    const overlap_left = @max(window_bounds.left, display_bounds.x);
                    const overlap_top = @max(window_bounds.top, display_bounds.y);
                    const overlap_right = @min(window_bounds.right, display_bounds.x + @as(i32, @intCast(display_bounds.width)));
                    const overlap_bottom = @min(window_bounds.bottom, display_bounds.y + @as(i32, @intCast(display_bounds.height)));
                    
                    if (overlap_right > overlap_left and overlap_bottom > overlap_top) {
                        const overlap_area = @as(u32, @intCast((overlap_right - overlap_left) * (overlap_bottom - overlap_top)));
                        
                        if (overlap_area > best_overlap_area) {
                            best_overlap_area = overlap_area;
                            best_display_id = disp.display_id;
                        }
                    }
                }
            }
            
            window.display_id = best_display_id;
        }
    }
    
    fn validateWindowPlacement(self: *Self, window_id: u64) !void {
        if (self.windows.getPtr(window_id)) |window| {
            const virtual_size = self.bezel_corrector.getVirtualDesktopSize();
            
            // Ensure window is within virtual desktop bounds
            if (window.x < 0) window.x = 0;
            if (window.y < 0) window.y = 0;
            
            if (window.x + @as(i32, @intCast(window.width)) > virtual_size.width) {
                window.x = @as(i32, @intCast(virtual_size.width)) - @as(i32, @intCast(window.width));
            }
            
            if (window.y + @as(i32, @intCast(window.height)) > virtual_size.height) {
                window.y = @as(i32, @intCast(virtual_size.height)) - @as(i32, @intCast(window.height));
            }
            
            try self.updateWindowDisplayAssociation(window_id);
        }
    }
    
    pub fn maximizeWindowToDisplay(self: *Self, window_id: u64, display_id: u32) !void {
        if (self.windows.getPtr(window_id)) |window| {
            if (self.displays.get(display_id)) |display_info| {
                const disp = display_info.display;
                const res = disp.getEffectiveResolution();
                
                window.x = disp.position_x;
                window.y = disp.position_y;
                window.width = res.width;
                window.height = res.height;
                window.is_maximized = true;
                window.display_id = display_id;
            }
        }
    }
    
    pub fn spanWindowAcrossDisplays(self: *Self, window_id: u64, display_ids: []const u32) !void {
        if (self.windows.getPtr(window_id)) |window| {
            if (display_ids.len == 0) return;
            
            var min_x: i32 = std.math.maxInt(i32);
            var min_y: i32 = std.math.maxInt(i32);
            var max_x: i32 = std.math.minInt(i32);
            var max_y: i32 = std.math.minInt(i32);
            
            // Calculate bounding rectangle of specified displays
            for (display_ids) |display_id| {
                if (self.displays.get(display_id)) |display_info| {
                    const disp = display_info.display;
                    const res = disp.getEffectiveResolution();
                    
                    min_x = @min(min_x, disp.position_x);
                    min_y = @min(min_y, disp.position_y);
                    max_x = @max(max_x, disp.position_x + @as(i32, @intCast(res.width)));
                    max_y = @max(max_y, disp.position_y + @as(i32, @intCast(res.height)));
                }
            }
            
            window.x = min_x;
            window.y = min_y;
            window.width = @intCast(max_x - min_x);
            window.height = @intCast(max_y - min_y);
            window.span_displays = true;
        }
    }
    
    fn relocateWindowsFromDisplay(self: *Self, removed_display_id: u32) !void {
        // Find primary display to move windows to
        var primary_display_id: u32 = 0;
        var display_iter = self.displays.iterator();
        while (display_iter.next()) |entry| {
            const display_info = entry.value_ptr;
            if (display_info.role == .primary) {
                primary_display_id = entry.key_ptr.*;
                break;
            }
        }
        
        // Move windows from removed display
        var window_iter = self.windows.iterator();
        while (window_iter.next()) |entry| {
            const window = entry.value_ptr;
            if (window.display_id == removed_display_id) {
                if (self.displays.get(primary_display_id)) |primary_info| {
                    const primary_display = primary_info.display;
                    window.x = primary_display.position_x;
                    window.y = primary_display.position_y;
                    window.display_id = primary_display_id;
                }
            }
        }
    }
    
    pub fn handleDisplayHotplug(self: *Self, display_id: u32, connected: bool) !void {
        if (connected) {
            // Display connected - would need to query hardware for display info
            // For now, create a basic display configuration
            const new_display = bezel.PhysicalDisplay.init(display_id);
            try self.addDisplay(new_display, .secondary);
        } else {
            // Display disconnected
            try self.removeDisplay(display_id);
        }
        
        // Call registered callback
        if (self.hotplug_callback) |callback| {
            callback(display_id, connected);
        }
    }
    
    pub fn setDisplayPowerState(self: *Self, display_id: u32, state: DisplayPowerState) !void {
        try self.display_power_states.put(display_id, state);
        
        // In real implementation, would control actual display power via DRM/DDC
        switch (state) {
            .on => {
                // Turn display on, restore brightness/settings
            },
            .standby => {
                // Put display in standby (quick wake)
            },
            .suspended => {
                // Suspend display (longer wake time)
            },
            .off => {
                // Turn display completely off
            },
        }
    }
    
    pub fn createVirtualDesktop(self: *Self, name: []const u8) !u32 {
        const id = @as(u32, @intCast(self.virtual_desktops.items.len));
        var desktop = VirtualDesktop.init(self.allocator, id, name);
        
        // Copy current display configuration
        var displays = std.ArrayList(u32).init(self.allocator);
        defer displays.deinit();
        
        var display_iter = self.displays.keyIterator();
        while (display_iter.next()) |display_id| {
            try displays.append(display_id.*);
        }
        
        desktop.displays = try displays.toOwnedSlice();
        try self.virtual_desktops.append(desktop);
        
        return id;
    }
    
    pub fn switchToVirtualDesktop(self: *Self, desktop_id: u32) !void {
        if (desktop_id < self.virtual_desktops.items.len) {
            // Deactivate current desktop
            if (self.getActiveDesktop()) |current| {
                current.active = false;
            }
            
            // Activate new desktop
            self.virtual_desktops.items[desktop_id].active = true;
            self.active_desktop_id = desktop_id;
            
            // Restore window layout for this desktop
            // In real implementation, would restore window positions and states
        }
    }
    
    pub fn getActiveDesktop(self: *Self) ?*VirtualDesktop {
        for (self.virtual_desktops.items) |*desktop| {
            if (desktop.active) return desktop;
        }
        return null;
    }
    
    pub fn setHotplugCallback(self: *Self, callback: fn(display_id: u32, connected: bool) void) void {
        self.hotplug_callback = callback;
    }
    
    pub fn getSpanningStatistics(self: *const Self) struct {
        total_displays: u32,
        active_displays: u32,
        virtual_desktop_size: struct { width: u32, height: u32 },
        total_windows: u32,
        spanning_windows: u32,
        primary_display_id: u32,
    } {
        var active_count: u32 = 0;
        var spanning_count: u32 = 0;
        var primary_id: u32 = 0;
        
        var display_iter = self.displays.iterator();
        while (display_iter.next()) |entry| {
            if (entry.value_ptr.active) active_count += 1;
            if (entry.value_ptr.role == .primary) primary_id = entry.key_ptr.*;
        }
        
        var window_iter = self.windows.iterator();
        while (window_iter.next()) |entry| {
            if (entry.value_ptr.span_displays) spanning_count += 1;
        }
        
        return .{
            .total_displays = @intCast(self.displays.count()),
            .active_displays = active_count,
            .virtual_desktop_size = self.bezel_corrector.getVirtualDesktopSize(),
            .total_windows = @intCast(self.windows.count()),
            .spanning_windows = spanning_count,
            .primary_display_id = primary_id,
        };
    }
};