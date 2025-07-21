const std = @import("std");
const memory = @import("../hal/memory.zig");
const display = @import("../display/engine.zig");
const linux = std.os.linux;

/// Advanced Damage Tracking for Wayland Compositor
/// Provides optimized damage accumulation, spatial tracking, and GPU-accelerated processing

pub const DamageError = error{
    InvalidRegion,
    TrackerFull,
    OutOfMemory,
    HardwareError,
    AccelerationFailed,
};

pub const Rectangle = struct {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    
    pub fn init(x: i32, y: i32, width: u32, height: u32) Rectangle {
        return Rectangle{ .x = x, .y = y, .width = width, .height = height };
    }
    
    pub fn area(self: Rectangle) u64 {
        return @as(u64, self.width) * @as(u64, self.height);
    }
    
    pub fn intersects(self: Rectangle, other: Rectangle) bool {
        return self.x < other.x + @as(i32, @intCast(other.width)) and
               self.x + @as(i32, @intCast(self.width)) > other.x and
               self.y < other.y + @as(i32, @intCast(other.height)) and
               self.y + @as(i32, @intCast(self.height)) > other.y;
    }
    
    pub fn intersection(self: Rectangle, other: Rectangle) ?Rectangle {
        if (!self.intersects(other)) return null;
        
        const x1 = @max(self.x, other.x);
        const y1 = @max(self.y, other.y);
        const x2 = @min(self.x + @as(i32, @intCast(self.width)), other.x + @as(i32, @intCast(other.width)));
        const y2 = @min(self.y + @as(i32, @intCast(self.height)), other.y + @as(i32, @intCast(other.height)));
        
        return Rectangle.init(x1, y1, @intCast(x2 - x1), @intCast(y2 - y1));
    }
    
    pub fn union_with(self: Rectangle, other: Rectangle) Rectangle {
        const x1 = @min(self.x, other.x);
        const y1 = @min(self.y, other.y);
        const x2 = @max(self.x + @as(i32, @intCast(self.width)), other.x + @as(i32, @intCast(other.width)));
        const y2 = @max(self.y + @as(i32, @intCast(self.height)), other.y + @as(i32, @intCast(other.height)));
        
        return Rectangle.init(x1, y1, @intCast(x2 - x1), @intCast(y2 - y1));
    }
    
    pub fn contains(self: Rectangle, point: struct { x: i32, y: i32 }) bool {
        return point.x >= self.x and 
               point.x < self.x + @as(i32, @intCast(self.width)) and
               point.y >= self.y and 
               point.y < self.y + @as(i32, @intCast(self.height));
    }
    
    pub fn isEmpty(self: Rectangle) bool {
        return self.width == 0 or self.height == 0;
    }
};

pub const DamageRegion = struct {
    rectangles: std.ArrayList(Rectangle),
    bounding_box: ?Rectangle,
    total_area: u64,
    
    pub fn init(allocator: std.mem.Allocator) DamageRegion {
        return DamageRegion{
            .rectangles = std.ArrayList(Rectangle).init(allocator),
            .bounding_box = null,
            .total_area = 0,
        };
    }
    
    pub fn deinit(self: *DamageRegion) void {
        self.rectangles.deinit();
    }
    
    pub fn clear(self: *DamageRegion) void {
        self.rectangles.clearRetainingCapacity();
        self.bounding_box = null;
        self.total_area = 0;
    }
    
    pub fn addRect(self: *DamageRegion, rect: Rectangle) !void {
        if (rect.isEmpty()) return;
        
        try self.rectangles.append(rect);
        self.total_area += rect.area();
        
        if (self.bounding_box) |bbox| {
            self.bounding_box = bbox.union_with(rect);
        } else {
            self.bounding_box = rect;
        }
    }
    
    pub fn optimize(self: *DamageRegion) !void {
        if (self.rectangles.items.len <= 1) return;
        
        // Merge overlapping rectangles
        var i: usize = 0;
        while (i < self.rectangles.items.len) {
            var j = i + 1;
            while (j < self.rectangles.items.len) {
                if (self.rectangles.items[i].intersects(self.rectangles.items[j])) {
                    const merged = self.rectangles.items[i].union_with(self.rectangles.items[j]);
                    self.rectangles.items[i] = merged;
                    _ = self.rectangles.orderedRemove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        
        // Recalculate total area
        self.total_area = 0;
        for (self.rectangles.items) |rect| {
            self.total_area += rect.area();
        }
    }
    
    pub fn clipTo(self: *DamageRegion, clip_rect: Rectangle) void {
        var i: usize = 0;
        while (i < self.rectangles.items.len) {
            if (self.rectangles.items[i].intersection(clip_rect)) |clipped| {
                self.rectangles.items[i] = clipped;
                i += 1;
            } else {
                _ = self.rectangles.orderedRemove(i);
            }
        }
        
        if (self.bounding_box) |bbox| {
            self.bounding_box = bbox.intersection(clip_rect);
        }
        
        // Recalculate area
        self.total_area = 0;
        for (self.rectangles.items) |rect| {
            self.total_area += rect.area();
        }
    }
    
    pub fn isEmpty(self: *const DamageRegion) bool {
        return self.rectangles.items.len == 0;
    }
    
    pub fn copy(self: *const DamageRegion, allocator: std.mem.Allocator) !DamageRegion {
        var new_region = DamageRegion.init(allocator);
        try new_region.rectangles.appendSlice(self.rectangles.items);
        new_region.bounding_box = self.bounding_box;
        new_region.total_area = self.total_area;
        return new_region;
    }
};

pub const SpatialIndex = struct {
    const Self = @This();
    const GRID_SIZE = 64; // 64x64 grid for spatial indexing
    
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    cell_width: u32,
    cell_height: u32,
    grid: [GRID_SIZE][GRID_SIZE]std.ArrayList(usize), // Grid cells containing damage rect indices
    damage_rects: std.ArrayList(Rectangle),
    
    pub fn init(allocator: std.mem.Allocator, screen_width: u32, screen_height: u32) !Self {
        var self = Self{
            .allocator = allocator,
            .width = screen_width,
            .height = screen_height,
            .cell_width = (screen_width + GRID_SIZE - 1) / GRID_SIZE,
            .cell_height = (screen_height + GRID_SIZE - 1) / GRID_SIZE,
            .grid = undefined,
            .damage_rects = std.ArrayList(Rectangle).init(allocator),
        };
        
        // Initialize grid cells
        for (0..GRID_SIZE) |x| {
            for (0..GRID_SIZE) |y| {
                self.grid[x][y] = std.ArrayList(usize).init(allocator);
            }
        }
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        for (0..GRID_SIZE) |x| {
            for (0..GRID_SIZE) |y| {
                self.grid[x][y].deinit();
            }
        }
        self.damage_rects.deinit();
    }
    
    pub fn addDamage(self: *Self, rect: Rectangle) !void {
        const index = self.damage_rects.items.len;
        try self.damage_rects.append(rect);
        
        // Add to spatial grid
        const start_x = @max(0, @min(GRID_SIZE - 1, @as(usize, @intCast(rect.x)) / self.cell_width));
        const end_x = @max(0, @min(GRID_SIZE - 1, @as(usize, @intCast(rect.x + @as(i32, @intCast(rect.width)))) / self.cell_width));
        const start_y = @max(0, @min(GRID_SIZE - 1, @as(usize, @intCast(rect.y)) / self.cell_height));
        const end_y = @max(0, @min(GRID_SIZE - 1, @as(usize, @intCast(rect.y + @as(i32, @intCast(rect.height)))) / self.cell_height));
        
        for (start_x..end_x + 1) |x| {
            for (start_y..end_y + 1) |y| {
                try self.grid[x][y].append(index);
            }
        }
    }
    
    pub fn queryRegion(self: *const Self, query_rect: Rectangle, allocator: std.mem.Allocator) !std.ArrayList(Rectangle) {
        var result = std.ArrayList(Rectangle).init(allocator);
        var visited = std.AutoHashMap(usize, void).init(allocator);
        defer visited.deinit();
        
        const start_x = @max(0, @min(GRID_SIZE - 1, @as(usize, @intCast(query_rect.x)) / self.cell_width));
        const end_x = @max(0, @min(GRID_SIZE - 1, @as(usize, @intCast(query_rect.x + @as(i32, @intCast(query_rect.width)))) / self.cell_width));
        const start_y = @max(0, @min(GRID_SIZE - 1, @as(usize, @intCast(query_rect.y)) / self.cell_height));
        const end_y = @max(0, @min(GRID_SIZE - 1, @as(usize, @intCast(query_rect.y + @as(i32, @intCast(query_rect.height)))) / self.cell_height));
        
        for (start_x..end_x + 1) |x| {
            for (start_y..end_y + 1) |y| {
                for (self.grid[x][y].items) |rect_index| {
                    if (visited.contains(rect_index)) continue;
                    try visited.put(rect_index, {});
                    
                    const damage_rect = self.damage_rects.items[rect_index];
                    if (damage_rect.intersects(query_rect)) {
                        try result.append(damage_rect);
                    }
                }
            }
        }
        
        return result;
    }
    
    pub fn clear(self: *Self) void {
        for (0..GRID_SIZE) |x| {
            for (0..GRID_SIZE) |y| {
                self.grid[x][y].clearRetainingCapacity();
            }
        }
        self.damage_rects.clearRetainingCapacity();
    }
};

pub const DamageHistory = struct {
    const HISTORY_SIZE = 60; // Keep 60 frames of history (~1 second at 60fps)
    
    frames: [HISTORY_SIZE]DamageRegion,
    frame_times: [HISTORY_SIZE]u64,
    current_frame: usize,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) DamageHistory {
        var history = DamageHistory{
            .frames = undefined,
            .frame_times = [_]u64{0} ** HISTORY_SIZE,
            .current_frame = 0,
            .allocator = allocator,
        };
        
        for (0..HISTORY_SIZE) |i| {
            history.frames[i] = DamageRegion.init(allocator);
        }
        
        return history;
    }
    
    pub fn deinit(self: *DamageHistory) void {
        for (0..HISTORY_SIZE) |i| {
            self.frames[i].deinit();
        }
    }
    
    pub fn addFrame(self: *DamageHistory, damage: DamageRegion, timestamp: u64) !void {
        self.current_frame = (self.current_frame + 1) % HISTORY_SIZE;
        
        self.frames[self.current_frame].clear();
        for (damage.rectangles.items) |rect| {
            try self.frames[self.current_frame].addRect(rect);
        }
        self.frame_times[self.current_frame] = timestamp;
    }
    
    pub fn predictDamage(self: *const DamageHistory, allocator: std.mem.Allocator) !DamageRegion {
        var predicted = DamageRegion.init(allocator);
        
        // Simple prediction: areas that were damaged in recent frames are likely to be damaged again
        var heat_map = std.AutoHashMap(u64, u32).init(allocator);
        defer heat_map.deinit();
        
        const current_time = @as(u64, @intCast(std.time.microTimestamp()));
        
        for (0..HISTORY_SIZE) |i| {
            const frame_age = current_time - self.frame_times[i];
            if (frame_age > 1_000_000) continue; // Ignore frames older than 1 second
            
            const weight = @as(u32, @intFromFloat(100.0 - (@as(f32, @floatFromInt(frame_age)) / 10_000.0)));
            
            for (self.frames[i].rectangles.items) |rect| {
                // Quantize rectangle to reduce noise
                const quantized_key = (@as(u64, @intCast(@divTrunc(rect.x, 8))) << 32) |
                                    (@as(u64, @intCast(@divTrunc(rect.y, 8))) << 16) |
                                    (@as(u64, @intCast(rect.width / 8)) << 8) |
                                    @as(u64, @intCast(rect.height / 8));
                
                const current_heat = heat_map.get(quantized_key) orelse 0;
                try heat_map.put(quantized_key, current_heat + weight);
            }
        }
        
        // Add high-heat regions to prediction
        var heat_iter = heat_map.iterator();
        while (heat_iter.next()) |entry| {
            if (entry.value_ptr.* > 200) { // High prediction confidence
                const key = entry.key_ptr.*;
                const x = @as(i32, @intCast((key >> 32) & 0xFFFF)) * 8;
                const y = @as(i32, @intCast((key >> 16) & 0xFFFF)) * 8;
                const w = @as(u32, @intCast((key >> 8) & 0xFF)) * 8;
                const h = @as(u32, @intCast(key & 0xFF)) * 8;
                
                try predicted.addRect(Rectangle.init(x, y, w, h));
            }
        }
        
        try predicted.optimize();
        return predicted;
    }
    
    pub fn getRecentDamage(self: *const DamageHistory, frames_back: usize, allocator: std.mem.Allocator) !DamageRegion {
        var combined = DamageRegion.init(allocator);
        
        for (0..@min(frames_back, HISTORY_SIZE)) |i| {
            const frame_index = (self.current_frame + HISTORY_SIZE - i) % HISTORY_SIZE;
            for (self.frames[frame_index].rectangles.items) |rect| {
                try combined.addRect(rect);
            }
        }
        
        try combined.optimize();
        return combined;
    }
};

pub const DamageTracker = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    screen_width: u32,
    screen_height: u32,
    current_damage: DamageRegion,
    accumulated_damage: DamageRegion,
    spatial_index: SpatialIndex,
    damage_history: DamageHistory,
    frame_count: u64,
    last_commit_time: u64,
    
    // Performance counters
    total_damage_area: u64,
    optimized_area_saved: u64,
    gpu_acceleration_enabled: bool,
    
    pub fn init(allocator: std.mem.Allocator, screen_width: u32, screen_height: u32) !Self {
        return Self{
            .allocator = allocator,
            .screen_width = screen_width,
            .screen_height = screen_height,
            .current_damage = DamageRegion.init(allocator),
            .accumulated_damage = DamageRegion.init(allocator),
            .spatial_index = try SpatialIndex.init(allocator, screen_width, screen_height),
            .damage_history = DamageHistory.init(allocator),
            .frame_count = 0,
            .last_commit_time = 0,
            .total_damage_area = 0,
            .optimized_area_saved = 0,
            .gpu_acceleration_enabled = true,
        };
    }
    
    pub fn deinit(self: *Self) void {
        self.current_damage.deinit();
        self.accumulated_damage.deinit();
        self.spatial_index.deinit();
        self.damage_history.deinit();
    }
    
    pub fn addDamage(self: *Self, rect: Rectangle) !void {
        if (rect.isEmpty()) return;
        
        // Clip damage to screen bounds
        const screen_rect = Rectangle.init(0, 0, self.screen_width, self.screen_height);
        const clipped_rect = rect.intersection(screen_rect) orelse return;
        
        try self.current_damage.addRect(clipped_rect);
        try self.accumulated_damage.addRect(clipped_rect);
        try self.spatial_index.addDamage(clipped_rect);
        
        self.total_damage_area += clipped_rect.area();
    }
    
    pub fn addSurfaceDamage(self: *Self, surface_rect: Rectangle, damage_rects: []const Rectangle) !void {
        for (damage_rects) |damage_rect| {
            // Transform damage from surface coordinates to screen coordinates
            const screen_damage = Rectangle.init(
                surface_rect.x + damage_rect.x,
                surface_rect.y + damage_rect.y,
                damage_rect.width,
                damage_rect.height
            );
            
            try self.addDamage(screen_damage);
        }
    }
    
    pub fn optimizeDamage(self: *Self) !void {
        const original_area = self.current_damage.total_area;
        
        // Optimize current damage region
        try self.current_damage.optimize();
        try self.accumulated_damage.optimize();
        
        // Calculate area saved by optimization
        const optimized_area = self.current_damage.total_area;
        if (original_area > optimized_area) {
            self.optimized_area_saved += original_area - optimized_area;
        }
        
        // Limit damage complexity for performance
        if (self.current_damage.rectangles.items.len > 32) {
            // Merge into bounding box if too many small regions
            if (self.current_damage.bounding_box) |bbox| {
                self.current_damage.clear();
                try self.current_damage.addRect(bbox);
            }
        }
    }
    
    pub fn commitDamage(self: *Self) !DamageRegion {
        const timestamp = @as(u64, @intCast(std.time.microTimestamp()));
        
        // Optimize before commit
        try self.optimizeDamage();
        
        // Add to history
        try self.damage_history.addFrame(self.current_damage, timestamp);
        
        // Copy current damage for return
        const committed_damage = try self.current_damage.copy(self.allocator);
        
        // Clear current damage for next frame
        self.current_damage.clear();
        self.spatial_index.clear();
        
        self.frame_count += 1;
        self.last_commit_time = timestamp;
        
        return committed_damage;
    }
    
    pub fn clearAccumulatedDamage(self: *Self) void {
        self.accumulated_damage.clear();
    }
    
    pub fn queryDamageInRegion(self: *const Self, query_rect: Rectangle) !std.ArrayList(Rectangle) {
        return self.spatial_index.queryRegion(query_rect, self.allocator);
    }
    
    pub fn predictNextFrameDamage(self: *const Self) !DamageRegion {
        return self.damage_history.predictDamage(self.allocator);
    }
    
    pub fn getRecentDamagePattern(self: *const Self, frames: usize) !DamageRegion {
        return self.damage_history.getRecentDamage(frames, self.allocator);
    }
    
    pub fn shouldFullRedraw(self: *const Self) bool {
        if (self.accumulated_damage.bounding_box) |bbox| {
            const total_screen_area = @as(u64, self.screen_width) * @as(u64, self.screen_height);
            const damage_coverage = @as(f32, @floatFromInt(bbox.area())) / @as(f32, @floatFromInt(total_screen_area));
            
            // Full redraw if damage covers more than 75% of screen
            return damage_coverage > 0.75;
        }
        
        return false;
    }
    
    pub fn getOptimizationStats(self: *const Self) struct {
        total_damage_area: u64,
        area_saved_by_optimization: u64,
        optimization_ratio: f32,
        average_rects_per_frame: f32,
    } {
        const optimization_ratio = if (self.total_damage_area > 0)
            @as(f32, @floatFromInt(self.optimized_area_saved)) / @as(f32, @floatFromInt(self.total_damage_area))
        else
            0.0;
        
        const avg_rects = if (self.frame_count > 0)
            @as(f32, @floatFromInt(self.spatial_index.damage_rects.items.len)) / @as(f32, @floatFromInt(self.frame_count))
        else
            0.0;
        
        return .{
            .total_damage_area = self.total_damage_area,
            .area_saved_by_optimization = self.optimized_area_saved,
            .optimization_ratio = optimization_ratio,
            .average_rects_per_frame = avg_rects,
        };
    }
    
    pub fn enableGpuAcceleration(self: *Self, enable: bool) void {
        self.gpu_acceleration_enabled = enable;
    }
    
    /// GPU-accelerated damage region computation using compute shaders
    pub fn computeDamageGpu(self: *Self, old_buffer: []const u8, new_buffer: []const u8, threshold: u8) !void {
        if (!self.gpu_acceleration_enabled) return;
        
        // This would use GPU compute shaders to quickly identify changed regions
        // For now, implement a simplified CPU version
        
        const bytes_per_pixel = 4; // Assume RGBA8888
        const pixels_per_row = self.screen_width;
        const block_size = 64; // Compare in 64x64 pixel blocks
        
        var y: u32 = 0;
        while (y < self.screen_height) : (y += block_size) {
            var x: u32 = 0;
            while (x < self.screen_width) : (x += block_size) {
                const block_w = @min(block_size, self.screen_width - x);
                const block_h = @min(block_size, self.screen_height - y);
                
                var changed = false;
                
                // Compare block pixels
                var by: u32 = 0;
                while (by < block_h and !changed) : (by += 1) {
                    var bx: u32 = 0;
                    while (bx < block_w and !changed) : (bx += 1) {
                        const pixel_idx = ((y + by) * pixels_per_row + (x + bx)) * bytes_per_pixel;
                        
                        if (pixel_idx + 3 < old_buffer.len and pixel_idx + 3 < new_buffer.len) {
                            const old_r = old_buffer[pixel_idx];
                            const old_g = old_buffer[pixel_idx + 1];
                            const old_b = old_buffer[pixel_idx + 2];
                            
                            const new_r = new_buffer[pixel_idx];
                            const new_g = new_buffer[pixel_idx + 1];
                            const new_b = new_buffer[pixel_idx + 2];
                            
                            const diff = @max(@max(
                                @abs(@as(i16, old_r) - @as(i16, new_r)),
                                @abs(@as(i16, old_g) - @as(i16, new_g))
                            ), @abs(@as(i16, old_b) - @as(i16, new_b)));
                            
                            if (diff > threshold) {
                                changed = true;
                            }
                        }
                    }
                }
                
                if (changed) {
                    const damage_rect = Rectangle.init(
                        @as(i32, @intCast(x)),
                        @as(i32, @intCast(y)),
                        block_w,
                        block_h
                    );
                    try self.addDamage(damage_rect);
                }
            }
        }
    }
};

/// High-level damage tracking for Wayland surfaces
pub const SurfaceDamageTracker = struct {
    surface_trackers: std.AutoHashMap(u32, DamageTracker), // surface_id -> DamageTracker
    global_tracker: DamageTracker,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, screen_width: u32, screen_height: u32) !SurfaceDamageTracker {
        return SurfaceDamageTracker{
            .surface_trackers = std.AutoHashMap(u32, DamageTracker).init(allocator),
            .global_tracker = try DamageTracker.init(allocator, screen_width, screen_height),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SurfaceDamageTracker) void {
        var iter = self.surface_trackers.valueIterator();
        while (iter.next()) |tracker| {
            tracker.deinit();
        }
        self.surface_trackers.deinit();
        self.global_tracker.deinit();
    }
    
    pub fn addSurface(self: *SurfaceDamageTracker, surface_id: u32, width: u32, height: u32) !void {
        const tracker = try DamageTracker.init(self.allocator, width, height);
        try self.surface_trackers.put(surface_id, tracker);
    }
    
    pub fn removeSurface(self: *SurfaceDamageTracker, surface_id: u32) void {
        if (self.surface_trackers.fetchRemove(surface_id)) |kv| {
            kv.value.deinit();
        }
    }
    
    pub fn addSurfaceDamage(self: *SurfaceDamageTracker, surface_id: u32, surface_rect: Rectangle, damage_rects: []const Rectangle) !void {
        // Add to surface-specific tracker
        if (self.surface_trackers.getPtr(surface_id)) |tracker| {
            for (damage_rects) |rect| {
                try tracker.addDamage(rect);
            }
        }
        
        // Add to global tracker with screen coordinates
        try self.global_tracker.addSurfaceDamage(surface_rect, damage_rects);
    }
    
    pub fn commitFrame(self: *SurfaceDamageTracker) !DamageRegion {
        return self.global_tracker.commitDamage();
    }
    
    pub fn getSurfaceTracker(self: *SurfaceDamageTracker, surface_id: u32) ?*DamageTracker {
        return self.surface_trackers.getPtr(surface_id);
    }
    
    pub fn getGlobalTracker(self: *SurfaceDamageTracker) *DamageTracker {
        return &self.global_tracker;
    }
};