const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const command = @import("command.zig");
const pci = @import("pci.zig");

// Interrupt Handling for NVIDIA GPU Events

pub const InterruptError = error{
    HandlerNotFound,
    RegistrationFailed,
    InvalidVector,
    DeviceNotFound,
    HardwareError,
    AccessDenied,
    OutOfMemory,
    HandlerBusy,
    TimeoutError,
};

pub const InterruptType = enum(u8) {
    command_completion = 0,    // Command completed
    memory_fault = 1,          // Memory access fault
    engine_error = 2,          // Engine error
    thermal_event = 3,         // Thermal event
    power_event = 4,           // Power management event
    display_hotplug = 5,       // Display hotplug/unplug
    pcie_error = 6,            // PCIe error
    firmware_error = 7,        // Firmware error
    timer_interrupt = 8,       // Timer interrupt
    doorbell_ring = 9,         // Doorbell ring
    
    pub fn toString(self: InterruptType) []const u8 {
        return switch (self) {
            .command_completion => "Command Completion",
            .memory_fault => "Memory Fault",
            .engine_error => "Engine Error",
            .thermal_event => "Thermal Event",
            .power_event => "Power Event",
            .display_hotplug => "Display Hotplug",
            .pcie_error => "PCIe Error",
            .firmware_error => "Firmware Error",
            .timer_interrupt => "Timer Interrupt",
            .doorbell_ring => "Doorbell Ring",
        };
    }
};

pub const InterruptPriority = enum(u8) {
    low = 0,
    normal = 1,
    high = 2,
    critical = 3,
    
    pub fn toString(self: InterruptPriority) []const u8 {
        return switch (self) {
            .low => "Low",
            .normal => "Normal",
            .high => "High",
            .critical => "Critical",
        };
    }
};

pub const InterruptVector = struct {
    vector_number: u32,
    interrupt_type: InterruptType,
    priority: InterruptPriority,
    engine: command.EngineType,
    enabled: bool,
    
    pub fn init(vector: u32, int_type: InterruptType, priority: InterruptPriority, engine: command.EngineType) InterruptVector {
        return InterruptVector{
            .vector_number = vector,
            .interrupt_type = int_type,
            .priority = priority,
            .engine = engine,
            .enabled = false,
        };
    }
};

pub const InterruptContext = struct {
    vector: InterruptVector,
    timestamp: u64,
    device_id: u32,
    additional_data: u64,
    
    pub fn init(vector: InterruptVector, device_id: u32, data: u64) InterruptContext {
        return InterruptContext{
            .vector = vector,
            .timestamp = std.time.nanoTimestamp(),
            .device_id = device_id,
            .additional_data = data,
        };
    }
};

pub const InterruptHandler = struct {
    handler_id: u32,
    handler_fn: *const fn (context: *InterruptContext) void,
    user_data: ?*anyopaque,
    enabled: bool,
    call_count: u64,
    last_called: u64,
    
    pub fn init(id: u32, handler: *const fn (context: *InterruptContext) void, data: ?*anyopaque) InterruptHandler {
        return InterruptHandler{
            .handler_id = id,
            .handler_fn = handler,
            .user_data = data,
            .enabled = true,
            .call_count = 0,
            .last_called = 0,
        };
    }
    
    pub fn call(self: *InterruptHandler, context: *InterruptContext) void {
        if (!self.enabled) return;
        
        self.call_count += 1;
        self.last_called = std.time.nanoTimestamp();
        
        self.handler_fn(context);
    }
    
    pub fn enable(self: *InterruptHandler) void {
        self.enabled = true;
    }
    
    pub fn disable(self: *InterruptHandler) void {
        self.enabled = false;
    }
};

pub const InterruptStats = struct {
    vector_number: u32,
    interrupt_type: InterruptType,
    total_count: u64,
    enabled_count: u64,
    handler_count: u32,
    last_interrupt: u64,
    average_latency_ns: u64,
    max_latency_ns: u64,
};

pub const InterruptController = struct {
    allocator: Allocator,
    vectors: std.ArrayList(InterruptVector),
    handlers: std.HashMap(u32, std.ArrayList(InterruptHandler), std.hash_map.DefaultContext(u32), 80),
    interrupt_stats: std.HashMap(u32, InterruptStats, std.hash_map.DefaultContext(u32), 80),
    next_handler_id: std.atomic.Value(u32),
    enabled: bool,
    
    // Hardware interrupt registers (would be memory-mapped in real implementation)
    interrupt_enable_reg: u32,
    interrupt_status_reg: u32,
    interrupt_mask_reg: u32,
    
    pub fn init(allocator: Allocator) InterruptController {
        return InterruptController{
            .allocator = allocator,
            .vectors = std.ArrayList(InterruptVector).init(allocator),
            .handlers = std.HashMap(u32, std.ArrayList(InterruptHandler), std.hash_map.DefaultContext(u32), 80).init(allocator),
            .interrupt_stats = std.HashMap(u32, InterruptStats, std.hash_map.DefaultContext(u32), 80).init(allocator),
            .next_handler_id = std.atomic.Value(u32).init(1),
            .enabled = false,
            .interrupt_enable_reg = 0,
            .interrupt_status_reg = 0,
            .interrupt_mask_reg = 0,
        };
    }
    
    pub fn deinit(self: *InterruptController) void {
        self.disable();
        
        var handler_iter = self.handlers.iterator();
        while (handler_iter.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.handlers.deinit();
        
        self.vectors.deinit();
        self.interrupt_stats.deinit();
    }
    
    pub fn initialize(self: *InterruptController) !void {
        // Initialize common interrupt vectors
        try self.addVector(0, .command_completion, .normal, .graphics);
        try self.addVector(1, .command_completion, .normal, .compute);
        try self.addVector(2, .command_completion, .normal, .copy);
        try self.addVector(3, .command_completion, .normal, .video_encode);
        try self.addVector(4, .command_completion, .normal, .video_decode);
        try self.addVector(5, .memory_fault, .high, .graphics);
        try self.addVector(6, .engine_error, .high, .graphics);
        try self.addVector(7, .thermal_event, .critical, .graphics);
        try self.addVector(8, .power_event, .high, .graphics);
        try self.addVector(9, .display_hotplug, .normal, .display);
        try self.addVector(10, .pcie_error, .critical, .graphics);
        try self.addVector(11, .firmware_error, .critical, .graphics);
        try self.addVector(12, .timer_interrupt, .low, .graphics);
        try self.addVector(13, .doorbell_ring, .normal, .graphics);
        
        // Initialize interrupt statistics
        for (self.vectors.items) |vector| {
            try self.interrupt_stats.put(vector.vector_number, InterruptStats{
                .vector_number = vector.vector_number,
                .interrupt_type = vector.interrupt_type,
                .total_count = 0,
                .enabled_count = 0,
                .handler_count = 0,
                .last_interrupt = 0,
                .average_latency_ns = 0,
                .max_latency_ns = 0,
            });
        }
        
        std.log.info("Initialized interrupt controller with {} vectors", .{self.vectors.items.len});
    }
    
    pub fn addVector(
        self: *InterruptController,
        vector_num: u32,
        int_type: InterruptType,
        priority: InterruptPriority,
        engine: command.EngineType,
    ) !void {
        const vector = InterruptVector.init(vector_num, int_type, priority, engine);
        try self.vectors.append(vector);
        
        // Create empty handler list for this vector
        try self.handlers.put(vector_num, std.ArrayList(InterruptHandler).init(self.allocator));
        
        std.log.debug("Added interrupt vector {}: {} ({} priority, {} engine)", .{
            vector_num,
            int_type.toString(),
            priority.toString(),
            engine.toString(),
        });
    }
    
    pub fn registerHandler(
        self: *InterruptController,
        vector_num: u32,
        handler_fn: *const fn (context: *InterruptContext) void,
        user_data: ?*anyopaque,
    ) !u32 {
        var handler_list = self.handlers.getPtr(vector_num) orelse return InterruptError.InvalidVector;
        
        const handler_id = self.next_handler_id.fetchAdd(1, .acq_rel);
        const handler = InterruptHandler.init(handler_id, handler_fn, user_data);
        
        try handler_list.append(handler);
        
        // Update stats
        if (self.interrupt_stats.getPtr(vector_num)) |stats| {
            stats.handler_count += 1;
        }
        
        std.log.debug("Registered interrupt handler {} for vector {}", .{ handler_id, vector_num });
        return handler_id;
    }
    
    pub fn unregisterHandler(self: *InterruptController, vector_num: u32, handler_id: u32) !void {
        var handler_list = self.handlers.getPtr(vector_num) orelse return InterruptError.InvalidVector;
        
        for (handler_list.items, 0..) |handler, i| {
            if (handler.handler_id == handler_id) {
                _ = handler_list.swapRemove(i);
                
                // Update stats
                if (self.interrupt_stats.getPtr(vector_num)) |stats| {
                    stats.handler_count -= 1;
                }
                
                std.log.debug("Unregistered interrupt handler {} from vector {}", .{ handler_id, vector_num });
                return;
            }
        }
        
        return InterruptError.HandlerNotFound;
    }
    
    pub fn enableVector(self: *InterruptController, vector_num: u32) !void {
        for (self.vectors.items) |*vector| {
            if (vector.vector_number == vector_num) {
                vector.enabled = true;
                
                // Update hardware register (simulate)
                self.interrupt_enable_reg |= (@as(u32, 1) << @intCast(vector_num));
                
                std.log.debug("Enabled interrupt vector {}", .{vector_num});
                return;
            }
        }
        
        return InterruptError.InvalidVector;
    }
    
    pub fn disableVector(self: *InterruptController, vector_num: u32) !void {
        for (self.vectors.items) |*vector| {
            if (vector.vector_number == vector_num) {
                vector.enabled = false;
                
                // Update hardware register (simulate)
                self.interrupt_enable_reg &= ~(@as(u32, 1) << @intCast(vector_num));
                
                std.log.debug("Disabled interrupt vector {}", .{vector_num});
                return;
            }
        }
        
        return InterruptError.InvalidVector;
    }
    
    pub fn enable(self: *InterruptController) void {
        self.enabled = true;
        
        // Enable all vectors by default
        for (self.vectors.items) |*vector| {
            vector.enabled = true;
            self.interrupt_enable_reg |= (@as(u32, 1) << @intCast(vector.vector_number));
        }
        
        std.log.info("Enabled interrupt controller");
    }
    
    pub fn disable(self: *InterruptController) void {
        self.enabled = false;
        
        // Disable all vectors
        for (self.vectors.items) |*vector| {
            vector.enabled = false;
        }
        
        self.interrupt_enable_reg = 0;
        
        std.log.info("Disabled interrupt controller");
    }
    
    pub fn handleInterrupt(self: *InterruptController, vector_num: u32, device_id: u32, data: u64) !void {
        if (!self.enabled) return;
        
        const start_time = std.time.nanoTimestamp();
        
        // Find the vector
        var vector: ?InterruptVector = null;
        for (self.vectors.items) |v| {
            if (v.vector_number == vector_num and v.enabled) {
                vector = v;
                break;
            }
        }
        
        if (vector == null) {
            return InterruptError.InvalidVector;
        }
        
        // Create interrupt context
        var context = InterruptContext.init(vector.?, device_id, data);
        
        // Call all registered handlers for this vector
        if (self.handlers.getPtr(vector_num)) |handler_list| {
            for (handler_list.items) |*handler| {
                handler.call(&context);
            }
        }
        
        // Update statistics
        if (self.interrupt_stats.getPtr(vector_num)) |stats| {
            stats.total_count += 1;
            stats.last_interrupt = context.timestamp;
            
            const latency = std.time.nanoTimestamp() - start_time;
            
            // Update average latency (simple moving average)
            if (stats.total_count == 1) {
                stats.average_latency_ns = latency;
            } else {
                stats.average_latency_ns = (stats.average_latency_ns + latency) / 2;
            }
            
            if (latency > stats.max_latency_ns) {
                stats.max_latency_ns = latency;
            }
        }
        
        std.log.debug("Handled interrupt vector {} for device {} (data: 0x{X})", .{
            vector_num,
            device_id,
            data,
        });
    }
    
    pub fn simulateInterrupt(self: *InterruptController, int_type: InterruptType, engine: command.EngineType, data: u64) !void {
        // Find a vector that matches the type and engine
        for (self.vectors.items) |vector| {
            if (vector.interrupt_type == int_type and vector.engine == engine) {
                try self.handleInterrupt(vector.vector_number, 0, data);
                return;
            }
        }
        
        return InterruptError.InvalidVector;
    }
    
    pub fn pollInterrupts(self: *InterruptController) u32 {
        // Simulate reading hardware interrupt status register
        // In real implementation, this would read from memory-mapped register
        
        var handled_count: u32 = 0;
        
        // Check for pending interrupts (simulate some random activity)
        var prng = std.rand.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
        const random = prng.random();
        
        for (self.vectors.items) |vector| {
            if (vector.enabled and random.boolean()) {
                // Simulate interrupt with random data
                const data = random.int(u64);
                self.handleInterrupt(vector.vector_number, 0, data) catch continue;
                handled_count += 1;
            }
        }
        
        return handled_count;
    }
    
    pub fn getVectorStats(self: *InterruptController, vector_num: u32) ?InterruptStats {
        return self.interrupt_stats.get(vector_num);
    }
    
    pub fn getAllStats(self: *InterruptController) ![]InterruptStats {
        var stats = try self.allocator.alloc(InterruptStats, self.interrupt_stats.count());
        
        var iterator = self.interrupt_stats.valueIterator();
        var i: usize = 0;
        while (iterator.next()) |stat| {
            stats[i] = stat.*;
            i += 1;
        }
        
        return stats;
    }
    
    pub fn printStats(self: *InterruptController) !void {
        const stats = try self.getAllStats();
        defer self.allocator.free(stats);
        
        std.log.info("=== Interrupt Controller Statistics ===");
        std.log.info("Controller enabled: {}", .{self.enabled});
        std.log.info("Total vectors: {}", .{self.vectors.items.len});
        std.log.info("");
        
        for (stats) |stat| {
            std.log.info("Vector {}: {} ({} handlers)", .{
                stat.vector_number,
                stat.interrupt_type.toString(),
                stat.handler_count,
            });
            std.log.info("  Total interrupts: {}", .{stat.total_count});
            std.log.info("  Average latency: {} ns", .{stat.average_latency_ns});
            std.log.info("  Max latency: {} ns", .{stat.max_latency_ns});
            
            if (stat.last_interrupt > 0) {
                const time_since = std.time.nanoTimestamp() - stat.last_interrupt;
                std.log.info("  Last interrupt: {} ns ago", .{time_since});
            }
            std.log.info("");
        }
    }
    
    pub fn clearStats(self: *InterruptController) void {
        var iterator = self.interrupt_stats.valueIterator();
        while (iterator.next()) |stats| {
            stats.total_count = 0;
            stats.enabled_count = 0;
            stats.last_interrupt = 0;
            stats.average_latency_ns = 0;
            stats.max_latency_ns = 0;
        }
        
        std.log.info("Cleared interrupt statistics");
    }
    
    pub fn enableHandler(self: *InterruptController, vector_num: u32, handler_id: u32) !void {
        const handler_list = self.handlers.getPtr(vector_num) orelse return InterruptError.InvalidVector;
        
        for (handler_list.items) |*handler| {
            if (handler.handler_id == handler_id) {
                handler.enable();
                std.log.debug("Enabled handler {} for vector {}", .{ handler_id, vector_num });
                return;
            }
        }
        
        return InterruptError.HandlerNotFound;
    }
    
    pub fn disableHandler(self: *InterruptController, vector_num: u32, handler_id: u32) !void {
        const handler_list = self.handlers.getPtr(vector_num) orelse return InterruptError.InvalidVector;
        
        for (handler_list.items) |*handler| {
            if (handler.handler_id == handler_id) {
                handler.disable();
                std.log.debug("Disabled handler {} for vector {}", .{ handler_id, vector_num });
                return;
            }
        }
        
        return InterruptError.HandlerNotFound;
    }
};

// Common interrupt handlers
pub fn commandCompletionHandler(context: *InterruptContext) void {
    std.log.debug("Command completion interrupt: engine={}, data=0x{X}", .{
        context.vector.engine.toString(),
        context.additional_data,
    });
    
    // In real implementation, this would:
    // 1. Read command completion status
    // 2. Update fence/semaphore states
    // 3. Wake up waiting threads
    // 4. Update command queue state
}

pub fn memoryFaultHandler(context: *InterruptContext) void {
    std.log.err("Memory fault interrupt: device={}, fault_addr=0x{X}", .{
        context.device_id,
        context.additional_data,
    });
    
    // In real implementation, this would:
    // 1. Read fault address and type
    // 2. Attempt recovery if possible
    // 3. Kill offending context if needed
    // 4. Report error to user space
}

pub fn engineErrorHandler(context: *InterruptContext) void {
    std.log.err("Engine error interrupt: engine={}, error_code=0x{X}", .{
        context.vector.engine.toString(),
        context.additional_data,
    });
    
    // In real implementation, this would:
    // 1. Read error registers
    // 2. Reset engine if needed
    // 3. Report error to application
    // 4. Update error statistics
}

pub fn thermalEventHandler(context: *InterruptContext) void {
    const temperature = @as(u32, @truncate(context.additional_data));
    
    std.log.warn("Thermal event: temperature={}°C", .{temperature});
    
    // In real implementation, this would:
    // 1. Check thermal sensors
    // 2. Adjust clock speeds if needed
    // 3. Update thermal management policy
    // 4. Notify user space thermal daemon
}

pub fn displayHotplugHandler(context: *InterruptContext) void {
    const connector_id = @as(u32, @truncate(context.additional_data));
    const connected = (context.additional_data >> 32) != 0;
    
    std.log.info("Display hotplug: connector={}, connected={}", .{ connector_id, connected });
    
    // In real implementation, this would:
    // 1. Detect connected displays
    // 2. Read EDID information
    // 3. Update display configuration
    // 4. Notify display manager
}

// Test functions
test "interrupt controller initialization" {
    const allocator = std.testing.allocator;
    
    var controller = InterruptController.init(allocator);
    defer controller.deinit();
    
    try controller.initialize();
    
    try std.testing.expect(controller.vectors.items.len > 0);
    try std.testing.expect(controller.handlers.count() > 0);
}

test "interrupt handler registration" {
    const allocator = std.testing.allocator;
    
    var controller = InterruptController.init(allocator);
    defer controller.deinit();
    
    try controller.initialize();
    
    // Register handler
    const handler_id = try controller.registerHandler(0, commandCompletionHandler, null);
    try std.testing.expect(handler_id == 1);
    
    // Enable controller and vector
    controller.enable();
    try controller.enableVector(0);
    
    // Test interrupt handling
    try controller.handleInterrupt(0, 0, 0x12345678);
    
    // Check stats
    const stats = controller.getVectorStats(0);
    try std.testing.expect(stats != null);
    try std.testing.expect(stats.?.total_count == 1);
}

test "interrupt vector management" {
    const allocator = std.testing.allocator;
    
    var controller = InterruptController.init(allocator);
    defer controller.deinit();
    
    // Add custom vector
    try controller.addVector(100, .timer_interrupt, .low, .graphics);
    
    // Enable/disable vector
    try controller.enableVector(100);
    try controller.disableVector(100);
    
    // Test invalid vector
    const result = controller.enableVector(999);
    try std.testing.expectError(InterruptError.InvalidVector, result);
}

test "interrupt simulation" {
    const allocator = std.testing.allocator;
    
    var controller = InterruptController.init(allocator);
    defer controller.deinit();
    
    try controller.initialize();
    controller.enable();
    
    // Register handler
    _ = try controller.registerHandler(0, commandCompletionHandler, null);
    
    // Simulate interrupt
    try controller.simulateInterrupt(.command_completion, .graphics, 0xDEADBEEF);
    
    // Check that interrupt was handled
    const stats = controller.getVectorStats(0);
    try std.testing.expect(stats != null);
    try std.testing.expect(stats.?.total_count == 1);
}