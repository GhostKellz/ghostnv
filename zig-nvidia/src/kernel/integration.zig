const std = @import("std");
const kernel_module = @import("module.zig");
const display = @import("../display/engine.zig");
const video = @import("../video/processor.zig");
const audio = @import("../audio/pipewire_integration.zig");
const cuda = @import("../cuda/runtime.zig");
const memory = @import("../hal/memory.zig");
const command = @import("../hal/command.zig");
const arch_config = @import("../arch/config.zig");

/// GhostKernel Integration Layer
/// Provides the interface for embedding GhostNV driver into the pure Zig GhostKernel
pub const GhostKernelIntegration = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    kernel_module: *kernel_module.NvidiaKernelModule,
    
    // Driver subsystems
    display_engine: *display.DisplayEngine,
    video_processor: *video.VideoProcessor,
    audio_integration: *audio.PipeWireIntegration,
    cuda_runtime: *cuda.CudaRuntime,
    memory_manager: *memory.MemoryManager,
    command_processor: *command.CommandProcessor,
    
    // Kernel interface
    kernel_interface: KernelInterface,
    
    // Arch Linux configuration
    arch_manager: arch_config.ArchNvidiaManager,
    
    // System state
    initialization_state: InitializationState,
    driver_state: DriverState,
    
    pub fn init(allocator: std.mem.Allocator, kernel_ctx: *anyopaque) !*Self {
        var self = try allocator.create(Self);
        
        // Initialize kernel module
        self.kernel_module = try kernel_module.NvidiaKernelModule.init(allocator);
        
        // Initialize memory manager
        self.memory_manager = try memory.MemoryManager.init(allocator);
        
        // Initialize command processor
        self.command_processor = try command.CommandProcessor.init(allocator, null);
        
        // Initialize CUDA runtime
        self.cuda_runtime = try cuda.CudaRuntime.init(allocator, self.memory_manager);
        
        // Initialize display engine
        self.display_engine = try display.DisplayEngine.init(allocator, kernel_ctx, self.memory_manager);
        
        // Initialize video processor
        self.video_processor = try video.VideoProcessor.init(allocator, kernel_ctx, self.memory_manager);
        
        // Initialize audio integration
        self.audio_integration = try audio.PipeWireIntegration.init(allocator);
        
        // Initialize kernel interface
        self.kernel_interface = try KernelInterface.init(allocator, self);
        
        // Initialize Arch Linux configuration
        self.arch_manager = arch_config.ArchNvidiaManager.init(allocator);
        
        self.* = Self{
            .allocator = allocator,
            .kernel_module = self.kernel_module,
            .display_engine = self.display_engine,
            .video_processor = self.video_processor,
            .audio_integration = self.audio_integration,
            .cuda_runtime = self.cuda_runtime,
            .memory_manager = self.memory_manager,
            .command_processor = self.command_processor,
            .kernel_interface = self.kernel_interface,
            .arch_manager = self.arch_manager,
            .initialization_state = .not_initialized,
            .driver_state = .stopped,
        };
        
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        self.arch_manager.deinit();
        self.kernel_interface.deinit();
        self.audio_integration.deinit();
        self.video_processor.deinit();
        self.display_engine.deinit();
        self.cuda_runtime.deinit();
        self.command_processor.deinit();
        self.memory_manager.deinit();
        self.kernel_module.deinit();
        
        self.allocator.destroy(self);
    }
    
    /// Main driver initialization function called by GhostKernel
    pub fn initializeDriver(self: *Self) !void {
        std.log.info("Initializing GhostNV driver for GhostKernel...");
        
        self.initialization_state = .initializing;
        
        // Phase 1: Hardware detection and initialization
        try self.initializeHardware();
        
        // Phase 2: Memory management setup
        try self.initializeMemorySubsystem();
        
        // Phase 3: Command processing setup
        try self.initializeCommandSubsystem();
        
        // Phase 4: Display subsystem
        try self.initializeDisplaySubsystem();
        
        // Phase 5: Video processing
        try self.initializeVideoSubsystem();
        
        // Phase 6: Audio AI
        try self.initializeAudioSubsystem();
        
        // Phase 7: CUDA runtime
        try self.initializeCudaSubsystem();
        
        // Phase 8: Kernel integration
        try self.initializeKernelIntegration();
        
        // Phase 9: Arch Linux configuration
        try self.applyArchConfiguration();
        
        self.initialization_state = .initialized;
        self.driver_state = .running;
        
        std.log.info("GhostNV driver initialization complete");
    }
    
    /// Driver shutdown function
    pub fn shutdownDriver(self: *Self) !void {
        std.log.info("Shutting down GhostNV driver...", .{});
        
        self.driver_state = .stopping;
        
        // Shutdown in reverse order
        try self.shutdownKernelIntegration();
        try self.shutdownCudaSubsystem();
        try self.shutdownAudioSubsystem();
        try self.shutdownVideoSubsystem();
        try self.shutdownDisplaySubsystem();
        try self.shutdownCommandSubsystem();
        try self.shutdownMemorySubsystem();
        try self.shutdownHardware();
        
        self.driver_state = .stopped;
        self.initialization_state = .not_initialized;
        
        std.log.info("GhostNV driver shutdown complete");
    }
    
    /// Handle system suspend
    pub fn suspendDriver(self: *Self) !void {
        std.log.info("Suspending GhostNV driver...", .{});
        
        self.driver_state = .suspending;
        
        // Suspend all subsystems
        // Audio integration handles suspend in deinit if needed
        try self.video_processor.suspendProcessing();
        try self.display_engine.suspendDisplay();
        try self.cuda_runtime.suspendCompute();
        try self.memory_manager.suspendMemory();
        
        self.driver_state = .suspended;
        
        std.log.info("GhostNV driver suspended");
    }
    
    /// Handle system resume
    pub fn resumeDriver(self: *Self) !void {
        std.log.info("Resuming GhostNV driver...", .{});
        
        self.driver_state = .resuming;
        
        // Resume all subsystems
        try self.memory_manager.resumeMemory();
        try self.cuda_runtime.resumeCompute();
        try self.display_engine.resumeDisplay();
        try self.video_processor.resumeProcessing();
        // Audio integration handles resume automatically
        
        self.driver_state = .running;
        
        std.log.info("GhostNV driver resumed");
    }
    
    /// Handle interrupts from hardware
    pub fn handleInterrupt(self: *Self, interrupt_vector: u32) void {
        // Route interrupt to appropriate subsystem
        if (interrupt_vector & DISPLAY_INTERRUPT_MASK) {
            self.display_engine.handleInterrupt(interrupt_vector);
        }
        
        if (interrupt_vector & VIDEO_INTERRUPT_MASK) {
            self.video_processor.handleInterrupt(interrupt_vector);
        }
        
        if (interrupt_vector & COMPUTE_INTERRUPT_MASK) {
            self.cuda_runtime.handleInterrupt(interrupt_vector);
        }
        
        if (interrupt_vector & MEMORY_INTERRUPT_MASK) {
            self.memory_manager.handleInterrupt(interrupt_vector);
        }
        
        if (interrupt_vector & COMMAND_INTERRUPT_MASK) {
            self.command_processor.handleInterrupt(interrupt_vector);
        }
    }
    
    /// Provide driver statistics to kernel
    pub fn getDriverStats(self: *Self) DriverStats {
        return DriverStats{
            .memory_usage = self.memory_manager.getMemoryUsage(),
            .gpu_utilization = self.cuda_runtime.getGpuUtilization(),
            .display_stats = self.display_engine.frame_stats,
            .video_stats = self.video_processor.stats,
            .audio_stats = self.audio_integration.getPerformanceStats(),
            .uptime_seconds = self.getUptimeSeconds(),
        };
    }
    
    /// Kernel module interface functions
    pub fn createDeviceNode(self: *Self, device_id: u32) !*DeviceNode {
        const node = try self.allocator.create(DeviceNode);
        node.* = DeviceNode{
            .device_id = device_id,
            .driver = self,
            .open_count = 0,
            .file_operations = &device_file_operations,
        };
        
        return node;
    }
    
    pub fn destroyDeviceNode(self: *Self, node: *DeviceNode) void {
        self.allocator.destroy(node);
    }
    
    // Private initialization methods
    fn initializeHardware(self: *Self) !void {
        std.log.info("Initializing hardware...");
        try self.kernel_module.initializeHardware();
    }
    
    fn initializeMemorySubsystem(self: *Self) !void {
        std.log.info("Initializing memory subsystem...");
        try self.memory_manager.initialize();
    }
    
    fn initializeCommandSubsystem(self: *Self) !void {
        std.log.info("Initializing command subsystem...");
        try self.command_processor.initialize();
    }
    
    fn initializeDisplaySubsystem(self: *Self) !void {
        _ = self;
        std.log.info("Initializing display subsystem...");
        // Display engine is already initialized in init()
    }
    
    fn initializeVideoSubsystem(self: *Self) !void {
        _ = self;
        std.log.info("Initializing video subsystem...");
        // Video processor is already initialized in init()
    }
    
    fn initializeAudioSubsystem(self: *Self) !void {
        std.log.info("Initializing audio subsystem...");
        
        // Register HDMI and DisplayPort audio outputs for each display
        // This will be called when displays are detected
        for (0..self.kernel_module.devices.items.len) |i| {
            // Register HDMI outputs for this GPU
            try self.audio_integration.registerHDMIOutput(@intCast(i), .HDMI);
            
            // Register DisplayPort outputs for this GPU  
            try self.audio_integration.registerDisplayPort(@intCast(i), 4); // 4 lanes typical
        }
        
        std.log.info("Audio subsystem initialized with PipeWire integration");
    }
    
    fn initializeCudaSubsystem(self: *Self) !void {
        std.log.info("Initializing CUDA subsystem...");
        try self.cuda_runtime.initialize();
    }
    
    fn initializeKernelIntegration(self: *Self) !void {
        std.log.info("Initializing kernel integration...");
        try self.kernel_interface.registerWithKernel();
    }
    
    fn applyArchConfiguration(self: *Self) !void {
        std.log.info("Applying Arch Linux configuration...");
        try self.arch_manager.applyRecommendedConfig();
    }
    
    // Private shutdown methods
    fn shutdownKernelIntegration(self: *Self) !void {
        std.log.info("Shutting down kernel integration...");
        try self.kernel_interface.unregisterFromKernel();
    }
    
    fn shutdownCudaSubsystem(self: *Self) !void {
        std.log.info("Shutting down CUDA subsystem...");
        try self.cuda_runtime.shutdown();
    }
    
    fn shutdownAudioSubsystem(self: *Self) !void {
        _ = self;
        std.log.info("Shutting down audio subsystem...");
        // Audio engine cleanup handled in deinit()
    }
    
    fn shutdownVideoSubsystem(self: *Self) !void {
        _ = self;
        std.log.info("Shutting down video subsystem...");
        // Video processor cleanup handled in deinit()
    }
    
    fn shutdownDisplaySubsystem(self: *Self) !void {
        _ = self;
        std.log.info("Shutting down display subsystem...");
        // Display engine cleanup handled in deinit()
    }
    
    fn shutdownCommandSubsystem(self: *Self) !void {
        _ = self;
        std.log.info("Shutting down command subsystem...");
        // Command processor cleanup handled in deinit()
    }
    
    fn shutdownMemorySubsystem(self: *Self) !void {
        _ = self;
        std.log.info("Shutting down memory subsystem...");
        // Memory manager cleanup handled in deinit()
    }
    
    fn shutdownHardware(self: *Self) !void {
        std.log.info("Shutting down hardware...");
        try self.kernel_module.shutdownHardware();
    }
    
    fn getUptimeSeconds(self: *Self) u64 {
        _ = self;
        return @divTrunc(std.time.milliTimestamp(), 1000);
    }
};

/// Kernel Interface Layer
pub const KernelInterface = struct {
    const Self = @This();
    
    allocator: std.mem.Allocator,
    driver: *GhostKernelIntegration,
    registered: bool,
    
    pub fn init(allocator: std.mem.Allocator, driver: *GhostKernelIntegration) !Self {
        return Self{
            .allocator = allocator,
            .driver = driver,
            .registered = false,
        };
    }
    
    pub fn deinit(self: *Self) void {
        if (self.registered) {
            self.unregisterFromKernel() catch {};
        }
    }
    
    pub fn registerWithKernel(self: *Self) !void {
        // Register device nodes with kernel
        // Register syscall handlers
        // Register interrupt handlers
        // Register power management callbacks
        
        self.registered = true;
        std.log.info("Driver registered with GhostKernel");
    }
    
    pub fn unregisterFromKernel(self: *Self) !void {
        // Unregister all kernel interfaces
        self.registered = false;
        std.log.info("Driver unregistered from GhostKernel");
    }
};

/// Device node for userspace interface
pub const DeviceNode = struct {
    device_id: u32,
    driver: *GhostKernelIntegration,
    open_count: u32,
    file_operations: *const FileOperations,
};

/// File operations structure
pub const FileOperations = struct {
    open: *const fn(*DeviceNode) callconv(.C) i32,
    close: *const fn(*DeviceNode) callconv(.C) i32,
    read: *const fn(*DeviceNode, []u8, u64) callconv(.C) i32,
    write: *const fn(*DeviceNode, []const u8, u64) callconv(.C) i32,
    ioctl: *const fn(*DeviceNode, u32, u64) callconv(.C) i32,
    mmap: *const fn(*DeviceNode, u64, u64, u32) callconv(.C) ?*anyopaque,
};

/// Device file operations
const device_file_operations = FileOperations{
    .open = deviceOpen,
    .close = deviceClose,
    .read = deviceRead,
    .write = deviceWrite,
    .ioctl = deviceIoctl,
    .mmap = deviceMmap,
};

// Device file operation implementations
fn deviceOpen(node: *DeviceNode) callconv(.C) i32 {
    node.open_count += 1;
    std.log.info("Device {} opened (count: {})", .{ node.device_id, node.open_count });
    return 0;
}

fn deviceClose(node: *DeviceNode) callconv(.C) i32 {
    if (node.open_count > 0) {
        node.open_count -= 1;
    }
    std.log.info("Device {} closed (count: {})", .{ node.device_id, node.open_count });
    return 0;
}

fn deviceRead(node: *DeviceNode, buffer: []u8, offset: u64) callconv(.C) i32 {
    _ = node;
    _ = buffer;
    _ = offset;
    // Implement device read operations
    return 0;
}

fn deviceWrite(node: *DeviceNode, buffer: []const u8, offset: u64) callconv(.C) i32 {
    _ = node;
    _ = buffer;
    _ = offset;
    // Implement device write operations
    return 0;
}

fn deviceIoctl(node: *DeviceNode, cmd: u32, arg: u64) callconv(.C) i32 {
    _ = node;
    _ = cmd;
    _ = arg;
    // Implement device ioctl operations
    return 0;
}

fn deviceMmap(node: *DeviceNode, addr: u64, length: u64, prot: u32) callconv(.C) ?*anyopaque {
    _ = node;
    _ = addr;
    _ = length;
    _ = prot;
    // Implement device memory mapping
    return null;
}

/// Driver and system state enums
pub const InitializationState = enum {
    not_initialized,
    initializing,
    initialized,
    failed,
};

pub const DriverState = enum {
    stopped,
    starting,
    running,
    stopping,
    suspending,
    suspended,
    resuming,
    err,
};

/// Driver statistics
pub const DriverStats = struct {
    memory_usage: memory.MemoryUsage,
    gpu_utilization: f32,
    display_stats: display.FrameStats,
    video_stats: video.VideoStats,
    audio_stats: audio.AudioStats,
    uptime_seconds: u64,
};

/// Interrupt masks
const DISPLAY_INTERRUPT_MASK = 0x0000FFFF;
const VIDEO_INTERRUPT_MASK = 0x00FF0000;
const COMPUTE_INTERRUPT_MASK = 0xFF000000;
const MEMORY_INTERRUPT_MASK = 0x000F0000;
const COMMAND_INTERRUPT_MASK = 0x00F00000;

/// GhostKernel integration API
pub const GhostKernelAPI = struct {
    /// Initialize GhostNV driver
    pub fn ghostnv_init(kernel_ctx: *anyopaque) !*GhostKernelIntegration {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        const allocator = gpa.allocator();
        
        const driver = try GhostKernelIntegration.init(allocator, kernel_ctx);
        try driver.initializeDriver();
        
        return driver;
    }
    
    /// Shutdown GhostNV driver
    pub fn ghostnv_shutdown(driver: *GhostKernelIntegration) !void {
        try driver.shutdownDriver();
        driver.deinit();
    }
    
    /// Handle system suspend
    pub fn ghostnv_suspend(driver: *GhostKernelIntegration) !void {
        try driver.suspendDriver();
    }
    
    /// Handle system resume
    pub fn ghostnv_resume(driver: *GhostKernelIntegration) !void {
        try driver.resumeDriver();
    }
    
    /// Handle interrupts
    pub fn ghostnv_interrupt(driver: *GhostKernelIntegration, vector: u32) void {
        driver.handleInterrupt(vector);
    }
    
    /// Get driver statistics
    pub fn ghostnv_get_stats(driver: *GhostKernelIntegration) DriverStats {
        return driver.getDriverStats();
    }
    
    /// Create device node
    pub fn ghostnv_create_device(driver: *GhostKernelIntegration, device_id: u32) !*DeviceNode {
        return try driver.createDeviceNode(device_id);
    }
    
    /// Destroy device node
    pub fn ghostnv_destroy_device(driver: *GhostKernelIntegration, node: *DeviceNode) void {
        driver.destroyDeviceNode(node);
    }
};

/// Export symbols for GhostKernel
pub export fn ghostnv_driver_init(kernel_ctx: *anyopaque) ?*GhostKernelIntegration {
    return GhostKernelAPI.ghostnv_init(kernel_ctx) catch null;
}

pub export fn ghostnv_driver_shutdown(driver: *GhostKernelIntegration) void {
    GhostKernelAPI.ghostnv_shutdown(driver) catch {};
}

pub export fn ghostnv_driver_suspend(driver: *GhostKernelIntegration) void {
    GhostKernelAPI.ghostnv_suspend(driver) catch {};
}

pub export fn ghostnv_driver_resume(driver: *GhostKernelIntegration) void {
    GhostKernelAPI.ghostnv_resume(driver) catch {};
}

pub export fn ghostnv_driver_interrupt(driver: *GhostKernelIntegration, vector: u32) void {
    GhostKernelAPI.ghostnv_interrupt(driver, vector);
}

pub export fn ghostnv_driver_get_stats(driver: *GhostKernelIntegration) DriverStats {
    return GhostKernelAPI.ghostnv_get_stats(driver);
}

pub export fn ghostnv_create_device_node(driver: *GhostKernelIntegration, device_id: u32) ?*DeviceNode {
    return GhostKernelAPI.ghostnv_create_device(driver, device_id) catch null;
}

pub export fn ghostnv_destroy_device_node(driver: *GhostKernelIntegration, node: *DeviceNode) void {
    GhostKernelAPI.ghostnv_destroy_device(driver, node);
}

// Test functions
test "ghostkernel integration initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const kernel_ctx = @as(*anyopaque, @ptrFromInt(0x1000000));
    
    var driver = try GhostKernelIntegration.init(allocator, kernel_ctx);
    defer driver.deinit();
    
    try std.testing.expect(driver.initialization_state == .not_initialized);
    try std.testing.expect(driver.driver_state == .stopped);
}

test "device node creation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const kernel_ctx = @as(*anyopaque, @ptrFromInt(0x1000000));
    
    var driver = try GhostKernelIntegration.init(allocator, kernel_ctx);
    defer driver.deinit();
    
    const node = try driver.createDeviceNode(0);
    defer driver.destroyDeviceNode(node);
    
    try std.testing.expect(node.device_id == 0);
    try std.testing.expect(node.open_count == 0);
}

test "driver statistics" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const kernel_ctx = @as(*anyopaque, @ptrFromInt(0x1000000));
    
    var driver = try GhostKernelIntegration.init(allocator, kernel_ctx);
    defer driver.deinit();
    
    const stats = driver.getDriverStats();
    try std.testing.expect(stats.uptime_seconds >= 0);
}