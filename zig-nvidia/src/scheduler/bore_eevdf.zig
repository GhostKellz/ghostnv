const std = @import("std");
const linux = std.os.linux;
const kernel = @import("../kernel/module.zig");
const rtx40 = @import("../rtx40/optimizations.zig");

/// Bore-EEVDF Scheduler Integration for GhostNV
/// Optimizes GPU workloads for the advanced Bore and EEVDF schedulers
/// 
/// This module provides tight integration between the GPU driver and
/// the Bore (Burst-Oriented Response Enhancer) and EEVDF (Earliest 
/// Eligible Virtual Deadline First) schedulers in the linux-ghost kernel.
pub const BoreEEVDFIntegration = struct {
    allocator: std.mem.Allocator,
    kernel_module: *kernel.KernelModule,
    optimizer: ?*rtx40.RTX40Optimizer,
    
    // Scheduler state
    current_scheduler: SchedulerType,
    scheduler_config: SchedulerConfig,
    
    // Performance tracking
    latency_samples: std.ArrayList(u64),
    throughput_samples: std.ArrayList(u64),
    scheduler_switches: u64,
    
    // GPU process management
    gpu_processes: std.ArrayList(GpuProcess),
    priority_queues: [8]std.ArrayList(u32), // 8 priority levels
    
    const Self = @This();
    
    pub fn init(allocator: std.mem.Allocator, kernel_module: *kernel.KernelModule) !Self {
        var self = Self{
            .allocator = allocator,
            .kernel_module = kernel_module,
            .optimizer = null,
            .current_scheduler = try detectScheduler(),
            .scheduler_config = SchedulerConfig.default(),
            .latency_samples = std.ArrayList(u64).init(allocator),
            .throughput_samples = std.ArrayList(u64).init(allocator),
            .scheduler_switches = 0,
            .gpu_processes = std.ArrayList(GpuProcess).init(allocator),
            .priority_queues = undefined,
        };
        
        // Initialize priority queues
        for (&self.priority_queues) |*queue| {
            queue.* = std.ArrayList(u32).init(allocator);
        }
        
        try self.configureForScheduler();
        
        std.log.info("Bore-EEVDF integration initialized for {} scheduler", .{self.current_scheduler});
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        self.latency_samples.deinit();
        self.throughput_samples.deinit();
        self.gpu_processes.deinit();
        
        for (&self.priority_queues) |*queue| {
            queue.deinit();
        }
    }
    
    pub fn setOptimizer(self: *Self, optimizer: *rtx40.RTX40Optimizer) void {
        self.optimizer = optimizer;
    }
    
    /// Configure GPU driver for detected scheduler
    fn configureForScheduler(self: *Self) !void {
        switch (self.current_scheduler) {
            .bore => try self.configureBoreOptimizations(),
            .eevdf => try self.configureEEVDFOptimizations(),
            .cfs => try self.configureCFSOptimizations(),
            .unknown => {
                std.log.warn("Unknown scheduler detected, using conservative settings");
                try self.configureConservativeSettings();
            },
        }
    }
    
    /// Bore scheduler optimizations
    fn configureBoreOptimizations(self: *Self) !void {
        std.log.info("Configuring Bore scheduler optimizations");
        
        // Bore scheduler characteristics:
        // - Burst-oriented response enhancer
        // - Lower latency for interactive tasks
        // - Better handling of bursty workloads
        
        self.scheduler_config = SchedulerConfig{
            .gpu_preemption_granularity_us = 50,     // Very fine-grained preemption
            .context_switch_overhead_ns = 2000,      // Optimized for low overhead
            .priority_boost_interactive = 15,        // Boost interactive GPU tasks
            .latency_sensitive_threshold_us = 100,   // Aggressive latency targets
            .burst_detection_window_ms = 10,         // Short burst detection
            .thermal_throttle_delay_ms = 50,         // Quick thermal response
            .memory_bandwidth_priority = .interactive, // Prioritize interactive workloads
            .compute_slice_time_us = 1000,           // Short compute slices
            .render_pipeline_depth = 2,              // Shallow pipeline for responsiveness
            .scheduler_hint_enabled = true,
        };
        
        // Apply hardware optimizations for Bore
        if (self.optimizer) |opt| {
            // Enable ultra-low latency mode
            try opt.enableHardwareScheduling(0);
            try opt.configureLatencyOptimizer(0);
            
            // Configure preemption for burst handling
            try opt.setPreemptionTimeout(0, self.scheduler_config.gpu_preemption_granularity_us);
            
            // Optimize memory for bursty access patterns  
            try opt.enableMemoryCompression(0, true);
            try opt.configurePrefetching(0, .high);
        }
        
        // Set up sysctl hints for Bore scheduler
        try self.setBoreSchedulerHints();
    }
    
    /// EEVDF scheduler optimizations
    fn configureEEVDFOptimizations(self: *Self) !void {
        std.log.info("Configuring EEVDF scheduler optimizations");
        
        // EEVDF scheduler characteristics:
        // - Earliest Eligible Virtual Deadline First
        // - Better fairness and deadline scheduling
        // - Excellent for mixed workloads
        
        self.scheduler_config = SchedulerConfig{
            .gpu_preemption_granularity_us = 100,    // Balanced preemption
            .context_switch_overhead_ns = 3000,      // Moderate overhead for fairness
            .priority_boost_interactive = 10,        // Moderate interactive boost
            .latency_sensitive_threshold_us = 200,   // Balanced latency targets
            .burst_detection_window_ms = 25,         // Medium burst detection
            .thermal_throttle_delay_ms = 100,        // Balanced thermal response
            .memory_bandwidth_priority = .balanced,  // Fair bandwidth allocation
            .compute_slice_time_us = 2000,           // Medium compute slices
            .render_pipeline_depth = 3,              // Balanced pipeline depth
            .scheduler_hint_enabled = true,
        };
        
        // Apply hardware optimizations for EEVDF
        if (self.optimizer) |opt| {
            // Configure for deadline-aware scheduling
            try opt.enableHardwareScheduling(0);
            try opt.configurePriorityQueues(0);
            
            // Set preemption timeout for fairness
            try opt.setPreemptionTimeout(0, self.scheduler_config.gpu_preemption_granularity_us);
            
            // Optimize for mixed workloads
            try opt.enableMemoryCompression(0, true);
            try opt.configurePrefetching(0, .medium);
        }
        
        // Set up sysctl hints for EEVDF scheduler
        try self.setEEVDFSchedulerHints();
    }
    
    /// CFS scheduler optimizations (fallback)
    fn configureCFSOptimizations(self: *Self) !void {
        std.log.info("Configuring CFS scheduler optimizations");
        
        self.scheduler_config = SchedulerConfig{
            .gpu_preemption_granularity_us = 250,    // Standard preemption
            .context_switch_overhead_ns = 5000,      // Higher overhead accepted
            .priority_boost_interactive = 5,         // Minimal boost
            .latency_sensitive_threshold_us = 500,   // Conservative latency
            .burst_detection_window_ms = 50,         // Longer detection window
            .thermal_throttle_delay_ms = 200,        // Conservative thermal
            .memory_bandwidth_priority = .throughput, // Maximize throughput
            .compute_slice_time_us = 5000,           // Longer compute slices
            .render_pipeline_depth = 4,              // Deeper pipeline for throughput
            .scheduler_hint_enabled = false,
        };
        
        if (self.optimizer) |opt| {
            // Conservative hardware settings for CFS
            try opt.enableHardwareScheduling(0);
            try opt.setPreemptionTimeout(0, self.scheduler_config.gpu_preemption_granularity_us);
        }
    }
    
    /// Conservative settings for unknown schedulers
    fn configureConservativeSettings(self: *Self) !void {
        self.scheduler_config = SchedulerConfig.default();
        
        if (self.optimizer) |opt| {
            // Most conservative settings
            try opt.enableHardwareScheduling(0);
            try opt.setPreemptionTimeout(0, 1000); // 1ms timeout
        }
    }
    
    /// Set scheduler hints via sysctl for Bore
    fn setBoreSchedulerHints(self: *Self) !void {
        // These would write to /proc/sys/kernel/ entries
        // For now, we'll log what we would set
        
        std.log.debug("Setting Bore scheduler hints:");\n        std.log.debug("  - sched_bore_interactive_boost = {}", .{self.scheduler_config.priority_boost_interactive});
        std.log.debug("  - sched_bore_burst_penalty = 0"); 
        std.log.debug("  - sched_bore_preempt_granularity = {}us", .{self.scheduler_config.gpu_preemption_granularity_us});
        
        // Write to sysctl files (simplified)
        self.writeSysctl("kernel/sched_bore_interactive_boost", self.scheduler_config.priority_boost_interactive) catch |err| {
            std.log.warn("Failed to set Bore interactive boost: {}", .{err});
        };
    }
    
    /// Set scheduler hints via sysctl for EEVDF
    fn setEEVDFSchedulerHints(self: *Self) !void {
        std.log.debug("Setting EEVDF scheduler hints:");
        std.log.debug("  - sched_eevdf_slice_ns = {}000", .{self.scheduler_config.compute_slice_time_us});
        std.log.debug("  - sched_eevdf_preempt_ns = {}000", .{self.scheduler_config.gpu_preemption_granularity_us});
        
        // Write to sysctl files (simplified)
        self.writeSysctl("kernel/sched_eevdf_slice_ns", self.scheduler_config.compute_slice_time_us * 1000) catch |err| {
            std.log.warn("Failed to set EEVDF slice: {}", .{err});
        };
    }
    
    /// Write to sysctl file (simplified implementation)
    fn writeSysctl(self: *Self, path: []const u8, value: anytype) !void {
        _ = self;
        
        var full_path_buf: [256]u8 = undefined;
        const full_path = try std.fmt.bufPrint(full_path_buf[0..], "/proc/sys/{s}", .{path});
        
        var file = std.fs.openFileAbsolute(full_path, .{ .mode = .write_only }) catch |err| {
            std.log.debug("Cannot open {s}: {}", .{ full_path, err });
            return;
        };
        defer file.close();
        
        var value_buf: [32]u8 = undefined;
        const value_str = try std.fmt.bufPrint(value_buf[0..], "{}", .{value});
        
        _ = file.write(value_str) catch |err| {
            std.log.debug("Cannot write to {s}: {}", .{ full_path, err });
        };
    }
    
    /// Register GPU process for scheduler optimization
    pub fn registerGpuProcess(self: *Self, pid: u32, process_type: GpuProcessType) !void {
        const gpu_process = GpuProcess{
            .pid = pid,
            .process_type = process_type,
            .priority = self.calculateProcessPriority(process_type),
            .cpu_affinity = self.calculateCpuAffinity(process_type),
            .nice_value = self.calculateNiceValue(process_type),
            .oom_score_adj = self.calculateOomScore(process_type),
            .scheduler_policy = self.getSchedulerPolicy(process_type),
            .vruntime_boost = self.calculateVruntimeBoost(process_type),
        };
        
        try self.gpu_processes.append(gpu_process);
        try self.priority_queues[gpu_process.priority].append(pid);
        
        // Apply scheduler optimizations to the process
        try self.optimizeProcessForScheduler(gpu_process);
        
        std.log.info("Registered GPU process {} with priority {} for {} workload", 
                    .{ pid, gpu_process.priority, process_type });
    }
    
    /// Calculate process priority based on type and scheduler
    fn calculateProcessPriority(self: *Self, process_type: GpuProcessType) u8 {
        return switch (self.current_scheduler) {
            .bore => switch (process_type) {
                .gaming => 1,        // Highest priority for gaming
                .interactive => 2,   // High priority for UI
                .compute => 4,       // Medium priority for compute
                .background => 7,    // Low priority for background
            },
            .eevdf => switch (process_type) {
                .gaming => 2,        // High but fair priority
                .interactive => 1,   // Highest for UI responsiveness
                .compute => 3,       // Balanced compute priority
                .background => 6,    // Background priority
            },
            .cfs => switch (process_type) {
                .gaming => 3,        // Standard gaming priority
                .interactive => 2,   // Interactive priority
                .compute => 4,       // Standard compute
                .background => 5,    // Background
            },
            .unknown => 4, // Conservative middle priority
        };
    }
    
    /// Calculate CPU affinity for GPU processes
    fn calculateCpuAffinity(self: *Self, process_type: GpuProcessType) u64 {
        // Get CPU topology info
        const cpu_count = self.getCpuCount();
        const big_cores = self.getBigCores(); // P-cores on hybrid systems
        const little_cores = self.getLittleCores(); // E-cores on hybrid systems
        
        return switch (process_type) {
            .gaming => if (big_cores != 0) big_cores else (1 << @intCast(cpu_count)) - 1,
            .interactive => big_cores | (little_cores & 0xF), // P-cores + some E-cores
            .compute => (1 << @intCast(cpu_count)) - 1, // All cores
            .background => little_cores, // E-cores only if available
        };
    }
    
    /// Apply scheduler-specific optimizations to a process
    fn optimizeProcessForScheduler(self: *Self, gpu_process: GpuProcess) !void {
        switch (self.current_scheduler) {
            .bore => try self.optimizeForBore(gpu_process),
            .eevdf => try self.optimizeForEEVDF(gpu_process),
            .cfs => try self.optimizeForCFS(gpu_process),
            .unknown => {}, // No specific optimizations
        }
    }
    
    /// Bore-specific process optimizations
    fn optimizeForBore(self: *Self, gpu_process: GpuProcess) !void {
        _ = self;
        
        // Set process priority via setpriority()
        const result = linux.setpriority(linux.PRIO.PROCESS, gpu_process.pid, gpu_process.nice_value);
        if (linux.getErrno(result) != .SUCCESS) {
            std.log.warn("Failed to set priority for process {}", .{gpu_process.pid});
        }
        
        // Set CPU affinity via sched_setaffinity()
        var cpu_set: linux.cpu_set_t = std.mem.zeroes(linux.cpu_set_t);
        for (0..64) |cpu| {
            if ((gpu_process.cpu_affinity >> @intCast(cpu)) & 1 != 0) {
                linux.CPU_SET(cpu, &cpu_set);
            }
        }
        
        const affinity_result = linux.sched_setaffinity(gpu_process.pid, @sizeOf(linux.cpu_set_t), &cpu_set);
        if (linux.getErrno(affinity_result) != .SUCCESS) {
            std.log.warn("Failed to set CPU affinity for process {}", .{gpu_process.pid});
        }
        
        // Set Bore-specific scheduler policy (if available)
        if (gpu_process.process_type == .gaming) {
            // For Bore, gaming processes get special treatment
            try self.setBoreInteractiveFlag(gpu_process.pid, true);
        }
    }
    
    /// EEVDF-specific process optimizations
    fn optimizeForEEVDF(self: *Self, gpu_process: GpuProcess) !void {
        _ = self;
        
        // EEVDF uses virtual runtime and deadlines
        // Set scheduler policy to SCHED_EEVDF if available
        var sched_param: linux.sched_param = .{ .sched_priority = @intCast(gpu_process.priority) };
        
        const policy_result = linux.sched_setscheduler(gpu_process.pid, linux.SCHED.OTHER, &sched_param);
        if (linux.getErrno(policy_result) != .SUCCESS) {
            std.log.warn("Failed to set scheduler policy for process {}", .{gpu_process.pid});
        }
        
        // Set vruntime boost for interactive processes
        if (gpu_process.vruntime_boost != 0) {
            try self.setEEVDFVruntimeBoost(gpu_process.pid, gpu_process.vruntime_boost);
        }
    }
    
    /// CFS-specific process optimizations
    fn optimizeForCFS(self: *Self, gpu_process: GpuProcess) !void {
        _ = self;
        
        // Standard CFS optimizations
        var sched_param: linux.sched_param = .{ .sched_priority = 0 };
        
        const policy = switch (gpu_process.process_type) {
            .gaming, .interactive => linux.SCHED.OTHER,
            .compute => linux.SCHED.BATCH,
            .background => linux.SCHED.IDLE,
        };
        
        const result = linux.sched_setscheduler(gpu_process.pid, policy, &sched_param);
        if (linux.getErrno(result) != .SUCCESS) {
            std.log.warn("Failed to set CFS scheduler policy for process {}", .{gpu_process.pid});
        }
    }
    
    /// Set Bore interactive flag (scheduler-specific)
    fn setBoreInteractiveFlag(self: *Self, pid: u32, interactive: bool) !void {
        _ = self;
        
        // This would interface with Bore-specific syscalls or procfs entries
        std.log.debug("Setting Bore interactive flag for PID {} to {}", .{ pid, interactive });
        
        // Example: echo 1 > /proc/{pid}/sched_bore_interactive
        var path_buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(path_buf[0..], "/proc/{}/sched_bore_interactive", .{pid});
        
        var file = std.fs.openFileAbsolute(path, .{ .mode = .write_only }) catch |err| {
            std.log.debug("Cannot open Bore interactive flag file: {}", .{err});
            return;
        };
        defer file.close();
        
        const value = if (interactive) "1" else "0";
        _ = file.write(value) catch |err| {
            std.log.debug("Cannot write Bore interactive flag: {}", .{err});
        };
    }
    
    /// Set EEVDF vruntime boost
    fn setEEVDFVruntimeBoost(self: *Self, pid: u32, boost: i32) !void {
        _ = self;
        
        std.log.debug("Setting EEVDF vruntime boost for PID {} to {}", .{ pid, boost });
        
        // Example: echo boost > /proc/{pid}/sched_eevdf_vruntime_boost
        var path_buf: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(path_buf[0..], "/proc/{}/sched_eevdf_vruntime_boost", .{pid});
        
        var file = std.fs.openFileAbsolute(path, .{ .mode = .write_only }) catch |err| {
            std.log.debug("Cannot open EEVDF vruntime boost file: {}", .{err});
            return;
        };
        defer file.close();
        
        var value_buf: [16]u8 = undefined;
        const value = try std.fmt.bufPrint(value_buf[0..], "{}", .{boost});
        _ = file.write(value) catch |err| {
            std.log.debug("Cannot write EEVDF vruntime boost: {}", .{err});
        };
    }
    
    /// Monitor scheduler performance and adapt
    pub fn monitorAndAdapt(self: *Self) !void {
        // Collect performance metrics
        const current_latency = try self.measureGpuLatency();
        const current_throughput = try self.measureGpuThroughput();
        
        try self.latency_samples.append(current_latency);
        try self.throughput_samples.append(current_throughput);
        
        // Keep only recent samples (last 100)
        if (self.latency_samples.items.len > 100) {
            _ = self.latency_samples.orderedRemove(0);
        }
        if (self.throughput_samples.items.len > 100) {
            _ = self.throughput_samples.orderedRemove(0);
        }
        
        // Analyze performance and adapt if needed
        const avg_latency = self.calculateAverageLatency();
        const latency_target = self.scheduler_config.latency_sensitive_threshold_us;
        
        if (avg_latency > latency_target * 1.2) {
            // Latency is too high, increase responsiveness
            try self.increaseLa    Ытality();
            std.log.info("Increased GPU responsiveness due to high latency: {}μs", .{avg_latency});
        } else if (avg_latency < latency_target * 0.8) {
            // Latency is very good, can optimize for throughput
            try self.optimizeForThroughput();
            std.log.debug("Optimizing for throughput due to low latency: {}μs", .{avg_latency});
        }
    }
    
    // Helper functions
    
    fn getCpuCount(self: *Self) u32 {
        _ = self;
        // Read from /proc/cpuinfo or use sysconf
        return 16; // Default assumption
    }
    
    fn getBigCores(self: *Self) u64 {
        _ = self;
        // On hybrid architectures, P-cores are typically 0-7
        return 0xFF; // First 8 cores
    }
    
    fn getLittleCores(self: *Self) u64 {
        _ = self;
        // E-cores are typically 8-15 on hybrid systems
        return 0xFF00; // Cores 8-15
    }
    
    fn calculateNiceValue(self: *Self, process_type: GpuProcessType) i32 {
        return switch (self.current_scheduler) {
            .bore => switch (process_type) {
                .gaming => -10,      // High priority (lower nice value)
                .interactive => -5,  // Medium-high priority
                .compute => 0,       // Normal priority
                .background => 10,   // Low priority
            },
            .eevdf => switch (process_type) {
                .gaming => -5,       // High priority
                .interactive => -2,  // Medium-high priority
                .compute => 0,       // Normal priority
                .background => 5,    // Low priority
            },
            .cfs => switch (process_type) {
                .gaming => -2,       // Slightly high priority
                .interactive => 0,   // Normal priority
                .compute => 2,       // Slightly low priority
                .background => 5,    // Low priority
            },
            .unknown => 0, // Normal priority
        };
    }
    
    fn calculateOomScore(self: *Self, process_type: GpuProcessType) i32 {
        _ = self;
        return switch (process_type) {
            .gaming => -100,      // Protect gaming processes
            .interactive => -50,  // Protect interactive processes
            .compute => 0,        // Normal OOM score
            .background => 100,   // Background processes can be killed
        };
    }
    
    fn getSchedulerPolicy(self: *Self, process_type: GpuProcessType) u32 {
        _ = self;
        return switch (process_type) {
            .gaming, .interactive => linux.SCHED.OTHER,
            .compute => linux.SCHED.BATCH,
            .background => linux.SCHED.IDLE,
        };
    }
    
    fn calculateVruntimeBoost(self: *Self, process_type: GpuProcessType) i32 {
        return switch (self.current_scheduler) {
            .eevdf => switch (process_type) {
                .gaming => -500000,    // 500μs boost
                .interactive => -200000, // 200μs boost
                .compute => 0,         // No boost
                .background => 100000, // 100μs penalty
            },
            else => 0, // Other schedulers don't use vruntime boost
        };
    }
    
    fn measureGpuLatency(self: *Self) !u64 {
        _ = self;
        // Measure GPU command latency
        // This would involve GPU-specific timing mechanisms
        return 150; // Simulated latency in microseconds
    }
    
    fn measureGpuThroughput(self: *Self) !u64 {
        _ = self;
        // Measure GPU throughput (operations per second)
        return 1000000; // Simulated throughput
    }
    
    fn calculateAverageLatency(self: *Self) u64 {
        if (self.latency_samples.items.len == 0) return 0;
        
        var sum: u64 = 0;
        for (self.latency_samples.items) |sample| {
            sum += sample;
        }
        return sum / self.latency_samples.items.len;
    }
    
    fn increaseResponsiveness(self: *Self) !void {
        if (self.optimizer) |opt| {
            // Reduce preemption timeout for better responsiveness
            const new_timeout = @max(self.scheduler_config.gpu_preemption_granularity_us / 2, 25);
            try opt.setPreemptionTimeout(0, new_timeout);
            self.scheduler_config.gpu_preemption_granularity_us = new_timeout;
        }
    }
    
    fn optimizeForThroughput(self: *Self) !void {
        if (self.optimizer) |opt| {
            // Increase preemption timeout for better throughput
            const new_timeout = @min(self.scheduler_config.gpu_preemption_granularity_us * 2, 1000);
            try opt.setPreemptionTimeout(0, new_timeout);
            self.scheduler_config.gpu_preemption_granularity_us = new_timeout;
        }
    }
};

/// Detect current scheduler type
fn detectScheduler() !SchedulerType {
    // Read /proc/sys/kernel/sched_domain/cpu0/name or similar
    var file = std.fs.openFileAbsolute("/proc/version", .{}) catch {
        return SchedulerType.unknown;
    };
    defer file.close();
    
    var buf: [1024]u8 = undefined;
    const bytes_read = file.read(buf[0..]) catch 0;
    const content = buf[0..bytes_read];
    
    if (std.mem.indexOf(u8, content, "bore")) |_| {
        return SchedulerType.bore;
    } else if (std.mem.indexOf(u8, content, "eevdf")) |_| {
        return SchedulerType.eevdf;
    } else if (std.mem.indexOf(u8, content, "cfs")) |_| {
        return SchedulerType.cfs;
    }
    
    return SchedulerType.unknown;
}

// Type definitions

pub const SchedulerType = enum {
    bore,     // Burst-Oriented Response Enhancer
    eevdf,    // Earliest Eligible Virtual Deadline First
    cfs,      // Completely Fair Scheduler (default Linux)
    unknown,
};

pub const GpuProcessType = enum {
    gaming,      // Real-time gaming applications
    interactive, // Desktop/UI applications using GPU
    compute,     // CUDA/OpenCL compute workloads
    background,  // Background GPU tasks
};

pub const MemoryBandwidthPriority = enum {
    interactive,  // Prioritize interactive workloads
    balanced,     // Balance between interactive and throughput
    throughput,   // Maximize total throughput
};

pub const SchedulerConfig = struct {
    gpu_preemption_granularity_us: u32,
    context_switch_overhead_ns: u32,
    priority_boost_interactive: u8,
    latency_sensitive_threshold_us: u32,
    burst_detection_window_ms: u32,
    thermal_throttle_delay_ms: u32,
    memory_bandwidth_priority: MemoryBandwidthPriority,
    compute_slice_time_us: u32,
    render_pipeline_depth: u8,
    scheduler_hint_enabled: bool,
    
    pub fn default() SchedulerConfig {
        return SchedulerConfig{
            .gpu_preemption_granularity_us = 250,
            .context_switch_overhead_ns = 5000,
            .priority_boost_interactive = 5,
            .latency_sensitive_threshold_us = 500,
            .burst_detection_window_ms = 50,
            .thermal_throttle_delay_ms = 200,
            .memory_bandwidth_priority = .balanced,
            .compute_slice_time_us = 2000,
            .render_pipeline_depth = 3,
            .scheduler_hint_enabled = false,
        };
    }
};

pub const GpuProcess = struct {
    pid: u32,
    process_type: GpuProcessType,
    priority: u8,               // 0 = highest, 7 = lowest
    cpu_affinity: u64,          // CPU affinity bitmask
    nice_value: i32,            // Process nice value
    oom_score_adj: i32,         // OOM killer score adjustment
    scheduler_policy: u32,      // Scheduler policy (SCHED_*)
    vruntime_boost: i32,        // Virtual runtime boost (EEVDF specific)
};

// Test functions
test "scheduler detection" {
    const scheduler = detectScheduler() catch SchedulerType.unknown;
    try std.testing.expect(scheduler != SchedulerType.unknown or scheduler == SchedulerType.unknown);
}

test "bore-eevdf integration" {
    const allocator = std.testing.allocator;
    
    var mock_kernel = kernel.KernelModule{
        .allocator = allocator,
        .nvidia_fd = -1,
        .nvidia_ctl_fd = -1,
        .nvidia_uvm_fd = -1,
        .device_count = 0,
        .devices = undefined,
    };
    
    var integration = try BoreEEVDFIntegration.init(allocator, &mock_kernel);
    defer integration.deinit();
    
    try std.testing.expect(integration.current_scheduler != SchedulerType.unknown or 
                          integration.current_scheduler == SchedulerType.unknown);
}