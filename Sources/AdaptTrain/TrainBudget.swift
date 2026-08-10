import Foundation

/// Resource envelope for a single `Trainer.run` invocation.
///
/// Whichever constraint binds first ends the run cleanly with a checkpoint —
/// exhausting a budget is a normal outcome, not an error.
public struct TrainBudget: Sendable, Hashable {
    /// Maximum optimizer steps to take in this run (not cumulative lifetime).
    public var maxSteps: Int
    /// Wall-clock limit measured from the start of `run`.
    public var maxWallClock: Duration
    /// Soft cap on MLX unified memory in megabytes (`Memory.memoryLimit`).
    public var maxMemoryMB: Int
    /// Stop when `ProcessInfo.thermalState` reaches or exceeds this level.
    public var stopOnThermal: ProcessInfo.ThermalState

    /// Creates a training budget.
    ///
    /// - Parameters:
    ///   - maxSteps: Optimizer steps for this run.
    ///   - maxWallClock: Wall-clock limit (default 1 hour).
    ///   - maxMemoryMB: MLX memory limit in MB (default 4096).
    ///   - stopOnThermal: Thermal threshold (default `.serious`).
    public init(
        maxSteps: Int,
        maxWallClock: Duration = .seconds(3600),
        maxMemoryMB: Int = 4096,
        stopOnThermal: ProcessInfo.ThermalState = .serious
    ) {
        self.maxSteps = maxSteps
        self.maxWallClock = maxWallClock
        self.maxMemoryMB = maxMemoryMB
        self.stopOnThermal = stopOnThermal
    }
}
