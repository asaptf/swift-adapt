import Foundation

/// Runtime device capability checks for on-device training (architecture §3).
///
/// Training is gated at **runtime**, not compile time: the package declares
/// iOS 18+ / macOS 15+, but LoRA training still needs enough unified memory.
/// Default floor: **6 GB** physical RAM.
public enum AdaptCapability: Sendable {
    /// Minimum physical memory required to attempt training (bytes).
    ///
    /// Architecture §3: "Training requires ≥ 6 GB RAM devices". This is a
    /// product floor, not a measured peak-step number for a specific model.
    public static let minimumTrainingMemoryBytes: UInt64 = 6 * 1_024 * 1_024 * 1_024

    /// Returns `true` when `physicalMemoryBytes` meets the training floor.
    public static func canTrain(physicalMemoryBytes: UInt64) -> Bool {
        physicalMemoryBytes >= minimumTrainingMemoryBytes
    }

    /// Current device physical memory from `ProcessInfo`.
    public static var devicePhysicalMemoryBytes: UInt64 {
        ProcessInfo.processInfo.physicalMemory
    }

    /// Whether this process's host meets the training memory floor.
    public static var deviceCanTrain: Bool {
        canTrain(physicalMemoryBytes: devicePhysicalMemoryBytes)
    }

    /// Throws ``AdaptScheduleError/insufficientMemory`` when below the floor.
    ///
    /// - Parameter physicalMemoryBytes: Injectable for tests; defaults to the
    ///   live `ProcessInfo` value.
    public static func requireTrainingMemory(
        physicalMemoryBytes: UInt64 = AdaptCapability.devicePhysicalMemoryBytes
    ) throws {
        guard canTrain(physicalMemoryBytes: physicalMemoryBytes) else {
            throw AdaptScheduleError.insufficientMemory(
                haveBytes: physicalMemoryBytes,
                needBytes: minimumTrainingMemoryBytes
            )
        }
    }
}
