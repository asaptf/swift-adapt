import Foundation

/// Errors thrown by AdaptSchedule (architecture §4.6, §8).
///
/// Distinctly named so multi-module imports do not collide. Conforms to
/// `LocalizedError` for host-app presentation. Contains **no** test-only cases.
public enum AdaptScheduleError: Error, LocalizedError, Sendable, Equatable {
    /// Device physical memory is below the training floor (§3: ≥ 6 GB).
    case insufficientMemory(haveBytes: UInt64, needBytes: UInt64)

    /// `ProcessInfo.thermalState` is at or above the abort threshold.
    case thermalStateTooHigh(ProcessInfo.ThermalState)

    /// Battery fraction is below the configured minimum (iOS policy).
    case batteryTooLow(level: Double, minimum: Double)

    /// Device is not on external power while the policy requires charging.
    case notCharging

    /// Exponential backoff after repeated gate **refusals** is still active.
    case backoffActive(until: Date, consecutiveRefusals: Int)

    /// Caller supplied an unusable configuration value.
    case invalidConfiguration(String)

    /// A pipeline stage failed with a recoverable description (no user text).
    case stageFailed(stage: PipelineStage, message: String)

    public var errorDescription: String? {
        switch self {
        case .insufficientMemory(let have, let need):
            let haveGB = Double(have) / 1_073_741_824.0
            let needGB = Double(need) / 1_073_741_824.0
            return String(
                format:
                    "Device has %.2f GB RAM; training requires at least %.0f GB (AdaptCapability).",
                haveGB,
                needGB
            )
        case .thermalStateTooHigh(let state):
            return "Aborting Adapt pipeline: thermal state is \(Self.thermalName(state))."
        case .batteryTooLow(let level, let minimum):
            return String(
                format: "Battery at %.0f%%; Adapt requires ≥ %.0f%% and charging.",
                level * 100,
                minimum * 100
            )
        case .notCharging:
            return "Device is not charging; Adapt night pipeline requires external power."
        case .backoffActive(let until, let n):
            return
                "Training deferred by exponential backoff after \(n) consecutive gate refusal(s); next eligible \(until)."
        case .invalidConfiguration(let message):
            return "Invalid AdaptSchedule configuration: \(message)"
        case .stageFailed(let stage, let message):
            return "Adapt pipeline stage \(stage.rawValue) failed: \(message)"
        }
    }

    private static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown(\(state.rawValue))"
        }
    }
}

extension ProcessInfo.ThermalState: @retroactive Comparable {
    public static func < (lhs: ProcessInfo.ThermalState, rhs: ProcessInfo.ThermalState) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}
