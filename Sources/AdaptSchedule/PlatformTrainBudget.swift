import AdaptTrain
import Foundation

/// Platform default ``TrainBudget`` values for the night pipeline (§4.6).
///
/// ## Do not treat these as measured phone budgets
///
/// §4.6 originally proposed:
/// - iOS: **300 steps or 8 minutes**, whichever first
/// - macOS: **2000 steps or 30 minutes**, whichever first
///
/// On a development Mac we later measured **100 steps of rank-8 LoRA over
/// Qwen3-4B-4bit in ~15 s** (attention-only, ~2.6 M trainable parameters) and
/// ~40 steps in ~6 s. Under that measurement, 300 steps is on the order of a
/// **minute on Mac**, not eight — the two halves of the iOS pair are not
/// describing the same run, and both numbers were written **before** any
/// measurement.
///
/// A phone is not that Mac: the realistic on-device base model is smaller, and
/// per-step cost differs. Architecture §7 still lists "iOS background budget
/// too short for meaningful training" as the project's largest empirical
/// unknown. **M4 makes the budget explicit and measurable; it does not assert
/// a phone resolution.**
///
/// | Field | iOS default | Basis | Status |
/// |---|---|---|---|
/// | `maxSteps` | 300 | §4.6 sketch | **Policy default** — Mac data suggests ~1 min for 4B rank-8; phone unknown |
/// | `maxWallClock` | 8 min | §4.6 sketch | **Guess pending device** — not calibrated to step cost |
/// | `maxSteps` (macOS) | 2000 | §4.6 sketch | Policy default; ~5 min for 4B rank-8 on measured Mac |
/// | `maxWallClock` (macOS) | 30 min | §4.6 sketch | Looser bound than step count for that Mac measurement |
/// | `stopOnThermal` | `.serious` | §4.6 | Product policy |
/// | `maxMemoryMB` | 4096 | Aligns with AdaptTrain default | Soft MLX cap, not a phone measurement |
public enum PlatformTrainBudget: Sendable {
    /// iOS night-pipeline default. See type docs for measurement caveats.
    public static var iOS: TrainBudget {
        TrainBudget(
            maxSteps: 300,
            maxWallClock: .seconds(8 * 60),
            maxMemoryMB: 4096,
            stopOnThermal: .serious
        )
    }

    /// macOS night-pipeline default. See type docs for measurement caveats.
    public static var macOS: TrainBudget {
        TrainBudget(
            maxSteps: 2000,
            maxWallClock: .seconds(30 * 60),
            maxMemoryMB: 4096,
            stopOnThermal: .serious
        )
    }

    /// Platform-appropriate default for the current OS.
    public static var current: TrainBudget {
        #if os(iOS)
        return iOS
        #else
        return macOS
        #endif
    }

    /// Human-readable justification table for docs / inspect output.
    public static var justificationSummary: String {
        """
        Train budget defaults (AdaptSchedule / PlatformTrainBudget):
        - iOS: 300 steps OR 8 minutes (whichever first). Source: architecture §4.6.
          Mac measurement (rank-8 LoRA, Qwen3-4B-4bit, attention-only): ~15 s / 100 steps
          ⇒ 300 steps ≈ 1 min on that Mac — not 8 min. Phone per-step cost unmeasured;
          wall-clock half remains a guess pending device (architecture §7).
        - macOS: 2000 steps OR 30 minutes. Source: architecture §4.6.
          Same Mac measurement ⇒ 2000 steps ≈ 5 min; wall clock is the looser bound.
        - Thermal abort: ProcessInfo.ThermalState.serious (§4.6).
        - maxMemoryMB: 4096 (AdaptTrain default soft cap).
        """
    }
}
