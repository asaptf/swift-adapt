import Foundation

/// Origin of a training example in the personalization pipeline.
///
/// Default importance weights (architecture §4.2) are the “secret sauce” of
/// how strongly each signal influences training when the caller does not set
/// an explicit `TrainingExample.weight`.
public enum SignalSource: String, Codable, Sendable, Hashable, CaseIterable {
    /// User rewrote generated text — highest-value gold signal.
    case explicitEdit
    /// User accepted generated text unchanged.
    case acceptance
    /// User dismissed a suggestion (negative / preference signal).
    case rejection
    /// App-provided seed or synthetic example.
    case synthetic

    /// Default importance weight for examples from this source (§4.2 table).
    ///
    /// | Source | Weight |
    /// |---|---|
    /// | `.explicitEdit` | 1.0 (gold) |
    /// | `.acceptance` | 0.6 |
    /// | `.rejection` | 0.4 |
    /// | `.synthetic` | 0.3 |
    public var defaultWeight: Double {
        switch self {
        case .explicitEdit: return 1.0
        case .acceptance: return 0.6
        case .rejection: return 0.4
        case .synthetic: return 0.3
        }
    }
}
