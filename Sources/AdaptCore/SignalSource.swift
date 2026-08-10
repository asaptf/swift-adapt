import Foundation

/// Origin of a training example in the personalization pipeline.
public enum SignalSource: String, Codable, Sendable, Hashable, CaseIterable {
    /// User rewrote generated text — highest-value gold signal.
    case explicitEdit
    /// User accepted generated text unchanged.
    case acceptance
    /// User dismissed a suggestion (negative / preference signal).
    case rejection
    /// App-provided seed or synthetic example.
    case synthetic
}
