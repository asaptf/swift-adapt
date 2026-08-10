import Foundation

/// Lifecycle state of an adapter version in the registry.
public enum AdapterStatus: String, Codable, Sendable, Hashable {
    /// Trained but not yet promoted (may be a checkpoint).
    case candidate
    /// Currently selected for inference for its lineage.
    case active
    /// Was active; demoted by rollback or a newer promotion.
    case rolledBack
    /// Retained but no longer eligible for promotion without re-evaluation.
    case archived
}
