import Foundation

/// Metadata for one versioned adapter artifact in a lineage.
///
/// Must never embed user training content (prompts/completions). Weights live
/// in a sibling `adapters.safetensors` file; this type is `version.json` only.
public struct AdapterVersion: Codable, Sendable, Hashable {
    /// Lineage this version belongs to.
    public let lineage: AdapterLineage
    /// Monotonic version number within the lineage (1-based).
    public let version: Int
    /// Parent version this was fine-tuned from, if any.
    public let parentVersion: Int?
    /// Data window metadata (counts and dates only).
    public let trainedOn: TrainingWindow
    /// Optional evaluation results (placeholder until M3).
    public let evalReport: EvalReport?
    /// Lifecycle status.
    public let status: AdapterStatus
    /// SHA-256 hex digest of the weights file for integrity checks.
    public let weightsDigest: String
    /// When this version metadata was written.
    public let createdAt: Date

    /// Creates adapter version metadata.
    public init(
        lineage: AdapterLineage,
        version: Int,
        parentVersion: Int? = nil,
        trainedOn: TrainingWindow,
        evalReport: EvalReport? = nil,
        status: AdapterStatus = .candidate,
        weightsDigest: String,
        createdAt: Date = Date()
    ) {
        self.lineage = lineage
        self.version = version
        self.parentVersion = parentVersion
        self.trainedOn = trainedOn
        self.evalReport = evalReport
        self.status = status
        self.weightsDigest = weightsDigest
        self.createdAt = createdAt
    }

    /// Returns a copy with an updated status.
    public func with(status: AdapterStatus) -> AdapterVersion {
        AdapterVersion(
            lineage: lineage,
            version: version,
            parentVersion: parentVersion,
            trainedOn: trainedOn,
            evalReport: evalReport,
            status: status,
            weightsDigest: weightsDigest,
            createdAt: createdAt
        )
    }
}
