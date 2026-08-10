import Foundation

/// Metadata for one versioned adapter artifact in a lineage.
///
/// Must never embed user training content (prompts/completions). Weights live
/// in a sibling `adapters.safetensors` file; this type is `version.json` only.
///
/// ## Codable forward-compatibility
///
/// New optional fields decode as `nil` when absent so older on-disk metadata
/// remains readable. Unknown keys are ignored by `JSONDecoder`.
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
    /// Prompt formatting convention used while training this adapter.
    ///
    /// `nil` means legacy metadata written before the field existed — treat as
    /// ``PromptFormatConvention/rawConcatenation`` via
    /// ``SFTPromptFormatter/convention(fromStored:)``. Inference must refuse to
    /// load an adapter whose stored convention disagrees with the session.
    public let promptFormat: PromptFormatConvention?

    /// Creates adapter version metadata.
    public init(
        lineage: AdapterLineage,
        version: Int,
        parentVersion: Int? = nil,
        trainedOn: TrainingWindow,
        evalReport: EvalReport? = nil,
        status: AdapterStatus = .candidate,
        weightsDigest: String,
        createdAt: Date = Date(),
        promptFormat: PromptFormatConvention? = nil
    ) {
        self.lineage = lineage
        self.version = version
        self.parentVersion = parentVersion
        self.trainedOn = trainedOn
        self.evalReport = evalReport
        self.status = status
        self.weightsDigest = weightsDigest
        self.createdAt = createdAt
        self.promptFormat = promptFormat
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
            createdAt: createdAt,
            promptFormat: promptFormat
        )
    }

    /// Returns a copy with an updated evaluation report (measurement and/or gate).
    public func with(evalReport: EvalReport?) -> AdapterVersion {
        AdapterVersion(
            lineage: lineage,
            version: version,
            parentVersion: parentVersion,
            trainedOn: trainedOn,
            evalReport: evalReport,
            status: status,
            weightsDigest: weightsDigest,
            createdAt: createdAt,
            promptFormat: promptFormat
        )
    }

    // Explicit Codable so a missing `promptFormat` key decodes as `nil`
    // (synthesized would too for Optional, but this documents the contract and
    // keeps encode stable if we add non-optional fields later).
    private enum CodingKeys: String, CodingKey {
        case lineage, version, parentVersion, trainedOn, evalReport
        case status, weightsDigest, createdAt, promptFormat
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        lineage = try c.decode(AdapterLineage.self, forKey: .lineage)
        version = try c.decode(Int.self, forKey: .version)
        parentVersion = try c.decodeIfPresent(Int.self, forKey: .parentVersion)
        trainedOn = try c.decode(TrainingWindow.self, forKey: .trainedOn)
        evalReport = try c.decodeIfPresent(EvalReport.self, forKey: .evalReport)
        status = try c.decode(AdapterStatus.self, forKey: .status)
        weightsDigest = try c.decode(String.self, forKey: .weightsDigest)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        promptFormat = try c.decodeIfPresent(PromptFormatConvention.self, forKey: .promptFormat)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(lineage, forKey: .lineage)
        try c.encode(version, forKey: .version)
        try c.encodeIfPresent(parentVersion, forKey: .parentVersion)
        try c.encode(trainedOn, forKey: .trainedOn)
        try c.encodeIfPresent(evalReport, forKey: .evalReport)
        try c.encode(status, forKey: .status)
        try c.encode(weightsDigest, forKey: .weightsDigest)
        try c.encode(createdAt, forKey: .createdAt)
        try c.encodeIfPresent(promptFormat, forKey: .promptFormat)
    }
}
