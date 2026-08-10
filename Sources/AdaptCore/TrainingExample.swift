import Foundation

/// A single training example, already formatted for the target task.
///
/// Examples live in the data layer only. Adapter metadata (`AdapterVersion`)
/// must never embed prompt or completion text.
public struct TrainingExample: Codable, Sendable, Hashable, Identifiable {
    /// Stable identifier for this example.
    public let id: UUID
    /// Model input (context / instruction).
    public let prompt: String
    /// Expected model output.
    public let completion: String
    /// Importance-sampling weight (higher = more influence when sampling).
    public let weight: Double
    /// When the example was captured on-device.
    public let capturedAt: Date
    /// How the example was obtained.
    public let source: SignalSource

    /// Creates a training example.
    ///
    /// - Parameter weight: Importance weight. When `nil` (the default), uses
    ///   `source.defaultWeight` from the §4.2 signal table. Pass an explicit
    ///   value to override.
    public init(
        id: UUID = UUID(),
        prompt: String,
        completion: String,
        weight: Double? = nil,
        capturedAt: Date = Date(),
        source: SignalSource
    ) {
        self.id = id
        self.prompt = prompt
        self.completion = completion
        self.weight = weight ?? source.defaultWeight
        self.capturedAt = capturedAt
        self.source = source
    }
}
