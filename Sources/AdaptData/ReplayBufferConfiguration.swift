import Foundation

/// Tunables for a lineage ``ReplayBuffer``.
public struct ReplayBufferConfiguration: Sendable, Equatable {
    /// Maximum examples retained for the lineage (capacity bound).
    public var maxExamples: Int

    /// How long examples are kept before TTL pruning deletes them —
    /// **including** examples never trained on and **including** IDs that may
    /// be referenced by a held-out pin. Default 30 days (§4.2).
    public var retention: Duration

    /// Maximum new captures accepted per UTC calendar day for this lineage.
    public var maxCapturesPerDay: Int

    /// Scrubber pipeline run at capture time before any disk write.
    public var scrubberPipeline: ScrubberPipeline

    /// Creates a configuration.
    ///
    /// - Parameters:
    ///   - maxExamples: Capacity bound (≥ 1).
    ///   - retention: TTL window (must be positive).
    ///   - maxCapturesPerDay: Daily privacy budget (≥ 1).
    ///   - scrubberPipeline: Capture-time scrubbers (defaults to builtins).
    public init(
        maxExamples: Int = 10_000,
        retention: Duration = .seconds(30 * 24 * 60 * 60),
        maxCapturesPerDay: Int = 200,
        scrubberPipeline: ScrubberPipeline = .builtins
    ) {
        self.maxExamples = maxExamples
        self.retention = retention
        self.maxCapturesPerDay = maxCapturesPerDay
        self.scrubberPipeline = scrubberPipeline
    }

    /// Validates configuration; throws ``AdaptDataError/invalidArgument(_:)`` if unusable.
    public func validate() throws {
        guard maxExamples >= 1 else {
            throw AdaptDataError.invalidArgument("maxExamples must be ≥ 1")
        }
        guard maxCapturesPerDay >= 1 else {
            throw AdaptDataError.invalidArgument("maxCapturesPerDay must be ≥ 1")
        }
        guard retention > .zero else {
            throw AdaptDataError.invalidArgument("retention must be positive")
        }
    }
}

extension ScrubberPipeline: Equatable {
    public static func == (lhs: ScrubberPipeline, rhs: ScrubberPipeline) -> Bool {
        // Compare by ordered scrubber names — pipelines are value-configured.
        lhs.scrubbers.map(\.name) == rhs.scrubbers.map(\.name)
    }
}
