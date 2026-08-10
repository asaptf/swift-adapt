import AdaptCore
import Foundation

/// Token-weighted mean cross-entropy aggregation (model-free).
///
/// Real forward passes live in ``HeldOutLossRunner``; this type only combines
/// per-example `(sum of CE, supervised token count)` into a single mean so
/// offline tests can check the arithmetic without MLX.
public enum HeldOutLoss {
    /// Result of scoring an adapter on held-out examples.
    public struct Result: Sendable, Equatable {
        /// Mean cross-entropy in nats per supervised token.
        public var meanCrossEntropyNats: Double
        /// Examples that contributed at least one supervised token.
        public var exampleCount: Int
        /// Total supervised tokens across those examples.
        public var supervisedTokenCount: Int
        /// Examples skipped (empty, over max length, or no supervised tokens).
        public var skippedExampleCount: Int

        public init(
            meanCrossEntropyNats: Double,
            exampleCount: Int,
            supervisedTokenCount: Int,
            skippedExampleCount: Int = 0
        ) {
            self.meanCrossEntropyNats = meanCrossEntropyNats
            self.exampleCount = exampleCount
            self.supervisedTokenCount = supervisedTokenCount
            self.skippedExampleCount = skippedExampleCount
        }

        /// ``EvalReport`` carrying this measurement (no gate decision).
        public var evalReport: EvalReport {
            EvalReport.heldOutCrossEntropy(
                meanNats: meanCrossEntropyNats,
                exampleCount: exampleCount,
                supervisedTokenCount: supervisedTokenCount,
                notes: skippedExampleCount > 0
                    ? "skipped \(skippedExampleCount) held-out example(s)"
                    : nil
            )
        }
    }

    /// One example's contribution: sum of per-token CE over supervised targets,
    /// and the number of those targets.
    public struct ExampleContribution: Sendable, Equatable {
        public var crossEntropySum: Double
        public var supervisedTokens: Int

        public init(crossEntropySum: Double, supervisedTokens: Int) {
            self.crossEntropySum = crossEntropySum
            self.supervisedTokens = supervisedTokens
        }
    }

    /// Token-weighted mean: `sum(ceSum) / sum(tokens)`.
    ///
    /// Returns `nil` when no supervised tokens are present. Per-example means
    /// must be converted to sums (`mean * tokenCount`) before calling this —
    /// averaging means would overweight short examples.
    public static func aggregate(
        _ contributions: [ExampleContribution],
        skippedExampleCount: Int = 0
    ) -> Result? {
        var sum = 0.0
        var tokens = 0
        var examples = 0
        for c in contributions {
            guard c.supervisedTokens > 0 else { continue }
            // Reject non-finite contributions rather than poisoning the mean.
            guard c.crossEntropySum.isFinite else { continue }
            sum += c.crossEntropySum
            tokens += c.supervisedTokens
            examples += 1
        }
        guard tokens > 0, examples > 0 else { return nil }
        return Result(
            meanCrossEntropyNats: sum / Double(tokens),
            exampleCount: examples,
            supervisedTokenCount: tokens,
            skippedExampleCount: skippedExampleCount
        )
    }
}
