import Foundation

/// Parameters for the §4.5 promotion gate.
///
/// Units for regression bounds are **absolute nats of mean per-token
/// cross-entropy** (`maxCrossEntropyRegressionNats`), not perplexity ratios.
public struct PromotionPolicy: Sendable, Codable, Equatable, Hashable {
    /// Absolute minimum held-out examples. Below this the gate **abstains**.
    ///
    /// Default 30. Abstention is not rejection and must not feed §4.6 backoff.
    public var minHeldOut: Int

    /// Target held-out share of the available pool (within the 10–20% band).
    ///
    /// The pin size is `max(minHeldOut, round(heldOutFraction * poolSize))`,
    /// clamped so it never exceeds `maxHeldOutFraction * poolSize`. When that
    /// clamp leaves fewer than `minHeldOut` examples, evaluation abstains.
    public var heldOutFraction: Double

    /// Upper bound of the held-out share (architecture: 10–20% band).
    public var maxHeldOutFraction: Double

    /// One-sided Wilcoxon significance level (default 0.05).
    public var alpha: Double

    /// Max allowed regression on **secondary** metrics, in absolute CE nats.
    ///
    /// The primary metric must improve *significantly*; secondaries must not
    /// regress beyond this point-estimate bound.
    public var maxCrossEntropyRegressionNats: Double

    /// Optional secondary metric comparisons (empty in the common CE-only path).
    public var secondaryMetrics: [SecondaryMetricBound]

    /// Creates a promotion policy with §4.5 redesign defaults.
    public init(
        minHeldOut: Int = 30,
        heldOutFraction: Double = 0.15,
        maxHeldOutFraction: Double = 0.20,
        alpha: Double = 0.05,
        maxCrossEntropyRegressionNats: Double = 0.02,
        secondaryMetrics: [SecondaryMetricBound] = []
    ) {
        self.minHeldOut = minHeldOut
        self.heldOutFraction = heldOutFraction
        self.maxHeldOutFraction = maxHeldOutFraction
        self.alpha = alpha
        self.maxCrossEntropyRegressionNats = maxCrossEntropyRegressionNats
        self.secondaryMetrics = secondaryMetrics
    }

    /// Validates policy ranges; throws ``AdaptEvalError/invalidPolicy`` on bad values.
    public func validate() throws {
        guard minHeldOut > 0 else {
            throw AdaptEvalError.invalidPolicy("minHeldOut must be > 0")
        }
        guard heldOutFraction > 0, heldOutFraction <= 1 else {
            throw AdaptEvalError.invalidPolicy("heldOutFraction must be in (0, 1]")
        }
        guard maxHeldOutFraction > 0, maxHeldOutFraction <= 1 else {
            throw AdaptEvalError.invalidPolicy("maxHeldOutFraction must be in (0, 1]")
        }
        guard heldOutFraction <= maxHeldOutFraction else {
            throw AdaptEvalError.invalidPolicy(
                "heldOutFraction (\(heldOutFraction)) must be ≤ maxHeldOutFraction (\(maxHeldOutFraction))"
            )
        }
        guard alpha > 0, alpha < 1 else {
            throw AdaptEvalError.invalidPolicy("alpha must be in (0, 1)")
        }
        guard maxCrossEntropyRegressionNats >= 0, maxCrossEntropyRegressionNats.isFinite else {
            throw AdaptEvalError.invalidPolicy(
                "maxCrossEntropyRegressionNats must be finite and ≥ 0"
            )
        }
    }

    /// Target pin size for a pool of `poolSize` examples.
    ///
    /// Returns the intended count before availability clamping. When the pool
    /// cannot support `minHeldOut` under `maxHeldOutFraction`, returns a value
    /// still below `minHeldOut` so callers can abstain.
    public func targetHeldOutCount(poolSize: Int) -> Int {
        guard poolSize > 0 else { return 0 }
        let byFraction = Int((heldOutFraction * Double(poolSize)).rounded())
        let maxAllowed = max(1, Int((maxHeldOutFraction * Double(poolSize)).rounded(.down)))
        let desired = max(minHeldOut, byFraction)
        return min(desired, maxAllowed, poolSize)
    }
}

/// Bound on a secondary metric used during promotion.
public struct SecondaryMetricBound: Sendable, Codable, Equatable, Hashable {
    /// Stable metric id (must not be the primary metric id).
    public var metric: String
    /// Whether lower scores are better for this metric.
    public var lowerIsBetter: Bool
    /// Maximum allowed regression of the candidate vs incumbent (point estimate).
    ///
    /// For lower-is-better: `candidate - incumbent ≤ maxRegression`.
    /// For higher-is-better: `incumbent - candidate ≤ maxRegression`.
    public var maxRegression: Double

    public init(metric: String, lowerIsBetter: Bool, maxRegression: Double) {
        self.metric = metric
        self.lowerIsBetter = lowerIsBetter
        self.maxRegression = maxRegression
    }
}
