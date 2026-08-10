import Foundation

/// Hyperparameters for a training run.
///
/// Pure configuration — no user data, no MLX types. Safe to log and sync.
public struct TrainConfig: Sendable, Codable, Hashable {
    /// Learning rate for AdamW.
    public var learningRate: Float
    /// AdamW β₁ / β₂.
    public var betas: (Float, Float)
    /// AdamW ε.
    public var eps: Float
    /// Decoupled weight decay.
    public var weightDecay: Float
    /// Whether to apply bias correction to moments (standard AdamW: true).
    public var biasCorrection: Bool
    /// Micro-batch size (examples per forward/backward).
    public var batchSize: Int
    /// Number of micro-batches accumulated before an optimizer step.
    public var gradientAccumulationSteps: Int
    /// Write a registry candidate every N optimizer steps.
    public var checkpointEvery: Int
    /// Seed for the batch-order generator (SplitMix64).
    public var seed: UInt64
    /// Max token length of prompt+completion; longer examples are skipped.
    public var maxSequenceLength: Int

    /// Creates training hyperparameters with production-friendly defaults.
    public init(
        learningRate: Float = 1e-5,
        betas: (Float, Float) = (0.9, 0.999),
        eps: Float = 1e-8,
        weightDecay: Float = 0.01,
        biasCorrection: Bool = true,
        batchSize: Int = 4,
        gradientAccumulationSteps: Int = 1,
        checkpointEvery: Int = 25,
        seed: UInt64 = 42,
        maxSequenceLength: Int = 2048
    ) {
        self.learningRate = learningRate
        self.betas = betas
        self.eps = eps
        self.weightDecay = weightDecay
        self.biasCorrection = biasCorrection
        self.batchSize = batchSize
        self.gradientAccumulationSteps = gradientAccumulationSteps
        self.checkpointEvery = checkpointEvery
        self.seed = seed
        self.maxSequenceLength = maxSequenceLength
    }

    // Manual Codable — tuples are not Codable.
    private enum CodingKeys: String, CodingKey {
        case learningRate, beta1, beta2, eps, weightDecay, biasCorrection
        case batchSize, gradientAccumulationSteps, checkpointEvery, seed, maxSequenceLength
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        learningRate = try c.decode(Float.self, forKey: .learningRate)
        let b1 = try c.decode(Float.self, forKey: .beta1)
        let b2 = try c.decode(Float.self, forKey: .beta2)
        betas = (b1, b2)
        eps = try c.decode(Float.self, forKey: .eps)
        weightDecay = try c.decode(Float.self, forKey: .weightDecay)
        biasCorrection = try c.decode(Bool.self, forKey: .biasCorrection)
        batchSize = try c.decode(Int.self, forKey: .batchSize)
        gradientAccumulationSteps = try c.decode(Int.self, forKey: .gradientAccumulationSteps)
        checkpointEvery = try c.decode(Int.self, forKey: .checkpointEvery)
        seed = try c.decode(UInt64.self, forKey: .seed)
        maxSequenceLength = try c.decode(Int.self, forKey: .maxSequenceLength)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(learningRate, forKey: .learningRate)
        try c.encode(betas.0, forKey: .beta1)
        try c.encode(betas.1, forKey: .beta2)
        try c.encode(eps, forKey: .eps)
        try c.encode(weightDecay, forKey: .weightDecay)
        try c.encode(biasCorrection, forKey: .biasCorrection)
        try c.encode(batchSize, forKey: .batchSize)
        try c.encode(gradientAccumulationSteps, forKey: .gradientAccumulationSteps)
        try c.encode(checkpointEvery, forKey: .checkpointEvery)
        try c.encode(seed, forKey: .seed)
        try c.encode(maxSequenceLength, forKey: .maxSequenceLength)
    }

    public static func == (lhs: TrainConfig, rhs: TrainConfig) -> Bool {
        lhs.learningRate == rhs.learningRate
            && lhs.betas.0 == rhs.betas.0 && lhs.betas.1 == rhs.betas.1
            && lhs.eps == rhs.eps
            && lhs.weightDecay == rhs.weightDecay
            && lhs.biasCorrection == rhs.biasCorrection
            && lhs.batchSize == rhs.batchSize
            && lhs.gradientAccumulationSteps == rhs.gradientAccumulationSteps
            && lhs.checkpointEvery == rhs.checkpointEvery
            && lhs.seed == rhs.seed
            && lhs.maxSequenceLength == rhs.maxSequenceLength
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(learningRate)
        hasher.combine(betas.0)
        hasher.combine(betas.1)
        hasher.combine(eps)
        hasher.combine(weightDecay)
        hasher.combine(biasCorrection)
        hasher.combine(batchSize)
        hasher.combine(gradientAccumulationSteps)
        hasher.combine(checkpointEvery)
        hasher.combine(seed)
        hasher.combine(maxSequenceLength)
    }
}
