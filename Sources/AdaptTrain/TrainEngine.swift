import AdaptCore
import Foundation
import MLX
import MLXNN

/// Loss function for one micro-batch: returns `(scalarLoss, tokenOrExampleCount)`.
///
/// Kept as a plain function over `Module` + arrays so tests can plug MSE on a
/// tiny synthetic stack without involving `LLMModel`.
public typealias MicrobatchLoss =
    (Module, [MLXArray]) -> (loss: MLXArray, count: MLXArray)

/// Mutable MLX session for the training loop.
///
/// **Not Sendable.** Must live entirely inside one `Trainer` actor isolation
/// (or a single-threaded test). Never pass `MLXArray` or this type across
/// `async` boundaries without `eval` + converting to Sendable snapshots.
///
/// The step API is **synchronous** so the owning actor can interleave
/// `await` checkpoint I/O between steps without sending the engine across
/// isolation domains.
public final class TrainEngine {
    /// Trainable module (LoRA-injected LLM or synthetic test module).
    public let model: Module
    /// Checkpointable AdamW.
    public let optimizer: CheckpointableAdamW
    /// Hyperparameters.
    public let config: TrainConfig
    /// Seeded batch cursor.
    public private(set) var batchIterator: SeededBatchIterator
    /// Lifetime optimizer steps completed.
    public private(set) var lifetimeSteps: Int
    /// Lifetime loss curve.
    public private(set) var lossHistory: [Float]
    /// Lifetime supervised token/example count.
    public private(set) var tokensProcessed: Int
    /// Parent registry version when this engine was restored, if any.
    public private(set) var parentVersion: Int?

    private let lossAndGrad: (Module, [MLXArray]) -> ([MLXArray], ModuleParameters)

    /// Creates a fresh engine (no resume).
    public init(
        model: Module,
        config: TrainConfig,
        datasetCount: Int,
        loss: @escaping MicrobatchLoss
    ) {
        self.model = model
        self.config = config
        self.optimizer = CheckpointableAdamW(
            learningRate: config.learningRate,
            betas: config.betas,
            eps: config.eps,
            weightDecay: config.weightDecay,
            biasCorrection: config.biasCorrection
        )
        self.batchIterator = SeededBatchIterator(
            datasetCount: datasetCount,
            batchSize: config.batchSize,
            generator: SeededGenerator(seed: config.seed),
            infinite: true
        )
        self.lifetimeSteps = 0
        self.lossHistory = []
        self.tokensProcessed = 0
        self.parentVersion = nil
        self.lossAndGrad = valueAndGrad(model: model) {
            (model: Module, arrays: [MLXArray]) -> [MLXArray] in
            let (l, c) = loss(model, arrays)
            return [l, c]
        }
        eval(model)
    }

    /// Restores engine state from a checkpoint (weights must already be on `model`).
    public func restore(
        state: TrainStateFile,
        moments: [String: MLXArray],
        parentVersion: Int
    ) {
        optimizer.importMoments(moments)
        optimizer.restoreStep(state.optimizerStep)
        batchIterator.restore(cursor: state.cursor)
        lifetimeSteps = state.step
        lossHistory = state.lossHistory
        tokensProcessed = state.tokensProcessed
        self.parentVersion = parentVersion
        eval(model)
        eval(Array(optimizer.exportMoments().values))
    }

    /// Performs one optimizer step with gradient accumulation.
    ///
    /// - Returns: `(stepLoss, tokensInStep)` or `nil` if no usable micro-batch
    ///   could be formed (empty data after filtering).
    public func stepOnce(
        microbatches: ([Int]) throws -> [MLXArray]?
    ) throws -> (loss: Float, tokens: Int)? {
        let accum = max(config.gradientAccumulationSteps, 1)
        var accumGrads: [String: MLXArray] = [:]
        var stepLossSum: Float = 0
        var stepTokenCount: Int = 0
        var microsDone = 0

        for _ in 0..<accum {
            guard let indices = batchIterator.nextBatchIndices() else {
                break
            }
            guard let arrays = try microbatches(indices), !arrays.isEmpty else {
                continue
            }

            let (result, grads) = lossAndGrad(model, arrays)
            let lvalue = result[0]
            let count = result[1]
            eval(lvalue, count)

            stepLossSum += lvalue.item(Float.self)
            stepTokenCount += count.item(Int.self)

            let scale = 1.0 / Float(accum)
            for (key, g) in grads.flattened() {
                let scaled = g * scale
                if let existing = accumGrads[key] {
                    accumGrads[key] = existing + scaled
                } else {
                    accumGrads[key] = scaled
                }
            }
            microsDone += 1
        }

        guard microsDone > 0 else { return nil }

        eval(Array(accumGrads.values))
        let gradTree = ModuleParameters.unflattened(accumGrads)
        optimizer.update(model: model, gradients: gradTree)
        eval(model)

        let meanLoss = stepLossSum / Float(microsDone)
        lifetimeSteps += 1
        lossHistory.append(meanLoss)
        tokensProcessed += stepTokenCount
        return (meanLoss, stepTokenCount)
    }

    /// Builds a `TrainStateFile` snapshot for the current engine position.
    public func makeStateFile(parentVersion: Int?) -> TrainStateFile {
        TrainStateFile(
            step: lifetimeSteps,
            optimizerStep: optimizer.step,
            seed: config.seed,
            cursor: batchIterator.cursor,
            lossHistory: lossHistory,
            tokensProcessed: tokensProcessed,
            parentVersion: parentVersion ?? self.parentVersion,
            config: config
        )
    }

    /// Records that `version` was just written so the next checkpoint chains from it.
    public func noteCheckpointed(version: Int) {
        parentVersion = version
    }
}
