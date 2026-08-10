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
    /// Micro-batches are weighted by their supervised-token (or example) counts so
    /// the accumulated gradient matches the equivalent full-batch mean. Gradients
    /// are materialized (`eval`) after each successful micro-batch so the lazy
    /// MLX graph does not retain every micro-batch's activations.
    ///
    /// Unusable micro-batches (builder returns `nil`) are skipped and drawing
    /// continues until `gradientAccumulationSteps` successful micros are filled
    /// or a full epoch of consecutive skips proves no usable data remains.
    ///
    /// - Returns: `(stepLoss, tokensInStep)` or `nil` if no usable micro-batch
    ///   could be formed after exhausting a full pass over the dataset.
    public func stepOnce(
        microbatches: ([Int]) throws -> [MLXArray]?
    ) throws -> (loss: Float, tokens: Int)? {
        let accum = max(config.gradientAccumulationSteps, 1)
        var accumGrads: [String: MLXArray] = [:]
        var weightedLossSum: Float = 0
        var stepTokenCount: Int = 0
        var microsDone = 0

        let datasetCount = batchIterator.datasetCount
        let batchSize = max(batchIterator.batchSize, 1)
        let drawsPerEpoch = max(1, (datasetCount + batchSize - 1) / batchSize)
        var consecutiveSkips = 0

        // Keep drawing until we fill `accum` successful micros, or a full epoch
        // of skips shows the corpus has no usable examples right now.
        while microsDone < accum {
            guard let indices = batchIterator.nextBatchIndices() else {
                break
            }
            guard let arrays = try microbatches(indices), !arrays.isEmpty else {
                consecutiveSkips += 1
                if consecutiveSkips >= drawsPerEpoch {
                    break
                }
                continue
            }

            consecutiveSkips = 0

            let (result, grads) = lossAndGrad(model, arrays)
            let lvalue = result[0]
            let count = result[1]
            eval(lvalue, count)

            let n = count.item(Int.self)
            guard n > 0 else {
                consecutiveSkips += 1
                if consecutiveSkips >= drawsPerEpoch {
                    break
                }
                continue
            }

            let lossValue = lvalue.item(Float.self)
            weightedLossSum += lossValue * Float(n)
            stepTokenCount += n

            // Weight by token count; divide by total after the loop so unequal
            // micro-batches match the full-batch mean gradient.
            for (key, g) in grads.flattened() {
                let weighted = g * Float(n)
                if let existing = accumGrads[key] {
                    accumGrads[key] = existing + weighted
                } else {
                    accumGrads[key] = weighted
                }
            }
            // Materialize after each micro-batch so the running sum does not
            // retain the full computation graph / activations of prior micros
            // (the reason gradient accumulation exists on 6 GB devices).
            eval(Array(accumGrads.values))
            microsDone += 1
        }

        guard microsDone > 0, stepTokenCount > 0 else { return nil }

        let invTotal = 1.0 / Float(stepTokenCount)
        for (key, g) in accumGrads {
            accumGrads[key] = g * invTotal
        }
        eval(Array(accumGrads.values))

        let gradTree = ModuleParameters.unflattened(accumGrads)
        optimizer.update(model: model, gradients: gradTree)
        eval(model)

        let meanLoss = weightedLossSum / Float(stepTokenCount)
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
