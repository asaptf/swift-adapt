import AdaptCore
import AdaptEval
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// MLX implementation of ``PerExampleScorer`` using the same completion-masked
/// CE as ``Trainer/llmCompletionLoss``.
///
/// ## Why scoring lives here
///
/// Architecture §7 confines mlx imports to AdaptTrain / AdaptInference.
/// AdaptEval owns the **decision procedure** (pure Swift: Wilcoxon, pin,
/// policy) and depends only on AdaptCore. The forward pass reuses training's
/// tokenization and loss so held-out nats are comparable to train loss.
///
/// The CLI (and later AdaptSchedule) construct this scorer around a loaded
/// model + adapter; unit tests never touch it.
/// Unchecked: MLX `Module` / tokenizer are used sequentially on the CLI path
/// (same discipline as `HeldOutLossRunner`); not shared across concurrent tasks.
public struct MLXPerExampleCrossEntropyScorer: PerExampleScorer, @unchecked Sendable {
    private let model: Module
    private let tokenizer: any Tokenizer
    private let maxSequenceLength: Int
    private let convention: PromptFormatConvention

    /// - Parameters:
    ///   - model: Module with the adapter under test already applied (or base).
    ///   - tokenizer: Same tokenizer used for training.
    ///   - maxSequenceLength: Drop examples that tokenize longer than this.
    ///   - convention: Prompt format convention used when the adapter was trained.
    public init(
        model: Module,
        tokenizer: any Tokenizer,
        maxSequenceLength: Int = 512,
        convention: PromptFormatConvention
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.maxSequenceLength = maxSequenceLength
        self.convention = convention
    }

    public func score(_ examples: [TrainingExample]) async throws -> [ExampleScore] {
        var results: [ExampleScore] = []
        results.reserveCapacity(examples.count)

        for example in examples {
            guard
                let tokenized = PromptCompletionBatch.tokenize(
                    example,
                    tokenizer: tokenizer,
                    maxLength: maxSequenceLength,
                    convention: convention
                )
            else {
                continue
            }
            guard let collated = PromptCompletionBatch.collate([tokenized]) else {
                continue
            }

            let (loss, tokenCount) = Trainer.llmCompletionLoss(
                model: model,
                arrays: [
                    collated.inputs, collated.targets, collated.lengths, collated.tokenWeights,
                ]
            )
            eval(loss, tokenCount)
            let meanCE = Double(loss.item(Float.self))
            let nTok = Int(tokenCount.item(Float.self).rounded())
            guard nTok > 0, meanCE.isFinite else { continue }

            results.append(
                ExampleScore(
                    exampleID: example.id,
                    primary: meanCE,
                    supervisedTokenCount: nTok
                )
            )
        }
        return results
    }
}
