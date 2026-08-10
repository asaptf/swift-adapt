import AdaptCore
import AdaptTrain
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Runs held-out mean cross-entropy on a live MLX module (CLI measurement path).
///
/// Uses the same tokenization, mask, and CE as ``Trainer/llmCompletionLoss`` so
/// train and measure numbers are comparable. Does **not** implement a promotion
/// gate, early stopping, or any decision procedure (architecture §4.5 / M3).
public enum HeldOutLossRunner {
    /// Scores `examples` under `model` + `tokenizer`.
    ///
    /// - Parameters:
    ///   - model: Module with LoRA already applied (or base-only).
    ///   - tokenizer: Same tokenizer used for training.
    ///   - examples: Held-out set (never trained on for a fair measurement).
    ///   - maxSequenceLength: Drop examples that tokenize longer than this.
    ///   - convention: Prompt format; pass the convention used when training.
    public static func measure(
        model: Module,
        tokenizer: any Tokenizer,
        examples: [TrainingExample],
        maxSequenceLength: Int = 512,
        convention: PromptFormatConvention
    ) throws -> HeldOutLoss.Result {
        var contributions: [HeldOutLoss.ExampleContribution] = []
        contributions.reserveCapacity(examples.count)
        var skipped = 0

        for example in examples {
            guard
                let tokenized = PromptCompletionBatch.tokenize(
                    example,
                    tokenizer: tokenizer,
                    maxLength: maxSequenceLength,
                    convention: convention
                )
            else {
                skipped += 1
                continue
            }
            guard let collated = PromptCompletionBatch.collate([tokenized]) else {
                skipped += 1
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
            guard nTok > 0, meanCE.isFinite else {
                skipped += 1
                continue
            }
            // llmCompletionLoss returns the **mean** over supervised tokens;
            // convert to a sum so aggregate stays token-weighted.
            contributions.append(
                HeldOutLoss.ExampleContribution(
                    crossEntropySum: meanCE * Double(nTok),
                    supervisedTokens: nTok
                )
            )
        }

        guard
            let result = HeldOutLoss.aggregate(
                contributions,
                skippedExampleCount: skipped
            )
        else {
            throw AdaptCLIError.invalidArgument(
                "held-out measurement produced no supervised tokens (examples=\(examples.count), skipped=\(skipped))"
            )
        }
        return result
    }
}
