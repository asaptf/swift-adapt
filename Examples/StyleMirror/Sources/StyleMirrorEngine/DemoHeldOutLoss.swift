import AdaptCore
import AdaptTrain
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Token-weighted mean CE aggregation + live MLX measurement (demo target only).
///
/// Mirrors `adapt-cli measure` arithmetic so numbers are comparable with the
/// seeded registry's recorded `EvalReport`s. **Measurement only** — never a
/// promotion decision by itself (see ``ProvisionalPromotionGate``).
enum DemoHeldOutLoss {
    struct Result: Sendable, Equatable {
        var meanCrossEntropyNats: Double
        var exampleCount: Int
        var supervisedTokenCount: Int
        var skippedExampleCount: Int

        var evalReport: EvalReport {
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

    struct ExampleContribution: Sendable, Equatable {
        var crossEntropySum: Double
        var supervisedTokens: Int
    }

    static func aggregate(
        _ contributions: [ExampleContribution],
        skippedExampleCount: Int = 0
    ) -> Result? {
        var sum = 0.0
        var tokens = 0
        var examples = 0
        for c in contributions {
            guard c.supervisedTokens > 0, c.crossEntropySum.isFinite else { continue }
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

    /// Scores `examples` under a live module (same mask as ``Trainer/llmCompletionLoss``).
    static func measure(
        model: Module,
        tokenizer: any Tokenizer,
        examples: [TrainingExample],
        maxSequenceLength: Int = 512,
        convention: PromptFormatConvention
    ) throws -> Result {
        var contributions: [ExampleContribution] = []
        contributions.reserveCapacity(examples.count)
        var skipped = 0

        for example in examples {
            guard
                let tokenized = PromptCompletionBatch.tokenize(
                    example,
                    tokenizer: tokenizer,
                    maxLength: maxSequenceLength,
                    convention: convention
                ),
                let collated = PromptCompletionBatch.collate([tokenized])
            else {
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
            contributions.append(
                ExampleContribution(
                    crossEntropySum: meanCE * Double(nTok),
                    supervisedTokens: nTok
                )
            )
        }

        guard
            let result = aggregate(contributions, skippedExampleCount: skipped)
        else {
            throw StyleMirrorError.invalidState(
                "held-out measurement produced no supervised tokens (examples=\(examples.count), skipped=\(skipped))"
            )
        }
        return result
    }
}
