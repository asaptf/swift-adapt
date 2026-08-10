import AdaptCore
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Tokenized example with a prompt-length boundary for loss masking.
public struct TokenizedExample: Sendable, Hashable {
    /// Full sequence: prompt tokens then completion tokens.
    public let tokens: [Int]
    /// Number of leading prompt tokens (loss is masked on these as targets).
    public let promptTokenCount: Int
    /// Importance weight from `TrainingExample.weight`.
    public let weight: Float
    /// Formatting convention used to produce `tokens`.
    public let convention: PromptFormatConvention

    /// Creates a tokenized example.
    public init(
        tokens: [Int],
        promptTokenCount: Int,
        weight: Float,
        convention: PromptFormatConvention = .rawConcatenation
    ) {
        self.tokens = tokens
        self.promptTokenCount = promptTokenCount
        self.weight = weight
        self.convention = convention
    }
}

/// Prompt/completion formatting and batching.
///
/// ## Masking convention (honored by M3 held-out perplexity)
///
/// Formatting is owned by ``SFTPromptFormatter`` (AdaptCore) so train and
/// generate share one code path:
///
/// 1. **Chat template** (when the tokenizer has one): apply the template over
///    user + assistant turns; `promptTokenCount` is the generation-prefix
///    length (user turn + start-of-assistant marker). Scaffold tokens are not
///    supervised.
/// 2. **Raw fallback** (no template): `encode(prompt) + encode(completion)` —
///    same rule train and generate use.
/// 3. Teacher-forcing: `inputs = tokens[..<n-1]`, `targets = tokens[1...]`.
/// 4. **Loss mask:** a target at sequence position `i` (predicting `tokens[i+1]`)
///    contributes only when `i + 1 >= promptTokenCount` — i.e. only assistant /
///    completion tokens are supervised.
/// 5. Padding positions are also masked (length mask), matching `LoRATrain.loss`.
/// 6. **Example weights:** each example's CE is multiplied by `TrainingExample.weight`
///    before the batch mean, so `SignalSource` importance is honored in the objective.
public enum PromptCompletionBatch {
    /// Wraps an `MLXLMCommon.Tokenizer` for ``SFTPromptFormatter``.
    public static func sftTokenizer(_ tokenizer: any Tokenizer) -> AnySFTTokenizer {
        AnySFTTokenizer(
            encode: { text, addSpecial in
                tokenizer.encode(text: text, addSpecialTokens: addSpecial)
            },
            applyChatTemplate: { messages, addGenerationPrompt in
                do {
                    let sendable: [[String: any Sendable]] = messages.map { dict in
                        dict.mapValues { $0 as any Sendable }
                    }
                    return try tokenizer.applyChatTemplate(
                        messages: sendable,
                        tools: nil,
                        additionalContext: ["add_generation_prompt": addGenerationPrompt]
                    )
                } catch TokenizerError.missingChatTemplate {
                    throw SFTFormattingError.missingChatTemplate
                }
            }
        )
    }

    /// Tokenizes a training example; returns `nil` if empty or over `maxLength`.
    public static func tokenize(
        _ example: TrainingExample,
        tokenizer: any Tokenizer,
        maxLength: Int,
        convention: PromptFormatConvention
    ) -> TokenizedExample? {
        let sft = sftTokenizer(tokenizer)
        return tokenize(example, sftTokenizer: sft, maxLength: maxLength, convention: convention)
    }

    /// Tokenizes via an ``SFTTokenizing`` (shared with inference tests).
    public static func tokenize(
        _ example: TrainingExample,
        sftTokenizer: some SFTTokenizing,
        maxLength: Int,
        convention: PromptFormatConvention
    ) -> TokenizedExample? {
        let formatted: SFTTrainingTokens
        do {
            formatted = try SFTPromptFormatter.formatTraining(
                prompt: example.prompt,
                completion: example.completion,
                tokenizer: sftTokenizer,
                convention: convention
            )
        } catch {
            return nil
        }
        guard formatted.tokens.count >= 2, formatted.tokens.count <= maxLength else {
            return nil
        }
        return TokenizedExample(
            tokens: formatted.tokens,
            promptTokenCount: formatted.promptTokenCount,
            weight: Float(example.weight),
            convention: formatted.convention
        )
    }

    /// Builds padded `(inputs, targets, lengths, tokenWeights)` for a micro-batch.
    ///
    /// - `tokenWeights` is the per-token loss multiplier: 0 for pad/prompt targets,
    ///   `example.weight` for completion targets. Callers can use it with a custom
    ///   loss or fold it into lengths-style masking.
    public static func collate(
        _ examples: [TokenizedExample]
    ) -> (inputs: MLXArray, targets: MLXArray, lengths: MLXArray, tokenWeights: MLXArray)? {
        guard !examples.isEmpty else { return nil }

        // Shifted sequences: inputs[:-1], targets[1:].
        let sequences: [(input: [Int32], target: [Int32], promptCount: Int, weight: Float)] =
            examples.map { ex in
                let ids = ex.tokens.map { Int32($0) }
                return (
                    input: Array(ids.dropLast()),
                    target: Array(ids.dropFirst()),
                    promptCount: ex.promptTokenCount,
                    weight: ex.weight
                )
            }

        let lengths = sequences.map { $0.input.count }
        let maxLen = lengths.max() ?? 0
        guard maxLen > 0 else { return nil }

        let batchSize = sequences.count
        var inputData = [Int32](repeating: 0, count: batchSize * maxLen)
        var targetData = [Int32](repeating: 0, count: batchSize * maxLen)
        var weightData = [Float](repeating: 0, count: batchSize * maxLen)

        for (row, seq) in sequences.enumerated() {
            let len = seq.input.count
            for col in 0..<len {
                inputData[row * maxLen + col] = seq.input[col]
                targetData[row * maxLen + col] = seq.target[col]
                // Target at col predicts tokens[col+1]. Supervise completion only.
                let predictedIndex = col + 1
                if predictedIndex >= seq.promptCount {
                    weightData[row * maxLen + col] = seq.weight
                }
            }
        }

        let inputs = MLXArray(inputData, [batchSize, maxLen])
        let targets = MLXArray(targetData, [batchSize, maxLen])
        let lengthsArr = MLXArray(lengths.map { Int32($0) })
        let weights = MLXArray(weightData, [batchSize, maxLen])
        return (inputs, targets, lengthsArr, weights)
    }

    /// Weighted cross-entropy loss over completion tokens only.
    ///
    /// Compatible with models that expose logits via a call matching
    /// `LoRATrain.loss` expectations. Used for LLM training; synthetic tests
    /// use a simpler MSE path instead.
    public static func weightedCompletionLoss(
        logits: MLXArray,
        targets: MLXArray,
        lengths: MLXArray,
        tokenWeights: MLXArray
    ) -> (loss: MLXArray, tokenCount: MLXArray) {
        // length mask: positions < length
        let seqLen = logits.dim(1)
        let lengthMask =
            MLXArray(0..<seqLen)[.newAxis, 0...] .< lengths[0..., .newAxis]
        let mask = lengthMask * tokenWeights
        let ntoks = mask.sum()
        // Avoid div-by-zero if a batch has no supervised tokens.
        let denom = MLX.maximum(ntoks, MLXArray(Float(1e-8)))
        let ce = (crossEntropy(logits: logits.asType(.float32), targets: targets) * mask).sum()
            / denom
        return (ce, ntoks)
    }
}
