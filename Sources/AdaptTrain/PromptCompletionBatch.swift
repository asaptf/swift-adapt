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

    /// Creates a tokenized example.
    public init(tokens: [Int], promptTokenCount: Int, weight: Float) {
        self.tokens = tokens
        self.promptTokenCount = promptTokenCount
        self.weight = weight
    }
}

/// Prompt/completion formatting and batching.
///
/// ## Masking convention (honored by M3 held-out perplexity)
///
/// 1. Encode `prompt` and `completion` **separately** with the tokenizer
///    (`encode(text:addSpecialTokens:)` — specials on the prompt only by default
///    via `encode(text:)`; completion uses `addSpecialTokens: false` so BOS is
///    not injected mid-sequence).
/// 2. Concatenate: `tokens = promptTokens + completionTokens`.
/// 3. Teacher-forcing: `inputs = tokens[..<n-1]`, `targets = tokens[1...]`.
/// 4. **Loss mask:** a target at sequence position `i` (predicting `tokens[i+1]`)
///    contributes only when `i + 1 >= promptTokenCount` — i.e. only completion
///    tokens are supervised. The first completion token is predicted from the
///    last prompt token, which is intentional.
/// 5. Padding positions are also masked (length mask), matching `LoRATrain.loss`.
/// 6. **Example weights:** each example's CE is multiplied by `TrainingExample.weight`
///    before the batch mean, so `SignalSource` importance is honored in the objective.
public enum PromptCompletionBatch {
    /// Tokenizes a training example; returns `nil` if empty or over `maxLength`.
    public static func tokenize(
        _ example: TrainingExample,
        tokenizer: any Tokenizer,
        maxLength: Int
    ) -> TokenizedExample? {
        let promptIDs = tokenizer.encode(text: example.prompt, addSpecialTokens: true)
        let completionIDs = tokenizer.encode(text: example.completion, addSpecialTokens: false)
        guard !completionIDs.isEmpty else { return nil }
        let tokens = promptIDs + completionIDs
        guard tokens.count >= 2, tokens.count <= maxLength else { return nil }
        return TokenizedExample(
            tokens: tokens,
            promptTokenCount: promptIDs.count,
            weight: Float(example.weight)
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
