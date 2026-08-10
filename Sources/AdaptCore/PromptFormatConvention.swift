import Foundation

/// How prompt/completion text is mapped to token ids for SFT train and generate.
///
/// **Train and generate must use the same convention** for a given adapter.
/// Divergence trains an adapter that inference then conditions wrongly — the
/// worst silent failure mode. The convention is recorded on
/// ``AdapterVersion/promptFormat`` so a session can refuse a mismatch.
public enum PromptFormatConvention: String, Codable, Sendable, Hashable, CaseIterable {
    /// Tokenizer chat template over a user turn (+ assistant for training).
    /// Loss is masked to the assistant span only.
    case chatTemplate

    /// Legacy raw concatenation: `encode(prompt) + encode(completion)` with no
    /// chat scaffolding. Used when the tokenizer has no chat template.
    case rawConcatenation
}

/// Errors from ``SFTPromptFormatter`` / ``SFTTokenizing``.
public enum SFTFormattingError: Error, LocalizedError, Sendable, Equatable {
    /// Tokenizer does not provide a chat template.
    case missingChatTemplate
    /// Encoding produced no tokens (or an empty supervised span).
    case emptyEncoding(String)

    public var errorDescription: String? {
        switch self {
        case .missingChatTemplate:
            return "Tokenizer has no chat template"
        case .emptyEncoding(let message):
            return "Empty encoding: \(message)"
        }
    }
}

/// Minimal tokenizer surface for shared train/generate formatting.
///
/// Deliberately independent of `MLXLMCommon.Tokenizer` so ``AdaptCore`` stays
/// MLX-free. Production code wraps the real tokenizer; tests use a stub.
public protocol SFTTokenizing: Sendable {
    /// Encode text to token ids.
    func encode(text: String, addSpecialTokens: Bool) -> [Int]

    /// Apply the model chat template to `messages`.
    ///
    /// - Parameter addGenerationPrompt: When `true`, append the tokens that open
    ///   an assistant turn (inference prefix / training prompt boundary). When
    ///   `false`, emit only the provided messages (full train sequence with an
    ///   assistant reply already present).
    /// - Throws: ``SFTFormattingError/missingChatTemplate`` when the tokenizer
    ///   has no template.
    func applyChatTemplate(
        messages: [[String: String]],
        addGenerationPrompt: Bool
    ) throws -> [Int]
}

/// Type-erased ``SFTTokenizing`` so call sites can wrap any tokenizer once.
public struct AnySFTTokenizer: SFTTokenizing, Sendable {
    private let _encode: @Sendable (String, Bool) -> [Int]
    private let _applyChatTemplate: @Sendable ([[String: String]], Bool) throws -> [Int]

    /// Creates a type-erased tokenizer.
    public init(
        encode: @escaping @Sendable (String, Bool) -> [Int],
        applyChatTemplate: @escaping @Sendable ([[String: String]], Bool) throws -> [Int]
    ) {
        self._encode = encode
        self._applyChatTemplate = applyChatTemplate
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        _encode(text, addSpecialTokens)
    }

    public func applyChatTemplate(
        messages: [[String: String]],
        addGenerationPrompt: Bool
    ) throws -> [Int] {
        try _applyChatTemplate(messages, addGenerationPrompt)
    }
}

/// Tokenized training sequence with a supervised-span boundary.
public struct SFTTrainingTokens: Sendable, Hashable {
    /// Full sequence: prompt (templated or raw) then completion tokens.
    public let tokens: [Int]
    /// Leading token count that is **not** supervised (prompt + scaffold).
    public let promptTokenCount: Int
    /// Convention used to produce `tokens`.
    public let convention: PromptFormatConvention

    /// Creates a training token sequence.
    public init(tokens: [Int], promptTokenCount: Int, convention: PromptFormatConvention) {
        self.tokens = tokens
        self.promptTokenCount = promptTokenCount
        self.convention = convention
    }
}

/// **Single source of truth** for SFT prompt formatting on train and generate paths.
///
/// Both ``AdaptTrain`` and ``AdaptInference`` call this type so the two sides
/// cannot silently disagree about raw-vs-template encoding.
public enum SFTPromptFormatter {
    /// Probes the tokenizer for a chat template; falls back to raw concatenation.
    public static func detectConvention(tokenizer: some SFTTokenizing) -> PromptFormatConvention {
        do {
            _ = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": "ping"]],
                addGenerationPrompt: true
            )
            return .chatTemplate
        } catch SFTFormattingError.missingChatTemplate {
            return .rawConcatenation
        } catch {
            // Unexpected probe failure: treat as no template so train and generate
            // still share the same fallback rather than diverging.
            return .rawConcatenation
        }
    }

    /// Formats a training example under `convention`.
    ///
    /// - Chat template: full = template(user, assistant); prompt boundary =
    ///   template(user, addGenerationPrompt: true). Loss must cover only
    ///   `tokens[promptTokenCount...]`.
    /// - Raw: `encode(prompt, specials) + encode(completion, no specials)`.
    public static func formatTraining(
        prompt: String,
        completion: String,
        tokenizer: some SFTTokenizing,
        convention: PromptFormatConvention
    ) throws -> SFTTrainingTokens {
        switch convention {
        case .rawConcatenation:
            let promptIDs = tokenizer.encode(text: prompt, addSpecialTokens: true)
            let completionIDs = tokenizer.encode(text: completion, addSpecialTokens: false)
            guard !completionIDs.isEmpty else {
                throw SFTFormattingError.emptyEncoding("completion produced no tokens")
            }
            let tokens = promptIDs + completionIDs
            guard tokens.count >= 2 else {
                throw SFTFormattingError.emptyEncoding("prompt+completion too short")
            }
            return SFTTrainingTokens(
                tokens: tokens,
                promptTokenCount: promptIDs.count,
                convention: .rawConcatenation
            )

        case .chatTemplate:
            let prefix = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": prompt]],
                addGenerationPrompt: true
            )
            let full = try tokenizer.applyChatTemplate(
                messages: [
                    ["role": "user", "content": prompt],
                    ["role": "assistant", "content": completion],
                ],
                addGenerationPrompt: false
            )
            guard !full.isEmpty else {
                throw SFTFormattingError.emptyEncoding("chat template produced no tokens")
            }
            // Prefer the generation-prefix length when it is a true prefix of the
            // full sequence (the HF/Qwen invariant). Otherwise fall back to a
            // conservative boundary: only tokens beyond the user-only template
            // without generation prompt are supervised.
            let promptTokenCount: Int
            if full.count >= prefix.count, Array(full.prefix(prefix.count)) == prefix {
                promptTokenCount = prefix.count
            } else {
                let userOnly = try tokenizer.applyChatTemplate(
                    messages: [["role": "user", "content": prompt]],
                    addGenerationPrompt: false
                )
                if full.count >= userOnly.count, Array(full.prefix(userOnly.count)) == userOnly {
                    promptTokenCount = userOnly.count
                } else {
                    // Last resort: do not supervise anything we cannot bound —
                    // better to skip via caller maxLength than to train on scaffold.
                    promptTokenCount = min(prefix.count, full.count)
                }
            }
            guard full.count > promptTokenCount else {
                throw SFTFormattingError.emptyEncoding(
                    "chat template left no assistant tokens to supervise"
                )
            }
            return SFTTrainingTokens(
                tokens: full,
                promptTokenCount: promptTokenCount,
                convention: .chatTemplate
            )
        }
    }

    /// Formats the generation prefix for `prompt` under `convention`.
    ///
    /// **Invariant:** for the same `prompt` and tokenizer, the returned ids are
    /// byte-identical to `formatTraining(...).tokens[0..<promptTokenCount]`.
    public static func formatGenerationPrefix(
        prompt: String,
        tokenizer: some SFTTokenizing,
        convention: PromptFormatConvention
    ) throws -> [Int] {
        switch convention {
        case .rawConcatenation:
            let ids = tokenizer.encode(text: prompt, addSpecialTokens: true)
            guard !ids.isEmpty else {
                throw SFTFormattingError.emptyEncoding("prompt produced no tokens")
            }
            return ids

        case .chatTemplate:
            let ids = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": prompt]],
                addGenerationPrompt: true
            )
            guard !ids.isEmpty else {
                throw SFTFormattingError.emptyEncoding("chat template prefix was empty")
            }
            return ids
        }
    }

    /// Resolves the convention recorded on adapter metadata.
    ///
    /// Legacy `version.json` without the field is treated as raw concatenation
    /// (the historical train/generate path).
    public static func convention(fromStored stored: PromptFormatConvention?) -> PromptFormatConvention {
        stored ?? .rawConcatenation
    }
}
