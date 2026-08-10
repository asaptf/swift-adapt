import Foundation

/// Errors raised by the StyleMirror demo engine seam.
public enum StyleMirrorError: Error, LocalizedError, Sendable, Equatable {
    /// Requested corpus item or round was not found.
    case notFound(String)
    /// Operation is invalid in the current engine state.
    case invalidState(String)
    /// Caller supplied a value the engine cannot accept.
    case invalidArgument(String)
    /// A blind-test guess referenced an unknown candidate or round.
    case unknownCandidate(String)
    /// Real engine cannot open the adapter registry (missing path, I/O).
    case registryUnavailable(String)
    /// Real engine cannot load the base model or tokenizer.
    case modelUnavailable(String)
    /// Generation failed (session, template, or runtime).
    case generationFailed(String)
    /// A generated reply landed far outside the shared length class.
    ///
    /// Deliberately **not** silently trimmed: a truncated reply would lie about
    /// what the model produced. The UI should surface this rather than show a
    /// fair three-way card layout.
    case lengthClassMismatch(role: ReplyRole, wordCount: Int, characterCount: Int)

    public var errorDescription: String? {
        switch self {
        case .notFound(let message):
            return "Not found: \(message)"
        case .invalidState(let message):
            return "Invalid state: \(message)"
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        case .unknownCandidate(let message):
            return "Unknown candidate: \(message)"
        case .registryUnavailable(let message):
            return "Registry unavailable: \(message)"
        case .modelUnavailable(let message):
            return "Model unavailable: \(message)"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        case .lengthClassMismatch(let role, let wordCount, let characterCount):
            return """
                Length class mismatch for \(role.rawValue): \
                \(wordCount) words / \(characterCount) chars \
                (expected \(BlindReplyPrompt.minWords)–\(BlindReplyPrompt.maxWords) words). \
                Reply was not trimmed.
                """
        }
    }
}
