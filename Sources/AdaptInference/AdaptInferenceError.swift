import AdaptCore
import Foundation

/// Errors originating from AdaptInference operations.
///
/// One distinctly named enum per module (architecture §8). Contains no
/// test-only cases.
public enum AdaptInferenceError: Error, LocalizedError, Sendable, Equatable {
    /// Caller supplied an unusable configuration (e.g. remote model without a downloader).
    case invalidArgument(String)
    /// Base model or tokenizer could not be loaded.
    case modelLoadFailed(String)
    /// Active adapter weights failed SHA-256 verification against registry metadata.
    case integrityMismatch(expected: String, actual: String)
    /// LoRA adapter could not be applied or unloaded.
    case adapterLoadFailed(String)
    /// Token generation failed.
    case generationFailed(String)
    /// Session was fused; adapter hot-swap is no longer possible.
    case fusedImmutable(String)
    /// Adapter was trained under a different prompt-format convention than this session.
    ///
    /// Serving a raw-trained adapter through a chat template (or the reverse)
    /// produces subtly wrong output with no other failure signal — refuse loudly.
    case promptFormatMismatch(adapter: PromptFormatConvention, session: PromptFormatConvention)

    public var errorDescription: String? {
        switch self {
        case .invalidArgument(let message):
            return "Invalid argument: \(message)"
        case .modelLoadFailed(let message):
            return "Model load failed: \(message)"
        case .integrityMismatch(let expected, let actual):
            return "Adapter integrity mismatch: expected \(expected), got \(actual)"
        case .adapterLoadFailed(let message):
            return "Adapter load failed: \(message)"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        case .fusedImmutable(let message):
            return "Fused session is immutable: \(message)"
        case .promptFormatMismatch(let adapter, let session):
            return
                "Prompt format mismatch: adapter was trained with \(adapter.rawValue), "
                + "session uses \(session.rawValue)"
        }
    }
}
