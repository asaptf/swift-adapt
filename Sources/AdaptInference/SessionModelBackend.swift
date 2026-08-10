import AdaptCore
import Foundation

/// Package-internal model/adapter backend used by ``AdaptSession``.
///
/// Production code uses ``MLXSessionBackend`` (over `ModelContainer` +
/// `LoRAContainer`). Tests inject a fake that records load/unload and emits
/// scripted tokens — keeping `swift test` network-free and model-free.
///
/// Not public: the host-facing surface is ``AdaptSession``.
///
/// Conformances are `@unchecked Sendable` (same pattern as
/// ``MLXSessionBackend``): all access is serialized by ``AdaptSession``.
package protocol SessionModelBackend: AnyObject, Sendable {
    /// Stable identity for the loaded base model instance.
    ///
    /// Unchanged across adapter swaps so tests can prove `reload()` does not
    /// re-load the base weights.
    var modelInstanceID: UUID { get }

    /// Whether LoRA layers are currently applied (unfused live layers).
    var hasAdapterLoaded: Bool { get }

    /// Whether adapters have been permanently fused into the base weights.
    var isFused: Bool { get }

    /// Prompt formatting convention detected (or configured) for this backend.
    ///
    /// Production backends probe the tokenizer once; fakes return a configured value.
    var promptFormatConvention: PromptFormatConvention { get }

    /// Apply a registry version directory (`adapter_config.json` +
    /// `adapters.safetensors`) as live LoRA layers.
    func loadAdapter(from directory: URL) async throws

    /// Remove live LoRA layers and restore the base model.
    func unloadAdapter() async throws

    /// Permanently fuse the currently loaded adapter into the base weights.
    ///
    /// After fuse, `loadAdapter` / `unloadAdapter` / further `fuse` must fail.
    func fuseAdapter() async throws

    /// Stream decoded text chunks for `prompt`.
    ///
    /// Must honor cooperative cancellation (`Task.isCancelled`) between chunks
    /// so stopping the consuming task ends generation promptly.
    ///
    /// Encoding of `prompt` must go through ``SFTPromptFormatter`` under
    /// `promptFormatConvention` so it matches training.
    func generate(
        prompt: String,
        options: GenerationOptions
    ) -> AsyncThrowingStream<String, Error>
}
