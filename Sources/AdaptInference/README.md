# AdaptInference

Adapter loading, hot-swap, and streaming generation over `mlx-swift-lm`.

## Contract

- **No network I/O in this module.** Model download and tokenizer file loading go
  through caller-supplied protocol seams (`MLXLMCommon.Downloader`,
  `MLXLMCommon.TokenizerLoader`). Hugging Face packages stay in `adapt-cli` (or
  a host app). A library that *cannot* reach the network is a stronger privacy
  claim than one that promises not to (architecture §3 / §5 App Store story).
- **Auto-load active adapter** for the lineage from `AdapterRegistry`, verifying
  the weights digest before applying. Loading unverified weights is a defect.
- **Zero-config cold start.** No active version → generate on the base model.
  Never error just because a lineage has never trained.
- **Unfused by default.** Live LoRA layers via `LoRAContainer.load(into:)` /
  `unload(from:)`. Swapping does **not** reload the base model.
- **Optional `fuse()`.** Permanently merges the current adapter into the base
  for a throughput mode. After fuse, `reload()` / unload are impossible — create
  a new session to bind a different version. Trade-off: fused ≈ faster matmuls;
  unfused ≈ free hot-swap.
- **MLX stays inside this module** (and `AdaptTrain`). Public API exchanges only
  `Sendable` values. Pattern matches `ModelContainer` / `AdaptTrain.Trainer`
  isolation rather than `@unchecked Sendable` on MLX types.

## Public surface

| Symbol | Role |
|---|---|
| `AdaptSession` | Actor: bind lineage + model, stream generate, reload, fuse |
| `ModelSource` | `.id(_:revision:)` or `.directory(_:)` |
| `GenerationOptions` | maxTokens / temperature / seed |
| `AdaptModelLoader` | Load `ModelContainer` / `ModelContext` via protocol seams |
| `AdaptInferenceError` | One module error enum (`LocalizedError`) |

### `AdaptSession` ergonomics

```swift
let session = try await AdaptSession(
    model: .id("mlx-community/Qwen3-0.6B-4bit"),
    lineage: emailStyle,
    registry: registry,
    tokenizerLoader: myTokenizerLoader,
    downloader: myHubDownloader   // omit for .directory (bundle weights)
)
for try await token in session.generate(prompt: draft) {
    print(token, terminator: "")
}
// After promote in the registry:
try await session.reload()        // picks up new active adapter, no model reload
```

### Injection seams

| Seam | Protocol | Who implements it |
|---|---|---|
| Download weights | `MLXLMCommon.Downloader` | `adapt-cli` `HubDownloader`, or host cache |
| Load tokenizer | `MLXLMCommon.TokenizerLoader` | `adapt-cli` `TransformersTokenizerLoader`, or bundle loader |
| Local-only path | `ModelSource.directory` + tokenizer loader | Host app ships weights in the bundle — **no downloader** |

### `reload()` semantics

- Re-reads `registry.activeVersion(for:verifyIntegrity: true)`.
- Unloads the previous live LoRA (if any) and loads the new directory.
- Same version already loaded → no-op.
- No active version → unload to base.
- **Never mid-generation.** If a generation is in flight, `reload()` **waits**
  until every stream finishes or is cancelled, then swaps. In-flight streams keep
  the adapter they started with. Cancel outstanding generations first if you need
  the new adapter immediately.
- After `fuse()` → throws `AdaptInferenceError.fusedImmutable`.

### Cancellation

Cancelling the task that consumes `generate` stops production promptly (cooperative
cancel between chunks) and leaves the session reusable for the next generate.

## Dependencies

- `AdaptCore`, `AdaptRegistry`
- `mlx-swift` / `mlx-swift-lm` (`MLX`, `MLXNN`, `MLXLMCommon`, `MLXLLM`)
- **Not** `swift-huggingface` / `swift-transformers`

## Manual protocol — adapter swap latency (M5 / §6)

> **Not run under `swift test`.** Unit tests stay network-free and model-free.
> Measure on an Apple Silicon Mac with a real rank-8 adapter.

Acceptance target (architecture §6 M5): **&lt; 500 ms** for a rank-8 adapter swap
on M-series (unload previous + load new, no base model reload).

```bash
cd /path/to/swift-adapt
REG="$PWD/.build/demo-registry-inference"
# Train + promote two candidates (or reuse an existing demo registry), then:

swift run -c release adapt-cli generate \
  --prompt "Decline a meeting that conflicts with your watch." \
  --model mlx-community/Qwen3-0.6B-4bit \
  --task style-mirror \
  --rank 8 \
  --num-layers 8 \
  --registry "$REG" \
  --measure-swap
```

`--measure-swap` loads the base once, applies the active adapter, then times a
second `reload()` cycle (unload + load the same active version). Print the
millisecond number; compare to 500 ms.

### Transcript (real run, 2026-08-10, Apple Silicon)

| Machine | Adapter | Measured swap | Pass (&lt; 500 ms)? |
|---|---|---|---|
| Apple Silicon M-series (release `adapt-cli generate --measure-swap`) | rank-8, 8 layers, Qwen3-0.6B-4bit, active v4 | **8.0 ms** (promote + `reload()` load path after `clearActive` unload) | **yes** |

Well under the 500 ms target. The timed path is a full on-disk
`LoRAContainer.from(directory:)` + `load(into:)` after the previous live
layers were unloaded — not a same-version no-op.

## Offline tests

```bash
swift test --filter AdaptInferenceTests
```

Uses a fake `SessionModelBackend` (no weights, no Metal). Covers active load +
digest refuse, zero-config base path, reload without model reload, rollback +
reload, and mid-generation cancellation.
