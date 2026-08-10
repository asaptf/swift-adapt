# AdaptCore

Shared vocabulary for the Adapt library: training examples, adapter lineage, LoRA configuration, version metadata, evaluation placeholders, and a seedable PRNG.

## Contract

- **Pure Swift.** Depends only on Foundation and CryptoKit. No MLX, no platform UI frameworks.
- **Metadata never carries user data.** `TrainingWindow` and `AdapterVersion` store counts, date ranges, and metric placeholders — never prompt/completion text.
- **Wire-compatible LoRA config.** `LoRAConfig` encodes as `adapter_config.json` with the same snake_case shape as `mlx-swift-lm`'s `LoRAConfiguration`, so registry directories load via upstream `LoRAContainer.from(directory:)` with zero glue.
- **Stable lineage IDs.** `AdapterLineage.lineageID` is a SHA-256 hex digest of canonical content. It is filesystem-safe and process-stable (not Swift `Hashable`).
- **Forward-compatible Codable.** Types decode legacy/minimal JSON and ignore unknown keys where appropriate so on-disk metadata survives schema growth.
- **Deterministic randomness.** `SeededGenerator` (SplitMix64) supports reproducible training shuffles; its `state` is serializable for train/resume.
- **Signal weights.** `SignalSource.defaultWeight` encodes the §4.2 importance table; `TrainingExample` uses it when weight is omitted.

Host apps and other Adapt modules import this package for types only; it performs no I/O and holds no shared mutable state.
