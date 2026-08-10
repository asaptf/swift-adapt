# Changelog

## 0.2.0 — 2026-08-10

Milestones 2 through 5, plus the inference layer that was split out of milestone 1.

Adapters trained with 0.1.0 are **not** comparable to adapters trained with 0.2.0: both the prompt formatting and
the default set of adapted modules changed. Retrain rather than carry old versions forward.

### Added

- **AdaptInference** — `AdaptSession`: generation with or without the active adapter, digest verification before a
  load, adapter hot-swap that does not reload the model, cancellation mid-generation, and top-p and
  repetition-penalty sampling.
- **AdaptData** — SQLite replay buffer, privacy budget, TTL pruning with a durable prune log, and a scrubber
  pipeline (email, IBAN, card, phone) that runs before anything is stored.
- **AdaptEval** — the promotion gate: a held-out set pinned to the lineage, paired per-example cross-entropy, a
  one-sided Wilcoxon signed-rank test at α = 0.05, a minimum-evidence floor, and a broken-pin outcome that is
  neither a pass nor a failure. Abstaining and refusing are distinct in the type system, not two readings of one
  boolean.
- **AdaptSchedule** — `AdaptPipeline` (`prune → sample → train → eval → promote`), thermal, battery and memory
  policy, backoff on refusal only, and `BGProcessingTask` registration.
- **AdaptMacros** — `@Personalizable(task:)`, `@Prompt`, `@Completion`.
- `adapt-cli eval`, and a script that seeds a seven-night registry by training seven adapters in seven separate
  processes, each resuming from the one before.
- Examples: `StyleMirror` (macOS demo) and `QuickReply` (iOS skeleton).

### Changed

- **Training and generation both go through the model's chat template.** One formatter serves both paths, the
  convention is recorded in the adapter's metadata, and a session refuses an adapter trained under a different
  convention instead of generating subtly wrong output. On Qwen3-4B-4bit at 100 steps this moved the loss from
  9.63 to 1.49.
- **LoRA target modules are explicit, and the default adapts attention projections only.** Adapting the MLP
  projections as well costs 7.3M parameters against 2.6M for no visible gain in style, and reaches a lower training
  loss largely by memorising more. This changes the shape of a default-configuration adapter.
- `EvalReport` carries the gate's decision fields: the primary metric, its direction, and the example and
  supervised-token counts behind the number.
- Per-module error enums (`AdaptCoreError`, `AdaptRegistryError`, `AdaptTrainError`, …) instead of one shared enum.

### Fixed

Twelve findings from an external review, the ones that could lose data or mislead:

- A checkpoint whose state file was not written last, so a crash mid-write could leave the registry unreadable.
- Gradient accumulation that weighted micro-batches equally regardless of how many tokens each contained.
- `.noData` returned when data merely ran short rather than out.
- LoRA initialisation that ignored the configured seed, making "same seed ⇒ same run" false.
- A missing digest verification before `generate` loads an adapter.
- Lineage IDs used in path construction without validation.

### Known limitations

- On-device training has not been measured on a physical iPhone. The iOS build compiles and links and the pipeline
  runs in the simulator; step cost, thermal behaviour and the background window on real hardware are unmeasured.
- Encrypted sync between devices (M6) is not built.
- The demo app promotes through a provisional threshold, not `AdaptEval`'s gate.
- The demo's blind test needs about 90 seconds in the app against 3.8 seconds for the same work in a smoke test.
- A rank-8 adapter over a corpus that is 20% Spanish and 20% Russian does not hold a non-English voice. The
  multilingual demo screen was cut rather than tuned.

## 0.1.0 — 2026-08-10

`AdaptCore`, `AdaptRegistry`, `AdaptTrain` and `adapt-cli`: versioned adapter storage with atomic promote and O(1)
rollback, and interruption-safe LoRA training over MLX that resumes to within 1e-5 of the uninterrupted run's loss
curve and final weights.
