# AdaptTrain

Resumable, interruption-safe on-device LoRA training over MLX.

## Contract

- **Own the step loop.** Does **not** call `MLXLLM.LoRATrain.train` — its progress
  callback only fires every N steps, so “cancellation loses ≤ 1 step” (§4.3) cannot
  be expressed. We call upstream only for pieces that fit (`LoRATrain.loss` CE
  primitive via shared `crossEntropy`, `LoRAContainer` for layer injection).
- **Own AdamW.** `MLXOptimizers.AdamW` state is not restorable through the public
  API (`stateStorage` is internal; `innerState()` is read-only/unkeyed). Moments
  `(m, v)` and step `t` are first-class serializable state in `CheckpointableAdamW`.
  Do not “fix” this back without an upstream restore path.
- **MLX stays inside this module.** `AdaptCore` / `AdaptRegistry` remain MLX-free.
  `Trainer` is an actor whose public API exchanges only `Sendable` values; all
  `MLXArray` / `Module` state is confined to the actor (pattern: `ModelContainer` /
  serial exclusive access).
- **Data seam.** `TrainingDataSource` is an ordered collection of
  `TrainingExample`. M2’s `ReplayBuffer` will satisfy it without reshaping `Trainer`.
- **No user data in outcomes.** `TrainOutcome` carries losses, rates, and version
  metadata only.

## Target modules (`LoRAConfig.keys`)

`keys` is the set of linear submodule names LoRA is injected into, combined with
`numLayers` (top-N). Upstream treats `keys: null` as “use the model’s
`loraDefaultKeys`” — for Qwen-style decoders that is **all seven** projections
(`q/k/v/o_proj` + `gate/up/down_proj`). That inheritance is model-dependent and
silent: the same `LoRAConfig()` would train different adapters on different
bases.

**Adapt defaults to an explicit attention-only set** so the adapted surface is
visible in config, lineage, and `adapt-cli inspect`:

| Setting | Keys | Params† | Tensors† | Adapter size† |
|---|---|---|---|---|
| **Default** (`LoRAConfig()` / `train --keys attention`) | `self_attn.{q,k,v,o}_proj` | 2,621,440 | 128 | 10.0 MB |
| Wide (`keys: LoRAConfig.allProjectionKeys` / `train --keys all`) | + `mlp.{gate,up,down}_proj` | 7,340,032 | 224 | 28.0 MB |

† Measured on `mlx-community/Qwen3-4B-4bit`, rank 8, 16 layers, F32
`adapters.safetensors` (100-step Nix fixture runs). MLP projections are ~3.8×
wider than attention; they dominate the wide size (~2.8× params vs attention-only).
`adapt-cli train` prints the resolved keys at start; `inspect` shows
`keys: …` per lineage. Legacy on-disk configs with `keys: null` still decode
(forward-compatible Codable).

Changing the default key set changes lineage identity (SHA-256 over
`LoRAConfig` JSON) — correct: a different key set is a different lineage.

### Stored dtype is F32 (do not “optimise” to fp16)

Adapter weights in `adapters.safetensors` are **float32**. Storing fp16 would
halve disk (~5 MB vs ~10 MB for the attention default) but the resume
guarantee asserts loss agreement to **1e-5** against an uninterrupted run, and
that oracle depends on exact restorable weights (and matching AdamW moments).
Quantising the very tensors that guarantee depends on is not a trade worth
making for disk. Keep F32 until an explicit, re-validated resume path exists.

## Checkpoint format

Each registry candidate `vN/` holds:

| File | Owner | Role |
|---|---|---|
| `adapter_config.json` | Registry | Upstream-compatible `LoRAConfig` (includes explicit `keys`) |
| `adapters.safetensors` | Registry | Trainable LoRA (or synthetic) weights **F32** |
| `version.json` | Registry | `AdapterVersion` metadata + digest |
| `train_state.json` | AdaptTrain | step, batch cursor, loss history, seed, config snapshot |
| `optimizer.safetensors` | AdaptTrain | AdamW moments keyed `m.<param>`, `v.<param>` |

Resume loads the **highest** version that has both AdaptTrain sidecars complete.
A crash between registry publish and sidecar write leaves a non-resumable candidate
that listing still tolerates (registry ignores incomplete train state; next run
resumes from the previous complete checkpoint).

### How resume matches uninterrupted loss

1. Adapter weights restored via `model.update(parameters:)`.
2. AdamW `(m, v, t)` restored exactly from `optimizer.safetensors` + `train_state.json`.
3. Batch order restored from `BatchCursor` (shuffled indices + offset + SplitMix64 state).
4. Next step therefore sees the same parameters, moments, and mini-batch as the
   uninterrupted run would have — so the loss sequence matches within float tolerance.

## Prompt / completion masking

Formatting is owned by **`AdaptCore.SFTPromptFormatter`** (shared with
AdaptInference) so train and generate cannot silently disagree:

1. **Detect** whether the tokenizer has a chat template
   (`PromptFormatConvention.chatTemplate` vs `.rawConcatenation`).
2. **Chat template:** apply the template over user (`prompt`) + assistant
   (`completion`); `promptTokenCount` is the generation-prefix length (user
   turn + start-of-assistant marker). Scaffold tokens are **not** supervised.
3. **Raw fallback** (no template): `encode(prompt) + encode(completion)` —
   same rule on the generate path.
4. Teacher-forcing shift: inputs = tokens[:-1], targets = tokens[1:].
5. **Loss only on assistant/completion tokens** (`predictedIndex >= promptTokenCount`).
6. Padding masked via length.
7. **`TrainingExample.weight`** multiplies per-token CE (defaults from
   `SignalSource.defaultWeight`). Not deferred — weighted in the objective.

The convention is written into `AdapterVersion.promptFormat` so a session that
would serve the adapter under a different convention fails with a typed error.

M3 held-out perplexity must use the same mask so numbers are comparable.
`MLXPerExampleCrossEntropyScorer` (this module) implements AdaptEval's
`PerExampleScorer` with that mask; the gate itself stays pure Swift in AdaptEval.

### Recipe finding → M3 (do not mini-build a gate here)

A controlled Qwen3-4B run (rank-8, 300 steps, 50-example fixture) memorized:
loss collapsed to **0.001** (~6 epochs) and fixture vocabulary (`lane`,
`cycle`, `scrap-side`) bled into unrelated answers. A 0.6B run at 100 steps
ended at loss ~2.62 without that collapse.

**M3's eval gate** (§4.5) already owns the correct response: pinned held-out
set, paired comparison, Wilcoxon, abstain below a floor. Do **not** add early
stopping or a held-out split inside AdaptTrain as a stopgap — it would be
thrown away. Until M3 lands, keep CLI step defaults conservative on small
corpora (see `adapt-cli` README).

## Budget & cancellation

`TrainBudget` binds on first of: `maxSteps`, `maxWallClock`, thermal threshold,
or cooperative `Task` cancellation. Exhaustion is a normal `TrainOutcome`, not an
error. Cancellation is checked at step boundaries (≤ 1 step lost) and still writes
a consistent checkpoint.

Memory: `Memory.memoryLimit` is set from `maxMemoryMB`. Peak step memory is
further controlled by `gradientAccumulationSteps`.

## Tests

`swift test` is offline and model-free. Suites use a tiny synthetic `Module` and a
stub `Tokenizer`. Real-model validation is **opt-in**:

```bash
ADAPT_REAL_MODEL_TESTS=1 swift test --filter RealModel
```

(That suite is not registered by default in M1 B1 — CLI slice B2 owns it.)

### Metal library (generated, not vendored)

SPM does not package Cmlx’s `default.metallib` automatically. Tests **do not**
commit one either — a frozen blob would silently drift from the floating
0.31.x mlx-swift pin. Instead `MetalBootstrap` runs
`scripts/ensure-mlx-metal-library.sh` on first use, compiling the resolved checkout
into a revision-keyed cache under `.build/mlx-metallib-cache/`, then copies the
metallib next to the test executable. Details:
[`Tests/AdaptTrainTests/README.md`](../../Tests/AdaptTrainTests/README.md).

## Public surface (summary)

- `Trainer` actor — `run` / `runLLM` (optional per-step `onStep` progress)
- `TrainConfig`, `TrainBudget`, `TrainOutcome`, `TrainStepProgress`, `TrainStopReason`
- `TrainingDataSource`, `ArrayTrainingData`
- `CheckpointableAdamW`, `SeededBatchIterator`, `PromptCompletionBatch`
- `TrainCheckpoint` / `TrainStateFile` (on-disk format)
- `AdaptTrainError`
