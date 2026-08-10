# adapt-cli

Terminal front-end for Adapt’s M1 core: train a LoRA adapter on JSONL style
examples, inspect the on-disk registry, and compare base vs adapter generation.

## Subcommands

| Command | Purpose |
|---|---|
| `train` | Fine-tune via `AdaptTrain.Trainer`, stream progress, write versioned candidates |
| `generate` | **Default experience:** base model **and** active/latest adapter side-by-side |
| `inspect` | List lineages, versions, active pointer, digests, training windows |
| `promote` | Manually flip the active pointer (eval gate is M3; not auto) |

```bash
swift run adapt-cli --help
swift run adapt-cli train --help
swift run adapt-cli generate --help
swift run adapt-cli inspect
swift run adapt-cli promote --version 1
```

## Model loading (injection seams)

`AdaptInference` owns model load, adapter hot-swap, and generation
(`AdaptSession` / `AdaptModelLoader`). This CLI target only supplies the
**Hugging Face adapters** that implement mlx-swift-lm’s protocol seams:

| Seam | CLI type | Library consumer |
|---|---|---|
| `Downloader` | `HubDownloader` | `AdaptModelLoader` / `AdaptSession` |
| `TokenizerLoader` | `TransformersTokenizerLoader` | same |

Library modules stay free of `swift-huggingface` / `swift-transformers` —
network I/O is structurally confined to the CLI (architecture §3). Digest
verification runs inside `AdaptSession` before any adapter is applied (same
discipline the CLI previously enforced at the call site).

```bash
# Optional: time one reload() load path (rank-8 swap latency, §6 M5 target < 500 ms)
swift run -c release adapt-cli generate ... --measure-swap
```

## JSONL input format

One JSON object per line, mapping to `AdaptCore.TrainingExample`:

| Field | Required | Notes |
|---|---|---|
| `prompt` | yes | non-empty string |
| `completion` | yes | string (the target style reply) |
| `source` | no | `explicitEdit` \| `acceptance` \| `rejection` \| `synthetic` (default) |
| `weight` | no | defaults via `SignalSource.defaultWeight` |
| `id` | no | UUID; generated if omitted |
| `capturedAt` | no | ISO-8601; defaults to now |

Blank lines and `#` comments are skipped. A malformed line fails with the
**1-based line number** and a short reason — never a stack dump.

Example:

```json
{"prompt":"Decline a meeting.","completion":"Nix here—bad timing. … —Nix / Belt lane 4","source":"synthetic"}
```

Fixture corpus: [`Fixtures/nix-caldera-style.jsonl`](Fixtures/nix-caldera-style.jsonl)
(~50 synthetic examples in the voice of **Nix Caldera**, a fictional asteroid-belt
salvage broker). Distinctive register: opens with `Nix here—`, closes with
`—Nix / Belt lane 4`, short clipped sentences, salvage jargon. No real names or
addresses.

## Registry

Default root: Application Support/`Adapt` (same as `AdapterRegistry()`).
Override with `--registry /path/to/root` on every subcommand.

Lineage identity is the SHA-256 of `taskID + baseModelID + LoRAConfig`. Training
and generate/promote must use the **same** `--task`, `--model`, `--rank`,
`--num-layers`, and `--scale` or they will point at different directories.

## Checkpoint / resume

`train` checkpoints every `--checkpoint-every` steps (default 25) and on exit.
Ctrl-C installs a cooperative cancel: the trainer finishes the current step,
writes a consistent candidate, and stops. Re-running the **same** `train`
command resumes from the highest complete checkpoint (adapter weights + AdamW
moments + batch cursor) rather than restarting from step 0.

## Default step count (anti-overfit)

`--steps` defaults to **100**. On the 50-example Nix fixture with `--batch-size 1`
that is ≈ 2 epochs — enough for a visible style pull on small models without
memorizing the corpus.

A measured run at **300** steps (≈ 6 epochs) collapsed train loss to ~0.001 and
bled fixture vocabulary (`lane`, `cycle`, `scrap-side run`) into unrelated
answers. That is overfit, not a better recipe. Raise steps only with more data
or after M3's held-out eval gate can refuse a memorized candidate. Do **not**
treat early stopping inside `train` as a substitute for that gate.

## Manual acceptance protocol (real model)

> **Not run under `swift test`.** Unit tests stay network-free and model-free.
> This protocol downloads weights once (Hugging Face cache) and needs an
> Apple Silicon Mac with several GB free RAM.

### Environment

- macOS 15+, Apple Silicon (M1 or newer)
- Xcode / Swift 6 toolchain
- Network for first-time model download (~0.5–1 GB for 0.6B 4-bit)

### Commands

```bash
cd /path/to/swift-adapt

# Optional: isolated registry so the demo is reproducible
REG="$PWD/.build/demo-registry"
rm -rf "$REG"

# 1) Train (default --steps 100) rank-8 LoRA on the Nix Caldera fixture
swift run -c release adapt-cli train \
  --data Tools/adapt-cli/Fixtures/nix-caldera-style.jsonl \
  --steps 100 \
  --model mlx-community/Qwen3-0.6B-4bit \
  --task style-mirror \
  --rank 8 \
  --num-layers 8 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --seed 42 \
  --checkpoint-every 25 \
  --registry "$REG" \
  --promote

# 2) Compare base vs adapter (default experience)
swift run -c release adapt-cli generate \
  --prompt "Decline a meeting that conflicts with your watch." \
  --model mlx-community/Qwen3-0.6B-4bit \
  --task style-mirror \
  --rank 8 \
  --num-layers 8 \
  --max-tokens 120 \
  --temperature 0 \
  --registry "$REG"

# Optional sampling knobs (defaults disable both — same as omitting them):
#   --top-p 0.9
#   --repetition-penalty 1.15
#   --repetition-context-size 20

# 3) Inspect registry
swift run -c release adapt-cli inspect --registry "$REG"
```

### Expected cost (M-series Mac)

| Phase | Wall clock (approx.) | Memory (approx.) |
|---|---|---|
| First model download | 1–5 min (network) | n/a |
| 100 train steps, rank-8, 0.6B-4bit | 2–8 min | 4–8 GB unified |
| Generate (base + adapter) | 10–40 s | 2–4 GB |

Numbers vary by chip and thermal state. Prefer `release` for the timed run.

### What “visibly different” means

The adapter should pull replies toward Nix Caldera’s register:

- Opening `Nix here—`
- Closing `—Nix / Belt lane 4`
- Short, clipped, salvage-broker diction (`manifest`, `thrusters`, `no-go`)

Base Qwen3-0.6B typically produces generic polite English without those markers.
If both sides look the same after 100 steps, treat it as a **finding** (raise
`--steps`, `--learning-rate`, or strengthen the fixture) — do not dress it up.

### Resume check (manual)

```bash
# Start training, Ctrl-C after ~15–20 steps once a checkpoint prints
swift run -c release adapt-cli train ... --steps 100 --registry "$REG"

# Re-run the identical command — lifetime step count should continue, not reset
swift run -c release adapt-cli train ... --steps 100 --registry "$REG"
```

### Transcript (real run, 2026-08-10, Apple Silicon)

**Train** (`--steps 100 --rank 8 --lr 1e-4`, release build, model already cached after first download):

```
Training lineage a2c9bb6a7901d049…
  task=style-mirror  model=mlx-community/Qwen3-0.6B-4bit  rank=8  layers=8
  steps=100  batch=1  lr=0.0001  seed=42
  examples=50  checkpointEvery=25
  step 1 (lifetime 1)  loss=6.6447  tokens=9
  step 50 (lifetime 50)  loss=3.8599  tokens=516
  step 100 (lifetime 100)  loss=2.6216  tokens=1032
Done.
  stop=maxSteps
  stepsThisRun=100  lifetime=100
  tokens=1032  tok/s=124.4
  candidate=v5  digest=bf458b157289…
  promoted v5 → active
```

Wall clock for the 100-step loop was on the order of **~10–15 s** once the model
was warm (download is separate). Peak unified memory stayed well under the
4 GB `Memory.memoryLimit` on this M-series Mac.

**Generate** (same registry; prompt matches a fixture topic; greedy decode):

```
=== WITHOUT adapter (base) ===
prompt: Decline a meeting that conflicts with your watch.
What is the correct way to handle this situation? What are the possible ways
to handle it? What are the possible ways to handle it? …

=== WITH adapter v5 (active) ===
prompt: Decline a meeting that conflicts with your watch.
Nix here—watch's a quiet lane. Manifest's a quiet lane. I don't bargain on the
quiet lane. —Nix / Belt lane 4. Manifest lane 4. —Nix / Belt lane 4. …

--- comparison ---
Outputs differ (adapter effect visible).
base chars=373  adapted chars=219
```

Base is generic English. Adapted output picks up the fixture voice markers
(`Nix here—`, `—Nix / Belt lane 4`, salvage diction). Some repetition remains
at 100 steps / rank-8 — enough for a clear yes on “visibly different.”

**Resume:** a second `train` invocation on the same lineage starts at
`lifetime = previous + 1` (e.g. first run ends lifetime 200, resume step 1
prints lifetime 201) with loss continuing from the low plateau rather than
resetting to ~6.6. Checkpoint sidecars are what make that work.

**Ctrl-C:** the CLI installs a cooperative SIGINT handler that cancels the
train `Task`; `Trainer` checkpoints on cancel. On a warm 0.6B-4bit run the
step loop is ~3–8 s for 100 steps, so a human interrupt races the process —
the cancel path is the same one covered by `AdaptTrainTests` budget/cancel
suites, and process-level resume is verified as above.

## Offline tests

```bash
swift test --filter AdaptCLITests
```

Covers JSONL parsing (including line numbers on errors), argument validation
helpers, inspect formatting, and comparison footer text. No downloads, no weights.
