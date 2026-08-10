# adapt-cli

Terminal front-end for Adapt’s M1 core: train a LoRA adapter on JSONL style
examples, inspect the on-disk registry, and compare base vs adapter generation.

## Subcommands

| Command | Purpose |
|---|---|
| `train` | Fine-tune via `AdaptTrain.Trainer`, stream progress, write versioned candidates |
| `generate` | **Default experience:** base model **and** active/latest adapter side-by-side |
| `inspect` | List lineages, versions, active pointer, digests, training windows |
| `eval` | Run the §4.5 promotion gate (pinned held-out, paired Wilcoxon) against the active adapter |
| `promote` | Manually flip the active pointer (**override**; prefer `eval --promote`) |
| `measure` | Held-out mean cross-entropy (nats/token) for one version — **measurement only**, not a gate |
| `export-demo-nights` | Split a combined JSONL into `night-N.jsonl` + `held-out.jsonl` |

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

Fixture corpora (two personas, two purposes — both synthetic; no real names or
addresses):

| File | Persona | Size | Use |
|---|---|---|---|
| [`Fixtures/nix-caldera-style.jsonl`](Fixtures/nix-caldera-style.jsonl) | **Nix Caldera** (asteroid-belt salvage broker; `Nix here—` … `—Nix / Belt lane 4`) | ~50 | CLI quick train / generate smoke |
| [`Fixtures/nix-caldera-seven-nights.jsonl`](Fixtures/nix-caldera-seven-nights.jsonl) | Nix Caldera | 240 | CLI seven-night partition smoke (not the StyleMirror stage registry) |
| [`Fixtures/renna-vale-seven-nights.jsonl`](Fixtures/renna-vale-seven-nights.jsonl) | **Renna Vale** (Harborfinch product lead; short, direct, `— renna`; en/es/ru) | 240 | StyleMirror demo registry — must match `SampleCorpus` persona and blind-test held-out humans |

## Held-out measurement (`measure`)

```bash
swift run adapt-cli measure \
  --data path/to/held-out.jsonl \
  --version 3 \
  --model mlx-community/Qwen3-4B-4bit \
  --task style-mirror \
  --rank 8 \
  --num-layers 8 \
  --registry "$REG" \
  --record
```

Reports **mean cross-entropy in nats per supervised token** (lower is better),
using the same completion mask as training. With `--record`, the value is stored
on the version’s `EvalReport` together with `primaryMetric`,
`primaryDirection=lowerIsBetter`, and example/token counts.

This is **not** the promotion gate. Use `eval` for the decision procedure
(pinned held-out set, paired Wilcoxon, abstain floor — architecture §4.5).
`measure` only produces the number; it never auto-promotes or rejects.

## Promotion gate (`eval`)

```bash
swift run adapt-cli eval \
  --data path/to/pool-or-held-out.jsonl \
  --version 7 \
  --model mlx-community/Qwen3-4B-4bit \
  --task style-mirror \
  --rank 8 \
  --num-layers 8 \
  --registry "$REG" \
  --record
# optional: --promote   # flips active only when the gate returns promote
```

Compares the candidate to the **active** adapter on a **lineage-pinned** held-out
set (`held_out_pin.json` beside the lineage). Outcomes:

| Verdict | Meaning | Exit (with `--promote`) |
|---|---|---|
| `PROMOTE` | Significant Wilcoxon improvement | promotes |
| `REFUSE` | Worse or not significant (feeds §4.6 backoff) | exit 1 |
| `ABSTAIN` | Below `minHeldOut` / no incumbent (not a refusal) | exit 3 |
| pin broken | Pinned IDs missing from `--data` | exit 2 |

`promote` without `eval` remains a **manual override** and prints a warning.

## Seven-night demo registry

```bash
# StyleMirror stage persona (Renna Vale, multilingual):
bash scripts/seed-demo-registry.sh Tools/adapt-cli/Fixtures/renna-vale-seven-nights.jsonl
# CLI quickstart persona (Nix Caldera) still works the same way:
bash scripts/seed-demo-registry.sh Tools/adapt-cli/Fixtures/nix-caldera-seven-nights.jsonl
# → .build/demo-registry/  (gitignored; ~200 MB — derive, don’t vendor)
swift run -c release adapt-cli inspect --registry .build/demo-registry
```

Pass the corpus as the first argument (or `DEMO_CORPUS`); when omitted the script
defaults to the Renna Vale fixture used by StyleMirror. Runs seven separate
`train` processes (night N resumes from night N−1’s adapter and optimizer state,
trains on that night’s new examples only), measures the shared held-out slice
after each night, and prints a summary table. Defaults:
`mlx-community/Qwen3-4B-4bit`, attention-only keys, 40 steps/night. Override with
`DEMO_MODEL`, `STEPS_PER_NIGHT`, `DEMO_REGISTRY`, etc.

## Registry

Default root: Application Support/`Adapt` (same as `AdapterRegistry()`).
Override with `--registry /path/to/root` on every subcommand.

Lineage identity is the SHA-256 of `taskID + baseModelID + LoRAConfig`. Training
and generate/promote must use the **same** `--task`, `--model`, `--rank`,
`--num-layers`, `--scale`, and **`--keys`** or they will point at different
directories.

### `--keys` (target modules)

Default is **attention only** (`self_attn.q/k/v/o_proj`) — not the model’s full
linear set. Upstream `keys: null` would silently include MLP projections and
~4× the parameters; Adapt makes the set explicit instead. Keys are
**layer-relative paths** (mlx-swift-lm `namedModules` form).

| Flag | Meaning |
|---|---|
| `--keys attention` (default) | `self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj` |
| `--keys all` / `--keys wide` | attention + `mlp.gate_proj,mlp.up_proj,mlp.down_proj` |
| `--keys model` | inherit model `loraDefaultKeys` (legacy upstream behaviour) |
| `--keys self_attn.q_proj,self_attn.v_proj` | explicit list (comma or repeated options) |

`inspect` prints the configured keys so you can answer “what does this adapter
adapt?” without opening safetensors.

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
