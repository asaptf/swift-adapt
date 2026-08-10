# Adapt — On-Device Continual Personalization for Apple Platforms

> Architecture & implementation plan. Written to be consumed milestone-by-milestone by an AI coding agent (Claude Code). Each milestone is self-contained: scope, deliverables, acceptance criteria, and explicit non-goals.

---

## 1. Vision

**Adapt** is a Swift library that lets any iOS/macOS app ship a language model that quietly gets better at *its user* — fully on-device, no server, no Python.

Apple Foundation Models supports custom LoRA adapters, but training happens offline on a Mac with Apple's toolkit. MLX Swift can train LoRA on-device, but only as example code. **Adapt productizes the gap**: data collection → background training → evaluation → safe rollout → cross-device sync, as a single coherent framework.

**One-line pitch:** `@Personalizable` on your data model → your app's LLM writes like your user within a week, offline.

### Why this cannot be done in Python
- Python does not run on iOS. Period.
- Requires deep OS integration: `BGProcessingTask`, thermal state, battery, App Lifecycle, CloudKit, Keychain.
- Unified memory on Apple Silicon makes on-device LoRA training feasible (MLX); the orchestration around it is the missing product.

### Design principles
1. **Privacy is structural, not a setting.** Raw user data never leaves the device. Only encrypted adapter weights sync.
2. **Never degrade.** A new adapter ships to the user only after passing an on-device eval gate; instant rollback is always possible.
3. **Invisible cost.** Training runs only when the device is charging, idle, and thermally nominal. The user should never notice.
4. **Model-agnostic.** Works with any MLX-loadable model via `mlx-swift-lm`; not tied to one architecture.
5. **Ergonomics first.** One macro + three lines of code for the 80% case; full control underneath.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Host App                            │
│                                                             │
│   @Personalizable structs        AdaptSession (inference)   │
└──────────┬──────────────────────────────┬───────────────────┘
           │ examples                     │ generate(with: adapter)
           ▼                              ▼
┌──────────────────┐            ┌──────────────────────┐
│   AdaptData      │            │   AdaptInference     │
│  replay buffer   │            │  adapter hot-swap    │
│  privacy budget  │            │  over mlx-swift-lm   │
│  PII scrubbing   │            └──────────┬───────────┘
└────────┬─────────┘                       │ loads
         │ batches                         ▼
         ▼                      ┌──────────────────────┐
┌──────────────────┐   writes   │   AdaptRegistry      │
│   AdaptTrain     │──────────► │  adapter versions    │
│  MLX LoRA loop   │            │  lineage, rollback   │
│  checkpointing   │            └──────────┬───────────┘
└────────▲─────────┘                       │ gate
         │ schedules                       ▼
┌────────┴─────────┐            ┌──────────────────────┐
│  AdaptSchedule   │            │    AdaptEval         │
│ BGProcessingTask │            │  on-device benchmark │
│ thermal/battery  │            │  A/B, regression gate│
└──────────────────┘            └──────────┬───────────┘
                                           │ approved adapters only
                                           ▼
                                ┌──────────────────────┐
                                │    AdaptSync         │
                                │ CloudKit, E2E crypto │
                                │ cross-device merge   │
                                └──────────────────────┘
```

---

## 3. Package Structure (SwiftPM)

```
adapt/
├── Package.swift
├── Sources/
│   ├── AdaptCore/          // shared types, protocols, errors, logging
│   ├── AdaptData/          // example capture, replay buffer, privacy
│   ├── AdaptTrain/         // MLX LoRA training loop
│   ├── AdaptRegistry/      // adapter storage, versioning, rollback
│   ├── AdaptEval/          // on-device evaluation & promotion gates
│   ├── AdaptSchedule/      // background task orchestration
│   ├── AdaptInference/     // adapter loading & hot-swap over mlx-swift-lm
│   ├── AdaptSync/          // CloudKit encrypted adapter sync
│   └── AdaptMacros/        // @Personalizable, @TrainingSignal
├── Tests/                  // mirrors Sources, + AdaptIntegrationTests
├── Examples/
│   ├── StyleMirror/        // macOS demo: email drafts in your style
│   └── QuickReply/         // iOS demo: personalized reply suggestions
└── Tools/
    └── adapt-cli/          // train/eval/inspect adapters from terminal
```

**Dependencies:** `mlx-swift` (0.31.x), `mlx-swift-lm` (3.31.x — models, tokenizers, LoRA layers), `swift-syntax` (602–603, macros), `swift-argument-parser` (1.8.x, `adapt-cli` only — already a transitive dependency of `mlx-swift`). Nothing else. No networking beyond CloudKit and model download.

> Version pins verified against the toolchain in use (Swift 6.3.3 / Xcode 26.6): `mlx-swift` 0.31.6, `mlx-swift-lm` 3.31.4, `swift-syntax` 603.0.2. `mlx-swift-lm` constrains `swift-syntax` to `602.0.0..<604.0.0`, which the M5 macro target must respect.

**Platforms:** macOS 15+, iOS 18+. Training requires ≥6 GB RAM devices (gate at runtime via `AdaptCapability`).

---

## 4. Module Specifications

### 4.1 AdaptCore

Shared vocabulary. No MLX imports here — pure Swift, fully testable on Linux CI.

```swift
/// A single training example, already formatted for the target task.
public struct TrainingExample: Codable, Sendable {
    public let id: UUID
    public let prompt: String
    public let completion: String
    public let weight: Double          // importance sampling weight
    public let capturedAt: Date
    public let source: SignalSource    // .explicitEdit, .acceptance, .rejection, .synthetic
}

/// Identity of an adapter lineage (one per personalization task).
public struct AdapterLineage: Codable, Sendable, Hashable {
    public let taskID: String          // e.g. "email-style"
    public let baseModelID: String     // e.g. "mlx-community/Qwen3-4B-4bit"
    public let loraConfig: LoRAConfig
}

/// Wire-compatible with mlx-swift-lm's `LoRAConfiguration` (see §9, delta 1).
public struct LoRAConfig: Codable, Sendable, Hashable {
    public struct LoRAParameters: Codable, Sendable, Hashable {
        public let rank: Int      // 8
        public let scale: Float   // 10.0 — upstream's multiplier, NOT PEFT alpha
        public let keys: [String]?  // nil = the model's own default target keys
    }
    public let numLayers: Int              // 16 — adapt the top N layers
    public let fineTuneType: FineTuneType  // .lora | .dora
    public let loraParameters: LoRAParameters
    // Encodes as num_layers / fine_tune_type / lora_parameters
}

public struct AdapterVersion: Codable, Sendable {
    public let lineage: AdapterLineage
    public let version: Int
    public let parentVersion: Int?
    public let trainedOn: TrainingWindow   // date range + example count, NOT the data
    public let evalReport: EvalReport?
    public let status: AdapterStatus       // .candidate, .active, .rolledBack, .archived
}
```

Key design decision: **adapter metadata never contains user data** — only counts, date ranges, and metric values. This makes the registry safe to sync and safe to log.

Second key decision: `AdapterLineage` exposes a **`lineageID` derived by SHA-256** over canonical content (taskID + baseModelID + loraConfig JSON), used as the on-disk directory name. It must never be derived from Swift's `Hashable`/`hashValue`, which is seeded per process and would orphan every stored adapter after a restart.

### 4.2 AdaptData — capture, buffer, privacy

Responsibilities:
- Turn app-level signals into `TrainingExample`s via the `@Personalizable` macro or manual API.
- Maintain a **replay buffer**: bounded, deduplicated, stratified by `SignalSource` and recency (mix old + new to fight catastrophic forgetting).
- **Privacy budget**: per-lineage cap on how many examples per day may be captured; configurable retention (default 30 days, then examples are deleted even if never trained on).
- **PII scrubbing pipeline**: pluggable `Scrubber` protocol; ships with regex-based scrubbers for emails, phone numbers, credit cards, IBANs. Scrubbing runs *at capture time*, so **unscrubbed text is never persisted** — but note that what remains is still the user's real prose, and is treated as sensitive throughout (encrypted at rest, never synced, TTL-pruned). Do not claim "raw text is never persisted"; §5.1 states the actual guarantee.

```swift
public protocol Scrubber: Sendable {
    func scrub(_ text: String) -> String
}

public actor ReplayBuffer {
    public func add(_ example: TrainingExample) async throws
    public func sample(count: Int, strategy: SamplingStrategy) async -> [TrainingExample]
    public func prune(olderThan: Date) async
    public var stats: BufferStats { get async }
}
```

Storage: SQLite via `GRDB`-free raw `SQLite3` (avoid dependency) or flat protobuf-style files — **decide in M2, prefer SQLite** with `SQLCipher`-style encryption via `Data Protection` file classes (`.completeUntilFirstUserAuthentication`).

Signal taxonomy (the secret sauce — what counts as a training example):
| Signal | Example | Weight |
|---|---|---|
| `.explicitEdit` | user rewrote a generated draft | 1.0 (gold) |
| `.acceptance` | user sent generated text unchanged | 0.6 |
| `.rejection` | user dismissed a suggestion | 0.4 (used as negative/DPO pair in later milestone) |
| `.synthetic` | app-provided seed examples | 0.3 |

### 4.3 AdaptTrain — the MLX LoRA loop

Wraps the LoRA training pattern from `mlx-swift-examples` into a resumable, interruption-safe engine.

```swift
public actor Trainer {
    public init(lineage: AdapterLineage, buffer: ReplayBuffer, config: TrainConfig)

    /// Runs until budget exhausted, data exhausted, or cancellation.
    /// ALWAYS checkpoint-safe: cancellation at any point loses ≤ 1 step.
    public func run(budget: TrainBudget) async throws -> TrainOutcome
}

public struct TrainBudget: Sendable {
    public var maxSteps: Int
    public var maxWallClock: Duration
    public var maxMemoryMB: Int          // enforced via MLX GPU memory limit
    public var stopOnThermal: ProcessInfo.ThermalState = .serious
}
```

Requirements:
- **Checkpoint every N steps** (default 25) to the registry as a `.candidate`; resume from checkpoint on next run. Background time on iOS is unpredictable — treat every step as potentially the last.
- Gradient accumulation to keep peak memory under budget on 6 GB devices.
- Loss curve + tokens/sec recorded into `TrainOutcome` for the eval report.
- Deterministic given (seed, data order) — needed for tests. Requires **our own seeded batch iterator**: upstream's `LoRABatchIterator` is `internal` and shuffles with the unseeded global RNG (§9, delta 3).
- LoRA layers from `mlx-swift-lm`, injected via `LoRAContainer.from(model:configuration:)`. **AdamW is implemented inside `AdaptTrain`, not taken from `MLXOptimizers`** — upstream's optimizer state is not restorable through its public API, which makes checkpoint/resume impossible (§9, delta 4).
- Own the step loop rather than calling `LoRATrain.train`: its progress callback fires every `stepsPerReport` steps, so it cannot express "lose ≤ 1 step" (§9, delta 2).

### 4.4 AdaptRegistry — versioning & rollback

A content-addressed store of adapter weights (`.safetensors`) + metadata (JSON), with lineage semantics:

- `promote(version:)` — mark candidate as active after eval gate passes.
- `rollback(to:)` — O(1) pointer flip; old adapter files retained until archived.
- `gc(keepLast: 3)` — prune old versions.
- Files stored under `Application Support/Adapt/<lineageID>/v<N>/`, protected with Data Protection. Each version directory holds `adapter_config.json` + `adapters.safetensors` — **exactly the filenames upstream's `LoRAContainer.from(directory:)` reads**, so a registry directory is directly loadable with no glue — plus our `version.json` metadata. The active-version pointer lives in `<lineageID>/state.json` and is the single source of truth; version-level status fields are reconciled from it on read.
- A version becomes visible **all-or-nothing** (staged, then moved into place). An incomplete version directory must never make listing throw — otherwise a process killed mid-checkpoint permanently bricks the lineage, since version numbering reads the listing.

Invariant: **exactly one active version per lineage** (or none → base model behavior). The registry is the single source of truth; inference and sync both read from it.

### 4.5 AdaptEval — the promotion gate

A candidate adapter becomes active only if it beats the incumbent on an on-device eval:

- **Held-out set**: 10–20% of replay buffer, stratified, never trained on.
- Metrics: perplexity on held-out completions (primary), plus optional app-defined `Rubric` scored by the base model itself (LLM-as-judge, on-device).
- **Regression gate**: candidate must not be worse than active by more than ε on any core metric; must be better on primary.
- Produces an `EvalReport` persisted with the version.

```swift
public struct PromotionPolicy: Sendable {
    public var minExamplesSinceLastPromotion: Int = 40
    public var maxPerplexityRegression: Double = 0.02
    public var requireJudgeWinRate: Double? = nil   // e.g. 0.55
}
```

> **Open question for M3 (raised in review, unresolved).** As specified, this gate is statistically underpowered: 10–20% of a buffer gated at 40 examples is a **held-out set of 4–8 examples**, and perplexity noise on 8 examples far exceeds a 0.02 threshold — the gate would mostly be measuring sampling noise, which undermines the "never degrade" promise it exists to keep. Before implementing M3, decide: (a) a minimum absolute held-out size, not just a percentage; (b) a **paired** comparison — candidate vs. incumbent scored on the *same* examples, compared per-example — rather than two independent averages; (c) a confidence interval or sign test, so the gate can answer "is this difference real?"; and (d) the units of `maxPerplexityRegression` — ratio or absolute nats? Undefined today.

### 4.6 AdaptSchedule — invisible orchestration

macOS: background `Task` + `NSBackgroundActivityScheduler`. iOS: `BGProcessingTask` with `requiresExternalPower = true`, `requiresNetworkConnectivity = false`.

The scheduler composes a nightly pipeline: `prune → sample → train(budget) → eval → maybe-promote → maybe-sync`, with each stage individually skippable and the whole pipeline cancellation-safe.

Policies (all overridable):
- Only when charging AND battery > 40% (iOS).
- Abort on thermal `.serious`.
- Budget defaults: 300 steps or 8 minutes, whichever first (iOS); 2000 steps or 30 min (macOS).
- Exponential backoff after repeated failed evals (don't burn battery retraining a plateau).

### 4.7 AdaptInference — hot-swap

Thin layer over `mlx-swift-lm` chat/generation APIs:

```swift
let session = try await AdaptSession(
    model: .id("mlx-community/Qwen3-4B-4bit"),
    lineage: emailStyle          // auto-loads active adapter, if any
)
for try await token in session.generate(prompt: draft) { ... }
```

- Adapter weights fused at load OR applied as live LoRA layers (keep unfused → enables instant swap without model reload; fusing is an optimization decided in M5).
- `session.reload()` picks up a newly promoted adapter between generations.
- If no adapter: transparently falls back to base model. Zero-config cold start.

### 4.8 AdaptSync — cross-device adapters

- CloudKit private database, custom zone per lineage.
- Adapter weights encrypted client-side (CryptoKit, symmetric key in iCloud Keychain) — CloudKit stores opaque blobs.
- Conflict strategy: **highest eval score wins**, not last-writer-wins; ties → newer. Losing adapter is kept locally as a candidate.

> **Open question for M6 (raised in review, unresolved).** Eval scores from two devices are **not comparable**: each was computed against that device's own held-out set, i.e. different data. Ranking them directly is apples-to-oranges. The incoming adapter must be **re-evaluated locally against the local held-out set** before it can be compared to the local incumbent — which means sync depends on AdaptEval, not just the registry, and a device may reach a different verdict than its peer. That asymmetry needs a defined resolution (likely: each device independently decides its own active version, and only the weights sync).
- Sync only `.active` versions + their eval reports. Never the replay buffer.

### 4.9 AdaptMacros

```swift
@Personalizable(task: "email-style")
struct EmailDraft {
    @Prompt var context: String        // thread summary, recipient
    @Completion var body: String       // what the user actually sent
}

// Generated: EmailDraft.capture(...) → scrub → TrainingExample → buffer,
// respecting the privacy budget, entirely off the caller's hot path.
```

Follows the `swift-extract` `@Extractable` pattern: the macro generates conformances to `PersonalizationSignal`, a static `capture` method, and compile-time validation (exactly one `@Completion`, at least one `@Prompt`).

---

## 5. Security & Privacy Model (cross-cutting)

1. Raw examples: encrypted at rest (Data Protection), never synced, TTL-pruned.
2. Adapters: derived weights only; encrypted client-side before CloudKit.
3. No telemetry. `AdaptDiagnostics` writes local OSLog only.
4. Public `wipe()` API per lineage: deletes buffer + all adapter versions + CloudKit records — for the app's "reset personalization" setting. Required for App Store privacy story.
5. Document threat model in `SECURITY.md`: adapters can memorize training data; mitigations = scrubbing at capture, low rank, dropout, and eval-time canary check (M6 stretch: membership-inference smoke test).

---

## 6. Milestones for Claude Code

> Feed one milestone at a time. Each ends with a green test suite and a runnable artifact. Do not start milestone N+1 until N's acceptance criteria pass.

### M1 — Core engine: types + registry + offline training loop (macOS only)
**Scope:** `AdaptCore`, `AdaptRegistry`, `AdaptTrain`, `adapt-cli`.
**Deliverables:**
- All core types with Codable round-trip tests.
- Registry with promote/rollback/gc + invariant tests.
- Trainer that fine-tunes a real small model (Qwen3-0.6B-4bit or similar) on a JSONL file via `adapt-cli train --data examples.jsonl --steps 100`, producing a versioned adapter in the registry.
- Checkpoint/resume: kill the process mid-training, resume, final loss matches uninterrupted run within tolerance.
**Acceptance:** `swift test` green; CLI demo trains on 50 hand-written style examples and `adapt-cli generate` shows visibly different output with adapter vs. without.
**Non-goals:** iOS, scheduling, macros, sync, eval gates.

### M2 — Data layer: replay buffer + privacy + scrubbing
**Scope:** `AdaptData`.
**Deliverables:** SQLite-backed encrypted buffer; stratified sampling; privacy budget enforcement; scrubber pipeline with built-in PII scrubbers + tests with adversarial fixtures; TTL pruning.
**Acceptance:** property tests — no raw PII string ever appears in the DB file (scan bytes); budget cannot be exceeded under concurrent capture (actor test).

### M3 — Eval & promotion gate
**Scope:** `AdaptEval`; wire into `adapt-cli` as `adapt-cli eval` / `adapt-cli promote`.
**Deliverables:** held-out split management; perplexity eval on MLX; `PromotionPolicy` engine; `EvalReport` persisted to registry; regression gate that provably blocks a deliberately-corrupted adapter (test fixture).
**Acceptance:** end-to-end CLI flow train → eval → auto-promote; corrupted adapter is rejected and active version unchanged.

### M4 — Scheduling + iOS support
**Scope:** `AdaptSchedule`; iOS target enablement across all modules; capability gating.
**Deliverables:** pipeline orchestrator (prune→train→eval→promote) as a single cancellation-safe `AdaptPipeline.run()`; `BGProcessingTask` integration + macOS scheduler; thermal/battery policies; `Examples/QuickReply` iOS app skeleton demonstrating background training on-device.
**Acceptance:** pipeline unit tests with simulated cancellation at every stage boundary; QuickReply trains a real adapter overnight on a physical device (manual test protocol documented in `Examples/QuickReply/TESTING.md`).

### M5 — Inference hot-swap + macros
**Scope:** `AdaptInference`, `AdaptMacros`, `Examples/StyleMirror`.
**Deliverables:** `AdaptSession` with adapter auto-load, `reload()` swap without model reload; `@Personalizable` macro with diagnostics tests (swift-syntax); StyleMirror macOS demo: paste 30 of your emails → nightly (or forced) training → drafts in your style.
**Acceptance:** macro expansion tests; swap latency < 500 ms for rank-8 adapter on M-series; StyleMirror demo produces qualitatively personalized drafts (documented example transcript).

**StyleMirror is the demo vehicle** (see §10) and its acceptance includes three built-in screens, not just a draft view: a **live training** screen (loss curve + tokens/sec, watchable for the 2–3 minutes a rank-8 LoRA over a 0.6B model actually takes), a **blind test** screen (base draft vs. adapter draft vs. the user's real archived reply, audience guesses which is human, running tally), and a **code-switching** view (the same voice across the user's languages). The poisoning demo — a deliberately corrupted buffer that the eval gate refuses to promote — depends on M3's gate and is staged here.

### M6 — Sync + hardening
**Scope:** `AdaptSync`, security pass, docs.
**Deliverables:** CloudKit encrypted sync with eval-score conflict resolution (integration-tested against a mock CKDatabase protocol); `wipe()`; `SECURITY.md`; DocC documentation; README with the "week one" quickstart.
**Acceptance:** conflict-resolution property tests; full E2E: train on "device A" store, sync, load on "device B" store, generate.

### Stretch (post-1.0)
- DPO from `.rejection` signals (preference pairs instead of SFT).
- KV-cache persist/restore → foundation for cross-device generation handoff ("Handoff" as a sister library).
- Membership-inference canary in the eval gate.
- Foundation Models adapter export bridge (if/when Apple's format is publicly writable).

---

## 7. Key Technical Risks & Decisions

| Risk | Mitigation |
|---|---|
| iOS background budget too short for meaningful training | Checkpoint-everything design (M1); tiny ranks; gradient accumulation; measure real budgets in M4 and tune defaults |
| Catastrophic forgetting / style collapse | Stratified replay sampling; low LR; regression gate blocks bad adapters |
| 4-bit base + LoRA training quality | Train LoRA in fp16 over frozen quantized base (QLoRA pattern, supported by MLX); validate quality in M1 acceptance |
| Adapter memorizes PII | Scrub at capture; rank/dropout limits; canary eval (stretch) |
| mlx-swift-lm API churn (3.x major) | Pin versions; isolate all mlx-swift-lm imports inside AdaptTrain/AdaptInference |

## 8. Conventions for the Coding Agent

- Swift 6 language mode, strict concurrency. All shared state behind actors.
- Swift Testing (not XCTest) for new tests.
- Every public symbol gets a doc comment; every module gets a `README.md` with its contract.
- Errors: one error enum per module, **distinctly named** — `AdaptCoreError`, `AdaptRegistryError`, `AdaptTrainError`, … — all conforming to `LocalizedError`. Not one type called `AdaptError` per module: Swift does not namespace types at the use site, so identically-named enums collide in any file importing two modules and force callers to write `AdaptRegistry.AdaptError`.
- Public API carries no test-only affordances: no fault-injection cases in public error enums, no test hooks in public surface (use `package` access — the test targets are in the same package).
- No new third-party dependencies without updating this document's §3.
- Commit per logical unit; each milestone ends with a tagged release `m1`, `m2`, …

---

## 9. Verified deltas (implementation log)

Findings from reading the **pinned upstream source** (`mlx-swift` 0.31.6, `mlx-swift-lm` 3.31.4) rather than assuming its API. Each contradicts something written above; the inline sections have been corrected and cross-reference this list. Recorded during M1, 2026-08-10.

**1. `LoRAConfig`'s fields were wrong.** §4.1 originally specified `rank` / `alpha` / `dropout` / `targetModules`. Upstream's `LoRAConfiguration` is `numLayers` / `fineTuneType` / `loraParameters{rank, scale, keys}`. Consequences: `alpha` is really `scale`, a direct multiplier rather than PEFT's `alpha/rank`; target modules are selected by `keys` (defaulting to the model's own `loraDefaultKeys`) combined with "top N layers", not by an explicit module list; and **there is no `dropout` at all** — upstream's `LoRALinear`/`QLoRALinear` do not implement it. Note this weakens §7's PII-memorization mitigations, which list dropout as one of them; rank and scrubbing now carry that load. Our `LoRAConfig` mirrors upstream's exact JSON encoding, so registry directories are loadable by upstream unchanged.

**2. `LoRATrain.train` cannot meet §4.3's interruption guarantee.** Its `progress` closure — the only way to stop it — is invoked every `stepsPerReport` / `stepsPerEval` / `saveEvery` iterations, never per step. "Cancellation loses ≤ 1 step" is inexpressible through it. `AdaptTrain` owns its step loop and calls upstream only for the pieces that do fit (`LoRATrain.loss`, `LoRAContainer`).

**3. Upstream's batch iterator is unusable for us.** `LoRABatchIterator` is `internal` to `MLXLLM` and shuffles via `indices.shuffle()` on the unseeded global RNG — neither reachable nor reproducible. §4.3's determinism requirement therefore needs our own seeded iterator (`AdaptCore.SeededGenerator`, SplitMix64).

**4. `MLXOptimizers` AdamW state cannot be checkpointed.** `OptimizerBase.stateStorage` is `internal`; the sole public accessor `innerState() -> [MLXArray]` is read-only and unkeyed, with no restore path. Resuming with correct moments is impossible through the public API — so M1's acceptance criterion ("resume, final loss matches uninterrupted run") is unachievable while using it. `AdaptTrain` implements AdamW itself with per-parameter-key `(m, v)` and step `t` as serializable state. This is a deliberate deviation from §4.3; the resume test is its oracle.

**5. Lineage directory naming was a latent data-loss bug.** §4.4 said `<lineage-hash>` while `AdapterLineage` is `Hashable`; using `hashValue` would have produced a *different directory name on every process launch*, orphaning all stored adapters after a restart. Now SHA-256 over canonical content, with a hardcoded-digest test so a change to the hash inputs fails loudly.

**Also found in review, fixed:** a version directory was assembled non-atomically, so a process killed mid-checkpoint left a partial `v<N>/` that made listing throw — and since version numbering reads the listing, that **permanently bricked the lineage**. On iOS, where checkpoints are written every ~25 steps into a process the OS kills at will, this was a routine path, not an edge case. Versions are now staged and moved into place atomically, incomplete directories are skipped, and `gc` sweeps orphaned staging dirs.

### M1 status

- **Slice A — done, verified.** `AdaptCore` (types, stable lineage IDs, seeded PRNG) + `AdaptRegistry` (atomic promote/rollback/gc, integrity digests, crash-safe pointer flips). 25 tests, offline, no warnings under Swift 6 strict concurrency.
- **Slice B1 — `AdaptTrain`**, carrying deltas 2–4.
- **Slice B2 — `adapt-cli`** (`train` / `generate` / `inspect`) and the real-model acceptance demo of §6 M1.

Unit tests are **network-free and model-free** throughout: MLX-level tests run against tiny synthetic modules and a stub tokenizer. Anything needing real weights is an opt-in suite, disabled by default.

---

## 10. The demo: StyleMirror as the launch artifact

The strongest demo is one where **training happens in front of the audience** — not "we trained this earlier, trust us". Three acts, roughly five minutes, delivered by `Examples/StyleMirror` (M5).

**Act 1 — Airplane mode.** The first gesture is switching on airplane mode in plain view. Theatrical, but it sells the entire premise instantly: nothing that follows can physically reach a server. This is the opening three seconds of the video, and the app makes offline a celebrated state rather than a warning — alongside a "bytes sent: 0" counter that stays at zero all demo long.

**Act 2 — Training, live.** Paste ~30 of your own sent emails, press Train. A rank-8 LoRA over a 0.6B model on an M-series Mac is two to three minutes — the right length for a scene. A live loss curve and tokens/sec fill the time, and the narration writes itself: at night, `BGProcessingTask` does exactly this unattended, on the charger. A compact v1→v7 timeline tells the seven-nights story with a rising eval score.

**Act 3 — Blind test.** One incoming email, three candidate replies: base model, adapter, and your real archived reply. The audience guesses which is human. If the adapter gets mistaken for you more often than the base model does, the demo has made its argument — and "guess which reply is human" is a ready-made interactive for LinkedIn/HN.

**Two amplifiers:**

- **Code-switching.** A user writing in three languages shows the adapter learned their voice *in each* — a generic model writes correctly, the adapter writes like them. No cloud personalization demo shows this, because none is handed this much personal data.
- **Poisoning.** For a technical audience: feed 20 examples of ALL-CAPS pirate slang, run the pipeline, and watch the **eval gate publicly refuse to promote** while the active version holds. This sells "never degrade" better than any slide, and it is precisely what separates a library from a script. Depends on M3.

**60-second launch cut:** airplane mode → timelapse of seven nights (v1→v7, eval score climbing) → blind test.

Because StyleMirror was already the M5 deliverable, none of this changes the architecture — it only adds the blind-test, live-training, and code-switching screens to M5's acceptance criteria (§6).

**Build order note:** the demo's UI is developed against a protocol seam with a scripted mock engine, so the app is buildable and reviewable before `AdaptInference` exists, then switched to the real engine when M5 lands. The seam is the deliverable that keeps demo work off M1's critical path.
