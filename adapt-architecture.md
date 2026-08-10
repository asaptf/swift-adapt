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

**Dependencies:** `mlx-swift`, `mlx-swift-lm` (models, tokenizers, LoRA layers), `swift-syntax` (macros). Nothing else. No networking beyond CloudKit.

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

public struct LoRAConfig: Codable, Sendable, Hashable {
    public var rank: Int = 8
    public var alpha: Float = 16
    public var dropout: Float = 0.05
    public var targetModules: [String] = ["q_proj", "v_proj"]
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

### 4.2 AdaptData — capture, buffer, privacy

Responsibilities:
- Turn app-level signals into `TrainingExample`s via the `@Personalizable` macro or manual API.
- Maintain a **replay buffer**: bounded, deduplicated, stratified by `SignalSource` and recency (mix old + new to fight catastrophic forgetting).
- **Privacy budget**: per-lineage cap on how many examples per day may be captured; configurable retention (default 30 days, then examples are deleted even if never trained on).
- **PII scrubbing pipeline**: pluggable `Scrubber` protocol; ships with regex-based scrubbers for emails, phone numbers, credit cards, IBANs. Scrubbing runs *at capture time* — raw text is never persisted.

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
- Deterministic given (seed, data order) — needed for tests.
- Uses `MLXOptimizers` AdamW; LoRA layers from `mlx-swift-lm`, injected via module surgery on the loaded model.

### 4.4 AdaptRegistry — versioning & rollback

A content-addressed store of adapter weights (`.safetensors`) + metadata (JSON), with lineage semantics:

- `promote(version:)` — mark candidate as active after eval gate passes.
- `rollback(to:)` — O(1) pointer flip; old adapter files retained until archived.
- `gc(keepLast: 3)` — prune old versions.
- Files stored under `Application Support/Adapt/<lineage-hash>/v<N>/`, protected with Data Protection.

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
- Errors: one `AdaptError` enum per module, all conforming to `LocalizedError`.
- No new third-party dependencies without updating this document's §3.
- Commit per logical unit; each milestone ends with a tagged release `m1`, `m2`, …
