# AdaptSchedule

Invisible night-pipeline orchestration for on-device personalization (architecture §4.6).

## Contract

- **One call.** `AdaptPipeline.run()` composes
  `prune → sample → train(budget) → eval → maybe-promote → (sync later)`.
- **Stages are independently skippable.** Disabling sample does **not** skip
  train/eval/promote — train falls back to the full buffer snapshot.
- **Cancellation-safe at every stage boundary.** Cooperative cancellation is
  checked before each stage; the returned `PipelineOutcome` names the stage
  where the run stopped. The registry and buffer are never left mid-promote.
- **Promotion only through AdaptEval.** The pipeline never compares raw scores.
  Runners must return `EvaluationResult` / `GateDecision`.
- **Backoff is refuse-only.** Gate **refuse** increments exponential backoff;
  **abstain** and **pinBroken** do not. Promote resets backoff.
- **Prune may break the held-out pin** (AdaptData TTL wins). Eval still runs and
  surfaces `.pinBroken`; optional `clearBrokenPinForRePin` deletes the pin file
  so the next eval can re-pin. Eval is never silently skipped because prune ran.

## Budget defaults (explicit, not “measured on phone”)

| Platform | `maxSteps` | `maxWallClock` | Basis | Measurement status |
|---|---|---|---|---|
| iOS | 300 | 8 min | §4.6 sketch | **Not a phone measurement.** Mac: ~15 s / 100 rank-8 steps on Qwen3-4B-4bit ⇒ 300 steps ≈ 1 min on that Mac — the two §4.6 halves disagree for that hardware. Phone cost unknown (§7). |
| macOS | 2000 | 30 min | §4.6 sketch | Same Mac data ⇒ ~5 min for 2000 steps; wall clock is the looser bound. |

Thermal abort: `.serious`. Soft MLX memory cap: 4096 MB (AdaptTrain default).

See `PlatformTrainBudget.justificationSummary`.

## Device policy

| Constraint | iOS | macOS |
|---|---|---|
| Charging required | yes | no (no UIDevice battery gate) |
| Min battery | ≥ 40% | n/a |
| Thermal abort | ≥ `.serious` | ≥ `.serious` |

## Capability gate

`AdaptCapability` refuses training when physical RAM &lt; **6 GB** (§3) with
`AdaptScheduleError.insufficientMemory`.

## Background registration

- **iOS:** `BGProcessingTask` via `AdaptBackgroundScheduler` with
  `requiresExternalPower = true`, `requiresNetworkConnectivity = false`.
- **macOS:** `NSBackgroundActivityScheduler` (default 24 h, QoS `.background`).

## iOS enablement notes

| Surface | Status |
|---|---|
| Package platforms | macOS 15+, iOS 18+ (already declared) |
| `AdaptCore` / `AdaptData` / `AdaptRegistry` / `AdaptEval` | Pure Swift (+ sqlite); Data Protection file class is `#if os(iOS)` and only meaningful on device |
| `AdaptSchedule` policy + pipeline sources | Written with iOS paths (`UIDevice` battery, `BGProcessingTask`) |
| `AdaptTrain` / `AdaptInference` / `AdaptSchedule` (via Train) | Depend on **mlx-swift** / **mlx-swift-lm** |
- **iOS device build: verified.** `xcodebuild -scheme AdaptTrain -destination 'generic/platform=iOS' -skipPackagePluginValidation build` → **BUILD SUCCEEDED**. The earlier `Validate plug-in "CudaBuild" in package "mlx-swift"` failure was missing that flag, not an upstream blocker: mlx-swift ships a build plugin that Xcode asks to validate, and the flag answers it. Compilation and linking for the device are therefore proven; the device *runtime* is not (Metal library layout in a real bundle, per-step cost, the `BGProcessingTask` window, behaviour under memory pressure) — those stay in `Examples/QuickReply/TESTING.md` as a manual protocol.
| Test-time metallib (`scripts/ensure-mlx-metal-library.sh`) | **macOS developer convenience** for offline unit tests; says nothing about an App Store / device metallib layout |
| MLX training on a physical phone | **Pending** — QuickReply `TESTING.md` protocol |
| iOS Simulator training | **Not** a meaningful stand-in for device Metal |

If a path cannot work on iOS today, we document it rather than ship a target that
only fails at runtime. QuickReply is the device protocol vehicle.

## Public surface

| Symbol | Role |
|---|---|
| `AdaptPipeline` | Actor: `run(configuration:)` |
| `PipelineConfiguration` | Stage flags, budgets, policies |
| `PipelineOutcome` / `PipelineStage` / `PipelineStopReason` | Structured result |
| `PipelineTrainRunner` / `PipelineEvalRunner` | Train/eval seams |
| `AdaptCapability` | ≥ 6 GB RAM gate |
| `DevicePolicy` / `DeviceEnvironmentReading` | Thermal / battery / charging |
| `PlatformTrainBudget` | Documented defaults |
| `BackoffPolicy` / `BackoffStore` | Refuse-only exponential backoff |
| `AdaptBackgroundScheduler` | iOS / macOS registration helpers |
| `AdaptScheduleError` | Typed errors (`LocalizedError`) |

## Offline tests

```bash
swift test --filter AdaptScheduleTests
```

Model-free: injectable train/eval runners, fixed device environment, temp
directories. No network.
