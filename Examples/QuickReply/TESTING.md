# QuickReply — manual device protocol (M4 acceptance)

> **This protocol is not executed by CI and was not run in the environment that
> authored M4.** Treat every step below as **pending device verification** until
> you complete it on a physical iPhone. Do not describe an unrun protocol as
> passing.

## Goal

Show that Adapt can train a real LoRA adapter overnight on a **physical iOS
device** under the §4.6 background policy, without a network dependency during
training.

## Prerequisites

| Item | Requirement |
|---|---|
| Device | Physical iPhone/iPad, iOS 18+, **≥ 6 GB RAM** (`AdaptCapability` floor) |
| Power | Charger connected for the overnight window |
| Battery | Start above **40%** (policy gate) |
| Thermal | Device cool / not under heavy concurrent load |
| Xcode | 16+ with the QuickReply iOS app target (or a host app embedding `QuickReplyEngine`) |
| Model | On-device MLX base that fits the device (recommendation for first pass: a **0.5–0.6B 4-bit** MLX community build, **not** a 4B phone run). Pre-download into the app container; **no network during the night task** |
| Code | `NoOpTrainRunner` **replaced** with a real `Trainer` + MLX model loader before the overnight run |

### Info.plist / capabilities

1. Background Modes → **Processing**
2. `BGTaskSchedulerPermittedIdentifiers` includes:
   ```
   ai.adapt.pipeline.nightly
   ```
   (or your custom id matching `QuickReplyConfiguration.backgroundTaskIdentifier`)

## Protocol

### Night 0 — install and seed

1. Install the app on the physical device (not Simulator for the overnight claim).
2. Launch once, unlock, confirm bootstrap status shows BGProcessingTask registered.
3. Capture **≥ 40** short reply examples (mix of acceptance + explicit edits).  
   Use real short messages you would actually send.
4. Optionally force one **foreground** pipeline run to confirm prune/sample/eval
   wiring and that capability/device policy do not immediately refuse on this hardware.
5. Note buffer `exampleCount` and that `AdaptCapability.deviceCanTrain` is true.

### Night 1 — charger overnight

1. Plug into power with battery **> 40%**.
2. Leave the app; do not force-quit.
3. Keep the device offline or airplane mode **optional but preferred** to prove
   `requiresNetworkConnectivity = false` (model must already be on disk).
4. Leave overnight (≥ 6 hours wall clock so the system has room to schedule
   `BGProcessingTask`).

### Morning — inspect artifacts

On device (or via Xcode container download), check Application Support:

```
QuickReply/
  registry/<lineageID>/
    state.json              # activeVersion if promote fired
    vN/adapters.safetensors # candidate weights
    vN/version.json
    vN/train_state.json     # if training ran far enough to checkpoint
    held_out_pin.json       # if eval pinned a set
    backoff_state.json      # if gate refused
  buffer/<lineageID>/buffer.sqlite
```

Record:

| Field | Value |
|---|---|
| Device model + iOS version | |
| Physical RAM | |
| Base model id | |
| Examples in buffer before night | |
| `BGProcessingTask` fired? (Console / `BGTaskScheduler` logs) | yes / no / unknown |
| Candidate versions created | |
| `TrainOutcome` stop reason (if logged) | maxSteps / maxWallClock / thermal / cancelled / noData |
| Steps completed | |
| Wall time of train stage | |
| Gate decision | promote / refuse / abstain / pinBroken / n/a |
| Active version after night | |
| Any thermal abort? | |

### Budget measurement (the M4 point)

§4.6 defaults are **not** phone measurements. After the run, fill:

| Budget half | Configured | Observed binder (which hit first?) |
|---|---|---|
| `maxSteps` (default 300) | | |
| `maxWallClock` (default 8 min) | | |

Mac reference only (not a phone): ~15 s / 100 rank-8 steps over Qwen3-4B-4bit
attention-only. Phones will differ; write the **phone** numbers you measure.

If training never starts, note which gate fired:

- `insufficientMemory`
- `notCharging` / `batteryTooLow`
- `thermalStateTooHigh`
- `backoffActive`
- System never scheduled `BGProcessingTask` (OS discretion — common)

## What CI already verified (do not re-claim as device results)

- `AdaptPipeline` stage order, skip semantics, cancellation boundaries
- Refuse feeds backoff; abstain does not
- Prune can break pins; eval reports `pinBroken` and can clear for re-pin
- Thermal / battery / capability gates with **simulated** environment
- Package builds for the declared platforms (macOS CI host)

## What remains pending until you run this protocol

- Real `BGProcessingTask` scheduling latency on a specific device
- Per-step MLX cost on that device for the chosen base model
- Whether 300 steps / 8 minutes is meaningful on that hardware
- MLX Metal library availability in an **App Store / device** build (test-time
  metallib generation on Mac is not evidence)

## Simulator note

The iOS **Simulator** is not a substitute for this protocol. Metal training
behaviour and background task scheduling differ enough that a green Simulator
run does **not** satisfy M4’s device acceptance.
