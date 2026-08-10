# StyleMirror

Flagship **macOS demo** for [Adapt](../..): train a LoRA adapter on a user’s own sent email style, entirely on-device — staged here as a **model layer** the UI plugs into.

> **This package ships the engine seam + scripted mock + corpus. No SwiftUI.**
> A separate human-written UI drives `StyleMirrorEngine`. Do not look for views here.

## Five scenes

1. **Airplane mode** — `NetworkReachability` streams online/offline; `OutboundTrafficMeter.bytesSent` stays at 0 because the app never originates requests (see below).
2. **Live training** — paste ~30 synthetic sent mails, run `train`; consume an `AsyncStream` of loss / tokens/sec / ETA for a live curve.
3. **Version timeline** — `adapterVersions()` returns v1…v7 (`AdapterVersion` from AdaptCore) with rising eval scores (“seven nights”).
4. **Blind test** — three candidates (base / adapted / human); engine owns identity + seeded shuffle; audience guess → reveal + tally.
5. **Code-switching + poisoning** — same voice in EN/ES/RU; poisoned ALL-CAPS pirate buffer is **refused** by the eval gate without changing the active adapter.

## Scripted vs real

| Piece | Now | Later (M5+) |
|---|---|---|
| `ScriptedEngine` | Deterministic mock (`SeededGenerator` loss noise, canned replies) | Replace with type backed by `AdaptTrain` + `AdaptInference` |
| Loss curve | Fake but noisy, wall-clock paced | Real MLX LoRA steps |
| Blind / code-switch text | `SampleCorpus` literals | Generated with base vs active adapter |
| Timeline / poisoning | Hard-coded AdaptCore metadata + gate verdict | `AdaptRegistry` + `AdaptEval` |

The type is named **`ScriptedEngine` on purpose** so nobody confuses it with on-device learning.

## Package layout

```
Examples/StyleMirror/          # own SwiftPM package (does not edit root Package.swift)
  Package.swift
  Sources/
    StyleMirrorEngine/         # library: protocol, scripted impl, corpus, offline helpers
    StyleMirror/               # executable placeholder (human replaces with SwiftUI @main)
  Tests/
    StyleMirrorEngineTests/
```

Depends on root package path `../..` → product **`AdaptCore` only**.

## Build & test

```bash
cd Examples/StyleMirror
swift build
swift test
```

Platform: **macOS 15+**, Swift 6.

## Drive the engine from UI code

```swift
import StyleMirrorEngine
import AdaptCore

let engine: any StyleMirrorEngine = ScriptedEngine(seed: 42)

// Training (UI development: ~20s; rehearsal: TrainingConfiguration.rehearsal ~150s)
let examples = SampleCorpus.trainingExamples()
for await progress in engine.train(examples: examples, configuration: .uiDevelopment) {
    // progress.loss, .validationLoss, .tokensPerSecond, .estimatedRemaining, .fractionComplete
    if progress.isFinished { break }
}

// Timeline
let versions = await engine.adapterVersions()
let active = await engine.activeVersion()

// Blind test
let round = try await engine.prepareBlindRound(incomingEmailID: "in-en-sprint")
// show round.candidates (shuffled; no roles)
let result = try await engine.submitBlindGuess(roundID: round.id, candidateID: chosenID)
// result.reveal, result.tally.adapterMistakenForHuman, …

// Code-switch + poison
let polyglot = await engine.codeSwitchingDemo()
let poison = await engine.runPoisoningScenario()  // verdict.promoted == false

// Offline chrome
for await status in NetworkReachability().updates { /* .online / .offline */ }
let bytes = await OutboundTrafficMeter.shared.bytesSent  // structural 0
```

Full protocol surface is documented on `StyleMirrorEngine` and the public types in `StyleMirrorTypes.swift`.

## Offline / bytes-sent honesty

- **`NetworkReachability`** — `NWPathMonitor` only; no outbound data.
- **`OutboundTrafficMeter`** — sole chokepoint that would count app-originated bytes. **No StyleMirror code calls `recordOutbound`.** The UI should bind to `bytesSent`, not hardcode `"0"`. If a future feature sends traffic, it must record here.

## Synthetic corpus

Persona: **Renna Vale** at fictional **Harborfinch** (`*.example` addresses). Correspondents and companies are invented. Voice is short, direct, em-dash heavy, sign-off `— renna`; base-model foils are over-polite boilerplate so the blind test has contrast.

## Placeholder executable

`Sources/StyleMirror/main.swift` only prints that the UI is not attached. Replace with the SwiftUI app; keep depending on `StyleMirrorEngine`.
