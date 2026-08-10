# QuickReply

iOS skeleton for Adapt M4: background personalization of short reply suggestions.

## What this is

- Captures sent/edited replies into `AdaptData.ReplayBuffer`
- Registers a `BGProcessingTask` via `AdaptSchedule` (`requiresExternalPower = true`,
  `requiresNetworkConnectivity = false`)
- Runs `AdaptPipeline` (prune → sample → train → eval → promote)

## What this is not

- **Not** a green CI proof that LoRA trains on iPhone. That is the **manual
  device protocol** in [`TESTING.md`](TESTING.md).
- Default train runner is `NoOpTrainRunner` so the skeleton builds without
  bundling a multi-GB model. Swap in a real `Trainer` for the device night.

## Build (library / host)

```bash
cd Examples/QuickReply
swift build
```

For a full iOS app with `BGTaskSchedulerPermittedIdentifiers`, open an Xcode
iOS app target that depends on `QuickReplyEngine` and list:

```
ai.adapt.pipeline.nightly
```

under **Background Modes → Processing** and Info.plist
`BGTaskSchedulerPermittedIdentifiers`.

## Stay out of StyleMirror

This example is independent of `Examples/StyleMirror` / `Sources/StyleMirror*`.
