# StyleMirror: the demo, screen by screen

`Examples/StyleMirror` is a macOS SwiftUI app built to be performed live in about five minutes. It runs on the real library: training, generation and the adapter registry are the ones in `Sources/`. If the model or the seeded registry is missing it falls back to a scripted engine, and says so — the status strip marks the run `SCRIPTED` in red, because which engine is running changes what every number on screen means.

The version history comes from `scripts/seed-demo-registry.sh`, which trains seven adapters in seven separate processes, each resuming from the one before. Nothing in the registry is a fixture.

## Act 1: airplane mode

The presenter switches on airplane mode in front of the audience before anything else happens. The app treats offline as the feature, not a degraded state, and a "0 B sent" counter stays on screen for the rest of the demo. That counter is a real measurement, not a printed zero: the app routes all outbound traffic through a single chokepoint, and it never calls it. The screen's argument: you don't have to trust a privacy policy when there is no network path to police.

## The night that made it worse

This is the screen the demo exists for, and it was not planned. Seeding the registry with seven real overnight runs produced this held-out loss curve: `4.06 → 3.70 → 3.38 → 3.42 → 3.19 → 3.12 → 3.34`. Night seven came out **0.22 nats worse than night six**, on ordinary mail, with nothing sabotaged — and it became the active adapter anyway, because promotion is manual until the evaluation module exists.

So the timeline at the bottom is evidence rather than decoration: worse loss sits higher, and night seven kinks upward. The panel states what a gate would have decided, from the measurements already recorded in the registry.

Then it is undone. Rollback is a pointer flip in the registry — no weights are rewritten — and the screen prints the duration it measured rather than a number someone typed. A staged failure invites the suspicion that it was staged; this one is in the data, and the fix is a capability that already exists.

The original scene — twenty examples of ALL-CAPS pirate slang, refused by the same comparison — is still there behind the second tab. It reads better once the audience has watched the check catch something real.

## Not ready yet: the blind test and the multilingual screen

Two screens are built and wired to the real engine but are **not** presentable, and the honest thing is to say why rather than photograph them at a flattering moment.

**The blind test** generates its three candidates on demand. In the app that takes about 30 seconds even in a release build, while the same work measured 1.9 seconds in a smoke test against a warm model — so something in the app's path is paying a cost the smoke test does not. The screen shows a progress indicator with a live clock while it waits, but the per-step counts the engine emits are not reaching it yet, so the bar cannot go determinate.

**Code-switching** renders correctly and generates for real, and the result does not support the claim the screen makes. English is passable. Spanish repeats itself and breaks off mid-word; Russian degenerates into a loop that is not a sentence. The base model answers sensibly in all three. With 20% of the training corpus in each non-English language and a rank-8 adapter, one adapter is not holding three voices — and the generation path does not apply a repetition penalty. Both are fixable; neither is fixed.

Until they are, there is no screenshot of them here. A demo that photographs its worst screen at its best instant is the thing this project is trying not to be.
