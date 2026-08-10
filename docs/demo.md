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

**The blind test does not finish in the app.** Its three candidates are generated on demand. A smoke harness measures that work at 3.8 seconds cold and 1.5 warm; the shipped app, release build, was still generating at **91 seconds** with the elapsed clock running. Caching the generation session fixed the harness and did not fix the app, so the two paths differ in something not yet identified — and at 91 seconds this is not a cold model load, which was my first guess and was wrong. The screen shows an honest indicator with a live clock, but the per-unit counts the engine emits still do not reach it, so it cannot go determinate.

**Code-switching** is no longer performed, and its claim has been retired rather than tuned. The screen renders and generates for real; the result does not support "one adapter learned your voice in every language". English is fine. Spanish becomes coherent once a repetition penalty is applied. Russian degenerates under every sampling setting tried — temperature 0 to 0.4, top-p 0.85 to 0.9, penalty 1.2 to 1.35 — which changes the failure mode without producing a voice. A rank-8 adapter over a corpus with roughly a fifth of its examples per non-English language is a capacity limit, and fixing it means re-training the seven nights and re-deriving every number in this document. That was not worth doing for one screenshot, so the screen now shows the real output and states the boundary.

Session caching and a yield between progress events fixed both symptoms **in the smoke harness**. Neither fixed the app. That gap is the open defect, and it is being investigated in the app's own path rather than in a harness — a fix verified only where the bug does not reproduce is not a fix.
