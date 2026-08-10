# AdaptEval

On-device evaluation and promotion gate for adapter versions (architecture §4.5 redesign).

## Contract

A candidate becomes active only when this module says so. The gate:

1. **Pins a held-out set per lineage** — chosen once from a persisted seed,
   stratified by `SignalSource` and recency. Every version of a lineage is
   scored on the **same** examples.
2. **Stores the pin beside the lineage** (`held_out_pin.json` next to
   `state.json`), **not** in the replay buffer. §4.2 prunes buffer examples
   after 30 days; a pin that lived only in the buffer would silently change the
   yardstick. Missing pinned IDs are **detected and reported**, never ignored.
3. **Abstains below a floor** — at least `minHeldOut` (default 30) examples, in
   addition to the 10–20% share rule. Abstention is **not** refusal and must
   not feed §4.6 exponential backoff. The type system enforces the distinction
   (`GateDecision.promote | .refuse | .abstain`).
4. **Compares pairs** — candidate and incumbent scores on the same examples;
   never two independent averages.
5. **Wilcoxon signed-rank** — one-sided, default `alpha = 0.05`. The report
   carries p-value, W⁺, and rank-biserial effect size so a significant-but-
   negligible gain is visible.
6. **Units are nats** — regression bounds are absolute mean per-token
   cross-entropy nats (`maxCrossEntropyRegressionNats`), not perplexity ratios.

## Where scoring lives

**Statistics and policy are pure Swift** in this module (no MLX). Per-example
scoring needs a forward pass; that lives in **AdaptTrain** as
`MLXPerExampleCrossEntropyScorer`, next to `Trainer.llmCompletionLoss`, so the
completion mask matches training and mlx imports stay confined to
AdaptTrain / AdaptInference (§7).

Inject any `PerExampleScorer` for tests (see `ClosurePerExampleScorer`).

## M2 seam

`HeldOutExampleSource` / `ArrayHeldOutSource` mirror `AdaptTrain`'s
`TrainingDataSource` / `ArrayTrainingData`. M2's `ReplayBuffer` will satisfy
the protocol without reshaping the gate.

## Public surface (summary)

| Symbol | Role |
|---|---|
| `PromotionPolicy` | Floors, alpha, CE regression bound |
| `PromotionGate` | Pure decide(candidate, incumbent) → `GateDecision` |
| `PromotionEvaluator` | Pin → resolve → score → decide |
| `GateDecision` | `.promote` / `.refuse` / `.abstain` |
| `EvaluationResult` | `.decided` or `.pinBroken` |
| `HeldOutPin` / `HeldOutPinStore` | Durable yardstick |
| `HeldOutSelector` | Stratified selection + resolve |
| `WilcoxonSignedRank` | One-sided signed-rank test |
| `PerExampleScorer` | Scoring seam (MLX impl in AdaptTrain) |
| `AdaptEvalError` | Typed errors |

## Offline tests

```bash
swift test --filter AdaptEvalTests
```

Wilcoxon is verified against hand-computed oracles (ties, small n), not smoke
assertions. The gate suite covers refuse / promote / abstain type distinction,
broken pins, and noise (identical adapters must not produce a significant win).
