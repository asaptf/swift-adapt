# AdaptData

On-device capture store: scrubbed replay buffer, per-lineage privacy budget, TTL
pruning (architecture §4.2).

## Contract

- **Actor-isolated.** All mutation goes through `ReplayBuffer`. SQLite is never
  shared across threads outside that actor.
- **Scrub at capture.** The configured `ScrubberPipeline` runs **before** any
  INSERT. Unscrubbed caller text is never written to the database file. What
  remains is still the user's prose and is treated as sensitive throughout.
- **Bounded + deduplicated.** Capacity (`maxExamples`) and content-hash
  deduplication within a lineage. Duplicates do not consume budget.
- **Privacy budget.** Per-lineage, per UTC calendar day capture cap
  (`maxCapturesPerDay`). Enforced under concurrent capture by actor
  serialisation + a transactional counter.
- **TTL pruning.** Default retention 30 days. Expired examples are **deleted
  even if never trained on**, and **even if referenced by a held-out pin**.
- **Observable pruning.** Every prune that deletes rows records a durable
  `PruneEvent` (IDs + cutoff + reason) and returns a `PruneResult`. Nothing
  silent: M3's gate can detect a broken pin and re-pin.
- **No network. No MLX.** Depends only on `AdaptCore` and system SQLite3.

## Seams (M2)

| Consumer | Protocol | How |
|---|---|---|
| AdaptTrain | `TrainingDataSource` | `extension ReplayBuffer: TrainingDataSource` in AdaptTrain |
| AdaptEval | `HeldOutExampleSource` | `extension ReplayBuffer: HeldOutExampleSource` in AdaptEval |

Both protocols require `examples() async throws -> [TrainingExample]`. The buffer
implements that method; empty extensions attach the conformances so AdaptData
does not import MLX (via AdaptTrain) and does not create a cycle.

Existing `ArrayTrainingData` / `ArrayHeldOutSource` remain for tests and CLI.

## On-disk schema (SQLite, `PRAGMA user_version = 1`)

```
examples(
  id TEXT PRIMARY KEY,          -- UUID
  lineage_id TEXT NOT NULL,
  prompt TEXT NOT NULL,         -- scrubbed
  completion TEXT NOT NULL,     -- scrubbed
  weight REAL NOT NULL,
  captured_at REAL NOT NULL,    -- timeIntervalSince1970
  source TEXT NOT NULL,         -- SignalSource raw value
  content_hash TEXT NOT NULL,   -- SHA-256 hex of prompt\0completion
  trained_count INTEGER NOT NULL DEFAULT 0
)
UNIQUE(lineage_id, content_hash)

privacy_budget(
  lineage_id TEXT NOT NULL,
  day TEXT NOT NULL,             -- YYYY-MM-DD UTC
  count INTEGER NOT NULL,
  PRIMARY KEY (lineage_id, day)
)

prune_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lineage_id TEXT NOT NULL,
  pruned_at REAL NOT NULL,
  cutoff REAL NOT NULL,
  reason TEXT NOT NULL,         -- ttl | capacity | manual
  deleted_ids TEXT NOT NULL     -- JSON array of UUID strings
)
```

Journal mode is `DELETE` (no WAL companion files) so a single-file byte scan is
meaningful in tests. Additive migrations bump `user_version`; this binary
creates v1 and can still **read** v2 (forward window) after an additive column
migration.

## Platform protection (encryption at rest)

| Platform | What Adapt applies | What you actually get |
|---|---|---|
| **iOS** | `FileProtectionType.completeUntilFirstUserAuthentication` on the database file | After first unlock post-boot, the file is encrypted at rest with a key tied to the device passcode. Inaccessible from cold boot until first unlock. |
| **macOS** | **No-op** — Apple does not offer the same per-file Data Protection classes | Confidentiality at rest depends on **FileVault** (full-disk encryption) when the user has enabled it. Adapt does **not** claim parity with iOS file-class encryption on macOS. |

There is **no SQLCipher** and no additional app-level ciphertext layer. "Encrypted
at rest" in §4.2 means Data Protection file classes where the OS provides them.

## Scrubbing limits (mitigation, not a guarantee)

Built-in pipeline order: email → IBAN → payment card → phone
(IBAN/card before phone so phone patterns do not carve up their digit groups).

| Scrubber | Catches | Does **not** catch (known gaps) |
|---|---|---|
| `EmailScrubber` | `user@host.tld`, light spacing, simple `at` / `dot` spoken forms | HTML entities (`&#64;`), zero-width joiners, unicode homoglyph domains, free-form paraphrase |
| `PhoneNumberScrubber` | 10–15 digit runs with common separators or single-space digit split | Word-only numbers, ambiguous digit IDs without phone-like grouping |
| `CreditCardScrubber` | 13–19 digit PANs that **pass Luhn** (optional spaces/dashes) | Invalid-Luhn digit runs (left intact on purpose), letter-mixed PANs, unicode digit look-alikes |
| `IBANScrubber` | ISO 13616-shaped IBANs with/without spaces that pass mod-97 | Odd hyphenation, country codes written as words |

Architecture §7 still lists **adapter memorisation of PII** as a live residual
risk. Rank limits and (eventually) canary eval carry the rest of the load.
Overstating scrubbing would be the worst place to be optimistic.

Adversarial fixtures in `AdaptDataTests` document which attacks the scrubbers
defeat and which defeat them — known gaps stay as green tests that assert the
limitation, not as deleted cases.

## TTL vs pinned held-out sets

**Pruning wins.** M3 pins a held-out set per lineage and reports `.pinBroken`
when pinned examples vanish. TTL **will** delete those examples. Exempting
held-out IDs from expiry would break §4.2's commitment that examples are deleted
after the retention window even if never trained on.

Consequences (also for the gate):

1. Prune is observable (`PruneResult` + durable `PruneEvent` log).
2. The gate must re-pin after breakage.
3. **Comparisons across a re-pin boundary are not directly comparable** — the
   yardstick changed.

## Public surface (summary)

| Symbol | Role |
|---|---|
| `ReplayBuffer` | Actor: `add` / `sample` / `prune` / `examples` / `stats` / `wipe` |
| `ReplayBufferConfiguration` | Capacity, retention, daily budget, scrubbers |
| `Scrubber` / `ScrubberPipeline` | Capture-time mitigation pipeline |
| `EmailScrubber`, `PhoneNumberScrubber`, `CreditCardScrubber`, `IBANScrubber` | Built-ins |
| `SamplingStrategy` | Stratified-by-source+recency or uniform |
| `BufferStats`, `PruneResult`, `PruneEvent`, `PruneReason` | Observability |
| `AdaptDataError` | Typed errors (`LocalizedError`) |

## Offline tests

```bash
swift test --filter AdaptDataTests
```

Property-style checks: raw PII byte-scan of the DB file after capture; concurrent
budget hammer; adversarial scrubber fixtures; stratified mix; TTL observability;
schema migration round-trip. Network-free and model-free.
