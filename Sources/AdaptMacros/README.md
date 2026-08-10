# AdaptMacros

Ergonomic compile-time entry point for capture (architecture §4.9).

## Contract

- **`@Personalizable(task:)`** on a `struct` synthesizes ``PersonalizationSignal``.
- **`@Prompt` / `@Completion`** are markers: exactly one `@Completion`, ≥1 `@Prompt`.
- **Serialisable types:** annotated properties must be `String` (compile-time diagnostic otherwise).
- **Capture goes through `AdaptData`.** ``PersonalizationSignal/capture(_:into:source:weight:)`` schedules ``ReplayBuffer/add(_:)`` on a `Task` so scrubbing, privacy budget, and SQLite I/O are off the caller's hot path. The macro does **not** invent a parallel store.
- **No MLX.** Depends only on `AdaptCore`, `AdaptData`, and the `AdaptMacrosPlugin` compiler plugin.
- **swift-syntax 602–603** (same range as `mlx-swift-lm`).

## Generated surface

```swift
@Personalizable(task: "email-style")
struct EmailDraft {
    @Prompt var context: String
    @Completion var body: String
}

// Synthesized:
extension EmailDraft: PersonalizationSignal {
    static var personalizationTaskID: String { "email-style" }
    func makeTrainingExample(source:weight:id:capturedAt:) -> TrainingExample
}

// Protocol extension (not re-emitted per type):
EmailDraft.capture(_:into:source:weight:) -> Task<ReplayBuffer.AddResult, Error>
```

Multiple `@Prompt` fields are joined with `"\n\n"` in declaration order.

## Compile-time diagnostics

| Condition | Message intent |
|---|---|
| Applied to class / enum / actor | Say the kind; fix-it: change keyword to `struct` |
| Zero `@Completion` | Require exactly one; say how to mark output |
| Two+ `@Completion` | List the properties; keep one, rest → `@Prompt` |
| Zero `@Prompt` | Require at least one input field |
| Non-`String` annotated property | Name the property and type; ask for `String` |
| Missing type annotation | Name the property; require explicit `String` |

## Offline tests

```bash
swift test --filter AdaptMacrosTests
```

Expansion + diagnostic assertions use `SwiftSyntaxMacrosTestSupport`. Capture-path
integration tests exercise scrubbing and privacy budget against a real
`ReplayBuffer` (model-free).
