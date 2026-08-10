/// Marks a stored property as model **input** (the prompt side of a training pair).
///
/// Used only as a compile-time marker read by ``Personalizable(task:)``. Expansion
/// produces no peer declarations.
@attached(peer)
public macro Prompt() = #externalMacro(module: "AdaptMacrosPlugin", type: "PromptMacro")

/// Marks a stored property as model **output** (the completion side of a training pair).
///
/// A ``Personalizable(task:)`` type must have **exactly one** `@Completion` property.
/// Expansion produces no peer declarations.
@attached(peer)
public macro Completion() = #externalMacro(module: "AdaptMacrosPlugin", type: "CompletionMacro")

/// Synthesizes a ``PersonalizationSignal`` conformance for a personalization task.
///
/// Apply to a `struct` whose fields are annotated with ``Prompt()`` and exactly one
/// ``Completion()``. The macro generates:
///
/// - `static var personalizationTaskID: String` (the `task` argument)
/// - `makeTrainingExample(source:weight:id:capturedAt:)` assembling prompt/completion
///
/// Capture into an ``ReplayBuffer`` (scrub + privacy budget) is provided by
/// ``PersonalizationSignal/capture(_:into:source:weight:)`` on the protocol extension —
/// scheduled off the caller's hot path as a `Task`.
///
/// ## Compile-time checks
///
/// - Applied only to a `struct` (class/enum/actor diagnose with a fix-it to `struct`)
/// - Exactly one `@Completion`
/// - At least one `@Prompt`
/// - Annotated properties must be of type `String`
///
/// ## Example
///
/// ```swift
/// @Personalizable(task: "email-style")
/// struct EmailDraft {
///     @Prompt var context: String
///     @Completion var body: String
/// }
///
/// let task = EmailDraft.capture(draft, into: buffer, source: .explicitEdit)
/// _ = try await task.value
/// ```
@attached(
    extension,
    conformances: PersonalizationSignal,
    names: named(personalizationTaskID), named(makeTrainingExample)
)
public macro Personalizable(task: String) =
    #externalMacro(module: "AdaptMacrosPlugin", type: "PersonalizableMacro")
