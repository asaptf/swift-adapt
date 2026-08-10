import AdaptCore
import AdaptData
import Foundation

/// A type that can produce a training signal for one personalization task.
///
/// Conformance is normally synthesized by ``Personalizable(task:)``. Host apps
/// may also conform manually when the macro is not a fit.
///
/// ## Capture path
///
/// ``capture(_:into:source:weight:)`` builds a ``TrainingExample`` and schedules
/// ``ReplayBuffer/add(_:)`` on a detached task so scrubbing, privacy-budget
/// checks, and SQLite I/O do not block the caller's hot path. Scrubbing always
/// happens inside the buffer — never in the macro expansion.
public protocol PersonalizationSignal: Sendable {
    /// Task identifier matching ``AdapterLineage/taskID`` for this signal.
    static var personalizationTaskID: String { get }

    /// Builds a ``TrainingExample`` from this instance's prompt/completion fields.
    ///
    /// Does **not** scrub text. Scrubbing is the buffer's responsibility at
    /// capture time so unscrubbed caller text is never persisted.
    ///
    /// - Parameters:
    ///   - source: How the example was obtained (drives default weight).
    ///   - weight: Optional importance override; `nil` uses `source.defaultWeight`.
    ///   - id: Example identity (defaulted by convenience overloads).
    ///   - capturedAt: Capture timestamp.
    func makeTrainingExample(
        source: SignalSource,
        weight: Double?,
        id: UUID,
        capturedAt: Date
    ) -> TrainingExample
}

extension PersonalizationSignal {
    /// Builds a training example with a fresh id and current timestamp.
    public func makeTrainingExample(
        source: SignalSource = .explicitEdit,
        weight: Double? = nil
    ) -> TrainingExample {
        makeTrainingExample(
            source: source,
            weight: weight,
            id: UUID(),
            capturedAt: Date()
        )
    }

    /// Schedules capture of `value` into `buffer` off the caller's hot path.
    ///
    /// Returns a `Task` that:
    /// 1. Builds a ``TrainingExample`` from the annotated fields
    /// 2. Calls ``ReplayBuffer/add(_:)``, which scrubs, enforces the daily
    ///    privacy budget, deduplicates, and persists
    ///
    /// Await `task.value` when you need the ``ReplayBuffer/AddResult`` or to
    /// surface ``AdaptDataError/privacyBudgetExceeded``. Discard the task for
    /// fire-and-forget capture.
    ///
    /// - Parameters:
    ///   - value: Instance whose `@Prompt` / `@Completion` fields form the pair.
    ///   - buffer: Lineage buffer that owns scrubbing and budget.
    ///   - source: Signal taxonomy value (default `.explicitEdit`).
    ///   - weight: Optional importance override.
    @discardableResult
    public static func capture(
        _ value: Self,
        into buffer: ReplayBuffer,
        source: SignalSource = .explicitEdit,
        weight: Double? = nil
    ) -> Task<ReplayBuffer.AddResult, Error> {
        Task {
            let example = value.makeTrainingExample(source: source, weight: weight)
            return try await buffer.add(example)
        }
    }
}
