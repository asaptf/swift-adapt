import AdaptCore
import Foundation

/// Engine seam driven by the StyleMirror UI.
///
/// Covers the five demo scenes: live training, version timeline, blind test,
/// code-switching, and the poisoning / eval-gate refusal. The UI depends only
/// on this protocol. Two implementations:
/// - ``ScriptedEngine`` — offline deterministic mock (tests, `--scripted` fallback).
/// - ``AdaptEngine`` — real ``Trainer`` + ``AdaptSession`` over a seeded registry.
///
/// All associated types used at the boundary are `Sendable`. Training progress
/// is an `AsyncStream` so the UI can `for await` on the main actor. Cancellation
/// of the consuming task is a **normal** terminal outcome (`wasCancelled == true`),
/// not a thrown error.
public protocol StyleMirrorEngine: Sendable {
    // MARK: Corpus

    /// Synthetic sent-mail training corpus (~30 examples).
    var sentEmails: [EmailMessage] { get async }

    /// Incoming emails that have a prepared three-way blind-test set.
    var blindTestIncomingIDs: [String] { get async }

    // MARK: Training

    /// Starts a training run over `examples` and streams progress until finish or cancel.
    ///
    /// On a **successful** finish (`isFinished && !wasCancelled`), the engine runs the
    /// same promotion gate used by ``runPoisoningScenario()``: the final
    /// ``TrainingProgress/gateOutcome`` is non-`nil`, ``GateVerdict/promoted`` is
    /// `true`, and ``activeVersion()`` advances to the new candidate. A **cancelled**
    /// or unfinished run yields no outcome, promotes nothing, and leaves the active
    /// version untouched — cancellation is a normal terminal outcome, not an error.
    ///
    /// - Parameters:
    ///   - examples: Training examples (typically derived from ``sentEmails``).
    ///   - configuration: Step count, wall-clock duration, and seed.
    /// - Returns: An `AsyncStream` of ``TrainingProgress``. The final element has
    ///   `isFinished == true`. If the consumer cancels, the stream ends with a
    ///   finished event where `wasCancelled == true` when the engine can observe it.
    func train(
        examples: [TrainingExample],
        configuration: TrainingConfiguration
    ) -> AsyncStream<TrainingProgress>

    // MARK: Version timeline

    /// Adapter versions for the "seven nights" timeline (typically v1…v7).
    ///
    /// Reuses ``AdapterVersion`` / ``TrainingWindow`` / ``EvalReport`` from AdaptCore.
    func adapterVersions() async -> [AdapterVersion]

    /// Currently active adapter, if any.
    func activeVersion() async -> AdapterVersion?

    // MARK: Blind test

    /// Prepares a round: loads the incoming email, builds three candidates, shuffles
    /// with the engine seed (reproducible), and retains the role map internally.
    ///
    /// - Parameter incomingEmailID: Corpus key from ``blindTestIncomingIDs``.
    func prepareBlindRound(incomingEmailID: String) async throws -> BlindTestRound

    /// Records an audience guess and returns the reveal plus updated tally.
    ///
    /// - Parameters:
    ///   - roundID: ID from ``BlindTestRound/id``.
    ///   - candidateID: ID of the chosen ``BlindCandidate``.
    func submitBlindGuess(roundID: UUID, candidateID: UUID) async throws -> BlindTestGuessResult

    /// Running blind-test scoreboard for this engine session.
    func blindTestTally() async -> BlindTestTally

    // MARK: Code-switching

    /// Same request answered in each of the user's languages, base vs. adapted.
    func codeSwitchingDemo() async -> CodeSwitchResult

    // MARK: Poisoning / eval gate

    /// Runs the poisoned-corpus pipeline. Produces a gate **refusal** with the
    /// failing metric and leaves the active version unchanged.
    ///
    /// Returns the same ``GateOutcome`` type as a successful training run's
    /// ``TrainingProgress/gateOutcome`` — one verdict shape, two states
    /// (`promoted` true after live train, false after poison).
    ///
    /// This is a legible result, not a thrown error — the system is protecting the user.
    func runPoisoningScenario() async -> GateOutcome
}
