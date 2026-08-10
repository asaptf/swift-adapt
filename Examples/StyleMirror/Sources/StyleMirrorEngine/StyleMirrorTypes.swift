import AdaptCore
import Foundation

// MARK: - Training

/// Wall-clock and step budget for a demo training run.
///
/// `duration` controls how long progress events are spread in real time so the
/// UI can develop against a ~20 s run and rehearse against a ~2.5 min run.
public struct TrainingConfiguration: Sendable, Equatable, Hashable {
    /// PRNG seed for the loss curve (and any training-side shuffle).
    public var seed: UInt64
    /// Total optimizer steps to simulate.
    public var totalSteps: Int
    /// Wall-clock duration over which steps are emitted.
    public var duration: Duration
    /// Emit a validation loss every N training steps (0 = never).
    public var validationInterval: Int

    /// Creates a training configuration.
    ///
    /// - Parameters:
    ///   - seed: Deterministic seed for the scripted curve.
    ///   - totalSteps: Number of progress steps (default 120).
    ///   - duration: Real-time span of the stream (default 20 s for UI work).
    ///   - validationInterval: Steps between validation points (default 10).
    public init(
        seed: UInt64 = 42,
        totalSteps: Int = 120,
        duration: Duration = .seconds(20),
        validationInterval: Int = 10
    ) {
        self.seed = seed
        self.totalSteps = totalSteps
        self.duration = duration
        self.validationInterval = validationInterval
    }

    /// Rehearsal preset: roughly 2.5 minutes, dense enough for a live curve.
    public static let rehearsal = TrainingConfiguration(
        seed: 42,
        totalSteps: 150,
        duration: .seconds(150),
        validationInterval: 10
    )

    /// Fast preset for UI development and automated tests.
    public static let uiDevelopment = TrainingConfiguration(
        seed: 42,
        totalSteps: 40,
        duration: .seconds(20),
        validationInterval: 5
    )

    /// Instant preset for unit tests (no intentional wall-clock delay).
    public static let unitTest = TrainingConfiguration(
        seed: 42,
        totalSteps: 24,
        duration: .milliseconds(0),
        validationInterval: 4
    )
}

/// One observation on the live training curve.
public struct TrainingProgress: Sendable, Equatable, Hashable {
    /// 1-based step index.
    public let step: Int
    /// Configured total steps for this run.
    public let totalSteps: Int
    /// Training loss at this step.
    public let loss: Double
    /// Held-out / validation loss when sampled; `nil` on pure training steps.
    public let validationLoss: Double?
    /// Approximate tokens processed per second.
    public let tokensPerSecond: Double
    /// Wall time since the run started.
    public let elapsed: Duration
    /// Estimated remaining wall time, if known.
    public let estimatedRemaining: Duration?
    /// Terminal flag: stream has completed (success or cancel).
    public let isFinished: Bool
    /// `true` when the consumer cancelled; cancellation is a normal outcome.
    public let wasCancelled: Bool
    /// Gate result on a **successful** terminal event only.
    ///
    /// Non-`nil` when `isFinished && !wasCancelled`: the candidate was evaluated
    /// and (on the scripted path) promoted. Cancelled or in-flight steps leave
    /// this `nil` and never change the active adapter.
    public let gateOutcome: GateOutcome?

    /// Creates a progress snapshot.
    public init(
        step: Int,
        totalSteps: Int,
        loss: Double,
        validationLoss: Double? = nil,
        tokensPerSecond: Double,
        elapsed: Duration,
        estimatedRemaining: Duration? = nil,
        isFinished: Bool = false,
        wasCancelled: Bool = false,
        gateOutcome: GateOutcome? = nil
    ) {
        self.step = step
        self.totalSteps = totalSteps
        self.loss = loss
        self.validationLoss = validationLoss
        self.tokensPerSecond = tokensPerSecond
        self.elapsed = elapsed
        self.estimatedRemaining = estimatedRemaining
        self.isFinished = isFinished
        self.wasCancelled = wasCancelled
        self.gateOutcome = gateOutcome
    }

    /// Fraction of steps completed in `0...1`.
    public var fractionComplete: Double {
        guard totalSteps > 0 else { return 1 }
        return min(1, Double(step) / Double(totalSteps))
    }
}

// MARK: - Blind test

/// Role of a reply candidate in the blind test.
public enum ReplyRole: String, Sendable, Codable, Hashable, CaseIterable {
    /// Stock base model (no adapter).
    case baseModel
    /// Model with the active personalization adapter.
    case adaptedModel
    /// The user's real archived reply (ground truth).
    case human
}

/// An email message used as corpus or blind-test context.
public struct EmailMessage: Sendable, Equatable, Hashable, Identifiable, Codable {
    public let id: String
    public let subject: String
    public let body: String
    /// BCP-47-ish language tag used by the demo (`en`, `es`, `ru`).
    public let language: String
    public let fromDisplayName: String
    public let toDisplayName: String

    public init(
        id: String,
        subject: String,
        body: String,
        language: String,
        fromDisplayName: String,
        toDisplayName: String
    ) {
        self.id = id
        self.subject = subject
        self.body = body
        self.language = language
        self.fromDisplayName = fromDisplayName
        self.toDisplayName = toDisplayName
    }
}

/// One of three reply options shown to the audience (identity hidden until reveal).
public struct BlindCandidate: Sendable, Equatable, Hashable, Identifiable {
    /// Opaque ID; the only handle the UI uses until reveal.
    public let id: UUID
    /// Reply body text.
    public let body: String

    public init(id: UUID = UUID(), body: String) {
        self.id = id
        self.body = body
    }
}

/// A prepared blind-test round: one incoming email and three shuffled candidates.
///
/// The engine owns the mapping from candidate ID → ``ReplyRole``. The UI must
/// not invent or assume roles; call ``StyleMirrorEngine/submitBlindGuess(roundID:candidateID:)``
/// and read the reveal from the result.
public struct BlindTestRound: Sendable, Equatable, Identifiable {
    public let id: UUID
    /// Stable corpus key for the incoming email.
    public let incomingEmailID: String
    public let incoming: EmailMessage
    /// Candidates in display order (already shuffled for this seed).
    public let candidates: [BlindCandidate]

    public init(
        id: UUID = UUID(),
        incomingEmailID: String,
        incoming: EmailMessage,
        candidates: [BlindCandidate]
    ) {
        self.id = id
        self.incomingEmailID = incomingEmailID
        self.incoming = incoming
        self.candidates = candidates
    }
}

/// Outcome of one audience guess.
public struct BlindTestGuessResult: Sendable, Equatable {
    public let roundID: UUID
    public let guessedCandidateID: UUID
    /// Role of the candidate the audience picked.
    public let guessedRole: ReplyRole
    /// Role that was the true human reply.
    public let humanCandidateID: UUID
    /// `true` when the audience correctly identified the human reply.
    public let identifiedHuman: Bool
    /// `true` when the audience picked the adapted model (mistook it for human).
    public let adapterMistakenForHuman: Bool
    /// Full reveal: every candidate ID → its true role.
    public let reveal: [UUID: ReplyRole]
    /// Running tally after this round.
    public let tally: BlindTestTally

    public init(
        roundID: UUID,
        guessedCandidateID: UUID,
        guessedRole: ReplyRole,
        humanCandidateID: UUID,
        identifiedHuman: Bool,
        adapterMistakenForHuman: Bool,
        reveal: [UUID: ReplyRole],
        tally: BlindTestTally
    ) {
        self.roundID = roundID
        self.guessedCandidateID = guessedCandidateID
        self.guessedRole = guessedRole
        self.humanCandidateID = humanCandidateID
        self.identifiedHuman = identifiedHuman
        self.adapterMistakenForHuman = adapterMistakenForHuman
        self.reveal = reveal
        self.tally = tally
    }
}

/// Aggregate blind-test scoreboard across rounds in a session.
public struct BlindTestTally: Sendable, Equatable, Hashable {
    /// Number of completed rounds.
    public let roundsPlayed: Int
    /// Times the audience correctly picked the human reply.
    public let humanCorrectlyIdentified: Int
    /// Times the audience picked the adapted model as "human".
    public let adapterMistakenForHuman: Int
    /// Times the audience picked the base model as "human".
    public let baseMistakenForHuman: Int

    public static let zero = BlindTestTally(
        roundsPlayed: 0,
        humanCorrectlyIdentified: 0,
        adapterMistakenForHuman: 0,
        baseMistakenForHuman: 0
    )

    public init(
        roundsPlayed: Int,
        humanCorrectlyIdentified: Int,
        adapterMistakenForHuman: Int,
        baseMistakenForHuman: Int
    ) {
        self.roundsPlayed = roundsPlayed
        self.humanCorrectlyIdentified = humanCorrectlyIdentified
        self.adapterMistakenForHuman = adapterMistakenForHuman
        self.baseMistakenForHuman = baseMistakenForHuman
    }
}

// MARK: - Code-switching

/// Demo languages for the code-switching scene.
public enum DemoLanguage: String, Sendable, Codable, Hashable, CaseIterable {
    case english = "en"
    case spanish = "es"
    case russian = "ru"

    /// Short display label suitable for a column header.
    public var displayName: String {
        switch self {
        case .english: return "English"
        case .spanish: return "Spanish"
        case .russian: return "Russian"
        }
    }
}

/// Base vs. adapted replies for one language.
public struct CodeSwitchLanguageResult: Sendable, Equatable, Hashable {
    public let language: DemoLanguage
    public let baseReply: String
    public let adaptedReply: String

    public init(language: DemoLanguage, baseReply: String, adaptedReply: String) {
        self.language = language
        self.baseReply = baseReply
        self.adaptedReply = adaptedReply
    }
}

/// Full code-switching scene payload for one shared request.
///
/// Success and unavailability are distinct. An empty `languages` array is never
/// a silent success: when the engine cannot produce pairs, `unavailabilityReason`
/// names why. Callers must not treat empty languages as "nothing to show" without
/// reading the reason.
public struct CodeSwitchResult: Sendable, Equatable {
    /// Shared intent / request summarized for the audience.
    public let requestSummary: String
    /// Per-language base vs. adapted pairs (typically en / es / ru).
    public let languages: [CodeSwitchLanguageResult]
    /// Non-`nil` when the demo could not produce language pairs.
    ///
    /// When set, `languages` is empty. When `nil`, `languages` is non-empty on a
    /// successful path.
    public let unavailabilityReason: String?

    /// Whether the result carries base/adapted pairs to display.
    public var isAvailable: Bool {
        unavailabilityReason == nil && !languages.isEmpty
    }

    public init(
        requestSummary: String,
        languages: [CodeSwitchLanguageResult],
        unavailabilityReason: String? = nil
    ) {
        self.requestSummary = requestSummary
        self.languages = languages
        self.unavailabilityReason = unavailabilityReason
    }

    /// Builds an explicit unavailable result (never a silent empty success).
    public static func unavailable(
        requestSummary: String,
        reason: String
    ) -> CodeSwitchResult {
        CodeSwitchResult(
            requestSummary: requestSummary,
            languages: [],
            unavailabilityReason: reason
        )
    }
}

// MARK: - Generation progress

/// One unit of multi-step generation work (blind test, code-switching).
///
/// Mirrors training's completed/total shape so the UI can show "step 3 of 6"
/// without inventing a percentage. Emitted in order; the final event has
/// `completed == total`.
public struct GenerationProgress: Sendable, Equatable, Hashable {
    /// Units finished so far (0…`total`).
    public let completed: Int
    /// Total units in this operation.
    public let total: Int
    /// Optional label for the unit just finished or about to run
    /// (e.g. `"english / base"`, `"base model"`).
    public let unitLabel: String?

    public init(completed: Int, total: Int, unitLabel: String? = nil) {
        self.completed = completed
        self.total = total
        self.unitLabel = unitLabel
    }

    /// Fraction complete in `0...1` when `total > 0`.
    public var fractionComplete: Double {
        guard total > 0 else { return 1 }
        return min(1, Double(completed) / Double(total))
    }
}

/// Callback for multi-step generation progress (blind test, code-switch).
///
/// **Async by design.** The engine `await`s each invocation before starting the
/// next unit of work, so a UI handler can hop to the MainActor and paint
/// determinate counts mid-generation. A fire-and-forget hop (`Task { @MainActor
/// in … }` without an await) is not sufficient: long Metal work can starve the
/// hop until the whole operation finishes, leaving the indicator indeterminate.
public typealias GenerationProgressHandler =
    @Sendable (GenerationProgress) async -> Void

// MARK: - Poisoning / eval gate

/// Metric that caused a promotion gate failure (or the primary metric on pass).
public struct GateMetric: Sendable, Equatable, Hashable {
    /// Stable machine name, e.g. `"held_out_perplexity"`.
    public let name: String
    /// Human-readable label for the UI.
    public let displayName: String
    /// Candidate's measured value.
    public let candidateValue: Double
    /// Incumbent (active) value when relevant.
    public let incumbentValue: Double?
    /// Threshold the candidate had to satisfy.
    public let threshold: Double
    /// Whether lower is better for this metric.
    public let lowerIsBetter: Bool

    public init(
        name: String,
        displayName: String,
        candidateValue: Double,
        incumbentValue: Double? = nil,
        threshold: Double,
        lowerIsBetter: Bool = true
    ) {
        self.name = name
        self.displayName = displayName
        self.candidateValue = candidateValue
        self.incumbentValue = incumbentValue
        self.threshold = threshold
        self.lowerIsBetter = lowerIsBetter
    }
}

/// First-class promotion-gate verdict — not an error.
///
/// Shared by the live-training happy path (`promoted == true`) and the
/// poisoning refusal (`promoted == false`). One type, two states.
public struct GateVerdict: Sendable, Equatable, Hashable {
    /// Whether the candidate was promoted to active.
    public let promoted: Bool
    /// Metric that failed (or the primary metric when promoted).
    public let primaryMetric: GateMetric
    /// Short audience-facing explanation.
    public let reason: String

    public init(promoted: Bool, primaryMetric: GateMetric, reason: String) {
        self.promoted = promoted
        self.primaryMetric = primaryMetric
        self.reason = reason
    }
}

/// Shared outcome of running the promotion gate against a trained candidate.
///
/// Used by both:
/// - a completed live training run (``verdict.promoted`` is `true`, active advances), and
/// - the poisoning scenario (``verdict.promoted`` is `false`, active unchanged).
///
/// Cancelled / unfinished training never produces a ``GateOutcome`` and never
/// mutates the active adapter.
public struct GateOutcome: Sendable, Equatable, Hashable {
    public let verdict: GateVerdict
    /// Adapter that was active when the gate ran.
    public let activeVersionBefore: AdapterVersion
    /// Adapter active after the gate (advanced on pass, same as before on refuse).
    public let activeVersionAfter: AdapterVersion
    /// Candidate that was evaluated (promoted or refused).
    public let candidate: AdapterVersion

    public init(
        verdict: GateVerdict,
        activeVersionBefore: AdapterVersion,
        activeVersionAfter: AdapterVersion,
        candidate: AdapterVersion
    ) {
        self.verdict = verdict
        self.activeVersionBefore = activeVersionBefore
        self.activeVersionAfter = activeVersionAfter
        self.candidate = candidate
    }
}

/// Historical name for ``GateOutcome`` (poisoning path). Prefer ``GateOutcome``.
@available(*, deprecated, renamed: "GateOutcome")
public typealias PoisoningOutcome = GateOutcome

// MARK: - Recorded comparison / rollback / best-vs-active

/// Result of a real registry rollback (O(1) pointer flip).
///
/// Elapsed time is measured, not staged — the demo shows this number.
public struct RollbackResult: Sendable, Equatable, Hashable {
    /// Adapter that was active before the rollback.
    public let fromVersion: AdapterVersion
    /// Adapter that is active after the rollback.
    public let toVersion: AdapterVersion
    /// Wall time for the registry pointer flip (including integrity check).
    public let elapsed: Duration

    public init(
        fromVersion: AdapterVersion,
        toVersion: AdapterVersion,
        elapsed: Duration
    ) {
        self.fromVersion = fromVersion
        self.toVersion = toVersion
        self.elapsed = elapsed
    }
}

/// Whether the active adapter is the best one we have measured.
///
/// For held-out cross-entropy (lower is better): a positive ``gapNats`` means
/// the active version is worse than the best recorded version by that many nats.
/// Ties set ``isActiveBest`` to `true` and ``gapNats`` to ~0.
public struct ActiveVersusBest: Sendable, Equatable, Hashable {
    /// Currently active adapter.
    public let active: AdapterVersion
    /// Version with the best recorded held-out CE (lowest nats). On a tie that
    /// includes the active version, this is the active version.
    public let bestMeasured: AdapterVersion
    /// Active version's recorded mean CE (nats/token).
    public let activeScore: Double
    /// Best recorded mean CE (nats/token).
    public let bestScore: Double
    /// `true` when the active score is equal to the best (including ties).
    public let isActiveBest: Bool
    /// `activeScore - bestScore` (positive ⇒ active is worse under lower-is-better).
    public let gapNats: Double

    public init(
        active: AdapterVersion,
        bestMeasured: AdapterVersion,
        activeScore: Double,
        bestScore: Double,
        isActiveBest: Bool,
        gapNats: Double
    ) {
        self.active = active
        self.bestMeasured = bestMeasured
        self.activeScore = activeScore
        self.bestScore = bestScore
        self.isActiveBest = isActiveBest
        self.gapNats = gapNats
    }
}

// MARK: - Network / offline

/// Coarse reachability for the airplane-mode scene.
public enum NetworkStatus: String, Sendable, Equatable, Hashable {
    case online
    case offline
}
