import AdaptCore
import Foundation

// =============================================================================
// SCRIPTED ENGINE — NOT REAL TRAINING
// =============================================================================
//
// This type is a **deterministic mock** of the StyleMirror demo engine. It exists
// so the UI and live demo can be built, reviewed, and rehearsed before
// `AdaptTrain` and `AdaptInference` land (M5).
//
// What it fakes:
//   • Loss curve + tokens/sec (SeededGenerator noise over a decaying trend)
//   • Wall-clock paced progress events (no MLX, no GPU)
//   • Blind-test replies (canned corpus text + seeded shuffle)
//   • Code-switching replies (canned multi-language pairs)
//   • Version timeline v1…v7 with rising eval scores (v7 active at 74)
//   • Successful train → gate pass, promote v8 (eval 75), shared GateOutcome
//   • Poisoning pipeline → same GateOutcome type, refusal, active unchanged
//
// What a future real backend replaces (same protocol, different type):
//   • `train` → AdaptTrain step loop over real examples / MLX LoRA
//   • Blind / code-switch generation → AdaptInference with base vs. active adapter
//   • Version timeline + poisoning → AdaptRegistry + AdaptEval promotion gate
//
// Name is intentionally honest: do not present ScriptedEngine as on-device learning.
// =============================================================================

/// Deterministic, seeded ``StyleMirrorEngine`` for UI development and live rehearsal.
///
/// Thread-safe via an internal actor. Same `seed` ⇒ identical loss curves and
/// identical blind-test shuffles.
public final class ScriptedEngine: StyleMirrorEngine, Sendable {
    private let state: State
    private let seed: UInt64
    private let lineage: AdapterLineage
    /// When non-`nil`, ``codeSwitchingDemo(progress:)`` returns an unavailable
    /// result with this reason instead of canned languages (offline failure path).
    private let codeSwitchFailureReason: String?

    /// Creates a scripted engine.
    ///
    /// - Parameters:
    ///   - seed: Master seed for loss curves and blind-test shuffles.
    ///   - codeSwitchFailureReason: When set, code-switching returns
    ///     ``CodeSwitchResult/unavailable(requestSummary:reason:)`` so tests can
    ///     exercise the explicit-failure path without a model.
    public init(seed: UInt64 = 42, codeSwitchFailureReason: String? = nil) {
        self.seed = seed
        self.codeSwitchFailureReason = codeSwitchFailureReason
        self.lineage = AdapterLineage(
            taskID: "email-style",
            baseModelID: "mlx-community/Qwen3-0.6B-4bit",
            loraConfig: LoRAConfig(rank: 8, scale: 10.0, numLayers: 16)
        )
        let versions = Self.makeTimeline(lineage: lineage, seed: seed)
        self.state = State(versions: versions, seed: seed)
    }

    // MARK: StyleMirrorEngine

    public var sentEmails: [EmailMessage] {
        get async { SampleCorpus.sentEmails }
    }

    public var blindTestIncomingIDs: [String] {
        get async { SampleCorpus.blindRounds.map(\.incoming.id) }
    }

    public func train(
        examples: [TrainingExample],
        configuration: TrainingConfiguration
    ) -> AsyncStream<TrainingProgress> {
        let totalSteps = max(1, configuration.totalSteps)
        let duration = configuration.duration
        let seed = configuration.seed
        let validationInterval = configuration.validationInterval
        let exampleCount = max(1, examples.count)
        let state = self.state
        let lineage = self.lineage
        let engineSeed = self.seed

        return AsyncStream { continuation in
            let task = Task {
                let curve = Self.makeLossCurve(
                    steps: totalSteps,
                    seed: seed,
                    validationInterval: validationInterval
                )
                let stepDelay = Self.delayPerStep(duration: duration, steps: totalSteps)
                let started = ContinuousClock.now

                for step in 1...totalSteps {
                    if Task.isCancelled {
                        // Cancelled runs promote nothing and carry no gate outcome.
                        let elapsed = started.duration(to: .now)
                        continuation.yield(
                            TrainingProgress(
                                step: step - 1,
                                totalSteps: totalSteps,
                                loss: step > 1 ? curve[step - 2].loss : curve[0].loss,
                                validationLoss: nil,
                                tokensPerSecond: 0,
                                elapsed: elapsed,
                                estimatedRemaining: .zero,
                                isFinished: true,
                                wasCancelled: true,
                                gateOutcome: nil
                            )
                        )
                        continuation.finish()
                        return
                    }

                    if stepDelay > .zero {
                        try? await Task.sleep(for: stepDelay)
                    }

                    if Task.isCancelled {
                        let elapsed = started.duration(to: .now)
                        let point = curve[step - 1]
                        continuation.yield(
                            TrainingProgress(
                                step: step,
                                totalSteps: totalSteps,
                                loss: point.loss,
                                validationLoss: point.validationLoss,
                                tokensPerSecond: 0,
                                elapsed: elapsed,
                                estimatedRemaining: .zero,
                                isFinished: true,
                                wasCancelled: true,
                                gateOutcome: nil
                            )
                        )
                        continuation.finish()
                        return
                    }

                    let point = curve[step - 1]
                    let elapsed = started.duration(to: .now)
                    let remainingSteps = totalSteps - step
                    let estimatedRemaining: Duration? = remainingSteps > 0 && stepDelay > .zero
                        ? stepDelay * remainingSteps
                        : (remainingSteps == 0 ? .zero : nil)
                    // Plausible tokens/sec: scales with example volume, mild noise from seed.
                    let baseTPS = 180.0 + Double(exampleCount % 40)
                    let tpsJitter = Double((seed &+ UInt64(step) &* 17) % 40)
                    let tokensPerSecond = baseTPS + tpsJitter

                    let finished = step == totalSteps
                    // Successful completion: same gate type as poisoning, but it passes
                    // and the new candidate becomes active (v7 → v8 on the first live run).
                    let outcome: GateOutcome? = if finished {
                        await state.promoteAfterTraining(lineage: lineage, seed: engineSeed)
                    } else {
                        nil
                    }

                    continuation.yield(
                        TrainingProgress(
                            step: step,
                            totalSteps: totalSteps,
                            loss: point.loss,
                            validationLoss: point.validationLoss,
                            tokensPerSecond: tokensPerSecond,
                            elapsed: elapsed,
                            estimatedRemaining: estimatedRemaining,
                            isFinished: finished,
                            wasCancelled: false,
                            gateOutcome: outcome
                        )
                    )
                }

                continuation.finish()
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    public func adapterVersions() async -> [AdapterVersion] {
        await state.versions
    }

    public func activeVersion() async -> AdapterVersion? {
        await state.activeVersion
    }

    public func prepareBlindRound(
        incomingEmailID: String,
        progress: GenerationProgressHandler?
    ) async throws -> BlindTestRound {
        guard let fixture = SampleCorpus.blindRounds.first(where: { $0.incoming.id == incomingEmailID }) else {
            throw StyleMirrorError.notFound("blind round '\(incomingEmailID)'")
        }
        // Scripted path: two synthetic units (base, adapter), human is corpus.
        let total = 2
        progress?(GenerationProgress(completed: 0, total: total, unitLabel: "base model"))
        progress?(GenerationProgress(completed: 1, total: total, unitLabel: "base model"))
        progress?(GenerationProgress(completed: 2, total: total, unitLabel: "adapter"))
        return await state.prepareBlindRound(fixture: fixture)
    }

    public func submitBlindGuess(roundID: UUID, candidateID: UUID) async throws -> BlindTestGuessResult {
        try await state.submitBlindGuess(roundID: roundID, candidateID: candidateID)
    }

    public func blindTestTally() async -> BlindTestTally {
        await state.tally
    }

    public func codeSwitchingDemo(progress: GenerationProgressHandler?) async -> CodeSwitchResult {
        let request = SampleCorpus.codeSwitch.requestSummary
        if let reason = codeSwitchFailureReason {
            return .unavailable(requestSummary: request, reason: reason)
        }
        let canned = SampleCorpus.codeSwitch
        let total = DemoLanguage.allCases.count * 2
        var completed = 0
        progress?(GenerationProgress(completed: 0, total: total, unitLabel: "loading"))
        for language in DemoLanguage.allCases {
            let name = language.displayName.lowercased()
            completed += 1
            progress?(
                GenerationProgress(completed: completed, total: total, unitLabel: "\(name) / base")
            )
            completed += 1
            progress?(
                GenerationProgress(completed: completed, total: total, unitLabel: "\(name) / adapter")
            )
        }
        return canned
    }

    public func runPoisoningScenario() async -> GateOutcome {
        await state.runPoisoning(lineage: lineage)
    }

    public func compareRecordedVersions(
        candidateVersion: Int,
        incumbentVersion: Int
    ) async throws -> GateOutcome {
        try await state.compareRecordedVersions(
            candidateVersion: candidateVersion,
            incumbentVersion: incumbentVersion
        )
    }

    public func rollbackToVersion(_ version: Int) async throws -> RollbackResult {
        try await state.rollbackToVersion(version)
    }

    public func restoreDemoStartingState() async throws {
        await state.restoreDemoStartingState(startingVersion: Self.demoStartingActiveVersion)
    }

    public func activeVersusBestMeasured() async -> ActiveVersusBest? {
        await state.activeVersusBestMeasured()
    }

    /// Opening active version for the scripted seven-night timeline (v7).
    public static let demoStartingActiveVersion = 7

    /// Test helper: replace the in-memory timeline (e.g. with CE-measured versions).
    func installVersionsForTesting(_ versions: [AdapterVersion]) async {
        await state.replaceVersions(versions)
    }

    // MARK: - Loss curve (scripted but noisy)

    struct LossPoint: Sendable, Equatable {
        var loss: Double
        var validationLoss: Double?
    }

    /// Builds a decaying loss series with step noise, plateaus, and lagged validation.
    ///
    /// Same `seed` always yields the same series. Intentionally **not** a clean
    /// exponential — that would look fake on stage.
    static func makeLossCurve(
        steps: Int,
        seed: UInt64,
        validationInterval: Int
    ) -> [LossPoint] {
        var rng = SeededGenerator(seed: seed &+ 0xC0FFEE)
        let startLoss = 2.85
        let floorLoss = 0.42
        let decay = 0.018

        var points: [LossPoint] = []
        points.reserveCapacity(steps)
        var current = startLoss
        var plateauLeft = 0

        for step in 1...steps {
            // Occasional plateau: hold loss nearly flat for 2–4 steps.
            if plateauLeft == 0 {
                let roll = Int(rng.next() % 100)
                if roll < 8, step > 5, step < steps - 3 {
                    plateauLeft = 2 + Int(rng.next() % 3)
                }
            }

            if plateauLeft > 0 {
                plateauLeft -= 1
                // Tiny jitter only.
                let micro = unitNoise(&rng) * 0.015
                current = max(floorLoss, current + micro)
            } else {
                let target = floorLoss + (startLoss - floorLoss) * exp(-decay * Double(step))
                // Pull toward target with noisy residual (heavier early, lighter late).
                let noiseScale = 0.12 * (0.35 + 0.65 * (1.0 - Double(step) / Double(steps)))
                let noise = unitNoise(&rng) * noiseScale
                // Mix previous with target so steps are autocorrelated, not IID.
                current = 0.55 * current + 0.45 * target + noise
                current = max(floorLoss * 0.9, current)
            }

            var validation: Double?
            if validationInterval > 0, step % validationInterval == 0 || step == steps {
                // Validation lags training slightly and is a bit higher.
                let lag = 0.04 + abs(unitNoise(&rng)) * 0.06
                validation = current + lag
            }

            points.append(LossPoint(loss: current, validationLoss: validation))
        }

        return points
    }

    /// Approx. standard normal via Box-Muller on seeded UInt64s.
    private static func unitNoise(_ rng: inout SeededGenerator) -> Double {
        let u1 = max(Double(rng.next() % 10_000) / 10_000.0, 1e-6)
        let u2 = Double(rng.next() % 10_000) / 10_000.0
        return sqrt(-2.0 * log(u1)) * cos(2.0 * Double.pi * u2)
    }

    static func delayPerStep(duration: Duration, steps: Int) -> Duration {
        guard steps > 0, duration > .zero else { return .zero }
        return duration / steps
    }

    // MARK: - Timeline

    static func makeTimeline(lineage: AdapterLineage, seed: UInt64) -> [AdapterVersion] {
        // Seven nights of overnight runs: style-match scores on held-out mail
        // (DESIGN.md §6.5 — 58…74, then live train promotes v8 at 75).
        let scores: [Double] = [58, 63, 66, 69, 71, 72, 74]
        let baseDate = Date(timeIntervalSince1970: 1_700_100_000)
        var versions: [AdapterVersion] = []

        for v in 1...7 {
            let start = baseDate.addingTimeInterval(Double(v - 1) * 86_400)
            let end = start.addingTimeInterval(86_400 - 1)
            let exampleCount = 18 + v * 4 + Int(seed % 3)
            let isActive = v == 7
            let digest = String(format: "%064x", seed &+ UInt64(v) &* 0x9E37)
            // Pad / trim to 64 hex chars for a plausible SHA-256 hex digest look.
            let weightsDigest = String(digest.prefix(64)).padding(toLength: 64, withPad: "0", startingAt: 0)

            let version = AdapterVersion(
                lineage: lineage,
                version: v,
                parentVersion: v == 1 ? nil : v - 1,
                trainedOn: TrainingWindow(start: start, end: end, exampleCount: exampleCount),
                evalReport: EvalReport(
                    primaryScore: scores[v - 1],
                    passedGate: true,
                    notes: "scripted timeline night \(v)",
                    // Style-match (higher is better) — not held-out CE. Keeps
                    // ``ProvisionalPromotionGate/heldOutCE(from:)`` from treating
                    // these as cross-entropy for the night-seven regression demo.
                    primaryMetric: "style_match",
                    primaryDirection: .higherIsBetter
                ),
                status: isActive ? .active : .archived,
                weightsDigest: weightsDigest,
                createdAt: end
            )
            versions.append(version)
        }
        return versions
    }
}

// MARK: - Session state

extension ScriptedEngine {
    /// Mutable session state for rounds, tally, and active version.
    actor State {
        var versions: [AdapterVersion]
        var tally: BlindTestTally = .zero
        private var openRounds: [UUID: OpenRound] = [:]
        private let seed: UInt64
        private var roundCounter: UInt64 = 0

        struct OpenRound {
            let roleByCandidate: [UUID: ReplyRole]
            let humanCandidateID: UUID
            var resolved: Bool
        }

        init(versions: [AdapterVersion], seed: UInt64) {
            self.versions = versions
            self.seed = seed
        }

        var activeVersion: AdapterVersion? {
            versions.first(where: { $0.status == .active })
        }

        func prepareBlindRound(fixture: SampleCorpus.BlindRoundFixture) -> BlindTestRound {
            roundCounter &+= 1
            var rng = SeededGenerator(seed: seed &+ roundCounter &* 0xA5A5)
            var tagged = fixture.bodiesByRole
            // Fisher–Yates with SeededGenerator.
            for i in stride(from: tagged.count - 1, through: 1, by: -1) {
                let j = Int(rng.next() % UInt64(i + 1))
                tagged.swapAt(i, j)
            }

            var roleByCandidate: [UUID: ReplyRole] = [:]
            var humanID: UUID?
            let candidates: [BlindCandidate] = tagged.map { role, body in
                // Deterministic UUID from seed + role + counter for reproducibility.
                let id = Self.deterministicID(seed: seed, round: roundCounter, role: role)
                roleByCandidate[id] = role
                if role == .human { humanID = id }
                return BlindCandidate(id: id, body: body)
            }
            // Fixtures always include exactly one human body.
            let resolvedHumanID = humanID ?? candidates[0].id

            let roundID = Self.deterministicID(seed: seed, round: roundCounter, role: nil)
            let round = BlindTestRound(
                id: roundID,
                incomingEmailID: fixture.incoming.id,
                incoming: fixture.incoming,
                candidates: candidates
            )
            openRounds[roundID] = OpenRound(
                roleByCandidate: roleByCandidate,
                humanCandidateID: resolvedHumanID,
                resolved: false
            )
            return round
        }

        func submitBlindGuess(roundID: UUID, candidateID: UUID) throws -> BlindTestGuessResult {
            guard var open = openRounds[roundID] else {
                throw StyleMirrorError.notFound("round \(roundID)")
            }
            guard !open.resolved else {
                throw StyleMirrorError.invalidState("round \(roundID) already scored")
            }
            guard let guessedRole = open.roleByCandidate[candidateID] else {
                throw StyleMirrorError.unknownCandidate(candidateID.uuidString)
            }

            open.resolved = true
            openRounds[roundID] = open

            let identifiedHuman = guessedRole == .human
            let adapterMistaken = guessedRole == .adaptedModel
            let baseMistaken = guessedRole == .baseModel

            tally = BlindTestTally(
                roundsPlayed: tally.roundsPlayed + 1,
                humanCorrectlyIdentified: tally.humanCorrectlyIdentified + (identifiedHuman ? 1 : 0),
                adapterMistakenForHuman: tally.adapterMistakenForHuman + (adapterMistaken ? 1 : 0),
                baseMistakenForHuman: tally.baseMistakenForHuman + (baseMistaken ? 1 : 0)
            )

            return BlindTestGuessResult(
                roundID: roundID,
                guessedCandidateID: candidateID,
                guessedRole: guessedRole,
                humanCandidateID: open.humanCandidateID,
                identifiedHuman: identifiedHuman,
                adapterMistakenForHuman: adapterMistaken,
                reveal: open.roleByCandidate,
                tally: tally
            )
        }

        /// Successful live training: gate passes, candidate becomes active.
        ///
        /// First Act 2 run promotes v8 (eval 75) over v7 (eval 74). Subsequent
        /// successful runs promote `active + 1` with a score one point above the
        /// incumbent so the chart keeps rising deterministically.
        func promoteAfterTraining(lineage: AdapterLineage, seed: UInt64) -> GateOutcome {
            let before = activeVersion
                ?? versions.last
                ?? ScriptedEngine.makeTimeline(lineage: lineage, seed: seed).last!

            let incumbentScore = before.evalReport?.primaryScore ?? 74
            let candidateScore = incumbentScore + 1
            let nextVersionNumber = before.version + 1
            let windowEnd = Date(timeIntervalSince1970: 1_700_100_000 + Double(nextVersionNumber) * 86_400)
            let digest = String(format: "%064x", seed &+ UInt64(nextVersionNumber) &* 0x9E37)
            let weightsDigest = String(digest.prefix(64)).padding(toLength: 64, withPad: "0", startingAt: 0)

            let candidate = AdapterVersion(
                lineage: lineage,
                version: nextVersionNumber,
                parentVersion: before.version,
                trainedOn: TrainingWindow(
                    start: windowEnd.addingTimeInterval(-86_400),
                    end: windowEnd,
                    exampleCount: SampleCorpus.sentEmails.count
                ),
                evalReport: EvalReport(
                    primaryScore: candidateScore,
                    passedGate: true,
                    notes: "live training run — gate passed"
                ),
                status: .active,
                weightsDigest: weightsDigest,
                createdAt: windowEnd
            )

            let metric = GateMetric(
                name: "style_match",
                displayName: "Style match",
                candidateValue: candidateScore,
                incumbentValue: incumbentScore,
                threshold: incumbentScore,
                lowerIsBetter: false
            )
            let verdict = GateVerdict(
                promoted: true,
                primaryMetric: metric,
                reason: """
                Gate: passed. v\(nextVersionNumber) promoted — eval \
                \(Int(candidateScore)) against v\(before.version)'s \(Int(incumbentScore)).
                """
            )

            // Archive prior active; append (or replace) the new active candidate.
            versions = versions.map { version in
                version.status == .active ? version.with(status: .archived) : version
            }
            if let idx = versions.firstIndex(where: { $0.version == nextVersionNumber }) {
                versions[idx] = candidate
            } else {
                versions.append(candidate)
            }

            return GateOutcome(
                verdict: verdict,
                activeVersionBefore: before,
                activeVersionAfter: candidate,
                candidate: candidate
            )
        }

        /// Provisional verdict from stored measurements (no re-measure).
        func compareRecordedVersions(
            candidateVersion: Int,
            incumbentVersion: Int
        ) throws -> GateOutcome {
            guard let candidate = versions.first(where: { $0.version == candidateVersion }) else {
                throw StyleMirrorError.notFound("version v\(candidateVersion)")
            }
            guard let incumbent = versions.first(where: { $0.version == incumbentVersion }) else {
                throw StyleMirrorError.notFound("version v\(incumbentVersion)")
            }
            guard let outcome = ProvisionalPromotionGate.evaluateRecorded(
                candidate: candidate,
                activeBefore: incumbent
            ) else {
                throw StyleMirrorError.invalidState(
                    "v\(candidateVersion) has no recorded held-out CE measurement to compare"
                )
            }
            return outcome
        }

        /// In-memory pointer flip (scripted stand-in for registry rollback).
        func rollbackToVersion(_ version: Int) throws -> RollbackResult {
            guard let from = activeVersion else {
                throw StyleMirrorError.invalidState("no active adapter to roll back from")
            }
            guard let target = versions.first(where: { $0.version == version }) else {
                throw StyleMirrorError.notFound("version v\(version)")
            }
            let started = ContinuousClock.now
            versions = versions.map { entry in
                if entry.version == version {
                    return entry.with(status: .active)
                }
                if entry.status == .active {
                    return entry.with(status: .rolledBack)
                }
                return entry
            }
            let elapsed = started.duration(to: .now)
            let to = versions.first(where: { $0.version == version }) ?? target.with(status: .active)
            return RollbackResult(fromVersion: from, toVersion: to, elapsed: elapsed)
        }

        /// Restores the demo's opening active version (v7). Does not fabricate data.
        func restoreDemoStartingState(startingVersion: Int) {
            guard versions.contains(where: { $0.version == startingVersion }) else {
                return
            }
            versions = versions.map { entry in
                if entry.version == startingVersion {
                    return entry.with(status: .active)
                }
                if entry.status == .active {
                    return entry.with(status: .archived)
                }
                return entry
            }
        }

        func activeVersusBestMeasured() -> ActiveVersusBest? {
            ProvisionalPromotionGate.activeVersusBest(versions: versions, active: activeVersion)
        }

        /// Replace the timeline with versions that carry held-out CE reports
        /// (test helper for recorded-comparison coverage).
        func replaceVersions(_ newVersions: [AdapterVersion]) {
            versions = newVersions
        }

        /// Poisoned corpus: same ``GateOutcome`` type, but the gate refuses.
        ///
        /// Candidate version is always `active + 1` (v9 after a successful Act 2
        /// that left v8 active; v8 if the presenter skips Act 2 and still has v7).
        /// Active version is never mutated.
        func runPoisoning(lineage: AdapterLineage) -> GateOutcome {
            let before = activeVersion
                ?? versions.last
                ?? ScriptedEngine.makeTimeline(lineage: lineage, seed: seed).last!

            let incumbentScore = before.evalReport?.primaryScore ?? 74
            let windowEnd = Date(timeIntervalSince1970: 1_700_800_000)
            let candidate = AdapterVersion(
                lineage: lineage,
                version: before.version + 1,
                parentVersion: before.version,
                trainedOn: TrainingWindow(
                    start: windowEnd.addingTimeInterval(-86_400),
                    end: windowEnd,
                    exampleCount: SampleCorpus.poisonedCompletions.count
                ),
                evalReport: EvalReport(
                    primaryScore: 41,
                    passedGate: false,
                    notes: "poisoned buffer — gate refused promotion"
                ),
                status: .candidate,
                weightsDigest: String(repeating: "a", count: 64),
                createdAt: windowEnd
            )

            let metric = GateMetric(
                name: "held_out_perplexity",
                displayName: "Held-out perplexity",
                candidateValue: 14.8,
                incumbentValue: 3.1,
                threshold: 3.1 * 1.02,
                lowerIsBetter: true
            )
            let verdict = GateVerdict(
                promoted: false,
                primaryMetric: metric,
                reason: """
                Candidate held-out perplexity 14.8 exceeds incumbent 3.1 by more than the \
                2% regression allowance. Poisoned ALL-CAPS pirate slang did not promote; \
                active adapter remains v\(before.version) (eval \(Int(incumbentScore))).
                """
            )

            // Active version intentionally unchanged — versions list keeps prior active.
            // Candidate is returned for UI (hollow timeline node) but not stored as active.
            return GateOutcome(
                verdict: verdict,
                activeVersionBefore: before,
                activeVersionAfter: before,
                candidate: candidate
            )
        }

        private static func deterministicID(seed: UInt64, round: UInt64, role: ReplyRole?) -> UUID {
            var bytes = [UInt8](repeating: 0, count: 16)
            var x = seed ^ (round &* 0x9E3779B97F4A7C15)
            if let role {
                x ^= UInt64(role.rawValue.utf8.reduce(0) { ($0 &<< 5) &+ $0 &+ UInt64($1) })
            } else {
                x ^= 0xDEADBEEF
            }
            for i in 0..<8 {
                bytes[i] = UInt8((x >> (i * 8)) & 0xff)
            }
            let y = x &* 0xBF58476D1CE4E5B9
            for i in 0..<8 {
                bytes[8 + i] = UInt8((y >> (i * 8)) & 0xff)
            }
            bytes[6] = (bytes[6] & 0x0f) | 0x40
            bytes[8] = (bytes[8] & 0x3f) | 0x80
            return UUID(uuid: (
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
                bytes[8], bytes[9], bytes[10], bytes[11],
                bytes[12], bytes[13], bytes[14], bytes[15]
            ))
        }
    }
}
