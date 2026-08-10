import AdaptCore
import Foundation
import Testing
@testable import StyleMirrorEngine

@Suite("BlindReplyPrompt")
struct BlindReplyPromptTests {
    @Test("base and adapter prompt construction share one path and identical text")
    func sharedPromptEquality() {
        let incoming = SampleCorpus.blindRounds[0].incoming
        let a = BlindReplyPrompt.generationPrompt(for: incoming)
        let b = BlindReplyPrompt.generationPrompt(for: incoming)
        #expect(a == b)
        #expect(a.contains(BlindReplyPrompt.sharedInstruction))
        #expect(a.contains("\(BlindReplyPrompt.minWords)–\(BlindReplyPrompt.maxWords) words"))
        #expect(a.contains(incoming.subject))
        #expect(a.contains(incoming.body.trimmingCharacters(in: .whitespacesAndNewlines)))
        // No dual call sites: the shared instruction is the only length constraint.
        #expect(a.components(separatedBy: BlindReplyPrompt.sharedInstruction).count == 2)
    }

    @Test("code-switch prompts share instruction and differ only by language name")
    func codeSwitchSharedInstruction() {
        let request = "Decline a meeting."
        let en = BlindReplyPrompt.codeSwitchPrompt(requestSummary: request, language: .english)
        let es = BlindReplyPrompt.codeSwitchPrompt(requestSummary: request, language: .spanish)
        #expect(en.contains(BlindReplyPrompt.sharedInstruction))
        #expect(es.contains(BlindReplyPrompt.sharedInstruction))
        #expect(en.contains("English"))
        #expect(es.contains("Spanish"))
        #expect(en.contains(request))
        #expect(es.contains(request))
    }

    @Test("word count and length class helpers")
    func lengthClassHelpers() {
        let short = "one two three"
        #expect(BlindReplyPrompt.wordCount(short) == 3)
        #expect(!BlindReplyPrompt.isInLengthClass(short))

        let words = (0..<50).map { "w\($0)" }.joined(separator: " ")
        #expect(BlindReplyPrompt.wordCount(words) == 50)
        #expect(BlindReplyPrompt.isInLengthClass(words))
        #expect(BlindReplyPrompt.lengthClassDeviation(words: 50) == 0)
        #expect(BlindReplyPrompt.lengthClassDeviation(words: 10) == 30)
        #expect(BlindReplyPrompt.lengthClassDeviation(words: 100) == 20)
    }
}

@Suite("BlindRoundSupport")
struct BlindRoundSupportTests {
    @Test("seeded shuffle is deterministic and roles stay complete")
    func shuffleDeterministic() {
        let fixture = SampleCorpus.blindRounds[0]
        let bodies = fixture.bodiesByRole
        let a = BlindRoundSupport.prepareRound(
            incomingEmailID: fixture.incoming.id,
            incoming: fixture.incoming,
            bodiesByRole: bodies,
            seed: 42,
            roundIndex: 1
        )
        let b = BlindRoundSupport.prepareRound(
            incomingEmailID: fixture.incoming.id,
            incoming: fixture.incoming,
            bodiesByRole: bodies,
            seed: 42,
            roundIndex: 1
        )
        #expect(a.round.candidates.map(\.id) == b.round.candidates.map(\.id))
        #expect(a.round.candidates.map(\.body) == b.round.candidates.map(\.body))
        #expect(Set(a.open.roleByCandidate.values) == Set(ReplyRole.allCases))
        #expect(a.open.roleByCandidate[a.open.humanCandidateID] == .human)
    }

    @Test("different seeds produce different orders")
    func shuffleDiffersBySeed() {
        let fixture = SampleCorpus.blindRounds[0]
        let a = BlindRoundSupport.prepareRound(
            incomingEmailID: fixture.incoming.id,
            incoming: fixture.incoming,
            bodiesByRole: fixture.bodiesByRole,
            seed: 1,
            roundIndex: 1
        )
        let b = BlindRoundSupport.prepareRound(
            incomingEmailID: fixture.incoming.id,
            incoming: fixture.incoming,
            bodiesByRole: fixture.bodiesByRole,
            seed: 2,
            roundIndex: 1
        )
        #expect(
            a.round.candidates.map(\.body) != b.round.candidates.map(\.body)
                || a.round.candidates.map(\.id) != b.round.candidates.map(\.id)
        )
    }

    @Test("scoreGuess tallies roles")
    func scoreGuess() throws {
        let fixture = SampleCorpus.blindRounds[0]
        let prepared = BlindRoundSupport.prepareRound(
            incomingEmailID: fixture.incoming.id,
            incoming: fixture.incoming,
            bodiesByRole: fixture.bodiesByRole,
            seed: 7,
            roundIndex: 3
        )
        let humanID = prepared.open.humanCandidateID
        let (result, tally) = try BlindRoundSupport.scoreGuess(
            open: prepared.open,
            roundID: prepared.round.id,
            candidateID: humanID,
            previousTally: .zero
        )
        #expect(result.identifiedHuman)
        #expect(tally.humanCorrectlyIdentified == 1)
        #expect(tally.roundsPlayed == 1)
    }
}

@Suite("ProvisionalPromotionGate")
struct ProvisionalPromotionGateTests {
    private func makeVersion(number: Int, ce: Double?) -> AdapterVersion {
        let lineage = AdapterLineage(
            taskID: "style-mirror",
            baseModelID: "mlx-community/Qwen3-4B-4bit",
            loraConfig: LoRAConfig(rank: 8, scale: 10, numLayers: 8)
        )
        return AdapterVersion(
            lineage: lineage,
            version: number,
            parentVersion: number > 1 ? number - 1 : nil,
            trainedOn: TrainingWindow(start: Date(), end: Date(), exampleCount: 10),
            evalReport: ce.map {
                EvalReport.heldOutCrossEntropy(
                    meanNats: $0,
                    exampleCount: 30,
                    supervisedTokenCount: 200
                )
            },
            status: number == 7 ? .active : .candidate,
            weightsDigest: String(repeating: "ab", count: 32)
        )
    }

    @Test("refuses when candidate CE is worse (higher) than incumbent")
    func refusesWorse() {
        let active = makeVersion(number: 7, ce: 2.2)
        let candidate = makeVersion(number: 8, ce: nil)
        let outcome = ProvisionalPromotionGate.evaluate(
            ProvisionalGateInput(
                candidateMeanCrossEntropyNats: 4.8,
                incumbentMeanCrossEntropyNats: 2.2,
                candidate: candidate,
                activeBefore: active
            )
        )
        #expect(outcome.verdict.promoted == false)
        #expect(outcome.activeVersionAfter.version == 7)
        #expect(outcome.candidate.status == .candidate)
        #expect(outcome.verdict.primaryMetric.name == ProvisionalPromotionGate.metricName)
        #expect(outcome.verdict.primaryMetric.candidateValue == 4.8)
        #expect(outcome.verdict.primaryMetric.incumbentValue == 2.2)
        #expect(outcome.verdict.reason.contains("Not the M3 gate"))
    }

    @Test("promotes when candidate CE is better (lower)")
    func promotesBetter() {
        let active = makeVersion(number: 7, ce: 2.5)
        let candidate = makeVersion(number: 8, ce: nil)
        let outcome = ProvisionalPromotionGate.evaluate(
            ProvisionalGateInput(
                candidateMeanCrossEntropyNats: 2.1,
                incumbentMeanCrossEntropyNats: 2.5,
                candidate: candidate,
                activeBefore: active
            )
        )
        #expect(outcome.verdict.promoted == true)
        #expect(outcome.activeVersionAfter.version == 8)
        #expect(outcome.activeVersionAfter.status == .active)
    }

    @Test("promotes on equal CE (no regression)")
    func promotesEqual() {
        let active = makeVersion(number: 7, ce: 2.3)
        let candidate = makeVersion(number: 8, ce: nil)
        let outcome = ProvisionalPromotionGate.evaluate(
            ProvisionalGateInput(
                candidateMeanCrossEntropyNats: 2.3,
                incumbentMeanCrossEntropyNats: 2.3,
                candidate: candidate,
                activeBefore: active
            )
        )
        #expect(outcome.verdict.promoted == true)
    }

    @Test("promotes when incumbent has no measurement")
    func promotesWithoutIncumbent() {
        let active = makeVersion(number: 1, ce: nil)
        let candidate = makeVersion(number: 2, ce: nil)
        let outcome = ProvisionalPromotionGate.evaluate(
            ProvisionalGateInput(
                candidateMeanCrossEntropyNats: 3.0,
                incumbentMeanCrossEntropyNats: nil,
                candidate: candidate,
                activeBefore: active
            )
        )
        #expect(outcome.verdict.promoted == true)
    }

    @Test("evaluateRecorded refuses worse candidate from stored measurements")
    func evaluateRecordedRefusesWorse() {
        let v6 = makeVersion(number: 6, ce: 3.123)
        let v7 = makeVersion(number: 7, ce: 3.341)
        let outcome = ProvisionalPromotionGate.evaluateRecorded(
            candidate: v7,
            activeBefore: v6
        )
        #expect(outcome != nil)
        #expect(outcome?.verdict.promoted == false)
        #expect(outcome?.verdict.primaryMetric.candidateValue == 3.341)
        #expect(outcome?.verdict.primaryMetric.incumbentValue == 3.123)
        #expect(outcome?.verdict.reason.contains("Not the M3 gate") == true)
    }

    @Test("evaluateRecorded promotes better candidate from stored measurements")
    func evaluateRecordedPromotesBetter() {
        let v5 = makeVersion(number: 5, ce: 3.193)
        let v6 = makeVersion(number: 6, ce: 3.123)
        let outcome = ProvisionalPromotionGate.evaluateRecorded(
            candidate: v6,
            activeBefore: v5
        )
        #expect(outcome?.verdict.promoted == true)
        #expect(outcome?.verdict.primaryMetric.candidateValue == 3.123)
        #expect(outcome?.verdict.primaryMetric.incumbentValue == 3.193)
    }

    @Test("activeVersusBest detects regression and accepts ties")
    func activeVersusBestGapAndTies() {
        let v5 = makeVersion(number: 5, ce: 3.193)
        let v6 = makeVersion(number: 6, ce: 3.123)
        let v7 = makeVersion(number: 7, ce: 3.341)
        let gap = ProvisionalPromotionGate.activeVersusBest(
            versions: [v5, v6, v7],
            active: v7
        )
        #expect(gap != nil)
        #expect(gap?.isActiveBest == false)
        #expect(gap?.bestMeasured.version == 6)
        #expect(abs((gap?.gapNats ?? 0) - (3.341 - 3.123)) < 1e-9)

        // Tie: two versions share the best score; active is one of them.
        let tiedBest = makeVersion(number: 8, ce: 3.123)
        let tie = ProvisionalPromotionGate.activeVersusBest(
            versions: [v5, v6, tiedBest],
            active: tiedBest
        )
        #expect(tie?.isActiveBest == true)
        #expect(abs(tie?.gapNats ?? 1) < 1e-12)
        #expect(tie?.bestMeasured.version == 8)

        // Active is uniquely best.
        let best = ProvisionalPromotionGate.activeVersusBest(
            versions: [v5, v6],
            active: v6
        )
        #expect(best?.isActiveBest == true)
        #expect(best?.bestMeasured.version == 6)
    }
}

@Suite("AdaptEngine error paths")
struct AdaptEngineErrorPathTests {
    @Test("missing registry root fails init or empty versions")
    func missingRegistry() async throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("style-mirror-no-reg-\(UUID().uuidString)", isDirectory: true)
        // AdapterRegistry creates the root if missing — so versions are simply empty.
        let config = AdaptEngineConfiguration(registryRoot: missing, heldOutJSONL: nil)
        let engine = try AdaptEngine(configuration: config, seed: 1)
        let versions = await engine.adapterVersions()
        #expect(versions.isEmpty)
        #expect(await engine.activeVersion() == nil)
    }

    @Test("prepareBlindRound without active adapter throws")
    func blindWithoutActive() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("style-mirror-empty-\(UUID().uuidString)", isDirectory: true)
        let config = AdaptEngineConfiguration(registryRoot: root, heldOutJSONL: nil)
        let engine = try AdaptEngine(configuration: config, seed: 1)
        await #expect(throws: StyleMirrorError.self) {
            try await engine.prepareBlindRound(incomingEmailID: "in-en-sprint")
        }
    }

    @Test("codeSwitchingDemo without active adapter returns unavailable with reason")
    func codeSwitchWithoutActive() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("style-mirror-codeswitch-\(UUID().uuidString)", isDirectory: true)
        let config = AdaptEngineConfiguration(registryRoot: root, heldOutJSONL: nil)
        let engine = try AdaptEngine(configuration: config, seed: 1)
        let result = await engine.codeSwitchingDemo()
        #expect(result.languages.isEmpty)
        #expect(!result.isAvailable)
        #expect(result.unavailabilityReason != nil)
        #expect(result.unavailabilityReason?.contains("no active adapter") == true)
        #expect(result.requestSummary == SampleCorpus.codeSwitch.requestSummary)
    }

    @Test("CodeSwitchResult.unavailable never looks like a silent empty success")
    func codeSwitchUnavailableFactory() {
        let result = CodeSwitchResult.unavailable(
            requestSummary: "Decline a meeting.",
            reason: "model unavailable: offline"
        )
        #expect(result.languages.isEmpty)
        #expect(!result.isAvailable)
        #expect(result.unavailabilityReason == "model unavailable: offline")
    }

    @Test("train with empty examples finishes without gate outcome")
    func trainEmptyExamples() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("style-mirror-train-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let config = AdaptEngineConfiguration(registryRoot: root, heldOutJSONL: nil)
        let engine = try AdaptEngine(configuration: config, seed: 1)
        var last: TrainingProgress?
        for await progress in engine.train(examples: [], configuration: .unitTest) {
            last = progress
        }
        #expect(last?.isFinished == true)
        #expect(last?.gateOutcome == nil)
    }

    @Test("lengthClassMismatch error describes role and counts")
    func lengthErrorDescription() {
        let error = StyleMirrorError.lengthClassMismatch(
            role: .baseModel,
            wordCount: 200,
            characterCount: 1200
        )
        let text = error.errorDescription ?? ""
        #expect(text.contains("baseModel"))
        #expect(text.contains("200"))
        #expect(text.contains("1200"))
        #expect(text.contains("not trimmed"))
    }
}

@Suite("DemoHeldOutLoss aggregate")
struct DemoHeldOutLossTests {
    @Test("token-weighted mean rejects empty contributions")
    func aggregateEmpty() {
        #expect(DemoHeldOutLoss.aggregate([]) == nil)
    }

    @Test("token-weighted mean is sum/tokens not mean-of-means")
    func aggregateWeighted() {
        let result = DemoHeldOutLoss.aggregate([
            .init(crossEntropySum: 10, supervisedTokens: 5), // mean 2
            .init(crossEntropySum: 30, supervisedTokens: 10), // mean 3
        ])
        #expect(result != nil)
        // (10+30)/(5+10) = 40/15 ≈ 2.666, not (2+3)/2 = 2.5
        #expect(abs(result!.meanCrossEntropyNats - (40.0 / 15.0)) < 1e-9)
        #expect(result!.exampleCount == 2)
        #expect(result!.supervisedTokenCount == 15)
    }
}
