import AdaptCore
import Foundation
import Testing
@testable import StyleMirrorEngine

@Suite("ScriptedEngine")
struct ScriptedEngineTests {
    // MARK: - Loss curve determinism

    @Test("same seed produces identical loss curve")
    func sameSeedIdenticalCurve() {
        let a = ScriptedEngine.makeLossCurve(steps: 40, seed: 42, validationInterval: 5)
        let b = ScriptedEngine.makeLossCurve(steps: 40, seed: 42, validationInterval: 5)
        #expect(a == b)
        #expect(a.count == 40)
    }

    @Test("different seeds produce different loss curves")
    func differentSeedsDifferentCurves() {
        let a = ScriptedEngine.makeLossCurve(steps: 40, seed: 1, validationInterval: 5)
        let b = ScriptedEngine.makeLossCurve(steps: 40, seed: 2, validationInterval: 5)
        #expect(a != b)
    }

    @Test("loss curve trends downward overall (not a pure flat or rising line)")
    func lossTrendsDown() {
        let curve = ScriptedEngine.makeLossCurve(steps: 80, seed: 99, validationInterval: 10)
        let early = curve.prefix(10).map(\.loss).reduce(0, +) / 10
        let late = curve.suffix(10).map(\.loss).reduce(0, +) / 10
        #expect(late < early)
        // Not a perfect exponential: some adjacent steps should not be strictly decreasing.
        var nonMonotone = 0
        for i in 1..<curve.count {
            if curve[i].loss >= curve[i - 1].loss { nonMonotone += 1 }
        }
        #expect(nonMonotone > 0)
    }

    // MARK: - Training stream

    @Test("training stream terminates with finished state")
    func trainingCompletes() async {
        let engine = ScriptedEngine(seed: 7)
        let examples = SampleCorpus.trainingExamples()
        var events: [TrainingProgress] = []
        for await progress in engine.train(examples: examples, configuration: .unitTest) {
            events.append(progress)
        }
        #expect(!events.isEmpty)
        #expect(events.last?.isFinished == true)
        #expect(events.last?.wasCancelled == false)
        #expect(events.last?.step == TrainingConfiguration.unitTest.totalSteps)
    }

    @Test("completed training yields passing gate verdict and promotes to v8")
    func trainingPromotesThroughGate() async {
        let engine = ScriptedEngine(seed: 42)
        let before = await engine.activeVersion()
        #expect(before?.version == 7)
        let v7Score = before?.evalReport?.primaryScore
        #expect(v7Score == 74)

        var last: TrainingProgress?
        for await progress in engine.train(
            examples: SampleCorpus.trainingExamples(),
            configuration: .unitTest
        ) {
            last = progress
            // In-flight steps never carry a gate outcome.
            if !progress.isFinished {
                #expect(progress.gateOutcome == nil)
            }
        }

        #expect(last?.isFinished == true)
        #expect(last?.wasCancelled == false)
        let outcome = last?.gateOutcome
        #expect(outcome != nil)
        #expect(outcome?.verdict.promoted == true)
        #expect(outcome?.activeVersionBefore.version == 7)
        #expect(outcome?.activeVersionAfter.version == 8)
        #expect(outcome?.candidate.version == 8)
        #expect(outcome?.candidate.status == .active)
        #expect(outcome?.activeVersionAfter.status == .active)

        let candidateScore = outcome?.candidate.evalReport?.primaryScore
        #expect(candidateScore != nil)
        #expect(candidateScore! > v7Score!)
        #expect(candidateScore == 75)
        #expect(outcome?.verdict.primaryMetric.candidateValue == 75)
        #expect(outcome?.verdict.primaryMetric.incumbentValue == 74)

        let active = await engine.activeVersion()
        #expect(active?.version == 8)
        #expect(active?.evalReport?.primaryScore == 75)

        let versions = await engine.adapterVersions()
        #expect(versions.map(\.version).contains(8))
        #expect(versions.filter { $0.status == .active }.count == 1)
        #expect(versions.first(where: { $0.version == 7 })?.status == .archived)
    }

    @Test("cancelled training promotes nothing and leaves active version unchanged")
    func cancelledTrainingDoesNotPromote() async {
        let engine = ScriptedEngine(seed: 7)
        let before = await engine.activeVersion()
        #expect(before?.version == 7)

        // Non-zero duration so cancellation can land mid-stream before the final step.
        let config = TrainingConfiguration(
            seed: 7,
            totalSteps: 200,
            duration: .milliseconds(800),
            validationInterval: 10
        )
        let stream = engine.train(examples: SampleCorpus.trainingExamples(), configuration: config)
        let consumer = Task { () -> [TrainingProgress] in
            var events: [TrainingProgress] = []
            for await progress in stream {
                events.append(progress)
                if progress.step >= 2 { break }
            }
            return events
        }
        try? await Task.sleep(for: .milliseconds(50))
        consumer.cancel()
        let events = await consumer.value

        #expect(!events.isEmpty)
        // No terminal success outcome may appear — either early break without finish,
        // or a cancelled finished event with nil gateOutcome.
        for event in events {
            #expect(event.gateOutcome == nil)
            if event.wasCancelled {
                #expect(event.isFinished)
            }
        }

        let after = await engine.activeVersion()
        #expect(after?.version == before?.version)
        #expect(after?.status == .active)
        let versions = await engine.adapterVersions()
        #expect(versions.count == 7)
        #expect(versions.last?.version == 7)
    }

    @Test("training cancellation is a normal finished outcome")
    func trainingCancellation() async {
        let engine = ScriptedEngine(seed: 7)
        // Use a non-zero duration so we can cancel mid-stream.
        let config = TrainingConfiguration(
            seed: 7,
            totalSteps: 200,
            duration: .milliseconds(500),
            validationInterval: 10
        )
        let stream = engine.train(examples: SampleCorpus.trainingExamples(), configuration: config)
        let iteratorTask = Task { () -> [TrainingProgress] in
            var events: [TrainingProgress] = []
            for await progress in stream {
                events.append(progress)
                if progress.step >= 2 {
                    break
                }
            }
            return events
        }
        // Allow a couple of steps, then cancel the consumer task.
        try? await Task.sleep(for: .milliseconds(50))
        iteratorTask.cancel()
        let events = await iteratorTask.value
        // Either we broke early after step >= 2, or cancellation yielded a cancel event.
        #expect(!events.isEmpty)
        if let last = events.last, last.wasCancelled {
            #expect(last.isFinished)
            #expect(last.gateOutcome == nil)
        } else {
            #expect(events.contains(where: { $0.step >= 1 }))
        }
    }

    @Test("training cancellation via task cancel completes without hanging")
    func trainingCancelThroughTermination() async {
        let engine = ScriptedEngine(seed: 3)
        let config = TrainingConfiguration(
            seed: 3,
            totalSteps: 500,
            duration: .seconds(30),
            validationInterval: 20
        )
        let stream = engine.train(examples: SampleCorpus.trainingExamples(), configuration: config)
        let consumer = Task { () -> [TrainingProgress] in
            var collected: [TrainingProgress] = []
            for await p in stream {
                collected.append(p)
                if collected.count >= 3 { break }
            }
            return collected
        }
        try? await Task.sleep(for: .milliseconds(40))
        consumer.cancel()
        let collected = await consumer.value
        // Cancellation must not hang; partial progress is fine.
        #expect(collected.count <= 500)
        if let last = collected.last, last.wasCancelled {
            #expect(last.isFinished)
            #expect(last.gateOutcome == nil)
        }
        // Active version must remain the overnight incumbent.
        let active = await engine.activeVersion()
        #expect(active?.version == 7)
    }

    // MARK: - Blind test shuffle + scoring

    @Test("same seed produces identical blind-test shuffle")
    func blindShuffleDeterministic() async throws {
        let a = ScriptedEngine(seed: 42)
        let b = ScriptedEngine(seed: 42)
        let id = SampleCorpus.blindRounds[0].incoming.id
        let roundA = try await a.prepareBlindRound(incomingEmailID: id)
        let roundB = try await b.prepareBlindRound(incomingEmailID: id)
        #expect(roundA.candidates.map(\.id) == roundB.candidates.map(\.id))
        #expect(roundA.candidates.map(\.body) == roundB.candidates.map(\.body))
    }

    @Test("different seeds produce different blind-test shuffles")
    func blindShuffleDiffersBySeed() async throws {
        let a = ScriptedEngine(seed: 1)
        let b = ScriptedEngine(seed: 2)
        let id = SampleCorpus.blindRounds[0].incoming.id
        let roundA = try await a.prepareBlindRound(incomingEmailID: id)
        let roundB = try await b.prepareBlindRound(incomingEmailID: id)
        // Bodies are the same set; order or IDs should differ for different seeds.
        let orderA = roundA.candidates.map(\.body)
        let orderB = roundB.candidates.map(\.body)
        let idsDiffer = roundA.candidates.map(\.id) != roundB.candidates.map(\.id)
        let orderDiffers = orderA != orderB
        #expect(idsDiffer || orderDiffers)
    }

    @Test("blind guess scores correct/incorrect and tally accumulates")
    func blindGuessAndTally() async throws {
        let engine = ScriptedEngine(seed: 42)
        let id = SampleCorpus.blindRounds[0].incoming.id
        let round = try await engine.prepareBlindRound(incomingEmailID: id)
        #expect(round.candidates.count == 3)

        // Guess first candidate, then use reveal to verify scoring.
        let first = round.candidates[0]
        let result = try await engine.submitBlindGuess(roundID: round.id, candidateID: first.id)
        #expect(result.reveal.count == 3)
        #expect(Set(result.reveal.values) == Set(ReplyRole.allCases))
        #expect(result.guessedRole == result.reveal[first.id])
        #expect(result.identifiedHuman == (result.guessedRole == .human))
        #expect(result.adapterMistakenForHuman == (result.guessedRole == .adaptedModel))
        #expect(result.tally.roundsPlayed == 1)

        // Second round — tally accumulates.
        let id2 = SampleCorpus.blindRounds[1].incoming.id
        let round2 = try await engine.prepareBlindRound(incomingEmailID: id2)
        // Intentionally pick human via reveal after guessing each until we find... better:
        // guess index 0 again and just check roundsPlayed == 2.
        let result2 = try await engine.submitBlindGuess(
            roundID: round2.id,
            candidateID: round2.candidates[0].id
        )
        #expect(result2.tally.roundsPlayed == 2)
        let tally = await engine.blindTestTally()
        #expect(tally.roundsPlayed == 2)
        #expect(
            tally.humanCorrectlyIdentified
                + tally.adapterMistakenForHuman
                + tally.baseMistakenForHuman
                == 2
        )
    }

    @Test("correct human guess increments humanCorrectlyIdentified")
    func correctHumanGuess() async throws {
        let engine = ScriptedEngine(seed: 11)
        let id = SampleCorpus.blindRounds[2].incoming.id
        let round = try await engine.prepareBlindRound(incomingEmailID: id)
        // Probe: submit first guess to get reveal... but that consumes the round.
        // Instead, prepare twice with same seed/counter — counter advances.
        // Use bodies to match human fixture text.
        let humanBody = SampleCorpus.blindRounds[2].human
        let humanCandidate = round.candidates.first { $0.body == humanBody }
        #expect(humanCandidate != nil)
        let result = try await engine.submitBlindGuess(
            roundID: round.id,
            candidateID: humanCandidate!.id
        )
        #expect(result.identifiedHuman)
        #expect(result.tally.humanCorrectlyIdentified == 1)
        #expect(result.adapterMistakenForHuman == false)
    }

    // MARK: - Poisoning / shared gate

    @Test("poisoning refuses promotion and leaves active version unchanged")
    func poisoningRefuses() async {
        let engine = ScriptedEngine(seed: 42)
        let before = await engine.activeVersion()
        #expect(before != nil)
        let outcome = await engine.runPoisoningScenario()
        #expect(outcome.verdict.promoted == false)
        #expect(outcome.activeVersionBefore.version == before?.version)
        #expect(outcome.activeVersionAfter.version == outcome.activeVersionBefore.version)
        #expect(outcome.activeVersionAfter.status == .active)
        #expect(outcome.candidate.status == .candidate)
        #expect(outcome.candidate.version == before!.version + 1)
        #expect(outcome.verdict.primaryMetric.name == "held_out_perplexity")
        #expect(outcome.verdict.primaryMetric.candidateValue > outcome.verdict.primaryMetric.threshold)

        // Active version after equals active version before (and still active on the engine).
        let after = await engine.activeVersion()
        #expect(after?.version == before?.version)
        #expect(after?.version == outcome.activeVersionAfter.version)
    }

    @Test("poisoning after a successful train refuses v9 and leaves v8 active")
    func poisoningAfterTrainRefusesNextCandidate() async {
        let engine = ScriptedEngine(seed: 42)
        for await progress in engine.train(
            examples: SampleCorpus.trainingExamples(),
            configuration: .unitTest
        ) {
            if progress.isFinished { break }
        }
        let afterTrain = await engine.activeVersion()
        #expect(afterTrain?.version == 8)

        let outcome = await engine.runPoisoningScenario()
        #expect(outcome.verdict.promoted == false)
        #expect(outcome.candidate.version == 9)
        #expect(outcome.activeVersionBefore.version == 8)
        #expect(outcome.activeVersionAfter.version == 8)
        #expect(await engine.activeVersion()?.version == 8)
    }

    @Test("pass and refusal share GateOutcome / GateVerdict type")
    func sharedGateOutcomeType() async {
        let engine = ScriptedEngine(seed: 42)

        var passOutcome: GateOutcome?
        for await progress in engine.train(
            examples: SampleCorpus.trainingExamples(),
            configuration: .unitTest
        ) {
            if let outcome = progress.gateOutcome {
                passOutcome = outcome
            }
        }
        let refuseOutcome = await engine.runPoisoningScenario()

        #expect(passOutcome != nil)
        // Same concrete type for both paths — one UI component, two states.
        let pass: GateOutcome = passOutcome!
        let refuse: GateOutcome = refuseOutcome
        #expect(type(of: pass) == type(of: refuse))
        #expect(type(of: pass.verdict) == GateVerdict.self)
        #expect(type(of: refuse.verdict) == GateVerdict.self)

        #expect(pass.verdict.promoted == true)
        #expect(refuse.verdict.promoted == false)
        // Both expose the deciding metric and the resulting active version.
        #expect(pass.verdict.primaryMetric.candidateValue > pass.verdict.primaryMetric.incumbentValue!)
        #expect(refuse.verdict.primaryMetric.candidateValue > refuse.verdict.primaryMetric.threshold)
        #expect(pass.activeVersionAfter.version == 8)
        #expect(refuse.activeVersionAfter.version == 8)
        #expect(pass.activeVersionAfter.version == refuse.activeVersionAfter.version)
    }

    // MARK: - Timeline

    @Test("adapter timeline has seven versions with rising scores")
    func timelineSevenNights() async {
        let engine = ScriptedEngine(seed: 42)
        let versions = await engine.adapterVersions()
        #expect(versions.count == 7)
        #expect(versions.map(\.version) == Array(1...7))
        let scores = versions.compactMap { $0.evalReport?.primaryScore }
        #expect(scores.count == 7)
        for i in 1..<scores.count {
            #expect(scores[i] > scores[i - 1])
        }
        #expect(versions.last?.status == .active)
        // Reuses AdaptCore types end-to-end.
        #expect(versions[0].trainedOn.exampleCount > 0)
        #expect(versions[0].lineage.taskID == "email-style")
    }

    // MARK: - Code-switching

    @Test("code-switching covers three languages with base and adapted")
    func codeSwitch() async {
        let engine = ScriptedEngine(seed: 1)
        let result = await engine.codeSwitchingDemo()
        #expect(result.languages.count == 3)
        #expect(result.isAvailable)
        #expect(result.unavailabilityReason == nil)
        let langs = Set(result.languages.map(\.language))
        #expect(langs == [.english, .spanish, .russian])
        for pair in result.languages {
            #expect(!pair.baseReply.isEmpty)
            #expect(!pair.adaptedReply.isEmpty)
            #expect(pair.baseReply != pair.adaptedReply)
        }
    }

    @Test("code-switching failure surfaces a reason instead of empty success")
    func codeSwitchFailureReason() async {
        let reason = "simulated generation failure for tests"
        let engine = ScriptedEngine(seed: 1, codeSwitchFailureReason: reason)
        let result = await engine.codeSwitchingDemo()
        #expect(result.languages.isEmpty)
        #expect(!result.isAvailable)
        #expect(result.unavailabilityReason == reason)
        #expect(result.requestSummary == SampleCorpus.codeSwitch.requestSummary)
    }

    @Test("code-switching progress events are ordered and end at total")
    func codeSwitchProgressOrdered() async {
        let engine = ScriptedEngine(seed: 1)
        let box = ProgressBox()
        let result = await engine.codeSwitchingDemo { event in
            box.append(event)
        }
        #expect(result.isAvailable)
        let events = box.snapshot()
        #expect(!events.isEmpty)
        // Completed is non-decreasing and stays within 0…total.
        var previous = -1
        for event in events {
            #expect(event.total == 6)
            #expect(event.completed >= previous)
            #expect(event.completed <= event.total)
            previous = event.completed
        }
        #expect(events.last?.completed == events.last?.total)
        #expect(events.last?.total == 6)
    }

    @Test("blind-round progress events are ordered and end at total")
    func blindProgressOrdered() async throws {
        let engine = ScriptedEngine(seed: 1)
        let id = SampleCorpus.blindRounds[0].incoming.id
        let box = ProgressBox()
        let round = try await engine.prepareBlindRound(incomingEmailID: id) { event in
            box.append(event)
        }
        #expect(round.candidates.count == 3)
        let events = box.snapshot()
        #expect(!events.isEmpty)
        var previous = -1
        for event in events {
            #expect(event.total == 2)
            #expect(event.completed >= previous)
            #expect(event.completed <= event.total)
            previous = event.completed
        }
        #expect(events.last?.completed == 2)
        #expect(events.last?.total == 2)
    }

    // MARK: - Outbound meter honesty

    @Test("outbound traffic meter starts at zero without recordOutbound calls")
    func meterStartsZero() async {
        let meter = OutboundTrafficMeter()
        let bytes = await meter.bytesSent
        let ops = await meter.operationCount
        #expect(bytes == 0)
        #expect(ops == 0)
        await meter.recordOutbound(bytes: 128)
        #expect(await meter.bytesSent == 128)
        #expect(await meter.operationCount == 1)
    }
}

/// Collects ``GenerationProgress`` from ``@Sendable`` handlers in tests.
private final class ProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [GenerationProgress] = []

    func append(_ event: GenerationProgress) {
        lock.lock()
        events.append(event)
        lock.unlock()
    }

    func snapshot() -> [GenerationProgress] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }
}
