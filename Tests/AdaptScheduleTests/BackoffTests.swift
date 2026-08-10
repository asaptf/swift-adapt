import AdaptCore
import AdaptData
import AdaptEval
import AdaptRegistry
import AdaptSchedule
import AdaptTrain
import Foundation
import Testing

@Suite("Backoff: refuse vs abstain")
struct BackoffTests {

    @Test("refusal feeds exponential backoff; subsequent run is deferred")
    func refusalFeedsBackoff() async throws {
        let root = try ScheduleTestSupport.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let lineage = ScheduleTestSupport.lineage("backoff-refuse")
        let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
        let buffer = try await ScheduleTestSupport.makeBuffer(
            lineage: lineage,
            root: root.appendingPathComponent("buf"),
            examples: [ScheduleTestSupport.example()]
        )

        let pipeline = ScheduleTestSupport.makePipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            train: ScheduleTestSupport.trainOutcome(lineage: lineage),
            evaluation: ScheduleTestSupport.decided(ScheduleTestSupport.refuseDecision())
        )

        let policy = BackoffPolicy(
            baseDelay: .seconds(3600),
            maxDelay: .seconds(16 * 3600),
            enabled: true
        )
        let config = PipelineConfiguration(
            trainBudget: TrainBudget(maxSteps: 1),
            devicePolicy: .macOS,
            backoffPolicy: policy,
            enforceCapabilityGate: false
        )

        let first = try await pipeline.run(configuration: config)
        #expect(first.didComplete)
        #expect(first.backoffState.consecutiveRefusals == 1)
        #expect(first.backoffState.nextEligibleAt != nil)
        #expect(first.evaluation?.decision?.feedsBackoff == true)

        let second = try await pipeline.run(configuration: config)
        guard case .policyBlocked(let error) = second.stopReason else {
            Issue.record("expected backoff policy block, got \(String(describing: second.stopReason))")
            return
        }
        guard case .backoffActive(_, let n) = error else {
            Issue.record("expected backoffActive, got \(error)")
            return
        }
        #expect(n == 1)
    }

    @Test("abstention does not feed backoff")
    func abstentionDoesNotFeedBackoff() async throws {
        let root = try ScheduleTestSupport.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let lineage = ScheduleTestSupport.lineage("backoff-abstain")
        let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
        let buffer = try await ScheduleTestSupport.makeBuffer(
            lineage: lineage,
            root: root.appendingPathComponent("buf"),
            examples: [ScheduleTestSupport.example()]
        )

        let pipeline = ScheduleTestSupport.makePipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            train: ScheduleTestSupport.trainOutcome(lineage: lineage),
            evaluation: ScheduleTestSupport.decided(ScheduleTestSupport.abstainDecision())
        )

        let config = PipelineConfiguration(
            trainBudget: TrainBudget(maxSteps: 1),
            devicePolicy: .macOS,
            backoffPolicy: BackoffPolicy(baseDelay: .seconds(3600), enabled: true),
            enforceCapabilityGate: false
        )

        let first = try await pipeline.run(configuration: config)
        #expect(first.didComplete)
        #expect(first.evaluation?.decision?.isAbstention == true)
        #expect(first.evaluation?.decision?.feedsBackoff == false)
        #expect(first.backoffState.consecutiveRefusals == 0)
        #expect(first.backoffState.nextEligibleAt == nil)

        // Second run must not be deferred.
        let second = try await pipeline.run(configuration: config)
        #expect(second.didComplete)
        #expect(second.backoffState.consecutiveRefusals == 0)
    }

    @Test("promote resets backoff after prior refusals")
    func promoteResetsBackoff() async throws {
        let root = try ScheduleTestSupport.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let lineage = ScheduleTestSupport.lineage("backoff-reset")
        let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
        let buffer = try await ScheduleTestSupport.makeBuffer(
            lineage: lineage,
            root: root.appendingPathComponent("buf"),
            examples: [ScheduleTestSupport.example()]
        )
        let lineageDir = await registry.lineageDirectoryURL(for: lineage)
        try FileManager.default.createDirectory(at: lineageDir, withIntermediateDirectories: true)

        let state = BackoffState(consecutiveRefusals: 3, nextEligibleAt: nil)
        try BackoffStore.save(state, to: lineageDir)

        // Store real candidate so promote can flip.
        let now = Date()
        let stored = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data(repeating: 7, count: 32),
            trainedOn: TrainingWindow(start: now, end: now, exampleCount: 1)
        )
        let train = TrainOutcome(
            stopReason: .maxSteps,
            stepsCompleted: 1,
            lifetimeSteps: 1,
            lossHistory: [1],
            tokensPerSecond: 1,
            tokensProcessed: 1,
            candidateVersion: stored
        )

        let pipeline = ScheduleTestSupport.makePipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            train: train,
            evaluation: ScheduleTestSupport.decided(ScheduleTestSupport.promoteDecision())
        )

        let config = PipelineConfiguration(
            trainBudget: TrainBudget(maxSteps: 1),
            devicePolicy: .macOS,
            backoffPolicy: BackoffPolicy(enabled: true),
            enforceCapabilityGate: false
        )
        let outcome = try await pipeline.run(configuration: config)
        #expect(outcome.promotedVersion == stored.version)
        #expect(outcome.backoffState.consecutiveRefusals == 0)
        #expect(outcome.backoffState.nextEligibleAt == nil)
        _ = state
    }

    @Test("BackoffPolicy doubles delay and caps at max")
    func delayMath() throws {
        let policy = BackoffPolicy(baseDelay: .seconds(10), maxDelay: .seconds(40))
        #expect(policy.delay(afterConsecutiveRefusals: 1) == .seconds(10))
        #expect(policy.delay(afterConsecutiveRefusals: 2) == .seconds(20))
        #expect(policy.delay(afterConsecutiveRefusals: 3) == .seconds(40))
        #expect(policy.delay(afterConsecutiveRefusals: 4) == .seconds(40))
        #expect(policy.delay(afterConsecutiveRefusals: 0) == .zero)
    }
}
