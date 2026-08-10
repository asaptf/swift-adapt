import AdaptCore
import AdaptData
import AdaptEval
import AdaptRegistry
import AdaptSchedule
import AdaptTrain
import Foundation

/// Mutex box for Sendable test closures.
final class LockedBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: T
    init(_ value: T) { _value = value }
    var value: T {
        get { lock.lock(); defer { lock.unlock() }; return _value }
        set { lock.lock(); defer { lock.unlock() }; _value = newValue }
    }
}

/// Shared fixtures for AdaptSchedule offline tests.
enum ScheduleTestSupport {
    static func lineage(_ task: String = "quick-reply") -> AdapterLineage {
        AdapterLineage(
            taskID: task,
            baseModelID: "test-model",
            loraConfig: LoRAConfig()
        )
    }

    static func tempRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("adapt-schedule-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func makeBuffer(
        lineage: AdapterLineage,
        root: URL,
        examples: [TrainingExample] = [],
        configuration: ReplayBufferConfiguration = ReplayBufferConfiguration(
            maxExamples: 10_000,
            retention: .seconds(30 * 24 * 60 * 60),
            maxCapturesPerDay: 10_000,
            scrubberPipeline: ScrubberPipeline(scrubbers: [])
        )
    ) async throws -> ReplayBuffer {
        let buffer = try ReplayBuffer(
            lineage: lineage,
            rootURL: root,
            configuration: configuration
        )
        for example in examples {
            _ = try await buffer.add(example)
        }
        return buffer
    }

    static func example(
        id: UUID = UUID(),
        prompt: String = "p",
        completion: String = "c",
        capturedAt: Date = Date(),
        source: SignalSource = .acceptance
    ) -> TrainingExample {
        TrainingExample(
            id: id,
            prompt: prompt,
            completion: completion,
            capturedAt: capturedAt,
            source: source
        )
    }

    static func candidateVersion(
        lineage: AdapterLineage,
        version: Int = 1
    ) -> AdapterVersion {
        let now = Date()
        return AdapterVersion(
            lineage: lineage,
            version: version,
            trainedOn: TrainingWindow(start: now, end: now, exampleCount: 1),
            status: .candidate,
            weightsDigest: String(repeating: "ab", count: 32)
        )
    }

    static func trainOutcome(
        lineage: AdapterLineage,
        version: Int = 1,
        steps: Int = 10
    ) -> TrainOutcome {
        TrainOutcome(
            stopReason: .maxSteps,
            stepsCompleted: steps,
            lifetimeSteps: steps,
            lossHistory: [1.0],
            tokensPerSecond: 100,
            tokensProcessed: 100,
            candidateVersion: candidateVersion(lineage: lineage, version: version)
        )
    }

    static func promoteDecision() -> GateDecision {
        // n=12 all-positive improvements → significant under default alpha.
        let wilcoxon = WilcoxonSignedRank.oneSidedGreater(Array(repeating: 1.0, count: 12))
        return .promote(
            PromotionEvidence(
                wilcoxon: wilcoxon,
                candidateMeanPrimary: 1.0,
                incumbentMeanPrimary: 2.0,
                meanPairedDifference: -1.0,
                exampleCount: 12,
                alpha: 0.05
            )
        )
    }

    static func refuseDecision() -> GateDecision {
        let wilcoxon = WilcoxonSignedRank.oneSidedGreater(Array(repeating: -1.0, count: 12))
        return .refuse(
            RefusalEvidence(
                reason: .primaryRegression,
                wilcoxon: wilcoxon,
                candidateMeanPrimary: 3.0,
                incumbentMeanPrimary: 2.0,
                meanPairedDifference: 1.0,
                exampleCount: 12,
                alpha: 0.05
            )
        )
    }

    static func abstainDecision() -> GateDecision {
        .abstain(.belowMinHeldOut(have: 5, need: 30))
    }

    static func decided(_ decision: GateDecision) -> EvaluationResult {
        .decided(decision, report: PromotionGate.makeReport(decision: decision))
    }

    static func makePipeline(
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        buffer: ReplayBuffer,
        train: TrainOutcome? = nil,
        evaluation: EvaluationResult? = nil,
        device: DeviceEnvironmentSnapshot = DeviceEnvironmentSnapshot()
    ) -> AdaptPipeline {
        let trainRunner = ClosureTrainRunner { lin, reg, examples, budget in
            _ = reg
            _ = examples
            _ = budget
            return train ?? trainOutcome(lineage: lin)
        }
        let evalRunner = ClosureEvalRunner { _, _, _, _, _ in
            evaluation ?? decided(abstainDecision())
        }
        return AdaptPipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            trainRunner: trainRunner,
            evalRunner: evalRunner,
            deviceEnvironment: FixedDeviceEnvironment(device)
        )
    }
}
