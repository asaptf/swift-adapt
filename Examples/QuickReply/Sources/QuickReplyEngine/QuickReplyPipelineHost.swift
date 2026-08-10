import AdaptCore
import AdaptEval
import AdaptSchedule
import AdaptTrain
import Foundation

/// Wires `AdaptPipeline` for the QuickReply demo.
///
/// ## Training runner
///
/// Real on-device LoRA needs a loaded MLX model (see TESTING.md). This host
/// defaults to ``NoOpTrainRunner`` so the **skeleton compiles and can exercise
/// prune/sample/eval wiring offline**. Replace `trainRunner` with a
/// `Trainer`-backed implementation before the overnight device protocol.
public actor QuickReplyPipelineHost {
    public let configuration: QuickReplyConfiguration
    public let store: QuickReplyStore
    public let pipeline: AdaptPipeline

    public init(
        configuration: QuickReplyConfiguration = QuickReplyConfiguration(),
        store: QuickReplyStore? = nil,
        trainRunner: (any PipelineTrainRunner)? = nil,
        evalRunner: (any PipelineEvalRunner)? = nil
    ) throws {
        self.configuration = configuration
        let resolvedStore = try store ?? QuickReplyStore(lineage: configuration.lineage)
        self.store = resolvedStore

        // Default eval: model-free abstain until the device protocol supplies
        // real scorers. Still returns AdaptEval types so promotion is gated.
        let resolvedEval =
            evalRunner
            ?? ClosureEvalRunner { _, _, source, policy, seed in
                // Without scorers we can only abstain (insufficient measurement).
                // Still go through GateDecision so promotion remains gated.
                _ = try await source.examples()
                _ = policy
                _ = seed
                let decision = GateDecision.abstain(.missingIncumbent)
                return .decided(decision, report: PromotionGate.makeReport(decision: decision))
            }

        self.pipeline = AdaptPipeline(
            lineage: configuration.lineage,
            registry: resolvedStore.registry,
            buffer: resolvedStore.buffer,
            trainRunner: trainRunner ?? NoOpTrainRunner(),
            evalRunner: resolvedEval,
            deviceEnvironment: SystemDeviceEnvironment()
        )
    }

    /// Runs one night pipeline under platform defaults.
    @discardableResult
    public func runNightly() async throws -> PipelineOutcome {
        var config = PipelineConfiguration(
            trainBudget: configuration.trainBudget,
            devicePolicy: .current,
            enforceCapabilityGate: true
        )
        // Sync is M6 — leave disabled.
        config.runSync = false
        return try await pipeline.run(configuration: config)
    }
}
