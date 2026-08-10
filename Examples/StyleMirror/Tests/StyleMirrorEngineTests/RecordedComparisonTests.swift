import AdaptCore
import AdaptRegistry
import Foundation
import Testing
@testable import StyleMirrorEngine

// MARK: - Shared fixtures

private func ceVersion(
    lineage: AdapterLineage,
    number: Int,
    ce: Double,
    status: AdapterStatus
) -> AdapterVersion {
    AdapterVersion(
        lineage: lineage,
        version: number,
        parentVersion: number > 1 ? number - 1 : nil,
        trainedOn: TrainingWindow(start: Date(), end: Date(), exampleCount: 10),
        evalReport: EvalReport.heldOutCrossEntropy(
            meanNats: ce,
            exampleCount: 30,
            supervisedTokenCount: 200
        ),
        status: status,
        weightsDigest: String(repeating: "ab", count: 32)
    )
}

// MARK: - ScriptedEngine recorded comparison / rollback / restore

@Suite("ScriptedEngine recorded comparison & rollback")
struct ScriptedEngineRecordedComparisonTests {
    private var lineage: AdapterLineage {
        AdapterLineage(
            taskID: "email-style",
            baseModelID: "mlx-community/Qwen3-0.6B-4bit",
            loraConfig: LoRAConfig(rank: 8, scale: 10.0, numLayers: 16)
        )
    }

    @Test("compareRecordedVersions refuses worse and promotes better (injected CE)")
    func compareBothDirections() async throws {
        let engine = ScriptedEngine(seed: 42)
        let lineage = self.lineage
        // Night-six best, night-seven regression — the demo's headline numbers.
        await engine.installVersionsForTesting([
            ceVersion(lineage: lineage, number: 6, ce: 3.123, status: .archived),
            ceVersion(lineage: lineage, number: 7, ce: 3.341, status: .active),
        ])

        let refuse = try await engine.compareRecordedVersions(
            candidateVersion: 7,
            incumbentVersion: 6
        )
        #expect(refuse.verdict.promoted == false)
        #expect(refuse.verdict.primaryMetric.candidateValue == 3.341)
        #expect(refuse.verdict.primaryMetric.incumbentValue == 3.123)
        // Active pointer unchanged by comparison.
        #expect(await engine.activeVersion()?.version == 7)

        let pass = try await engine.compareRecordedVersions(
            candidateVersion: 6,
            incumbentVersion: 7
        )
        #expect(pass.verdict.promoted == true)
        #expect(pass.verdict.primaryMetric.candidateValue == 3.123)
        #expect(pass.verdict.primaryMetric.incumbentValue == 3.341)
    }

    @Test("rollback changes active pointer and reports a measured duration")
    func rollbackReportsDuration() async throws {
        let engine = ScriptedEngine(seed: 42)
        let before = await engine.activeVersion()
        #expect(before?.version == 7)

        let result = try await engine.rollbackToVersion(6)
        #expect(result.fromVersion.version == 7)
        #expect(result.toVersion.version == 6)
        #expect(result.toVersion.status == .active)
        // Duration is measured (may be zero on a fast in-memory flip).
        #expect(result.elapsed >= .zero)
        #expect(await engine.activeVersion()?.version == 6)
    }

    @Test("restoreDemoStartingState puts v7 back as active")
    func restoreStartingState() async throws {
        let engine = ScriptedEngine(seed: 42)
        _ = try await engine.rollbackToVersion(6)
        #expect(await engine.activeVersion()?.version == 6)

        try await engine.restoreDemoStartingState()
        #expect(await engine.activeVersion()?.version == 7)
        #expect(await engine.activeVersion()?.status == .active)
    }

    @Test("activeVersusBestMeasured answers correctly including ties")
    func activeVersusBest() async throws {
        let engine = ScriptedEngine(seed: 42)
        let lineage = self.lineage
        await engine.installVersionsForTesting([
            ceVersion(lineage: lineage, number: 5, ce: 3.193, status: .archived),
            ceVersion(lineage: lineage, number: 6, ce: 3.123, status: .archived),
            ceVersion(lineage: lineage, number: 7, ce: 3.341, status: .active),
        ])

        let gap = await engine.activeVersusBestMeasured()
        #expect(gap != nil)
        #expect(gap?.isActiveBest == false)
        #expect(gap?.bestMeasured.version == 6)
        #expect(gap?.active.version == 7)
        #expect(abs((gap?.gapNats ?? 0) - (3.341 - 3.123)) < 1e-9)

        // Tie for best: active shares the lowest CE.
        await engine.installVersionsForTesting([
            ceVersion(lineage: lineage, number: 6, ce: 3.123, status: .archived),
            ceVersion(lineage: lineage, number: 7, ce: 3.123, status: .active),
        ])
        let tie = await engine.activeVersusBestMeasured()
        #expect(tie?.isActiveBest == true)
        #expect(abs(tie?.gapNats ?? 1) < 1e-12)
    }

    @Test("poisoning path still refuses and leaves active unchanged")
    func poisoningStillWorks() async {
        let engine = ScriptedEngine(seed: 42)
        let before = await engine.activeVersion()?.version
        let outcome = await engine.runPoisoningScenario()
        #expect(outcome.verdict.promoted == false)
        #expect(await engine.activeVersion()?.version == before)
    }
}

// MARK: - AdaptEngine over a real temp registry (model-free)

@Suite("AdaptEngine recorded comparison & rollback")
struct AdaptEngineRecordedComparisonTests {
    private func makeTempRegistry() throws -> (AdapterRegistry, URL, AdapterLineage) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("style-mirror-recorded-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let registry = try AdapterRegistry(rootURL: root)
        let lineage = AdapterLineage(
            taskID: "style-mirror",
            baseModelID: "mlx-community/Qwen3-4B-4bit",
            loraConfig: LoRAConfig(
                rank: 8,
                scale: 10,
                keys: LoRAConfig.defaultAttentionKeys,
                numLayers: 8
            )
        )
        return (registry, root, lineage)
    }

    private func teardown(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func storeMeasured(
        registry: AdapterRegistry,
        lineage: AdapterLineage,
        ce: Double,
        weights: String
    ) async throws -> AdapterVersion {
        let stored = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data(weights.utf8),
            trainedOn: TrainingWindow(start: Date(), end: Date(), exampleCount: 10)
        )
        let report = EvalReport.heldOutCrossEntropy(
            meanNats: ce,
            exampleCount: 30,
            supervisedTokenCount: 200
        )
        try await registry.recordEvalReport(
            lineage: lineage,
            version: stored.version,
            report: report
        )
        return try await registry.version(
            for: lineage,
            version: stored.version,
            verifyIntegrity: false
        )
    }

    @Test("compareRecordedVersions both directions over registry measurements")
    func compareBothDirections() async throws {
        let (registry, root, lineage) = try makeTempRegistry()
        defer { teardown(root) }

        let v1 = try await storeMeasured(
            registry: registry, lineage: lineage, ce: 3.123, weights: "w1"
        )
        let v2 = try await storeMeasured(
            registry: registry, lineage: lineage, ce: 3.341, weights: "w2"
        )
        try await registry.promote(lineage: lineage, version: v2.version)

        let config = AdaptEngineConfiguration(
            registryRoot: root,
            heldOutJSONL: nil,
            demoStartingActiveVersion: v2.version
        )
        let engine = try AdaptEngine(configuration: config, seed: 1)

        let refuse = try await engine.compareRecordedVersions(
            candidateVersion: v2.version,
            incumbentVersion: v1.version
        )
        #expect(refuse.verdict.promoted == false)
        #expect(refuse.verdict.primaryMetric.candidateValue == 3.341)
        #expect(refuse.verdict.primaryMetric.incumbentValue == 3.123)

        let pass = try await engine.compareRecordedVersions(
            candidateVersion: v1.version,
            incumbentVersion: v2.version
        )
        #expect(pass.verdict.promoted == true)
        #expect(pass.verdict.primaryMetric.candidateValue == 3.123)

        // Comparison never moves the active pointer.
        #expect(await engine.activeVersion()?.version == v2.version)
    }

    @Test("rollback flips active pointer and restore returns starting version")
    func rollbackAndRestore() async throws {
        let (registry, root, lineage) = try makeTempRegistry()
        defer { teardown(root) }

        let v1 = try await storeMeasured(
            registry: registry, lineage: lineage, ce: 3.123, weights: "w1"
        )
        let v2 = try await storeMeasured(
            registry: registry, lineage: lineage, ce: 3.341, weights: "w2"
        )
        try await registry.promote(lineage: lineage, version: v2.version)

        let config = AdaptEngineConfiguration(
            registryRoot: root,
            heldOutJSONL: nil,
            demoStartingActiveVersion: v2.version
        )
        let engine = try AdaptEngine(configuration: config, seed: 1)

        let result = try await engine.rollbackToVersion(v1.version)
        #expect(result.fromVersion.version == v2.version)
        #expect(result.toVersion.version == v1.version)
        #expect(result.elapsed >= .zero)
        #expect(await engine.activeVersion()?.version == v1.version)

        try await engine.restoreDemoStartingState()
        #expect(await engine.activeVersion()?.version == v2.version)

        let gap = await engine.activeVersusBestMeasured()
        #expect(gap?.isActiveBest == false)
        #expect(gap?.bestMeasured.version == v1.version)
        #expect(gap?.active.version == v2.version)
    }

    @Test("activeVersusBestMeasured treats CE ties as best")
    func activeVersusBestTies() async throws {
        let (registry, root, lineage) = try makeTempRegistry()
        defer { teardown(root) }

        let v1 = try await storeMeasured(
            registry: registry, lineage: lineage, ce: 3.123, weights: "w1"
        )
        let v2 = try await storeMeasured(
            registry: registry, lineage: lineage, ce: 3.123, weights: "w2"
        )
        try await registry.promote(lineage: lineage, version: v2.version)

        let config = AdaptEngineConfiguration(
            registryRoot: root,
            heldOutJSONL: nil,
            demoStartingActiveVersion: v2.version
        )
        let engine = try AdaptEngine(configuration: config, seed: 1)
        let gap = await engine.activeVersusBestMeasured()
        #expect(gap?.isActiveBest == true)
        #expect(abs(gap?.gapNats ?? 1) < 1e-12)
        #expect(gap?.bestMeasured.version == v2.version)
        #expect(v1.version == 1)
    }
}
