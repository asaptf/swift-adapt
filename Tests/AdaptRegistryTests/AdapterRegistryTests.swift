import AdaptCore
import AdaptRegistry
import Foundation
import Testing

@Suite("AdapterRegistry")
struct AdapterRegistryTests {
    private func makeRegistry() throws -> (AdapterRegistry, URL) {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("AdaptRegistryTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        let registry = try AdapterRegistry(rootURL: temp)
        return (registry, temp)
    }

    private func teardown(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private var sampleLineage: AdapterLineage {
        AdapterLineage(
            taskID: "email-style",
            baseModelID: "mlx-community/Qwen3-0.6B-4bit",
            loraConfig: LoRAConfig()
        )
    }

    private var sampleWindow: TrainingWindow {
        TrainingWindow(
            start: Date(timeIntervalSince1970: 1_700_000_000),
            end: Date(timeIntervalSince1970: 1_700_086_400),
            exampleCount: 10
        )
    }

    @Test("store candidate writes layout and remains inactive")
    func storeCandidateLayout() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        let weights = Data("fake-weights-v1".utf8)
        let stored = try await registry.storeCandidate(
            lineage: lineage,
            weights: weights,
            trainedOn: sampleWindow
        )

        #expect(stored.version == 1)
        #expect(stored.status == .candidate)
        #expect(stored.weightsDigest == AdapterRegistry.sha256Hex(weights))

        let dir = await registry.directoryURL(for: lineage, version: 1)
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("adapter_config.json").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("adapters.safetensors").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("version.json").path))

        let active = try await registry.activeVersion(for: lineage)
        #expect(active == nil)

        // adapter_config.json must be upstream-shaped.
        let configData = try Data(contentsOf: dir.appendingPathComponent("adapter_config.json"))
        let configJSON = try #require(String(data: configData, encoding: .utf8))
        #expect(configJSON.contains("num_layers"))
        #expect(configJSON.contains("lora_parameters"))
    }

    @Test("promote sets single active; second promote demotes first")
    func promoteInvariant() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        let v1 = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("w1".utf8),
            trainedOn: sampleWindow
        )
        let v2 = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("w2".utf8),
            trainedOn: sampleWindow,
            parentVersion: v1.version
        )

        try await registry.promote(lineage: lineage, version: v1.version)
        var active = try await registry.activeVersion(for: lineage)
        #expect(active?.version == 1)
        #expect(active?.status == .active)

        try await registry.promote(lineage: lineage, version: v2.version)
        active = try await registry.activeVersion(for: lineage)
        #expect(active?.version == 2)

        let all = try await registry.listVersions(for: lineage)
        let actives = all.filter { $0.status == .active }
        #expect(actives.count == 1)
        #expect(actives.first?.version == 2)
        #expect(all.first { $0.version == 1 }?.status == .rolledBack)
    }

    @Test("concurrent promotes leave at most one active")
    func concurrentPromotes() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        for i in 1...8 {
            _ = try await registry.storeCandidate(
                lineage: lineage,
                weights: Data("weights-\(i)".utf8),
                trainedOn: sampleWindow
            )
        }

        await withTaskGroup(of: Void.self) { group in
            for version in 1...8 {
                group.addTask {
                    try? await registry.promote(lineage: lineage, version: version)
                }
            }
        }

        let all = try await registry.listVersions(for: lineage)
        let actives = all.filter { $0.status == .active }
        #expect(actives.count == 1)

        let active = try await registry.activeVersion(for: lineage)
        #expect(active != nil)
        #expect(active?.status == .active)
    }

    @Test("rollback is pointer flip and restores prior version")
    func rollbackPointerFlip() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        let w1 = Data("weights-one".utf8)
        let w2 = Data("weights-two".utf8)
        _ = try await registry.storeCandidate(lineage: lineage, weights: w1, trainedOn: sampleWindow)
        _ = try await registry.storeCandidate(lineage: lineage, weights: w2, trainedOn: sampleWindow)

        try await registry.promote(lineage: lineage, version: 1)
        try await registry.promote(lineage: lineage, version: 2)

        let urlBefore = await registry.weightsURL(for: lineage, version: 1)
        let bytesBefore = try Data(contentsOf: urlBefore)

        try await registry.rollback(lineage: lineage, to: 1)

        let active = try await registry.activeVersion(for: lineage)
        #expect(active?.version == 1)
        #expect(active?.status == .active)

        // Weights for v1 untouched.
        let bytesAfter = try Data(contentsOf: urlBefore)
        #expect(bytesBefore == bytesAfter)
        #expect(bytesAfter == w1)

        // v2 still on disk.
        let v2url = await registry.weightsURL(for: lineage, version: 2)
        #expect(FileManager.default.fileExists(atPath: v2url.path))
    }

    @Test("integrity mismatch surfaces typed error")
    func integrityMismatch() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        _ = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("original".utf8),
            trainedOn: sampleWindow
        )

        // Corrupt weights on disk.
        let weightsURL = await registry.weightsURL(for: lineage, version: 1)
        try Data("corrupted".utf8).write(to: weightsURL)

        do {
            _ = try await registry.version(for: lineage, version: 1, verifyIntegrity: true)
            Issue.record("expected integrity error")
        } catch let error as AdaptRegistry.AdaptError {
            guard case .integrityMismatch = error else {
                Issue.record("expected integrityMismatch, got \(error)")
                return
            }
        } catch {
            Issue.record("expected AdaptRegistry.AdaptError, got \(error)")
        }
    }

    @Test("crash mid-promote leaves consistent active pointer")
    func crashSafetyMidPromote() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        _ = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("w1".utf8),
            trainedOn: sampleWindow
        )
        _ = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("w2".utf8),
            trainedOn: sampleWindow
        )

        try await registry.promote(lineage: lineage, version: 1)
        let before = try await registry.activeVersion(for: lineage)
        #expect(before?.version == 1)

        // Simulate kill between status updates and state.json flip.
        await registry.setFaultBeforePointerFlip(true)
        do {
            try await registry.promote(lineage: lineage, version: 2)
            Issue.record("expected injected fault")
        } catch let error as AdaptRegistry.AdaptError {
            if case .injectedFault = error {
                // expected
            } else {
                Issue.record("expected injectedFault, got \(error)")
            }
        } catch {
            Issue.record("expected AdaptRegistry.AdaptError, got \(error)")
        }

        // Registry must still open consistently — old active (v1) via state.json.
        let after = try await registry.activeVersion(for: lineage)
        #expect(after?.version == 1)

        let all = try await registry.listVersions(for: lineage)
        #expect(all.filter { $0.status == .active }.count == 1)
    }

    @Test("crash after weights write leaves no corrupt pointer and no partial version")
    func crashAfterWeightsWrite() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        _ = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("w1".utf8),
            trainedOn: sampleWindow
        )
        try await registry.promote(lineage: lineage, version: 1)

        await registry.setFaultAfterWeightsWrite(true)
        do {
            _ = try await registry.storeCandidate(
                lineage: lineage,
                weights: Data("partial".utf8),
                trainedOn: sampleWindow
            )
            Issue.record("expected injected fault")
        } catch let error as AdaptRegistry.AdaptError {
            if case .injectedFault = error {
                // expected
            } else {
                Issue.record("expected injectedFault, got \(error)")
            }
        } catch {
            Issue.record("expected AdaptRegistry.AdaptError, got \(error)")
        }

        // Active still v1; incomplete store must not be listable as a full version.
        let active = try await registry.activeVersion(for: lineage)
        #expect(active?.version == 1)

        // Re-open via a fresh actor on the same root to simulate process restart.
        let reopened = try AdapterRegistry(rootURL: root)
        let reopenedActive = try await reopened.activeVersion(for: lineage)
        #expect(reopenedActive?.version == 1)
    }

    @Test("gc never deletes active version")
    func gcKeepsActive() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        for i in 1...5 {
            _ = try await registry.storeCandidate(
                lineage: lineage,
                weights: Data("w\(i)".utf8),
                trainedOn: sampleWindow
            )
        }
        try await registry.promote(lineage: lineage, version: 1)

        try await registry.gc(lineage: lineage, keepLast: 2)

        let remaining = try await registry.listVersions(for: lineage)
        let numbers = Set(remaining.map(\.version))
        // keepLast 2 → v4, v5; plus active v1
        #expect(numbers.contains(1))
        #expect(numbers.contains(4))
        #expect(numbers.contains(5))
        #expect(!numbers.contains(2))
        #expect(!numbers.contains(3))

        let active = try await registry.activeVersion(for: lineage)
        #expect(active?.version == 1)
    }

    @Test("store increments versions and supports clearActive")
    func versioningAndClear() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        let a = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("a".utf8),
            trainedOn: sampleWindow
        )
        let b = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("b".utf8),
            trainedOn: sampleWindow
        )
        #expect(a.version == 1)
        #expect(b.version == 2)

        try await registry.promote(lineage: lineage, version: 2)
        try await registry.clearActive(lineage: lineage)
        let active = try await registry.activeVersion(for: lineage)
        #expect(active == nil)
    }
}

// Test-only helpers to toggle package fault flags from the test target.
extension AdapterRegistry {
    func setFaultBeforePointerFlip(_ value: Bool) {
        faultBeforePointerFlip = value
    }

    func setFaultAfterWeightsWrite(_ value: Bool) {
        faultAfterWeightsWrite = value
    }
}
