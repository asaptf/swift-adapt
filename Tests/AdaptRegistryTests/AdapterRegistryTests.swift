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
        } catch let error as AdaptRegistryError {
            guard case .integrityMismatch = error else {
                Issue.record("expected integrityMismatch, got \(error)")
                return
            }
        } catch {
            Issue.record("expected AdaptRegistryError, got \(error)")
        }
    }

    /// Mid-store crash must not brick the lineage: listVersions stays usable,
    /// active pointer is unchanged, a subsequent store succeeds with a sane
    /// version number, and the same holds after reopening the registry root.
    @Test("crash mid-storeCandidate leaves lineage usable for list/store/reopen")
    func crashMidStoreLeavesLineageUsable() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        let v1 = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("w1".utf8),
            trainedOn: sampleWindow
        )
        try await registry.promote(lineage: lineage, version: v1.version)

        await registry.setFaultAfterWeightsWrite(true)
        do {
            _ = try await registry.storeCandidate(
                lineage: lineage,
                weights: Data("partial".utf8),
                trainedOn: sampleWindow
            )
            Issue.record("expected injected fault")
        } catch let error as RegistryTestFault {
            guard case .injected = error else {
                Issue.record("expected RegistryTestFault.injected, got \(error)")
                return
            }
        } catch {
            Issue.record("expected RegistryTestFault, got \(error)")
        }

        // listVersions must not throw; only the previously complete version is visible.
        let listed = try await registry.listVersions(for: lineage)
        #expect(listed.map(\.version) == [1])

        let active = try await registry.activeVersion(for: lineage)
        #expect(active?.version == 1)

        // Subsequent store must succeed with a version number that does not collide.
        let afterCrash = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("after-crash".utf8),
            trainedOn: sampleWindow
        )
        #expect(afterCrash.version >= 2)
        #expect(afterCrash.status == .candidate)

        let listedAfter = try await registry.listVersions(for: lineage)
        #expect(listedAfter.map(\.version).sorted() == [1, afterCrash.version].sorted())
        #expect(listedAfter.count == 2)

        // Simulated process restart on the same root.
        let reopened = try AdapterRegistry(rootURL: root)
        let reopenedListed = try await reopened.listVersions(for: lineage)
        #expect(reopenedListed.map(\.version).sorted() == [1, afterCrash.version].sorted())
        let reopenedActive = try await reopened.activeVersion(for: lineage)
        #expect(reopenedActive?.version == 1)

        // Another store after reopen still works.
        let afterReopen = try await reopened.storeCandidate(
            lineage: lineage,
            weights: Data("after-reopen".utf8),
            trainedOn: sampleWindow
        )
        #expect(afterReopen.version > afterCrash.version)
    }

    /// Incomplete `vN/` left by an older build (no version.json) must be ignored
    /// by listVersions and must not block allocation of a free next version.
    @Test("legacy partial version directory is skipped and does not brick next store")
    func legacyPartialVersionDirectorySelfHeals() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        _ = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("w1".utf8),
            trainedOn: sampleWindow
        )

        // Plant a legacy partial v2/ (weights only, no version.json) as older builds did.
        let lineageDir = root.appendingPathComponent(lineage.lineageID, isDirectory: true)
        let partialV2 = lineageDir.appendingPathComponent("v2", isDirectory: true)
        try FileManager.default.createDirectory(at: partialV2, withIntermediateDirectories: true)
        try Data("orphan-weights".utf8).write(
            to: partialV2.appendingPathComponent("adapters.safetensors")
        )

        let listed = try await registry.listVersions(for: lineage)
        #expect(listed.map(\.version) == [1])

        // Next store must not reuse 2 (would collide) and must succeed.
        let stored = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("w-new".utf8),
            trainedOn: sampleWindow
        )
        #expect(stored.version >= 3)
        #expect(FileManager.default.fileExists(
            atPath: lineageDir.appendingPathComponent("v\(stored.version)").path
        ))
        #expect(FileManager.default.fileExists(
            atPath: lineageDir
                .appendingPathComponent("v\(stored.version)")
                .appendingPathComponent("version.json").path
        ))
    }

    @Test("gc removes leftover staging directories")
    func gcCleansStagingDirectories() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        _ = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("w1".utf8),
            trainedOn: sampleWindow
        )

        // Leave a staging dir as a crashed store would.
        let lineageDir = root.appendingPathComponent(lineage.lineageID, isDirectory: true)
        let staging = lineageDir.appendingPathComponent(
            ".staging-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try Data("junk".utf8).write(to: staging.appendingPathComponent("adapters.safetensors"))

        try await registry.gc(lineage: lineage, keepLast: 10)

        let remaining = try FileManager.default.contentsOfDirectory(
            at: lineageDir,
            includingPropertiesForKeys: nil
        )
        #expect(!remaining.contains { $0.lastPathComponent.hasPrefix(".staging-") })
        #expect(remaining.contains { $0.lastPathComponent == "v1" })
    }

    @Test("crash mid-promote leaves consistent active pointer; list and reopen agree")
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
        } catch let error as RegistryTestFault {
            guard case .injected = error else {
                Issue.record("expected RegistryTestFault.injected, got \(error)")
                return
            }
        } catch {
            Issue.record("expected RegistryTestFault, got \(error)")
        }

        // state.json is the source of truth — active remains v1.
        let after = try await registry.activeVersion(for: lineage)
        #expect(after?.version == 1)

        let all = try await registry.listVersions(for: lineage)
        #expect(all.filter { $0.status == .active }.count == 1)
        #expect(all.first { $0.version == 1 }?.status == .active)

        // Reopen on same root: pointer still v1; promote to v2 still works.
        let reopened = try AdapterRegistry(rootURL: root)
        let reopenedActive = try await reopened.activeVersion(for: lineage)
        #expect(reopenedActive?.version == 1)
        try await reopened.promote(lineage: lineage, version: 2)
        let promoted = try await reopened.activeVersion(for: lineage)
        #expect(promoted?.version == 2)
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

    /// Default `activeVersion` is metadata-only; corrupted weights still return
    /// metadata, while `verifyIntegrity: true` surfaces the mismatch.
    @Test("activeVersion does not hash weights by default")
    func activeVersionSkipsIntegrityByDefault() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        _ = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("original".utf8),
            trainedOn: sampleWindow
        )
        try await registry.promote(lineage: lineage, version: 1)

        let weightsURL = await registry.weightsURL(for: lineage, version: 1)
        try Data("corrupted".utf8).write(to: weightsURL)

        let meta = try await registry.activeVersion(for: lineage)
        #expect(meta?.version == 1)

        do {
            _ = try await registry.activeVersion(for: lineage, verifyIntegrity: true)
            Issue.record("expected integrityMismatch")
        } catch let error as AdaptRegistryError {
            guard case .integrityMismatch = error else {
                Issue.record("expected integrityMismatch, got \(error)")
                return
            }
        } catch {
            Issue.record("expected AdaptRegistryError, got \(error)")
        }

        // Promote must still refuse unverified/corrupt weights.
        do {
            try await registry.promote(lineage: lineage, version: 1)
            Issue.record("expected promote to fail integrity check")
        } catch let error as AdaptRegistryError {
            guard case .integrityMismatch = error else {
                Issue.record("expected integrityMismatch on promote, got \(error)")
                return
            }
        } catch {
            Issue.record("expected AdaptRegistryError, got \(error)")
        }
    }

    @Test("lineageID with path separators is rejected")
    func rejectsPathTraversalLineageID() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let badIDs = [
            "../escape",
            "foo/bar",
            String(repeating: "b", count: 63),  // wrong length (63)
            String(repeating: "b", count: 65),  // wrong length (65)
            String(repeating: "g", count: 64),  // g is not hex
            String(repeating: "A", count: 64),  // uppercase not allowed
            "../../tmp",
            "abc",
        ]
        for bad in badIDs {
            do {
                _ = try await registry.listVersions(lineageID: bad)
                Issue.record("expected invalidLineageID for \(bad)")
            } catch let error as AdaptRegistryError {
                guard case .invalidLineageID = error else {
                    Issue.record("expected invalidLineageID for \(bad), got \(error)")
                    return
                }
            } catch {
                Issue.record("expected AdaptRegistryError for \(bad), got \(error)")
            }
        }

        // Valid 64-char lowercase hex is accepted (empty lineage → []).
        let valid = String(repeating: "ab", count: 32)  // 64 chars of a,b
        let listed = try await registry.listVersions(lineageID: valid)
        #expect(listed.isEmpty)

        // promote/gc/clearActive also reject.
        do {
            try await registry.promote(lineageID: "../x", version: 1)
            Issue.record("expected promote to reject")
        } catch let error as AdaptRegistryError {
            guard case .invalidLineageID = error else {
                Issue.record("expected invalidLineageID on promote, got \(error)")
                return
            }
        }

        do {
            try await registry.gc(lineageID: "not-a-digest", keepLast: 1)
            Issue.record("expected gc to reject")
        } catch let error as AdaptRegistryError {
            guard case .invalidLineageID = error else {
                Issue.record("expected invalidLineageID on gc, got \(error)")
                return
            }
        }
    }

    @Test("recordEvalReport writes measurement without changing status or active")
    func recordEvalReport() async throws {
        let (registry, root) = try makeRegistry()
        defer { teardown(root) }

        let lineage = sampleLineage
        let stored = try await registry.storeCandidate(
            lineage: lineage,
            weights: Data("w-eval".utf8),
            trainedOn: sampleWindow
        )
        #expect(stored.evalReport == nil)

        let report = EvalReport.heldOutCrossEntropy(
            meanNats: 1.25,
            exampleCount: 30,
            supervisedTokenCount: 400
        )
        try await registry.recordEvalReport(
            lineage: lineage,
            version: stored.version,
            report: report
        )

        let reloaded = try await registry.version(
            for: lineage,
            version: stored.version,
            verifyIntegrity: false
        )
        #expect(reloaded.evalReport?.primaryScore == 1.25)
        #expect(reloaded.evalReport?.primaryMetric == EvalReport.metricMeanCrossEntropyNats)
        #expect(reloaded.evalReport?.primaryDirection == .lowerIsBetter)
        #expect(reloaded.evalReport?.exampleCount == 30)
        #expect(reloaded.status == .candidate)
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
