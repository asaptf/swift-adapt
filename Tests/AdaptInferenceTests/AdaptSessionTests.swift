import AdaptCore
import AdaptInference
import AdaptRegistry
import Foundation
import Testing

@Suite("AdaptSession")
struct AdaptSessionTests {

    // MARK: - Active load + integrity

    @Test("active adapter is loaded and applied on init")
    func loadsActiveAdapter() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage
        let weights = Data("adapter-v1-weights".utf8)
        let stored = try await InferenceTestSupport.store(
            registry: reg,
            lineage: lineage,
            weights: weights,
            promote: true
        )
        let dir = await reg.directoryURL(for: lineage, version: stored.version)

        let fake = FakeSessionBackend(
            baseChunks: ["base-out"],
            adapterChunks: [dir.path: ["adapted-out"]]
        )
        let session = try await AdaptSession(
            backend: fake,
            lineage: lineage,
            registry: reg
        )

        #expect(await session.loadedVersion == stored.version)
        #expect(fake.loadCount == 1)
        #expect(fake.loadedDirectory?.path == dir.path)

        let text = try await session.generateText(prompt: "hi")
        #expect(text == "adapted-out")
        #expect(fake.generateCount == 1)
    }

    @Test("digest mismatch is refused with typed integrity error")
    func digestMismatchRefused() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage
        let stored = try await InferenceTestSupport.store(
            registry: reg,
            lineage: lineage,
            weights: Data("good-weights".utf8),
            promote: true
        )
        // Corrupt after promote (promote verified the original digest).
        try await InferenceTestSupport.corruptWeights(
            registry: reg,
            lineage: lineage,
            version: stored.version
        )

        let fake = FakeSessionBackend()
        await #expect(throws: AdaptInferenceError.self) {
            _ = try await AdaptSession(
                backend: fake,
                lineage: lineage,
                registry: reg
            )
        }
        // Backend must never have been asked to load corrupt weights.
        #expect(fake.loadCount == 0)

        // Typed case check.
        do {
            _ = try await AdaptSession(
                backend: FakeSessionBackend(),
                lineage: lineage,
                registry: reg
            )
            Issue.record("expected integrity error")
        } catch let error as AdaptInferenceError {
            guard case .integrityMismatch = error else {
                Issue.record("expected integrityMismatch, got \(error)")
                return
            }
        } catch {
            Issue.record("expected AdaptInferenceError, got \(error)")
        }
    }

    // MARK: - Zero-config base path

    @Test("no active version → base-model generation works")
    func zeroConfigBasePath() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage

        let fake = FakeSessionBackend(baseChunks: ["cold", "start"])
        let session = try await AdaptSession(
            backend: fake,
            lineage: lineage,
            registry: reg
        )

        #expect(await session.loadedVersion == nil)
        #expect(fake.loadCount == 0)
        let text = try await session.generateText(prompt: "x")
        #expect(text == "coldstart")
    }

    // MARK: - useBaseModel (base-vs-adapter without registry thrash)

    @Test("useBaseModel unloads adapter without changing registry active pointer")
    func useBaseModelUnloadsWithoutRegistryChange() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage
        let stored = try await InferenceTestSupport.store(
            registry: reg,
            lineage: lineage,
            weights: Data("adapter-base-swap".utf8),
            promote: true
        )
        let dir = await reg.directoryURL(for: lineage, version: stored.version)

        let fake = FakeSessionBackend(
            baseChunks: ["base-only"],
            adapterChunks: [dir.path: ["with-adapter"]]
        )
        let session = try await AdaptSession(
            backend: fake,
            lineage: lineage,
            registry: reg
        )
        #expect(await session.loadedVersion == stored.version)
        #expect(try await session.generateText(prompt: "p") == "with-adapter")

        try await session.useBaseModel()
        #expect(await session.loadedVersion == nil)
        #expect(fake.unloadCount >= 1)
        #expect(try await session.generateText(prompt: "p") == "base-only")

        // Registry active pointer untouched — reload re-applies the same adapter.
        let active = try await reg.activeVersion(for: lineage, verifyIntegrity: false)
        #expect(active?.version == stored.version)
        try await session.reload()
        #expect(await session.loadedVersion == stored.version)
        #expect(try await session.generateText(prompt: "p") == "with-adapter")
    }

    // MARK: - reload without model reload

    @Test("reload after promotion swaps adapter without reloading the model")
    func reloadAfterPromotion() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage

        let v1 = try await InferenceTestSupport.store(
            registry: reg,
            lineage: lineage,
            weights: Data("weights-v1".utf8),
            promote: true
        )
        let dir1 = await reg.directoryURL(for: lineage, version: v1.version)

        let v2 = try await InferenceTestSupport.store(
            registry: reg,
            lineage: lineage,
            weights: Data("weights-v2".utf8),
            promote: false
        )
        let dir2 = await reg.directoryURL(for: lineage, version: v2.version)

        let modelID = UUID()
        let fake = FakeSessionBackend(
            baseChunks: ["base"],
            adapterChunks: [
                dir1.path: ["from-v1"],
                dir2.path: ["from-v2"],
            ],
            modelInstanceID: modelID
        )
        let session = try await AdaptSession(
            backend: fake,
            lineage: lineage,
            registry: reg
        )
        #expect(await session.loadedVersion == v1.version)
        #expect(try await session.generateText(prompt: "p") == "from-v1")

        // Promote v2, then reload — must swap adapters, keep same model instance.
        try await reg.promote(lineage: lineage, version: v2.version)
        try await session.reload()

        #expect(await session.loadedVersion == v2.version)
        #expect(await session.modelInstanceID == modelID)
        #expect(fake.loadCount == 2)
        #expect(fake.unloadCount >= 1)
        #expect(try await session.generateText(prompt: "p") == "from-v2")
    }

    @Test("rollback then reload returns to previous adapter")
    func rollbackThenReload() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage

        let v1 = try await InferenceTestSupport.store(
            registry: reg,
            lineage: lineage,
            weights: Data("weights-v1".utf8),
            promote: true
        )
        let dir1 = await reg.directoryURL(for: lineage, version: v1.version)

        let v2 = try await InferenceTestSupport.store(
            registry: reg,
            lineage: lineage,
            weights: Data("weights-v2".utf8),
            promote: true
        )
        let dir2 = await reg.directoryURL(for: lineage, version: v2.version)

        let fake = FakeSessionBackend(
            adapterChunks: [
                dir1.path: ["v1-out"],
                dir2.path: ["v2-out"],
            ]
        )
        let session = try await AdaptSession(
            backend: fake,
            lineage: lineage,
            registry: reg
        )
        #expect(await session.loadedVersion == v2.version)
        #expect(try await session.generateText(prompt: "p") == "v2-out")

        try await reg.rollback(lineage: lineage, to: v1.version)
        try await session.reload()

        #expect(await session.loadedVersion == v1.version)
        #expect(try await session.generateText(prompt: "p") == "v1-out")
    }

    // MARK: - Cancellation

    @Test("cancellation mid-generation stops promptly and session remains reusable")
    func cancellationThenReuse() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage

        // Slow chunks so we can cancel mid-stream.
        let fake = FakeSessionBackend(
            baseChunks: ["a", "b", "c", "d", "e", "f", "g", "h"],
            chunkDelayNanoseconds: 50_000_000 // 50 ms
        )
        let session = try await AdaptSession(
            backend: fake,
            lineage: lineage,
            registry: reg
        )

        let collectTask = Task {
            var parts: [String] = []
            for try await chunk in await session.generate(prompt: "slow") {
                parts.append(chunk)
                if parts.count == 2 {
                    break // leave the for-await; termination cancels generation
                }
            }
            return parts
        }
        let partial = try await collectTask.value
        #expect(partial.count == 2)
        #expect(partial == ["a", "b"])

        // Session still works for a full generation afterwards.
        // Speed up remaining tests on this backend.
        fake.chunkDelayNanoseconds = 0
        let full = try await session.generateText(prompt: "again")
        #expect(full == "abcdefgh")
        #expect(fake.generateCount == 2)
    }

    @Test("reload waits for in-flight generation before swapping")
    func reloadWaitsForGeneration() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage

        let v1 = try await InferenceTestSupport.store(
            registry: reg,
            lineage: lineage,
            weights: Data("w1".utf8),
            promote: true
        )
        let dir1 = await reg.directoryURL(for: lineage, version: v1.version)
        let v2 = try await InferenceTestSupport.store(
            registry: reg,
            lineage: lineage,
            weights: Data("w2".utf8),
            promote: false
        )
        let dir2 = await reg.directoryURL(for: lineage, version: v2.version)

        let fake = FakeSessionBackend(
            adapterChunks: [
                dir1.path: ["one", "two", "three"],
                dir2.path: ["NEW"],
            ],
            chunkDelayNanoseconds: 30_000_000
        )
        let session = try await AdaptSession(
            backend: fake,
            lineage: lineage,
            registry: reg
        )

        try await reg.promote(lineage: lineage, version: v2.version)

        // Start a generation that will hold the in-flight counter.
        let genTask = Task {
            var out = ""
            for try await chunk in await session.generate(prompt: "hold") {
                out += chunk
            }
            return out
        }

        // Give generation time to enter beginGeneration.
        try await Task.sleep(nanoseconds: 20_000_000)

        let reloadTask = Task {
            try await session.reload()
        }

        let generated = try await genTask.value
        // In-flight stream must have used v1 (adapter bound at start).
        #expect(generated == "onetwothree")

        try await reloadTask.value
        #expect(await session.loadedVersion == v2.version)
        fake.chunkDelayNanoseconds = 0
        #expect(try await session.generateText(prompt: "x") == "NEW")
    }

    @Test("fuse marks session immutable for reload")
    func fuseBlocksReload() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage
        let stored = try await InferenceTestSupport.store(
            registry: reg,
            lineage: lineage,
            weights: Data("fuse-me".utf8),
            promote: true
        )
        let dir = await reg.directoryURL(for: lineage, version: stored.version)
        let fake = FakeSessionBackend(adapterChunks: [dir.path: ["fused"]])
        let session = try await AdaptSession(
            backend: fake,
            lineage: lineage,
            registry: reg
        )
        try await session.fuse()
        #expect(await session.isFused)

        await #expect(throws: AdaptInferenceError.self) {
            try await session.reload()
        }
    }
}
