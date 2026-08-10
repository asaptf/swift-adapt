import AdaptCore
import AdaptInference
import AdaptRegistry
import Foundation
import Testing

@Suite("Prompt format mismatch")
struct PromptFormatMismatchTests {
    @Test("adapter trained under chatTemplate is refused by raw session")
    func chatTemplateAdapterRefusedByRawSession() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage

        let stored = try await reg.storeCandidate(
            lineage: lineage,
            weights: Data("tmpl-weights".utf8),
            trainedOn: TrainingWindow(start: Date(), end: Date(), exampleCount: 3),
            promptFormat: .chatTemplate
        )
        try await reg.promote(lineage: lineage, version: stored.version)

        let fake = FakeSessionBackend(promptFormatConvention: .rawConcatenation)
        do {
            _ = try await AdaptSession(
                backend: fake,
                lineage: lineage,
                registry: reg
            )
            Issue.record("expected promptFormatMismatch")
        } catch let error as AdaptInferenceError {
            guard case .promptFormatMismatch(let adapter, let session) = error else {
                Issue.record("expected promptFormatMismatch, got \(error)")
                return
            }
            #expect(adapter == .chatTemplate)
            #expect(session == .rawConcatenation)
        } catch {
            Issue.record("expected AdaptInferenceError, got \(error)")
        }
        #expect(fake.loadCount == 0)
    }

    @Test("adapter trained under raw is refused by chatTemplate session")
    func rawAdapterRefusedByChatSession() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage

        let stored = try await reg.storeCandidate(
            lineage: lineage,
            weights: Data("raw-weights".utf8),
            trainedOn: TrainingWindow(start: Date(), end: Date(), exampleCount: 3),
            promptFormat: .rawConcatenation
        )
        try await reg.promote(lineage: lineage, version: stored.version)

        let fake = FakeSessionBackend(promptFormatConvention: .chatTemplate)
        await #expect(throws: AdaptInferenceError.self) {
            _ = try await AdaptSession(
                backend: fake,
                lineage: lineage,
                registry: reg
            )
        }
        #expect(fake.loadCount == 0)
    }

    @Test("matching conventions load successfully")
    func matchingConventionsLoad() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage

        let stored = try await reg.storeCandidate(
            lineage: lineage,
            weights: Data("ok-weights".utf8),
            trainedOn: TrainingWindow(start: Date(), end: Date(), exampleCount: 1),
            promptFormat: .chatTemplate
        )
        try await reg.promote(lineage: lineage, version: stored.version)
        let dir = await reg.directoryURL(for: lineage, version: stored.version)

        let fake = FakeSessionBackend(
            baseChunks: ["base"],
            adapterChunks: [dir.path: ["adapted"]],
            promptFormatConvention: .chatTemplate
        )
        let session = try await AdaptSession(
            backend: fake,
            lineage: lineage,
            registry: reg
        )
        #expect(await session.loadedVersion == stored.version)
        #expect(fake.loadCount == 1)
    }

    @Test("legacy nil promptFormat is treated as rawConcatenation")
    func legacyNilTreatedAsRaw() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }
        let lineage = InferenceTestSupport.lineage

        // storeCandidate with nil promptFormat (legacy path).
        let stored = try await reg.storeCandidate(
            lineage: lineage,
            weights: Data("legacy-weights".utf8),
            trainedOn: TrainingWindow(start: Date(), end: Date(), exampleCount: 1),
            promptFormat: nil
        )
        try await reg.promote(lineage: lineage, version: stored.version)
        #expect(stored.promptFormat == nil)

        let dir = await reg.directoryURL(for: lineage, version: stored.version)
        let matching = FakeSessionBackend(
            adapterChunks: [dir.path: ["legacy-ok"]],
            promptFormatConvention: .rawConcatenation
        )
        let session = try await AdaptSession(
            backend: matching,
            lineage: lineage,
            registry: reg
        )
        #expect(await session.loadedVersion == stored.version)

        // Chat-template session must refuse legacy raw adapter.
        let mismatch = FakeSessionBackend(promptFormatConvention: .chatTemplate)
        do {
            _ = try await AdaptSession(
                backend: mismatch,
                lineage: lineage,
                registry: reg
            )
            Issue.record("expected mismatch for legacy raw vs chat session")
        } catch let error as AdaptInferenceError {
            guard case .promptFormatMismatch(let adapter, let sessionFormat) = error else {
                Issue.record("expected promptFormatMismatch, got \(error)")
                return
            }
            #expect(adapter == .rawConcatenation)
            #expect(sessionFormat == .chatTemplate)
        }
    }
}
