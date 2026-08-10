import AdaptCore
import AdaptData
import AdaptMacros
import Foundation
import Testing

// MARK: - Fixture (real macro expansion at compile time)

@Personalizable(task: "email-style")
struct EmailDraft {
    @Prompt var context: String
    @Completion var body: String
}

@Personalizable(task: "multi")
struct MultiPromptDraft {
    @Prompt var a: String
    @Prompt var b: String
    @Completion var out: String
}

@Suite("PersonalizationSignal capture → AdaptData")
struct CaptureIntegrationTests {

    @Test("capture scrubs PII before persistence")
    func captureScrubsPII() async throws {
        let (buffer, root) = try makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxExamples: 100,
                maxCapturesPerDay: 50,
                scrubberPipeline: .builtins
            )
        )
        defer { teardown(root) }

        let draft = EmailDraft(
            context: "Reply to alice@example.com about Q3",
            body: "Thanks Alice — call me at 415-555-0100."
        )

        let result = try await EmailDraft.capture(draft, into: buffer, source: .explicitEdit).value
        guard case .inserted(let id) = result else {
            Issue.record("expected inserted, got \(result)")
            return
        }

        let examples = try await buffer.examples()
        #expect(examples.count == 1)
        let stored = try #require(examples.first)
        #expect(stored.id == id)
        #expect(!stored.prompt.contains("alice@example.com"))
        #expect(stored.prompt.contains("[EMAIL]"))
        #expect(!stored.completion.contains("415-555-0100"))
        // Built-in phone scrubber replaces digit runs with a token.
        #expect(stored.completion.contains("[PHONE]") || !stored.completion.contains("555"))

        // Raw PII must not appear as UTF-8 anywhere in the database file.
        #expect(try !fileContainsASCII("alice@example.com", fileURL: buffer.databaseURL))
        #expect(try !fileContainsASCII("415-555-0100", fileURL: buffer.databaseURL))
    }

    @Test("capture respects privacy budget")
    func captureRespectsBudget() async throws {
        let limit = 2
        let (buffer, root) = try makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxExamples: 100,
                maxCapturesPerDay: limit,
                scrubberPipeline: ScrubberPipeline(scrubbers: [])
            )
        )
        defer { teardown(root) }

        for i in 0..<limit {
            let draft = EmailDraft(context: "p\(i)", body: "c\(i)")
            let result = try await EmailDraft.capture(draft, into: buffer, source: .acceptance).value
            guard case .inserted = result else {
                Issue.record("expected insert \(i)")
                return
            }
        }

        let overflow = EmailDraft(context: "overflow", body: "nope")
        do {
            _ = try await EmailDraft.capture(overflow, into: buffer).value
            Issue.record("expected privacyBudgetExceeded")
        } catch let error as AdaptDataError {
            guard case .privacyBudgetExceeded = error else {
                Issue.record("wrong AdaptDataError: \(error)")
                return
            }
        }

        let stats = try await buffer.stats()
        #expect(stats.exampleCount == limit)
        #expect(stats.capturesToday == limit)
    }

    @Test("makeTrainingExample joins multiple prompts and uses source weight")
    func makeTrainingExampleSurface() {
        let value = MultiPromptDraft(a: "hello", b: "world", out: "done")
        #expect(MultiPromptDraft.personalizationTaskID == "multi")

        let example = value.makeTrainingExample(source: .acceptance)
        #expect(example.prompt == "hello\n\nworld")
        #expect(example.completion == "done")
        #expect(example.source == .acceptance)
        #expect(example.weight == SignalSource.acceptance.defaultWeight)
    }

    @Test("capture is scheduled off the caller's stack (Task)")
    func captureReturnsTask() async throws {
        let (buffer, root) = try makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxExamples: 10,
                maxCapturesPerDay: 10,
                scrubberPipeline: ScrubberPipeline(scrubbers: [])
            )
        )
        defer { teardown(root) }

        let draft = EmailDraft(context: "ctx", body: "body")
        let task: Task<ReplayBuffer.AddResult, Error> = EmailDraft.capture(draft, into: buffer)
        let result = try await task.value
        guard case .inserted = result else {
            Issue.record("expected inserted")
            return
        }
    }
}

// MARK: - Local helpers (no dependency on AdaptDataTests)

private func makeLineage(taskID: String = "email-style") -> AdapterLineage {
    AdapterLineage(
        taskID: taskID,
        baseModelID: "mlx-community/Qwen3-0.6B-4bit",
        loraConfig: LoRAConfig()
    )
}

private func makeTempRoot() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("AdaptMacrosTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func teardown(_ url: URL) {
    try? FileManager.default.removeItem(at: url)
}

private func makeBuffer(
    configuration: ReplayBufferConfiguration
) throws -> (ReplayBuffer, URL) {
    let root = try makeTempRoot()
    let buffer = try ReplayBuffer(
        lineage: makeLineage(),
        rootURL: root,
        configuration: configuration
    )
    return (buffer, root)
}

private func fileContainsASCII(_ needle: String, fileURL: URL) throws -> Bool {
    let data = try Data(contentsOf: fileURL)
    guard let needleData = needle.data(using: .utf8), !needleData.isEmpty else {
        return false
    }
    return data.range(of: needleData) != nil
}
