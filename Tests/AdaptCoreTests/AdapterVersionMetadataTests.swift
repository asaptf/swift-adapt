import AdaptCore
import Foundation
import Testing

@Suite("Adapter metadata privacy & forward-compat")
struct AdapterVersionMetadataTests {
    @Test("encoded AdapterVersion never contains prompt/completion text")
    func noUserDataInMetadata() throws {
        let secretPrompt = "SECRET_PROMPT_TEXT_XYZ_NEVER_IN_JSON"
        let secretCompletion = "SECRET_COMPLETION_TEXT_ABC_NEVER_IN_JSON"

        // Construct a fully-populated version. Training content stays in TrainingExample only.
        _ = TrainingExample(
            prompt: secretPrompt,
            completion: secretCompletion,
            weight: 1.0,
            source: .explicitEdit
        )

        let lineage = AdapterLineage(
            taskID: "email-style",
            baseModelID: "mlx-community/Qwen3-4B-4bit",
            loraConfig: LoRAConfig()
        )
        let version = AdapterVersion(
            lineage: lineage,
            version: 1,
            parentVersion: nil,
            trainedOn: TrainingWindow(
                start: Date(timeIntervalSince1970: 1_700_000_000),
                end: Date(timeIntervalSince1970: 1_700_086_400),
                exampleCount: 42
            ),
            evalReport: EvalReport(primaryScore: 1.23, passedGate: true, notes: "ok"),
            status: .candidate,
            weightsDigest: String(repeating: "ab", count: 32),
            createdAt: Date(timeIntervalSince1970: 1_700_100_000)
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(version)
        let json = try #require(String(data: data, encoding: .utf8))

        #expect(!json.contains(secretPrompt))
        #expect(!json.contains(secretCompletion))
        #expect(!json.contains("prompt"))
        #expect(!json.contains("completion"))
        #expect(json.contains("exampleCount") || json.contains("example_count") || json.contains("42"))
    }

    @Test("EvalReport decodes minimal legacy JSON")
    func evalReportLegacyDecode() throws {
        let minimal = Data(#"{}"#.utf8)
        let report = try JSONDecoder().decode(EvalReport.self, from: minimal)
        #expect(report.primaryScore == nil)
        #expect(report.passedGate == nil)
        #expect(report.notes == nil)

        let partial = Data(#"{"primaryScore":0.5}"#.utf8)
        let partialReport = try JSONDecoder().decode(EvalReport.self, from: partial)
        #expect(partialReport.primaryScore == 0.5)
        #expect(partialReport.passedGate == nil)

        // Unknown keys must not cause failure.
        let withUnknown = Data(#"{"primaryScore":1.0,"futureMetric":99,"nested":{"x":1}}"#.utf8)
        let future = try JSONDecoder().decode(EvalReport.self, from: withUnknown)
        #expect(future.primaryScore == 1.0)
    }

    @Test("TrainingWindow holds only metadata")
    func trainingWindowMetadataOnly() throws {
        let window = TrainingWindow(
            start: Date(timeIntervalSince1970: 0),
            end: Date(timeIntervalSince1970: 100),
            exampleCount: 7
        )
        let data = try JSONEncoder().encode(window)
        let json = try #require(String(data: data, encoding: .utf8))
        #expect(json.contains("7") || json.contains("exampleCount"))
        #expect(!json.contains("prompt"))
        #expect(!json.contains("completion"))
    }
}
