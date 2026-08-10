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
            createdAt: Date(timeIntervalSince1970: 1_700_100_000),
            promptFormat: .chatTemplate
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(version)
        let json = try #require(String(data: data, encoding: .utf8))

        #expect(!json.contains(secretPrompt))
        #expect(!json.contains(secretCompletion))
        // Metadata may name the *format convention* (`promptFormat`) but must never
        // embed training fields (`"prompt"` / `"completion"` as JSON keys for text).
        #expect(!json.contains("\"prompt\""))
        #expect(!json.contains("\"completion\""))
        #expect(!json.contains(secretPrompt))
        #expect(json.contains("exampleCount") || json.contains("example_count") || json.contains("42"))
    }

    @Test("EvalReport decodes minimal legacy JSON")
    func evalReportLegacyDecode() throws {
        let minimal = Data(#"{}"#.utf8)
        let report = try JSONDecoder().decode(EvalReport.self, from: minimal)
        #expect(report.primaryScore == nil)
        #expect(report.passedGate == nil)
        #expect(report.notes == nil)
        #expect(report.primaryMetric == nil)
        #expect(report.primaryDirection == nil)
        #expect(report.exampleCount == nil)
        #expect(report.supervisedTokenCount == nil)

        let partial = Data(#"{"primaryScore":0.5}"#.utf8)
        let partialReport = try JSONDecoder().decode(EvalReport.self, from: partial)
        #expect(partialReport.primaryScore == 0.5)
        #expect(partialReport.passedGate == nil)
        #expect(partialReport.primaryMetric == nil)

        // Unknown keys must not cause failure.
        let withUnknown = Data(#"{"primaryScore":1.0,"futureMetric":99,"nested":{"x":1}}"#.utf8)
        let future = try JSONDecoder().decode(EvalReport.self, from: withUnknown)
        #expect(future.primaryScore == 1.0)
    }

    @Test("EvalReport round-trips held-out CE measurement fields")
    func evalReportHeldOutRoundTrip() throws {
        let report = EvalReport.heldOutCrossEntropy(
            meanNats: 2.345678,
            exampleCount: 30,
            supervisedTokenCount: 412,
            notes: "measurement only"
        )
        let data = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(EvalReport.self, from: data)
        #expect(decoded.primaryScore == 2.345678)
        #expect(decoded.primaryMetric == EvalReport.metricMeanCrossEntropyNats)
        #expect(decoded.primaryDirection == .lowerIsBetter)
        #expect(decoded.exampleCount == 30)
        #expect(decoded.supervisedTokenCount == 412)
        #expect(decoded.passedGate == nil)
        #expect(decoded.notes == "measurement only")
    }

    @Test("EvalReport gateDecision drives passedGate and round-trips")
    func evalReportGateDecision() throws {
        let promote = EvalReport(
            primaryScore: 1.2,
            gateDecision: .promote,
            wilcoxonPValue: 0.01,
            effectSize: 0.8
        )
        #expect(promote.passedGate == true)
        #expect(promote.gateDecision == .promote)

        let refuse = EvalReport(gateDecision: .refuse)
        #expect(refuse.passedGate == false)

        let abstain = EvalReport(gateDecision: .abstain)
        #expect(abstain.passedGate == nil)

        let encoder = JSONEncoder()
        let data = try encoder.encode(promote)
        let decoded = try JSONDecoder().decode(EvalReport.self, from: data)
        #expect(decoded.gateDecision == .promote)
        #expect(decoded.passedGate == true)
        #expect(decoded.wilcoxonPValue == 0.01)
        #expect(decoded.effectSize == 0.8)

        // Legacy JSON without gate fields still decodes.
        let minimal = Data(#"{}"#.utf8)
        let legacy = try JSONDecoder().decode(EvalReport.self, from: minimal)
        #expect(legacy.gateDecision == nil)
        #expect(legacy.wilcoxonPValue == nil)
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
