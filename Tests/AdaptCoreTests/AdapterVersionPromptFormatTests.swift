import AdaptCore
import Foundation
import Testing

@Suite("AdapterVersion promptFormat forward-compat")
struct AdapterVersionPromptFormatTests {
    private func sampleVersion(promptFormat: PromptFormatConvention? = nil) -> AdapterVersion {
        AdapterVersion(
            lineage: AdapterLineage(
                taskID: "email-style",
                baseModelID: "mlx-community/Qwen3-4B-4bit",
                loraConfig: LoRAConfig()
            ),
            version: 1,
            parentVersion: nil,
            trainedOn: TrainingWindow(
                start: Date(timeIntervalSince1970: 1_700_000_000),
                end: Date(timeIntervalSince1970: 1_700_086_400),
                exampleCount: 10
            ),
            evalReport: nil,
            status: .candidate,
            weightsDigest: String(repeating: "cd", count: 32),
            createdAt: Date(timeIntervalSince1970: 1_700_100_000),
            promptFormat: promptFormat
        )
    }

    @Test("legacy metadata without promptFormat still decodes")
    func legacyDecodeWithoutField() throws {
        // Minimal pre-field shape (no promptFormat key).
        let legacyJSON = """
            {
              "lineage": {
                "taskID": "email-style",
                "baseModelID": "mlx-community/Qwen3-4B-4bit",
                "loraConfig": {
                  "num_layers": 16,
                  "fine_tune_type": "lora",
                  "lora_parameters": { "rank": 8, "scale": 10 }
                }
              },
              "version": 2,
              "trainedOn": {
                "start": "2023-11-14T22:13:20Z",
                "end": "2023-11-15T22:13:20Z",
                "exampleCount": 5
              },
              "status": "candidate",
              "weightsDigest": "\(String(repeating: "ab", count: 32))",
              "createdAt": "2023-11-16T00:00:00Z"
            }
            """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let version = try decoder.decode(AdapterVersion.self, from: Data(legacyJSON.utf8))
        #expect(version.promptFormat == nil)
        #expect(SFTPromptFormatter.convention(fromStored: version.promptFormat) == .rawConcatenation)
        #expect(version.version == 2)
    }

    @Test("promptFormat round-trips through Codable")
    func promptFormatRoundTrip() throws {
        let original = sampleVersion(promptFormat: .chatTemplate)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(original)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(AdapterVersion.self, from: data)
        #expect(decoded.promptFormat == .chatTemplate)

        let json = try #require(String(data: data, encoding: .utf8))
        #expect(json.contains("promptFormat") || json.contains("chatTemplate"))
    }

    @Test("with(status:) preserves promptFormat")
    func withStatusPreservesFormat() {
        let v = sampleVersion(promptFormat: .chatTemplate)
        let promoted = v.with(status: .active)
        #expect(promoted.promptFormat == .chatTemplate)
        #expect(promoted.status == .active)
    }
}
