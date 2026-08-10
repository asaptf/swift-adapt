import AdaptCore
import Foundation
import Testing

@Suite("LoRAConfig wire compatibility")
struct LoRAConfigTests {
    @Test("encodes upstream snake_case JSON keys")
    func encodesUpstreamKeys() throws {
        let config = LoRAConfig(
            rank: 8,
            scale: 10.0,
            keys: ["q_proj", "v_proj"],
            numLayers: 16,
            fineTuneType: .lora
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(config)
        let json = try #require(String(data: data, encoding: .utf8))

        #expect(json.contains("\"num_layers\""))
        #expect(json.contains("\"fine_tune_type\""))
        #expect(json.contains("\"lora_parameters\""))
        #expect(json.contains("\"rank\""))
        #expect(json.contains("\"scale\""))
        #expect(json.contains("\"keys\""))
        #expect(!json.contains("\"numLayers\""))
        #expect(!json.contains("\"dropout\""))
        #expect(!json.contains("\"alpha\""))
        #expect(!json.contains("\"targetModules\""))

        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let root = try #require(object)
        #expect(root["num_layers"] as? Int == 16)
        #expect(root["fine_tune_type"] as? String == "lora")
        let params = try #require(root["lora_parameters"] as? [String: Any])
        #expect(params["rank"] as? Int == 8)
        #expect(params["keys"] as? [String] == ["q_proj", "v_proj"])
    }

    @Test("defaults match upstream-friendly values")
    func defaults() {
        let config = LoRAConfig()
        #expect(config.numLayers == 16)
        #expect(config.fineTuneType == .lora)
        #expect(config.loraParameters.rank == 8)
        #expect(config.loraParameters.scale == 10.0)
        #expect(config.loraParameters.keys == nil)
    }

    @Test("round-trips through Codable")
    func roundTrip() throws {
        let original = LoRAConfig(
            rank: 16,
            scale: 32.0,
            keys: nil,
            numLayers: 8,
            fineTuneType: .dora
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(LoRAConfig.self, from: data)
        #expect(decoded == original)
    }
}
