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
            keys: ["self_attn.q_proj", "self_attn.v_proj"],
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
        #expect(params["keys"] as? [String] == ["self_attn.q_proj", "self_attn.v_proj"])
    }

    @Test("defaults are attention-only (explicit, not model-inherited)")
    func defaults() {
        let config = LoRAConfig()
        #expect(config.numLayers == 16)
        #expect(config.fineTuneType == .lora)
        #expect(config.loraParameters.rank == 8)
        #expect(config.loraParameters.scale == 10.0)
        #expect(config.loraParameters.keys == LoRAConfig.defaultAttentionKeys)
        #expect(
            config.loraParameters.keys
                == [
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                ]
        )
        #expect(
            config.keysDescription
                == "self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj"
        )
    }

    @Test("explicit key set is honoured")
    func explicitKeysHonoured() {
        let narrow = LoRAConfig(rank: 8, keys: ["self_attn.q_proj", "self_attn.v_proj"])
        #expect(narrow.loraParameters.keys == ["self_attn.q_proj", "self_attn.v_proj"])

        let wide = LoRAConfig(rank: 8, keys: LoRAConfig.allProjectionKeys)
        #expect(wide.loraParameters.keys == LoRAConfig.allProjectionKeys)
        #expect(wide.loraParameters.keys?.contains("mlp.gate_proj") == true)
        #expect(wide.loraParameters.keys?.contains("mlp.down_proj") == true)

        let inherit = LoRAConfig(rank: 8, keys: nil)
        #expect(inherit.loraParameters.keys == nil)
        #expect(inherit.keysDescription == "(model defaults)")
    }

    @Test("defaultAttentionKeys is a strict subset of allProjectionKeys")
    func attentionSubsetOfAll() {
        let attention = Set(LoRAConfig.defaultAttentionKeys)
        let all = Set(LoRAConfig.allProjectionKeys)
        #expect(attention.isSubset(of: all))
        #expect(attention.count == 4)
        #expect(all.count == 7)
        #expect(!attention.contains("mlp.gate_proj"))
        #expect(all.contains("mlp.up_proj"))
        // Layer-relative paths (mlx-swift-lm namedModules), not bare names.
        #expect(attention.allSatisfy { $0.hasPrefix("self_attn.") })
        #expect(all.contains(where: { $0.hasPrefix("mlp.") }))
    }

    @Test("round-trips through Codable")
    func roundTrip() throws {
        let original = LoRAConfig(
            rank: 16,
            scale: 32.0,
            keys: LoRAConfig.defaultAttentionKeys,
            numLayers: 8,
            fineTuneType: .dora
        )
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(LoRAConfig.self, from: data)
        #expect(decoded == original)
    }

    @Test("legacy adapter metadata with keys null still decodes")
    func legacyKeysNullDecodes() throws {
        // Pre-change on-disk shape: keys omitted or JSON null → Swift nil.
        let withNull = """
            {
              "num_layers": 16,
              "fine_tune_type": "lora",
              "lora_parameters": {
                "rank": 8,
                "scale": 10.0,
                "keys": null
              }
            }
            """.data(using: .utf8)!
        let decodedNull = try JSONDecoder().decode(LoRAConfig.self, from: withNull)
        #expect(decodedNull.loraParameters.keys == nil)
        #expect(decodedNull.loraParameters.rank == 8)
        #expect(decodedNull.numLayers == 16)

        // keys field entirely absent (older writers that never emitted it).
        let omitted = """
            {
              "num_layers": 8,
              "fine_tune_type": "lora",
              "lora_parameters": {
                "rank": 16,
                "scale": 20.0
              }
            }
            """.data(using: .utf8)!
        let decodedOmitted = try JSONDecoder().decode(LoRAConfig.self, from: omitted)
        #expect(decodedOmitted.loraParameters.keys == nil)
        #expect(decodedOmitted.loraParameters.rank == 16)
        #expect(decodedOmitted.numLayers == 8)
    }

    @Test("resolved keys are recorded in encoded adapter_config shape")
    func resolvedKeysRecordedInJSON() throws {
        let config = LoRAConfig()
        let data = try JSONEncoder().encode(config)
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let params = try #require(object["lora_parameters"] as? [String: Any])
        let keys = try #require(params["keys"] as? [String])
        #expect(keys == LoRAConfig.defaultAttentionKeys)
        // Not null — operators reading adapter_config.json see the real set.
        #expect(!(params["keys"] is NSNull))
    }
}
