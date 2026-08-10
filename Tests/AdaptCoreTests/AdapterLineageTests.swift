import AdaptCore
import Foundation
import Testing

@Suite("AdapterLineage stable ID")
struct AdapterLineageTests {
    /// Fixed fixture used to pin the lineage digest (default attention-only keys).
    private static let fixtureLineage = AdapterLineage(
        taskID: "email-style",
        baseModelID: "mlx-community/Qwen3-4B-4bit",
        loraConfig: LoRAConfig(
            rank: 8,
            scale: 10.0,
            keys: LoRAConfig.defaultAttentionKeys,
            numLayers: 16,
            fineTuneType: .lora
        )
    )

    /// Hardcoded expected SHA-256 hex of the canonical payload for `fixtureLineage`.
    /// If hashing inputs or LoRAConfig encoding change, this test must fail loudly.
    ///
    /// Note: changing the default `keys` from `nil` to attention-only correctly
    /// changes lineage identity (a different key set is a different lineage).
    private static let expectedLineageID =
        "4bd8eff834adbfe272a87d435bb0244b4c8203199e59bbe2945c19a278069ba0"

    @Test("lineageID is stable and matches hardcoded digest")
    func stableHardcodedDigest() {
        let id1 = Self.fixtureLineage.lineageID
        let id2 = AdapterLineage(
            taskID: "email-style",
            baseModelID: "mlx-community/Qwen3-4B-4bit",
            loraConfig: LoRAConfig()
        ).lineageID

        #expect(id1 == id2)
        #expect(id1.count == 64)
        #expect(id1.allSatisfy { $0.isHexDigit })
        #expect(id1 == Self.expectedLineageID)
    }

    @Test("different content yields different lineage IDs")
    func differentContentDifferentID() {
        let a = AdapterLineage(taskID: "email-style", baseModelID: "model-a")
        let b = AdapterLineage(taskID: "email-style", baseModelID: "model-b")
        let c = AdapterLineage(taskID: "chat-style", baseModelID: "model-a")
        let d = AdapterLineage(
            taskID: "email-style",
            baseModelID: "model-a",
            loraConfig: LoRAConfig(rank: 16)
        )
        #expect(a.lineageID != b.lineageID)
        #expect(a.lineageID != c.lineageID)
        #expect(a.lineageID != d.lineageID)
    }

    /// Distinct LoRAConfig values must never share a lineageID (would silently
    /// merge on-disk weight trees). Covers rank, scale, layers, type, and keys.
    @Test("distinct LoRAConfigs never collide on lineageID")
    func distinctLoRAConfigsNeverCollide() {
        let baseTask = "email-style"
        let baseModel = "model-a"
        let configs: [LoRAConfig] = [
            LoRAConfig(),
            LoRAConfig(rank: 16),
            LoRAConfig(rank: 8, scale: 20.0),
            LoRAConfig(rank: 8, scale: 10.0, keys: ["attn.q"]),
            LoRAConfig(rank: 8, scale: 10.0, keys: nil, numLayers: 8),
            LoRAConfig(rank: 8, scale: 10.0, keys: nil, numLayers: 16, fineTuneType: .dora),
            LoRAConfig(rank: 8, scale: 10.0, keys: LoRAConfig.allProjectionKeys),
            // Explicit nil keys (model defaults) must differ from attention default.
            LoRAConfig(rank: 8, scale: 10.0, keys: nil, numLayers: 16),
        ]
        let ids = configs.map {
            AdapterLineage(taskID: baseTask, baseModelID: baseModel, loraConfig: $0).lineageID
        }
        #expect(Set(ids).count == ids.count)
    }

    @Test("key set change changes lineageID")
    func keySetChangesLineage() {
        let attention = AdapterLineage(
            taskID: "t",
            baseModelID: "m",
            loraConfig: LoRAConfig(rank: 8, keys: LoRAConfig.defaultAttentionKeys)
        )
        let wide = AdapterLineage(
            taskID: "t",
            baseModelID: "m",
            loraConfig: LoRAConfig(rank: 8, keys: LoRAConfig.allProjectionKeys)
        )
        let modelDefault = AdapterLineage(
            taskID: "t",
            baseModelID: "m",
            loraConfig: LoRAConfig(rank: 8, keys: nil)
        )
        #expect(attention.lineageID != wide.lineageID)
        #expect(attention.lineageID != modelDefault.lineageID)
        #expect(wide.lineageID != modelDefault.lineageID)
    }

    @Test("lineageID is filesystem-safe")
    func filesystemSafe() {
        let id = Self.fixtureLineage.lineageID
        #expect(!id.contains("/"))
        #expect(!id.contains(":"))
        #expect(!id.contains(" "))
        #expect(id == id.lowercased())
    }
}
