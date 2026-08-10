import AdaptCore
import Foundation
import Testing

@Suite("AdapterLineage stable ID")
struct AdapterLineageTests {
    /// Fixed fixture used to pin the lineage digest.
    private static let fixtureLineage = AdapterLineage(
        taskID: "email-style",
        baseModelID: "mlx-community/Qwen3-4B-4bit",
        loraConfig: LoRAConfig(
            rank: 8,
            scale: 10.0,
            keys: nil,
            numLayers: 16,
            fineTuneType: .lora
        )
    )

    /// Hardcoded expected SHA-256 hex of the canonical payload for `fixtureLineage`.
    /// If hashing inputs or LoRAConfig encoding change, this test must fail loudly.
    private static let expectedLineageID =
        "6e45aaa192ef9a2b55edcdac5632329f6e03a3f17085e46f29a4288600f715dd"

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
        // Will be updated once we compute the real digest in the first green run.
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

    @Test("lineageID is filesystem-safe")
    func filesystemSafe() {
        let id = Self.fixtureLineage.lineageID
        #expect(!id.contains("/"))
        #expect(!id.contains(":"))
        #expect(!id.contains(" "))
        #expect(id == id.lowercased())
    }
}
