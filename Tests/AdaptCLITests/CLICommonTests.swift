import AdaptCLI
import AdaptCore
import Foundation
import Testing

@Suite("CLICommon")
struct CLICommonTests {

    @Test("makeLineage rejects empty task")
    func emptyTask() {
        do {
            _ = try CLICommon.makeLineage(
                taskID: "",
                modelID: "m",
                rank: 8,
                scale: 10,
                numLayers: 8
            )
            Issue.record("expected throw")
        } catch let error as AdaptCLIError {
            #expect(error.localizedDescription.contains("task"))
        } catch {
            Issue.record("unexpected \(error)")
        }
    }

    @Test("makeLineage rejects non-positive rank")
    func badRank() {
        do {
            _ = try CLICommon.makeLineage(
                taskID: "t",
                modelID: "m",
                rank: 0,
                scale: 10,
                numLayers: 8
            )
            Issue.record("expected throw")
        } catch let error as AdaptCLIError {
            #expect(error.localizedDescription.contains("rank"))
        } catch {
            Issue.record("unexpected \(error)")
        }
    }

    @Test("makeLineage is stable for same inputs")
    func stableLineageID() throws {
        let a = try CLICommon.makeLineage(
            taskID: "style-mirror",
            modelID: CLICommon.defaultModelID,
            rank: 8,
            scale: 10,
            numLayers: 8
        )
        let b = try CLICommon.makeLineage(
            taskID: "style-mirror",
            modelID: CLICommon.defaultModelID,
            rank: 8,
            scale: 10,
            numLayers: 8
        )
        #expect(a.lineageID == b.lineageID)
        #expect(a.lineageID.count == 64)
        // Default keys are attention-only, not nil.
        #expect(a.loraConfig.loraParameters.keys == LoRAConfig.defaultAttentionKeys)
    }

    @Test("makeLineage honours explicit keys")
    func makeLineageExplicitKeys() throws {
        let wide = try CLICommon.makeLineage(
            taskID: "t",
            modelID: "m",
            rank: 8,
            scale: 10,
            numLayers: 16,
            keys: LoRAConfig.allProjectionKeys
        )
        #expect(wide.loraConfig.loraParameters.keys == LoRAConfig.allProjectionKeys)

        let inherit = try CLICommon.makeLineage(
            taskID: "t",
            modelID: "m",
            rank: 8,
            scale: 10,
            numLayers: 16,
            keys: nil
        )
        #expect(inherit.loraConfig.loraParameters.keys == nil)
        #expect(wide.lineageID != inherit.lineageID)
    }

    @Test("parseKeys presets and explicit lists")
    func parseKeys() throws {
        #expect(try CLICommon.parseKeys(["attention"]) == LoRAConfig.defaultAttentionKeys)
        #expect(try CLICommon.parseKeys(["all"]) == LoRAConfig.allProjectionKeys)
        #expect(try CLICommon.parseKeys(["wide"]) == LoRAConfig.allProjectionKeys)
        #expect(try CLICommon.parseKeys(["model"]) == nil)
        #expect(
            try CLICommon.parseKeys(["self_attn.q_proj", "self_attn.v_proj"])
                == ["self_attn.q_proj", "self_attn.v_proj"]
        )
        #expect(
            try CLICommon.parseKeys(["self_attn.q_proj,self_attn.v_proj"])
                == ["self_attn.q_proj", "self_attn.v_proj"]
        )
        #expect(
            try CLICommon.parseKeys(["mlp.gate_proj,mlp.up_proj,mlp.down_proj"]) == [
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ]
        )
    }

    @Test("parseKeys rejects empty and mixed presets")
    func parseKeysRejects() {
        do {
            _ = try CLICommon.parseKeys([])
            Issue.record("expected throw")
        } catch let error as AdaptCLIError {
            #expect(error.localizedDescription.contains("keys"))
        } catch {
            Issue.record("unexpected \(error)")
        }
        do {
            _ = try CLICommon.parseKeys(["attention", "q_proj"])
            Issue.record("expected throw")
        } catch let error as AdaptCLIError {
            #expect(error.localizedDescription.contains("preset"))
        } catch {
            Issue.record("unexpected \(error)")
        }
    }

    @Test("resolvePath expands tilde")
    func tildeExpand() {
        let url = CLICommon.resolvePath("~/Documents")
        #expect(!url.path.contains("~"))
        #expect(url.path.hasSuffix("Documents") || url.path.contains("Documents"))
    }
}
