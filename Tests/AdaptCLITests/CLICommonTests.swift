import AdaptCLI
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
    }

    @Test("resolvePath expands tilde")
    func tildeExpand() {
        let url = CLICommon.resolvePath("~/Documents")
        #expect(!url.path.contains("~"))
        #expect(url.path.hasSuffix("Documents") || url.path.contains("Documents"))
    }
}
