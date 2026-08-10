import AdaptCLI
import Foundation
import Testing

@Suite("InspectFormatter")
struct InspectFormatterTests {

    @Test("empty registry message")
    func emptyRegistry() {
        let text = InspectFormatter.format(rootPath: "/tmp/adapt", lineages: [])
        #expect(text.contains("Registry root: /tmp/adapt"))
        #expect(text.contains("(no lineages)"))
    }

    @Test("formats active version and digests")
    func formatsLineage() {
        let summary = InspectFormatter.LineageSummary(
            lineageID: "abc123",
            taskID: "style-mirror",
            baseModelID: "mlx-community/Qwen3-0.6B-4bit",
            rank: 8,
            activeVersion: 2,
            versions: [
                .init(
                    version: 1,
                    status: "candidate",
                    digestPrefix: "deadbeefcafe",
                    exampleCount: 50
                ),
                .init(
                    version: 2,
                    status: "active",
                    digestPrefix: "cafebabef00d",
                    exampleCount: 50,
                    parentVersion: 1
                ),
            ]
        )
        let text = InspectFormatter.format(rootPath: "/tmp/r", lineages: [summary])
        #expect(text.contains("Lineage abc123"))
        #expect(text.contains("task:       style-mirror"))
        #expect(text.contains("active:     v2"))
        #expect(text.contains("digest=deadbeefcafe"))
        #expect(text.contains("parent=v1"))
        #expect(text.contains("[active]"))
    }

    @Test("shows none when no active")
    func noActive() {
        let summary = InspectFormatter.LineageSummary(
            lineageID: "xyz",
            activeVersion: nil,
            versions: []
        )
        let text = InspectFormatter.format(rootPath: "/tmp/r", lineages: [summary])
        #expect(text.contains("active:     (none — base model)"))
        #expect(text.contains("versions:   (none)"))
    }
}
