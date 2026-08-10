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
            keys: [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            ],
            keysAreModelDefaults: false,
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
        #expect(
            text.contains(
                "keys:       self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj"
            )
        )
        #expect(text.contains("active:     v2"))
        #expect(text.contains("digest=deadbeefcafe"))
        #expect(text.contains("parent=v1"))
        #expect(text.contains("[active]"))
    }

    @Test("formats held-out eval measurement on versions")
    func formatsEvalMeasurement() {
        let summary = InspectFormatter.LineageSummary(
            lineageID: "abc",
            taskID: "style-mirror",
            activeVersion: 1,
            versions: [
                .init(
                    version: 1,
                    status: "active",
                    digestPrefix: "deadbeefcafe",
                    exampleCount: 30,
                    primaryScore: 2.5,
                    primaryMetric: "mean_cross_entropy_nats",
                    primaryDirection: "lowerIsBetter"
                ),
            ]
        )
        let text = InspectFormatter.format(rootPath: "/tmp/r", lineages: [summary])
        #expect(text.contains("eval=mean_cross_entropy_nats=2.500000"))
        #expect(text.contains("lowerIsBetter"))
    }

    @Test("surfaces model-defaults when keys were nil")
    func surfacesModelDefaults() {
        let summary = InspectFormatter.LineageSummary(
            lineageID: "legacy",
            taskID: "style-mirror",
            rank: 8,
            keys: nil,
            keysAreModelDefaults: true,
            activeVersion: 1,
            versions: [
                .init(version: 1, status: "active", digestPrefix: "abc", exampleCount: 10)
            ]
        )
        let text = InspectFormatter.format(rootPath: "/tmp/r", lineages: [summary])
        #expect(text.contains("keys:       (model defaults)"))
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
        #expect(text.contains("keys:       (model defaults)"))
    }
}
