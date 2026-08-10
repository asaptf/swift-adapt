import AdaptCore
import AdaptRegistry
import Foundation

enum InferenceTestSupport {
    static func makeRegistry() throws -> (AdapterRegistry, URL) {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("AdaptInferenceTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        return (try AdapterRegistry(rootURL: temp), temp)
    }

    static func teardown(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    static var lineage: AdapterLineage {
        AdapterLineage(
            taskID: "inference-test",
            baseModelID: "synthetic/tiny",
            loraConfig: LoRAConfig(rank: 8, scale: 10.0, numLayers: 2)
        )
    }

    /// Stores a candidate with opaque weight bytes and optional promote.
    @discardableResult
    static func store(
        registry: AdapterRegistry,
        lineage: AdapterLineage,
        weights: Data,
        promote: Bool = false
    ) async throws -> AdapterVersion {
        let version = try await registry.storeCandidate(
            lineage: lineage,
            weights: weights,
            trainedOn: TrainingWindow(start: Date(), end: Date(), exampleCount: 1)
        )
        // Write a minimal adapter_config is already done by storeCandidate.
        // For fake backends we only need the directory path to exist.
        if promote {
            try await registry.promote(lineage: lineage, version: version.version)
        }
        return version
    }

    static func corruptWeights(
        registry: AdapterRegistry,
        lineage: AdapterLineage,
        version: Int
    ) async throws {
        let root = await registry.rootURL
        let url = root
            .appendingPathComponent(lineage.lineageID, isDirectory: true)
            .appendingPathComponent("v\(version)", isDirectory: true)
            .appendingPathComponent("adapters.safetensors")
        try Data("corrupted-weights".utf8).write(to: url, options: .atomic)
    }
}
