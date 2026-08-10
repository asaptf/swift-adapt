import AdaptCore
import AdaptData
import Foundation

enum AdaptDataTestSupport {
    static func makeLineage(taskID: String = "email-style") -> AdapterLineage {
        AdapterLineage(
            taskID: taskID,
            baseModelID: "mlx-community/Qwen3-0.6B-4bit",
            loraConfig: LoRAConfig()
        )
    }

    static func makeTempRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("AdaptDataTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static func teardown(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    static func makeBuffer(
        configuration: ReplayBufferConfiguration = ReplayBufferConfiguration(
            maxExamples: 1_000,
            retention: .seconds(30 * 24 * 60 * 60),
            maxCapturesPerDay: 500
        ),
        taskID: String = "email-style"
    ) throws -> (ReplayBuffer, URL) {
        let root = try makeTempRoot()
        let lineage = makeLineage(taskID: taskID)
        let buffer = try ReplayBuffer(
            lineage: lineage,
            rootURL: root,
            configuration: configuration
        )
        return (buffer, root)
    }

    static func example(
        id: UUID = UUID(),
        prompt: String,
        completion: String,
        source: SignalSource = .explicitEdit,
        capturedAt: Date = Date(),
        weight: Double? = nil
    ) -> TrainingExample {
        TrainingExample(
            id: id,
            prompt: prompt,
            completion: completion,
            weight: weight,
            capturedAt: capturedAt,
            source: source
        )
    }

    /// True if `needle` UTF-8 bytes appear anywhere in `fileURL` contents.
    static func fileContainsASCII(_ needle: String, fileURL: URL) throws -> Bool {
        let data = try Data(contentsOf: fileURL)
        guard let needleData = needle.data(using: .utf8), !needleData.isEmpty else {
            return false
        }
        return data.range(of: needleData) != nil
    }
}
