import AdaptCore
import AdaptData
import AdaptRegistry
import Foundation

/// On-device directories for the QuickReply demo (buffer + registry).
public struct QuickReplyStore: Sendable {
    public let rootURL: URL
    public let lineage: AdapterLineage
    public let registry: AdapterRegistry
    public let buffer: ReplayBuffer

    public init(lineage: AdapterLineage, rootURL: URL? = nil) throws {
        let fm = FileManager.default
        let root: URL
        if let rootURL {
            root = rootURL
        } else {
            let support = try fm.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            root = support.appendingPathComponent("QuickReply", isDirectory: true)
        }
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        self.rootURL = root
        self.lineage = lineage
        self.registry = try AdapterRegistry(
            rootURL: root.appendingPathComponent("registry", isDirectory: true)
        )
        self.buffer = try ReplayBuffer(
            lineage: lineage,
            rootURL: root.appendingPathComponent("buffer", isDirectory: true),
            configuration: ReplayBufferConfiguration(
                maxExamples: 5_000,
                retention: .seconds(30 * 24 * 60 * 60),
                maxCapturesPerDay: 200
            )
        )
    }

    /// Captures a reply the user actually sent (acceptance signal).
    public func captureReply(context: String, body: String) async throws {
        let example = TrainingExample(
            prompt: context,
            completion: body,
            source: .acceptance
        )
        _ = try await buffer.add(example)
    }

    /// Captures a user edit of a suggested reply (highest-weight signal).
    public func captureEdit(context: String, body: String) async throws {
        let example = TrainingExample(
            prompt: context,
            completion: body,
            source: .explicitEdit
        )
        _ = try await buffer.add(example)
    }
}
