import AdaptCore
import AdaptRegistry
import ArgumentParser
import Foundation

/// `adapt-cli inspect` — list lineages, versions, active pointer, digests, windows.
public struct InspectCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "inspect",
        abstract: "List adapter lineages, versions, active pointer, digests, and training windows."
    )

    @Option(name: .long, help: "Registry root directory (default: Application Support/Adapt).")
    var registry: String?

    @Option(name: .long, help: "Filter to a single lineage id (full hex digest).")
    var lineage: String?

    public init() {}

    public func run() async throws {
        let registry = try CLICommon.openRegistry(root: self.registry)
        let root = await registry.rootURL
        let ids: [String]
        if let only = lineage {
            ids = [only]
        } else {
            ids = try CLICommon.listLineageIDs(rootURL: root)
        }

        var summaries: [InspectFormatter.LineageSummary] = []
        for id in ids {
            let versions = try await registry.listVersions(lineageID: id)
            let active = try await registry.activeVersion(lineageID: id)?.version
            if versions.isEmpty && lineage == nil {
                // Empty lineage dir with only state.json — still show it.
                summaries.append(
                    InspectFormatter.LineageSummary(
                        lineageID: id,
                        activeVersion: active,
                        versions: []
                    )
                )
            } else {
                summaries.append(
                    InspectFormatter.summary(
                        lineageID: id,
                        versions: versions,
                        activeVersion: active
                    )
                )
            }
        }

        print(InspectFormatter.format(rootPath: root.path, lineages: summaries))
    }
}
