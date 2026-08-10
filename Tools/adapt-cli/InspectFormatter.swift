import AdaptCore
import Foundation

/// Formats registry listings for the `inspect` subcommand (pure, testable).
public enum InspectFormatter {
    /// One lineage summary for display.
    public struct LineageSummary: Sendable, Equatable {
        public var lineageID: String
        public var taskID: String?
        public var baseModelID: String?
        public var rank: Int?
        /// Configured LoRA target modules; `nil` means inherit model defaults.
        public var keys: [String]?
        /// Whether `keys` was omitted (`null`) in stored config.
        public var keysAreModelDefaults: Bool
        public var activeVersion: Int?
        public var versions: [VersionSummary]

        public init(
            lineageID: String,
            taskID: String? = nil,
            baseModelID: String? = nil,
            rank: Int? = nil,
            keys: [String]? = nil,
            keysAreModelDefaults: Bool = false,
            activeVersion: Int? = nil,
            versions: [VersionSummary] = []
        ) {
            self.lineageID = lineageID
            self.taskID = taskID
            self.baseModelID = baseModelID
            self.rank = rank
            self.keys = keys
            self.keysAreModelDefaults = keysAreModelDefaults
            self.activeVersion = activeVersion
            self.versions = versions
        }

        /// Operator-facing keys line (what this adapter adapts).
        public var keysDescription: String {
            if keysAreModelDefaults {
                return "(model defaults)"
            }
            guard let keys else {
                return "(model defaults)"
            }
            if keys.isEmpty {
                return "(empty)"
            }
            return keys.joined(separator: ", ")
        }
    }

    /// One version row for display.
    public struct VersionSummary: Sendable, Equatable {
        public var version: Int
        public var status: String
        public var digestPrefix: String
        public var exampleCount: Int
        public var trainedStart: Date?
        public var trainedEnd: Date?
        public var createdAt: Date?
        public var parentVersion: Int?
        /// Primary eval score when present (e.g. held-out CE nats).
        public var primaryScore: Double?
        /// Metric id for `primaryScore` (e.g. `mean_cross_entropy_nats`).
        public var primaryMetric: String?
        /// `lowerIsBetter` / `higherIsBetter` when known.
        public var primaryDirection: String?

        public init(
            version: Int,
            status: String,
            digestPrefix: String,
            exampleCount: Int,
            trainedStart: Date? = nil,
            trainedEnd: Date? = nil,
            createdAt: Date? = nil,
            parentVersion: Int? = nil,
            primaryScore: Double? = nil,
            primaryMetric: String? = nil,
            primaryDirection: String? = nil
        ) {
            self.version = version
            self.status = status
            self.digestPrefix = digestPrefix
            self.exampleCount = exampleCount
            self.trainedStart = trainedStart
            self.trainedEnd = trainedEnd
            self.createdAt = createdAt
            self.parentVersion = parentVersion
            self.primaryScore = primaryScore
            self.primaryMetric = primaryMetric
            self.primaryDirection = primaryDirection
        }
    }

    /// Renders a multi-line human-readable registry report.
    public static func format(
        rootPath: String,
        lineages: [LineageSummary],
        iso8601: ISO8601DateFormatter = ISO8601DateFormatter()
    ) -> String {
        var lines: [String] = []
        lines.append("Registry root: \(rootPath)")
        if lineages.isEmpty {
            lines.append("(no lineages)")
            return lines.joined(separator: "\n")
        }

        for lineage in lineages {
            lines.append("")
            lines.append("Lineage \(lineage.lineageID)")
            if let task = lineage.taskID {
                lines.append("  task:       \(task)")
            }
            if let model = lineage.baseModelID {
                lines.append("  base model: \(model)")
            }
            if let rank = lineage.rank {
                lines.append("  rank:       \(rank)")
            }
            // Always show keys so operators can answer "what does this adapter adapt?"
            lines.append("  keys:       \(lineage.keysDescription)")
            if let active = lineage.activeVersion {
                lines.append("  active:     v\(active)")
            } else {
                lines.append("  active:     (none — base model)")
            }

            if lineage.versions.isEmpty {
                lines.append("  versions:   (none)")
                continue
            }

            lines.append("  versions:")
            for v in lineage.versions {
                let parent =
                    v.parentVersion.map { " parent=v\($0)" } ?? ""
                let window: String
                if let start = v.trainedStart, let end = v.trainedEnd {
                    window =
                        " window=\(iso8601.string(from: start))…\(iso8601.string(from: end)) examples=\(v.exampleCount)"
                } else {
                    window = " examples=\(v.exampleCount)"
                }
                let created =
                    v.createdAt.map { " created=\(iso8601.string(from: $0))" } ?? ""
                let eval: String
                if let score = v.primaryScore {
                    let metric = v.primaryMetric ?? "primaryScore"
                    let direction = v.primaryDirection.map { " (\($0))" } ?? ""
                    eval = String(format: " eval=%@=%.6f%@", metric, score, direction)
                } else {
                    eval = ""
                }
                lines.append(
                    "    v\(v.version)  [\(v.status)]  digest=\(v.digestPrefix)…\(parent)\(window)\(created)\(eval)"
                )
            }
        }
        return lines.joined(separator: "\n")
    }

    /// Builds a summary from stored adapter versions.
    public static func summary(
        lineageID: String,
        versions: [AdapterVersion],
        activeVersion: Int?
    ) -> LineageSummary {
        let first = versions.first
        let configuredKeys = first?.lineage.loraConfig.loraParameters.keys
        return LineageSummary(
            lineageID: lineageID,
            taskID: first?.lineage.taskID,
            baseModelID: first?.lineage.baseModelID,
            rank: first?.lineage.loraConfig.loraParameters.rank,
            keys: configuredKeys,
            keysAreModelDefaults: configuredKeys == nil,
            activeVersion: activeVersion,
            versions: versions.map { meta in
                VersionSummary(
                    version: meta.version,
                    status: meta.status.rawValue,
                    digestPrefix: String(meta.weightsDigest.prefix(12)),
                    exampleCount: meta.trainedOn.exampleCount,
                    trainedStart: meta.trainedOn.start,
                    trainedEnd: meta.trainedOn.end,
                    createdAt: meta.createdAt,
                    parentVersion: meta.parentVersion,
                    primaryScore: meta.evalReport?.primaryScore,
                    primaryMetric: meta.evalReport?.primaryMetric,
                    primaryDirection: meta.evalReport?.primaryDirection?.rawValue
                )
            }
        )
    }
}
