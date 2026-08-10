import AdaptCore
import Foundation

/// Deterministic partition of a style-mirror fixture into overnight slices + held-out.
///
/// Seven nights of continual training each need **new** examples; a single held-out
/// slice is never trained on and is reused to measure every version. This type is
/// pure data plumbing — no training, no gate logic.
///
/// ## Contamination rule
///
/// Training masks loss to the assistant (completion) span. If the same completion
/// text appears in both a night slice and the held-out set, held-out CE measures
/// memorisation rather than generalisation. ``partition`` therefore:
/// 1. Requires every example to carry a **distinct** completion string.
/// 2. Assigns each unique completion wholly to either a night or held-out
///    (layout: nights first, held-out tail, in input order).
/// 3. Fails loudly with counts when the corpus cannot satisfy the requested
///    night count and held-out size without completion overlap.
public enum DemoCorpus {
    /// Default night count for the StyleMirror “seven nights” demo.
    public static let defaultNightCount = 7
    /// Default held-out size (enough for a stable mean CE; M3’s floor is separate).
    public static let defaultHeldOutCount = 30
    /// Examples per night when the combined fixture is sized for equal slices.
    public static let defaultExamplesPerNight = 30

    /// Result of partitioning a combined corpus.
    public struct Partition: Sendable, Equatable {
        /// Per-night training slices in order (night 1 … night N).
        public var nights: [[TrainingExample]]
        /// Held-out examples never used for training.
        public var heldOut: [TrainingExample]

        public init(nights: [[TrainingExample]], heldOut: [TrainingExample]) {
            self.nights = nights
            self.heldOut = heldOut
        }

        /// Total examples accounted for.
        public var totalCount: Int {
            nights.reduce(0) { $0 + $1.count } + heldOut.count
        }

        /// Completions used for training (union of all nights).
        public var trainCompletions: Set<String> {
            Set(nights.flatMap { $0.map(\.completion) })
        }

        /// Completions reserved for held-out measurement.
        public var heldOutCompletions: Set<String> {
            Set(heldOut.map(\.completion))
        }
    }

    /// Partitions `examples` into `nightCount` equal night slices and a held-out tail.
    ///
    /// Layout (index order preserved within each slice):
    /// ```
    /// [ night1 | night2 | … | nightN | held-out ]
    /// ```
    /// Each night has `(examples.count - heldOutCount) / nightCount` examples.
    /// Any remainder after equal night division is absorbed into the held-out tail
    /// so nights stay equal-sized (stable resume batch cursors across nights).
    ///
    /// **Structural non-contamination:** every completion string must be unique in
    /// the corpus. Partitioning unique completions (one example each) guarantees
    /// held-out completions never appear in any night. Duplicate completions fail
    /// with counts rather than emitting a contaminated split.
    ///
    /// - Throws: ``AdaptCLIError/invalidArgument`` when sizes cannot form the
    ///   partition or when completion texts are not unique.
    public static func partition(
        _ examples: [TrainingExample],
        nightCount: Int = defaultNightCount,
        heldOutCount: Int = defaultHeldOutCount
    ) throws -> Partition {
        guard nightCount > 0 else {
            throw AdaptCLIError.invalidArgument("nightCount must be > 0")
        }
        guard heldOutCount > 0 else {
            throw AdaptCLIError.invalidArgument("heldOutCount must be > 0")
        }

        let total = examples.count
        let uniqueCompletions = Set(examples.map(\.completion))
        let uniqueCount = uniqueCompletions.count

        // Fail loudly on duplicate completions — re-pairing the same reply under
        // new prompts pads volume without new training signal and leaks across splits.
        if uniqueCount != total {
            throw AdaptCLIError.invalidArgument(
                """
                demo corpus completions must be unique: \
                lines=\(total) unique_completions=\(uniqueCount) \
                duplicates=\(total - uniqueCount). \
                Expand the fixture so every line has a distinct completion text.
                """
            )
        }

        let minimum = heldOutCount + nightCount
        guard total >= minimum else {
            throw AdaptCLIError.invalidArgument(
                """
                need at least heldOutCount (\(heldOutCount)) + nightCount (\(nightCount)) \
                unique completions; got lines=\(total) unique_completions=\(uniqueCount)
                """
            )
        }

        let trainBudget = total - heldOutCount
        let perNight = trainBudget / nightCount
        guard perNight > 0 else {
            throw AdaptCLIError.invalidArgument(
                """
                heldOutCount \(heldOutCount) leaves no examples for \(nightCount) nights \
                (lines=\(total) unique_completions=\(uniqueCount))
                """
            )
        }

        let trainUsed = perNight * nightCount
        // Remainder of the train budget joins held-out so nights stay equal length.
        let heldOutStart = trainUsed
        var nights: [[TrainingExample]] = []
        nights.reserveCapacity(nightCount)
        for night in 0..<nightCount {
            let start = night * perNight
            let end = start + perNight
            nights.append(Array(examples[start..<end]))
        }
        let heldOut = Array(examples[heldOutStart...])

        let partition = Partition(nights: nights, heldOut: heldOut)

        // Structural guarantee (redundant when completions are unique, but the
        // numbers in the error message are what operators need when it fails).
        let overlap = partition.trainCompletions.intersection(partition.heldOutCompletions)
        if !overlap.isEmpty {
            throw AdaptCLIError.invalidArgument(
                """
                held-out contaminated by training completions: \
                overlap=\(overlap.count) held_out=\(heldOut.count) \
                train=\(trainUsed) unique_completions=\(uniqueCount). \
                Partition on unique completions failed; fix the corpus.
                """
            )
        }

        return partition
    }

    /// Writes night + held-out JSONL files under `directory`.
    ///
    /// Files: `night-1.jsonl` … `night-N.jsonl`, `held-out.jsonl`.
    public static func writePartition(
        _ partition: Partition,
        to directory: URL
    ) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        for (index, night) in partition.nights.enumerated() {
            let url = directory.appendingPathComponent("night-\(index + 1).jsonl")
            try writeJSONL(night, to: url)
        }
        try writeJSONL(partition.heldOut, to: directory.appendingPathComponent("held-out.jsonl"))
    }

    /// Encodes examples as JSONL (prompt/completion/source only — stable for fixtures).
    public static func writeJSONL(_ examples: [TrainingExample], to url: URL) throws {
        var lines: [String] = []
        lines.reserveCapacity(examples.count)
        for ex in examples {
            // Compact, deterministic keys. Omit ephemeral id/capturedAt so re-runs match.
            let source = ex.source.rawValue
            let prompt = escapeJSONString(ex.prompt)
            let completion = escapeJSONString(ex.completion)
            lines.append(
                #"{"prompt":\#(prompt),"completion":\#(completion),"source":"\#(source)"}"#
            )
        }
        let text = lines.joined(separator: "\n") + (lines.isEmpty ? "" : "\n")
        do {
            try text.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            throw AdaptCLIError.invalidArgument(
                "failed to write JSONL at \(url.path): \(error.localizedDescription)"
            )
        }
    }

    private static func escapeJSONString(_ string: String) -> String {
        // Use JSONSerialization for correct escaping.
        let data = try! JSONSerialization.data(withJSONObject: string, options: .fragmentsAllowed)
        return String(data: data, encoding: .utf8)!
    }
}
