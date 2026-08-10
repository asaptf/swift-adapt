import AdaptCore
import Foundation

/// Minimal JSONL loader for held-out measurement inside the demo target.
///
/// Same wire format as `adapt-cli` (`prompt` / `completion` required). Lives here
/// so StyleMirrorEngine does not depend on the CLI module.
enum DemoJSONL {
    struct Row: Decodable {
        var prompt: String
        var completion: String
        var source: SignalSource?
        var weight: Double?
        var id: UUID?
        var capturedAt: Date?
    }

    static func load(from url: URL) throws -> [TrainingExample] {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw StyleMirrorError.invalidArgument(
                "could not read JSONL at \(url.path): \(error.localizedDescription)"
            )
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw StyleMirrorError.invalidArgument("JSONL is not valid UTF-8: \(url.path)")
        }
        return try parse(text: text)
    }

    static func parse(text: String) throws -> [TrainingExample] {
        var examples: [TrainingExample] = []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)

        for (index, rawLine) in lines.enumerated() {
            let lineNumber = index + 1
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix("#") { continue }

            let row: Row
            do {
                row = try decoder.decode(Row.self, from: Data(line.utf8))
            } catch {
                throw StyleMirrorError.invalidArgument(
                    "malformed JSONL line \(lineNumber): \(error.localizedDescription)"
                )
            }
            guard !row.prompt.isEmpty else {
                throw StyleMirrorError.invalidArgument(
                    "malformed JSONL line \(lineNumber): empty prompt"
                )
            }
            let source = row.source ?? .synthetic
            examples.append(
                TrainingExample(
                    id: row.id ?? UUID(),
                    prompt: row.prompt,
                    completion: row.completion,
                    weight: row.weight ?? source.defaultWeight,
                    capturedAt: row.capturedAt ?? Date(),
                    source: source
                )
            )
        }
        return examples
    }
}
