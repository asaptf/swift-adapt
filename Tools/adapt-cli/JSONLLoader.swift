import AdaptCore
import CryptoKit
import Foundation

/// Parses training examples from JSON Lines (one JSON object per line).
///
/// ## Wire format
///
/// Required keys:
/// - `prompt` (string)
/// - `completion` (string)
///
/// Optional keys:
/// - `source` — `explicitEdit` | `acceptance` | `rejection` | `synthetic`
///   (default `synthetic`)
/// - `weight` — importance double; when omitted, uses `source.defaultWeight`
/// - `id` — UUID string; when omitted, a **content-stable** UUID is derived
///   from `prompt` + `completion` so held-out pins survive reloads
/// - `capturedAt` — ISO-8601 date; defaults to a content-stable epoch offset
///
/// Blank lines and lines whose first non-whitespace character is `#` are skipped.
/// A malformed object fails with the 1-based line number (never a stack trace).
public enum JSONLLoader {
    /// Decodable row used only for parsing; maps onto ``TrainingExample``.
    struct Row: Decodable {
        var prompt: String
        var completion: String
        var source: SignalSource?
        var weight: Double?
        var id: UUID?
        var capturedAt: Date?
    }

    /// Loads examples from a UTF-8 JSONL file.
    ///
    /// - Throws: ``AdaptCLIError/malformedJSONL(line:detail:)`` or
    ///   ``AdaptCLIError/fileNotFound(_:)``.
    public static func load(from url: URL) throws -> [TrainingExample] {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw AdaptCLIError.fileNotFound(url.path)
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw AdaptCLIError.malformedJSONL(line: 1, detail: "file is not valid UTF-8")
        }
        return try parse(text: text)
    }

    /// Parses JSONL text into examples (test seam).
    public static func parse(text: String) throws -> [TrainingExample] {
        var examples: [TrainingExample] = []
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        for (index, rawLine) in lines.enumerated() {
            let lineNumber = index + 1
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }
            if line.hasPrefix("#") { continue }

            let lineData = Data(line.utf8)
            let row: Row
            do {
                row = try decoder.decode(Row.self, from: lineData)
            } catch {
                throw AdaptCLIError.malformedJSONL(
                    line: lineNumber,
                    detail: humanizeDecodeError(error)
                )
            }

            guard !row.prompt.isEmpty else {
                throw AdaptCLIError.malformedJSONL(
                    line: lineNumber,
                    detail: "prompt must be a non-empty string"
                )
            }
            // Empty completion is allowed (rare) but reject missing via Decodable.

            let source = row.source ?? .synthetic
            // Content-stable id/capturedAt when omitted so AdaptEval pins remain
            // valid across process restarts (random UUIDs would break the yardstick).
            let id = row.id ?? stableID(prompt: row.prompt, completion: row.completion)
            let capturedAt =
                row.capturedAt
                ?? stableCapturedAt(prompt: row.prompt, completion: row.completion)
            examples.append(
                TrainingExample(
                    id: id,
                    prompt: row.prompt,
                    completion: row.completion,
                    weight: row.weight,
                    capturedAt: capturedAt,
                    source: source
                )
            )
        }

        return examples
    }

    /// Deterministic UUID from prompt/completion (first 16 bytes of SHA-256).
    public static func stableID(prompt: String, completion: String) -> UUID {
        var payload = Data()
        payload.append(contentsOf: prompt.utf8)
        payload.append(0)
        payload.append(contentsOf: completion.utf8)
        let digest = SHA256.hash(data: payload)
        let bytes = Array(digest.prefix(16))
        return UUID(
            uuid: (
                bytes[0], bytes[1], bytes[2], bytes[3],
                bytes[4], bytes[5], bytes[6], bytes[7],
                bytes[8], bytes[9], bytes[10], bytes[11],
                bytes[12], bytes[13], bytes[14], bytes[15]
            )
        )
    }

    /// Deterministic capture time so recency stratification is stable across loads.
    public static func stableCapturedAt(prompt: String, completion: String) -> Date {
        var payload = Data()
        payload.append(contentsOf: prompt.utf8)
        payload.append(0)
        payload.append(contentsOf: completion.utf8)
        let digest = SHA256.hash(data: payload)
        // Mix bytes into a second-resolution offset from a fixed epoch.
        var value: UInt64 = 0
        for (i, b) in digest.prefix(8).enumerated() {
            value |= UInt64(b) << (8 * i)
        }
        let offset = TimeInterval(value % 86_400_000) // ~1000 days of range
        return Date(timeIntervalSince1970: 1_700_000_000 + offset)
    }

    /// Short, operator-friendly decode error (no stack traces).
    public static func humanizeDecodeError(_ error: Error) -> String {
        if let decoding = error as? DecodingError {
            switch decoding {
            case .keyNotFound(let key, _):
                return "missing key \"\(key.stringValue)\""
            case .typeMismatch(_, let context):
                let path = context.codingPath.map(\.stringValue).joined(separator: ".")
                return path.isEmpty ? "type mismatch" : "type mismatch at \"\(path)\""
            case .valueNotFound(_, let context):
                let path = context.codingPath.map(\.stringValue).joined(separator: ".")
                return path.isEmpty ? "missing value" : "missing value at \"\(path)\""
            case .dataCorrupted(let context):
                if let underlying = context.underlyingError {
                    return "invalid JSON (\(underlying.localizedDescription))"
                }
                return "invalid JSON"
            @unknown default:
                return decoding.localizedDescription
            }
        }
        return error.localizedDescription
    }
}
