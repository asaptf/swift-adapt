import Foundation

/// Writes files atomically via a temporary sibling + replace, never partial in-place mutation.
enum AtomicFileWriter {
    /// Writes `data` to `destination` by creating a unique temp file in the same directory,
    /// then replacing the destination. Applies Data Protection after the write.
    static func write(data: Data, to destination: URL) throws {
        let directory = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        let tempName = ".tmp-\(UUID().uuidString)-\(destination.lastPathComponent)"
        let tempURL = directory.appendingPathComponent(tempName)

        do {
            try data.write(to: tempURL, options: .atomic)
            try FileProtection.apply(tempURL)

            if FileManager.default.fileExists(atPath: destination.path) {
                _ = try FileManager.default.replaceItemAt(
                    destination,
                    withItemAt: tempURL,
                    backupItemName: nil,
                    options: .usingNewMetadataOnly
                )
            } else {
                try FileManager.default.moveItem(at: tempURL, to: destination)
            }
            try FileProtection.apply(destination)
        } catch {
            try? FileManager.default.removeItem(at: tempURL)
            throw AdaptRegistryError.ioFailed(
                "Failed to atomically write \(destination.lastPathComponent): \(error.localizedDescription)"
            )
        }
    }

    /// Encodes `value` as pretty-printed JSON and writes it atomically.
    static func writeJSON<T: Encodable>(_ value: T, to destination: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data: Data
        do {
            data = try encoder.encode(value)
        } catch {
            throw AdaptRegistryError.codingFailed(
                "Failed to encode \(destination.lastPathComponent): \(error.localizedDescription)"
            )
        }
        try write(data: data, to: destination)
    }

    /// Decodes JSON from `url` using ISO-8601 dates.
    static func readJSON<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw AdaptRegistryError.ioFailed(
                "Failed to read \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        do {
            return try decoder.decode(type, from: data)
        } catch {
            throw AdaptRegistryError.codingFailed(
                "Failed to decode \(url.lastPathComponent): \(error.localizedDescription)"
            )
        }
    }
}
