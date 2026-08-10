import AdaptCore
import Foundation

/// Reads and writes ``HeldOutPin`` files beside a lineage directory.
///
/// File name: `held_out_pin.json` next to `state.json` — durable across buffer
/// TTL pruning. Pure Foundation I/O; no MLX, no registry dependency.
public enum HeldOutPinStore: Sendable {
    /// On-disk file name inside a lineage directory.
    public static let fileName = "held_out_pin.json"

    /// URL of the pin file inside `lineageDirectory`.
    public static func pinURL(in lineageDirectory: URL) -> URL {
        lineageDirectory.appendingPathComponent(fileName)
    }

    /// Loads a pin if present; returns `nil` when the file does not exist.
    public static func load(from lineageDirectory: URL) throws -> HeldOutPin? {
        let url = pinURL(in: lineageDirectory)
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return nil }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw AdaptEvalError.pinIO(
                "Failed to read \(fileName): \(error.localizedDescription)"
            )
        }
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(HeldOutPin.self, from: data)
        } catch {
            throw AdaptEvalError.pinIO(
                "Failed to decode \(fileName): \(error.localizedDescription)"
            )
        }
    }

    /// Atomically writes `pin` into the lineage directory.
    public static func save(_ pin: HeldOutPin, to lineageDirectory: URL) throws {
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: lineageDirectory, withIntermediateDirectories: true)
        } catch {
            throw AdaptEvalError.pinIO(
                "Failed to create lineage directory: \(error.localizedDescription)"
            )
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data: Data
        do {
            data = try encoder.encode(pin)
        } catch {
            throw AdaptEvalError.pinIO(
                "Failed to encode held-out pin: \(error.localizedDescription)"
            )
        }

        let destination = pinURL(in: lineageDirectory)
        let tempName = ".tmp-\(UUID().uuidString)-\(fileName)"
        let tempURL = lineageDirectory.appendingPathComponent(tempName)
        do {
            try data.write(to: tempURL, options: .atomic)
            if fm.fileExists(atPath: destination.path) {
                _ = try fm.replaceItemAt(
                    destination,
                    withItemAt: tempURL,
                    backupItemName: nil,
                    options: .usingNewMetadataOnly
                )
            } else {
                try fm.moveItem(at: tempURL, to: destination)
            }
        } catch {
            try? fm.removeItem(at: tempURL)
            throw AdaptEvalError.pinIO(
                "Failed to write \(fileName): \(error.localizedDescription)"
            )
        }
    }

    /// Loads an existing pin or creates, persists, and returns a new one.
    ///
    /// The pin is created **once** per lineage directory. Subsequent calls
    /// return the same pin regardless of changes to `pool` composition (IDs
    /// that later disappear are detected at resolve time).
    public static func loadOrCreate(
        lineageDirectory: URL,
        lineageID: String,
        pool: [TrainingExample],
        policy: PromotionPolicy,
        seed: UInt64,
        mode: HeldOutPinMode = .stratifiedFraction
    ) throws -> HeldOutPin {
        if let existing = try load(from: lineageDirectory) {
            return existing
        }
        let pin = try HeldOutSelector.select(
            from: pool,
            policy: policy,
            seed: seed,
            lineageID: lineageID,
            mode: mode
        )
        try save(pin, to: lineageDirectory)
        return pin
    }
}
