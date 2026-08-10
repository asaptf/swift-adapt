import AdaptCore
import Foundation
import MLX
import MLXNN

/// On-disk train-loop state written next to registry adapter files.
///
/// ## Checkpoint format (per registry version directory `vN/`)
///
/// | File | Writer | Contents |
/// |---|---|---|
/// | `adapter_config.json` | AdaptRegistry | `LoRAConfig` (upstream shape) |
/// | `adapters.safetensors` | AdaptRegistry | trainable LoRA weights |
/// | `version.json` | AdaptRegistry | `AdapterVersion` metadata |
/// | `train_state.json` | AdaptTrain | step, cursor, loss curve, hyperparams |
/// | `optimizer.safetensors` | AdaptTrain | AdamW moments (`m.*`, `v.*`) |
///
/// A version is **resumable** only when both AdaptTrain sidecars exist. Incomplete
/// sidecars are skipped so a crash mid-checkpoint never resumes with half state.
///
/// ## Write order (interruption safety)
///
/// 1. `optimizer.safetensors` is written **atomically** (serialize to memory, then
///    temp file + replace).
/// 2. `train_state.json` is written **last** (also atomically).
///
/// `isComplete` requires both files to exist, so a process killed during the
/// optimizer write never advertises a resumable checkpoint: the state file is
/// absent until every durable part is in place. Resume also treats load failures
/// (truncated safetensors, corrupt JSON) as "skip this version".
public struct TrainStateFile: Codable, Sendable, Hashable {
    /// Format version for forward-compatible decoding.
    public var schemaVersion: Int
    /// Lifetime optimizer steps completed before this checkpoint.
    public var step: Int
    /// AdamW `t` (matches `step` for one update per step).
    public var optimizerStep: Int
    /// Original run seed (informational; cursor carries live RNG state).
    public var seed: UInt64
    /// Batch iterator cursor.
    public var cursor: BatchCursor
    /// Full loss history up to this checkpoint (lifetime).
    public var lossHistory: [Float]
    /// Lifetime tokens that contributed to the loss.
    public var tokensProcessed: Int
    /// Parent adapter version, if any.
    public var parentVersion: Int?
    /// Config snapshot for diagnostics (resume uses the live `TrainConfig`).
    public var config: TrainConfig

    /// Creates a train-state payload.
    public init(
        schemaVersion: Int = 1,
        step: Int,
        optimizerStep: Int,
        seed: UInt64,
        cursor: BatchCursor,
        lossHistory: [Float],
        tokensProcessed: Int,
        parentVersion: Int?,
        config: TrainConfig
    ) {
        self.schemaVersion = schemaVersion
        self.step = step
        self.optimizerStep = optimizerStep
        self.seed = seed
        self.cursor = cursor
        self.lossHistory = lossHistory
        self.tokensProcessed = tokensProcessed
        self.parentVersion = parentVersion
        self.config = config
    }
}

/// Names of AdaptTrain sidecar files inside a registry version directory.
public enum TrainCheckpointFiles {
    /// JSON train state.
    public static let trainState = "train_state.json"
    /// AdamW moments safetensors.
    public static let optimizer = "optimizer.safetensors"
}

/// Load/save helpers for train sidecars (Sendable I/O only — arrays stay local).
public enum TrainCheckpoint {
    /// Returns true when both sidecars exist for a version directory.
    ///
    /// Existence alone is not integrity: resume must still attempt load and skip
    /// on failure. Existence + write-state-last means a crash mid-optimizer-write
    /// cannot leave both files present from a partial new checkpoint.
    public static func isComplete(at versionDirectory: URL) -> Bool {
        let fm = FileManager.default
        let state = versionDirectory.appendingPathComponent(TrainCheckpointFiles.trainState)
        let opt = versionDirectory.appendingPathComponent(TrainCheckpointFiles.optimizer)
        return fm.fileExists(atPath: state.path) && fm.fileExists(atPath: opt.path)
    }

    /// Writes optimizer moments first (atomic), then `train_state.json` last.
    ///
    /// Ordering guarantees a checkpoint is only advertised as complete once every
    /// part is durable. The optimizer file is never left as a truncated in-place
    /// write: moments are serialized with `MLX.saveToData` then written via
    /// temp + replace.
    public static func write(
        state: TrainStateFile,
        moments: [String: MLXArray],
        to versionDirectory: URL
    ) throws {
        try FileManager.default.createDirectory(
            at: versionDirectory,
            withIntermediateDirectories: true
        )

        // 1. Optimizer first (atomic).
        let optURL = versionDirectory.appendingPathComponent(TrainCheckpointFiles.optimizer)
        let optData: Data
        do {
            optData = try MLX.saveToData(arrays: moments)
        } catch {
            throw AdaptTrainError.checkpointFailed(
                "serialize optimizer: \(error.localizedDescription)"
            )
        }
        do {
            try writeAtomically(optData, to: optURL)
        } catch {
            throw AdaptTrainError.checkpointFailed(
                "write optimizer: \(error.localizedDescription)"
            )
        }

        // 2. State last — only after optimizer is durable. Completeness requires both.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        let stateData: Data
        do {
            stateData = try encoder.encode(state)
        } catch {
            throw AdaptTrainError.checkpointFailed(
                "encode train_state: \(error.localizedDescription)"
            )
        }
        let stateURL = versionDirectory.appendingPathComponent(TrainCheckpointFiles.trainState)
        do {
            try writeAtomically(stateData, to: stateURL)
        } catch {
            throw AdaptTrainError.checkpointFailed(
                "write train_state: \(error.localizedDescription)"
            )
        }
    }

    /// Loads train state JSON.
    public static func loadState(from versionDirectory: URL) throws -> TrainStateFile {
        let url = versionDirectory.appendingPathComponent(TrainCheckpointFiles.trainState)
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw AdaptTrainError.checkpointFailed("read train_state: \(error.localizedDescription)")
        }
        do {
            return try JSONDecoder().decode(TrainStateFile.self, from: data)
        } catch {
            throw AdaptTrainError.checkpointFailed("decode train_state: \(error.localizedDescription)")
        }
    }

    /// Loads optimizer moments from safetensors.
    public static func loadMoments(from versionDirectory: URL) throws -> [String: MLXArray] {
        let url = versionDirectory.appendingPathComponent(TrainCheckpointFiles.optimizer)
        do {
            return try MLX.loadArrays(url: url)
        } catch {
            throw AdaptTrainError.checkpointFailed("read optimizer: \(error.localizedDescription)")
        }
    }

    /// Serializes trainable parameters to safetensors `Data` for the registry.
    public static func weightsData(from model: Module) throws -> Data {
        let parameters = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
        do {
            return try MLX.saveToData(arrays: parameters)
        } catch {
            throw AdaptTrainError.checkpointFailed("serialize weights: \(error.localizedDescription)")
        }
    }

    /// Loads adapter weights from a registry weights file into `model`.
    public static func loadWeights(into model: Module, from weightsURL: URL) throws {
        let arrays: [String: MLXArray]
        do {
            arrays = try MLX.loadArrays(url: weightsURL)
        } catch {
            throw AdaptTrainError.checkpointFailed("load weights: \(error.localizedDescription)")
        }
        model.update(parameters: ModuleParameters.unflattened(arrays))
        eval(model)
    }

    /// Writes `data` via a unique temp sibling then replace/move (never partial in-place).
    private static func writeAtomically(_ data: Data, to destination: URL) throws {
        let directory = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let tempName = ".tmp-\(UUID().uuidString)-\(destination.lastPathComponent)"
        let tempURL = directory.appendingPathComponent(tempName)
        do {
            try data.write(to: tempURL, options: .atomic)
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
        } catch {
            try? FileManager.default.removeItem(at: tempURL)
            throw error
        }
    }
}
