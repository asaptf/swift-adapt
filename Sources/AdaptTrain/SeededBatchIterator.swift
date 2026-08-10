import AdaptCore
import Foundation

/// Deterministic batch index iterator seeded from `AdaptCore.SeededGenerator`.
///
/// Upstream `LoRABatchIterator` is `internal` and calls `indices.shuffle()` on
/// the unseeded global RNG — neither reachable nor reproducible. Batch order
/// here is a pure function of `(seed / generator state, dataset count, batchSize)`.
public struct SeededBatchIterator: Sendable {
    /// Shuffled indices into the dataset for the current epoch.
    public private(set) var indices: [Int]
    /// Next offset into `indices`.
    public private(set) var index: Int
    /// Generator used for reshuffles at epoch boundaries.
    public private(set) var generator: SeededGenerator
    /// Examples per batch.
    public let batchSize: Int
    /// Dataset size.
    public let datasetCount: Int
    /// When true, reshuffle and continue forever; when false, stop at epoch end.
    public let infinite: Bool

    /// Creates an iterator and performs the initial shuffle.
    public init(
        datasetCount: Int,
        batchSize: Int,
        generator: SeededGenerator,
        infinite: Bool = true,
        index: Int = 0,
        indices: [Int]? = nil
    ) {
        precondition(batchSize > 0)
        precondition(datasetCount >= 0)
        self.datasetCount = datasetCount
        self.batchSize = batchSize
        self.generator = generator
        self.infinite = infinite
        self.index = index
        if let indices, indices.count == datasetCount {
            self.indices = indices
        } else {
            self.indices = Array(0..<datasetCount)
            if datasetCount > 1 {
                self.indices.shuffle(using: &self.generator)
            }
        }
    }

    /// Checkpointable cursor state.
    public var cursor: BatchCursor {
        BatchCursor(indices: indices, index: index, generatorState: generator.state)
    }

    /// Restores from a saved cursor.
    public mutating func restore(cursor: BatchCursor) {
        self.indices = cursor.indices
        self.index = cursor.index
        self.generator = SeededGenerator(state: cursor.generatorState)
    }

    /// Returns the next batch of dataset indices, or `nil` when finite and exhausted.
    public mutating func nextBatchIndices() -> [Int]? {
        if datasetCount == 0 { return nil }

        if index >= indices.count {
            if !infinite { return nil }
            indices = Array(0..<datasetCount)
            if datasetCount > 1 {
                indices.shuffle(using: &generator)
            }
            index = 0
        }

        let end = min(index + batchSize, indices.count)
        let batch = Array(indices[index..<end])
        index = end
        return batch
    }
}

/// Serializable batch-iterator position for checkpoints.
public struct BatchCursor: Codable, Sendable, Hashable {
    /// Permuted dataset indices for the current epoch.
    public var indices: [Int]
    /// Offset into `indices`.
    public var index: Int
    /// `SeededGenerator.state` for the next reshuffle.
    public var generatorState: UInt64

    /// Creates a batch cursor.
    public init(indices: [Int], index: Int, generatorState: UInt64) {
        self.indices = indices
        self.index = index
        self.generatorState = generatorState
    }
}
