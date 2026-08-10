import Foundation
import MLX
import MLXNN

/// Exclusive transfer of a non-`Sendable` `Module` into ``Trainer``.
///
/// `MLXArray` / `Module` must not cross concurrency domains. This box is
/// `@unchecked Sendable` solely so ownership can move into the trainer actor;
/// the trainer serializes all access. Callers must not touch the model while
/// a `run` is in progress.
public struct SendingModule: @unchecked Sendable {
    /// The module being transferred.
    public let model: Module

    /// Creates a transfer wrapper.
    public init(_ model: Module) {
        self.model = model
    }
}

/// `@unchecked Sendable` loss function for actor transfer.
///
/// The body captures only what the trainer invokes under its isolation.
public struct SendingLoss: @unchecked Sendable {
    /// Loss implementation.
    public let body: MicrobatchLoss

    /// Creates a loss transfer wrapper.
    public init(_ body: @escaping MicrobatchLoss) {
        self.body = body
    }
}

/// `@unchecked Sendable` micro-batch builder for actor transfer.
public struct SendingMicrobatch: @unchecked Sendable {
    /// Maps dataset indices to loss arrays.
    public let body: ([Int]) throws -> [MLXArray]?

    /// Creates a micro-batch builder wrapper.
    public init(_ body: @escaping ([Int]) throws -> [MLXArray]?) {
        self.body = body
    }
}
