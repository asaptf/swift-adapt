import Foundation

/// Package-internal fault used only by crash-safety tests.
///
/// Not part of the public `AdaptRegistryError` surface — adopters must never see or
/// switch on this type. Tests in this package catch it directly.
package enum RegistryTestFault: Error, Equatable, Sendable {
    /// Injected mid-operation crash for resilience tests.
    case injected(String)
}
