import Foundation

/// On-disk pointer file (`state.json`) for which version is active in a lineage.
struct LineageState: Codable, Sendable, Equatable {
    /// Active version number, or `nil` for base-model behavior.
    var activeVersion: Int?

    init(activeVersion: Int? = nil) {
        self.activeVersion = activeVersion
    }
}
