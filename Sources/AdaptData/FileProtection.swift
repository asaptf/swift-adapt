import Foundation

/// Applies Data Protection attributes where the platform supports them.
///
/// ## Platform guarantees
///
/// - **iOS:** files are marked
///   `FileProtectionType.completeUntilFirstUserAuthentication`. After first
///   unlock following boot, the system encrypts the file at rest with a key
///   tied to the device passcode; the file is inaccessible from a cold boot
///   until the user unlocks once.
/// - **macOS:** Apple does **not** offer the same per-file Data Protection
///   classes. This helper is a no-op. Confidentiality at rest on macOS
///   depends on FileVault (full-disk encryption) when the user has enabled
///   it — Adapt cannot claim parity with iOS file-class encryption.
enum FileProtection {
    /// Sets `.completeUntilFirstUserAuthentication` on iOS; no-op on macOS.
    static func apply(_ url: URL) throws {
        #if os(iOS)
        try FileManager.default.setAttributes(
            [.protectionKey: FileProtectionType.completeUntilFirstUserAuthentication],
            ofItemAtPath: url.path
        )
        #endif
    }
}
