import Foundation

/// Applies Data Protection attributes where supported.
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
