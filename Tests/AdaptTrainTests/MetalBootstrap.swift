import Foundation
import Testing

/// Ensures MLX can find `default.metallib` when running under `swift test`.
///
/// Upstream Cmlx looks for a bundle named `mlx-swift_Cmlx` (compile-time
/// `SWIFTPM_BUNDLE`) or a metallib colocated with the process binary. SPM
/// packages our fixture as `Adapt_AdaptTrainTests.bundle/mlx-swift_Cmlx.bundle/`,
/// which is not always visible to `NSBundle.allBundles` until accessed. This
/// bootstrap copies the metallib next to the test executable so load succeeds.
enum MetalBootstrap {
    private static let lock = NSLock()
    // Guarded by `lock`; nonisolated(unsafe) is the standard pattern for
    // one-time process setup under a mutex.
    nonisolated(unsafe) private static var didRun = false

    static func ensureMetallib() {
        lock.lock()
        defer { lock.unlock() }
        guard !didRun else { return }
        didRun = true

        // Touch Bundle.module so the resource bundle is loaded into the process.
        let moduleBundle = Bundle.module
        let source: URL?
        if let nested = moduleBundle.url(forResource: "mlx-swift_Cmlx", withExtension: "bundle") {
            let candidate = nested.appendingPathComponent("default.metallib")
            source = FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
        } else {
            source =
                moduleBundle.url(
                    forResource: "default",
                    withExtension: "metallib",
                    subdirectory: "mlx-swift_Cmlx.bundle"
                )
                ?? moduleBundle.url(forResource: "default", withExtension: "metallib")
        }
        guard let source else { return }
        copyNextToExecutable(from: source)
    }

    private static func copyNextToExecutable(from source: URL) {
        // Executable path: .../AdaptPackageTests.xctest/Contents/MacOS/AdaptPackageTests
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        let dir = exe.deletingLastPathComponent()
        let destinations = [
            dir.appendingPathComponent("mlx.metallib"),
            dir.appendingPathComponent("default.metallib"),
            dir.deletingLastPathComponent() // Contents
                .appendingPathComponent("Resources")
                .appendingPathComponent("mlx-swift_Cmlx.bundle")
                .appendingPathComponent("default.metallib"),
        ]
        for dest in destinations {
            do {
                try FileManager.default.createDirectory(
                    at: dest.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                if FileManager.default.fileExists(atPath: dest.path) {
                    try FileManager.default.removeItem(at: dest)
                }
                try FileManager.default.copyItem(at: source, to: dest)
            } catch {
                // Best-effort: other destinations may still work.
                continue
            }
        }
    }
}
