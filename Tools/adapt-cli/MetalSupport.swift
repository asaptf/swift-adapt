import Foundation

/// Ensures MLX can find `default.metallib` when running `adapt-cli`.
///
/// Mirrors the test-host bootstrap: compile the resolved mlx-swift Metal
/// sources (or reuse the revision-keyed cache) and install the metallib next
/// to the process binary. Safe to call repeatedly.
public enum MetalSupport {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var didRun = false
    nonisolated(unsafe) private static var retainedBundle: Bundle?

    /// Locates/builds the metallib and places it where Cmlx can load it.
    public static func ensureMetallib() throws {
        lock.lock()
        defer { lock.unlock() }
        guard !didRun else { return }
        didRun = true

        let packageRoot = try findPackageRoot()
        let metallib = try runEnsureScript(packageRoot: packageRoot)
        try installNextToExecutable(metallib: metallib)
        try stageBundle(metallib: metallib, packageRoot: packageRoot)
    }

    // MARK: - Script

    private static func runEnsureScript(packageRoot: URL) throws -> URL {
        let script = packageRoot
            .appendingPathComponent("scripts")
            .appendingPathComponent("ensure-mlx-metal-library.sh")
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw AdaptCLIError.metal("build script not found at \(script.path)")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/bash")
        process.arguments = [script.path]
        process.currentDirectoryURL = packageRoot
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()

        let outText = String(data: stdout.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let errText = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        guard process.terminationStatus == 0 else {
            let combined = [errText, outText]
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: "\n")
            throw AdaptCLIError.metal(
                "ensure-mlx-metal-library.sh failed (status \(process.terminationStatus)): \(combined)"
            )
        }

        let currentPointer = packageRoot
            .appendingPathComponent(".build/mlx-metallib-cache/CURRENT")
        if let pointer = try? String(contentsOf: currentPointer, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !pointer.isEmpty
        {
            let url = URL(fileURLWithPath: pointer)
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        if let pathLine = outText.split(separator: "\n").reversed()
            .first(where: { $0.hasPrefix("path: ") })
        {
            let path = pathLine.dropFirst("path: ".count)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let url = URL(fileURLWithPath: String(path))
            if FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        throw AdaptCLIError.metal("metallib missing after build under .build/mlx-metallib-cache/")
    }

    // MARK: - Install

    private static func installNextToExecutable(metallib: URL) throws {
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        let dir = exe.deletingLastPathComponent()
        let fm = FileManager.default
        for name in ["mlx.metallib", "default.metallib"] {
            let dest = dir.appendingPathComponent(name)
            if fm.fileExists(atPath: dest.path) {
                try? fm.removeItem(at: dest)
            }
            try fm.copyItem(at: metallib, to: dest)
        }
        // Resources/ sibling used by some Cmlx probes.
        let resources = dir.appendingPathComponent("Resources")
        try? fm.createDirectory(at: resources, withIntermediateDirectories: true)
        for name in ["mlx.metallib", "default.metallib"] {
            let dest = resources.appendingPathComponent(name)
            if fm.fileExists(atPath: dest.path) {
                try? fm.removeItem(at: dest)
            }
            try? fm.copyItem(at: metallib, to: dest)
        }
    }

    private static func stageBundle(metallib: URL, packageRoot: URL) throws {
        let rev =
            (try? String(
                contentsOf: packageRoot
                    .appendingPathComponent(".build/mlx-metallib-cache/CURRENT.revision"),
                encoding: .utf8
            ))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
            ?? "current"
        let bundleDir = packageRoot
            .appendingPathComponent(".build/mlx-metallib-cache")
            .appendingPathComponent(rev)
            .appendingPathComponent("mlx-swift_Cmlx.bundle")
        let dest = bundleDir.appendingPathComponent("default.metallib")
        let fm = FileManager.default
        try fm.createDirectory(at: bundleDir, withIntermediateDirectories: true)
        if fm.fileExists(atPath: dest.path) {
            try? fm.removeItem(at: dest)
        }
        try fm.copyItem(at: metallib, to: dest)
        if let bundle = Bundle(url: bundleDir) {
            retainedBundle = bundle
            _ = bundle.resourceURL
        }
    }

    private static func findPackageRoot() throws -> URL {
        var starts: [URL] = [
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        ]
        for arg in CommandLine.arguments {
            if arg.hasPrefix("/") || arg.hasPrefix(".") {
                starts.append(
                    URL(fileURLWithPath: arg).resolvingSymlinksInPath()
                        .deletingLastPathComponent()
                )
            }
        }
        for start in starts {
            var dir = start
            for _ in 0..<16 {
                let packageSwift = dir.appendingPathComponent("Package.swift")
                let script = dir
                    .appendingPathComponent("scripts")
                    .appendingPathComponent("ensure-mlx-metal-library.sh")
                if FileManager.default.fileExists(atPath: packageSwift.path),
                   FileManager.default.fileExists(atPath: script.path)
                {
                    return dir
                }
                let parent = dir.deletingLastPathComponent()
                if parent.path == dir.path { break }
                dir = parent
            }
        }
        throw AdaptCLIError.metal(
            "could not locate package root (Package.swift + scripts/ensure-mlx-metal-library.sh). Run from the repository root."
        )
    }
}
