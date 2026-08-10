import Foundation

/// Ensures MLX can find `default.metallib` when running under `swift test`.
///
/// ## Why this exists
///
/// Upstream Cmlx looks for a bundle named `mlx-swift_Cmlx` (compile-time
/// `SWIFTPM_BUNDLE`) or a metallib colocated with the process binary. SwiftPM
/// does not produce that metallib for Cmlx, so MLX kernel load fails under
/// `swift test` unless we supply one.
///
/// ## Why it is not a committed fixture
///
/// Packaging a prebuilt `default.metallib` under `Tests/` freezes kernels at
/// whatever mlx-swift revision built them. `Package.swift` pins mlx-swift as
/// `.upToNextMinor(from: "0.31.4")`, so the resolved checkout can drift within
/// 0.31.x while a vendored blob stays frozen — a silent mismatch worse than a
/// build error.
///
/// Instead, `MetalBootstrap` invokes `scripts/ensure-mlx-metal-library.sh` on
/// first use. That script compiles the resolved `.build/checkouts/mlx-swift`
/// Metal sources into an untracked, revision-keyed cache under
/// `.build/mlx-metallib-cache/`. A stamp file records the source revision; a
/// different resolved mlx-swift forces a rebuild. Cache hits are near-instant.
///
/// If generation cannot succeed (no checkout, no `xcrun metal`), we fail hard
/// with the script's actionable message — never skip MLX tests quietly.
///
/// See `Tests/AdaptTrainTests/README.md`.
enum MetalBootstrap {
    private static let lock = NSLock()
    // Guarded by `lock`; nonisolated(unsafe) is the standard pattern for
    // one-time process setup under a mutex.
    nonisolated(unsafe) private static var didRun = false
    // Keep the synthetic bundle alive so it remains in `Bundle.allBundles`
    // for Cmlx's SwiftPM load path.
    nonisolated(unsafe) private static var retainedBundle: Bundle?

    /// Locate/build the metallib matching the resolved mlx-swift and place it
    /// where Cmlx can load it. Safe to call repeatedly; runs once per process.
    static func ensureMetallib() {
        lock.lock()
        defer { lock.unlock() }
        guard !didRun else { return }
        didRun = true

        do {
            let packageRoot = try findPackageRoot()
            let metallib = try ensureCachedMetallib(packageRoot: packageRoot)
            try installForCmlx(metallib: metallib, packageRoot: packageRoot)
        } catch {
            // Loud failure: a green suite that never loaded Metal is the
            // outcome we are trying to prevent.
            fatalError(
                """
                MetalBootstrap failed to prepare default.metallib for AdaptTrainTests.

                \(error)

                MLX tests require a metallib built from the resolved mlx-swift
                checkout. See Tests/AdaptTrainTests/README.md.
                """
            )
        }
    }

    // MARK: - Cache / script

    private static func ensureCachedMetallib(packageRoot: URL) throws -> URL {
        let script = packageRoot
            .appendingPathComponent("scripts")
            .appendingPathComponent("ensure-mlx-metal-library.sh")
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw BootstrapError.missingScript(script.path)
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

        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        let outText = String(data: outData, encoding: .utf8) ?? ""
        let errText = String(data: errData, encoding: .utf8) ?? ""

        guard process.terminationStatus == 0 else {
            let combined = [errText, outText]
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
                .joined(separator: "\n")
            throw BootstrapError.scriptFailed(
                status: process.terminationStatus,
                output: combined.isEmpty ? "(no output)" : combined
            )
        }

        // Prefer the CURRENT pointer the script writes; fall back to parsing
        // a "path: …" line so we still work if the pointer write races.
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

        throw BootstrapError.metallibMissingAfterBuild
    }

    // MARK: - Install where Cmlx looks

    /// Cmlx (`device.cpp`) probes, in order:
    /// 1. `current_binary_dir()/mlx.metallib` (dladdr of Cmlx code — the test
    ///    host binary when statically linked)
    /// 2. `current_binary_dir()/Resources/mlx.metallib`
    /// 3. SwiftPM bundles: `{bundle.resourceURL}/mlx-swift_Cmlx.bundle/default.metallib`
    /// 4. `current_binary_dir()/Resources/default.metallib`
    ///
    /// Under modern SwiftPM, the process argv[0] is `swiftpm-testing-helper`
    /// (inside Xcode.app), not the test host. We must install next to
    /// `AdaptPackageTests.xctest`, not next to argv[0].
    private static func installForCmlx(metallib: URL, packageRoot: URL) throws {
        let hostDirs = try testHostInstallDirectories(packageRoot: packageRoot)
        var wrote = 0
        var errors: [String] = []

        for hostDir in hostDirs {
            let destinations = [
                hostDir.appendingPathComponent("mlx.metallib"),
                hostDir.appendingPathComponent("default.metallib"),
                hostDir.deletingLastPathComponent() // Contents
                    .appendingPathComponent("Resources")
                    .appendingPathComponent("mlx.metallib"),
                hostDir.deletingLastPathComponent()
                    .appendingPathComponent("Resources")
                    .appendingPathComponent("default.metallib"),
                hostDir.deletingLastPathComponent()
                    .appendingPathComponent("Resources")
                    .appendingPathComponent("mlx-swift_Cmlx.bundle")
                    .appendingPathComponent("default.metallib"),
            ]
            for dest in destinations {
                do {
                    try copyFile(from: metallib, to: dest)
                    wrote += 1
                } catch {
                    errors.append("\(dest.path): \(error)")
                }
            }
        }

        // Also stage a loadable bundle under the cache and retain it so
        // NSBundle.allBundles exposes the SWIFTPM_BUNDLE path to Cmlx.
        do {
            let stagedBundle = try stageSwiftPMBundle(
                metallib: metallib,
                packageRoot: packageRoot
            )
            if let bundle = Bundle(url: stagedBundle) {
                retainedBundle = bundle
                // Touch resource URL so the bundle is fully realized.
                _ = bundle.resourceURL
                wrote += 1
            }
        } catch {
            errors.append("stage bundle: \(error)")
        }

        if wrote == 0 {
            throw BootstrapError.copyFailed(
                errors.isEmpty ? "no install locations found" : errors.joined(separator: "\n")
            )
        }
    }

    private static func stageSwiftPMBundle(metallib: URL, packageRoot: URL) throws -> URL {
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
        try copyFile(from: metallib, to: dest)
        return bundleDir
    }

    private static func copyFile(from source: URL, to dest: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: dest.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if fm.fileExists(atPath: dest.path) {
            try fm.removeItem(at: dest)
        }
        try fm.copyItem(at: source, to: dest)
    }

    /// Directories to place colocated metallibs (typically
    /// `…/AdaptPackageTests.xctest/Contents/MacOS`).
    private static func testHostInstallDirectories(packageRoot: URL) throws -> [URL] {
        var dirs: [URL] = []
        var seen = Set<String>()

        func add(_ url: URL) {
            let path = url.standardizedFileURL.path
            guard !seen.contains(path) else { return }
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
                  isDir.boolValue
            else { return }
            seen.insert(path)
            dirs.append(url)
        }

        // 1. Paths mentioned on the command line (swiftpm-testing-helper passes
        //    the real test bundle path as an argument).
        for arg in CommandLine.arguments {
            let url = URL(fileURLWithPath: arg).resolvingSymlinksInPath()
            if arg.contains("AdaptPackageTests.xctest") {
                if url.pathExtension == "xctest" {
                    add(url.appendingPathComponent("Contents/MacOS"))
                } else if url.lastPathComponent == "AdaptPackageTests" {
                    add(url.deletingLastPathComponent())
                } else if url.path.contains(".xctest/") {
                    // Strip to …/Something.xctest/Contents/MacOS
                    var cursor = url
                    while cursor.path != "/", cursor.pathExtension != "xctest" {
                        cursor = cursor.deletingLastPathComponent()
                    }
                    if cursor.pathExtension == "xctest" {
                        add(cursor.appendingPathComponent("Contents/MacOS"))
                    }
                }
            }
        }

        // 2. Scan package .build for the test host (covers argv layouts that
        //    omit the path, and multi-arch build folders). Stay shallow: only
        //    triple directories under .build, never walk checkouts/.
        let buildRoot = packageRoot.appendingPathComponent(".build")
        let fm = FileManager.default
        if let triples = try? fm.contentsOfDirectory(
            at: buildRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) {
            for triple in triples {
                for config in ["debug", "release"] {
                    let candidate = triple
                        .appendingPathComponent(config)
                        .appendingPathComponent("AdaptPackageTests.xctest")
                    if fm.fileExists(atPath: candidate.path) {
                        add(candidate.appendingPathComponent("Contents/MacOS"))
                    }
                }
                // Some layouts put the product directly under the triple.
                let direct = triple.appendingPathComponent("AdaptPackageTests.xctest")
                if fm.fileExists(atPath: direct.path) {
                    add(direct.appendingPathComponent("Contents/MacOS"))
                }
            }
        }

        if dirs.isEmpty {
            throw BootstrapError.testHostNotFound
        }
        return dirs
    }

    // MARK: - Package root

    /// Walk up from argv paths and the CWD until we find a directory that
    /// contains both `Package.swift` and `scripts/ensure-mlx-metal-library.sh`.
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
        throw BootstrapError.packageRootNotFound
    }

    // MARK: - Errors

    private enum BootstrapError: Error, CustomStringConvertible {
        case missingScript(String)
        case scriptFailed(status: Int32, output: String)
        case metallibMissingAfterBuild
        case copyFailed(String)
        case packageRootNotFound
        case testHostNotFound

        var description: String {
            switch self {
            case .missingScript(let path):
                return "build script not found at \(path)"
            case .scriptFailed(let status, let output):
                return """
                scripts/ensure-mlx-metal-library.sh exited with status \(status):

                \(output)
                """
            case .metallibMissingAfterBuild:
                return """
                build script succeeded but no metallib was found under \
                .build/mlx-metallib-cache/. Check script output and cache permissions.
                """
            case .copyFailed(let detail):
                return "could not install metallib for the test host:\n\(detail)"
            case .packageRootNotFound:
                return """
                could not locate package root (Package.swift + scripts/ensure-mlx-metal-library.sh) \
                from the test process. Run 'swift test' from the repository root.
                """
            case .testHostNotFound:
                return """
                could not locate AdaptPackageTests.xctest under .build/. \
                Run 'swift test' from the repository root so the test host is built.
                """
            }
        }
    }
}
