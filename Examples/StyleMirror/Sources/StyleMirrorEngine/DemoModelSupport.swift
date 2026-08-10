import AdaptCore
import AdaptInference
import AdaptRegistry
import Foundation
import HuggingFace
import MLXLMCommon
import Tokenizers

// MARK: - Hugging Face seams (demo host only — not library modules)

/// Hugging Face hub bridge for `MLXLMCommon.Downloader`.
///
/// Confined to the StyleMirror demo target so AdaptInference stays network-free.
public struct StyleMirrorHubDownloader: Downloader, Sendable {
    private let client: HubClient

    public init(client: HubClient = HubClient()) {
        self.client = client
    }

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        guard let repoID = Repo.ID(rawValue: id) else {
            throw StyleMirrorError.modelUnavailable(
                "invalid Hugging Face repository id '\(id)'"
            )
        }
        let rev = revision ?? "main"
        do {
            return try await client.downloadSnapshot(
                of: repoID,
                revision: rev,
                matching: patterns,
                progressHandler: { @MainActor progress in
                    progressHandler(progress)
                }
            )
        } catch {
            throw StyleMirrorError.modelUnavailable(
                "download failed for \(id): \(error.localizedDescription)"
            )
        }
    }
}

/// Loads `Tokenizers.AutoTokenizer` and adapts it to `MLXLMCommon.Tokenizer`.
public struct StyleMirrorTokenizerLoader: TokenizerLoader, Sendable {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        do {
            let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
            return StyleMirrorBridgedTokenizer(upstream)
        } catch {
            throw StyleMirrorError.modelUnavailable(
                "tokenizer load failed at \(directory.path): \(error.localizedDescription)"
            )
        }
    }
}

struct StyleMirrorBridgedTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        var addGenerationPrompt = true
        if let last = messages.last, let role = last["role"] as? String, role == "assistant" {
            addGenerationPrompt = false
        }
        var jinjaContext = additionalContext ?? [:]
        if let explicit = jinjaContext["add_generation_prompt"] as? Bool {
            addGenerationPrompt = explicit
            jinjaContext.removeValue(forKey: "add_generation_prompt")
        }
        let contextForJinja: [String: any Sendable]? = jinjaContext.isEmpty ? nil : jinjaContext
        do {
            return try upstream.applyChatTemplate(
                messages: messages,
                chatTemplate: nil,
                addGenerationPrompt: addGenerationPrompt,
                truncation: false,
                maxLength: nil,
                tools: tools,
                additionalContext: contextForJinja
            )
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

// MARK: - Metal bootstrap

/// Ensures MLX can find `default.metallib` when running the real StyleMirror engine.
public enum StyleMirrorMetalSupport {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var didRun = false
    nonisolated(unsafe) private static var retainedBundle: Bundle?

    public static func ensureMetallib() throws {
        lock.lock()
        defer { lock.unlock() }
        guard !didRun else { return }
        didRun = true

        let packageRoot = try findAdaptPackageRoot()
        let metallib = try runEnsureScript(packageRoot: packageRoot)
        try installNextToExecutable(metallib: metallib)
        try stageBundle(metallib: metallib, packageRoot: packageRoot)
    }

    private static func runEnsureScript(packageRoot: URL) throws -> URL {
        let script = packageRoot
            .appendingPathComponent("scripts")
            .appendingPathComponent("ensure-mlx-metal-library.sh")
        guard FileManager.default.fileExists(atPath: script.path) else {
            throw StyleMirrorError.modelUnavailable(
                "Metal build script not found at \(script.path)"
            )
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
            throw StyleMirrorError.modelUnavailable(
                "ensure-mlx-metal-library.sh failed: \(combined)"
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

        throw StyleMirrorError.modelUnavailable(
            "metallib missing after build under .build/mlx-metallib-cache/"
        )
    }

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

    /// Walks up from cwd / argv looking for the Adapt package root.
    public static func findAdaptPackageRoot() throws -> URL {
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
            for _ in 0..<20 {
                let packageSwift = dir.appendingPathComponent("Package.swift")
                let script = dir
                    .appendingPathComponent("scripts")
                    .appendingPathComponent("ensure-mlx-metal-library.sh")
                let sources = dir.appendingPathComponent("Sources/AdaptCore")
                if FileManager.default.fileExists(atPath: packageSwift.path),
                   FileManager.default.fileExists(atPath: script.path),
                   FileManager.default.fileExists(atPath: sources.path)
                {
                    return dir
                }
                // StyleMirror nested package: parent is Adapt root.
                let parentSources = dir
                    .deletingLastPathComponent()
                    .deletingLastPathComponent()
                    .appendingPathComponent("Sources/AdaptCore")
                if dir.lastPathComponent == "StyleMirror",
                   FileManager.default.fileExists(atPath: parentSources.path)
                {
                    return dir.deletingLastPathComponent().deletingLastPathComponent()
                }
                let parent = dir.deletingLastPathComponent()
                if parent.path == dir.path { break }
                dir = parent
            }
        }
        throw StyleMirrorError.modelUnavailable(
            "could not locate Adapt package root (Package.swift + scripts/ensure-mlx-metal-library.sh)"
        )
    }
}

// MARK: - Model load helpers

enum DemoModelLoader {
    static func loadContext(
        modelID: String,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContext {
        try StyleMirrorMetalSupport.ensureMetallib()
        do {
            return try await AdaptModelLoader.loadContext(
                source: .id(modelID),
                downloader: StyleMirrorHubDownloader(),
                tokenizerLoader: StyleMirrorTokenizerLoader(),
                progressHandler: progressHandler
            )
        } catch let error as StyleMirrorError {
            throw error
        } catch {
            throw StyleMirrorError.modelUnavailable(error.localizedDescription)
        }
    }

    static func makeSession(
        modelID: String,
        lineage: AdapterLineage,
        registry: AdapterRegistry,
        loadActiveAdapter: Bool
    ) async throws -> AdaptSession {
        try StyleMirrorMetalSupport.ensureMetallib()
        do {
            return try await AdaptSession(
                model: .id(modelID),
                lineage: lineage,
                registry: registry,
                tokenizerLoader: StyleMirrorTokenizerLoader(),
                downloader: StyleMirrorHubDownloader(),
                loadActiveAdapter: loadActiveAdapter
            )
        } catch let error as StyleMirrorError {
            throw error
        } catch {
            throw StyleMirrorError.modelUnavailable(error.localizedDescription)
        }
    }
}
