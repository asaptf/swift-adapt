import AdaptCore
import CryptoKit
import Foundation

/// Versioned adapter store with atomic promote/rollback and integrity checks.
///
/// All shared mutable state is actor-isolated. On-disk layout:
/// ```
/// <root>/<lineageID>/v<N>/adapter_config.json
/// <root>/<lineageID>/v<N>/adapters.safetensors
/// <root>/<lineageID>/v<N>/version.json
/// <root>/<lineageID>/state.json
/// ```
public actor AdapterRegistry {
    /// Configurable root directory for all lineages.
    public let rootURL: URL

    /// Optional fault injection for crash-safety tests. Production code leaves this nil.
    package var faultBeforePointerFlip: Bool = false
    /// When true, the next store completes weights + config writes then throws before version.json.
    package var faultAfterWeightsWrite: Bool = false

    private let fileManager = FileManager.default

    /// Creates a registry rooted at `rootURL`, creating the directory if needed.
    ///
    /// - Parameter rootURL: Store root. Defaults to Application Support/Adapt.
    public init(rootURL: URL? = nil) throws {
        let resolved: URL
        if let rootURL {
            resolved = rootURL
        } else {
            let appSupport = try fileManager.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
            resolved = appSupport.appendingPathComponent("Adapt", isDirectory: true)
        }
        self.rootURL = resolved
        try fileManager.createDirectory(at: resolved, withIntermediateDirectories: true)
        try FileProtection.apply(resolved)
    }

    // MARK: - Public API

    /// Stores a new candidate version for `lineage` with the given weight bytes.
    ///
    /// Writes `adapter_config.json`, `adapters.safetensors`, and `version.json` under
    /// `v<N>/` where N is the next free version number. Does not change the active pointer.
    ///
    /// - Returns: The stored `AdapterVersion` metadata (status `.candidate`).
    @discardableResult
    public func storeCandidate(
        lineage: AdapterLineage,
        weights: Data,
        trainedOn: TrainingWindow,
        parentVersion: Int? = nil,
        evalReport: EvalReport? = nil
    ) throws -> AdapterVersion {
        let lineageID = lineage.lineageID
        let lineageDir = lineageDirectory(for: lineageID)
        try fileManager.createDirectory(at: lineageDir, withIntermediateDirectories: true)
        try FileProtection.apply(lineageDir)

        // Ensure state.json exists so the lineage is always openable.
        if !fileManager.fileExists(atPath: stateURL(lineageID: lineageID).path) {
            try AtomicFileWriter.writeJSON(LineageState(), to: stateURL(lineageID: lineageID))
        }

        let versionNumber = try nextVersionNumber(lineageID: lineageID)
        let versionDir = versionDirectory(lineageID: lineageID, version: versionNumber)
        try fileManager.createDirectory(at: versionDir, withIntermediateDirectories: true)
        try FileProtection.apply(versionDir)

        let digest = Self.sha256Hex(weights)

        // 1. adapter_config.json (upstream-compatible LoRAConfig)
        try AtomicFileWriter.writeJSON(
            lineage.loraConfig,
            to: versionDir.appendingPathComponent("adapter_config.json")
        )

        // 2. weights blob
        try AtomicFileWriter.write(
            data: weights,
            to: versionDir.appendingPathComponent("adapters.safetensors")
        )

        if faultAfterWeightsWrite {
            faultAfterWeightsWrite = false
            throw AdaptError.injectedFault("after weights write, before version.json")
        }

        let metadata = AdapterVersion(
            lineage: lineage,
            version: versionNumber,
            parentVersion: parentVersion,
            trainedOn: trainedOn,
            evalReport: evalReport,
            status: .candidate,
            weightsDigest: digest,
            createdAt: Date()
        )

        // 3. version.json
        try AtomicFileWriter.writeJSON(
            metadata,
            to: versionDir.appendingPathComponent("version.json")
        )

        return metadata
    }

    /// Promotes a candidate (or any stored version) to active for its lineage.
    ///
    /// Demotes the previous active version to `.rolledBack`, sets the new version
    /// to `.active`, then atomically flips `state.json`. At most one active version
    /// remains. Weight files are never moved.
    public func promote(lineage: AdapterLineage, version: Int) throws {
        try promote(lineageID: lineage.lineageID, version: version)
    }

    /// Promotes by lineage ID (directory name).
    public func promote(lineageID: String, version: Int) throws {
        let versionDir = versionDirectory(lineageID: lineageID, version: version)
        guard fileManager.fileExists(atPath: versionDir.path) else {
            throw AdaptError.notFound("version v\(version) in lineage \(lineageID)")
        }

        // Verify integrity before flipping the pointer.
        _ = try loadVersion(lineageID: lineageID, version: version, verifyIntegrity: true)

        var state = try loadState(lineageID: lineageID)
        let previousActive = state.activeVersion

        if previousActive == version {
            // Already active — ensure status field is correct and return.
            try updateVersionStatus(lineageID: lineageID, version: version, status: .active)
            return
        }

        // Update version.json statuses first; state.json last (source of truth for active).
        if let previousActive {
            try updateVersionStatus(lineageID: lineageID, version: previousActive, status: .rolledBack)
        }
        try updateVersionStatus(lineageID: lineageID, version: version, status: .active)

        if faultBeforePointerFlip {
            faultBeforePointerFlip = false
            throw AdaptError.injectedFault("before state.json pointer flip")
        }

        state.activeVersion = version
        try AtomicFileWriter.writeJSON(state, to: stateURL(lineageID: lineageID))
    }

    /// Rolls back to a previous version via an O(1) pointer flip.
    ///
    /// Does not move or rewrite weight files. The formerly active version is marked
    /// `.rolledBack`. Passing a missing version fails without changing state.
    public func rollback(lineage: AdapterLineage, to version: Int) throws {
        try rollback(lineageID: lineage.lineageID, to: version)
    }

    /// Rolls back by lineage ID.
    public func rollback(lineageID: String, to version: Int) throws {
        let versionDir = versionDirectory(lineageID: lineageID, version: version)
        guard fileManager.fileExists(atPath: versionDir.path) else {
            throw AdaptError.notFound("version v\(version) in lineage \(lineageID)")
        }

        _ = try loadVersion(lineageID: lineageID, version: version, verifyIntegrity: true)

        var state = try loadState(lineageID: lineageID)
        let previousActive = state.activeVersion

        if previousActive == version {
            return
        }

        if let previousActive {
            try updateVersionStatus(lineageID: lineageID, version: previousActive, status: .rolledBack)
        }
        try updateVersionStatus(lineageID: lineageID, version: version, status: .active)

        if faultBeforePointerFlip {
            faultBeforePointerFlip = false
            throw AdaptError.injectedFault("before state.json pointer flip on rollback")
        }

        state.activeVersion = version
        try AtomicFileWriter.writeJSON(state, to: stateURL(lineageID: lineageID))
    }

    /// Clears the active pointer so the lineage uses base-model behavior.
    public func clearActive(lineage: AdapterLineage) throws {
        try clearActive(lineageID: lineage.lineageID)
    }

    /// Clears the active pointer by lineage ID.
    public func clearActive(lineageID: String) throws {
        var state = try loadState(lineageID: lineageID)
        if let previous = state.activeVersion {
            try updateVersionStatus(lineageID: lineageID, version: previous, status: .rolledBack)
        }
        state.activeVersion = nil
        try AtomicFileWriter.writeJSON(state, to: stateURL(lineageID: lineageID))
    }

    /// Returns the active version metadata, or `nil` if none (base-model behavior).
    public func activeVersion(for lineage: AdapterLineage) throws -> AdapterVersion? {
        try activeVersion(lineageID: lineage.lineageID)
    }

    /// Returns the active version by lineage ID.
    public func activeVersion(lineageID: String) throws -> AdapterVersion? {
        let state = try loadState(lineageID: lineageID)
        guard let version = state.activeVersion else { return nil }
        return try loadVersion(lineageID: lineageID, version: version, verifyIntegrity: true)
    }

    /// Lists all stored versions for a lineage, sorted by version number ascending.
    ///
    /// Status fields are reconciled against `state.json` so at most one reports `.active`.
    public func listVersions(for lineage: AdapterLineage) throws -> [AdapterVersion] {
        try listVersions(lineageID: lineage.lineageID)
    }

    /// Lists versions by lineage ID.
    public func listVersions(lineageID: String) throws -> [AdapterVersion] {
        let lineageDir = lineageDirectory(for: lineageID)
        guard fileManager.fileExists(atPath: lineageDir.path) else {
            return []
        }

        let state = try loadState(lineageID: lineageID)
        let contents: [URL]
        do {
            contents = try fileManager.contentsOfDirectory(
                at: lineageDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            throw AdaptError.ioFailed("Failed to list lineage \(lineageID): \(error.localizedDescription)")
        }

        var versions: [AdapterVersion] = []
        for url in contents {
            let name = url.lastPathComponent
            guard name.hasPrefix("v"),
                  let number = Int(name.dropFirst())
            else { continue }

            var meta = try loadVersion(lineageID: lineageID, version: number, verifyIntegrity: false)
            // Reconcile status with state.json (source of truth for active).
            if state.activeVersion == number {
                meta = meta.with(status: .active)
            } else if meta.status == .active {
                meta = meta.with(status: .rolledBack)
            }
            versions.append(meta)
        }
        return versions.sorted { $0.version < $1.version }
    }

    /// Loads one version and optionally verifies the weights digest.
    public func version(
        for lineage: AdapterLineage,
        version: Int,
        verifyIntegrity: Bool = true
    ) throws -> AdapterVersion {
        try loadVersion(
            lineageID: lineage.lineageID,
            version: version,
            verifyIntegrity: verifyIntegrity
        )
    }

    /// Absolute URL of a version's weights file (for later modules / CLI).
    public func weightsURL(for lineage: AdapterLineage, version: Int) -> URL {
        versionDirectory(lineageID: lineage.lineageID, version: version)
            .appendingPathComponent("adapters.safetensors")
    }

    /// Absolute URL of a version directory.
    public func directoryURL(for lineage: AdapterLineage, version: Int) -> URL {
        versionDirectory(lineageID: lineage.lineageID, version: version)
    }

    /// Deletes old versions, keeping the most recent `keepLast` plus always the active one.
    ///
    /// Never deletes the active version regardless of `keepLast`. Archived/candidate
    /// versions outside the retention window are removed from disk.
    public func gc(lineage: AdapterLineage, keepLast: Int) throws {
        try gc(lineageID: lineage.lineageID, keepLast: keepLast)
    }

    /// GC by lineage ID.
    public func gc(lineageID: String, keepLast: Int) throws {
        guard keepLast >= 0 else {
            throw AdaptError.invalidOperation("keepLast must be >= 0")
        }

        let versions = try listVersions(lineageID: lineageID)
        guard !versions.isEmpty else { return }

        let state = try loadState(lineageID: lineageID)
        let active = state.activeVersion

        // Keep the last `keepLast` by version number; always keep active.
        let sortedDesc = versions.sorted { $0.version > $1.version }
        var keep = Set(sortedDesc.prefix(keepLast).map(\.version))
        if let active {
            keep.insert(active)
        }

        for meta in versions where !keep.contains(meta.version) {
            let dir = versionDirectory(lineageID: lineageID, version: meta.version)
            do {
                try fileManager.removeItem(at: dir)
            } catch {
                throw AdaptError.ioFailed(
                    "Failed to GC v\(meta.version): \(error.localizedDescription)"
                )
            }
        }
    }

    // MARK: - Paths

    private func lineageDirectory(for lineageID: String) -> URL {
        rootURL.appendingPathComponent(lineageID, isDirectory: true)
    }

    private func versionDirectory(lineageID: String, version: Int) -> URL {
        lineageDirectory(for: lineageID).appendingPathComponent("v\(version)", isDirectory: true)
    }

    private func stateURL(lineageID: String) -> URL {
        lineageDirectory(for: lineageID).appendingPathComponent("state.json")
    }

    // MARK: - Internals

    private func nextVersionNumber(lineageID: String) throws -> Int {
        let versions = try listVersions(lineageID: lineageID)
        return (versions.map(\.version).max() ?? 0) + 1
    }

    private func loadState(lineageID: String) throws -> LineageState {
        let url = stateURL(lineageID: lineageID)
        if !fileManager.fileExists(atPath: url.path) {
            return LineageState()
        }
        return try AtomicFileWriter.readJSON(LineageState.self, from: url)
    }

    private func loadVersion(
        lineageID: String,
        version: Int,
        verifyIntegrity: Bool
    ) throws -> AdapterVersion {
        let versionDir = versionDirectory(lineageID: lineageID, version: version)
        let metaURL = versionDir.appendingPathComponent("version.json")
        guard fileManager.fileExists(atPath: metaURL.path) else {
            throw AdaptError.notFound("version.json for v\(version) in \(lineageID)")
        }

        var meta = try AtomicFileWriter.readJSON(AdapterVersion.self, from: metaURL)

        // Reconcile active status with state pointer.
        let state = try loadState(lineageID: lineageID)
        if state.activeVersion == version {
            meta = meta.with(status: .active)
        } else if meta.status == .active {
            meta = meta.with(status: .rolledBack)
        }

        if verifyIntegrity {
            let weightsURL = versionDir.appendingPathComponent("adapters.safetensors")
            guard fileManager.fileExists(atPath: weightsURL.path) else {
                throw AdaptError.notFound("adapters.safetensors for v\(version)")
            }
            let data: Data
            do {
                data = try Data(contentsOf: weightsURL)
            } catch {
                throw AdaptError.ioFailed("Failed to read weights: \(error.localizedDescription)")
            }
            let actual = Self.sha256Hex(data)
            if actual != meta.weightsDigest {
                throw AdaptError.integrityMismatch(expected: meta.weightsDigest, actual: actual)
            }
        }

        return meta
    }

    private func updateVersionStatus(lineageID: String, version: Int, status: AdapterStatus) throws {
        let metaURL = versionDirectory(lineageID: lineageID, version: version)
            .appendingPathComponent("version.json")
        guard fileManager.fileExists(atPath: metaURL.path) else {
            throw AdaptError.notFound("version.json for v\(version)")
        }
        let meta = try AtomicFileWriter.readJSON(AdapterVersion.self, from: metaURL)
        let updated = meta.with(status: status)
        try AtomicFileWriter.writeJSON(updated, to: metaURL)
    }

    /// SHA-256 hex digest of `data`.
    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
