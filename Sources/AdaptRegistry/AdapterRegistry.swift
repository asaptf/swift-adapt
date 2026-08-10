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
///
/// A version directory becomes visible under `v<N>/` only after a full staged
/// write is moved into place atomically. Incomplete or unreadable version
/// directories (including leftovers from older builds) are ignored by listing
/// and never brick a lineage.
public actor AdapterRegistry {
    /// Configurable root directory for all lineages.
    public let rootURL: URL

    /// Optional fault injection for crash-safety tests. Production code leaves this nil/false.
    /// Package-internal only — not a public affordance.
    package var faultBeforePointerFlip: Bool = false
    /// When true, the next store completes weights + config writes then throws before version.json.
    package var faultAfterWeightsWrite: Bool = false

    private let fileManager = FileManager.default

    /// Prefix for staged version directories that are not yet committed as `v<N>`.
    private static let stagingPrefix = ".staging-"

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
    /// Builds the version in a private `.staging-<uuid>/` directory, then moves
    /// it into place as `v<N>/` in one filesystem operation so a partial version
    /// is never observable under a `v<N>` name. Does not change the active pointer.
    ///
    /// - Returns: The stored `AdapterVersion` metadata (status `.candidate`).
    @discardableResult
    public func storeCandidate(
        lineage: AdapterLineage,
        weights: Data,
        trainedOn: TrainingWindow,
        parentVersion: Int? = nil,
        evalReport: EvalReport? = nil,
        promptFormat: PromptFormatConvention? = nil
    ) throws -> AdapterVersion {
        let lineageID = lineage.lineageID
        let lineageDir = try lineageDirectory(for: lineageID)
        try fileManager.createDirectory(at: lineageDir, withIntermediateDirectories: true)
        try FileProtection.apply(lineageDir)

        // Ensure state.json exists so the lineage is always openable.
        let statePath = try stateURL(lineageID: lineageID)
        if !fileManager.fileExists(atPath: statePath.path) {
            try AtomicFileWriter.writeJSON(LineageState(), to: statePath)
        }

        let versionNumber = try nextVersionNumber(lineageID: lineageID)
        let versionDir = try versionDirectory(lineageID: lineageID, version: versionNumber)

        // Stage fully, then rename into `v<N>/` so observers never see a partial version.
        let stagingName = "\(Self.stagingPrefix)\(UUID().uuidString)"
        let stagingDir = lineageDir.appendingPathComponent(stagingName, isDirectory: true)
        try fileManager.createDirectory(at: stagingDir, withIntermediateDirectories: true)
        try FileProtection.apply(stagingDir)

        do {
            let digest = Self.sha256Hex(weights)

            // 1. adapter_config.json (upstream-compatible LoRAConfig)
            try AtomicFileWriter.writeJSON(
                lineage.loraConfig,
                to: stagingDir.appendingPathComponent("adapter_config.json")
            )

            // 2. weights blob
            try AtomicFileWriter.write(
                data: weights,
                to: stagingDir.appendingPathComponent("adapters.safetensors")
            )

            if faultAfterWeightsWrite {
                faultAfterWeightsWrite = false
                throw RegistryTestFault.injected("after weights write, before version.json")
            }

            let metadata = AdapterVersion(
                lineage: lineage,
                version: versionNumber,
                parentVersion: parentVersion,
                trainedOn: trainedOn,
                evalReport: evalReport,
                status: .candidate,
                weightsDigest: digest,
                createdAt: Date(),
                promptFormat: promptFormat
            )

            // 3. version.json — last file before the version becomes visible.
            try AtomicFileWriter.writeJSON(
                metadata,
                to: stagingDir.appendingPathComponent("version.json")
            )

            // Publish: single rename within the lineage directory (atomic on APFS/HFS+).
            guard !fileManager.fileExists(atPath: versionDir.path) else {
                try? fileManager.removeItem(at: stagingDir)
                throw AdaptRegistryError.ioFailed(
                    "Refusing to overwrite existing version directory v\(versionNumber)"
                )
            }
            try fileManager.moveItem(at: stagingDir, to: versionDir)
            try FileProtection.apply(versionDir)

            return metadata
        } catch {
            // Leave staging in place only for injected crash tests; clean other failures.
            // GC removes leftover staging directories either way.
            if !(error is RegistryTestFault) {
                try? fileManager.removeItem(at: stagingDir)
            }
            throw error
        }
    }

    /// Promotes a candidate (or any stored version) to active for its lineage.
    ///
    /// Verifies weights integrity before flipping the pointer. Demotes the previous
    /// active version to `.rolledBack`, sets the new version to `.active`, then
    /// atomically flips `state.json`. At most one active version remains. Weight
    /// files are never moved.
    public func promote(lineage: AdapterLineage, version: Int) throws {
        try promote(lineageID: lineage.lineageID, version: version)
    }

    /// Promotes by lineage ID (directory name).
    ///
    /// Always verifies weights integrity — an unverified adapter must never become active.
    public func promote(lineageID: String, version: Int) throws {
        try Self.validateLineageID(lineageID)
        let versionDir = try versionDirectory(lineageID: lineageID, version: version)
        guard fileManager.fileExists(atPath: versionDir.path) else {
            throw AdaptRegistryError.notFound("version v\(version) in lineage \(lineageID)")
        }

        // Verify integrity before flipping the pointer.
        _ = try loadVersion(lineageID: lineageID, version: version, verifyIntegrity: true)

        let state = try loadState(lineageID: lineageID)
        if state.activeVersion == version {
            // Already active — ensure status field is correct and return.
            try updateVersionStatus(lineageID: lineageID, version: version, status: .active)
            return
        }

        try commitActivePointer(lineageID: lineageID, newActive: version)
    }

    /// Rolls back to a previous version via an O(1) pointer flip.
    ///
    /// Verifies the target's weights integrity before flipping. Does not move or
    /// rewrite weight files. The formerly active version is marked `.rolledBack`.
    /// Passing a missing version fails without changing state.
    public func rollback(lineage: AdapterLineage, to version: Int) throws {
        try rollback(lineageID: lineage.lineageID, to: version)
    }

    /// Rolls back by lineage ID.
    ///
    /// Always verifies weights integrity of the rollback target.
    public func rollback(lineageID: String, to version: Int) throws {
        try Self.validateLineageID(lineageID)
        let versionDir = try versionDirectory(lineageID: lineageID, version: version)
        guard fileManager.fileExists(atPath: versionDir.path) else {
            throw AdaptRegistryError.notFound("version v\(version) in lineage \(lineageID)")
        }

        _ = try loadVersion(lineageID: lineageID, version: version, verifyIntegrity: true)

        let state = try loadState(lineageID: lineageID)
        if state.activeVersion == version {
            return
        }

        try commitActivePointer(lineageID: lineageID, newActive: version)
    }

    /// Records an evaluation report on an existing version's `version.json`.
    ///
    /// Does not change the active pointer, weights, or status. Intended for
    /// post-train **measurements** (e.g. held-out cross-entropy) and later for
    /// M3 gate results. Replaces any previous `evalReport` on that version.
    public func recordEvalReport(
        lineage: AdapterLineage,
        version: Int,
        report: EvalReport
    ) throws {
        try recordEvalReport(lineageID: lineage.lineageID, version: version, report: report)
    }

    /// Records an evaluation report by lineage ID.
    public func recordEvalReport(
        lineageID: String,
        version: Int,
        report: EvalReport
    ) throws {
        try Self.validateLineageID(lineageID)
        let metaURL = try versionDirectory(lineageID: lineageID, version: version)
            .appendingPathComponent("version.json")
        guard fileManager.fileExists(atPath: metaURL.path) else {
            throw AdaptRegistryError.notFound("version.json for v\(version) in \(lineageID)")
        }
        let meta = try AtomicFileWriter.readJSON(AdapterVersion.self, from: metaURL)
        let updated = meta.with(evalReport: report)
        try AtomicFileWriter.writeJSON(updated, to: metaURL)
    }

    /// Clears the active pointer so the lineage uses base-model behavior.
    public func clearActive(lineage: AdapterLineage) throws {
        try clearActive(lineageID: lineage.lineageID)
    }

    /// Clears the active pointer by lineage ID.
    public func clearActive(lineageID: String) throws {
        try Self.validateLineageID(lineageID)
        try commitActivePointer(lineageID: lineageID, newActive: nil)
    }

    /// Returns the active version metadata, or `nil` if none (base-model behavior).
    ///
    /// By default this is a **metadata-only** read: it does **not** hash the
    /// weights file. Integrity is enforced on promote/rollback and when loading
    /// weights for train/inference. Pass `verifyIntegrity: true` to demand a
    /// full SHA-256 check of `adapters.safetensors`.
    public func activeVersion(
        for lineage: AdapterLineage,
        verifyIntegrity: Bool = false
    ) throws -> AdapterVersion? {
        try activeVersion(lineageID: lineage.lineageID, verifyIntegrity: verifyIntegrity)
    }

    /// Returns the active version by lineage ID.
    ///
    /// - Parameter verifyIntegrity: When `true`, SHA-256s the weights file.
    ///   Default `false` so polling the active pointer (e.g. session reload)
    ///   stays cheap; use `true` when about to load weights.
    public func activeVersion(
        lineageID: String,
        verifyIntegrity: Bool = false
    ) throws -> AdapterVersion? {
        try Self.validateLineageID(lineageID)
        let state = try loadState(lineageID: lineageID)
        guard let version = state.activeVersion else { return nil }
        return try loadVersion(
            lineageID: lineageID,
            version: version,
            verifyIntegrity: verifyIntegrity
        )
    }

    /// Lists all **complete, readable** stored versions for a lineage, sorted
    /// by version number ascending.
    ///
    /// Incomplete or unreadable `vN` directories (missing/corrupt `version.json`)
    /// and staging leftovers are skipped so a single bad directory cannot take
    /// down the lineage. Status fields are reconciled against `state.json` so
    /// at most one reports `.active`.
    public func listVersions(for lineage: AdapterLineage) throws -> [AdapterVersion] {
        try listVersions(lineageID: lineage.lineageID)
    }

    /// Lists complete versions by lineage ID. Partial/unreadable directories are ignored.
    public func listVersions(lineageID: String) throws -> [AdapterVersion] {
        try Self.validateLineageID(lineageID)
        let lineageDir = try lineageDirectory(for: lineageID)
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
            throw AdaptRegistryError.ioFailed("Failed to list lineage \(lineageID): \(error.localizedDescription)")
        }

        var versions: [AdapterVersion] = []
        for url in contents {
            let name = url.lastPathComponent
            guard let number = Self.parseVersionDirectoryName(name) else { continue }

            // Skip incomplete/unreadable versions rather than failing the lineage.
            guard var meta = try? loadVersion(
                lineageID: lineageID,
                version: number,
                verifyIntegrity: false
            ) else {
                continue
            }

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
    ///
    /// - Parameter verifyIntegrity: Defaults to `true` because callers typically
    ///   load a specific version in order to use its weights.
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
    ///
    /// Lineage IDs from `AdapterLineage` are always valid SHA-256 hex; this path
    /// helper traps only if that invariant is broken.
    public func weightsURL(for lineage: AdapterLineage, version: Int) -> URL {
        versionDirectoryUnchecked(lineageID: lineage.lineageID, version: version)
            .appendingPathComponent("adapters.safetensors")
    }

    /// Absolute URL of a version directory.
    public func directoryURL(for lineage: AdapterLineage, version: Int) -> URL {
        versionDirectoryUnchecked(lineageID: lineage.lineageID, version: version)
    }

    /// Deletes old versions, keeping the most recent `keepLast` plus always the active one.
    ///
    /// Never deletes the active version regardless of `keepLast`. Archived/candidate
    /// versions outside the retention window are removed from disk. Also removes
    /// leftover `.staging-*` directories (crashed mid-store garbage).
    public func gc(lineage: AdapterLineage, keepLast: Int) throws {
        try gc(lineageID: lineage.lineageID, keepLast: keepLast)
    }

    /// GC by lineage ID. Also purges staging leftovers.
    public func gc(lineageID: String, keepLast: Int) throws {
        try Self.validateLineageID(lineageID)
        guard keepLast >= 0 else {
            throw AdaptRegistryError.invalidOperation("keepLast must be >= 0")
        }

        try removeStagingDirectories(lineageID: lineageID)

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
            let dir = try versionDirectory(lineageID: lineageID, version: meta.version)
            do {
                try fileManager.removeItem(at: dir)
            } catch {
                throw AdaptRegistryError.ioFailed(
                    "Failed to GC v\(meta.version): \(error.localizedDescription)"
                )
            }
        }
    }

    // MARK: - Paths

    /// Validates a public `lineageID` path component.
    ///
    /// Lineage IDs are SHA-256 hex by construction (`AdapterLineage.lineageID`):
    /// exactly 64 lowercase hex characters. Reject anything else before it can
    /// escape the registry root via `/`, `..`, or other path tricks.
    package static func validateLineageID(_ lineageID: String) throws {
        guard lineageID.count == 64 else {
            throw AdaptRegistryError.invalidLineageID(
                "expected 64-char SHA-256 hex, got length \(lineageID.count)"
            )
        }
        for scalar in lineageID.unicodeScalars {
            let isDigit = scalar >= "0" && scalar <= "9"
            let isLowerHex = scalar >= "a" && scalar <= "f"
            guard isDigit || isLowerHex else {
                throw AdaptRegistryError.invalidLineageID(
                    "expected lowercase hex [0-9a-f], got \(lineageID)"
                )
            }
        }
    }

    private func lineageDirectory(for lineageID: String) throws -> URL {
        try Self.validateLineageID(lineageID)
        return rootURL.appendingPathComponent(lineageID, isDirectory: true)
    }

    private func versionDirectory(lineageID: String, version: Int) throws -> URL {
        try lineageDirectory(for: lineageID).appendingPathComponent("v\(version)", isDirectory: true)
    }

    /// Path helper for `AdapterLineage`-sourced IDs (already valid SHA-256 hex).
    private func versionDirectoryUnchecked(lineageID: String, version: Int) -> URL {
        rootURL
            .appendingPathComponent(lineageID, isDirectory: true)
            .appendingPathComponent("v\(version)", isDirectory: true)
    }

    private func stateURL(lineageID: String) throws -> URL {
        try lineageDirectory(for: lineageID).appendingPathComponent("state.json")
    }

    // MARK: - Internals

    /// Parses `"v12"` → `12`; returns nil for staging dirs and other names.
    private static func parseVersionDirectoryName(_ name: String) -> Int? {
        guard name.hasPrefix("v"), name.count > 1 else { return nil }
        return Int(name.dropFirst())
    }

    /// Next free version number for `lineageID`.
    ///
    /// **Invariant:** the next number is `1 + max` over **every** on-disk
    /// directory named `v<N>` (complete, partial, or unreadable). We deliberately
    /// do **not** use only `listVersions`, which skips incomplete directories —
    /// otherwise a leftover partial `v2/` would cause the next store to reuse `2`
    /// and collide on the publish rename. Staging dirs (`.staging-*`) are never
    /// version numbers and do not affect allocation.
    private func nextVersionNumber(lineageID: String) throws -> Int {
        let maxOnDisk = try highestVersionDirectoryNumber(lineageID: lineageID)
        return maxOnDisk + 1
    }

    /// Highest `N` among sibling directories named `vN`, or `0` if none.
    private func highestVersionDirectoryNumber(lineageID: String) throws -> Int {
        let lineageDir = try lineageDirectory(for: lineageID)
        guard fileManager.fileExists(atPath: lineageDir.path) else { return 0 }

        let contents: [URL]
        do {
            // Include hidden names so we still see everything; version dirs are not hidden.
            contents = try fileManager.contentsOfDirectory(
                at: lineageDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: []
            )
        } catch {
            throw AdaptRegistryError.ioFailed(
                "Failed to scan version dirs for \(lineageID): \(error.localizedDescription)"
            )
        }

        var maxN = 0
        for url in contents {
            if let n = Self.parseVersionDirectoryName(url.lastPathComponent) {
                maxN = max(maxN, n)
            }
        }
        return maxN
    }

    private func loadState(lineageID: String) throws -> LineageState {
        let url = try stateURL(lineageID: lineageID)
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
        let versionDir = try versionDirectory(lineageID: lineageID, version: version)
        let metaURL = versionDir.appendingPathComponent("version.json")
        guard fileManager.fileExists(atPath: metaURL.path) else {
            throw AdaptRegistryError.notFound("version.json for v\(version) in \(lineageID)")
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
                throw AdaptRegistryError.notFound("adapters.safetensors for v\(version)")
            }
            let data: Data
            do {
                data = try Data(contentsOf: weightsURL)
            } catch {
                throw AdaptRegistryError.ioFailed("Failed to read weights: \(error.localizedDescription)")
            }
            let actual = Self.sha256Hex(data)
            if actual != meta.weightsDigest {
                throw AdaptRegistryError.integrityMismatch(expected: meta.weightsDigest, actual: actual)
            }
        }

        return meta
    }

    private func updateVersionStatus(lineageID: String, version: Int, status: AdapterStatus) throws {
        let metaURL = try versionDirectory(lineageID: lineageID, version: version)
            .appendingPathComponent("version.json")
        guard fileManager.fileExists(atPath: metaURL.path) else {
            throw AdaptRegistryError.notFound("version.json for v\(version)")
        }
        let meta = try AtomicFileWriter.readJSON(AdapterVersion.self, from: metaURL)
        let updated = meta.with(status: status)
        try AtomicFileWriter.writeJSON(updated, to: metaURL)
    }

    /// Shared pointer-flip used by `promote`, `rollback`, and `clearActive`.
    ///
    /// Sequence: demote previous active's `version.json` → set target's
    /// `version.json` (if any) → optionally inject a test fault → atomically
    /// write `state.json`. Callers are responsible for existence/integrity checks
    /// and for the "already active" short-circuit where behavior differs.
    private func commitActivePointer(lineageID: String, newActive: Int?) throws {
        var state = try loadState(lineageID: lineageID)
        let previousActive = state.activeVersion

        if let previousActive, previousActive != newActive {
            try updateVersionStatus(lineageID: lineageID, version: previousActive, status: .rolledBack)
        }
        if let newActive {
            try updateVersionStatus(lineageID: lineageID, version: newActive, status: .active)
        }

        if faultBeforePointerFlip {
            faultBeforePointerFlip = false
            throw RegistryTestFault.injected("before state.json pointer flip")
        }

        state.activeVersion = newActive
        try AtomicFileWriter.writeJSON(state, to: try stateURL(lineageID: lineageID))
    }

    /// Removes leftover `.staging-*` directories under a lineage (crashed stores).
    private func removeStagingDirectories(lineageID: String) throws {
        let lineageDir = try lineageDirectory(for: lineageID)
        guard fileManager.fileExists(atPath: lineageDir.path) else { return }

        let contents: [URL]
        do {
            contents = try fileManager.contentsOfDirectory(
                at: lineageDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: []
            )
        } catch {
            throw AdaptRegistryError.ioFailed(
                "Failed to list staging dirs for \(lineageID): \(error.localizedDescription)"
            )
        }

        for url in contents where url.lastPathComponent.hasPrefix(Self.stagingPrefix) {
            do {
                try fileManager.removeItem(at: url)
            } catch {
                throw AdaptRegistryError.ioFailed(
                    "Failed to remove staging \(url.lastPathComponent): \(error.localizedDescription)"
                )
            }
        }
    }

    /// SHA-256 hex digest of `data`.
    public static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
