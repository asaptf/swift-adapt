import AdaptCore
import CryptoKit
import Foundation
import SQLite3

/// Schema version written by this build for new databases.
let currentSchemaVersion = 1
/// Highest schema version this binary can still **read** (forward-compat window).
///
/// v2 is an additive migration used in tests (`notes` column). Production still
/// creates v1; readers tolerate v2 so a DB that underwent a schema change keeps
/// opening without data loss.
let maxReadableSchemaVersion = 2

/// Raw SQLite3 persistence for one lineage's replay buffer.
///
/// Not thread-safe: owned exclusively by ``ReplayBuffer``'s actor isolation.
final class SQLiteBufferStore: @unchecked Sendable {
    private var db: OpaquePointer?
    let databaseURL: URL
    let lineageID: String

    init(databaseURL: URL, lineageID: String) throws {
        self.databaseURL = databaseURL
        self.lineageID = lineageID

        let directory = databaseURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        var handle: OpaquePointer?
        let flags = SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX
        let openStatus = databaseURL.path.withCString { path in
            sqlite3_open_v2(path, &handle, flags, nil)
        }
        guard openStatus == SQLITE_OK, let handle else {
            throw AdaptDataError.storageFailed(
                "sqlite3_open_v2 failed for \(databaseURL.lastPathComponent): \(openStatus)"
            )
        }
        self.db = handle

        do {
            try exec("PRAGMA journal_mode=DELETE;")
            try exec("PRAGMA foreign_keys=ON;")
            try exec("PRAGMA busy_timeout=5000;")
            try migrateIfNeeded()
            try FileProtection.apply(databaseURL)
        } catch {
            sqlite3_close(handle)
            self.db = nil
            throw error
        }
    }

    deinit {
        if let db {
            sqlite3_close(db)
        }
    }

    // MARK: - Schema

    private func migrateIfNeeded() throws {
        let version = try userVersion()
        if version > maxReadableSchemaVersion {
            throw AdaptDataError.unsupportedSchemaVersion(
                found: version,
                supported: maxReadableSchemaVersion
            )
        }
        if version == 0 {
            try applyV1Schema()
            try setUserVersion(currentSchemaVersion)
            return
        }
        // v1 and additive v2 are both readable with the same column set for
        // core fields. Auto-upgrade from v1 → v2 is opt-in (tests / future).
        if version >= 1 {
            return
        }
        throw AdaptDataError.corruptDatabase("unexpected schema version \(version)")
    }

    private func applyV1Schema() throws {
        try exec(
            """
            CREATE TABLE IF NOT EXISTS examples (
                id TEXT PRIMARY KEY NOT NULL,
                lineage_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                completion TEXT NOT NULL,
                weight REAL NOT NULL,
                captured_at REAL NOT NULL,
                source TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                trained_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_examples_lineage_captured
                ON examples(lineage_id, captured_at);
            CREATE INDEX IF NOT EXISTS idx_examples_lineage_source
                ON examples(lineage_id, source);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_examples_lineage_hash
                ON examples(lineage_id, content_hash);

            CREATE TABLE IF NOT EXISTS privacy_budget (
                lineage_id TEXT NOT NULL,
                day TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (lineage_id, day)
            );

            CREATE TABLE IF NOT EXISTS prune_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lineage_id TEXT NOT NULL,
                pruned_at REAL NOT NULL,
                cutoff REAL NOT NULL,
                reason TEXT NOT NULL,
                deleted_ids TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prune_events_lineage
                ON prune_events(lineage_id, pruned_at);
            """
        )
    }

    /// Applies a test-only / forward migration that bumps schema and adds a column.
    ///
    /// Used by migration safety tests. Production code paths call
    /// ``migrateIfNeeded()`` only.
    func applyTestMigrationToV2() throws {
        let version = try userVersion()
        guard version == 1 else {
            throw AdaptDataError.invalidArgument(
                "test migration expects schema v1, found \(version)"
            )
        }
        try exec("ALTER TABLE examples ADD COLUMN notes TEXT NOT NULL DEFAULT '';")
        try setUserVersion(2)
    }

    // MARK: - Examples

    struct StoredExample: Sendable {
        var example: TrainingExample
        var contentHash: String
        var trainedCount: Int
    }

    func insert(example: TrainingExample, contentHash: String) throws {
        let sql = """
            INSERT INTO examples
            (id, lineage_id, prompt, completion, weight, captured_at, source, content_hash, trained_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0);
            """
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }

        try bindText(stmt, 1, example.id.uuidString)
        try bindText(stmt, 2, lineageID)
        try bindText(stmt, 3, example.prompt)
        try bindText(stmt, 4, example.completion)
        try bindDouble(stmt, 5, example.weight)
        try bindDouble(stmt, 6, example.capturedAt.timeIntervalSince1970)
        try bindText(stmt, 7, example.source.rawValue)
        try bindText(stmt, 8, contentHash)

        let status = sqlite3_step(stmt)
        if status == SQLITE_CONSTRAINT {
            throw AdaptDataError.invalidArgument(
                "duplicate content hash or id for example \(example.id)"
            )
        }
        guard status == SQLITE_DONE else {
            throw AdaptDataError.storageFailed("insert example failed: \(status) \(errmsg())")
        }
    }

    func exampleExists(id: UUID) throws -> Bool {
        let sql = "SELECT 1 FROM examples WHERE lineage_id = ? AND id = ? LIMIT 1;"
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bindText(stmt, 1, lineageID)
        try bindText(stmt, 2, id.uuidString)
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    func contentHashExists(_ hash: String) throws -> Bool {
        let sql = "SELECT 1 FROM examples WHERE lineage_id = ? AND content_hash = ? LIMIT 1;"
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bindText(stmt, 1, lineageID)
        try bindText(stmt, 2, hash)
        return sqlite3_step(stmt) == SQLITE_ROW
    }

    func fetchAll() throws -> [TrainingExample] {
        let sql = """
            SELECT id, prompt, completion, weight, captured_at, source
            FROM examples
            WHERE lineage_id = ?
            ORDER BY captured_at ASC, id ASC;
            """
        return try fetchExamples(sql: sql, bind: { stmt in
            try bindText(stmt, 1, lineageID)
        })
    }

    func fetchIDsCapturedBefore(_ cutoff: Date) throws -> [UUID] {
        let sql = """
            SELECT id FROM examples
            WHERE lineage_id = ? AND captured_at < ?
            ORDER BY captured_at ASC, id ASC;
            """
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bindText(stmt, 1, lineageID)
        try bindDouble(stmt, 2, cutoff.timeIntervalSince1970)

        var ids: [UUID] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let cString = sqlite3_column_text(stmt, 0) else { continue }
            let text = String(cString: cString)
            if let id = UUID(uuidString: text) {
                ids.append(id)
            }
        }
        return ids
    }

    func oldestIDs(limit: Int) throws -> [UUID] {
        guard limit > 0 else { return [] }
        let sql = """
            SELECT id FROM examples
            WHERE lineage_id = ?
            ORDER BY captured_at ASC, id ASC
            LIMIT ?;
            """
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bindText(stmt, 1, lineageID)
        sqlite3_bind_int64(stmt, 2, Int64(limit))

        var ids: [UUID] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let cString = sqlite3_column_text(stmt, 0) else { continue }
            let text = String(cString: cString)
            if let id = UUID(uuidString: text) {
                ids.append(id)
            }
        }
        return ids
    }

    func delete(ids: [UUID]) throws {
        guard !ids.isEmpty else { return }
        try exec("BEGIN IMMEDIATE;")
        do {
            let sql = "DELETE FROM examples WHERE lineage_id = ? AND id = ?;"
            let stmt = try prepare(sql)
            defer { sqlite3_finalize(stmt) }
            for id in ids {
                sqlite3_reset(stmt)
                sqlite3_clear_bindings(stmt)
                try bindText(stmt, 1, lineageID)
                try bindText(stmt, 2, id.uuidString)
                let status = sqlite3_step(stmt)
                guard status == SQLITE_DONE else {
                    throw AdaptDataError.storageFailed("delete failed: \(status) \(errmsg())")
                }
            }
            try exec("COMMIT;")
        } catch {
            try? exec("ROLLBACK;")
            throw error
        }
    }

    func count() throws -> Int {
        let sql = "SELECT COUNT(*) FROM examples WHERE lineage_id = ?;"
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bindText(stmt, 1, lineageID)
        guard sqlite3_step(stmt) == SQLITE_ROW else {
            throw AdaptDataError.storageFailed("count failed: \(errmsg())")
        }
        return Int(sqlite3_column_int64(stmt, 0))
    }

    func countsBySource() throws -> [SignalSource: Int] {
        let sql = """
            SELECT source, COUNT(*) FROM examples
            WHERE lineage_id = ?
            GROUP BY source;
            """
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bindText(stmt, 1, lineageID)

        var result: [SignalSource: Int] = [:]
        for source in SignalSource.allCases {
            result[source] = 0
        }
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let cString = sqlite3_column_text(stmt, 0) else { continue }
            let raw = String(cString: cString)
            guard let source = SignalSource(rawValue: raw) else { continue }
            result[source] = Int(sqlite3_column_int64(stmt, 1))
        }
        return result
    }

    func captureBounds() throws -> (oldest: Date?, newest: Date?) {
        let sql = """
            SELECT MIN(captured_at), MAX(captured_at)
            FROM examples WHERE lineage_id = ?;
            """
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bindText(stmt, 1, lineageID)
        guard sqlite3_step(stmt) == SQLITE_ROW else {
            return (nil, nil)
        }
        if sqlite3_column_type(stmt, 0) == SQLITE_NULL {
            return (nil, nil)
        }
        let oldest = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 0))
        let newest = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 1))
        return (oldest, newest)
    }

    // MARK: - Privacy budget

    func captureCount(on day: String) throws -> Int {
        let sql = """
            SELECT count FROM privacy_budget
            WHERE lineage_id = ? AND day = ?;
            """
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bindText(stmt, 1, lineageID)
        try bindText(stmt, 2, day)
        if sqlite3_step(stmt) == SQLITE_ROW {
            return Int(sqlite3_column_int64(stmt, 0))
        }
        return 0
    }

    /// Atomically increments the daily counter if `current + delta <= limit`.
    ///
    /// Returns the new count, or `nil` if the budget would be exceeded.
    func tryIncrementBudget(day: String, delta: Int, limit: Int) throws -> Int? {
        precondition(delta > 0)
        try exec("BEGIN IMMEDIATE;")
        do {
            let current = try captureCount(on: day)
            let next = current + delta
            if next > limit {
                try exec("ROLLBACK;")
                return nil
            }
            let sql = """
                INSERT INTO privacy_budget (lineage_id, day, count)
                VALUES (?, ?, ?)
                ON CONFLICT(lineage_id, day) DO UPDATE SET count = excluded.count;
                """
            let stmt = try prepare(sql)
            defer { sqlite3_finalize(stmt) }
            try bindText(stmt, 1, lineageID)
            try bindText(stmt, 2, day)
            sqlite3_bind_int64(stmt, 3, Int64(next))
            let status = sqlite3_step(stmt)
            guard status == SQLITE_DONE else {
                throw AdaptDataError.storageFailed("budget upsert failed: \(status) \(errmsg())")
            }
            try exec("COMMIT;")
            return next
        } catch {
            try? exec("ROLLBACK;")
            throw error
        }
    }

    // MARK: - Prune events

    func recordPruneEvent(
        prunedAt: Date,
        cutoff: Date,
        reason: PruneReason,
        deletedIDs: [UUID]
    ) throws -> Int64 {
        let encoder = JSONEncoder()
        let data = try encoder.encode(deletedIDs.map(\.uuidString))
        guard let json = String(data: data, encoding: .utf8) else {
            throw AdaptDataError.storageFailed("failed to encode prune deleted_ids")
        }
        let sql = """
            INSERT INTO prune_events (lineage_id, pruned_at, cutoff, reason, deleted_ids)
            VALUES (?, ?, ?, ?, ?);
            """
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bindText(stmt, 1, lineageID)
        try bindDouble(stmt, 2, prunedAt.timeIntervalSince1970)
        try bindDouble(stmt, 3, cutoff.timeIntervalSince1970)
        try bindText(stmt, 4, reason.rawValue)
        try bindText(stmt, 5, json)
        let status = sqlite3_step(stmt)
        guard status == SQLITE_DONE else {
            throw AdaptDataError.storageFailed("insert prune_event failed: \(status) \(errmsg())")
        }
        return sqlite3_last_insert_rowid(db)
    }

    func fetchPruneEvents(limit: Int = 100) throws -> [PruneEvent] {
        let sql = """
            SELECT id, pruned_at, cutoff, reason, deleted_ids
            FROM prune_events
            WHERE lineage_id = ?
            ORDER BY id DESC
            LIMIT ?;
            """
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bindText(stmt, 1, lineageID)
        sqlite3_bind_int64(stmt, 2, Int64(max(0, limit)))

        var events: [PruneEvent] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            let id = sqlite3_column_int64(stmt, 0)
            let prunedAt = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 1))
            let cutoff = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 2))
            guard let reasonC = sqlite3_column_text(stmt, 3),
                  let idsC = sqlite3_column_text(stmt, 4)
            else { continue }
            let reasonRaw = String(cString: reasonC)
            let idsJSON = String(cString: idsC)
            guard let reason = PruneReason(rawValue: reasonRaw),
                  let data = idsJSON.data(using: .utf8),
                  let strings = try? JSONDecoder().decode([String].self, from: data)
            else { continue }
            let ids = strings.compactMap(UUID.init(uuidString:))
            events.append(
                PruneEvent(
                    id: id,
                    lineageID: lineageID,
                    prunedAt: prunedAt,
                    cutoff: cutoff,
                    deletedIDs: ids,
                    reason: reason
                )
            )
        }
        return events
    }

    func wipeAll() throws {
        try exec("BEGIN IMMEDIATE;")
        do {
            let stmt1 = try prepare("DELETE FROM examples WHERE lineage_id = ?;")
            defer { sqlite3_finalize(stmt1) }
            try bindText(stmt1, 1, lineageID)
            guard sqlite3_step(stmt1) == SQLITE_DONE else {
                throw AdaptDataError.storageFailed("wipe examples failed: \(errmsg())")
            }

            let stmt2 = try prepare("DELETE FROM privacy_budget WHERE lineage_id = ?;")
            defer { sqlite3_finalize(stmt2) }
            try bindText(stmt2, 1, lineageID)
            guard sqlite3_step(stmt2) == SQLITE_DONE else {
                throw AdaptDataError.storageFailed("wipe budget failed: \(errmsg())")
            }

            let stmt3 = try prepare("DELETE FROM prune_events WHERE lineage_id = ?;")
            defer { sqlite3_finalize(stmt3) }
            try bindText(stmt3, 1, lineageID)
            guard sqlite3_step(stmt3) == SQLITE_DONE else {
                throw AdaptDataError.storageFailed("wipe prune_events failed: \(errmsg())")
            }
            try exec("COMMIT;")
            try exec("VACUUM;")
        } catch {
            try? exec("ROLLBACK;")
            throw error
        }
    }

    /// Forces journal flush so on-disk bytes match committed state (for tests).
    func checkpointForByteScan() throws {
        // DELETE journal mode: no WAL. A no-op barrier keeps the API stable.
        try exec("PRAGMA schema_version;")
    }

    func userVersion() throws -> Int {
        let stmt = try prepare("PRAGMA user_version;")
        defer { sqlite3_finalize(stmt) }
        guard sqlite3_step(stmt) == SQLITE_ROW else {
            throw AdaptDataError.storageFailed("PRAGMA user_version failed: \(errmsg())")
        }
        return Int(sqlite3_column_int(stmt, 0))
    }

    // MARK: - Internals

    private func setUserVersion(_ version: Int) throws {
        try exec("PRAGMA user_version = \(version);")
    }

    private func fetchExamples(
        sql: String,
        bind: (OpaquePointer) throws -> Void
    ) throws -> [TrainingExample] {
        let stmt = try prepare(sql)
        defer { sqlite3_finalize(stmt) }
        try bind(stmt)

        var examples: [TrainingExample] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            guard let idC = sqlite3_column_text(stmt, 0),
                  let promptC = sqlite3_column_text(stmt, 1),
                  let completionC = sqlite3_column_text(stmt, 2),
                  let sourceC = sqlite3_column_text(stmt, 5)
            else {
                throw AdaptDataError.corruptDatabase("NULL column in examples row")
            }
            let idText = String(cString: idC)
            guard let id = UUID(uuidString: idText) else {
                throw AdaptDataError.corruptDatabase("invalid example id \(idText)")
            }
            let prompt = String(cString: promptC)
            let completion = String(cString: completionC)
            let weight = sqlite3_column_double(stmt, 3)
            let capturedAt = Date(timeIntervalSince1970: sqlite3_column_double(stmt, 4))
            let sourceRaw = String(cString: sourceC)
            guard let source = SignalSource(rawValue: sourceRaw) else {
                throw AdaptDataError.corruptDatabase("unknown SignalSource \(sourceRaw)")
            }
            examples.append(
                TrainingExample(
                    id: id,
                    prompt: prompt,
                    completion: completion,
                    weight: weight,
                    capturedAt: capturedAt,
                    source: source
                )
            )
        }
        return examples
    }

    private func prepare(_ sql: String) throws -> OpaquePointer {
        var stmt: OpaquePointer?
        let status = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
        guard status == SQLITE_OK, let stmt else {
            throw AdaptDataError.storageFailed("prepare failed: \(status) \(errmsg()) — \(sql)")
        }
        return stmt
    }

    private func exec(_ sql: String) throws {
        var errorMessage: UnsafeMutablePointer<CChar>?
        let status = sqlite3_exec(db, sql, nil, nil, &errorMessage)
        if status != SQLITE_OK {
            let message = errorMessage.map { String(cString: $0) } ?? errmsg()
            sqlite3_free(errorMessage)
            throw AdaptDataError.storageFailed("exec failed (\(status)): \(message)")
        }
    }

    private func errmsg() -> String {
        guard let db, let cString = sqlite3_errmsg(db) else { return "unknown" }
        return String(cString: cString)
    }

    private func bindText(_ stmt: OpaquePointer, _ index: Int32, _ value: String) throws {
        let status = value.withCString { cString in
            // SQLITE_TRANSIENT: SQLite copies the bytes.
            sqlite3_bind_text(stmt, index, cString, -1, unsafeBitCast(-1, to: sqlite3_destructor_type.self))
        }
        guard status == SQLITE_OK else {
            throw AdaptDataError.storageFailed("bind text failed: \(status)")
        }
    }

    private func bindDouble(_ stmt: OpaquePointer, _ index: Int32, _ value: Double) throws {
        let status = sqlite3_bind_double(stmt, index, value)
        guard status == SQLITE_OK else {
            throw AdaptDataError.storageFailed("bind double failed: \(status)")
        }
    }
}

// MARK: - Content hash

enum ExampleContentHash {
    /// SHA-256 hex of `prompt \\0 completion` (UTF-8). Dedup key within a lineage.
    static func hash(prompt: String, completion: String) -> String {
        var data = Data()
        data.append(contentsOf: prompt.utf8)
        data.append(0)
        data.append(contentsOf: completion.utf8)
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - Day key

enum PrivacyBudgetDay {
    /// UTC calendar day `YYYY-MM-DD` used as the budget bucket.
    static func key(for date: Date, calendar: Calendar = .init(identifier: .gregorian)) -> String {
        var cal = calendar
        cal.timeZone = TimeZone(secondsFromGMT: 0)!
        let comps = cal.dateComponents([.year, .month, .day], from: date)
        let y = comps.year ?? 1970
        let m = comps.month ?? 1
        let d = comps.day ?? 1
        return String(format: "%04d-%02d-%02d", y, m, d)
    }
}
