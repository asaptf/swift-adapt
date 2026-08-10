import AdaptCore
import AdaptData
import Foundation
import Testing

@Suite("Schema migration safety")
struct SchemaMigrationTests {

    @Test("database written at v1 opens and retains rows after additive schema change")
    func roundTripAfterSchemaChange() async throws {
        let root = try AdaptDataTestSupport.makeTempRoot()
        defer { AdaptDataTestSupport.teardown(root) }

        let lineage = AdaptDataTestSupport.makeLineage(taskID: "migration-task")
        let config = ReplayBufferConfiguration(
            maxCapturesPerDay: 50,
            scrubberPipeline: ScrubberPipeline(scrubbers: [])
        )

        let id = UUID()
        let prompt = "stable-prompt-\(id.uuidString.prefix(8))"
        let completion = "stable-completion"

        do {
            let buffer = try ReplayBuffer(
                lineage: lineage,
                rootURL: root,
                configuration: config
            )
            #expect(try await buffer.schemaVersion() == 1)
            _ = try await buffer.add(
                AdaptDataTestSupport.example(
                    id: id,
                    prompt: prompt,
                    completion: completion,
                    capturedAt: Date(timeIntervalSince1970: 1_700_000_000)
                )
            )
            // Simulate a schema change applied by a newer build.
            try await buffer.applyTestMigrationToV2()
            #expect(try await buffer.schemaVersion() == 2)

            let rows = try await buffer.examples()
            #expect(rows.count == 1)
            #expect(rows[0].id == id)
            #expect(rows[0].prompt == prompt)
            #expect(rows[0].completion == completion)
        }

        // Re-open from disk: reader tolerates v2 and still returns core columns.
        let reopened = try ReplayBuffer(
            lineage: lineage,
            rootURL: root,
            configuration: config
        )
        #expect(try await reopened.schemaVersion() == 2)
        let again = try await reopened.examples()
        #expect(again.count == 1)
        #expect(again[0].id == id)
        #expect(again[0].prompt == prompt)
        #expect(again[0].completion == completion)
    }

    @Test("unsupported future schema version fails clearly")
    func unsupportedSchemaFails() async throws {
        let root = try AdaptDataTestSupport.makeTempRoot()
        defer { AdaptDataTestSupport.teardown(root) }

        let lineage = AdaptDataTestSupport.makeLineage(taskID: "future-schema")
        let buffer = try ReplayBuffer(
            lineage: lineage,
            rootURL: root,
            configuration: ReplayBufferConfiguration(scrubberPipeline: ScrubberPipeline(scrubbers: []))
        )
        // Corrupt user_version beyond maxReadableSchemaVersion (2).
        // Use a second connection via sqlite3 CLI would be heavy; instead rewrite
        // by applying v2 then manually... We only expose applyTestMigrationToV2.
        // Force version 99 via a tiny raw open in-process:
        try forceUserVersion(databaseURL: buffer.databaseURL, version: 99)

        #expect(throws: AdaptDataError.self) {
            _ = try ReplayBuffer(
                lineage: lineage,
                rootURL: root,
                configuration: ReplayBufferConfiguration()
            )
        }
    }
}

// MARK: - Helpers

import SQLite3

private func forceUserVersion(databaseURL: URL, version: Int) throws {
    var db: OpaquePointer?
    let status = databaseURL.path.withCString { path in
        sqlite3_open_v2(path, &db, SQLITE_OPEN_READWRITE, nil)
    }
    guard status == SQLITE_OK, let db else {
        throw AdaptDataError.storageFailed("open for forceUserVersion failed")
    }
    defer { sqlite3_close(db) }
    let sql = "PRAGMA user_version = \(version);"
    var err: UnsafeMutablePointer<CChar>?
    let execStatus = sqlite3_exec(db, sql, nil, nil, &err)
    if execStatus != SQLITE_OK {
        let message = err.map { String(cString: $0) } ?? "?"
        sqlite3_free(err)
        throw AdaptDataError.storageFailed(message)
    }
}
