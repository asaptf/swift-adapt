import AdaptCLI
import AdaptCore
import Foundation
import Testing

@Suite("DemoCorpus partition")
struct DemoCorpusTests {

    private func examples(_ n: Int) -> [TrainingExample] {
        (0..<n).map { i in
            TrainingExample(
                prompt: "prompt-\(i)",
                completion: "completion-\(i)",
                source: .synthetic
            )
        }
    }

    @Test("seven nights get equal new slices; held-out is disjoint")
    func equalNightsAndHeldOut() throws {
        let all = examples(240)
        let part = try DemoCorpus.partition(all, nightCount: 7, heldOutCount: 30)
        #expect(part.nights.count == 7)
        #expect(part.nights.allSatisfy { $0.count == 30 })
        #expect(part.heldOut.count == 30)
        #expect(part.totalCount == 240)

        // Night slices are prefix blocks; held-out is the tail.
        #expect(part.nights[0].map(\.prompt) == (0..<30).map { "prompt-\($0)" })
        #expect(part.nights[6].map(\.prompt) == (180..<210).map { "prompt-\($0)" })
        #expect(part.heldOut.map(\.prompt) == (210..<240).map { "prompt-\($0)" })

        // No train example appears in held-out.
        let trainPrompts = Set(part.nights.flatMap { $0.map(\.prompt) })
        let heldPrompts = Set(part.heldOut.map(\.prompt))
        #expect(trainPrompts.isDisjoint(with: heldPrompts))
        #expect(trainPrompts.count == 210)
    }

    @Test("remainder after equal nights is absorbed into held-out")
    func remainderGoesToHeldOut() throws {
        // 100 examples, heldOut=20 → train budget 80 → 7 nights * 11 = 77, held-out gets 23
        let all = examples(100)
        let part = try DemoCorpus.partition(all, nightCount: 7, heldOutCount: 20)
        #expect(part.nights.count == 7)
        #expect(part.nights.allSatisfy { $0.count == 11 })
        #expect(part.heldOut.count == 100 - 7 * 11)
        #expect(part.totalCount == 100)
    }

    @Test("rejects too-small corpora")
    func rejectsTooSmall() {
        let all = examples(10)
        #expect(throws: AdaptCLIError.self) {
            _ = try DemoCorpus.partition(all, nightCount: 7, heldOutCount: 30)
        }
    }

    @Test("writePartition emits night and held-out files")
    func writePartitionFiles() throws {
        let all = examples(37) // 7*4 + 9 held-out when heldOutCount=9 → train 28 → 7*4
        let part = try DemoCorpus.partition(all, nightCount: 7, heldOutCount: 9)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("demo-corpus-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try DemoCorpus.writePartition(part, to: dir)
        for i in 1...7 {
            let url = dir.appendingPathComponent("night-\(i).jsonl")
            #expect(FileManager.default.fileExists(atPath: url.path))
            let loaded = try JSONLLoader.load(from: url)
            #expect(loaded.count == part.nights[i - 1].count)
        }
        let held = try JSONLLoader.load(from: dir.appendingPathComponent("held-out.jsonl"))
        #expect(held.count == part.heldOut.count)
        #expect(held.map(\.prompt) == part.heldOut.map(\.prompt))
    }
}
