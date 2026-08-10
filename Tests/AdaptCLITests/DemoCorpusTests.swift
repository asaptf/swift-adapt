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

        // No train example appears in held-out (pair-level and completion-level).
        let trainPrompts = Set(part.nights.flatMap { $0.map(\.prompt) })
        let heldPrompts = Set(part.heldOut.map(\.prompt))
        #expect(trainPrompts.isDisjoint(with: heldPrompts))
        #expect(trainPrompts.count == 210)
        #expect(part.trainCompletions.isDisjoint(with: part.heldOutCompletions))
        #expect(part.trainCompletions.count == 210)
        #expect(part.heldOutCompletions.count == 30)
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
        #expect(part.trainCompletions.isDisjoint(with: part.heldOutCompletions))
    }

    @Test("rejects too-small corpora")
    func rejectsTooSmall() {
        let all = examples(10)
        #expect(throws: AdaptCLIError.self) {
            _ = try DemoCorpus.partition(all, nightCount: 7, heldOutCount: 30)
        }
    }

    @Test("rejects duplicate completions with counts")
    func rejectsDuplicateCompletions() {
        var all = examples(40)
        // Re-pair an existing completion under a new prompt — classic contamination pad.
        all.append(
            TrainingExample(
                prompt: "prompt-dup",
                completion: "completion-0",
                source: .synthetic
            )
        )
        #expect(throws: AdaptCLIError.self) {
            _ = try DemoCorpus.partition(all, nightCount: 7, heldOutCount: 10)
        }
        do {
            _ = try DemoCorpus.partition(all, nightCount: 7, heldOutCount: 10)
            Issue.record("expected partition to throw on duplicate completions")
        } catch let error as AdaptCLIError {
            let message = error.errorDescription ?? ""
            #expect(message.contains("unique_completions="))
            #expect(message.contains("duplicates="))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test("held-out and train completion texts never overlap")
    func zeroCompletionOverlap() throws {
        let all = examples(240)
        let part = try DemoCorpus.partition(all, nightCount: 7, heldOutCount: 30)
        let overlap = part.trainCompletions.intersection(part.heldOutCompletions)
        #expect(overlap.isEmpty)
        #expect(overlap.count == 0)
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

    @Test("seven-night fixture has unique completions equal to line count")
    func sevenNightFixtureUniqueCompletions() throws {
        let url = fixtureURL("nix-caldera-seven-nights.jsonl")
        let examples = try JSONLLoader.load(from: url)
        #expect(examples.count == 240)

        let completions = examples.map(\.completion)
        let unique = Set(completions)
        #expect(unique.count == examples.count)

        let part = try DemoCorpus.partition(examples, nightCount: 7, heldOutCount: 30)
        #expect(part.nights.count == 7)
        #expect(part.nights.allSatisfy { $0.count == 30 })
        #expect(part.heldOut.count == 30)
        let overlap = part.trainCompletions.intersection(part.heldOutCompletions)
        #expect(overlap.isEmpty, "held-out contaminated: \(overlap.count) shared completions")
    }

    @Test("Renna Vale seven-night fixture: unique completions, zero train/held-out overlap")
    func rennaSevenNightFixtureUniqueAndDisjoint() throws {
        let url = fixtureURL("renna-vale-seven-nights.jsonl")
        let examples = try JSONLLoader.load(from: url)
        #expect(examples.count == 240)

        let completions = examples.map(\.completion)
        #expect(Set(completions).count == examples.count)

        let part = try DemoCorpus.partition(examples, nightCount: 7, heldOutCount: 30)
        #expect(part.nights.count == 7)
        #expect(part.nights.allSatisfy { $0.count == 30 })
        #expect(part.heldOut.count == 30)
        let overlap = part.trainCompletions.intersection(part.heldOutCompletions)
        #expect(overlap.isEmpty, "held-out contaminated: \(overlap.count) shared completions")
    }

    @Test("Renna Vale seven-night fixture is multilingual (en/es/ru non-trivial)")
    func rennaSevenNightLanguageMix() throws {
        let url = fixtureURL("renna-vale-seven-nights.jsonl")
        let examples = try JSONLLoader.load(from: url)
        let part = try DemoCorpus.partition(examples, nightCount: 7, heldOutCount: 30)

        func language(of example: TrainingExample) -> String {
            let prompt = example.prompt.lowercased()
            if prompt.contains("in spanish") { return "es" }
            if prompt.contains("in russian") { return "ru" }
            if prompt.contains("in english") { return "en" }
            // Fallback on completion script when prompt omits the tag.
            let completion = example.completion
            if completion.unicodeScalars.contains(where: {
                (0x0400...0x04FF).contains($0.value)
            }) {
                return "ru"
            }
            return "en"
        }

        let trainLangs = part.nights.flatMap { $0.map(language(of:)) }
        let heldLangs = part.heldOut.map(language(of:))
        let allLangs = trainLangs + heldLangs

        func count(_ langs: [String], _ code: String) -> Int {
            langs.filter { $0 == code }.count
        }

        let total = allLangs.count
        #expect(total == 240)
        let en = count(allLangs, "en")
        let es = count(allLangs, "es")
        let ru = count(allLangs, "ru")
        // Non-trivial multilingual: each language ≥ 15% of the corpus.
        #expect(en >= 36, "english too scarce: \(en)/\(total)")
        #expect(es >= 36, "spanish too scarce: \(es)/\(total)")
        #expect(ru >= 36, "russian too scarce: \(ru)/\(total)")
        #expect(es + ru >= total / 3, "non-english share too small: es=\(es) ru=\(ru)")

        // Held-out and train both carry every language (code-switching claim).
        for code in ["en", "es", "ru"] {
            #expect(trainLangs.contains(code), "train missing \(code)")
            #expect(heldLangs.contains(code), "held-out missing \(code)")
        }
    }

    private func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // DemoCorpusTests.swift
            .deletingLastPathComponent() // AdaptCLITests
            .deletingLastPathComponent() // Tests
            .appendingPathComponent("Tools/adapt-cli/Fixtures/\(name)")
    }
}
