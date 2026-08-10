import AdaptCLI
import AdaptCore
import Foundation
import Testing

@Suite("JSONLLoader")
struct JSONLLoaderTests {

    @Test("parses required prompt and completion")
    func parsesRequiredFields() throws {
        let text = """
            {"prompt":"Hello","completion":"World"}
            """
        let examples = try JSONLLoader.parse(text: text)
        #expect(examples.count == 1)
        #expect(examples[0].prompt == "Hello")
        #expect(examples[0].completion == "World")
        #expect(examples[0].source == .synthetic)
        #expect(examples[0].weight == SignalSource.synthetic.defaultWeight)
    }

    @Test("applies optional source and weight")
    func optionalSourceAndWeight() throws {
        let text = """
            {"prompt":"p","completion":"c","source":"explicitEdit","weight":0.9}
            """
        let examples = try JSONLLoader.parse(text: text)
        #expect(examples[0].source == .explicitEdit)
        #expect(examples[0].weight == 0.9)
    }

    @Test("defaults weight from source when weight omitted")
    func defaultWeightFromSource() throws {
        let text = """
            {"prompt":"p","completion":"c","source":"acceptance"}
            """
        let examples = try JSONLLoader.parse(text: text)
        #expect(examples[0].weight == SignalSource.acceptance.defaultWeight)
    }

    @Test("skips blank lines and hash comments")
    func skipsBlankAndComments() throws {
        let text = """

            # comment
            {"prompt":"a","completion":"b"}

            {"prompt":"c","completion":"d"}
            """
        let examples = try JSONLLoader.parse(text: text)
        #expect(examples.count == 2)
    }

    @Test("reports line number on invalid JSON")
    func malformedJSONLineNumber() {
        let text = """
            {"prompt":"ok","completion":"ok"}
            {not json}
            {"prompt":"x","completion":"y"}
            """
        do {
            _ = try JSONLLoader.parse(text: text)
            Issue.record("expected throw")
        } catch let error as AdaptCLIError {
            guard case .malformedJSONL(let line, let detail) = error else {
                Issue.record("wrong case \(error)")
                return
            }
            #expect(line == 2)
            #expect(!detail.isEmpty)
            #expect(error.localizedDescription.contains("line 2"))
        } catch {
            Issue.record("unexpected \(error)")
        }
    }

    @Test("reports missing prompt key with line number")
    func missingPromptKey() {
        let text = """
            {"completion":"only completion"}
            """
        do {
            _ = try JSONLLoader.parse(text: text)
            Issue.record("expected throw")
        } catch let error as AdaptCLIError {
            guard case .malformedJSONL(let line, let detail) = error else {
                Issue.record("wrong case \(error)")
                return
            }
            #expect(line == 1)
            #expect(detail.contains("prompt"))
        } catch {
            Issue.record("unexpected \(error)")
        }
    }

    @Test("rejects empty prompt")
    func emptyPrompt() {
        let text = """
            {"prompt":"","completion":"c"}
            """
        do {
            _ = try JSONLLoader.parse(text: text)
            Issue.record("expected throw")
        } catch let error as AdaptCLIError {
            guard case .malformedJSONL(let line, _) = error else {
                Issue.record("wrong case \(error)")
                return
            }
            #expect(line == 1)
        } catch {
            Issue.record("unexpected \(error)")
        }
    }

    @Test("fixture corpus has ~50 valid examples")
    func fixtureCorpus() throws {
        // Locate package root from this source file.
        let thisFile = URL(fileURLWithPath: #filePath)
        let packageRoot = thisFile
            .deletingLastPathComponent() // AdaptCLITests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // package root
        let fixture = packageRoot
            .appendingPathComponent("Tools/adapt-cli/Fixtures/nix-caldera-style.jsonl")
        #expect(FileManager.default.fileExists(atPath: fixture.path))
        let examples = try JSONLLoader.load(from: fixture)
        #expect(examples.count >= 45)
        #expect(examples.count <= 60)
        // Distinctive voice markers should appear in most completions.
        let withSignoff = examples.filter { $0.completion.contains("—Nix / Belt lane 4") }
        #expect(withSignoff.count == examples.count)
        let withOpen = examples.filter { $0.completion.hasPrefix("Nix here—") }
        #expect(withOpen.count == examples.count)
    }
}
