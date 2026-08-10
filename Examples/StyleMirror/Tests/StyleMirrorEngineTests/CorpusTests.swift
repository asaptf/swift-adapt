import Foundation
import Testing
@testable import StyleMirrorEngine

@Suite("SampleCorpus")
struct CorpusTests {
    /// Word budget for blind-test candidates (`DESIGN.md` §4.3).
    private static let blindWordMin = 40
    private static let blindWordMax = 80
    /// Sibling length tolerance: max/min word counts must be ≤ this ratio (~±15%).
    private static let blindSiblingWordRatio = 1.15
    /// Code-switching base replies: two-sentence class (`DESIGN.md` §8.5).
    private static let codeSwitchBaseWordMax = 30

    @Test("corpus loads ~30 sent emails across en/es/ru")
    func corpusLoads() {
        let emails = SampleCorpus.sentEmails
        #expect(emails.count >= 28)
        #expect(emails.count <= 40)
        let langs = Set(emails.map(\.language))
        #expect(langs.contains("en"))
        #expect(langs.contains("es"))
        #expect(langs.contains("ru"))
        for email in emails {
            #expect(!email.body.isEmpty)
            #expect(!email.subject.isEmpty)
            #expect(!email.id.isEmpty)
        }
        // All ids unique.
        #expect(Set(emails.map(\.id)).count == emails.count)
    }

    @Test("every blind incoming has exactly three distinct candidates and one human")
    func blindFixturesShape() {
        #expect(SampleCorpus.blindRounds.count >= 4)
        for fixture in SampleCorpus.blindRounds {
            let bodies = [fixture.base, fixture.adapted, fixture.human]
            #expect(bodies.allSatisfy { !$0.isEmpty })
            #expect(Set(bodies).count == 3, "candidates must be pairwise distinct for \(fixture.incoming.id)")
            let roles = fixture.bodiesByRole.map(\.0)
            #expect(roles.filter { $0 == .human }.count == 1)
            #expect(roles.filter { $0 == .baseModel }.count == 1)
            #expect(roles.filter { $0 == .adaptedModel }.count == 1)
            // Base should look more formal / longer on average for EN fixtures.
            if fixture.incoming.language == "en" {
                #expect(fixture.base.count > fixture.human.count)
            }
        }
    }

    @Test("blind candidates share 40–80 word length class and stay within ±15%")
    func blindCandidateLengthFairness() {
        for fixture in SampleCorpus.blindRounds {
            let counts = [
                wordCount(fixture.base),
                wordCount(fixture.adapted),
                wordCount(fixture.human),
            ]
            for count in counts {
                #expect(
                    count >= Self.blindWordMin && count <= Self.blindWordMax,
                    "\(fixture.incoming.id): word count \(count) outside \(Self.blindWordMin)–\(Self.blindWordMax)"
                )
            }
            let minCount = counts.min()!
            let maxCount = counts.max()!
            let ratio = Double(maxCount) / Double(minCount)
            #expect(
                ratio <= Self.blindSiblingWordRatio + 1e-9,
                "\(fixture.incoming.id): sibling word ratio \(ratio) > \(Self.blindSiblingWordRatio) (\(counts))"
            )
        }
    }

    @Test("blind candidates apply sign-off convention uniformly (all stripped)")
    func blindCandidateSignOffUniformity() {
        for fixture in SampleCorpus.blindRounds {
            let flags = [
                containsSignOff(fixture.base),
                containsSignOff(fixture.adapted),
                containsSignOff(fixture.human),
            ]
            // Convention: signatures stripped on every candidate so presence alone
            // cannot identify the base model (DESIGN.md §4.3).
            #expect(
                flags.allSatisfy { $0 == false },
                "\(fixture.incoming.id): expected no sign-offs; flags=\(flags)"
            )
            // Guard the uniformity rule itself: if the convention ever flips to
            // "all signed", every candidate must still agree.
            #expect(Set(flags).count == 1, "\(fixture.incoming.id): mixed sign-off presence \(flags)")
        }
    }

    @Test("blind incoming bodies lead with substance, not a greeting line")
    func blindIncomingLeadsWithQuestion() {
        for fixture in SampleCorpus.blindRounds {
            let lines = fixture.incoming.body
                .split(whereSeparator: \.isNewline)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            #expect(lines.count >= 1, "\(fixture.incoming.id): empty body")
            let head = Array(lines.prefix(2))
            for line in head {
                #expect(
                    !isGreetingOnly(line),
                    "\(fixture.incoming.id): greeting-only line in first two: \(line)"
                )
            }
            // At least one of the first two lines should pose the question / ask.
            let headText = head.joined(separator: " ")
            #expect(
                headText.contains("?") || headText.contains("¿"),
                "\(fixture.incoming.id): first two lines should carry the question"
            )
        }
    }

    @Test("code-switching base replies stay in the two-sentence word budget")
    func codeSwitchBaseWordBudget() {
        let result = SampleCorpus.codeSwitch
        #expect(result.languages.count == 3)
        for pair in result.languages {
            let baseWords = wordCount(pair.baseReply)
            #expect(
                baseWords > 0 && baseWords <= Self.codeSwitchBaseWordMax,
                "\(pair.language): base reply \(baseWords) words exceeds \(Self.codeSwitchBaseWordMax)"
            )
            #expect(!containsSignOff(pair.baseReply), "\(pair.language): base reply has a multi-line sign-off")
            // Adapter stays personal and shorter-register; must differ from base.
            #expect(!pair.adaptedReply.isEmpty)
            #expect(pair.baseReply != pair.adaptedReply)
        }
    }

    @Test("trainingExamples maps one-to-one from sent emails")
    func trainingExamplesMap() {
        let examples = SampleCorpus.trainingExamples()
        #expect(examples.count == SampleCorpus.sentEmails.count)
        #expect(examples.allSatisfy { $0.source == .synthetic })
        #expect(Set(examples.map(\.id)).count == examples.count)
    }

    @Test("poisoned completions are non-empty and obviously synthetic")
    func poisoned() {
        #expect(SampleCorpus.poisonedCompletions.count >= 3)
        #expect(SampleCorpus.poisonedCompletions.allSatisfy { !$0.isEmpty })
        #expect(SampleCorpus.poisonedCompletions.allSatisfy { $0 == $0.uppercased() })
    }

    // MARK: - Helpers

    private func wordCount(_ text: String) -> Int {
        text.split { $0.isWhitespace || $0.isNewline }.count
    }

    /// Detects closing signature blocks / sign-offs used by the corpus.
    private func containsSignOff(_ text: String) -> Bool {
        let lowered = text.lowercased()
        let phrases = [
            "— renna",
            "-- renna",
            "best regards",
            "warm regards",
            "saludos cordiales",
            "с уважением",
            "renna vale",
            "product lead, harborfinch",
        ]
        if phrases.contains(where: { lowered.contains($0) }) {
            return true
        }
        // Lone closing initial used in internal sent mail.
        let trimmedLines = text
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        if let last = trimmedLines.last?.lowercased(), last == "r" || last == "— r" {
            return true
        }
        return false
    }

    /// True when a line is only a greeting / salutation with no substance.
    private func isGreetingOnly(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        let lowered = trimmed.lowercased()
        // Bare name address.
        if lowered == "renna" || lowered == "renna," || lowered == "renna!" {
            return true
        }
        let greetingPrefixes = [
            "hi ", "hi,", "hello", "hey ", "hey,",
            "hola", "dear ", "estimado", "estimada",
            "привет", "здравствуйте", "hi team",
        ]
        // Greeting-only if the whole line is a short salutation (no question, little body).
        let wordCount = trimmed.split { $0.isWhitespace }.count
        let looksLikeGreeting = greetingPrefixes.contains { prefix in
            lowered == prefix.trimmingCharacters(in: .whitespacesAndNewlines)
                || lowered.hasPrefix(prefix)
        }
        if looksLikeGreeting && wordCount <= 4 && !trimmed.contains("?") && !trimmed.contains("¿") {
            return true
        }
        return false
    }
}
