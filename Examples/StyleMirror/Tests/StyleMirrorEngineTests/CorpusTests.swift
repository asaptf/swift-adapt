import Foundation
import Testing
@testable import StyleMirrorEngine

@Suite("SampleCorpus")
struct CorpusTests {
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
}
