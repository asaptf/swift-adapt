import AdaptCore
import AdaptData
import Foundation
import Testing

@Suite("PII never lands in DB bytes")
struct PIIByteScanTests {

    @Test("raw PII strings are absent from the database file after capture")
    func byteScanNoRawPII() async throws {
        let (buffer, root) = try AdaptDataTestSupport.makeBuffer(
            configuration: ReplayBufferConfiguration(
                maxExamples: 50,
                maxCapturesPerDay: 50
            )
        )
        defer { AdaptDataTestSupport.teardown(root) }

        // Distinct, high-entropy strings that must not appear as raw bytes.
        let email = "secret.user.\(UUID().uuidString.prefix(8))@pii-test.example"
        let card = "4111111111111111"
        let phone = "415-555-2671"
        let ibanCompact = "GB82WEST12345698765432"
        let ibanSpaced = "GB82 WEST 1234 5698 7654 32"

        #expect(CreditCardScrubber.luhnValid(card))
        #expect(IBANScrubber.mod97Valid(ibanCompact))

        _ = try await buffer.add(
            AdaptDataTestSupport.example(
                prompt: "Write to \(email) about payment",
                completion: "Use card \(card) or IBAN \(ibanSpaced); call \(phone)."
            )
        )

        try await buffer.prepareForByteScan()
        let dbURL = buffer.databaseURL
        #expect(FileManager.default.fileExists(atPath: dbURL.path))

        let needles = [email, card, phone, ibanCompact, "WEST1234", "555-2671"]
        for needle in needles {
            let present = try AdaptDataTestSupport.fileContainsASCII(needle, fileURL: dbURL)
            #expect(!present, "raw PII still in DB file: \(needle)")
        }

        // Scrub tokens should be present (proves write path ran scrubbers).
        let tokens = ["[EMAIL]", "[CARD]", "[PHONE]", "[IBAN]"]
        for token in tokens {
            let present = try AdaptDataTestSupport.fileContainsASCII(token, fileURL: dbURL)
            #expect(present, "expected scrub token \(token) in DB")
        }

        let examples = try await buffer.examples()
        #expect(examples.count == 1)
        #expect(!examples[0].prompt.contains(email))
        #expect(!examples[0].completion.contains(card))
    }
}
