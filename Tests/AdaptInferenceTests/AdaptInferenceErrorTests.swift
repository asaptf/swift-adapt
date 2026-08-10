import AdaptInference
import Testing

@Suite("AdaptInferenceError")
struct AdaptInferenceErrorTests {
    @Test("error descriptions are non-empty and distinct")
    func descriptions() {
        let cases: [AdaptInferenceError] = [
            .invalidArgument("x"),
            .modelLoadFailed("x"),
            .integrityMismatch(expected: "a", actual: "b"),
            .adapterLoadFailed("x"),
            .generationFailed("x"),
            .fusedImmutable("x"),
            .promptFormatMismatch(adapter: .chatTemplate, session: .rawConcatenation),
        ]
        for error in cases {
            let d = error.errorDescription
            #expect(d != nil)
            #expect(!(d ?? "").isEmpty)
        }
        #expect(
            AdaptInferenceError.integrityMismatch(expected: "a", actual: "b")
                .errorDescription?
                .contains("a") == true
        )
    }
}
