import AdaptCLI
import Testing

@Suite("GenerateComparison")
struct GenerateComparisonTests {

    @Test("identical outputs note lack of adapter effect")
    func identical() {
        let text = GenerateComparison.format(base: "same", adapted: "same")
        #expect(text.contains("identical"))
    }

    @Test("different outputs report visible effect")
    func different() {
        let text = GenerateComparison.format(
            base: "Hello, how can I help?",
            adapted: "Nix here—hold the thrusters. —Nix / Belt lane 4"
        )
        #expect(text.contains("differ"))
        #expect(text.contains("base chars="))
    }
}
