import AdaptCore
import Testing

@Suite("SignalSource default weights")
struct SignalSourceWeightTests {
    @Test("§4.2 weight table")
    func weightTable() {
        #expect(SignalSource.explicitEdit.defaultWeight == 1.0)
        #expect(SignalSource.acceptance.defaultWeight == 0.6)
        #expect(SignalSource.rejection.defaultWeight == 0.4)
        #expect(SignalSource.synthetic.defaultWeight == 0.3)
    }

    @Test("TrainingExample uses source default when weight omitted")
    func exampleDefaultWeight() {
        let gold = TrainingExample(prompt: "p", completion: "c", source: .explicitEdit)
        #expect(gold.weight == 1.0)

        let accept = TrainingExample(prompt: "p", completion: "c", source: .acceptance)
        #expect(accept.weight == 0.6)

        let reject = TrainingExample(prompt: "p", completion: "c", source: .rejection)
        #expect(reject.weight == 0.4)

        let synth = TrainingExample(prompt: "p", completion: "c", source: .synthetic)
        #expect(synth.weight == 0.3)
    }

    @Test("explicit weight overrides source default")
    func explicitOverride() {
        let ex = TrainingExample(
            prompt: "p",
            completion: "c",
            weight: 2.5,
            source: .synthetic
        )
        #expect(ex.weight == 2.5)
    }
}
