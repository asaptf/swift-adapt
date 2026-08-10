import AdaptCore
import AdaptTrain
import MLX
import Testing

@Suite("Prompt/completion masking & weights")
struct PromptWeightingTests {
    @Test("tokenize concatenates prompt then completion")
    func tokenizeOrder() {
        let tok = FakeTokenizer()
        let ex = TrainingExample(
            prompt: "AB",
            completion: "CD",
            weight: 0.6,
            source: .acceptance
        )
        let t = PromptCompletionBatch.tokenize(ex, tokenizer: tok, maxLength: 64)!
        // BOS + A + B + C + D
        #expect(t.promptTokenCount == 3) // BOS + A + B
        #expect(t.tokens.count == 5)
        #expect(t.weight == 0.6)
    }

    @Test("collate masks prompt targets and applies example weight")
    func collateMask() {
        TestSupport.prepareMLX()
        let tok = FakeTokenizer()
        let ex = TrainingExample(prompt: "A", completion: "B", weight: 2.0, source: .explicitEdit)
        let t = PromptCompletionBatch.tokenize(ex, tokenizer: tok, maxLength: 64)!
        let collated = PromptCompletionBatch.collate([t])!
        eval(collated.tokenWeights)
        let w = collated.tokenWeights.asArray(Float.self)
        // tokens: [BOS, A, B] → inputs len 2, targets predict A then B
        // predict A (index 1) is still prompt → weight 0
        // predict B (index 2) is completion → weight 2.0
        #expect(w.count >= 2)
        #expect(w[0] == 0)
        #expect(w[1] == 2.0)
    }
}
