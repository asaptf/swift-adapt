import AdaptCore
import AdaptTrain
import MLX
import Testing

@Suite("Chat template training mask")
struct ChatTemplateMaskingTests {
    @Test("templated collate zeros weights on prompt/scaffold targets only")
    func collateMasksScaffold() {
        TestSupport.prepareMLX()
        let tok = FakeTokenizer(hasChatTemplate: true)
        let ex = TrainingExample(
            prompt: "AB",
            completion: "CD",
            weight: 1.5,
            source: .synthetic
        )
        let t = PromptCompletionBatch.tokenize(
            ex,
            tokenizer: tok,
            maxLength: 128,
            convention: .chatTemplate
        )!
        #expect(t.convention == .chatTemplate)
        #expect(t.promptTokenCount > 0)
        #expect(t.tokens.count > t.promptTokenCount)

        let collated = PromptCompletionBatch.collate([t])!
        eval(collated.tokenWeights)
        let weights = collated.tokenWeights.asArray(Float.self)
        // For each target index col predicting tokens[col+1]:
        // weight is 0 when col+1 < promptTokenCount, else example weight.
        for col in 0..<weights.count {
            let predictedIndex = col + 1
            if predictedIndex >= t.promptTokenCount {
                #expect(weights[col] == 1.5)
            } else {
                #expect(weights[col] == 0)
            }
        }
    }

    @Test("train and generate prefixes are byte-identical via shared formatter")
    func trainGenerateInvariantThroughBatch() throws {
        let tok = FakeTokenizer(hasChatTemplate: true)
        let prompt = "Ship the crate."
        let completion = "Already on the thrusters."
        let ex = TrainingExample(prompt: prompt, completion: completion, source: .synthetic)

        let trained = PromptCompletionBatch.tokenize(
            ex,
            tokenizer: tok,
            maxLength: 256,
            convention: .chatTemplate
        )!
        let genPrefix = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: prompt,
            tokenizer: tok,
            convention: .chatTemplate
        )
        #expect(Array(trained.tokens.prefix(trained.promptTokenCount)) == genPrefix)
    }

    @Test("no-template tokenizer uses raw on both sides")
    func noTemplateRawBothSides() throws {
        let tok = FakeTokenizer(hasChatTemplate: false)
        #expect(SFTPromptFormatter.detectConvention(tokenizer: tok) == .rawConcatenation)

        let prompt = "Ping"
        let completion = "Pong"
        let ex = TrainingExample(prompt: prompt, completion: completion, source: .synthetic)
        let trained = PromptCompletionBatch.tokenize(
            ex,
            tokenizer: tok,
            maxLength: 64,
            convention: .rawConcatenation
        )!
        let gen = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: prompt,
            tokenizer: tok,
            convention: .rawConcatenation
        )
        #expect(Array(trained.tokens.prefix(trained.promptTokenCount)) == gen)
        #expect(trained.convention == .rawConcatenation)
    }
}
