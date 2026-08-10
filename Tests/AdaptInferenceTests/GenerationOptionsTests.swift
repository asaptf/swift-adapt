import AdaptInference
import MLXLMCommon
import Testing

@Suite("GenerationOptions")
struct GenerationOptionsTests {

    // MARK: - Neutral defaults → unchanged mlx mapping

    @Test("defaults map to topP=1.0 and no repetition processor (nil penalty)")
    func neutralDefaultsMapUnchanged() throws {
        let options = GenerationOptions()
        #expect(options.maxTokens == 128)
        #expect(options.temperature == 0)
        #expect(options.seed == nil)
        #expect(options.topP == 1.0)
        #expect(options.repetitionPenalty == 1.0)
        #expect(options.repetitionContextSize == 20)

        let params = try options.asGenerateParameters()
        #expect(params.maxTokens == 128)
        #expect(params.temperature == 0)
        #expect(params.seed == nil)
        #expect(params.topP == 1.0)
        // Identity penalty must omit the processor path, matching historical
        // GenerateParameters(maxTokens:temperature:seed:) behaviour.
        #expect(params.repetitionPenalty == nil)
        #expect(params.repetitionContextSize == 20)
        #expect(params.processor() == nil)
    }

    @Test("explicit knobs map onto GenerateParameters fields")
    func explicitKnobsMap() throws {
        let options = GenerationOptions(
            maxTokens: 64,
            temperature: 0.7,
            seed: 42,
            topP: 0.9,
            repetitionPenalty: 1.15,
            repetitionContextSize: 32
        )
        let params = try options.asGenerateParameters()
        #expect(params.maxTokens == 64)
        #expect(params.temperature == 0.7)
        #expect(params.seed == 42)
        #expect(params.topP == 0.9)
        #expect(params.repetitionPenalty == 1.15)
        #expect(params.repetitionContextSize == 32)
        #expect(params.processor() != nil)
    }

    // MARK: - Validation

    @Test("topP outside (0, 1] throws invalidArgument")
    func topPOutOfRange() {
        let cases: [Float] = [0, -0.1, 1.01, .nan, .infinity, -.infinity]
        for value in cases {
            let options = GenerationOptions(topP: value)
            #expect(throws: AdaptInferenceError.self) {
                try options.validate()
            }
            #expect(throws: AdaptInferenceError.self) {
                try options.asGenerateParameters()
            }
        }
    }

    @Test("topP boundary 1.0 and near-zero positive are accepted")
    func topPBoundariesAccepted() throws {
        try GenerationOptions(topP: 1.0).validate()
        try GenerationOptions(topP: 0.0001).validate()
        let params = try GenerationOptions(topP: 0.95).asGenerateParameters()
        #expect(params.topP == 0.95)
    }

    @Test("repetitionPenalty below 1.0 throws invalidArgument")
    func repetitionPenaltyBelowOne() {
        let cases: [Float] = [0.99, 0, -1, .nan, .infinity, -.infinity]
        for value in cases {
            let options = GenerationOptions(repetitionPenalty: value)
            do {
                try options.validate()
                Issue.record("expected validate() to throw for penalty \(value)")
            } catch let error as AdaptInferenceError {
                guard case .invalidArgument(let message) = error else {
                    Issue.record("wrong error case: \(error)")
                    continue
                }
                #expect(message.contains("repetitionPenalty"))
            } catch {
                Issue.record("unexpected error type: \(error)")
            }
        }
    }

    @Test("repetitionPenalty 1.0 and above is accepted")
    func repetitionPenaltyAccepted() throws {
        try GenerationOptions(repetitionPenalty: 1.0).validate()
        try GenerationOptions(repetitionPenalty: 1.15).validate()
        try GenerationOptions(repetitionPenalty: 2.0).validate()
    }

    @Test("repetitionContextSize below 1 throws invalidArgument")
    func repetitionContextSizeInvalid() {
        for size in [0, -1, -100] {
            let options = GenerationOptions(repetitionContextSize: size)
            #expect(throws: AdaptInferenceError.self) {
                try options.validate()
            }
        }
    }

    @Test("session generate fails fast on invalid options without calling the backend")
    func sessionRejectsInvalidOptions() async throws {
        let (reg, root) = try InferenceTestSupport.makeRegistry()
        defer { InferenceTestSupport.teardown(root) }

        let fake = FakeSessionBackend(baseChunks: ["should-not-run"])
        let session = try await AdaptSession(
            backend: fake,
            lineage: InferenceTestSupport.lineage,
            registry: reg,
            skipInitialLoad: true
        )

        let bad = GenerationOptions(topP: 0, repetitionPenalty: 0.5)
        do {
            _ = try await session.generateText(prompt: "hi", options: bad)
            Issue.record("expected generateText to throw")
        } catch let error as AdaptInferenceError {
            if case .invalidArgument = error {
                // expected
            } else {
                Issue.record("expected invalidArgument, got \(error)")
            }
        } catch {
            Issue.record("unexpected error: \(error)")
        }
        #expect(fake.generateCount == 0)
    }
}
