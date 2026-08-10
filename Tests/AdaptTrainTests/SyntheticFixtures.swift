import AdaptCore
import AdaptRegistry
import AdaptTrain
import Foundation
import MLX
import MLXLMCommon
import MLXNN
// SFTTokenizing / SFTFormattingError live in AdaptCore (already imported).

/// Tiny fully-trainable module for model-free tests (LoRA-shaped: two matrices).
final class TinyTrainable: Module {
    @ParameterInfo(key: "lora_a") var loraA: MLXArray
    @ParameterInfo(key: "lora_b") var loraB: MLXArray

    init(seed: UInt64 = 0) {
        // Deterministic fixed init (not MLXRandom) so tests are reproducible.
        self._loraA.wrappedValue = MLXArray(
            [Float](repeating: 0, count: 8).enumerated().map { i, _ in
                Float(i + 1) * 0.01 + Float(seed % 7) * 0.001
            },
            [4, 2]
        )
        self._loraB.wrappedValue = MLXArray(
            [Float](repeating: 0, count: 8).enumerated().map { i, _ in
                Float(i + 1) * 0.02
            },
            [2, 4]
        )
        super.init()
    }

    /// y = x @ A @ B  (batch, 4) → (batch, 4)
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(matmul(x, loraA), loraB)
    }
}

/// Stub tokenizer: maps each Character to an id.
///
/// Synthetic chat template markers (when `hasChatTemplate` is true):
/// - `200` = bos/template start
/// - `201` = user-turn open
/// - `202` = assistant-turn open
/// - `203` = turn close
///
/// Generation prefix = template(user, addGenerationPrompt: true) is a true
/// prefix of full = template(user+assistant, addGenerationPrompt: false).
struct FakeTokenizer: Tokenizer, SFTTokenizing, Sendable {
    /// When false, `applyChatTemplate` throws `missingChatTemplate`.
    var hasChatTemplate: Bool

    init(hasChatTemplate: Bool = false) {
        self.hasChatTemplate = hasChatTemplate
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        if addSpecialTokens { ids.append(1) }
        for scalar in text.unicodeScalars {
            ids.append(Int(scalar.value % 200) + 10)
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds
            .filter { !skipSpecialTokens || ($0 != 1 && $0 < 200) }
            .compactMap { id -> String? in
                guard id >= 10 else { return nil }
                return UnicodeScalar(id - 10).map(String.init)
            }
            .joined()
    }

    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { "<s>" }
    var eosToken: String? { "</s>" }
    var unknownToken: String? { "<unk>" }

    // MARK: SFTTokenizing

    func applyChatTemplate(
        messages: [[String: String]],
        addGenerationPrompt: Bool
    ) throws -> [Int] {
        guard hasChatTemplate else { throw SFTFormattingError.missingChatTemplate }
        var ids: [Int] = [200]
        for message in messages {
            let role = message["role"] ?? ""
            let content = message["content"] ?? ""
            if role == "user" {
                ids.append(201)
            } else if role == "assistant" {
                ids.append(202)
            } else {
                ids.append(204)
            }
            ids.append(contentsOf: encode(text: content, addSpecialTokens: false))
            ids.append(203)
        }
        if addGenerationPrompt {
            ids.append(202)
        }
        return ids
    }

    // MARK: MLXLMCommon.Tokenizer

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        guard hasChatTemplate else { throw TokenizerError.missingChatTemplate }
        var addGen = true
        if let last = messages.last, let role = last["role"] as? String, role == "assistant" {
            addGen = false
        }
        if let explicit = additionalContext?["add_generation_prompt"] as? Bool {
            addGen = explicit
        }
        let stringMessages: [[String: String]] = messages.map { dict in
            var out: [String: String] = [:]
            for (k, v) in dict {
                if let s = v as? String { out[k] = s }
            }
            return out
        }
        // Bridge SFTFormattingError → TokenizerError for the MLX path.
        do {
            return try applyChatTemplate(messages: stringMessages, addGenerationPrompt: addGen)
        } catch SFTFormattingError.missingChatTemplate {
            throw TokenizerError.missingChatTemplate
        }
    }
}

/// Sendable float pairs so micro-batch closures can be built without capturing MLXArray.
struct FloatPair: Sendable {
    let x: [Float]
    let y: [Float]
}

enum TestSupport {
    /// Call before any MLX evaluation so the metallib is discoverable.
    static func prepareMLX() {
        MetalBootstrap.ensureMetallib()
    }

    static func makeRegistry() throws -> (AdapterRegistry, URL) {
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("AdaptTrainTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        return (try AdapterRegistry(rootURL: temp), temp)
    }

    static func teardown(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    static var lineage: AdapterLineage {
        AdapterLineage(
            taskID: "train-test",
            baseModelID: "synthetic/tiny",
            loraConfig: LoRAConfig(rank: 2, scale: 1.0, numLayers: 1)
        )
    }

    static func syntheticPairs(count: Int = 16) -> [FloatPair] {
        (0..<count).map { i in
            let base = Float(i) * 0.1
            return FloatPair(
                x: [base, base + 0.1, base + 0.2, base + 0.3],
                y: [base * 0.5, base * 0.5 + 0.1, base * 0.5 + 0.2, base * 0.5 + 0.3]
            )
        }
    }

    static func mseLoss(model: Module, arrays: [MLXArray]) -> (MLXArray, MLXArray) {
        let tiny = model as! TinyTrainable
        let x = arrays[0]
        let y = arrays[1]
        let pred = tiny(x)
        let loss = mean(square(pred - y))
        let count = MLXArray(Float(x.dim(0)))
        return (loss, count)
    }

    static func collatePairs(indices: [Int], pairs: [FloatPair]) -> [MLXArray]? {
        guard !indices.isEmpty else { return nil }
        var xData: [Float] = []
        var yData: [Float] = []
        for i in indices {
            xData.append(contentsOf: pairs[i].x)
            yData.append(contentsOf: pairs[i].y)
        }
        let n = indices.count
        return [
            MLXArray(xData, [n, 4]),
            MLXArray(yData, [n, 4]),
        ]
    }

    static func dummyExamples(count: Int) -> [TrainingExample] {
        (0..<count).map { i in
            TrainingExample(
                prompt: "p\(i)",
                completion: "c\(i)",
                source: .synthetic
            )
        }
    }

    static func makeLoss() -> SendingLoss {
        SendingLoss(mseLoss)
    }

    static func makeMicrobatch(pairs: [FloatPair]) -> SendingMicrobatch {
        SendingMicrobatch { indices in
            collatePairs(indices: indices, pairs: pairs)
        }
    }

    static func snapshotParams(_ model: TinyTrainable) -> [String: [Float]] {
        var out: [String: [Float]] = [:]
        for (key, arr) in model.trainableParameters().flattened() {
            eval(arr)
            out[key] = arr.asArray(Float.self)
        }
        return out
    }

    static func paramClose(
        _ a: [String: [Float]],
        _ b: [String: [Float]],
        tol: Float
    ) -> Bool {
        guard a.keys.sorted() == b.keys.sorted() else { return false }
        for key in a.keys {
            let va = a[key]!
            let vb = b[key]!
            guard va.count == vb.count else { return false }
            for (x, y) in zip(va, vb) where abs(x - y) > tol {
                return false
            }
        }
        return true
    }
}
