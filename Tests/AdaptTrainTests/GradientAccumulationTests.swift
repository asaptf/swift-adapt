import AdaptTrain
import MLX
import MLXNN
import Testing

@Suite("Gradient accumulation")
struct GradientAccumulationTests {
    /// Accumulating k micro-batches of size b matches one full batch of size k*b
    /// (equal microbatch sizes, mean loss, grads scaled by 1/k).
    @Test("k micro-batches match one full batch within tolerance")
    func accumulationMatchesFullBatch() {
        TestSupport.prepareMLX()
        let k = 4
        let micro = 2
        let full = k * micro

        let pairs = TestSupport.syntheticPairs(count: full)

        func oneStep(model: TinyTrainable, batchIndices: [Int], scale: Float) -> ModuleParameters {
            let arrays = TestSupport.collatePairs(indices: batchIndices, pairs: pairs)!
            let vg = valueAndGrad(model: model) { (m: Module, arrs: [MLXArray]) -> [MLXArray] in
                let (l, c) = TestSupport.mseLoss(model: m, arrays: arrs)
                return [l, c]
            }
            let (_, grads) = vg(model, arrays)
            if scale == 1 { return grads }
            var scaled: [String: MLXArray] = [:]
            for (key, g) in grads.flattened() {
                scaled[key] = g * scale
            }
            return ModuleParameters.unflattened(scaled)
        }

        let modelFull = TinyTrainable(seed: 1)
        eval(modelFull)
        let fullGrads = oneStep(model: modelFull, batchIndices: Array(0..<full), scale: 1)

        let modelAccum = TinyTrainable(seed: 1)
        eval(modelAccum)
        var accum: [String: MLXArray] = [:]
        for m in 0..<k {
            let start = m * micro
            let idxs = Array(start..<(start + micro))
            let g = oneStep(model: modelAccum, batchIndices: idxs, scale: 1.0 / Float(k))
            for (key, arr) in g.flattened() {
                if let e = accum[key] {
                    accum[key] = e + arr
                } else {
                    accum[key] = arr
                }
            }
        }
        eval(Array(accum.values))

        let fullFlat = Dictionary(uniqueKeysWithValues: fullGrads.flattened())
        for (key, gFull) in fullFlat {
            let gAcc = accum[key]!
            eval(gFull, gAcc)
            let a = gFull.asArray(Float.self)
            let b = gAcc.asArray(Float.self)
            #expect(a.count == b.count)
            for (x, y) in zip(a, b) {
                #expect(abs(x - y) < 1e-4)
            }
        }
    }
}
