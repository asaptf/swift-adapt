import AdaptTrain
import MLX
import MLXNN
import Testing

@Suite("CheckpointableAdamW")
struct AdamWTests {
    /// Hand-computed AdamW for a scalar parameter over two steps.
    ///
    /// Setup: θ₀ = 1.0, g₀ = 0.5, g₁ = 0.25
    /// lr = 0.1, β = (0.9, 0.999), ε = 1e-8, wd = 0, biasCorrection = true
    ///
    /// Step 1 (t=1):
    ///   m = 0.1 * 0.5 = 0.05
    ///   v = 0.001 * 0.25 = 0.00025
    ///   m̂ = 0.05 / (1-0.9) = 0.5
    ///   v̂ = 0.00025 / (1-0.999) = 0.25
    ///   update = 0.1 * 0.5 / (√0.25 + ε) ≈ 0.1
    ///   θ₁ = 1.0 - 0.1 = 0.9
    @Test("matches hand-computed scalar updates")
    func handComputed() {
        TestSupport.prepareMLX()
        final class ScalarModel: Module {
            @ParameterInfo(key: "w") var w: MLXArray
            init(value: Float) {
                self._w.wrappedValue = MLXArray(value)
                super.init()
            }
        }

        let model = ScalarModel(value: 1.0)
        eval(model)
        let opt = CheckpointableAdamW(
            learningRate: 0.1,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weightDecay: 0,
            biasCorrection: true
        )

        let g1 = ModuleParameters.unflattened(["w": MLXArray(Float(0.5))])
        opt.update(model: model, gradients: g1)
        eval(model)
        let w1: Float = model.w.item()
        #expect(abs(w1 - 0.9) < 1e-5)

        let g2 = ModuleParameters.unflattened(["w": MLXArray(Float(0.25))])
        opt.update(model: model, gradients: g2)
        eval(model)
        let w2: Float = model.w.item()

        // Step 2 hand computation → θ ≈ 0.806782
        #expect(abs(w2 - 0.806782) < 1e-4)
        #expect(opt.step == 2)
    }

    @Test("export/import moments round-trips")
    func momentRoundTrip() {
        TestSupport.prepareMLX()
        final class ScalarModel: Module {
            @ParameterInfo(key: "w") var w: MLXArray
            init(value: Float) {
                self._w.wrappedValue = MLXArray(value)
                super.init()
            }
        }
        let model = ScalarModel(value: 2)
        let opt = CheckpointableAdamW(learningRate: 1e-2, weightDecay: 0)
        let g = ModuleParameters.unflattened(["w": MLXArray(Float(1))])
        opt.update(model: model, gradients: g)
        eval(model)

        let exported = opt.exportMoments()
        let opt2 = CheckpointableAdamW(learningRate: 1e-2, weightDecay: 0)
        opt2.importMoments(exported)
        opt2.restoreStep(opt.step)

        #expect(opt2.step == 1)
        let m0: Float = opt.momentM["w"]!.item()
        let m1: Float = opt2.momentM["w"]!.item()
        #expect(abs(m0 - m1) < 1e-7)
    }
}
