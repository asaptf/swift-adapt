import Foundation
import MLX
import MLXNN

/// Checkpointable AdamW with per-parameter-key first/second moments and step `t`.
///
/// ## Why this is not `MLXOptimizers.AdamW` (deviation from architecture §4.3)
///
/// Upstream `OptimizerBase.stateStorage` is `internal`. The only public peek,
/// `innerState() -> [MLXArray]`, is **read-only and unkeyed** — there is no
/// restore path. M1 acceptance requires a resumed run's loss to match an
/// uninterrupted run, so moments must survive interruption as first-class
/// serializable state. Do **not** "fix" this back to `MLXOptimizers` without
/// an upstream restore API.
///
/// Standard AdamW (Loshchilov & Hutter, ICLR 2019) with decoupled weight decay.
/// Defaults: β = (0.9, 0.999), ε = 1e-8, bias correction on.
public final class CheckpointableAdamW {
    /// Learning rate.
    public var learningRate: Float
    /// (β₁, β₂).
    public var betas: (Float, Float)
    /// Numerical stability ε.
    public var eps: Float
    /// Decoupled weight decay coefficient.
    public var weightDecay: Float
    /// Apply bias correction to m̂ / v̂.
    public var biasCorrection: Bool

    /// Optimizer step count `t` (shared across parameters, like common AdamW).
    public private(set) var step: Int

    /// Per-parameter first moment, keyed like `Module.trainableParameters().flattened()`.
    private var m: [String: MLXArray]
    /// Per-parameter second moment.
    private var v: [String: MLXArray]

    /// Creates an AdamW optimizer with empty moment state.
    public init(
        learningRate: Float,
        betas: (Float, Float) = (0.9, 0.999),
        eps: Float = 1e-8,
        weightDecay: Float = 0.01,
        biasCorrection: Bool = true,
        step: Int = 0,
        m: [String: MLXArray] = [:],
        v: [String: MLXArray] = [:]
    ) {
        self.learningRate = learningRate
        self.betas = betas
        self.eps = eps
        self.weightDecay = weightDecay
        self.biasCorrection = biasCorrection
        self.step = step
        self.m = m
        self.v = v
    }

    /// Applies one AdamW update to `model` using `gradients` (same nesting as trainable params).
    public func update(model: Module, gradients: ModuleParameters) {
        let flatGrads = Dictionary(uniqueKeysWithValues: gradients.flattened())
        let flatParams = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
        var updated: [String: MLXArray] = [:]

        step += 1
        let t = step
        let (b1, b2) = betas

        for (key, grad) in flatGrads {
            guard let parameter = flatParams[key] else { continue }

            var mKey = m[key] ?? MLXArray.zeros(like: parameter)
            var vKey = v[key] ?? MLXArray.zeros(like: parameter)

            // Decoupled weight decay on the parameter (AdamW).
            var param = parameter
            if weightDecay != 0 {
                param = param * (1 - learningRate * weightDecay)
            }

            mKey = b1 * mKey + (1 - b1) * grad
            vKey = b2 * vKey + (1 - b2) * square(grad)

            let update: MLXArray
            if biasCorrection {
                let c1 = learningRate / (1 - pow(b1, Float(t)))
                let c2 = 1 / (1 - pow(b2, Float(t)))
                // m̂ = m/(1-β1^t), v̂ = v/(1-β2^t); step = lr * m̂ / (√v̂ + ε)
                update = (c1 * mKey) / (sqrt(vKey * c2) + eps)
            } else {
                update = learningRate * mKey / (sqrt(vKey) + eps)
            }

            updated[key] = param - update
            m[key] = mKey
            v[key] = vKey
        }

        eval(updated)
        eval(m)
        eval(v)

        model.update(parameters: ModuleParameters.unflattened(updated))
    }

    /// Flat snapshot of moments for safetensors I/O.
    ///
    /// Keys: `"m.<paramKey>"`, `"v.<paramKey>"`. Step lives in `train_state.json`.
    public func exportMoments() -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in m {
            out["m.\(key)"] = value
        }
        for (key, value) in v {
            out["v.\(key)"] = value
        }
        return out
    }

    /// Restores moments from ``exportMoments()`` keys.
    public func importMoments(_ arrays: [String: MLXArray]) {
        m.removeAll(keepingCapacity: true)
        v.removeAll(keepingCapacity: true)
        for (key, value) in arrays {
            if key.hasPrefix("m.") {
                m[String(key.dropFirst(2))] = value
            } else if key.hasPrefix("v.") {
                v[String(key.dropFirst(2))] = value
            }
        }
    }

    /// Sets the optimizer step counter (used when restoring a checkpoint).
    public func restoreStep(_ t: Int) {
        step = t
    }

    /// Package/test access to first moments.
    package var momentM: [String: MLXArray] { m }
    /// Package/test access to second moments.
    package var momentV: [String: MLXArray] { v }
}
