import Foundation

/// Seedable SplitMix64 generator for reproducible shuffling and sampling.
///
/// Upstream MLX batch iterators use the unseeded global RNG; Adapt owns
/// deterministic order via this type so training tests can hardcode sequences.
public struct SeededGenerator: RandomNumberGenerator, Sendable {
    private var state: UInt64

    /// Creates a generator from a 64-bit seed.
    public init(seed: UInt64) {
        self.state = seed
    }

    /// Advances state and returns the next 64-bit value (SplitMix64).
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z &>> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z &>> 27)) &* 0x94D049BB133111EB
        return z ^ (z &>> 31)
    }
}
