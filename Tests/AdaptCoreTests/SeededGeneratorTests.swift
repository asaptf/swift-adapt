import AdaptCore
import Testing

@Suite("SeededGenerator")
struct SeededGeneratorTests {
    /// Hardcoded SplitMix64 sequence for seed 42 (first 5 outputs).
    private static let expectedSeed42: [UInt64] = [
        0xBDD732262FEB6E95,
        0x28EFE333B266F103,
        0x47526757130F9F52,
        0x581CE1FF0E4AE394,
        0x09BC585A244823F2,
    ]

    @Test("same seed produces identical sequence")
    func sameSeedIdentical() {
        var a = SeededGenerator(seed: 42)
        var b = SeededGenerator(seed: 42)
        let seqA = (0..<16).map { _ in a.next() }
        let seqB = (0..<16).map { _ in b.next() }
        #expect(seqA == seqB)
    }

    @Test("different seeds produce different sequences")
    func differentSeedsDiffer() {
        var a = SeededGenerator(seed: 1)
        var b = SeededGenerator(seed: 2)
        let seqA = (0..<8).map { _ in a.next() }
        let seqB = (0..<8).map { _ in b.next() }
        #expect(seqA != seqB)
    }

    @Test("sequence is stable across runs (hardcoded values)")
    func hardcodedSequence() {
        var g = SeededGenerator(seed: 42)
        let first = g.next()
        let second = g.next()
        let third = g.next()
        let fourth = g.next()
        let fifth = g.next()

        // Placeholder values — replaced after computing real SplitMix64 outputs.
        #expect(first == Self.expectedSeed42[0])
        #expect(second == Self.expectedSeed42[1])
        #expect(third == Self.expectedSeed42[2])
        #expect(fourth == Self.expectedSeed42[3])
        #expect(fifth == Self.expectedSeed42[4])
    }
}
