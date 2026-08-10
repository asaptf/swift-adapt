import AdaptCore
import AdaptEval
import Foundation
import Testing

@Suite("PromotionEvaluator")
struct PromotionEvaluatorTests {

    private func pool(count: Int) -> [TrainingExample] {
        (0..<count).map { i in
            TrainingExample(
                id: UUID(uuidString: String(format: "00000000-0000-4000-8000-%012d", i))!,
                prompt: "p\(i)",
                completion: "c\(i)",
                capturedAt: Date(timeIntervalSince1970: Double(1_700_000_000 + i)),
                source: SignalSource.allCases[i % SignalSource.allCases.count]
            )
        }
    }

    /// Scorer: primary = constant + small id-derived term so pairs are stable.
    private func constantScorer(mean: Double) -> ClosurePerExampleScorer {
        ClosurePerExampleScorer { examples in
            examples.map { ex in
                ExampleScore(
                    exampleID: ex.id,
                    primary: mean,
                    supervisedTokenCount: 8
                )
            }
        }
    }

    @Test("end-to-end promote path with fake scorers")
    func e2ePromote() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("adapt-eval-e2e-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let examples = pool(count: 200)
        let evaluator = PromotionEvaluator(
            policy: PromotionPolicy(minHeldOut: 30, alpha: 0.05),
            seed: 42
        )
        let result = try await evaluator.evaluate(
            lineageID: String(repeating: "f", count: 64),
            lineageDirectory: dir,
            source: ArrayHeldOutSource(examples),
            candidateScorer: constantScorer(mean: 1.0),
            incumbentScorer: constantScorer(mean: 2.0)
        )
        guard case .decided(let decision, let report) = result else {
            Issue.record("expected decided, got \(result)")
            return
        }
        #expect(decision.shouldPromote)
        #expect(report.gateDecision == .promote)
        #expect(report.passedGate == true)
        // Pin file must exist.
        #expect(try HeldOutPinStore.load(from: dir) != nil)
    }

    @Test("end-to-end refuse corrupted candidate")
    func e2eRefuseCorrupted() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("adapt-eval-corr-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let examples = pool(count: 200)
        let evaluator = PromotionEvaluator(
            policy: PromotionPolicy(minHeldOut: 30),
            seed: 1
        )
        let result = try await evaluator.evaluate(
            lineageID: String(repeating: "0", count: 64),
            lineageDirectory: dir,
            source: ArrayHeldOutSource(examples),
            candidateScorer: constantScorer(mean: 9.0),
            incumbentScorer: constantScorer(mean: 1.0)
        )
        guard case .decided(let decision, let report) = result else {
            Issue.record("expected decided, got \(result)")
            return
        }
        #expect(!decision.shouldPromote)
        #expect(decision.feedsBackoff)
        #expect(report.gateDecision == .refuse)
        #expect(report.passedGate == false)
    }

    @Test("broken pin surfaces as pinBroken not refuse")
    func brokenPinNotRefuse() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("adapt-eval-brk-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let full = pool(count: 200)
        let policy = PromotionPolicy(minHeldOut: 30)
        let lineageID = String(repeating: "1", count: 64)
        // Create pin against full pool.
        _ = try HeldOutPinStore.loadOrCreate(
            lineageDirectory: dir,
            lineageID: lineageID,
            pool: full,
            policy: policy,
            seed: 5
        )
        // Evaluate with a source that lost most examples.
        let evaluator = PromotionEvaluator(policy: policy, seed: 5)
        let result = try await evaluator.evaluate(
            lineageID: lineageID,
            lineageDirectory: dir,
            source: ArrayHeldOutSource(Array(full.prefix(5))),
            candidateScorer: constantScorer(mean: 1.0),
            incumbentScorer: constantScorer(mean: 2.0)
        )
        guard case .pinBroken(let missing, let report) = result else {
            Issue.record("expected pinBroken, got \(result)")
            return
        }
        #expect(!missing.isEmpty)
        #expect(report.missingPinnedExampleCount == missing.count)
        #expect(!result.shouldPromote)
        #expect(report.gateDecision == nil)
    }
}
