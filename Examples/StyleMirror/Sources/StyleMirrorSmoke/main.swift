import AdaptCore
import Foundation
import StyleMirrorEngine

/// Real-path acceptance harness for ``AdaptEngine``.
///
/// Not the UI. Exercises train → versions → blind test → code-switch → poison
/// against the seeded demo registry and prints character counts so length-class
/// fairness can be judged honestly.
///
/// ```
/// cd Examples/StyleMirror
/// swift run -c release StyleMirrorSmoke
/// # optional flags:
/// #   --steps 20
/// #   --skip-train
/// #   --skip-poison
/// ```

@main
struct StyleMirrorSmoke {
    static func main() async {
        let args = Array(CommandLine.arguments.dropFirst())
        let steps = intFlag("--steps", in: args) ?? 20
        let skipTrain = args.contains("--skip-train")
        let skipPoison = args.contains("--skip-poison")
        let skipBlind = args.contains("--skip-blind")
        let skipCode = args.contains("--skip-code")

        print("=== StyleMirrorSmoke (AdaptEngine) ===")
        do {
            let config = try AdaptEngineConfiguration.seededDemo()
            print("  model:    \(config.modelID)")
            print("  registry: \(config.registryRoot.path)")
            print("  held-out: \(config.heldOutJSONL?.path ?? "(none)")")
            print("  steps:    \(steps)")

            let engine = try AdaptEngine(configuration: config, seed: 42)

            let versions = await engine.adapterVersions()
            let active = await engine.activeVersion()
            print("")
            print("--- Version history ---")
            print("  versions: \(versions.map(\.version))")
            if let active {
                let ce = active.evalReport?.primaryScore.map { String(format: "%.4f", $0) } ?? "n/a"
                print("  active:   v\(active.version)  held-out CE=\(ce)")
            } else {
                print("  active:   (none)")
            }

            if !skipTrain {
                print("")
                print("--- Act 2: train (\(steps) steps) ---")
                let examples = SampleCorpus.trainingExamples()
                let trainConfig = TrainingConfiguration(
                    seed: 42,
                    totalSteps: steps,
                    duration: .zero,
                    validationInterval: 0
                )
                let wallStart = ContinuousClock.now
                var losses: [Double] = []
                var last: TrainingProgress?
                for await progress in engine.train(examples: examples, configuration: trainConfig) {
                    last = progress
                    if !progress.isFinished {
                        losses.append(progress.loss)
                        if progress.step == 1 || progress.step % max(1, steps / 5) == 0 {
                            print(
                                String(
                                    format: "  step %d/%d  loss=%.4f  tok/s=%.1f",
                                    progress.step,
                                    progress.totalSteps,
                                    progress.loss,
                                    progress.tokensPerSecond
                                )
                            )
                        }
                    }
                }
                let wall = wallStart.duration(to: .now)
                let wallSec =
                    Double(wall.components.seconds)
                    + Double(wall.components.attoseconds) / 1e18
                if let last {
                    let lo = losses.min().map { String(format: "%.4f", $0) } ?? "n/a"
                    let hi = losses.max().map { String(format: "%.4f", $0) } ?? "n/a"
                    print(
                        String(
                            format: "  done steps=%d wall=%.1fs loss_range=[%@, %@] tps=%.1f",
                            last.step,
                            wallSec,
                            lo,
                            hi,
                            last.tokensPerSecond
                        )
                    )
                    if let outcome = last.gateOutcome {
                        let incumbent =
                            outcome.verdict.primaryMetric.incumbentValue.map { String($0) } ?? "n/a"
                        print(
                            "  gate promoted=\(outcome.verdict.promoted) candidate=v\(outcome.candidate.version) metric=\(outcome.verdict.primaryMetric.candidateValue) vs incumbent=\(incumbent)"
                        )
                        print("  reason: \(outcome.verdict.reason.prefix(200))…")
                    } else {
                        print("  gate: (no outcome — cancelled or failed)")
                    }
                }
                if let active = await engine.activeVersion() {
                    print("  active now: v\(active.version)")
                }
            }

            if !skipBlind {
                print("")
                print("--- Act 3: blind test ---")
                let id = SampleCorpus.blindRounds[0].incoming.id
                do {
                    let round = try await engine.prepareBlindRound(incomingEmailID: id)
                    print("  incoming: \(round.incoming.subject)")
                    for (index, candidate) in round.candidates.enumerated() {
                        let label = Character(UnicodeScalar(65 + index)!)
                        let words = BlindReplyPrompt.wordCount(candidate.body)
                        print(
                            "  Reply \(label): \(candidate.body.count) chars, \(words) words"
                        )
                        print("    \(preview(candidate.body, max: 160))")
                    }
                } catch {
                    print("  FAILED: \(error.localizedDescription)")
                }
            }

            if !skipCode {
                print("")
                print("--- Code-switching (one language sample) ---")
                let result = await engine.codeSwitchingDemo()
                if let en = result.languages.first(where: { $0.language == .english })
                    ?? result.languages.first
                {
                    print("  language: \(en.language.displayName)")
                    print(
                        "  base:    \(en.baseReply.count) chars — \(preview(en.baseReply, max: 120))"
                    )
                    print(
                        "  adapter: \(en.adaptedReply.count) chars — \(preview(en.adaptedReply, max: 120))"
                    )
                } else {
                    print("  (no language results)")
                }
            }

            if !skipPoison {
                print("")
                print("--- Poisoning (provisional threshold) ---")
                let poison = await engine.runPoisoningScenario()
                let metric = poison.verdict.primaryMetric
                print("  promoted=\(poison.verdict.promoted)")
                print(
                    "  candidate CE=\(metric.candidateValue)  incumbent CE=\(metric.incumbentValue.map { String($0) } ?? "n/a")"
                )
                print("  active before=v\(poison.activeVersionBefore.version) after=v\(poison.activeVersionAfter.version)")
                print("  candidate=v\(poison.candidate.version) status=\(poison.candidate.status)")
                print("  reason: \(poison.verdict.reason.prefix(280))")
            }

            print("")
            print("=== smoke complete ===")
        } catch {
            fputs("StyleMirrorSmoke fatal: \(error.localizedDescription)\n", stderr)
            exit(1)
        }
    }

    private static func preview(_ text: String, max: Int) -> String {
        let oneLine = text.replacingOccurrences(of: "\n", with: " ")
        if oneLine.count <= max { return oneLine }
        return String(oneLine.prefix(max)) + "…"
    }

    private static func intFlag(_ name: String, in args: [String]) -> Int? {
        guard let idx = args.firstIndex(of: name), idx + 1 < args.count else { return nil }
        return Int(args[idx + 1])
    }
}
