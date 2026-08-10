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
        let appSim = args.contains("--app-sim")

        print("=== StyleMirrorSmoke (AdaptEngine) ===")
        do {
            let config = try AdaptEngineConfiguration.seededDemo()
            print("  model:    \(config.modelID)")
            print("  registry: \(config.registryRoot.path)")
            print("  held-out: \(config.heldOutJSONL?.path ?? "(none)")")
            print("  steps:    \(steps)")
            if appSim {
                print("  mode:     app-sim (MainActor + awaited progress hop)")
                try await MainActorAppSim.run(config: config)
                print("")
                print("=== smoke complete ===")
                return
            }

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
                let wallSec = durationSeconds(wall)
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
                    let wallStart = ContinuousClock.now
                    let timer = ProgressTimer(started: wallStart)
                    let progressBox = ProgressCollector()
                    let round = try await engine.prepareBlindRound(incomingEmailID: id) { event in
                        let mark = timer.mark()
                        progressBox.append(event)
                        print(
                            String(
                                format: "  progress: %d/%d%@  +%.2fs (t=%.2fs)",
                                event.completed,
                                event.total,
                                event.unitLabel.map { " (\($0))" } ?? "",
                                mark.delta,
                                mark.elapsed
                            )
                        )
                        // No MainActor hop here — harness is not the app path.
                    }
                    let wallSec = durationSeconds(wallStart.duration(to: .now))
                    print(String(format: "  preparation wall=%.2fs", wallSec))
                    let progressEvents = progressBox.snapshot()
                    if let last = progressEvents.last {
                        print(
                            "  progress events: \(progressEvents.count) (final \(last.completed)/\(last.total))"
                        )
                    } else {
                        print("  progress events: 0 — HANDLER NEVER INVOKED")
                    }

                    // Second call: model should already be cached on the engine.
                    let warmStart = ContinuousClock.now
                    let id2 = SampleCorpus.blindRounds[
                        min(1, SampleCorpus.blindRounds.count - 1)
                    ].incoming.id
                    _ = try await engine.prepareBlindRound(incomingEmailID: id2)
                    print(
                        String(
                            format: "  second-round wall=%.2fs (cached session expected)",
                            durationSeconds(warmStart.duration(to: .now))
                        )
                    )

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
                print("--- Code-switching (all languages) ---")
                let wallStart = ContinuousClock.now
                let timer = ProgressTimer(started: wallStart)
                let progressBox = ProgressCollector()
                let result = await engine.codeSwitchingDemo { event in
                    let mark = timer.mark()
                    progressBox.append(event)
                    print(
                        String(
                            format: "  progress: %d/%d%@  +%.2fs (t=%.2fs)",
                            event.completed,
                            event.total,
                            event.unitLabel.map { " (\($0))" } ?? "",
                            mark.delta,
                            mark.elapsed
                        )
                    )
                }
                let wallSec = durationSeconds(wallStart.duration(to: .now))
                print(String(format: "  wall=%.2fs", wallSec))
                if let reason = result.unavailabilityReason {
                    print("  UNAVAILABLE: \(reason)")
                } else if result.languages.isEmpty {
                    print("  (no language results — engine returned empty without a reason)")
                } else {
                    print("  languages: \(result.languages.count)")
                    for pair in result.languages {
                        print("  --- \(pair.language.displayName) ---")
                        print(
                            "  base:    \(pair.baseReply.count) chars — \(preview(pair.baseReply, max: 200))"
                        )
                        print(
                            "  adapter: \(pair.adaptedReply.count) chars — \(preview(pair.adaptedReply, max: 200))"
                        )
                    }
                    let progressEvents = progressBox.snapshot()
                    if let last = progressEvents.last {
                        print(
                            "  progress events: \(progressEvents.count) (final \(last.completed)/\(last.total))"
                        )
                    } else {
                        print("  progress events: 0 — HANDLER NEVER INVOKED")
                    }
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

/// MainActor surface matching `DemoState.nextRound` + awaited progress hop.
@MainActor
private enum MainActorAppSim {
    static func run(config: AdaptEngineConfiguration) async throws {
        let wallStart = ContinuousClock.now
        func mark(_ label: String) {
            print(String(format: "  [+%.2fs] %@", durationSeconds(wallStart.duration(to: .now)), label))
            fflush(stdout)
        }

        mark("building engine on MainActor")
        let engine = try AdaptEngine(configuration: config, seed: 42)
        let active = await engine.activeVersion()
        mark("active=\(active.map { "v\($0.version)" } ?? "nil")")

        let id = SampleCorpus.blindRounds[0].incoming.id
        let progressBox = ProgressCollector()
        mark("prepareBlindRound START")
        let round = try await engine.prepareBlindRound(incomingEmailID: id) { event in
            await MainActor.run {
                progressBox.append(event)
                print(
                    String(
                        format: "  [+%.2fs] progress hop landed %d/%d %@ (hits=%d)",
                        durationSeconds(wallStart.duration(to: .now)),
                        event.completed,
                        event.total,
                        event.unitLabel ?? "",
                        progressBox.snapshot().count
                    )
                )
                fflush(stdout)
            }
        }
        mark("prepareBlindRound DONE candidates=\(round.candidates.count) progressHits=\(progressBox.snapshot().count)")

        mark("second round START")
        let id2 = SampleCorpus.blindRounds[min(1, SampleCorpus.blindRounds.count - 1)].incoming.id
        _ = try await engine.prepareBlindRound(incomingEmailID: id2)
        mark("second round DONE")
    }
}

private func durationSeconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds)
        + Double(duration.components.attoseconds) / 1e18
}

/// Thread-safe progress event bag for the smoke harness (progress callbacks
/// may run off the main task).
private final class ProgressCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [GenerationProgress] = []

    func append(_ event: GenerationProgress) {
        lock.lock()
        events.append(event)
        lock.unlock()
    }

    func snapshot() -> [GenerationProgress] {
        lock.lock()
        defer { lock.unlock() }
        return events
    }
}

/// Thread-safe wall-clock marks between progress events.
private final class ProgressTimer: @unchecked Sendable {
    private let lock = NSLock()
    private let started: ContinuousClock.Instant
    private var last: ContinuousClock.Instant

    init(started: ContinuousClock.Instant) {
        self.started = started
        self.last = started
    }

    struct Mark {
        let delta: Double
        let elapsed: Double
    }

    func mark() -> Mark {
        lock.lock()
        defer { lock.unlock() }
        let now = ContinuousClock.now
        let delta = durationSeconds(last.duration(to: now))
        let elapsed = durationSeconds(started.duration(to: now))
        last = now
        return Mark(delta: delta, elapsed: elapsed)
    }
}
