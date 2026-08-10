import AdaptCore
import Observation
import StyleMirrorEngine
import SwiftUI

/// One point on the live loss curve.
public struct LossPoint: Identifiable, Sendable, Equatable {
    public let id: Int
    public let step: Int
    public let loss: Double
    public let validationLoss: Double?
}

/// All demo state, driven entirely through ``StyleMirrorEngine``.
///
/// The view layer never fabricates data: numbers on screen come from the engine
/// or from the user's own pasted text. Nothing here is hardcoded to make a claim
/// look better — notably ``bytesSent``, which is read from the engine's outbound
/// traffic meter rather than printed as a literal zero.
@MainActor
@Observable
public final class DemoState {

    /// The five screens, in performance order (`DESIGN.md` §5).
    public enum Screen: Int, CaseIterable, Identifiable, Sendable {
        case offline, train, blindTest, languages, gate

        public var id: Int { rawValue }

        /// Tab label as it appears in the status strip (§8.1).
        var tabTitle: String {
            switch self {
            case .offline: "1 Offline"
            case .train: "2 Train"
            case .blindTest: "3 Blind test"
            case .languages: "Languages"
            case .gate: "Gate"
            }
        }
    }

    // MARK: Navigation & chrome

    public var screen: Screen = .offline
    public var networkStatus: NetworkStatus = .online
    public var bytesSent: UInt64 = 0

    /// Seconds since the network dropped — the clock that proves the zero is live.
    public var secondsOffline: Int = 0

    // MARK: Version history

    public var versions: [AdapterVersion] = []
    public var activeVersion: AdapterVersion?

    /// Version label used across the strip and identity chips, e.g. `v8`.
    public var activeVersionLabel: String {
        activeVersion.map { "v\($0.version)" } ?? "none"
    }

    // MARK: Act 2 — training

    public var pastedCorpus: String = ""
    public var isTraining = false
    public var lossPoints: [LossPoint] = []
    public var progress: TrainingProgress?
    public var promotionMessage: String?

    /// Verdict from the most recent completed run — the gate passing (§4.2).
    public var liveGateOutcome: GateOutcome?

    /// Stage default is the realistic ~2.5 min run; ⌘⌥F drops to a ~20 s pass
    /// for UI iteration and rehearsal (§10 only requires ≥ 1 update/s).
    public var trainingConfiguration: TrainingConfiguration = .rehearsal

    /// Emails detected in the pasted corpus — blocks separated by blank lines.
    public var pastedEmailCount: Int {
        pastedCorpus
            .components(separatedBy: "\n\n")
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .count
    }

    /// The pasted corpus as training examples — one per blank-line-separated block.
    ///
    /// The demo trains on what the presenter actually pasted. Mirrors
    /// `SampleCorpus.trainingExamples`' shape so the real `AdaptTrain` backend
    /// receives the same prompt/completion convention.
    ///
    /// Source is `.explicitEdit` (weight 1.0): this is prose the user wrote
    /// themselves, which is the gold signal — not `.synthetic`, which §4.2
    /// reserves for app-provided seed examples.
    public var pastedTrainingExamples: [TrainingExample] {
        pastedCorpus
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .map { body in
                TrainingExample(
                    prompt: "Write a reply email in your own voice.",
                    completion: body,
                    source: .explicitEdit
                )
            }
    }

    /// Rough token estimate (~4 characters per token) for the corpus chips.
    public var pastedTokenEstimate: Int {
        pastedCorpus.isEmpty ? 0 : max(1, pastedCorpus.count / 4)
    }

    // MARK: Act 3 — blind test

    public var round: BlindTestRound?
    public var pickedCandidateID: UUID?
    public var revealResult: BlindTestGuessResult?
    public var tally = BlindTestTally(
        roundsPlayed: 0, humanCorrectlyIdentified: 0,
        adapterMistakenForHuman: 0, baseMistakenForHuman: 0
    )
    private var incomingIDs: [String] = []
    private var roundIndex = 0

    // MARK: Languages

    public var codeSwitch: CodeSwitchResult?

    // MARK: Gate — the poisoning scene

    public var gateOutcome: GateOutcome?
    public var isRunningGate = false
    /// The verdict panel appears only after the deliberate pause (§7), so the
    /// checklist can already show real measured values while it resolves.
    public var verdictVisible = false
    /// How many checklist rows have resolved, so they can appear in sequence.
    public var resolvedChecklistRows = 0

    // MARK: Dependencies

    /// Recreated on reset: a demo session accumulates promotions and tally, so
    /// restoring the opening state (§5's ⌘⇧R) needs a fresh engine, not a reload
    /// of the mutated one.
    private let makeEngine: @Sendable () -> any StyleMirrorEngine
    private var engine: any StyleMirrorEngine
    private let reachability = NetworkReachability()

    public init(makeEngine: @escaping @Sendable () -> any StyleMirrorEngine) {
        self.makeEngine = makeEngine
        self.engine = makeEngine()
    }

    // MARK: Lifecycle

    /// Loads version history and starts observing network state and traffic.
    public func start() async {
        versions = await engine.adapterVersions()
        activeVersion = await engine.activeVersion()
        incomingIDs = await engine.blindTestIncomingIDs
        tally = await engine.blindTestTally()

        await withTaskGroup(of: Void.self) { group in
            group.addTask { [weak self] in await self?.observeNetwork() }
            group.addTask { [weak self] in await self?.tickClocks() }
        }
    }

    private func observeNetwork() async {
        for await status in reachability.updates {
            let wasOnline = networkStatus == .online
            networkStatus = status
            if status == .offline, wasOnline { secondsOffline = 0 }
        }
    }

    /// Drives the offline clock and re-reads the traffic meter once a second.
    ///
    /// The counter is polled rather than assumed: if anything ever did send a
    /// byte, this screen would say so (§6.2).
    private func tickClocks() async {
        while !Task.isCancelled {
            try? await Task.sleep(for: .seconds(1))
            if networkStatus == .offline { secondsOffline += 1 }
            bytesSent = await OutboundTrafficMeter.shared.bytesSent
        }
    }

    // MARK: Act 2

    /// Runs a training pass, streaming progress into the chart.
    public func train() async {
        guard !isTraining else { return }
        isTraining = true
        lossPoints = []
        progress = nil
        promotionMessage = nil

        for await update in engine.train(
            examples: pastedTrainingExamples,
            configuration: trainingConfiguration
        ) {
            progress = update
            if !update.isFinished {
                lossPoints.append(
                    LossPoint(
                        id: update.step,
                        step: update.step,
                        loss: update.loss,
                        validationLoss: update.validationLoss
                    )
                )
            }
        }

        isTraining = false

        // The gate's verdict comes from the engine, never from the UI assuming
        // success: a cancelled run carries no outcome and promotes nothing.
        if let outcome = progress?.gateOutcome, outcome.verdict.promoted {
            liveGateOutcome = outcome
            promotionMessage = Self.promotionLine(outcome)
        }
        versions = await engine.adapterVersions()
        activeVersion = await engine.activeVersion()
    }

    /// "Gate: passed. v8 promoted — eval 75 against v7's 74." (§8.3), built from
    /// the verdict's own numbers rather than a fixture.
    private static func promotionLine(_ outcome: GateOutcome) -> String {
        let metric = outcome.verdict.primaryMetric
        let candidate = metric.candidateValue.demoNumber(0)
        let incumbent = metric.incumbentValue?.demoNumber(0) ?? "—"
        return "Gate: passed. v\(outcome.activeVersionAfter.version) promoted — "
            + "eval \(candidate) against v\(outcome.activeVersionBefore.version)'s \(incumbent)."
    }

    // MARK: Gate

    /// Runs the poisoned batch through the pipeline, resolving the checklist rows
    /// in sequence so the audience can read each one (§7).
    public func runPoisoning() async {
        guard !isRunningGate else { return }
        isRunningGate = true
        gateOutcome = nil
        verdictVisible = false
        resolvedChecklistRows = 0

        // Measured up front so the checklist shows real values as it resolves;
        // only the verdict panel waits for the pause.
        gateOutcome = await engine.runPoisoningScenario()

        for row in 1...3 {
            resolvedChecklistRows = row
            try? await Task.sleep(for: Motion.checklistRowGap)
        }
        // Stillness before the verdict: hesitation reads as deliberation.
        try? await Task.sleep(for: Motion.verdictHold)

        verdictVisible = true
        activeVersion = await engine.activeVersion()
        isRunningGate = false
    }

    // MARK: Act 3

    public func nextRound() async {
        guard !incomingIDs.isEmpty else { return }
        let id = incomingIDs[roundIndex % incomingIDs.count]
        roundIndex += 1
        pickedCandidateID = nil
        revealResult = nil
        round = try? await engine.prepareBlindRound(incomingEmailID: id)
    }

    public func pick(_ candidateID: UUID) {
        guard revealResult == nil else { return }
        pickedCandidateID = candidateID
    }

    public func reveal() async {
        guard let round, let pickedCandidateID, revealResult == nil else { return }
        revealResult = try? await engine.submitBlindGuess(
            roundID: round.id, candidateID: pickedCandidateID
        )
        if let revealResult { tally = revealResult.tally }
    }

    /// The result line shown after a reveal (§8.4).
    public var revealResultLine: String? {
        guard let revealResult else { return nil }
        switch revealResult.guessedRole {
        case .human: return "Correct. That one was human."
        case .adaptedModel: return "That was the adapter."
        case .baseModel: return "That was the base model."
        }
    }

    // MARK: Languages

    public func loadCodeSwitching() async {
        guard codeSwitch == nil else { return }
        codeSwitch = await engine.codeSwitchingDemo()
    }

    // MARK: Presenter controls

    /// Space fires the current screen's primary action so the presenter never
    /// hunts with the mouse (§5).
    public func firePrimaryAction() async {
        switch screen {
        case .train where !isTraining && !pastedCorpus.isEmpty:
            await train()
        case .blindTest:
            if round == nil || revealResult != nil {
                await nextRound()
            } else {
                await reveal()
            }
        case .gate where !isRunningGate:
            await runPoisoning()
        default:
            break
        }
    }

    /// Applies launch-time options: open on a screen, optionally preload the
    /// sample corpus, optionally fire that screen's primary action.
    ///
    /// Used for rehearsal and screenshot capture so no synthetic input is needed.
    public func applyLaunchOptions(_ options: LaunchOptions = .fromCommandLine()) async {
        if options.preloadSampleCorpus, pastedCorpus.isEmpty {
            pastedCorpus = await engine.sentEmails
                .map(\.body)
                .joined(separator: "\n\n")
        }
        if let screen = options.screen {
            self.screen = screen
        }
        guard options.autorun else { return }
        // Let the screen settle so a capture shows the action's result rather
        // than its first frame.
        try? await Task.sleep(for: .milliseconds(600))
        await firePrimaryAction()
    }

    /// ⌘⌥F — toggles between the stage-realistic run and a fast pass.
    public func toggleFastMode() {
        trainingConfiguration =
            trainingConfiguration.totalSteps == TrainingConfiguration.rehearsal.totalSteps
            ? .uiDevelopment : .rehearsal
    }

    /// ⌘⇧R — resets the whole demo without relaunching.
    public func reset() async {
        lossPoints = []
        progress = nil
        promotionMessage = nil
        isTraining = false
        round = nil
        pickedCandidateID = nil
        revealResult = nil
        codeSwitch = nil
        liveGateOutcome = nil
        gateOutcome = nil
        verdictVisible = false
        resolvedChecklistRows = 0
        isRunningGate = false
        roundIndex = 0
        pastedCorpus = ""
        screen = .offline

        engine = makeEngine()
        versions = await engine.adapterVersions()
        activeVersion = await engine.activeVersion()
        incomingIDs = await engine.blindTestIncomingIDs
        tally = await engine.blindTestTally()
    }
}
