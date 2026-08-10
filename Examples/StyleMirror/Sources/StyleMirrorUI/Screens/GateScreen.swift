import AdaptCore
import StyleMirrorEngine
import SwiftUI

/// The gate scene (`DESIGN.md` §4.5).
///
/// The primary argument is the regression that actually happened: on the seeded
/// registry, night seven measured worse than night six on ordinary mail and became
/// active anyway, because promotion is manual until the evaluation module exists.
/// The timeline below is the evidence — worse held-out loss sits higher, so night
/// seven kinks upward.
///
/// The deliberately poisoned batch is kept as a second case. It reads better once
/// the audience has watched the same comparison catch a real failure.
public struct GateScreen: View {
    private let state: DemoState

    public init(state: DemoState) {
        self.state = state
    }

    /// Poisoned fixtures from `DESIGN.md` §8.6 — deliberately absurd, so nobody
    /// mistakes a demo artefact for real mail.
    private let poisonedRows = [
        "ARRR, THE QUARTERLY BOOTY BE ATTACHED, MATEY",
        "YE SCURVY DEADLINE BE SLIPPIN, SAVVY??",
        "SHIVER ME TIMBERS, APPROVE THE INVOICE OR WALK THE PLANK",
    ]

    public var body: some View {
        VStack(spacing: Space.l) {
            HStack(alignment: .top, spacing: Space.l) {
                caseCard
                verdictCard
            }
            .frame(height: 500)

            VersionTimeline(
                versions: state.versions,
                activeVersion: state.activeVersion?.version,
                rejected: rejectedNode
            )
        }
        .task { await state.loadRegressionCase() }
    }

    // MARK: Left — the case

    private var caseCard: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            casePicker

            switch state.gateCase {
            case .regression: regressionCase
            case .poisoned: poisonedCase
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardSurface()
        .frame(width: 500)
        .frame(maxHeight: .infinity, alignment: .top)
    }

    private var casePicker: some View {
        HStack(spacing: Space.xs) {
            ForEach(DemoState.GateCase.allCases) { option in
                let isSelected = state.gateCase == option
                Button {
                    state.gateCase = option
                } label: {
                    Text(option.title)
                        .textStyle(.label)
                        .foregroundStyle(isSelected ? Palette.ink : Palette.inkTertiary)
                        .padding(.horizontal, Space.s)
                        .frame(height: 26)
                        .background(
                            Capsule().fill(isSelected ? Palette.surfaceRaised : .clear)
                        )
                }
                .buttonStyle(.plain)
            }
        }
    }

    @ViewBuilder private var regressionCase: some View {
        if let comparison = state.activeVsBest, !comparison.isActiveBest {
            Text("Night \(comparison.active.version) came out worse")
                .textStyle(.headline)
                .foregroundStyle(Palette.ink)

            VStack(spacing: 0) {
                measurementRow(
                    label: "Active",
                    version: comparison.active.version,
                    value: comparison.activeScore,
                    isWorse: true
                )
                Rectangle().fill(Palette.border).frame(height: 1)
                measurementRow(
                    label: "Best measured",
                    version: comparison.bestMeasured.version,
                    value: comparison.bestScore,
                    isWorse: false
                )
            }

            Text(
                "Ordinary mail, nothing sabotaged. It is "
                    + "\(comparison.gapNats.demoNumber(2)) nats worse and it shipped anyway, "
                    + "because promotion stays manual until the evaluation module exists."
            )
            .textStyle(.bodyS)
            .foregroundStyle(Palette.inkSecondary)
            .fixedSize(horizontal: false, vertical: true)

            Spacer(minLength: Space.m)

            Button("Roll back to v\(comparison.bestMeasured.version)") {
                Task { await state.rollBackToBest() }
            }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(state.isRollingBack || state.rollbackResult != nil)
        } else if state.activeVsBest != nil {
            Text("The active adapter is the best one measured")
                .textStyle(.headline)
                .foregroundStyle(Palette.ink)
            Text("Nothing to roll back to. Re-seed the registry to restore the demo's starting state.")
                .textStyle(.bodyS)
                .foregroundStyle(Palette.inkSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        } else {
            EmptyStateMessage(text: "No measured versions in this registry yet.")
        }
    }

    private func measurementRow(
        label: String,
        version: Int,
        value: Double,
        isWorse: Bool
    ) -> some View {
        HStack(spacing: Space.s) {
            SectionLabel(label)
            Text("v\(version)")
                .textStyle(.dataM)
                .foregroundStyle(Palette.ink)
            Spacer(minLength: Space.s)
            Text(value.demoNumber(3))
                .textStyle(.dataM)
                .foregroundStyle(isWorse ? Palette.dataRed : Palette.accent)
        }
        .frame(height: 44)
    }

    @ViewBuilder private var poisonedCase: some View {
        Text("Training batch — 20 examples")
            .textStyle(.headline)
            .foregroundStyle(Palette.ink)

        VStack(alignment: .leading, spacing: Space.xs) {
            ForEach(poisonedRows, id: \.self) { row in
                Text(row)
                    .textStyle(.dataM)
                    .foregroundStyle(Palette.ink)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            Text("+ 17 more like this")
                .textStyle(.dataM)
                .foregroundStyle(Palette.inkTertiary)
        }

        Spacer(minLength: Space.m)

        Button("Train on this batch") {
            Task { await state.runPoisoning() }
        }
        .buttonStyle(PrimaryButtonStyle())
        .disabled(state.isRunningGate)
    }

    // MARK: Right — the verdict

    private var verdictCard: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            PipelineStepper(reachedStages: reachedStages)

            switch state.gateCase {
            case .regression: regressionVerdict
            case .poisoned: poisonedVerdict
            }

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardSurface()
        .frame(width: 836)
        .frame(maxHeight: .infinity, alignment: .top)
        .animation(Motion.verdictPanel, value: state.verdictVisible)
        .animation(Motion.verdictPanel, value: state.rollbackResult?.toVersion.version)
    }

    @ViewBuilder private var regressionVerdict: some View {
        // Prefer the retained comparison: after a rollback the live one is
        // degenerate (active *is* best), and reading it printed "v6 measured 0.00
        // nats worse than v6". The story to tell is the one that was just fixed.
        if let comparison = state.lastRegressionComparison ?? state.activeVsBest,
            comparison.gapNats > 0
        {
            RegressionVerdictPanel(
                activeVersion: comparison.active.version,
                activeScore: comparison.activeScore,
                bestVersion: comparison.bestMeasured.version,
                bestScore: comparison.bestScore,
                gapNats: comparison.gapNats,
                metricName: metricName,
                rolledBackTo: state.rollbackResult?.toVersion.version
            )

            if let rollback = state.rollbackResult {
                HStack(spacing: Space.xs) {
                    Image(systemName: "checkmark.seal.fill")
                        .font(.system(size: 16))
                        .foregroundStyle(Palette.accent)
                    Text(
                        "Rolled back in \(milliseconds(rollback.elapsed)) ms — a pointer flip, "
                            + "no weights rewritten."
                    )
                    .textStyle(.bodyS)
                    .foregroundStyle(Palette.ink)
                }
                .padding(.horizontal, Space.s)
                .padding(.vertical, Space.xs)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: Radius.control, style: .continuous)
                        .fill(Palette.accentWash)
                )
                .transition(.opacity)
            }
        } else {
            EmptyStateMessage(text: "Comparing the active adapter against the best measured one…")
        }
    }

    /// Metric name as recorded, so the sentence names what was measured.
    private var metricName: String {
        state.versions.compactMap(\.evalReport).last?.primaryMetric?
            .replacingOccurrences(of: "_", with: " ") ?? "held-out loss"
    }

    @ViewBuilder private var poisonedVerdict: some View {
        if state.gateOutcome == nil && !state.isRunningGate {
            EmptyStateMessage(text: "The gate has nothing to judge. Train a batch to see it work.")
        } else {
            checklist
            Spacer(minLength: Space.m)
            if state.verdictVisible, let outcome = state.gateOutcome {
                VerdictPanel(outcome: outcome)
                    .transition(.opacity.combined(with: .offset(y: Motion.riseDistance)))
            }
        }
    }

    private var checklist: some View {
        VStack(spacing: 0) {
            ForEach(Array(checklistRows.enumerated()), id: \.offset) { index, row in
                if index > 0 {
                    Rectangle().fill(Palette.border).frame(height: 1)
                }
                row
                    .opacity(state.resolvedChecklistRows > index ? 1 : 0)
                    .offset(y: state.resolvedChecklistRows > index ? 0 : Motion.checklistRise)
            }
        }
        .animation(Motion.checklistRow, value: state.resolvedChecklistRows)
    }

    /// Values come from the gate's own measurement, not from the spec's fixtures —
    /// what the audience reads must be what was computed.
    private var checklistRows: [GateChecklistRow] {
        guard let outcome = state.gateOutcome else { return [] }
        let metric = outcome.verdict.primaryMetric
        let metricPassed =
            metric.lowerIsBetter
            ? metric.candidateValue <= metric.threshold
            : metric.candidateValue >= metric.threshold

        return [
            GateChecklistRow(
                passed: true,
                title: "Training completed",
                value: "\(state.trainingConfiguration.totalSteps) steps"
            ),
            GateChecklistRow(
                passed: metricPassed,
                title: metric.displayName,
                value: metric.candidateValue.demoNumber(2),
                comparison: "(limit \(metric.threshold.demoNumber(2)))"
            ),
        ]
    }

    // MARK: Values

    private var reachedStages: Int {
        switch state.gateCase {
        case .regression:
            if state.rollbackResult != nil { return 3 }
            return state.regressionOutcome == nil ? 1 : 3
        case .poisoned:
            if state.gateOutcome == nil && !state.isRunningGate { return 0 }
            return max(1, state.resolvedChecklistRows)
        }
    }

    /// Only the poisoned case produces a refused candidate outside the promoted
    /// history; the regression is already *in* the timeline, which is the point.
    private var rejectedNode: RejectedNode? {
        guard state.gateCase == .poisoned,
            let outcome = state.gateOutcome,
            !outcome.verdict.promoted,
            let score = outcome.candidate.evalReport?.primaryScore
        else { return nil }
        return RejectedNode(versionLabel: "v\(outcome.candidate.version)", score: score)
    }

    private func milliseconds(_ duration: Duration) -> String {
        let parts = duration.components
        let value = Double(parts.seconds) * 1000 + Double(parts.attoseconds) / 1e15
        return value.demoNumber(1)
    }
}
