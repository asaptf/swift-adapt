import AdaptCore
import StyleMirrorEngine
import SwiftUI

/// The poisoning scene — the eval gate refusing to ship a worse adapter
/// (`DESIGN.md` §4.5).
///
/// The batch card carries **no warning styling**: the UI does not know the batch
/// is poisoned, which is exactly the point. Only the gate finds out, and only by
/// measuring.
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
                batchCard
                pipelineCard
            }
            .frame(height: 500)

            VersionTimeline(
                versions: state.versions,
                activeVersion: state.activeVersion?.version,
                rejected: rejectedNode
            )
        }
    }

    // MARK: Left — the batch

    private var batchCard: some View {
        VStack(alignment: .leading, spacing: Space.m) {
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
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardSurface()
        .frame(width: 500)
        .frame(maxHeight: .infinity, alignment: .top)
    }

    // MARK: Right — the pipeline

    private var pipelineCard: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            PipelineStepper(reachedStages: reachedStages)

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

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardSurface()
        .frame(width: 836)
        .frame(maxHeight: .infinity, alignment: .top)
        .animation(Motion.verdictPanel, value: state.verdictVisible)
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
                value: format(metric.candidateValue),
                comparison: "(limit \(format(metric.threshold)))"
            ),
            GateChecklistRow(
                passed: outcome.verdict.promoted,
                title: "Style match",
                value: score(outcome.candidate),
                comparison: "(active adapter \(score(outcome.activeVersionAfter)))"
            ),
        ]
    }

    private var reachedStages: Int {
        if state.gateOutcome == nil && !state.isRunningGate { return 0 }
        return max(1, state.resolvedChecklistRows)
    }

    private var rejectedNode: RejectedNode? {
        guard let outcome = state.gateOutcome, !outcome.verdict.promoted else { return nil }
        return RejectedNode(
            versionLabel: "v\(outcome.candidate.version)",
            score: outcome.candidate.evalReport?.primaryScore ?? 0
        )
    }

    private func format(_ value: Double) -> String {
        value.demoNumber(2)
    }

    private func score(_ version: AdapterVersion) -> String {
        version.evalReport?.primaryScore?.demoNumber(0) ?? "—"
    }
}
