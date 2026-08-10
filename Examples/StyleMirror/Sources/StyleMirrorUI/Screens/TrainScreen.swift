import StyleMirrorEngine
import SwiftUI

/// Act 2 — training, live (`DESIGN.md` §4.2).
///
/// The training state has to hold an audience for two to three minutes without a
/// narrator, so the chart is the largest object on screen and every number beside
/// it is live.
public struct TrainScreen: View {
    @Bindable private var state: DemoState
    @FocusState private var pasteFieldFocused: Bool

    public init(state: DemoState) {
        self.state = state
    }

    private var hasStarted: Bool { state.isTraining || !state.lossPoints.isEmpty }

    public var body: some View {
        if hasStarted {
            trainingLayout
        } else {
            pasteLayout
        }
    }

    // MARK: Before training

    private var pasteLayout: some View {
        VStack(spacing: 0) {
            Text("Train on your sent mail.")
                .textStyle(.title)
                .foregroundStyle(Palette.ink)
            Spacer().frame(height: Space.xs)
            Text("Paste 20–50 sent emails. They stay in memory, on this Mac, and are gone when you quit.")
                .textStyle(.bodyS)
                .foregroundStyle(Palette.inkSecondary)
                .multilineTextAlignment(.center)

            Spacer().frame(height: Space.l)
            pasteField
            Spacer().frame(height: Space.m)
            corpusChips
            Spacer().frame(height: Space.l)

            Button("Train") {
                Task { await state.train() }
            }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(state.pastedCorpus.isEmpty)
        }
        .frame(maxWidth: 760)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var pasteField: some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: $state.pastedCorpus)
                .textStyle(.dataM)
                .foregroundStyle(Palette.ink)
                .scrollContentBackground(.hidden)
                .padding(Space.s)
                .focused($pasteFieldFocused)

            if state.pastedCorpus.isEmpty {
                Text("⌘V to paste your sent mail")
                    .textStyle(.dataM)
                    .foregroundStyle(Palette.inkTertiary)
                    .padding(Space.s)
                    .padding(.leading, 5)
                    .allowsHitTesting(false)
            }
        }
        .frame(width: 760, height: 320)
        .background(
            RoundedRectangle(cornerRadius: Radius.control, style: .continuous)
                .fill(Palette.surface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Radius.control, style: .continuous)
                .strokeBorder(Palette.border, lineWidth: 1)
        )
    }

    /// Counts come from the pasted text, not from a fixture — if the presenter
    /// pastes 31 emails the chip says 31.
    private var corpusChips: some View {
        HStack(spacing: Space.xs) {
            StatChip("\(state.pastedEmailCount) emails")
            StatChip("\(state.pastedTokenEstimate.demoNumber) tokens")
            StatChip("≈ \(estimatedMinutes) min on this Mac")
        }
        .opacity(state.pastedCorpus.isEmpty ? 0 : 1)
        .animation(Motion.easeOut, value: state.pastedCorpus.isEmpty)
    }

    private var estimatedMinutes: Int {
        let seconds = Double(state.trainingConfiguration.duration.components.seconds)
        return max(1, Int((seconds / 60).rounded()))
    }

    // MARK: During and after training

    private var trainingLayout: some View {
        VStack(spacing: Space.l) {
            HStack(alignment: .top, spacing: Space.l) {
                chartCard
                metricColumn
            }
            VersionTimeline(
                versions: state.versions,
                activeVersion: state.activeVersion?.version
            )
            .animation(Motion.timelineNode, value: state.versions.count)
        }
    }

    private var chartCard: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            HStack(alignment: .firstTextBaseline) {
                Text("Training candidate v\(candidateVersion)")
                    .textStyle(.headline)
                    .foregroundStyle(Palette.ink)
                Spacer()
                Text("LoRA r=\(rank) · \(state.trainingConfiguration.totalSteps) steps")
                    .textStyle(.dataS)
                    .foregroundStyle(Palette.inkSecondary)
            }

            if let promotion = state.promotionMessage {
                promotionRow(promotion)
            }

            LossChart(
                points: state.lossPoints,
                totalSteps: state.trainingConfiguration.totalSteps
            )
        }
        .cardSurface()
        .frame(width: 900, height: 500)
    }

    /// Promotion is routine, so it arrives as a quiet row — no modal, no
    /// confetti. That restraint is what the gate screen later cashes in (§4.2).
    private func promotionRow(_ message: String) -> some View {
        HStack(spacing: Space.xs) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 16))
                .foregroundStyle(Palette.accent)
            Text(message)
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

    private var metricColumn: some View {
        VStack(spacing: Space.m) {
            MetricTile(
                label: "Step",
                value: "\(state.progress?.step ?? 0) / \(state.trainingConfiguration.totalSteps)",
                widestValue: "888 / 888"
            )
            MetricTile(
                label: "Tokens per second",
                value: (state.progress?.tokensPerSecond ?? 0).demoNumber(0),
                unit: "tok/s",
                widestValue: "8,888"
            )
            MetricTile(
                label: "Elapsed",
                value: clock(state.progress?.elapsed),
                widestValue: "88:88"
            )
            MetricTile(
                label: "Remaining",
                value: "~" + clock(state.progress?.estimatedRemaining),
                widestValue: "~88:88"
            )
        }
        .frame(width: 436)
    }

    // MARK: Values

    private var candidateVersion: Int {
        (state.versions.map(\.version).max() ?? 0) + (state.isTraining ? 1 : 0)
    }

    private var rank: Int {
        state.activeVersion?.lineage.loraConfig.loraParameters.rank ?? 16
    }

    private func clock(_ duration: Duration?) -> String {
        guard let duration else { return "0:00" }
        let total = Int(duration.components.seconds)
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
