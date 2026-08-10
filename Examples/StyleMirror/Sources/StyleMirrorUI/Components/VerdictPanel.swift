import AdaptCore
import StyleMirrorEngine
import SwiftUI

/// The three pipeline stages, lighting up as they are reached (`DESIGN.md` §4.5).
public struct PipelineStepper: View {
    private let reachedStages: Int

    public init(reachedStages: Int) {
        self.reachedStages = reachedStages
    }

    private let stages = ["Train", "Evaluate", "Gate"]

    public var body: some View {
        HStack(spacing: Space.s) {
            ForEach(Array(stages.enumerated()), id: \.element) { index, stage in
                HStack(spacing: Space.xxs) {
                    if index < reachedStages {
                        Image(systemName: "checkmark.circle")
                            .font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(Palette.accent)
                    }
                    SectionLabel(
                        stage,
                        color: index < reachedStages ? Palette.ink : Palette.inkTertiary
                    )
                }
                if index < stages.count - 1 {
                    Text("→")
                        .textStyle(.dataS)
                        .foregroundStyle(Palette.inkTertiary)
                }
            }
        }
        .animation(Motion.checklistRow, value: reachedStages)
    }
}

/// One resolved evaluation check (`DESIGN.md` §6.9).
public struct GateChecklistRow: View {
    private let passed: Bool
    private let title: String
    private let value: String
    private let comparison: String?

    public init(passed: Bool, title: String, value: String, comparison: String? = nil) {
        self.passed = passed
        self.title = title
        self.value = value
        self.comparison = comparison
    }

    public var body: some View {
        HStack(spacing: Space.s) {
            Image(systemName: passed ? "checkmark.circle" : "xmark.circle")
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(passed ? Palette.accent : Palette.dataRed)
            Text(title)
                .textStyle(.bodyS)
                .foregroundStyle(Palette.ink)
            Spacer(minLength: Space.s)
            HStack(spacing: Space.xxs) {
                Text(value)
                    .textStyle(.dataS)
                    .foregroundStyle(passed ? Palette.inkSecondary : Palette.dataRed)
                if let comparison {
                    Text(comparison)
                        .textStyle(.dataS)
                        .foregroundStyle(Palette.inkTertiary)
                }
            }
        }
        .frame(height: 44)
    }
}

/// The gate's decision (`DESIGN.md` §6.10).
///
/// Green panel, red numbers, calm type: the refusal is a **success state**. The
/// candidate failed; the system did exactly its job. The panel is never restyled
/// red — that would turn the product's best moment into an error dialog.
public struct VerdictPanel: View {
    private let outcome: GateOutcome

    /// Drives the shield's 0.92 → 1 settle as the panel lands.
    @State private var shieldScale: CGFloat = 0.92

    public init(outcome: GateOutcome) {
        self.outcome = outcome
    }

    private var verdict: GateVerdict { outcome.verdict }

    public var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(spacing: Space.s) {
                Image(systemName: "checkmark.shield.fill")
                    .font(.system(size: 28))
                    .foregroundStyle(Palette.accent)
                    .scaleEffect(shieldScale)
                Text(verdict.promoted ? "Promoted." : "Not promoted.")
                    .textStyle(.headline)
                    .foregroundStyle(Palette.ink)
            }

            Text(explanation)
                .textStyle(.bodyS)
                .foregroundStyle(Palette.inkSecondary)
                .fixedSize(horizontal: false, vertical: true)

            Rectangle().fill(Palette.border).frame(height: 1)

            HStack(spacing: Space.xs) {
                Circle().fill(Palette.accent).frame(width: 8, height: 8)
                Text(
                    "Active adapter — v\(outcome.activeVersionAfter.version) · "
                        + (verdict.promoted ? "updated" : "unchanged")
                )
                .textStyle(.dataM)
                .foregroundStyle(Palette.ink)
            }

            Text("The same gate runs after every training pass, including the ones that happen while you sleep.")
                .textStyle(.caption)
                .foregroundStyle(Palette.inkTertiary)
        }
        .padding(Space.card)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: Radius.control, style: .continuous)
                .fill(Palette.accent.opacity(0.08))
        )
        .overlay(
            RoundedRectangle(cornerRadius: Radius.control, style: .continuous)
                .strokeBorder(Palette.accent.opacity(0.25), lineWidth: 1)
        )
        .task {
            withAnimation(Motion.verdictShield) { shieldScale = 1 }
        }
    }

    /// Built from the verdict's own numbers so the sentence can never drift from
    /// what was measured.
    private var explanation: String {
        let candidate = score(outcome.candidate)
        let active = score(outcome.activeVersionAfter)
        if verdict.promoted {
            return "Candidate v\(outcome.candidate.version) writes closer to you than your "
                + "previous adapter. It was scored against held-out samples of your own mail "
                + "and won — \(candidate) to \(score(outcome.activeVersionBefore))."
        }
        return "Candidate v\(outcome.candidate.version) writes worse than your active adapter. "
            + "It was scored against held-out samples of your own mail and lost — "
            + "\(candidate) to \(active). v\(outcome.activeVersionAfter.version) remains active. "
            + "Nothing changed."
    }

    private func score(_ version: AdapterVersion) -> String {
        version.evalReport?.primaryScore?.demoNumber(0) ?? "—"
    }
}
