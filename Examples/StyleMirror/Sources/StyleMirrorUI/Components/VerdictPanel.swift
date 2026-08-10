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
    /// Overrides the headline. The Gate screen judges an **already active**
    /// version retrospectively ("would not have been promoted"), which is a
    /// different sentence from refusing a fresh candidate.
    private let headlineOverride: String?

    /// Drives the shield's 0.92 → 1 settle as the panel lands.
    @State private var shieldScale: CGFloat = 0.92

    public init(outcome: GateOutcome, headline: String? = nil) {
        self.outcome = outcome
        self.headlineOverride = headline
    }

    private var verdict: GateVerdict { outcome.verdict }

    public var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(spacing: Space.s) {
                Image(systemName: "checkmark.shield.fill")
                    .font(.system(size: 28))
                    .foregroundStyle(Palette.accent)
                    .scaleEffect(shieldScale)
                Text(headlineOverride ?? (verdict.promoted ? "Promoted." : "Not promoted."))
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

            Text("Provisional check: held-out loss against the active adapter. The full promotion gate arrives with the evaluation module.")
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
    /// Built from the verdict's own numbers so the sentence can never drift from
    /// what was measured. Phrased for the metric's direction: held-out loss is
    /// better when it is *lower*, so "beat" means a smaller number.
    private var explanation: String {
        let candidate = score(outcome.candidate)
        let incumbent = score(
            verdict.promoted ? outcome.activeVersionBefore : outcome.activeVersionAfter
        )
        let metric = verdict.primaryMetric.displayName.lowercased()
        if verdict.promoted {
            return "Candidate v\(outcome.candidate.version) writes closer to you than your "
                + "previous adapter. Measured on held-out samples of your own mail it beat it "
                + "on \(metric) — \(candidate) against \(incumbent)."
        }
        return "Candidate v\(outcome.candidate.version) writes worse than your active adapter. "
            + "Measured on held-out samples of your own mail it lost on \(metric) — "
            + "\(candidate) against \(incumbent). "
            + "v\(outcome.activeVersionAfter.version) remains active. Nothing changed."
    }

    private func score(_ version: AdapterVersion) -> String {
        version.evalReport?.primaryScore?.demoNumber(2) ?? "—"
    }
}

/// The retrospective verdict: the active adapter is *itself* the regression.
///
/// Deliberately a separate view rather than a mode on ``VerdictPanel``. That panel
/// says "a candidate was refused and the incumbent is untouched", which is the
/// opposite of what happened here — the worse version is the one that shipped.
/// Reusing it produced a screenshot claiming "v6 remains active. Nothing changed."
/// while the status strip said v7 was active: three false statements in one
/// paragraph, on the same screen as the truth.
public struct RegressionVerdictPanel: View {
    private let activeVersion: Int
    private let activeScore: Double
    private let bestVersion: Int
    private let bestScore: Double
    private let gapNats: Double
    private let metricName: String
    private let rolledBackTo: Int?

    @State private var shieldScale: CGFloat = 0.92

    public init(
        activeVersion: Int,
        activeScore: Double,
        bestVersion: Int,
        bestScore: Double,
        gapNats: Double,
        metricName: String,
        rolledBackTo: Int? = nil
    ) {
        self.activeVersion = activeVersion
        self.activeScore = activeScore
        self.bestVersion = bestVersion
        self.bestScore = bestScore
        self.gapNats = gapNats
        self.metricName = metricName
        self.rolledBackTo = rolledBackTo
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: Space.s) {
            HStack(spacing: Space.s) {
                Image(systemName: "checkmark.shield.fill")
                    .font(.system(size: 28))
                    .foregroundStyle(Palette.accent)
                    .scaleEffect(shieldScale)
                // Guard the degenerate case rather than asserting a regression
                // that the numbers do not support.
                Text(gapNats > 0 ? "Would not have been promoted." : "No regression to report.")
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
                Text(activeRowText)
                    .textStyle(.dataM)
                    .foregroundStyle(Palette.ink)
            }

            Text("Provisional check: recorded held-out measurements. The full promotion gate arrives with the evaluation module.")
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
        .task { withAnimation(Motion.verdictShield) { shieldScale = 1 } }
    }

    private var explanation: String {
        guard gapNats > 0 else {
            return "v\(activeVersion) is the best measured adapter in this lineage "
                + "(\(activeScore.demoNumber(2)) on \(metricName)). Nothing to roll back."
        }
        return "v\(activeVersion) measured \(gapNats.demoNumber(2)) nats worse than "
            + "v\(bestVersion) on \(metricName) — \(activeScore.demoNumber(2)) against "
            + "\(bestScore.demoNumber(2)) — and became active anyway. A gate would have "
            + "kept v\(bestVersion)."
    }

    private var activeRowText: String {
        if let rolledBackTo {
            return "Active adapter — v\(rolledBackTo) · restored"
        }
        return "Active adapter — v\(activeVersion) · the regression is still live"
    }
}
