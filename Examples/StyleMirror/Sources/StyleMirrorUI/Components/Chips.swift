import StyleMirrorEngine
import SwiftUI

/// A neutral data capsule — the Act 2 corpus stats (`DESIGN.md` §4.2).
public struct StatChip: View {
    private let text: String

    public init(_ text: String) {
        self.text = text
    }

    public var body: some View {
        Text(text)
            .textStyle(.dataM)
            .foregroundStyle(Palette.inkSecondary)
            .padding(.horizontal, Space.s)
            .frame(height: 28)
            .background(Capsule().fill(Palette.surfaceRaised))
    }
}

/// A caps section label (`DESIGN.md` §3, Label style).
public struct SectionLabel: View {
    private let text: String
    private let color: Color

    public init(_ text: String, color: Color = Palette.inkTertiary) {
        self.text = text
        self.color = color
    }

    public var body: some View {
        Text(text)
            .textStyle(.label)
            .foregroundStyle(color)
    }
}

/// Post-reveal identity chip for a blind-test card (`DESIGN.md` §6.6).
///
/// Models get hues, the human gets ink: base model wears a grey wash, the
/// adapter an accent wash, and the human an ink outline with no fill at all.
public struct IdentityChip: View {
    private let role: ReplyRole
    private let activeVersionLabel: String

    public init(role: ReplyRole, activeVersionLabel: String) {
        self.role = role
        self.activeVersionLabel = activeVersionLabel
    }

    public var body: some View {
        HStack(spacing: Space.xxs) {
            if role == .human {
                Image(systemName: "person.fill")
                    .font(.system(size: 10, weight: .semibold))
            }
            Text(title)
                .textStyle(.label)
                .kerning(0.8)
        }
        .foregroundStyle(foreground)
        .padding(.horizontal, 10)
        .frame(height: 24)
        .background(Capsule().fill(fill))
        .overlay(
            Capsule().strokeBorder(
                role == .human ? Palette.ink : .clear,
                lineWidth: role == .human ? 1.5 : 0
            )
        )
    }

    private var title: String {
        switch role {
        case .baseModel: "Base model"
        case .adaptedModel: "Adapter \(activeVersionLabel)"
        case .human: "Human"
        }
    }

    private var foreground: Color {
        switch role {
        case .baseModel: Palette.inkSecondary
        case .adaptedModel: Palette.accent
        case .human: Palette.ink
        }
    }

    private var fill: Color {
        switch role {
        case .baseModel: Palette.baseModel.opacity(0.14)
        case .adaptedModel: Palette.accentWash
        case .human: .clear
        }
    }
}

/// The blind-test scoreboard (`DESIGN.md` §6.7).
///
/// The count is the one and only count-up in the app, because it is the app's
/// scoreboard — everything else swaps instantly to avoid flicker on capture.
public struct TallyChip: View {
    private let tally: BlindTestTally

    public init(tally: BlindTestTally) {
        self.tally = tally
    }

    public var body: some View {
        Text("adapter picked as human · \(tally.adapterMistakenForHuman) of \(tally.roundsPlayed)")
            .textStyle(.dataM)
            .foregroundStyle(Palette.ink)
            .contentTransition(.numericText())
            .padding(.horizontal, Space.s)
            .frame(height: 28)
            .background(Capsule().fill(Palette.surfaceRaised))
            .animation(Motion.resultLine, value: tally.adapterMistakenForHuman)
    }
}
