import StyleMirrorEngine
import SwiftUI

/// One blind-test candidate (`DESIGN.md` §6.6).
///
/// Before the reveal the three cards are **pixel-identical** — same type, same
/// length class, no per-role styling. Any visual difference would leak the answer
/// to the audience, which would quietly destroy the whole test.
public struct ReplyCard: View {
    private let title: String
    private let body_: String
    private let isPicked: Bool
    private let revealedRole: ReplyRole?
    private let activeVersionLabel: String
    private let chipVisible: Bool
    private let onPick: () -> Void

    public init(
        title: String,
        body: String,
        isPicked: Bool,
        revealedRole: ReplyRole?,
        activeVersionLabel: String,
        chipVisible: Bool,
        onPick: @escaping () -> Void
    ) {
        self.title = title
        self.body_ = body
        self.isPicked = isPicked
        self.revealedRole = revealedRole
        self.activeVersionLabel = activeVersionLabel
        self.chipVisible = chipVisible
        self.onPick = onPick
    }

    private var isRevealed: Bool { revealedRole != nil }

    public var body: some View {
        VStack(alignment: .leading, spacing: Space.m) {
            HStack {
                Text(title)
                    .textStyle(.headline)
                    .foregroundStyle(Palette.ink)
                Spacer()
                if let revealedRole {
                    IdentityChip(role: revealedRole, activeVersionLabel: activeVersionLabel)
                        .opacity(chipVisible ? 1 : 0)
                        .offset(y: chipVisible ? 0 : Motion.chipRise)
                }
            }

            Text(body_)
                .textStyle(.body)
                .foregroundStyle(Palette.ink)
                .lineLimit(10)
                .fixedSize(horizontal: false, vertical: true)

            Spacer(minLength: Space.m)

            if !isRevealed {
                Button("This is the human", action: onPick)
                    .buttonStyle(QuietButtonStyle(isSelected: isPicked))
                    .transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Space.card)
        .background(
            RoundedRectangle(cornerRadius: Radius.card, style: .continuous)
                .fill(Palette.surface)
        )
        .overlay(
            RoundedRectangle(cornerRadius: Radius.card, style: .continuous)
                .strokeBorder(borderColor, lineWidth: borderWidth)
        )
        // Width is the card's *outer* size: applying it before the padding would
        // make each card 437 + 48 pt and overflow the 1360 pt content box.
        .frame(width: 437)
        .frame(minHeight: 400, alignment: .top)
        .animation(Motion.pick, value: isPicked)
        .animation(Motion.revealBorder, value: isRevealed)
    }

    /// Post-reveal the human card wears ink and the adapter card wears accent —
    /// models get hues, the person gets ink (§2.3).
    private var borderColor: Color {
        if let revealedRole {
            switch revealedRole {
            case .human: return Palette.ink
            case .adaptedModel: return Palette.accent
            case .baseModel: return Palette.border
            }
        }
        return isPicked ? Palette.ink : Palette.border
    }

    private var borderWidth: CGFloat {
        if let revealedRole {
            return revealedRole == .baseModel ? 1 : 2
        }
        return isPicked ? 2 : 1
    }
}
