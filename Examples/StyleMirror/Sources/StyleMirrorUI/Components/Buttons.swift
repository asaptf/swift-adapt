import SwiftUI

/// Primary action button — accent fill, 44 pt tall (`DESIGN.md` §6.8).
///
/// Disabled state uses `surfaceRaised` with a tertiary label rather than a
/// dimmed accent, so a disabled button never reads as a faded green claim.
public struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    public init() {}

    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .textStyle(.button)
            .foregroundStyle(isEnabled ? Palette.onAccent : Palette.inkTertiary)
            .frame(height: 44)
            .padding(.horizontal, 20)
            .background(
                RoundedRectangle(cornerRadius: Radius.control, style: .continuous)
                    .fill(isEnabled ? Palette.accent : Palette.surfaceRaised)
            )
            .opacity(configuration.isPressed ? 0.85 : 1)
            .animation(Motion.pick, value: configuration.isPressed)
    }
}

/// Quiet button — hairline border, no fill (`DESIGN.md` §6.8).
///
/// There is deliberately no destructive style: nothing in this app destroys
/// anything.
public struct QuietButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    @State private var isHovering = false

    /// When true the button fills with ink and its label flips to `bg` — the
    /// "picked" state of a blind-test card (§6.6).
    private let isSelected: Bool

    public init(isSelected: Bool = false) {
        self.isSelected = isSelected
    }

    public func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .textStyle(.button)
            .foregroundStyle(isSelected ? Palette.bg : Palette.ink)
            .frame(height: 36)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: Radius.control, style: .continuous)
                    .fill(fill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Radius.control, style: .continuous)
                    .strokeBorder(isSelected ? .clear : Palette.border, lineWidth: 1)
            )
            .opacity(isEnabled ? 1 : 0.4)
            .onHover { isHovering = $0 }
            .animation(Motion.pick, value: isSelected)
            .animation(Motion.pick, value: isHovering)
    }

    private var fill: Color {
        if isSelected { return Palette.ink }
        return isHovering ? Palette.surfaceRaised : .clear
    }
}
