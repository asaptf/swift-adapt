import SwiftUI

/// The persistent 48 pt strip present on all five screens (`DESIGN.md` §4).
///
/// This is where "0 B sent" lives all demo long — the claim stays on screen even
/// while the audience is looking at something else.
public struct StatusStrip: View {
    @Bindable private var state: DemoState

    public init(state: DemoState) {
        self.state = state
    }

    public var body: some View {
        HStack(spacing: 0) {
            leading
            Spacer(minLength: Space.l)
            tabs
            Spacer(minLength: Space.l)
            trailing
        }
        .padding(.horizontal, Space.l)
        .frame(height: WindowGeometry.statusStripHeight)
        .frame(maxWidth: .infinity)
        .background(Palette.bg)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Palette.border).frame(height: 1)
        }
    }

    private var leading: some View {
        HStack(spacing: Space.s) {
            Text("StyleMirror")
                .textStyle(.label)
                .foregroundStyle(Palette.ink)
            Text("active adapter: \(state.activeVersionLabel)")
                .textStyle(.dataS)
                .foregroundStyle(Palette.inkSecondary)
            // Stated, not hidden: with the scripted engine the numbers on screen
            // are staged, and nothing else on the strip would reveal it.
            if state.isUsingScriptedEngine {
                Text("scripted")
                    .textStyle(.label)
                    .foregroundStyle(Palette.dataRed)
            }
        }
        .frame(width: 320, alignment: .leading)
    }

    private var tabs: some View {
        HStack(spacing: Space.l) {
            ForEach(DemoState.Screen.allCases) { screen in
                let isActive = state.screen == screen
                Button {
                    state.screen = screen
                } label: {
                    // Mixed case per the copy deck (§8.1) — the Label style's
                    // uppercasing is reserved for section labels and chips.
                    Text(screen.tabTitle)
                        .textStyle(.button)
                        .foregroundStyle(isActive ? Palette.ink : Palette.inkTertiary)
                        .padding(.vertical, Space.xs)
                        .overlay(alignment: .bottom) {
                            Rectangle()
                                .fill(isActive ? Palette.accent : .clear)
                                .frame(height: 2)
                        }
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var trailing: some View {
        HStack(spacing: Space.s) {
            NetworkPill(status: state.networkStatus)
            ByteCounterStrip(bytesSent: state.bytesSent)
        }
        .frame(width: 320, alignment: .trailing)
    }
}
