import StyleMirrorEngine
import SwiftUI

/// The window: persistent status strip above one of five screens.
///
/// Screens crossfade, never slide (`DESIGN.md` §7), and the presenter's controls
/// are all keyboard: ⌘1–⌘5 pick a screen, Space fires the current screen's
/// primary action, ⌘⇧R resets, ⌘⌥F toggles the fast training pass.
public struct RootView: View {
    private let state: DemoState

    public init(state: DemoState) {
        self.state = state
    }

    public var body: some View {
        VStack(spacing: 0) {
            StatusStrip(state: state)
            screen
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(Space.windowPadding)
                .background(Palette.bg)
        }
        .frame(width: WindowGeometry.width, height: WindowGeometry.height)
        .background(Palette.bg)
        .task { await state.start() }
        // Focusable so the window receives Space when no text field owns focus;
        // a focused paste field still consumes it first (§5).
        .focusable()
        .focusEffectDisabled()
        .onKeyPress(.space) {
            // `onKeyPress` respects the focus chain, so a focused paste field
            // consumes the space instead of starting a training run.
            Task { await state.firePrimaryAction() }
            return .handled
        }
        .background(shortcuts)
    }

    @ViewBuilder private var screen: some View {
        ZStack {
            switch state.screen {
            case .offline:
                OfflineScreen(state: state)
            case .train:
                TrainScreen(state: state)
            case .blindTest:
                BlindTestScreen(state: state)
            case .languages:
                LanguagesScreen(state: state)
            case .gate:
                GateScreen(state: state)
            }
        }
        .animation(Motion.screenSwitch, value: state.screen)
    }

    /// Zero-size buttons carrying the ⌘ shortcuts. Modified keys cannot collide
    /// with text entry, so these are safe as global bindings.
    private var shortcuts: some View {
        VStack(spacing: 0) {
            ForEach(DemoState.Screen.allCases) { target in
                Button("") { state.screen = target }
                    .keyboardShortcut(
                        KeyEquivalent(Character("\(target.rawValue + 1)")),
                        modifiers: .command
                    )
            }
            Button("") { Task { await state.reset() } }
                .keyboardShortcut("r", modifiers: [.command, .shift])
            Button("") { state.toggleFastMode() }
                .keyboardShortcut("f", modifiers: [.command, .option])
        }
        .frame(width: 0, height: 0)
        .opacity(0)
        .accessibilityHidden(true)
    }
}

/// A centered secondary line, used for every empty state (`DESIGN.md` §8.7).
///
/// No illustrations: an empty state is only ever visible when the demo is run
/// out of order, and it should read as an instruction, not a decoration.
struct EmptyStateMessage: View {
    let text: String

    var body: some View {
        Text(text)
            .textStyle(.bodyS)
            .foregroundStyle(Palette.inkSecondary)
            .multilineTextAlignment(.center)
            .frame(maxWidth: 520)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
