import StyleMirrorEngine
import SwiftUI

/// Act 1 — the airplane-mode moment (`DESIGN.md` §4.1).
///
/// The whole screen is a three-second read with no narration: airplane,
/// "Offline", a giant green 0 above the words "bytes sent to network". Nothing
/// else is on screen, deliberately.
public struct OfflineScreen: View {
    private let state: DemoState

    /// How many of the three offline element groups have appeared. Driven by a
    /// staggered task rather than an idle animation, so the frame is static once
    /// the transition finishes (§7).
    @State private var revealed = 0

    public init(state: DemoState) {
        self.state = state
    }

    private var isOffline: Bool { state.networkStatus == .offline }

    public var body: some View {
        ZStack {
            if isOffline {
                offlineContent
            } else {
                waitingContent
                    .transition(.opacity)
            }
        }
        .frame(maxWidth: 760)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task(id: isOffline) {
            guard isOffline else {
                revealed = 0
                return
            }
            for step in 1...3 {
                withAnimation(Motion.offlineElementIn) { revealed = step }
                try? await Task.sleep(for: .seconds(Motion.offlineStagger))
            }
        }
    }

    // MARK: Before the toggle

    private var waitingContent: some View {
        VStack(spacing: 0) {
            NetworkPill(status: state.networkStatus)
            Spacer().frame(height: Space.l)
            Text("Turn on Airplane Mode.")
                .textStyle(.display)
                .foregroundStyle(Palette.ink)
            Spacer().frame(height: Space.s)
            Text("The demo starts when the network ends.")
                .textStyle(.bodyS)
                .foregroundStyle(Palette.inkSecondary)
        }
        .multilineTextAlignment(.center)
    }

    // MARK: After the toggle — the video's opening shot

    private var offlineContent: some View {
        VStack(spacing: 0) {
            ZStack {
                Circle().fill(Palette.accentWash)
                Image(systemName: "airplane")
                    .font(.system(size: 44, weight: .medium))
                    .foregroundStyle(Palette.accent)
            }
            .frame(width: 96, height: 96)
            .rising(revealed >= 1)

            Spacer().frame(height: Space.l)

            Text("Offline. Nothing leaves this Mac.")
                .textStyle(.display)
                .foregroundStyle(Palette.ink)
                .multilineTextAlignment(.center)
                .rising(revealed >= 2)

            Spacer().frame(height: Space.xxl)

            VStack(spacing: 0) {
                SectionLabel("Bytes sent to network")
                Spacer().frame(height: Space.xs)
                Text("\(state.bytesSent)")
                    .textStyle(.numeralXL)
                    .foregroundStyle(state.bytesSent == 0 ? Palette.accent : Palette.dataRed)
                Spacer().frame(height: Space.s)
                Text("since airplane mode · \(elapsedClock)")
                    .textStyle(.dataS)
                    .foregroundStyle(Palette.inkSecondary)
            }
            .rising(revealed >= 3)
        }
    }

    /// The clock proves the zero is live; the zero itself never animates (§4.1).
    private var elapsedClock: String {
        let total = state.secondsOffline
        return String(format: "%02d:%02d", total / 60, total % 60)
    }
}

extension View {
    /// Fade in while rising, the offline screen's only entrance (§7).
    fileprivate func rising(_ shown: Bool) -> some View {
        opacity(shown ? 1 : 0)
            .offset(y: shown ? 0 : Motion.riseDistance)
    }
}
