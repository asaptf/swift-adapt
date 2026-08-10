import StyleMirrorEngine
import SwiftUI

/// Network state as a capsule (`DESIGN.md` §6.1).
///
/// Connected is rendered **neutral, not as a warning** — and offline is the
/// celebrated state, because offline is the product claim. The dot never pulses.
public struct NetworkPill: View {
    private let status: NetworkStatus

    public init(status: NetworkStatus) {
        self.status = status
    }

    private var isOffline: Bool { status == .offline }

    public var body: some View {
        HStack(spacing: Space.xs) {
            Circle()
                .fill(isOffline ? Palette.accent : Palette.inkTertiary)
                .frame(width: 8, height: 8)
            Image(systemName: isOffline ? "airplane" : "network")
                .font(.system(size: 12, weight: .semibold))
            Text(isOffline ? "Offline" : "Connected")
                .textStyle(.label)
        }
        .foregroundStyle(isOffline ? Palette.accent : Palette.inkTertiary)
        .padding(.horizontal, Space.s)
        .frame(height: 28)
        .background(Capsule().fill(isOffline ? Palette.accentWash : .clear))
        .overlay(
            Capsule().strokeBorder(isOffline ? .clear : Palette.border, lineWidth: 1)
        )
        .animation(Motion.pillCrossfade, value: isOffline)
    }
}

/// The strip form of the byte counter (`DESIGN.md` §6.2).
///
/// The nonzero branch is deliberate build-time honesty: it will never fire in the
/// demo, and it exists so the zero on screen is a measurement rather than a label.
public struct ByteCounterStrip: View {
    private let bytesSent: UInt64

    public init(bytesSent: UInt64) {
        self.bytesSent = bytesSent
    }

    public var body: some View {
        Text("\(bytesSent) B sent")
            .textStyle(.dataM)
            .foregroundStyle(bytesSent == 0 ? Palette.ink : Palette.dataRed)
            .reservingWidth(for: "0 B sent", style: .dataM, alignment: .trailing)
    }
}
