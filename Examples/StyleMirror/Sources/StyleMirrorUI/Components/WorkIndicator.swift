import SwiftUI

/// Progress treatment for an operation that genuinely takes a while — generating
/// several replies on-device, for instance.
///
/// `DESIGN.md` §7 forbids anything that pulses **while idle**, so that the frame
/// is pixel-static between data events and the encoder spends bits on the moments
/// that matter. This is not that: work is in progress, and the elapsed seconds are
/// real data. Idle decoration and a progress report are different things, and the
/// spec has been amended to say so.
///
/// Deliberately restrained: one hairline sweep and a counting clock. No spinner
/// with a shadow, no bouncing dots.
public struct WorkIndicator: View {
    private let message: String
    /// Number of completed and total units when the caller knows them; a bare
    /// elapsed clock otherwise, because inventing a percentage would be a lie.
    private let completed: Int?
    private let total: Int?

    @State private var elapsed: Int = 0
    @State private var sweep: CGFloat = -0.3

    public init(message: String, completed: Int? = nil, total: Int? = nil) {
        self.message = message
        self.completed = completed
        self.total = total
    }

    public var body: some View {
        VStack(spacing: Space.m) {
            Text(message)
                .textStyle(.bodyS)
                .foregroundStyle(Palette.inkSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 520)

            track

            Text(statusLine)
                .textStyle(.dataS)
                .foregroundStyle(Palette.inkTertiary)
                .reservingWidth(for: "step 8 of 8 · 888 s", style: .dataS, alignment: .center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .task {
            // One clock, one sweep; both stop when the view goes away.
            withAnimation(.linear(duration: 1.6).repeatForever(autoreverses: false)) {
                sweep = 1.0
            }
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                elapsed += 1
            }
        }
    }

    private var track: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Palette.grid)
                Capsule()
                    .fill(Palette.accent)
                    .frame(width: geo.size.width * fillFraction)
                    .offset(x: total == nil ? geo.size.width * sweep : 0)
            }
            .clipShape(Capsule())
        }
        .frame(width: 320, height: 3)
    }

    /// Determinate when the caller knows the unit count, otherwise a short sweep
    /// that conveys motion without claiming a proportion.
    private var fillFraction: CGFloat {
        guard let total, total > 0, let completed else { return 0.3 }
        return CGFloat(completed) / CGFloat(total)
    }

    private var statusLine: String {
        if let total, let completed {
            return "step \(completed) of \(total) · \(elapsed) s"
        }
        return "\(elapsed) s"
    }
}
