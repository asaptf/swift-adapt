import AdaptCore
import SwiftUI

/// A candidate the gate refused, plotted at its true score (`DESIGN.md` §6.5).
public struct RejectedNode: Equatable, Sendable {
    public let versionLabel: String
    public let score: Double

    public init(versionLabel: String, score: Double) {
        self.versionLabel = versionLabel
        self.score = score
    }
}

/// The "seven nights" timeline (`DESIGN.md` §6.5).
///
/// Eval score is mapped over 55–80 so the polyline visibly *ascends* — the rising
/// line is the story, and it has to read from meters away.
public struct VersionTimeline: View {
    private let versions: [AdapterVersion]
    private let activeVersion: Int?
    private let rejected: RejectedNode?

    /// Score window the plot maps onto its height.
    private let scoreRange: ClosedRange<Double> = 55...80
    /// Vertical room reserved for the labels above and below each node.
    private let labelInset: CGFloat = 22

    public init(
        versions: [AdapterVersion],
        activeVersion: Int?,
        rejected: RejectedNode? = nil
    ) {
        self.versions = versions
        self.activeVersion = activeVersion
        self.rejected = rejected
    }

    public var body: some View {
        HStack(spacing: Space.l) {
            legend
            plot
        }
        .cardSurface()
        .frame(height: 158)
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            SectionLabel("Overnight runs")
            Text("7 nights · unattended · on battery")
                .textStyle(.caption)
                .foregroundStyle(Palette.inkTertiary)
            Text("score = style match on held-out mail")
                .textStyle(.caption)
                .foregroundStyle(Palette.inkTertiary)
        }
        .frame(width: 240, alignment: .leading)
    }

    private var plot: some View {
        GeometryReader { geo in
            let slots = versions.count + (rejected == nil ? 0 : 1)
            let positions = nodePositions(in: geo.size, slots: slots)

            ZStack(alignment: .topLeading) {
                // Connecting polyline through the promoted versions only.
                Path { path in
                    for (index, point) in positions.promoted.enumerated() {
                        index == 0 ? path.move(to: point) : path.addLine(to: point)
                    }
                }
                .stroke(Palette.accent.opacity(0.5), lineWidth: 2)

                // Dashed connector out to the refused candidate.
                if let last = positions.promoted.last, let rejectedPoint = positions.rejected {
                    Path { path in
                        path.move(to: last)
                        path.addLine(to: rejectedPoint)
                    }
                    .stroke(
                        Palette.inkTertiary,
                        style: StrokeStyle(lineWidth: 1, dash: [3, 3])
                    )
                }

                ForEach(Array(versions.enumerated()), id: \.element.version) { index, version in
                    node(
                        at: positions.promoted[index],
                        label: "v\(version.version)",
                        score: score(of: version),
                        isActive: version.version == activeVersion
                    )
                }

                if let rejected, let point = positions.rejected {
                    rejectedNode(at: point, node: rejected)
                }
            }
        }
    }

    // MARK: Nodes

    private func node(at point: CGPoint, label: String, score: Double?, isActive: Bool) -> some View {
        let tint = isActive ? Palette.accent : Palette.accent.opacity(0.45)
        return VStack(spacing: Space.xxs) {
            Text(label)
                .textStyle(.dataS)
                .foregroundStyle(isActive ? Palette.accent : Palette.inkTertiary)
            ZStack {
                Circle()
                    .fill(tint)
                    .frame(width: isActive ? 12 : 8, height: isActive ? 12 : 8)
                if isActive {
                    // Static radar-blip emphasis — it never pulses (§6.5).
                    Circle()
                        .strokeBorder(Palette.accent, lineWidth: 2)
                        .frame(width: 20, height: 20)
                }
            }
            .frame(height: 20)
            Text(score.map { String(format: "%.0f", $0) } ?? "—")
                .textStyle(.dataS)
                .foregroundStyle(isActive ? Palette.accent : Palette.inkSecondary)
        }
        .position(x: point.x, y: point.y)
        .transition(.scale(scale: 0.6).combined(with: .opacity))
    }

    private func rejectedNode(at point: CGPoint, node: RejectedNode) -> some View {
        VStack(spacing: Space.xxs) {
            Text(node.versionLabel)
                .textStyle(.dataS)
                .foregroundStyle(Palette.inkTertiary)
            Circle()
                .strokeBorder(Palette.dataRed, lineWidth: 1.5)
                .frame(width: 8, height: 8)
                .frame(height: 20)
            Text(String(format: "%.0f", node.score))
                .textStyle(.dataS)
                .foregroundStyle(Palette.dataRed)
            Text("not promoted")
                .textStyle(.caption)
                .foregroundStyle(Palette.inkTertiary)
        }
        .position(x: point.x, y: point.y)
    }

    // MARK: Geometry

    private func nodePositions(
        in size: CGSize,
        slots: Int
    ) -> (promoted: [CGPoint], rejected: CGPoint?) {
        guard slots > 0 else { return ([], nil) }
        let plotHeight = max(size.height - labelInset * 2, 1)

        func x(_ index: Int) -> CGFloat {
            size.width * (CGFloat(index) + 0.5) / CGFloat(slots)
        }
        func y(_ score: Double) -> CGFloat {
            let clamped = min(max(score, scoreRange.lowerBound), scoreRange.upperBound)
            let fraction =
                (clamped - scoreRange.lowerBound)
                / (scoreRange.upperBound - scoreRange.lowerBound)
            return labelInset + plotHeight * (1 - CGFloat(fraction))
        }

        let promoted = versions.enumerated().map { index, version in
            CGPoint(x: x(index), y: y(score(of: version) ?? scoreRange.lowerBound))
        }
        let rejectedPoint = rejected.map { CGPoint(x: x(versions.count), y: y($0.score)) }
        return (promoted, rejectedPoint)
    }

    private func score(of version: AdapterVersion) -> Double? {
        version.evalReport?.primaryScore
    }
}
