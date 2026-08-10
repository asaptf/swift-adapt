import AdaptCore
import SwiftUI

/// A candidate the gate refused, plotted at its measured value (`DESIGN.md` §6.5).
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
/// Plots whatever metric the registry actually recorded on each version, read
/// from `EvalReport`. The values are held-out cross-entropy in nats per token:
/// a real measurement on mail the adapter never trained on, not a 0–100 score
/// nobody computes.
///
/// The axis increases upward, so a "lower is better" metric produces the
/// familiar falling curve and a "higher is better" one produces a rising line —
/// the same geometry serves both, and only the caption changes.
public struct VersionTimeline: View {
    private let versions: [AdapterVersion]
    private let activeVersion: Int?
    private let rejected: RejectedNode?

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

    // MARK: Legend

    private var legend: some View {
        VStack(alignment: .leading, spacing: Space.xs) {
            SectionLabel("Overnight runs")
            Text("7 nights · unattended · on battery")
                .textStyle(.caption)
                .foregroundStyle(Palette.inkTertiary)
            Text(metricCaption)
                .textStyle(.caption)
                .foregroundStyle(Palette.inkTertiary)
        }
        .frame(width: 240, alignment: .leading)
    }

    /// Names the measurement rather than asserting a score. If the registry
    /// recorded no metric there is nothing honest to claim, so say that.
    private var metricCaption: String {
        guard let report = versions.compactMap(\.evalReport).last,
            report.primaryScore != nil
        else {
            return "not yet measured"
        }
        // Name the metric the registry recorded rather than assuming it is a
        // loss: with a higher-is-better score, "held-out loss · higher is
        // better" is a contradiction the screen would state confidently.
        let name = report.primaryMetric?.replacingOccurrences(of: "_", with: " ")
            ?? "measured score"
        let direction =
            report.primaryDirection == .higherIsBetter ? "higher is better" : "lower is better"
        return "\(name) on unseen mail · \(direction)"
    }

    // MARK: Plot

    private var plot: some View {
        GeometryReader { geo in
            let slots = versions.count + (rejected == nil ? 0 : 1)
            let positions = nodePositions(in: geo.size, slots: slots)

            ZStack(alignment: .topLeading) {
                Path { path in
                    for (index, point) in positions.promoted.enumerated() {
                        index == 0 ? path.move(to: point) : path.addLine(to: point)
                    }
                }
                .stroke(Palette.accent.opacity(0.5), lineWidth: 2)

                if let last = positions.promoted.last, let rejectedPoint = positions.rejected {
                    Path { path in
                        path.move(to: last)
                        path.addLine(to: rejectedPoint)
                    }
                    .stroke(Palette.inkTertiary, style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                }

                ForEach(Array(versions.enumerated()), id: \.element.version) { index, version in
                    if index < positions.promoted.count {
                        node(
                            at: positions.promoted[index],
                            label: "v\(version.version)",
                            value: value(of: version),
                            isActive: version.version == activeVersion
                        )
                    }
                }

                if let rejected, let point = positions.rejected {
                    rejectedNode(at: point, node: rejected)
                }
            }
        }
    }

    private func node(at point: CGPoint, label: String, value: Double?, isActive: Bool) -> some View {
        VStack(spacing: Space.xxs) {
            Text(label)
                .textStyle(.dataS)
                .foregroundStyle(isActive ? Palette.accent : Palette.inkTertiary)
            ZStack {
                Circle()
                    .fill(isActive ? Palette.accent : Palette.accent.opacity(0.45))
                    .frame(width: isActive ? 12 : 8, height: isActive ? 12 : 8)
                if isActive {
                    // Static radar-blip emphasis — it never pulses (§6.5).
                    Circle()
                        .strokeBorder(Palette.accent, lineWidth: 2)
                        .frame(width: 20, height: 20)
                }
            }
            .frame(height: 20)
            Text(value.map { $0.demoNumber(2) } ?? "—")
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
            Text(node.score.demoNumber(2))
                .textStyle(.dataS)
                .foregroundStyle(Palette.dataRed)
            Text("not promoted")
                .textStyle(.caption)
                .foregroundStyle(Palette.inkTertiary)
        }
        .position(x: point.x, y: point.y)
    }

    // MARK: Geometry

    /// Range covering every plotted value, padded so nodes never sit on the edge.
    ///
    /// Derived from the data rather than hardcoded: the values depend on the model
    /// and corpus, and a fixed window would clip or flatten a real measurement.
    private var valueRange: ClosedRange<Double> {
        var values = versions.compactMap(value(of:))
        if let rejected { values.append(rejected.score) }
        guard let low = values.min(), let high = values.max() else { return 0...1 }
        guard high > low else { return (low - 0.5)...(high + 0.5) }
        let padding = (high - low) * 0.15
        return (low - padding)...(high + padding)
    }

    private func nodePositions(
        in size: CGSize,
        slots: Int
    ) -> (promoted: [CGPoint], rejected: CGPoint?) {
        guard slots > 0 else { return ([], nil) }
        let plotHeight = max(size.height - labelInset * 2, 1)
        let range = valueRange
        let span = range.upperBound - range.lowerBound

        func x(_ index: Int) -> CGFloat {
            size.width * (CGFloat(index) + 0.5) / CGFloat(slots)
        }
        // Value increases upward, so a falling loss reads as a falling line.
        func y(_ value: Double) -> CGFloat {
            let clamped = min(max(value, range.lowerBound), range.upperBound)
            let fraction = span > 0 ? (clamped - range.lowerBound) / span : 0.5
            return labelInset + plotHeight * (1 - CGFloat(fraction))
        }

        let midpoint = range.lowerBound + span / 2
        let promoted = versions.enumerated().map { index, version in
            CGPoint(x: x(index), y: y(value(of: version) ?? midpoint))
        }
        let rejectedPoint = rejected.map { CGPoint(x: x(versions.count), y: y($0.score)) }
        return (promoted, rejectedPoint)
    }

    private func value(of version: AdapterVersion) -> Double? {
        version.evalReport?.primaryScore
    }
}
