import Charts
import SwiftUI

/// The Act 2 centerpiece — it must reward three minutes of staring
/// (`DESIGN.md` §6.3).
///
/// Interpolation is monotone rather than a spline on purpose: an overshooting
/// loss curve is a lie. Axes are furniture — gridlines only, no strokes or ticks.
public struct LossChart: View {
    private let points: [LossPoint]
    private let totalSteps: Int

    /// Diameter-derived symbol area for the live head, animated one tick per new
    /// point — a heartbeat at data rate, never an idle pulse (§7).
    @State private var headSize: CGFloat = 38

    public init(points: [LossPoint], totalSteps: Int) {
        self.points = points
        self.totalSteps = totalSteps
    }

    private var head: LossPoint? { points.last }

    /// Exponential moving average of the per-step loss.
    ///
    /// Real training at batch size 1 is violently noisy — each step sees one
    /// example, so raw loss swings across the whole axis and the trend is
    /// invisible. The scripted engine's tidy decay was a fiction; this is what
    /// the data actually looks like. So the raw series stays (faint, because
    /// hiding it would be a lie about the measurement) and the trend is drawn
    /// over it, labelled as smoothed.
    private var smoothed: [LossPoint] {
        guard points.count > 1 else { return points }
        let alpha = 0.2
        var value = points[0].loss
        return points.map { point in
            value += alpha * (point.loss - value)
            return LossPoint(
                id: point.id,
                step: point.step,
                loss: value,
                validationLoss: point.validationLoss
            )
        }
    }

    /// Label stride chosen from the run length, aiming for four to six labels.
    ///
    /// A fixed stride only suits one run length: at 40 steps a stride of 50
    /// leaves a single "0" on the axis, which is how this was first found.
    private var xAxisValues: [Int] {
        let target = max(1, totalSteps / 5)
        let stride = [1, 2, 5, 10, 25, 50, 100, 250, 500].first { $0 >= target } ?? 1000
        return Array(Swift.stride(from: 0, through: max(totalSteps, 1), by: stride))
    }

    public var body: some View {
        Chart {
            ForEach(points) { point in
                AreaMark(
                    x: .value("Step", point.step),
                    y: .value("Loss", point.loss)
                )
                .interpolationMethod(.monotone)
                .foregroundStyle(
                    .linearGradient(
                        colors: [Palette.accent.opacity(0.07), Palette.accent.opacity(0)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )

                LineMark(
                    x: .value("Step", point.step),
                    y: .value("Loss", point.loss),
                    series: .value("Series", "raw")
                )
                .interpolationMethod(.monotone)
                .lineStyle(StrokeStyle(lineWidth: 1, lineCap: .round, lineJoin: .round))
                .foregroundStyle(Palette.accent.opacity(0.25))
            }

            ForEach(smoothed) { point in
                LineMark(
                    x: .value("Step", point.step),
                    y: .value("Loss", point.loss),
                    series: .value("Series", "trend")
                )
                .interpolationMethod(.monotone)
                .lineStyle(StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round))
                .foregroundStyle(Palette.accent)
            }

            if let head = smoothed.last {
                PointMark(
                    x: .value("Step", head.step),
                    y: .value("Loss", head.loss)
                )
                .symbolSize(headSize)
                .foregroundStyle(Palette.accent)
                .annotation(position: .topTrailing, spacing: Space.xs) {
                    valueTag(head.loss)
                }
            }
        }
        .chartXScale(domain: 0...max(totalSteps, 1))
        .chartYScale(domain: 0...4.0)
        .chartXAxis {
            AxisMarks(values: xAxisValues) { value in
                AxisValueLabel {
                    Text(value.as(Int.self).map(String.init) ?? "")
                        .textStyle(.dataS)
                        .foregroundStyle(Palette.inkTertiary)
                }
            }
        }
        .chartYAxis {
            AxisMarks(values: .stride(by: 1.0)) { value in
                AxisGridLine().foregroundStyle(Palette.grid)
                AxisValueLabel {
                    Text(value.as(Double.self).map { String(format: "%.1f", $0) } ?? "")
                        .textStyle(.dataS)
                        .foregroundStyle(Palette.inkTertiary)
                }
            }
        }
        .chartPlotStyle { plot in
            plot.padding(.init(top: Space.m, leading: 0, bottom: 0, trailing: Space.m))
        }
        .animation(Motion.chartAppend, value: points.count)
        .task(id: points.count) {
            guard head != nil else { return }
            withAnimation(Motion.chartHeadTick) { headSize = 50 }
            try? await Task.sleep(for: .milliseconds(150))
            withAnimation(Motion.chartHeadTick) { headSize = 38 }
        }
    }

    /// Fixed-width so it never resizes; it moves only when the head moves (§6.3).
    private func valueTag(_ loss: Double) -> some View {
        Text(String(format: "%.2f", loss))
            .textStyle(.dataS)
            .foregroundStyle(Palette.ink)
            .reservingWidth(for: "0.00", style: .dataS, alignment: .center)
            .padding(.horizontal, Space.xs)
            .padding(.vertical, Space.xxs)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Palette.surfaceRaised)
            )
    }
}
