import SwiftUI

/// One live metric (`DESIGN.md` §6.4).
///
/// Values obey the zero-jitter rules: monospaced digits in a container sized for
/// the widest string, and swaps are instant — a number sliding at 2 Hz reads as
/// flicker on 60 fps capture.
public struct MetricTile: View {
    private let label: String
    private let value: String
    private let unit: String?
    /// Widest string this tile's value can reach, used to reserve layout.
    private let widestValue: String

    public init(label: String, value: String, unit: String? = nil, widestValue: String) {
        self.label = label
        self.value = value
        self.unit = unit
        self.widestValue = widestValue
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            SectionLabel(label)
            Spacer(minLength: Space.xs)
            HStack(alignment: .firstTextBaseline, spacing: Space.xs) {
                Text(value)
                    .textStyle(.numeralL)
                    .foregroundStyle(Palette.ink)
                    .reservingWidth(for: widestValue, style: .numeralL, alignment: .leading)
                if let unit {
                    Text(unit)
                        .textStyle(.bodyS)
                        .foregroundStyle(Palette.inkSecondary)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardSurface()
        .frame(height: 113)
    }
}
