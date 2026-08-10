import AppKit
import SwiftUI

/// The type scale from `DESIGN.md` §3.
///
/// System fonts only: SF Pro (Display above 20 pt, Text below — the system font
/// switches automatically) and SF Mono for **every number that can change** and
/// every piece of data. Never a third face.
///
/// Apply with ``SwiftUI/View/textStyle(_:)`` rather than `.font(...)` so tracking,
/// kerning, and leading come along with the size.
public struct TextStyle: Sendable {
    let size: CGFloat
    let weight: Font.Weight
    let monospaced: Bool
    /// Target total line height in pt; `nil` keeps the font's natural leading.
    let leading: CGFloat?
    /// Letter spacing applied uniformly (SwiftUI `.tracking`).
    let tracking: CGFloat
    /// Letter spacing that respects font kerning pairs (SwiftUI `.kerning`).
    let kerning: CGFloat
    let uppercase: Bool

    init(
        size: CGFloat,
        weight: Font.Weight,
        monospaced: Bool = false,
        leading: CGFloat? = nil,
        tracking: CGFloat = 0,
        kerning: CGFloat = 0,
        uppercase: Bool = false
    ) {
        self.size = size
        self.weight = weight
        self.monospaced = monospaced
        self.leading = leading
        self.tracking = tracking
        self.kerning = kerning
        self.uppercase = uppercase
    }

    var font: Font {
        .system(size: size, weight: weight, design: monospaced ? .monospaced : .default)
    }

    /// Extra spacing needed to reach ``leading``, derived from the real font
    /// metrics. SwiftUI's `lineSpacing` is *additional* space, not total leading.
    var extraLineSpacing: CGFloat {
        guard let leading else { return 0 }
        let nsFont =
            monospaced
            ? NSFont.monospacedSystemFont(ofSize: size, weight: weight.nsWeight)
            : NSFont.systemFont(ofSize: size, weight: weight.nsWeight)
        let naturalLineHeight = nsFont.ascender - nsFont.descender + nsFont.leading
        return max(0, leading - naturalLineHeight)
    }

    // MARK: - The scale

    /// The hero "0" — Act 1 only.
    public static let numeralXL = TextStyle(size: 96, weight: .medium, monospaced: true)
    /// Metric tile values.
    public static let numeralL = TextStyle(size: 32, weight: .medium, monospaced: true)
    /// One statement per screen ("Offline. Nothing leaves this Mac.").
    public static let display = TextStyle(size: 40, weight: .semibold, leading: 46, tracking: -0.4)
    /// Screen titles.
    public static let title = TextStyle(size: 28, weight: .semibold, leading: 34)
    /// Card titles ("Reply A").
    public static let headline = TextStyle(size: 20, weight: .semibold, leading: 25)
    /// Email and reply text.
    public static let body = TextStyle(size: 17, weight: .regular, leading: 25)
    /// Supporting prose, checklist rows.
    public static let bodyS = TextStyle(size: 15, weight: .regular, leading: 21)
    /// All buttons.
    public static let button = TextStyle(size: 15, weight: .semibold)
    /// Section labels, tile labels, chips.
    public static let label = TextStyle(
        size: 12, weight: .semibold, kerning: 1.2, uppercase: true
    )
    /// Poisoned examples, the strip counter.
    public static let dataM = TextStyle(size: 15, weight: .regular, monospaced: true, leading: 22)
    /// Axis labels, timeline scores, captions.
    public static let dataS = TextStyle(size: 13, weight: .regular, monospaced: true)
    /// Fine print.
    public static let caption = TextStyle(size: 12, weight: .regular)
}

extension View {
    /// Applies a ``TextStyle`` — font, tracking, kerning, and computed leading.
    public func textStyle(_ style: TextStyle) -> some View {
        self
            .font(style.font)
            .tracking(style.tracking)
            .kerning(style.kerning)
            .lineSpacing(style.extraLineSpacing)
            .textCase(style.uppercase ? .uppercase : nil)
    }

    /// Reserves horizontal room for the widest string a live value can reach, so
    /// the container never resizes as digits change (§3, zero-jitter rule 2).
    ///
    /// Numeric changes must swap instantly — never animate glyphs or count up.
    public func reservingWidth(
        for widestValue: String,
        style: TextStyle,
        alignment: Alignment = .trailing
    ) -> some View {
        let width = (widestValue as NSString).size(
            withAttributes: [
                .font: style.monospaced
                    ? NSFont.monospacedSystemFont(ofSize: style.size, weight: style.weight.nsWeight)
                    : NSFont.systemFont(ofSize: style.size, weight: style.weight.nsWeight)
            ]
        ).width
        return frame(width: ceil(width), alignment: alignment)
    }
}

extension Font.Weight {
    /// Bridges to `NSFont.Weight` for metric queries.
    fileprivate var nsWeight: NSFont.Weight {
        switch self {
        case .ultraLight: .ultraLight
        case .thin: .thin
        case .light: .light
        case .regular: .regular
        case .medium: .medium
        case .semibold: .semibold
        case .bold: .bold
        case .heavy: .heavy
        case .black: .black
        default: .regular
        }
    }
}
