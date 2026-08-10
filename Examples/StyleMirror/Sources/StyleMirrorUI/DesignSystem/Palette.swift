import AppKit
import SwiftUI

/// Color tokens from `DESIGN.md` §2.
///
/// Values are exact and validated (contrast, CVD separation) — do not tune by eye.
/// Every token resolves light/dark automatically through `NSColor`'s dynamic
/// provider, so views never branch on `colorScheme`.
///
/// One accent rule (§1): green means exactly one thing — *on-device / yours /
/// protected*. Nothing else is ever green.
public enum Palette {

    // MARK: - Surfaces

    /// Window background.
    public static let bg = dynamic(dark: 0x0E_1116, light: 0xF2_F4F6)

    /// All cards and the chart plot area.
    public static let surface = dynamic(dark: 0x16_1B22, light: 0xFF_FFFF)

    /// Chips on cards, chart value tag.
    public static let surfaceRaised = dynamic(dark: 0x1C_232C, light: 0xF5_F7F9)

    /// 1 px border on all cards and fields. Depth comes from this, never shadows.
    public static let border = dynamic(dark: 0x2A_323C, light: 0xD9_DEE3)

    /// Chart gridlines only — recessive by design.
    public static let grid = dynamic(dark: 0x23_2A33, light: 0xE9_EDF0)

    // MARK: - Text

    /// Primary text. 15.8:1 on `surface` (dark), 17.8:1 (light).
    public static let ink = dynamic(dark: 0xF2_F5F7, light: 0x14_181C)

    /// Secondary text. 8.2:1 on `surface` (dark), 8.0:1 (light).
    public static let inkSecondary = dynamic(dark: 0xA9_B4BE, light: 0x49_525A)

    /// Tertiary text and labels.
    ///
    /// - Important: In light appearance this is only legible on white `surface`
    ///   (4.6:1); on `bg` it drops to 4.2:1, so use ``inkSecondary`` there (§2.2).
    public static let inkTertiary = dynamic(dark: 0x7D_8894, light: 0x6C_7680)

    // MARK: - Identity

    /// The one accent: on-device / adapter / offline / gate holding.
    ///
    /// Text-safe in both appearances — 5.2:1 on `surface` (dark), 4.8:1 (light).
    public static let accent = dynamic(dark: 0x1F_A24A, light: 0x0B_8340)

    /// Fill behind green content — accent at 12% (dark) / 10% (light).
    public static let accentWash = dynamicComponents(
        dark: (0x1F_A24A, 0.12),
        light: (0x0B_8340, 0.10)
    )

    /// Base-model identity: deliberately chromaless grey.
    ///
    /// "Generic and characterless" is the argument the demo makes, so the color
    /// makes it too (§2.3). Models get hues; the human gets ink outline only.
    public static let baseModel = dynamic(dark: 0x7D_8894, light: 0x83_8C95)

    /// A metric that failed — and the small mark beside it. Never a panel,
    /// border, or background: the rejected thing is red, the *system* is green.
    public static let dataRed = dynamic(dark: 0xFF_453A, light: 0xC6_2F35)

    /// Label color on an accent-filled button.
    public static let onAccent = dynamic(dark: 0x0E_1116, light: 0xFF_FFFF)

    // MARK: - Dynamic construction

    private static func dynamic(dark: UInt32, light: UInt32) -> Color {
        dynamicComponents(dark: (dark, 1.0), light: (light, 1.0))
    }

    private static func dynamicComponents(
        dark: (hex: UInt32, alpha: Double),
        light: (hex: UInt32, alpha: Double)
    ) -> Color {
        Color(
            nsColor: NSColor(name: nil) { appearance in
                let isDark = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua
                let token = isDark ? dark : light
                return NSColor(hex: token.hex, alpha: token.alpha)
            }
        )
    }
}

extension NSColor {
    /// Creates a color from a packed `0xRRGGBB` value in the sRGB space.
    fileprivate convenience init(hex: UInt32, alpha: Double) {
        self.init(
            srgbRed: Double((hex >> 16) & 0xFF) / 255.0,
            green: Double((hex >> 8) & 0xFF) / 255.0,
            blue: Double(hex & 0xFF) / 255.0,
            alpha: alpha
        )
    }
}
