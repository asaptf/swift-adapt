import SwiftUI

/// Spacing, radii, and window geometry from `DESIGN.md` §4.
///
/// The scale is closed: 4 · 8 · 12 · 16 · 24 · 32 · 48 · 64. Nothing off-scale.
public enum Space {
    public static let xxs: CGFloat = 4
    public static let xs: CGFloat = 8
    public static let s: CGFloat = 12
    public static let m: CGFloat = 16
    public static let l: CGFloat = 24
    public static let xl: CGFloat = 32
    public static let xxl: CGFloat = 48
    public static let xxxl: CGFloat = 64

    /// Window content padding below the status strip.
    public static let windowPadding: CGFloat = 40
    /// Padding inside every card, and the gap between cards.
    public static let card: CGFloat = 24
    /// Default gap inside a stack.
    public static let stack: CGFloat = 16
}

/// Corner radii. All continuous (`.continuous`), never circular.
public enum Radius {
    /// Cards.
    public static let card: CGFloat = 12
    /// Fields, buttons, inner panels.
    public static let control: CGFloat = 8
}

/// Fixed window geometry.
///
/// 1440 × 810 is exactly 16:9, so the screen recording needs no crop and at 2×
/// Retina (2880 × 1620) it downscales to 1080p at a clean 1.5× (§4).
public enum WindowGeometry {
    public static let width: CGFloat = 1440
    public static let height: CGFloat = 810

    /// Height of the persistent status strip present on all five screens.
    public static let statusStripHeight: CGFloat = 48

    /// Content area below the status strip: 1440 × 762.
    public static var contentHeight: CGFloat { height - statusStripHeight }

    /// Inner content after `windowPadding` on all sides: 1360 × 682.
    public static var innerWidth: CGFloat { width - Space.windowPadding * 2 }
    public static var innerHeight: CGFloat { contentHeight - Space.windowPadding * 2 }
}

extension View {
    /// Applies the standard card treatment: `surface` fill, 1 px `border`
    /// hairline, radius 12, no shadow (§4 — depth comes from surface steps).
    public func cardSurface(padding: CGFloat = Space.card) -> some View {
        self
            .padding(padding)
            .background(
                RoundedRectangle(cornerRadius: Radius.card, style: .continuous)
                    .fill(Palette.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: Radius.card, style: .continuous)
                    .strokeBorder(Palette.border, lineWidth: 1)
            )
    }
}
