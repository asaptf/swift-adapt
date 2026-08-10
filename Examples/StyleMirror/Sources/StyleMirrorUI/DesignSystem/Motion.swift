import SwiftUI

/// Animation tokens from `DESIGN.md` §7.
///
/// Global rules: animate **transform and opacity only**; every spring is
/// critically damped (`dampingFraction: 1.0`) so nothing overshoots; nothing
/// loops while idle, so between data events the frame is pixel-static and the
/// video encoder spends bits on the moments that matter.
public enum Motion {

    /// The house curve — "ease-out".
    public static let easeOut = Animation.timingCurve(0.2, 0.8, 0.2, 1, duration: 0.25)

    /// Ease-out at an explicit duration.
    public static func easeOut(_ duration: Double) -> Animation {
        .timingCurve(0.2, 0.8, 0.2, 1, duration: duration)
    }

    // MARK: Act 1 — network drop

    /// Pill crossfade to OFFLINE.
    public static let pillCrossfade = easeOut(0.25)
    /// Pre-toggle content fading out.
    public static let preStateOut = easeOut(0.20)
    /// Each offline element fading in while rising 12 pt.
    public static let offlineElementIn = easeOut(0.45)
    /// Stagger between symbol → headline → counter.
    public static let offlineStagger: Double = 0.08
    /// Distance offline elements rise as they appear.
    public static let riseDistance: CGFloat = 12

    // MARK: Act 2 — chart & promotion

    /// The line extending to a new point. Linear: data arrives at a steady rate.
    public static let chartAppend = Animation.linear(duration: 0.25)
    /// The live head's heartbeat, 1.0 → 1.15 → 1.0, at data rate — not an idle pulse.
    public static let chartHeadTick = Animation.easeInOut(duration: 0.15)
    /// Promotion row fading in under the chart header.
    public static let promotionRow = easeOut(0.30)
    /// The eighth timeline node scaling 0.6 → 1.
    public static let timelineNode = Animation.spring(response: 0.35, dampingFraction: 1.0)

    // MARK: Act 3 — pick & reveal

    /// Border and button fill states when a card is picked.
    public static let pick = easeOut(0.15)
    /// Quiet buttons fading out at reveal.
    public static let revealButtonsOut = easeOut(0.20)
    /// Identity chips fading in while rising 8 pt.
    public static let revealChip = easeOut(0.35)
    /// Stagger between chips, left to right.
    public static let revealChipStagger: Double = 0.12
    /// Distance chips rise as they appear.
    public static let chipRise: CGFloat = 8
    /// Card borders fading in after the chips.
    public static let revealBorder = easeOut(0.40)
    /// Result line, and the app's only count-up (the tally is the scoreboard).
    public static let resultLine = easeOut(0.30)

    // MARK: Gate

    /// Each checklist row fading in while rising 6 pt as its check resolves.
    public static let checklistRow = easeOut(0.25)
    /// Minimum gap between checklist rows resolving.
    public static let checklistRowGap: Duration = .milliseconds(600)
    /// Distance checklist rows rise as they appear.
    public static let checklistRise: CGFloat = 6

    /// Stillness held after the last check, before the verdict appears.
    ///
    /// - Important: Load-bearing. Hesitation reads as deliberation, and the calm
    ///   that follows reads as confidence. Do not shorten it (§7).
    public static let verdictHold: Duration = .milliseconds(800)

    /// Verdict panel fading in while rising 12 pt.
    public static let verdictPanel = easeOut(0.50)
    /// The shield scaling 0.92 → 1 as the verdict lands.
    public static let verdictShield = Animation.spring(response: 0.40, dampingFraction: 1.0)

    // MARK: Navigation

    /// Screen switch — crossfade, never a slide.
    public static let screenSwitch = easeOut(0.25)
}
