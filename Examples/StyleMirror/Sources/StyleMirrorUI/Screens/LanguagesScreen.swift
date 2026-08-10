import StyleMirrorEngine
import SwiftUI

/// Code-switching — one adapter, three languages (`DESIGN.md` §4.4).
///
/// The hierarchy does the arguing: the base reply is dimmer and smaller than the
/// adapter reply. Columns stack vertically rather than forming a grid of nine
/// cells, which keeps it from reading as a spreadsheet.
///
/// The scene no longer claims the adapter learned the voice in every language.
/// Measured: English holds, Spanish is coherent only with a repetition penalty,
/// Russian degenerates under every sampling setting tried. That is a capacity
/// limit of a rank-8 adapter over a corpus that is ~20% per non-English language,
/// so the screen shows the real output and names the boundary instead.
public struct LanguagesScreen: View {
    private let state: DemoState

    public init(state: DemoState) {
        self.state = state
    }

    public var body: some View {
        Group {
            if state.activeVersion == nil {
                EmptyStateMessage(
                    text: "Train an adapter first. This screen compares it against the base model in three languages."
                )
            } else if let result = state.codeSwitch, !result.languages.isEmpty {
                content(result)
            } else if state.codeSwitch != nil {
                // An empty result is a failure the engine did not report. Never
                // print the argument over nothing.
                EmptyStateMessage(
                    text: "Nothing to compare — \(state.unavailableReason ?? "the engine reported no languages for this request")."
                )
            } else {
                WorkIndicator(
                    message: "Answering the same request in three languages, base and adapter.",
                    completed: state.workProgress?.completed,
                    total: state.workProgress?.total
                )
            }
        }
        .task { await state.loadCodeSwitching() }
    }

    private func content(_ result: CodeSwitchResult) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Code-switching.")
                .textStyle(.title)
                .foregroundStyle(Palette.ink)
            Spacer().frame(height: Space.xs)
            Text("One adapter, three languages — and the honest state of it.")
                .textStyle(.bodyS)
                .foregroundStyle(Palette.inkSecondary)

            Spacer().frame(height: Space.l)

            HStack(alignment: .top, spacing: Space.l) {
                ForEach(result.languages, id: \.language) { language in
                    column(language, requestSummary: result.requestSummary)
                }
            }

            Spacer(minLength: Space.m)

            Text("English holds. Spanish needs a repetition penalty to stay coherent. Russian does not hold — a rank-8 adapter with about a fifth of the corpus per non-English language is the limit of this recipe, not a setting to tweak.")
                .textStyle(.caption)
                .foregroundStyle(Palette.inkTertiary)
                .frame(maxWidth: .infinity, alignment: .center)
        }
    }

    private func column(_ result: CodeSwitchLanguageResult, requestSummary: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(endonym(result.language))
                .textStyle(.headline)
                .foregroundStyle(Palette.ink)

            Spacer().frame(height: Space.xs)
            Text("RE: \(requestSummary)")
                .textStyle(.dataS)
                .foregroundStyle(Palette.inkSecondary)
                .lineLimit(1)

            Spacer().frame(height: Space.m)
            replyBlock(
                label: "Base model",
                labelColor: Palette.inkTertiary,
                railColor: Palette.baseModel,
                text: result.baseReply,
                textStyle: .bodyS,
                textColor: Palette.inkSecondary,
                fill: .clear,
                lineLimit: 8
            )

            Spacer().frame(height: Space.m)
            replyBlock(
                label: "Adapter \(state.activeVersionLabel)",
                labelColor: Palette.accent,
                railColor: Palette.accent,
                text: result.adaptedReply,
                textStyle: .body,
                textColor: Palette.ink,
                fill: Palette.accent.opacity(0.06),
                lineLimit: 6
            )

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardSurface()
        .frame(width: 437)
        // Fixed, not minimum: an over-long reply must not push the bottom
        // caption over the cards (§4.4 sizes these at ~520).
        .frame(height: 520, alignment: .top)
    }

    private func replyBlock(
        label: String,
        labelColor: Color,
        railColor: Color,
        text: String,
        textStyle: TextStyle,
        textColor: Color,
        fill: Color,
        lineLimit: Int
    ) -> some View {
        HStack(alignment: .top, spacing: Space.s) {
            Rectangle()
                .fill(railColor)
                .frame(width: 3)
            VStack(alignment: .leading, spacing: Space.xs) {
                SectionLabel(label, color: labelColor)
                Text(text)
                    .textStyle(textStyle)
                    .foregroundStyle(textColor)
                    .lineLimit(lineLimit)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, Space.xs)
        .padding(.trailing, Space.xs)
        .background(fill)
    }

    /// Each language named in itself — the engine's `displayName` is English-only.
    private func endonym(_ language: DemoLanguage) -> String {
        switch language {
        case .english: "English"
        case .spanish: "Español"
        case .russian: "Русский"
        }
    }
}
