import StyleMirrorEngine
import SwiftUI

/// Act 3 — the blind test (`DESIGN.md` §4.3).
///
/// The engine owns which candidate is which; this view never learns a role until
/// the reveal comes back, so there is nothing here that could leak the answer.
public struct BlindTestScreen: View {
    private let state: DemoState

    /// Chips appear staggered left→right after the reveal (§7).
    @State private var visibleChips = 0

    public init(state: DemoState) {
        self.state = state
    }

    public var body: some View {
        Group {
            if state.activeVersion == nil {
                EmptyStateMessage(
                    text: "No active adapter yet. Train one in Act 2 — the blind test needs something to hide."
                )
            } else if let round = state.round {
                content(round)
            } else {
                EmptyStateMessage(text: "Preparing a round…")
            }
        }
        .task {
            if state.round == nil { await state.nextRound() }
        }
        .task(id: state.revealResult?.roundID) {
            guard state.revealResult != nil else {
                visibleChips = 0
                return
            }
            try? await Task.sleep(for: .seconds(0.2))
            for index in 1...3 {
                withAnimation(Motion.revealChip) { visibleChips = index }
                try? await Task.sleep(for: .seconds(Motion.revealChipStagger))
            }
        }
    }

    private func content(_ round: BlindTestRound) -> some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Text("Blind test")
                    .textStyle(.title)
                    .foregroundStyle(Palette.ink)
                Spacer()
                TallyChip(tally: state.tally)
            }

            Spacer().frame(height: Space.xs)
            HStack {
                Text("One of these three replies is human. Pick it.")
                    .textStyle(.bodyS)
                    .foregroundStyle(Palette.inkSecondary)
                Spacer()
            }

            Spacer().frame(height: Space.m)
            incomingCard(round)

            Spacer().frame(height: Space.l)
            HStack(alignment: .top, spacing: Space.l) {
                ForEach(Array(round.candidates.enumerated()), id: \.element.id) { index, candidate in
                    ReplyCard(
                        title: "Reply \(["A", "B", "C"][min(index, 2)])",
                        body: candidate.body,
                        isPicked: state.pickedCandidateID == candidate.id,
                        revealedRole: state.revealResult?.reveal[candidate.id],
                        activeVersionLabel: state.activeVersionLabel,
                        chipVisible: visibleChips > index,
                        onPick: { state.pick(candidate.id) }
                    )
                }
            }

            Spacer(minLength: Space.l)
            footer
        }
    }

    private func incomingCard(_ round: BlindTestRound) -> some View {
        VStack(alignment: .leading, spacing: Space.xxs) {
            SectionLabel("Incoming")
            Text("\(round.incoming.fromDisplayName) — \(round.incoming.subject)")
                .textStyle(.bodyS)
                .foregroundStyle(Palette.inkSecondary)
            Text(round.incoming.body)
                .textStyle(.body)
                .foregroundStyle(Palette.ink)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .cardSurface(padding: Space.m)
    }

    private var footer: some View {
        HStack(spacing: Space.m) {
            Spacer()
            if let line = state.revealResultLine {
                Text(line)
                    .textStyle(.bodyS)
                    .foregroundStyle(Palette.inkSecondary)
                    .transition(.opacity)
            }
            if state.revealResult == nil {
                Button("Reveal") { Task { await state.reveal() } }
                    .buttonStyle(PrimaryButtonStyle())
                    .disabled(state.pickedCandidateID == nil)
            } else {
                Button("Next round") { Task { await state.nextRound() } }
                    .buttonStyle(PrimaryButtonStyle())
            }
            Spacer()
        }
        .frame(height: 54)
        .animation(Motion.resultLine, value: state.revealResult?.roundID)
    }
}
