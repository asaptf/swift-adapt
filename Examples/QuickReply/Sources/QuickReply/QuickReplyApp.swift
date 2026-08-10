import AdaptSchedule
import QuickReplyEngine
import SwiftUI

/// iOS skeleton for overnight on-device personalization (architecture §6 M4).
///
/// This target is intentionally thin: capture UI + background registration.
/// Real LoRA training on a phone is validated via the manual protocol in
/// `TESTING.md` — not claimed by CI.
@main
struct QuickReplyApp: App {
    @State private var model = QuickReplyAppModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .task {
                    await model.bootstrap()
                }
        }
    }
}

@MainActor
@Observable
final class QuickReplyAppModel {
    var status: String = "Starting…"
    var lastOutcome: String = "—"
    var captureCountHint: String = "0"
    private var host: QuickReplyPipelineHost?

    func bootstrap() async {
        do {
            let host = try QuickReplyPipelineHost()
            self.host = host
            #if os(iOS)
            QuickReplyBackground.register(host: host)
            try QuickReplyBackground.schedule()
            status = "Registered BGProcessingTask (external power, no network)"
            #else
            status = "macOS host — use forced run; BGProcessingTask is iOS-only"
            #endif
            await refreshStats()
        } catch {
            status = "Bootstrap failed: \(error.localizedDescription)"
        }
    }

    func captureSample() async {
        guard let host else { return }
        do {
            try await host.store.captureReply(
                context: "Colleague: Can you send the deck?",
                body: "On it — shipping the deck in five."
            )
            status = "Captured sample reply"
            await refreshStats()
        } catch {
            status = "Capture failed: \(error.localizedDescription)"
        }
    }

    func runPipelineNow() async {
        guard let host else { return }
        status = "Running pipeline…"
        do {
            let outcome = try await host.runNightly()
            lastOutcome = summarize(outcome)
            status = "Pipeline finished"
            await refreshStats()
        } catch {
            status = "Pipeline error: \(error.localizedDescription)"
        }
    }

    private func refreshStats() async {
        guard let host else { return }
        do {
            let stats = try await host.store.buffer.stats()
            captureCountHint = "\(stats.exampleCount)"
        } catch {
            captureCountHint = "?"
        }
    }

    private func summarize(_ outcome: PipelineOutcome) -> String {
        let executed = outcome.stages.filter(\.executed).map(\.stage.rawValue).joined(separator: "→")
        return "stop=\(String(describing: outcome.stopReason)) stages=[\(executed)]"
    }
}

struct ContentView: View {
    @Bindable var model: QuickReplyAppModel

    var body: some View {
        NavigationStack {
            Form {
                Section("Status") {
                    Text(model.status)
                    LabeledContent("Buffer examples", value: model.captureCountHint)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Last outcome")
                        Text(model.lastOutcome)
                            .font(.caption.monospaced())
                    }
                }
                Section("Actions") {
                    Button("Capture sample reply") {
                        Task { await model.captureSample() }
                    }
                    Button("Run pipeline now (foreground)") {
                        Task { await model.runPipelineNow() }
                    }
                }
                Section("Device protocol") {
                    Text(
                        "Overnight training on a physical iPhone is documented in TESTING.md. CI does not run that path."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("QuickReply")
        }
    }
}
