import StyleMirrorEngine
import StyleMirrorUI
import SwiftUI

/// StyleMirror — the Adapt flagship demo.
///
/// Fixed 1440 × 810 content size: exactly 16:9, so the stage recording needs no
/// crop and downscales to 1080p cleanly at 2× Retina (`DESIGN.md` §4).
@main
struct StyleMirrorApp: App {
    @State private var state = DemoState(makeEngine: StyleMirrorApp.makeEngine)

    var body: some Scene {
        WindowGroup("StyleMirror") {
            RootView(state: state)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .defaultSize(width: WindowGeometry.width, height: WindowGeometry.height)
    }

    /// Prefers the real engine — actual training and generation over the seeded
    /// seven-night registry — and falls back to the scripted one when it cannot
    /// be built (no model on this machine, no seeded registry) or when
    /// `--scripted` is passed.
    ///
    /// The fallback is deliberately visible rather than silent: which engine is
    /// running changes what every number on screen means, so
    /// ``DemoState/isUsingRealEngine`` reports it to the UI.
    nonisolated static func makeEngine() -> any StyleMirrorEngine {
        if LaunchOptions.fromCommandLine().forceScripted {
            return ScriptedEngine(seed: 42)
        }
        do {
            return try AdaptEngine.seededDemo()
        } catch {
            FileHandle.standardError.write(
                Data("StyleMirror: falling back to the scripted engine — \(error)\n".utf8)
            )
            return ScriptedEngine(seed: 42)
        }
    }
}
