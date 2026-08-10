import StyleMirrorEngine
import StyleMirrorUI
import SwiftUI

/// StyleMirror — the Adapt flagship demo.
///
/// Fixed 1440 × 810 content size: exactly 16:9, so the stage recording needs no
/// crop and downscales to 1080p cleanly at 2× Retina (`DESIGN.md` §4).
@main
struct StyleMirrorApp: App {
    @State private var state = DemoState { ScriptedEngine(seed: 42) }

    var body: some Scene {
        WindowGroup("StyleMirror") {
            RootView(state: state)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
        .defaultSize(width: WindowGeometry.width, height: WindowGeometry.height)
    }
}
