// Placeholder entry point. The human UI layer replaces this with a SwiftUI @main App.
// Model layer lives entirely in StyleMirrorEngine — import that, drive StyleMirrorEngine.

import StyleMirrorEngine

@main
enum StyleMirrorPlaceholder {
    static func main() {
        print("StyleMirror model layer only — UI not yet attached.")
        print("Use StyleMirrorEngine (ScriptedEngine) from the app target.")
        _ = SampleCorpus.sentEmails.count
    }
}
