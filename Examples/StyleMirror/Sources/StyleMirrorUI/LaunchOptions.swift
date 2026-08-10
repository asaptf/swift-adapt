import Foundation

/// Launch-time options for rehearsal and for capturing screenshots.
///
/// These exist so a screen can be reached without driving the app through
/// synthetic keyboard and mouse events — which is both fragile and rude on a
/// machine somebody is using. Everything here is reachable through the normal UI
/// as well; nothing is only available via a flag.
///
/// ```
/// StyleMirror --screen train --preload-sample-corpus --fast --autorun
/// ```
public struct LaunchOptions: Sendable {
    /// Screen to open on, instead of Act 1.
    public var screen: DemoState.Screen?
    /// Fire the screen's primary action shortly after launch.
    public var autorun: Bool
    /// Fill the paste field with the synthetic sample corpus.
    ///
    /// A rehearsal aid only: in a real demo the presenter pastes their own mail,
    /// and the app trains on exactly what was pasted.
    public var preloadSampleCorpus: Bool
    /// Use the ~20 s pass instead of the stage-realistic ~2.5 min one.
    public var fastTraining: Bool

    public init(
        screen: DemoState.Screen? = nil,
        autorun: Bool = false,
        preloadSampleCorpus: Bool = false,
        fastTraining: Bool = false
    ) {
        self.screen = screen
        self.autorun = autorun
        self.preloadSampleCorpus = preloadSampleCorpus
        self.fastTraining = fastTraining
    }

    /// Parses `--screen <name>`, `--autorun`, `--preload-sample-corpus`.
    ///
    /// Unknown arguments are ignored rather than fatal: the app is a demo, and
    /// refusing to launch over a stray flag would be the wrong trade.
    public static func fromCommandLine(
        _ arguments: [String] = CommandLine.arguments
    ) -> LaunchOptions {
        var options = LaunchOptions()
        var index = 1
        while index < arguments.count {
            switch arguments[index] {
            case "--screen":
                if index + 1 < arguments.count {
                    options.screen = DemoState.Screen(argumentName: arguments[index + 1])
                    index += 1
                }
            case "--autorun":
                options.autorun = true
            case "--preload-sample-corpus":
                options.preloadSampleCorpus = true
            case "--fast":
                options.fastTraining = true
            default:
                break
            }
            index += 1
        }
        return options
    }
}

extension DemoState.Screen {
    /// Maps a command-line name onto a screen.
    init?(argumentName: String) {
        switch argumentName.lowercased() {
        case "offline", "1": self = .offline
        case "train", "2": self = .train
        case "blind", "blindtest", "blind-test", "3": self = .blindTest
        case "languages", "4": self = .languages
        case "gate", "5": self = .gate
        default: return nil
        }
    }
}
