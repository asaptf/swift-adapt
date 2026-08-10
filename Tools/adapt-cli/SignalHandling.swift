import Dispatch
import Foundation

/// Cooperative Ctrl-C → cancellation for long-running CLI work.
///
/// Installs a `SIGINT` dispatch source that invokes `onInterrupt`. The trainer
/// already checks `Task.isCancelled` at step boundaries and writes a consistent
/// checkpoint, so re-running the same `train` command resumes.
public final class InterruptHandler: @unchecked Sendable {
    private let source: DispatchSourceSignal

    /// Begins watching for SIGINT.
    public init(onInterrupt: @escaping @Sendable () -> Void) {
        // Ignore the default terminate so we can checkpoint cleanly.
        signal(SIGINT, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
        source.setEventHandler {
            fputs("\n^C — cancelling (checkpoint will be written)…\n", stderr)
            onInterrupt()
        }
        source.resume()
        self.source = source
    }

    deinit {
        source.cancel()
        signal(SIGINT, SIG_DFL)
    }
}

/// Runs an async body under a Task that Ctrl-C cancels cooperatively.
public func withInterruptibleTask<T: Sendable>(
    _ body: @escaping @Sendable () async throws -> T
) async throws -> T {
    let work = Task {
        try await body()
    }
    let handler = InterruptHandler {
        work.cancel()
    }
    defer { withExtendedLifetime(handler) {} }
    return try await work.value
}
