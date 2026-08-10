import AdaptSchedule
import Foundation

#if os(iOS)
import BackgroundTasks
#endif

/// Registers and schedules the iOS `BGProcessingTask` for QuickReply.
///
/// §4.6 flags (enforced by ``AdaptBackgroundScheduler``):
/// - `requiresExternalPower = true`
/// - `requiresNetworkConnectivity = false`
public enum QuickReplyBackground: Sendable {
    /// Call from `application(_:didFinishLaunchingWithOptions:)` / app `init`.
    @MainActor
    public static func register(
        identifier: String = AdaptBackgroundScheduler.defaultProcessingTaskIdentifier,
        host: QuickReplyPipelineHost
    ) {
        #if os(iOS)
        // `host` is an actor (Sendable). The launch handler is `@Sendable` and may
        // run on BGTaskScheduler's private serial queue (`using: nil` in
        // AdaptBackgroundScheduler) — not on the main actor.
        AdaptBackgroundScheduler.registerProcessingTask(identifier: identifier) { task in
            let completer = ProcessingTaskCompleter(task)
            let work = Task {
                var success = false
                do {
                    _ = try await host.runNightly()
                    success = true
                } catch {
                    success = false
                }
                completer.complete(success: success)
                // Always request the next night window. `scheduleProcessingTask`
                // is `@MainActor` on AdaptBackgroundScheduler; this Task does not
                // inherit main-actor isolation from the launch handler.
                await MainActor.run {
                    try? AdaptBackgroundScheduler.scheduleProcessingTask(identifier: identifier)
                }
            }
            task.expirationHandler = {
                work.cancel()
                // Complete here so the system is not left waiting if cancellation
                // is slow to surface inside the pipeline. The completer is once-only.
                completer.complete(success: false)
            }
        }
        #else
        _ = identifier
        _ = host
        #endif
    }

    /// Submits the next processing request (call after register + on complete).
    @MainActor
    public static func schedule(
        identifier: String = AdaptBackgroundScheduler.defaultProcessingTaskIdentifier
    ) throws {
        #if os(iOS)
        try AdaptBackgroundScheduler.scheduleProcessingTask(identifier: identifier)
        #else
        _ = identifier
        #endif
    }
}

#if os(iOS)
/// Bridges a non-`Sendable` `BGProcessingTask` into concurrent work under Swift 6.
///
/// Why `@unchecked Sendable` is safe here:
/// - Apple's contract for `BGProcessingTask` is that the launch handler may start
///   work on any queue and must call `setTaskCompleted(success:)` exactly once.
/// - All access to the task from concurrent contexts goes through this box, which
///   serializes completion with a lock and ignores subsequent calls.
/// - We never expose the raw task or claim it is generally Sendable — only that
///   this single-completion handoff matches the OS API's thread-safety model.
private final class ProcessingTaskCompleter: @unchecked Sendable {
    private let task: BGProcessingTask
    private let lock = NSLock()
    private var completed = false

    init(_ task: BGProcessingTask) {
        self.task = task
    }

    func complete(success: Bool) {
        lock.lock()
        defer { lock.unlock() }
        guard !completed else { return }
        completed = true
        task.setTaskCompleted(success: success)
    }
}
#endif
