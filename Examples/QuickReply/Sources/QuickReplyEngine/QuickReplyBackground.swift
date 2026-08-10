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
        AdaptBackgroundScheduler.registerProcessingTask(identifier: identifier) { task in
            let work = Task {
                do {
                    _ = try await host.runNightly()
                    task.setTaskCompleted(success: true)
                } catch {
                    task.setTaskCompleted(success: false)
                }
                // Always request the next night window.
                try? AdaptBackgroundScheduler.scheduleProcessingTask(identifier: identifier)
            }
            task.expirationHandler = {
                work.cancel()
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
