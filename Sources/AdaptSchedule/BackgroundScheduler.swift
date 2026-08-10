import Foundation

#if os(iOS)
import BackgroundTasks
#endif

#if os(macOS)
import AppKit
#endif

/// Background registration helpers for the night pipeline (architecture §4.6).
///
/// ## Platform policies
///
/// | Platform | Mechanism | Constraints |
/// |---|---|---|
/// | **iOS** | `BGProcessingTask` | `requiresExternalPower = true`, `requiresNetworkConnectivity = false` |
/// | **macOS** | `NSBackgroundActivityScheduler` | Repeating, app-controlled quality of service |
///
/// The scheduler only **schedules** work. Host apps still construct an
/// ``AdaptPipeline`` with real train/eval runners inside the handler. This type
/// does not retain user data.
public enum AdaptBackgroundScheduler: Sendable {
    /// Default BGTaskScheduler identifier. Host apps must also list it under
    /// `BGTaskSchedulerPermittedIdentifiers` in Info.plist.
    public static let defaultProcessingTaskIdentifier = "ai.adapt.pipeline.nightly"

    #if os(iOS)
    /// Registers a `BGProcessingTask` handler.
    ///
    /// Call once at launch before the app finishes launching. The system may
    /// invoke `handler` later when power/network constraints allow.
    ///
    /// - Parameters:
    ///   - identifier: Must match Info.plist.
    ///   - handler: Receives the task; must call `setTaskCompleted` and may
    ///     schedule the next request.
    @MainActor
    public static func registerProcessingTask(
        identifier: String = AdaptBackgroundScheduler.defaultProcessingTaskIdentifier,
        handler: @escaping @Sendable (BGProcessingTask) -> Void
    ) {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: identifier,
            using: nil
        ) { task in
            guard let processing = task as? BGProcessingTask else {
                task.setTaskCompleted(success: false)
                return
            }
            handler(processing)
        }
    }

    /// Submits a `BGProcessingTaskRequest` with §4.6 power/network flags.
    ///
    /// - Parameters:
    ///   - identifier: Task id registered at launch.
    ///   - earliestBeginDate: Optional deferral.
    @MainActor
    public static func scheduleProcessingTask(
        identifier: String = AdaptBackgroundScheduler.defaultProcessingTaskIdentifier,
        earliestBeginDate: Date? = nil
    ) throws {
        let request = BGProcessingTaskRequest(identifier: identifier)
        request.requiresExternalPower = true
        request.requiresNetworkConnectivity = false
        request.earliestBeginDate = earliestBeginDate
        try BGTaskScheduler.shared.submit(request)
    }

    /// §4.6 iOS flags documented for hosts that build their own request.
    public static var iOSProcessingRequiresExternalPower: Bool { true }
    /// §4.6: training is fully offline.
    public static var iOSProcessingRequiresNetworkConnectivity: Bool { false }
    #endif

    #if os(macOS)
    /// Creates a configured `NSBackgroundActivityScheduler` for the night pipeline.
    ///
    /// The caller retains the scheduler for the app lifetime and sets
    /// `repeats = true` as desired. Interval default: 24 hours.
    public static func makeMacScheduler(
        identifier: String = AdaptBackgroundScheduler.defaultProcessingTaskIdentifier,
        interval: TimeInterval = 24 * 60 * 60
    ) -> NSBackgroundActivityScheduler {
        let scheduler = NSBackgroundActivityScheduler(identifier: identifier)
        scheduler.repeats = true
        scheduler.interval = interval
        scheduler.qualityOfService = .background
        return scheduler
    }
    #endif
}

/// Host-facing description of scheduler policy (shared by README / QuickReply).
public enum AdaptBackgroundPolicySummary: Sendable {
    public static var text: String {
        """
        Background scheduling (AdaptSchedule):
        - iOS: BGProcessingTask with requiresExternalPower=true,
          requiresNetworkConnectivity=false (§4.6).
        - macOS: NSBackgroundActivityScheduler, default 24h interval, QoS .background.
        - Device policy (iOS): charging AND battery ≥ 40%; abort on thermal .serious.
        - Device policy (macOS): thermal .serious only (no UIDevice battery gate).
        """
    }
}
