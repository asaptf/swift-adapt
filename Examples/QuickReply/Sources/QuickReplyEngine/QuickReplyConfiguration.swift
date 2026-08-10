import AdaptCore
import AdaptSchedule
import AdaptTrain
import Foundation

/// Host-tunable knobs for the QuickReply demo (architecture §6 M4).
public struct QuickReplyConfiguration: Sendable {
    /// Personalization task id (lineage key component).
    public var taskID: String
    /// Base model id the demo would train against on device.
    public var baseModelID: String
    /// BGProcessingTask identifier (must match Info.plist).
    public var backgroundTaskIdentifier: String
    /// Night train budget (defaults to platform policy with documented caveats).
    public var trainBudget: TrainBudget

    public init(
        taskID: String = "quick-reply",
        baseModelID: String = "mlx-community/Qwen3-0.6B-4bit",
        backgroundTaskIdentifier: String = AdaptBackgroundScheduler.defaultProcessingTaskIdentifier,
        trainBudget: TrainBudget = PlatformTrainBudget.current
    ) {
        self.taskID = taskID
        self.baseModelID = baseModelID
        self.backgroundTaskIdentifier = backgroundTaskIdentifier
        self.trainBudget = trainBudget
    }

    /// Lineage identity for this demo.
    public var lineage: AdapterLineage {
        AdapterLineage(taskID: taskID, baseModelID: baseModelID)
    }
}
