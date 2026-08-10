import AdaptData

/// M2 seam: the on-disk replay buffer is a valid training data source without
/// reshaping ``Trainer`` or ``TrainingDataSource``.
extension ReplayBuffer: TrainingDataSource {}
