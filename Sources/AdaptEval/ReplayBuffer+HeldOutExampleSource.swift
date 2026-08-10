import AdaptData

/// M2 seam: the on-disk replay buffer is a valid held-out example source without
/// reshaping ``PromotionEvaluator`` or ``HeldOutExampleSource``.
extension ReplayBuffer: HeldOutExampleSource {}
