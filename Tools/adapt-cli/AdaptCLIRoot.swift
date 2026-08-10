import ArgumentParser

/// Root command for the Adapt on-device personalization CLI.
public struct AdaptCLIRoot: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "adapt-cli",
        abstract: "On-device LoRA train / generate / inspect for Adapt (M1).",
        discussion: """
            Trains rank-8 LoRA adapters with AdaptTrain, stores versioned candidates
            in AdaptRegistry, and compares base vs adapter generation.

            Model loading lives inside this CLI temporarily; AdaptInference (M5)
            will own load + hot-swap.
            """,
        subcommands: [
            TrainCommand.self,
            GenerateCommand.self,
            InspectCommand.self,
            PromoteCommand.self,
            MeasureCommand.self,
            ExportDemoNightsCommand.self,
        ],
        defaultSubcommand: nil
    )

    public init() {}
}
