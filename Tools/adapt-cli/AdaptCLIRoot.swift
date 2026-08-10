import ArgumentParser

/// Root command for the Adapt on-device personalization CLI.
public struct AdaptCLIRoot: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "adapt-cli",
        abstract: "On-device LoRA train / generate / inspect / eval for Adapt.",
        discussion: """
            Trains rank-8 LoRA adapters with AdaptTrain, stores versioned candidates
            in AdaptRegistry, runs the AdaptEval promotion gate, and compares base
            vs adapter generation.
            """,
        subcommands: [
            TrainCommand.self,
            GenerateCommand.self,
            InspectCommand.self,
            EvalCommand.self,
            PromoteCommand.self,
            MeasureCommand.self,
            ExportDemoNightsCommand.self,
        ],
        defaultSubcommand: nil
    )

    public init() {}
}
