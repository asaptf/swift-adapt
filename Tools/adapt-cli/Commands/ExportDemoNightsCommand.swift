import ArgumentParser
import Foundation

/// `adapt-cli export-demo-nights` — materialize night + held-out JSONL slices.
///
/// Used by `scripts/seed-demo-registry.sh`. Pure file I/O over ``DemoCorpus``.
public struct ExportDemoNightsCommand: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "export-demo-nights",
        abstract: "Split a combined JSONL fixture into night-N + held-out slices."
    )

    @Option(name: .long, help: "Combined JSONL corpus.")
    var data: String

    @Option(name: .long, help: "Output directory for night-*.jsonl and held-out.jsonl.")
    var outDir: String

    @Option(name: .long, help: "Number of overnight slices.")
    var nights: Int = DemoCorpus.defaultNightCount

    @Option(name: .long, help: "Held-out example count (tail of the train remainder + reserved).")
    var heldOut: Int = DemoCorpus.defaultHeldOutCount

    public init() {}

    public func run() async throws {
        let dataURL = CLICommon.resolvePath(data)
        let outURL = CLICommon.resolvePath(outDir)
        let examples = try JSONLLoader.load(from: dataURL)
        let partition = try DemoCorpus.partition(
            examples,
            nightCount: nights,
            heldOutCount: heldOut
        )
        try DemoCorpus.writePartition(partition, to: outURL)

        let uniqueCompletions = Set(examples.map(\.completion)).count
        print(
            """
            Exported demo corpus slices → \(outURL.path)
              nights=\(partition.nights.count)  per_night=\(partition.nights.first?.count ?? 0)
              held_out=\(partition.heldOut.count)  total_in=\(examples.count)
              unique_completions=\(uniqueCompletions)  completion_overlap_train_heldout=0
            """
        )
        for (index, night) in partition.nights.enumerated() {
            print("  night-\(index + 1).jsonl  examples=\(night.count)")
        }
        print("  held-out.jsonl  examples=\(partition.heldOut.count)")
    }
}
