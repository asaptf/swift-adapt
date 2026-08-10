// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Adapt",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(name: "AdaptCore", targets: ["AdaptCore"]),
        .library(name: "AdaptRegistry", targets: ["AdaptRegistry"]),
        .library(name: "AdaptTrain", targets: ["AdaptTrain"]),
        .library(name: "AdaptInference", targets: ["AdaptInference"]),
        .executable(name: "adapt-cli", targets: ["adapt-cli"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.4")),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        // Declared explicitly for adapt-cli (also transitive via mlx-swift).
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMinor(from: "1.8.0")),
        // CLI-only model download + tokenization (mlx-swift-lm 3.x decoupled these).
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "AdaptCore",
            path: "Sources/AdaptCore",
            exclude: ["README.md"]
        ),
        .target(
            name: "AdaptRegistry",
            dependencies: ["AdaptCore"],
            path: "Sources/AdaptRegistry",
            exclude: ["README.md"]
        ),
        .target(
            name: "AdaptTrain",
            dependencies: [
                "AdaptCore",
                "AdaptRegistry",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/AdaptTrain",
            exclude: ["README.md"]
        ),
        .target(
            name: "AdaptInference",
            dependencies: [
                "AdaptCore",
                "AdaptRegistry",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/AdaptInference",
            exclude: ["README.md"]
        ),
        // Pure CLI logic (JSONL, formatting, shared options) — testable offline.
        .target(
            name: "AdaptCLI",
            dependencies: [
                "AdaptCore",
                "AdaptRegistry",
                "AdaptTrain",
                "AdaptInference",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Tools/adapt-cli",
            exclude: ["README.md", "Fixtures", "AdaptCLIMain.swift"]
        ),
        .executableTarget(
            name: "adapt-cli",
            dependencies: ["AdaptCLI"],
            path: "Tools/adapt-cli",
            exclude: [
                "README.md",
                "Fixtures",
                // Library sources (owned by the AdaptCLI target):
                "AdaptCLIError.swift",
                "AdaptCLIRoot.swift",
                "CLICommon.swift",
                "InspectFormatter.swift",
                "JSONLLoader.swift",
                "MetalSupport.swift",
                "ModelLoading.swift",
                "SignalHandling.swift",
                "Commands",
            ],
            sources: ["AdaptCLIMain.swift"]
        ),
        .testTarget(
            name: "AdaptCoreTests",
            dependencies: ["AdaptCore"],
            path: "Tests/AdaptCoreTests"
        ),
        .testTarget(
            name: "AdaptRegistryTests",
            dependencies: ["AdaptRegistry", "AdaptCore"],
            path: "Tests/AdaptRegistryTests"
        ),
        .testTarget(
            name: "AdaptTrainTests",
            dependencies: [
                "AdaptTrain",
                "AdaptCore",
                "AdaptRegistry",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Tests/AdaptTrainTests",
            // No committed metallib. MetalBootstrap builds default.metallib from
            // the resolved mlx-swift checkout into .build/mlx-metallib-cache/
            // (revision-keyed) on first AdaptTrainTests setup. See
            // Tests/AdaptTrainTests/README.md.
            exclude: ["README.md"]
        ),
        .testTarget(
            name: "AdaptInferenceTests",
            dependencies: [
                "AdaptInference",
                "AdaptCore",
                "AdaptRegistry",
                // For GenerationOptions → GenerateParameters mapping assertions.
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Tests/AdaptInferenceTests"
        ),
        .testTarget(
            name: "AdaptCLITests",
            dependencies: ["AdaptCLI", "AdaptCore"],
            path: "Tests/AdaptCLITests"
        ),
    ]
)
