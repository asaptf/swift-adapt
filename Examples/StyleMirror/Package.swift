// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "StyleMirror",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .executable(name: "StyleMirror", targets: ["StyleMirror"]),
        .executable(name: "StyleMirrorSmoke", targets: ["StyleMirrorSmoke"]),
        .library(name: "StyleMirrorEngine", targets: ["StyleMirrorEngine"]),
    ],
    dependencies: [
        .package(path: "../.."),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.4")),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "StyleMirrorEngine",
            dependencies: [
                .product(name: "AdaptCore", package: "swift-adapt"),
                .product(name: "AdaptRegistry", package: "swift-adapt"),
                .product(name: "AdaptTrain", package: "swift-adapt"),
                .product(name: "AdaptInference", package: "swift-adapt"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/StyleMirrorEngine"
        ),
        .target(
            name: "StyleMirrorUI",
            dependencies: ["StyleMirrorEngine"],
            path: "Sources/StyleMirrorUI"
        ),
        .executableTarget(
            name: "StyleMirror",
            dependencies: ["StyleMirrorUI", "StyleMirrorEngine"],
            path: "Sources/StyleMirror"
        ),
        // Acceptance harness for AdaptEngine (real model). Not used by the UI.
        .executableTarget(
            name: "StyleMirrorSmoke",
            dependencies: [
                "StyleMirrorEngine",
                .product(name: "AdaptCore", package: "swift-adapt"),
            ],
            path: "Sources/StyleMirrorSmoke"
        ),
        .testTarget(
            name: "StyleMirrorEngineTests",
            dependencies: [
                "StyleMirrorEngine",
                .product(name: "AdaptCore", package: "swift-adapt"),
                .product(name: "AdaptInference", package: "swift-adapt"),
                .product(name: "AdaptRegistry", package: "swift-adapt"),
            ],
            path: "Tests/StyleMirrorEngineTests"
        ),
    ]
)
