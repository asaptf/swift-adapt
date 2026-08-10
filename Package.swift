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
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.4")),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
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
            // Ships `mlx-swift_Cmlx.bundle/default.metallib` so SPM test
            // executables can load MLX's Metal library (upstream Cmlx looks for
            // this exact bundle name via SWIFTPM_BUNDLE).
            resources: [
                .copy("MetalSupport/mlx-swift_Cmlx.bundle"),
            ]
        ),
    ]
)
