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
    ]
)
