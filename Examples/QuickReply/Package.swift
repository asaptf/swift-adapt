// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "QuickReply",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "QuickReplyEngine", targets: ["QuickReplyEngine"]),
        .executable(name: "QuickReply", targets: ["QuickReply"]),
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .target(
            name: "QuickReplyEngine",
            dependencies: [
                .product(name: "AdaptCore", package: "swift-adapt"),
                .product(name: "AdaptData", package: "swift-adapt"),
                .product(name: "AdaptRegistry", package: "swift-adapt"),
                .product(name: "AdaptEval", package: "swift-adapt"),
                .product(name: "AdaptTrain", package: "swift-adapt"),
                .product(name: "AdaptSchedule", package: "swift-adapt"),
            ],
            path: "Sources/QuickReplyEngine"
        ),
        .executableTarget(
            name: "QuickReply",
            dependencies: ["QuickReplyEngine"],
            path: "Sources/QuickReply"
        ),
    ]
)
