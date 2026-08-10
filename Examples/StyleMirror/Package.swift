// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "StyleMirror",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .executable(name: "StyleMirror", targets: ["StyleMirror"]),
        .library(name: "StyleMirrorEngine", targets: ["StyleMirrorEngine"]),
    ],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .target(
            name: "StyleMirrorEngine",
            dependencies: [
                .product(name: "AdaptCore", package: "swift-adapt-demo"),
            ],
            path: "Sources/StyleMirrorEngine"
        ),
        .executableTarget(
            name: "StyleMirror",
            dependencies: ["StyleMirrorEngine"],
            path: "Sources/StyleMirror"
        ),
        .testTarget(
            name: "StyleMirrorEngineTests",
            dependencies: [
                "StyleMirrorEngine",
                .product(name: "AdaptCore", package: "swift-adapt-demo"),
            ],
            path: "Tests/StyleMirrorEngineTests"
        ),
    ]
)
