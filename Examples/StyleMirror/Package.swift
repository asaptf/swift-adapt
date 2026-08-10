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
                .product(name: "AdaptCore", package: "swift-adapt"),
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
        .testTarget(
            name: "StyleMirrorEngineTests",
            dependencies: [
                "StyleMirrorEngine",
                .product(name: "AdaptCore", package: "swift-adapt"),
            ],
            path: "Tests/StyleMirrorEngineTests"
        ),
    ]
)
