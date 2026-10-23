// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "RecipeFactoryUI",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "RecipeFactoryUI",
            path: "Sources/RecipeFactoryUI",
            resources: [.process("Resources")]
        )
    ]
)
