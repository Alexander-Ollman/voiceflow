// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "VoiceFlowApp",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "VoiceFlowApp", targets: ["VoiceFlowApp"])
    ],
    targets: [
        .executableTarget(
            name: "VoiceFlowApp",
            dependencies: ["VoiceFlowFFI"],
            path: "Sources/VoiceFlowApp",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ],
            linkerSettings: [
                .linkedFramework("Carbon")
            ]
        ),
        .systemLibrary(
            name: "VoiceFlowFFI",
            path: "Sources/VoiceFlowFFI"
        )
    ]
)
