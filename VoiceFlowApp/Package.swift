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
    dependencies: [
        .package(url: "https://github.com/moonshine-ai/moonshine-swift.git", from: "0.0.48")
    ],
    targets: [
        .executableTarget(
            name: "VoiceFlowApp",
            dependencies: [
                "VoiceFlowFFI",
                .product(name: "Moonshine", package: "moonshine-swift")
            ],
            path: "Sources/VoiceFlowApp",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ],
            linkerSettings: [
                .linkedFramework("Carbon"),
                .linkedFramework("ServiceManagement"),
                .linkedFramework("ScreenCaptureKit"),
                .linkedLibrary("c++")
            ]
        ),
        .systemLibrary(
            name: "VoiceFlowFFI",
            path: "Sources/VoiceFlowFFI"
        )
    ]
)
