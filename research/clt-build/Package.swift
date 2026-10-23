// swift-tools-version: 6.2
// Builds the AIME input-method executable without Xcode (CLT only); resources come from the official bundle.
import PackageDescription
let package = Package(
    name: "AIMEApp",
    defaultLocalization: "zh-Hans",
    platforms: [.macOS(.v26)],
    dependencies: [.package(path: "../../Packages/AIMEKit")],
    targets: [
        .executableTarget(
            name: "AIME",
            dependencies: [
                .product(name: "RimeKit", package: "AIMEKit"),
                .product(name: "AIMECore", package: "AIMEKit"),
                .product(name: "AIMEPanel", package: "AIMEKit"),
                .product(name: "AIMEAI", package: "AIMEKit"),
            ],
            path: "Sources/AIME",
            exclude: ["AIME.entitlements", "Info.plist", "Resources"],
            swiftSettings: [.swiftLanguageMode(.v6)],
            linkerSettings: [.linkedFramework("InputMethodKit"), .linkedFramework("Carbon"),
                             .unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"])]
        ),
    ]
)
