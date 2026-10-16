import AppKit
import Foundation

@main
struct PRUpdateTests {
    @MainActor
    static func main() throws {
        var checks = 0
        func check(_ condition: Bool, _ message: String) {
            precondition(condition, message)
            checks += 1
        }
        let suite = "voiceflow.pr-update.tests.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        var usage: GlobeKeyController.FnUsage? = .showEmoji
        var allowWrites = true
        var controller = GlobeKeyController(defaults: defaults, readUsage: { usage }, writeUsage: {
            if allowWrites { usage = $0 }
        })
        controller.claim()
        check(usage == .doNothing, "Claim parks the system action")
        check(controller.originalUsage == .showEmoji, "Claim saves the previous action")
        controller.claim()
        check(controller.originalUsage == .showEmoji, "Repeated claim preserves the saved action")
        controller.restore()
        check(usage == .showEmoji && controller.originalUsage == nil, "Release restores and clears recovery")

        controller.claim()
        usage = .startDictation
        controller.claim()
        check(controller.originalUsage == .startDictation, "Reclaim records an external preference change")
        controller.restore()
        check(usage == .startDictation, "Release respects the latest external choice")

        controller.claim()
        usage = .changeInputSource
        controller.restore()
        check(usage == .changeInputSource && controller.originalUsage == nil, "Release leaves external changes alone")

        controller.claim()
        allowWrites = false
        controller.restore()
        check(controller.originalUsage == .changeInputSource, "Failed restore keeps recovery")
        allowWrites = true
        // Simulate a relaunch with the same persisted recovery record.
        controller = GlobeKeyController(defaults: defaults, readUsage: { usage }, writeUsage: { usage = $0 })
        controller.restore()
        check(usage == .changeInputSource && controller.originalUsage == nil, "Relaunch can recover a failed restore")

        usage = nil
        controller.claim()
        check(controller.originalUsage == nil, "Unknown system state is left untouched")
        usage = .doNothing
        controller.claim()
        check(controller.originalUsage == nil, "Existing Do Nothing preference needs no recovery")

        let history = RecordingWaveformHistory()
        for _ in 0..<100 { history.append(0.5) }
        check(history.levels.count == 37, "Microphone history stays bounded")
        check(history.levels.allSatisfy { $0 == 0.5 }, "Repeated identical samples still advance history")
        for _ in 0..<37 { history.append(0) }
        check(history.levels.allSatisfy { $0 == 0 }, "Silence fully clears previous speech")
        history.append(.nan)
        history.append(.infinity)
        history.append(-1)
        history.append(5)
        check(history.levels.suffix(4) == [0, 0, 0, 1], "Invalid and out-of-range samples are clamped")
        history.reset()
        check(history.levels.count == 37 && history.levels.allSatisfy { $0 == 0 }, "New recordings start with a clean waveform")

        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("voiceflow-install-test-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        let staged = root.appendingPathComponent(".staging")
        let installed = root.appendingPathComponent("VoiceFlow.app")
        func bundle(_ url: URL, value: String) throws {
            try fm.createDirectory(at: url, withIntermediateDirectories: true)
            try value.write(to: url.appendingPathComponent("version"), atomically: true, encoding: .utf8)
        }
        func version() throws -> String { try String(contentsOf: installed.appendingPathComponent("version"), encoding: .utf8) }
        try bundle(staged, value: "first")
        try AppBundleInstaller.promote(staged: staged, to: installed)
        check(try version() == "first", "Fresh installation succeeds")
        try bundle(staged, value: "second")
        try AppBundleInstaller.promote(staged: staged, to: installed)
        check(try version() == "second", "Existing installation is replaced")
        do {
            // Missing staged bundle forces the final rename to fail after backup.
            try AppBundleInstaller.promote(staged: staged, to: installed)
            preconditionFailure("Missing staging must fail")
        } catch {
            check(try version() == "second", "Failed promotion restores the previous installation")
        }
        check(try fm.contentsOfDirectory(atPath: root.path) == ["VoiceFlow.app"], "No backup or staging residue remains")
        print("Passed \(checks) PR lifecycle checks")
    }
}
