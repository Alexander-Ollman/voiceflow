import AppKit
import Foundation
import Darwin

@main
struct TroubleshootingTests {
    @MainActor
    static func main() async throws {
        NSApplication.shared.setActivationPolicy(.prohibited)
        var checks = 0
        func check(_ value: Bool, _ message: String) {
            precondition(value, message)
            checks += 1
        }
        let model = DictationDiagnostics()
        let field = NSTextView(frame: NSRect(x: 0, y: 0, width: 400, height: 100))
        let window = NSWindow(contentRect: field.frame, styleMask: .titled, backing: .buffered, defer: false)
        window.contentView = field
        window.makeFirstResponder(field)
        model.field = field
        model.begin()
        model.registered(id: 1, status: -9878)
        check(model.registration[1] == -9878, "Registration errors stay visible")
        model.registered(id: 1, status: 0)
        model.receivedHotkey(id: 2)
        check(!model.hotkeyReceived, "Other actions do not pass Dictate")
        model.receivedHotkey(id: 1)
        model.captureStarted()
        check(model.running && model.recording, "A focused field starts capture")
        model.audioLevel(0)
        check(model.results[.microphone] != .passed, "Silence does not pass the microphone signal")
        model.audioLevel(0.2)
        check(model.results[.microphone] == .passed, "A real signal passes the meter")
        model.transcribing()
        model.recognized("Private test sentence")
        model.formatted(success: true)
        check(!model.passed, "Recognition and formatting alone cannot pass insertion")
        check(model.preparePaste("Private test sentence", applicationIsActive: true), "Focused practice paste is authorized")
        check(!model.passed, "Sending a key event does not prove insertion")
        model.fieldChanged("Different text")
        check(!model.passed, "Unrelated typing cannot complete the test")
        model.fieldChanged("Private test sentence")
        check(model.passed && !model.running, "Actual matching field change completes the test")
        check(!model.summary.contains("Private test sentence"), "Check summary excludes the transcript")
        model.reset()
        check(!model.passed && !model.hotkeyReceived, "Reset invalidates previous success")

        model.captureStarted()
        model.transcribing()
        model.recognized("Button test")
        model.formatted(success: true)
        _ = model.preparePaste("Button test", applicationIsActive: true)
        model.fieldChanged("Button test")
        check(model.verified && !model.passed, "A button-only test cannot pass the hotkey requirement")
        model.reset()
        model.captureStarted()
        model.transcribing()
        model.tick(now: Date().addingTimeInterval(31))
        check(model.slowStage, "Stalled stages have visible troubleshooting guidance")
        model.end()
        check(!model.preparePaste("Late output", applicationIsActive: true), "Closing the check suppresses late output")
        check(!model.running, "Suppressed completion releases the in-flight test")
        model.begin()
        model.field = field
        model.captureStarted()
        model.transcribing()
        model.recognized("")
        check(model.results[.speech] != .passed && !model.running, "Empty recognition is distinct from success")
        model.reset()
        model.captureStarted()
        model.transcribing()
        model.recognized("Hello")
        model.formatted(success: false)
        check(model.results[.formatting] != .passed && !model.running, "Formatter failure remains distinct from no speech")
        model.reset()
        model.captureStarted()
        check(!model.preparePaste("Hello", applicationIsActive: false), "Background app cannot authorize test insertion")

        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        process.arguments = ["-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(30)"]
        process.standardOutput = output
        try process.run()
        _ = try output.fileHandleForReading.read(upToCount: 6)
        var heartbeat = false
        let beat = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 30_000_000)
            heartbeat = true
        }
        let start = Date()
        let stopped = await ManagedProcessShutdown.stop(process, grace: 0.15)
        let responsiveDuringStop = heartbeat
        await beat.value
        check(stopped && !process.isRunning, "A TERM-resistant owned helper is killed")
        check(Date().timeIntervalSince(start) < 1.5, "Process shutdown has a deadline")
        check(responsiveDuringStop, "Main actor remains responsive during helper shutdown")
        let socketPath = "/tmp/vf-stalled-\(UUID().uuidString).sock"
        let daemon = Process()
        let ready = Pipe()
        daemon.executableURL = URL(fileURLWithPath: "/usr/bin/python3")
        daemon.arguments = ["-c", "import socket,sys,time; s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[1]); s.listen(); print('ready',flush=True); c,_=s.accept(); c.recv(4096); time.sleep(30)", socketPath]
        daemon.standardOutput = ready
        try daemon.run()
        _ = try ready.fileHandleForReading.read(upToCount: 6)
        let probeStart = Date()
        let timedOut = await Task.detached {
            do {
                _ = try ParakeetASREngine.sendCommandSync(["command": "check"], socketPath: socketPath)
                return false
            } catch { return true }
        }.value
        check(timedOut && Date().timeIntervalSince(probeStart) < 3, "An unresponsive daemon health probe uses the short deadline")
        _ = await ManagedProcessShutdown.stop(daemon, grace: 0.1)
        try? FileManager.default.removeItem(atPath: socketPath)

        let cleanupStart = Date()
        await ManagedProcessShutdown.finishCleanup(timeout: 0.05) { Thread.sleep(forTimeInterval: 0.3) }
        check(Date().timeIntervalSince(cleanupStart) < 0.25, "A blocked native cleanup cannot freeze app termination")
        // Let the timed-out work finish to exercise the once-only completion race.
        try await Task.sleep(nanoseconds: 350_000_000)
        print("Passed \(checks) troubleshooting checks")
    }
}
