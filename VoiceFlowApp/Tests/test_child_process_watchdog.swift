import Foundation
import Darwin

/// Standalone lifecycle test.
///
/// Run from VoiceFlowApp/:
///   swiftc Sources/VoiceFlowApp/ChildProcessWatchdog.swift \
///     Tests/test_child_process_watchdog.swift \
///     -o /tmp/voiceflow-watchdog-test && /tmp/voiceflow-watchdog-test
@main
struct ChildProcessWatchdogTest {
    static func main() throws {
        if CommandLine.arguments.contains("--probe") {
            try runProbe()
            return
        }

        let probe = Process()
        let output = Pipe()
        probe.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
        probe.arguments = ["--probe"]
        probe.standardOutput = output
        probe.standardError = FileHandle.nullDevice
        try probe.run()
        probe.waitUntilExit()

        let data = output.fileHandleForReading.readDataToEndOfFile()
        guard let text = String(data: data, encoding: .utf8),
              let childPID = Int32(text.trimmingCharacters(in: .whitespacesAndNewlines))
        else {
            throw TestFailure("Probe did not report a child PID")
        }

        let deadline = Date().addingTimeInterval(4)
        while kill(childPID, 0) == 0, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.1)
        }
        guard kill(childPID, 0) != 0 else {
            kill(childPID, SIGKILL)
            throw TestFailure("Watchdog did not terminate child PID \(childPID)")
        }
        print("PASS: watchdog terminated child PID \(childPID) after parent exit")
    }

    private static func runProbe() throws {
        let child = Process()
        child.executableURL = URL(fileURLWithPath: "/bin/sleep")
        child.arguments = ["30"]
        try child.run()
        guard ChildProcessWatchdog.start(childPID: child.processIdentifier, role: "test-helper") != nil else {
            child.terminate()
            throw TestFailure("Could not start watchdog")
        }
        print(child.processIdentifier)
        fflush(stdout)
        Thread.sleep(forTimeInterval: 0.2)
        // Return without stopping either process, simulating a main-app exit
        // that bypasses applicationWillTerminate.
    }
}

private struct TestFailure: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}
