import Foundation

/// Keeps a VoiceFlow-owned helper tied to the lifetime of the main app even
/// when AppKit cannot run `applicationWillTerminate` (for example, SIGABRT).
///
/// `Process` children are re-parented when their parent dies on macOS. A tiny
/// detached /bin/sh monitor therefore records both PIDs and their start-time
/// tokens, then terminates the helper if the VoiceFlow process disappears.
/// Start-time comparison prevents a recycled PID from being mistaken for the
/// original app or helper.
final class ChildProcessWatchdog {
    private let process: Process
    private let role: String

    private init(process: Process, role: String) {
        self.process = process
        self.role = role
    }

    static func start(childPID: Int32, role: String) -> ChildProcessWatchdog? {
        let parentPID = ProcessInfo.processInfo.processIdentifier
        guard let parentStart = processStartToken(pid: parentPID),
              let childStart = processStartToken(pid: childPID)
        else {
            NSLog("[ChildWatchdog] %@: could not read process identity", role)
            return nil
        }

        let script = #"""
        parent="$1"
        child="$2"
        expected_parent_start="$3"
        expected_child_start="$4"
        role="$5"

        same_process() {
          pid="$1"
          expected="$2"
          current=$(/bin/ps -p "$pid" -o lstart= 2>/dev/null | /usr/bin/xargs)
          [ -n "$current" ] && [ "$current" = "$expected" ]
        }

        while same_process "$parent" "$expected_parent_start" && \
              same_process "$child" "$expected_child_start"; do
          /bin/sleep 1
        done

        if ! same_process "$parent" "$expected_parent_start" && \
             same_process "$child" "$expected_child_start"; then
          echo "[$(/bin/date '+%Y-%m-%d %H:%M:%S')] $role: parent lost; terminating PID $child"
          /bin/kill -TERM "$child" 2>/dev/null || true
          i=0
          while same_process "$child" "$expected_child_start" && [ "$i" -lt 20 ]; do
            /bin/sleep 0.1
            i=$((i + 1))
          done
          if same_process "$child" "$expected_child_start"; then
            echo "[$(/bin/date '+%Y-%m-%d %H:%M:%S')] $role: graceful stop timed out; killing PID $child"
            /bin/kill -KILL "$child" 2>/dev/null || true
          fi
        fi
        """#

        let monitor = Process()
        monitor.executableURL = URL(fileURLWithPath: "/bin/sh")
        monitor.arguments = [
            "-c", script, "voiceflow-child-watchdog",
            String(parentPID), String(childPID), parentStart, childStart, role,
        ]
        monitor.qualityOfService = .background

        let logPath = "/tmp/voiceflow_child_watchdog.log"
        if !FileManager.default.fileExists(atPath: logPath) {
            FileManager.default.createFile(atPath: logPath, contents: nil)
        }
        let logHandle = FileHandle(forWritingAtPath: logPath) ?? .nullDevice
        _ = try? logHandle.seekToEnd()
        monitor.standardOutput = logHandle
        monitor.standardError = logHandle

        do {
            try monitor.run()
            NSLog("[ChildWatchdog] %@: monitoring parent PID %d and child PID %d",
                  role, parentPID, childPID)
            return ChildProcessWatchdog(process: monitor, role: role)
        } catch {
            NSLog("[ChildWatchdog] %@: failed to launch: %@", role, error.localizedDescription)
            return nil
        }
    }

    /// Stop the monitor after its child has exited normally. It usually exits
    /// by itself on the next poll; terminate it immediately during app shutdown
    /// so VoiceFlow does not leave a short-lived monitor behind.
    func stop() {
        if process.isRunning {
            kill(process.processIdentifier, SIGKILL)
        }
        NSLog("[ChildWatchdog] %@: stopped", role)
    }

    private static func processStartToken(pid: Int32) -> String? {
        let output = Pipe()
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/ps")
        task.arguments = ["-p", String(pid), "-o", "lstart="]
        task.standardOutput = output
        task.standardError = FileHandle.nullDevice
        do {
            try task.run()
            task.waitUntilExit()
            guard task.terminationStatus == 0 else { return nil }
            let data = output.fileHandleForReading.readDataToEndOfFile()
            let token = String(data: data, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return token?.isEmpty == false ? token : nil
        } catch {
            return nil
        }
    }
}
