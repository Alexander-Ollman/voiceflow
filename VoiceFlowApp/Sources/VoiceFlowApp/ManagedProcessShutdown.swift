import Foundation
import Darwin

/// Stops only a Process owned by this app. Polling yields to the main run loop;
/// neither graceful termination nor SIGKILL is followed by an unbounded wait.
@MainActor
enum ManagedProcessShutdown {
    static func stop(_ process: Process?, grace: TimeInterval = 1.5) async -> Bool {
        guard let process, process.isRunning else { return true }
        process.terminate()
        let deadline = Date().addingTimeInterval(grace)
        while process.isRunning, Date() < deadline {
            do { try await Task.sleep(nanoseconds: 50_000_000) } catch { break }
        }
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        let killDeadline = Date().addingTimeInterval(0.5)
        while process.isRunning, Date() < killDeadline {
            do { try await Task.sleep(nanoseconds: 20_000_000) } catch { break }
        }
        return !process.isRunning
    }
}

/// A native cleanup call may be waiting for an in-flight inference lock. Give
/// it a bounded background opportunity to finish without freezing AppKit.
extension ManagedProcessShutdown {
    static func finishCleanup(timeout: TimeInterval = 2, work: @escaping @Sendable () -> Void) async {
        await withCheckedContinuation { continuation in
            let completion = ShutdownCompletion(continuation)
            DispatchQueue.global(qos: .userInitiated).async {
                work()
                completion.finish()
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + timeout) { completion.finish() }
        }
    }
}

private final class ShutdownCompletion: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    init(_ continuation: CheckedContinuation<Void, Never>) { self.continuation = continuation }
    func finish() {
        lock.lock()
        let pending = continuation
        continuation = nil
        lock.unlock()
        pending?.resume()
    }
}
