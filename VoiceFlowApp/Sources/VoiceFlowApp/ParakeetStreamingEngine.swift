import Foundation
import Darwin

/// Streaming wrapper over the parakeet_asr_daemon stream_* protocol.
///
/// Mirrors the shape of MoonshineStreamingEngine so VoiceFlowApp can swap
/// between them with minimal wiring changes. Buffers AudioRecorder chunks
/// to 1s, ships as int16 PCM over a persistent unix-socket session, and
/// publishes finalized + draft text on the main actor.
///
/// Notes:
///   - Requires the daemon to be running with the Parakeet model preloaded.
///     This engine doesn't manage the daemon — `ParakeetASREngine` does.
///   - parakeet-tdt-0.6b-v2 is offline-trained; streaming output is for live
///     display only. The final transcript should come from a batch
///     `transcribe` call on the full audio buffer.
@MainActor
final class ParakeetStreamingEngine: ObservableObject {
    @Published var partialText: String = ""
    @Published var finalizedText: String = ""
    @Published var draftText: String = ""

    /// `true` while a stream session is open and feeding.
    @Published private(set) var isActive: Bool = false

    private nonisolated static let sampleRate: Int = 16000
    private nonisolated static let chunkSamples: Int = 16000      // 1s
    private nonisolated static let socketPath: String = "/tmp/voiceflow_parakeet_daemon.sock"
    private nonisolated static let ioTimeoutSec: Int = 30

    private var pendingSamples: [Float] = []
    private var sessionId: String?
    private var sockFD: Int32 = -1

    /// Serializes all socket I/O for one session. One outstanding feed at a time.
    private let ioQueue = DispatchQueue(label: "voiceflow.parakeet-streaming.io")

    // MARK: - Session lifecycle

    func beginSession() throws {
        endSessionSilently()

        partialText = ""
        finalizedText = ""
        draftText = ""
        pendingSamples.removeAll(keepingCapacity: true)

        let fd = try Self.connectSocket()
        do {
            let response = try Self.exchange(fd: fd, message: [
                "command": "stream_open",
                "sample_rate": Self.sampleRate,
            ])
            guard (response["status"] as? String) == "ok",
                  let sid = response["session_id"] as? String else {
                let msg = (response["message"] as? String) ?? "stream_open failed"
                throw ParakeetStreamingError.daemonError(msg)
            }
            self.sessionId = sid
            self.sockFD = fd
            self.isActive = true
            NSLog("[ParakeetStreaming] Session opened: %@", sid)
        } catch {
            Darwin.close(fd)
            throw error
        }
    }

    /// Feed a chunk of 16kHz mono float32 audio (typically called from main
    /// actor by the AudioRecorder tap dispatch).
    func feedAudioChunk(_ samples: [Float]) {
        guard isActive, sockFD >= 0, let sid = sessionId else { return }
        pendingSamples.append(contentsOf: samples)

        while pendingSamples.count >= Self.chunkSamples {
            let chunk = Array(pendingSamples.prefix(Self.chunkSamples))
            pendingSamples.removeFirst(Self.chunkSamples)
            let b64 = Self.encodePCM16LEBase64(chunk)
            let fd = self.sockFD
            ioQueue.async { [weak self] in
                let response: [String: Any]
                do {
                    response = try Self.exchange(fd: fd, message: [
                        "command": "stream_feed",
                        "session_id": sid,
                        "audio": b64,
                    ])
                } catch {
                    NSLog("[ParakeetStreaming] feed error: %@", error.localizedDescription)
                    return
                }
                let finalized = (response["finalized"] as? String) ?? ""
                let draft = (response["draft"] as? String) ?? ""
                Task { @MainActor [weak self] in
                    guard let self = self, self.isActive else { return }
                    self.finalizedText = finalized
                    self.draftText = draft
                    let composed = Self.composeDisplay(finalized: finalized, draft: draft)
                    // Sticky: parakeet-mlx resets draft_tokens at sentence
                    // boundaries (silence_gap). Before finalized starts
                    // committing (~20s right-context lag), partialText would
                    // briefly go empty → overlay flickers to pill mode. Keep
                    // the last non-empty partial visible across these resets.
                    if !composed.isEmpty {
                        self.partialText = composed
                    }
                }
            }
        }
    }

    /// Close the stream and release socket resources. Final transcript should
    /// come from a separate batch `transcribe` call — this method does not
    /// return it.
    func endSession() {
        guard isActive else { return }
        let fd = sockFD
        let sid = sessionId ?? ""
        sockFD = -1
        sessionId = nil
        isActive = false
        pendingSamples.removeAll(keepingCapacity: true)

        ioQueue.async {
            defer { Darwin.close(fd) }
            do {
                _ = try Self.exchange(fd: fd, message: [
                    "command": "stream_close",
                    "session_id": sid,
                ])
            } catch {
                NSLog("[ParakeetStreaming] close error: %@", error.localizedDescription)
            }
        }
    }

    private func endSessionSilently() {
        if isActive { endSession() }
    }

    // MARK: - Helpers

    private nonisolated static func composeDisplay(finalized: String, draft: String) -> String {
        let f = finalized.trimmingCharacters(in: .whitespaces)
        let d = draft.trimmingCharacters(in: .whitespaces)
        if f.isEmpty { return d }
        if d.isEmpty { return f }
        return f + " " + d
    }

    private nonisolated static func encodePCM16LEBase64(_ samples: [Float]) -> String {
        var data = Data(capacity: samples.count * 2)
        for s in samples {
            let clamped = max(-1.0, min(1.0, s))
            let i = Int16(clamped * 32767.0)
            withUnsafeBytes(of: i.littleEndian) { data.append(contentsOf: $0) }
        }
        return data.base64EncodedString()
    }

    // MARK: - Unix socket plumbing

    private nonisolated static func connectSocket() throws -> Int32 {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw ParakeetStreamingError.socketError("socket() failed")
        }
        var ok = false
        defer { if !ok { Darwin.close(fd) } }

        var timeout = timeval(tv_sec: ioTimeoutSec, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        let sunPathSize = MemoryLayout.size(ofValue: addr.sun_path)
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: sunPathSize) { dest in
                pathBytes.withUnsafeBufferPointer { src in
                    let n = min(src.count, sunPathSize)
                    dest.update(from: src.baseAddress!, count: n)
                }
            }
        }
        let connectResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.connect(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connectResult == 0 else {
            throw ParakeetStreamingError.socketError(
                "connect failed: \(String(cString: strerror(errno)))"
            )
        }
        ok = true
        return fd
    }

    private nonisolated static func exchange(
        fd: Int32, message: [String: Any]
    ) throws -> [String: Any] {
        let payload = try JSONSerialization.data(withJSONObject: message)
        var lenBE = UInt32(payload.count).bigEndian
        let header = Data(bytes: &lenBE, count: 4)
        try sendAll(fd: fd, data: header)
        try sendAll(fd: fd, data: payload)

        let respHeader = try recvExact(fd: fd, count: 4)
        let respLen = respHeader.withUnsafeBytes {
            UInt32(bigEndian: $0.load(as: UInt32.self))
        }
        guard respLen > 0, respLen < 100 * 1024 * 1024 else {
            throw ParakeetStreamingError.socketError("invalid response length \(respLen)")
        }
        let respData = try recvExact(fd: fd, count: Int(respLen))
        guard let dict = try JSONSerialization.jsonObject(with: respData) as? [String: Any] else {
            throw ParakeetStreamingError.socketError("invalid JSON response")
        }
        return dict
    }

    private nonisolated static func sendAll(fd: Int32, data: Data) throws {
        try data.withUnsafeBytes { buf in
            var sent = 0
            let total = buf.count
            let base = buf.baseAddress!
            while sent < total {
                let n = Darwin.send(fd, base.advanced(by: sent), total - sent, 0)
                guard n > 0 else {
                    throw ParakeetStreamingError.socketError(
                        "send failed: \(String(cString: strerror(errno)))"
                    )
                }
                sent += n
            }
        }
    }

    private nonisolated static func recvExact(fd: Int32, count: Int) throws -> Data {
        var data = Data(count: count)
        var received = 0
        try data.withUnsafeMutableBytes { buf in
            let base = buf.baseAddress!
            while received < count {
                let n = Darwin.recv(fd, base.advanced(by: received), count - received, 0)
                guard n > 0 else {
                    throw ParakeetStreamingError.socketError(
                        "recv failed: \(String(cString: strerror(errno)))"
                    )
                }
                received += n
            }
        }
        return data
    }
}

enum ParakeetStreamingError: LocalizedError {
    case socketError(String)
    case daemonError(String)

    var errorDescription: String? {
        switch self {
        case .socketError(let m): return "Parakeet streaming socket error: \(m)"
        case .daemonError(let m): return "Parakeet streaming daemon error: \(m)"
        }
    }
}
