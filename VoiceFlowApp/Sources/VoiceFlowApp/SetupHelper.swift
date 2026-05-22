import Foundation

/// Downloads and verifies the two models VoiceFlow ships with: Parakeet TDT 0.6B v2
/// (STT) and Bonsai-8B Q1_0 (LLM). Centralizes the install logic so both the
/// onboarding wizard and the Models settings page can share the same code path.
///
/// Parakeet is downloaded by the Python daemon on first `load_model` call (via
/// parakeet-mlx → Hugging Face Hub). Bonsai is a single GGUF file we fetch
/// directly via URLSession into the VoiceFlow models directory.
/// SetupHelper is intentionally not @MainActor: SwiftUI views' implicit init runs
/// off the main actor in Swift 6, and an `@StateObject` that constructs a MainActor
/// type from there triggers actor-isolation runtime warnings (which on macOS can
/// manifest as a blank view). Mutations of @Published vars are dispatched to main
/// explicitly inside the methods that need it.
final class SetupHelper: ObservableObject {

    enum Phase: Equatable {
        case idle
        case downloadingBonsai(bytesDone: Int64, bytesTotal: Int64)
        /// Parakeet doesn't expose download progress directly — we poll the HF
        /// cache directory on disk and report observed size vs. an estimate.
        case loadingParakeet(bytesDone: Int64, bytesTotal: Int64)
        case complete
        case failed(String)
    }

    @Published var phase: Phase = .idle

    // Public switches — true once the corresponding artifact is on disk/loaded.
    @Published var bonsaiInstalled: Bool = false
    @Published var parakeetInstalled: Bool = false

    private var parakeetEngine: ParakeetASREngine?
    private var bonsaiDownloadTask: URLSessionDownloadTask?

    init() {
        refreshInstallState()
    }

    @MainActor
    private func ensureParakeetEngine() -> ParakeetASREngine {
        if let e = parakeetEngine { return e }
        let e = ParakeetASREngine()
        parakeetEngine = e
        return e
    }

    // MARK: - Paths

    static let bonsaiFilename = "Bonsai-8B-Q1_0.gguf"
    static let bonsaiURL = URL(string: "https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B-Q1_0.gguf")!
    /// Approximate size used purely for the UI rundown.
    static let bonsaiApproxBytes: Int64 = 1_152_000_000  // ~1.1 GB
    /// Approximate Parakeet bf16 weight size on Hugging Face cache.
    static let parakeetApproxBytes: Int64 = 1_200_000_000  // ~1.2 GB
    static var totalApproxBytes: Int64 { bonsaiApproxBytes + parakeetApproxBytes }

    static var modelsDirectory: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("com.era-laboratories.voiceflow/models")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static var bonsaiPath: URL {
        modelsDirectory.appendingPathComponent(bonsaiFilename)
    }

    static var parakeetCachePath: URL {
        // Hugging Face hub default cache — parakeet-mlx uses this layout.
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2")
    }

    // MARK: - State refresh

    func refreshInstallState() {
        bonsaiInstalled = FileManager.default.fileExists(atPath: Self.bonsaiPath.path)
        parakeetInstalled = FileManager.default.fileExists(atPath: Self.parakeetCachePath.path)
    }

    var isFullyInstalled: Bool {
        bonsaiInstalled && parakeetInstalled
    }

    // MARK: - Setup driver

    /// Run the complete install: Bonsai GGUF download, then Parakeet model load.
    /// Idempotent — skips steps that already have artifacts present.
    @MainActor
    func runSetup() async {
        phase = .idle
        do {
            if !bonsaiInstalled {
                try await downloadBonsai()
            }
            await loadParakeet()
            refreshInstallState()
            phase = .complete
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    // MARK: - Bonsai download

    private func downloadBonsai() async throws {
        phase = .downloadingBonsai(bytesDone: 0, bytesTotal: Self.bonsaiApproxBytes)

        let dest = Self.bonsaiPath
        let tmp = dest.appendingPathExtension("part")

        // Resume support: if a .part file exists, treat its size as already-downloaded.
        // Simpler approach for now: always re-download from scratch.
        try? FileManager.default.removeItem(at: tmp)

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            // IMPORTANT: do NOT pass a completion handler to downloadTask(with:).
            // Apple docs: "If you create a download task with a completion handler,
            // neither the URLSessionTaskDelegate nor URLSessionDownloadTaskDelegate
            // methods of your session delegate are called for that task." That bug
            // froze the progress bar at 0/1099 MB in v2.0.4 and earlier — the
            // download was actually running but didWriteData was silently muted.
            // Delegate-only: progress fires, file move happens in didFinishDownloadingTo,
            // failures surface via didCompleteWithError.
            let delegate = DownloadProgressDelegate(
                destination: dest,
                onProgress: { [weak self] done, total in
                    Task { @MainActor in
                        self?.phase = .downloadingBonsai(
                            bytesDone: done,
                            bytesTotal: max(total, Self.bonsaiApproxBytes)
                        )
                    }
                },
                onComplete: { result in
                    continuation.resume(with: result)
                }
            )

            let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
            var req = URLRequest(url: Self.bonsaiURL)
            req.timeoutInterval = 60
            let task = session.downloadTask(with: req)
            self.bonsaiDownloadTask = task
            task.resume()
        }

        bonsaiInstalled = true
    }

    // MARK: - Parakeet load (downloads via parakeet-mlx if cache miss)

    @MainActor
    private func loadParakeet() async {
        phase = .loadingParakeet(bytesDone: 0, bytesTotal: Self.parakeetApproxBytes)
        let engine = ensureParakeetEngine()

        // Poll the HF cache directory while parakeet-mlx downloads weights.
        // Cancellation: we set this token to true when load completes.
        let pollDone = ParakeetPollFlag()
        Task.detached { [weak self] in
            while !pollDone.done {
                let bytes = SetupHelper.directorySize(SetupHelper.parakeetCachePath)
                await MainActor.run { [weak self] in
                    guard let self = self else { return }
                    if case .loadingParakeet = self.phase {
                        self.phase = .loadingParakeet(
                            bytesDone: bytes,
                            bytesTotal: max(bytes, Self.parakeetApproxBytes)
                        )
                    }
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }

        await engine.loadModel()
        pollDone.done = true

        if case .ready = engine.state {
            parakeetInstalled = true
        }
    }

    /// Compute total bytes under a directory — fast walk via FileManager enumerator.
    private static func directorySize(_ url: URL) -> Int64 {
        guard FileManager.default.fileExists(atPath: url.path) else { return 0 }
        let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey]
        )
        var total: Int64 = 0
        while let f = enumerator?.nextObject() as? URL {
            if let v = try? f.resourceValues(forKeys: [.fileSizeKey]),
               let s = v.fileSize {
                total += Int64(s)
            }
        }
        return total
    }

    // MARK: - Progress helpers for UI

    var progressFraction: Double? {
        switch phase {
        case .downloadingBonsai(let done, let total):
            guard total > 0 else { return nil }
            return Double(done) / Double(total) * 0.5  // Bonsai is the first half
        case .loadingParakeet(let done, let total):
            guard total > 0 else { return 0.5 }
            return 0.5 + Double(done) / Double(total) * 0.5
        case .complete:
            return 1.0
        default:
            return nil
        }
    }

    var phaseLabel: String {
        switch phase {
        case .idle:
            return ""
        case .downloadingBonsai(let done, let total):
            let mb = Double(done) / 1_048_576.0
            let totalMB = Double(total) / 1_048_576.0
            return String(format: "Downloading Bonsai-8B GGUF — %.0f / %.0f MB", mb, totalMB)
        case .loadingParakeet(let done, let total):
            let mb = Double(done) / 1_048_576.0
            let totalMB = Double(total) / 1_048_576.0
            if mb < 1.0 {
                return "Loading Parakeet TDT 0.6B (first run, ~1.2 GB)…"
            }
            return String(format: "Downloading Parakeet TDT 0.6B — %.0f / %.0f MB", mb, totalMB)
        case .complete:
            return "Setup complete. Both models installed."
        case .failed(let msg):
            return "Setup failed: \(msg)"
        }
    }
}

/// Simple flag wrapper so the detached polling Task can be told to stop.
final class ParakeetPollFlag: @unchecked Sendable {
    var done: Bool = false
}

enum SetupError: Error, LocalizedError {
    case downloadFailed(String)

    var errorDescription: String? {
        switch self {
        case .downloadFailed(let msg): return msg
        }
    }
}

/// URLSession delegate that surfaces byte-level download progress AND owns
/// the final move-to-destination step. Delegate-only — no completion handler
/// on the underlying downloadTask, otherwise didWriteData/didFinishDownloadingTo
/// stay silent (Apple's docs: passing a completion handler suppresses the
/// delegate's progress callbacks for that task).
final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    let destination: URL
    let onProgress: (Int64, Int64) -> Void
    let onComplete: (Result<Void, Error>) -> Void
    private var hasCompleted = false

    init(
        destination: URL,
        onProgress: @escaping (Int64, Int64) -> Void,
        onComplete: @escaping (Result<Void, Error>) -> Void
    ) {
        self.destination = destination
        self.onProgress = onProgress
        self.onComplete = onComplete
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // location is a temp URL that gets deleted shortly after this method
        // returns, so the move has to be synchronous here.
        guard !hasCompleted else { return }
        do {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
            hasCompleted = true
            onComplete(.success(()))
        } catch {
            hasCompleted = true
            onComplete(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard !hasCompleted else { return }
        if let error = error {
            hasCompleted = true
            onComplete(.failure(error))
        }
        // Success was already signaled from didFinishDownloadingTo.
    }
}
