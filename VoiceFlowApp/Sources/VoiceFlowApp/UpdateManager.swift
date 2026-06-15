import AppKit
import SwiftUI

/// In-app self-updater. Pulls the latest **already-built, signed, notarized**
/// VoiceFlow.zip from GitHub Releases, verifies its signature is ours, and
/// atomically swaps `/Applications/VoiceFlow.app`.
///
/// IMPORTANT — user data: every byte of user data (stats/history.jsonl,
/// corrections, snippets, personas, profiles, hotkeys, models, config) lives in
/// `~/Library/...`, keyed to the bundle id `com.era-laboratories.voiceflow`.
/// The swap touches ONLY the `.app` bundle and never `~/Library`, and the new
/// build keeps the same bundle id + signature, so all of it persists untouched.
@MainActor
final class UpdateManager: ObservableObject {
    static let shared = UpdateManager()

    // The public distribution channel — same one the website's Download links use.
    private static let repo = "Alexander-Ollman/voiceflow"
    private static let assetName = "VoiceFlow.zip"
    private static let teamID = "JVSQ3LCY64" // Era Laboratories Developer ID

    enum State: Equatable {
        case idle
        case checking
        case upToDate
        case available(version: String)
        case downloading(progress: Double)
        case verifying
        case readyToInstall(version: String) // staged on disk, will apply on quit / on click
        case failed(String)
    }

    @Published private(set) var state: State = .idle
    /// When on, an available update downloads + verifies silently and applies on
    /// next quit (zero-click). When off, the user installs with one click. Backed
    /// by the same UserDefaults key the Settings toggle (`@AppStorage`) writes.
    var autoUpdateEnabled: Bool {
        UserDefaults.standard.object(forKey: "autoUpdateEnabled") as? Bool ?? true
    }

    /// Path to a verified, staged `.app` waiting to be swapped in (if any).
    private var stagedApp: URL?
    private var stagedVersion: String?
    private var checkTimer: Timer?

    var currentVersion: String {
        (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "0"
    }

    // MARK: - Lifecycle

    /// Begin automatic background checks (called once at launch).
    func startAutomaticChecks() {
        // First check shortly after launch, then every 6 hours.
        DispatchQueue.main.asyncAfter(deadline: .now() + 8) { [weak self] in
            Task { await self?.check(userInitiated: false) }
        }
        let t = Timer.scheduledTimer(withTimeInterval: 6 * 3600, repeats: true) { [weak self] _ in
            Task { await self?.check(userInitiated: false) }
        }
        RunLoop.main.add(t, forMode: .common)
        checkTimer = t
    }

    /// Called from `applicationWillTerminate`: if an update is staged, apply it
    /// silently now (the app is already quitting, so no relaunch).
    func applyStagedUpdateOnQuitIfNeeded() {
        guard let app = stagedApp else { return }
        runSwap(newApp: app, relaunch: false)
    }

    // MARK: - Check

    func check(userInitiated: Bool) async {
        if case .downloading = state { return }
        if case .verifying = state { return }
        if userInitiated { state = .checking }
        guard let (version, assetURL) = await fetchLatestWithRetry() else {
            if userInitiated {
                state = .failed("Couldn't reach the update server — try again later.")
            } else {
                scheduleRetrySoon() // transient; don't wait the full 6h
            }
            return
        }
        guard Self.isNewer(version, than: currentVersion) else {
            if userInitiated { state = .upToDate }
            return
        }
        // Already have this version staged? Don't re-download.
        if stagedVersion == version, stagedApp != nil {
            state = .readyToInstall(version: version)
            return
        }
        if autoUpdateEnabled {
            await downloadVerifyStage(version: version, assetURL: assetURL)
        } else {
            state = .available(version: version)
        }
    }

    /// Manual "Install & Relaunch": download + verify if needed, then swap now.
    func downloadAndInstall() async {
        guard let (version, assetURL) = await fetchLatestWithRetry() else {
            state = .failed("Couldn't reach the update server — try again later.")
            return
        }
        guard Self.isNewer(version, than: currentVersion) else { state = .upToDate; return }
        if stagedVersion != version || stagedApp == nil {
            await downloadVerifyStage(version: version, assetURL: assetURL)
        }
        guard let app = stagedApp else { return }
        installNow(newApp: app)
    }

    /// Apply a staged update immediately and relaunch.
    func installNow(newApp: URL) {
        runSwap(newApp: newApp, relaunch: true)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            NSApp.terminate(nil)
        }
    }

    // MARK: - Network

    /// Resolve the latest version + asset URL. Uses GitHub's plain
    /// `releases/latest` web redirect (→ `/releases/tag/vX.Y.Z`) instead of the
    /// REST API: no 60/hr rate limit, and it stays up when api.github.com is
    /// returning 504s to unauthenticated clients. The asset lives at the
    /// well-known `releases/latest/download/<asset>` path — no API needed.
    private func fetchLatest() async throws -> (version: String, assetURL: URL) {
        let latestURL = URL(string: "https://github.com/\(Self.repo)/releases/latest")!
        var req = URLRequest(url: latestURL)
        req.httpMethod = "HEAD"
        req.setValue("VoiceFlow/\(currentVersion)", forHTTPHeaderField: "User-Agent")
        req.timeoutInterval = 20
        req.cachePolicy = .reloadIgnoringLocalCacheData
        let (_, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, let finalURL = http.url else {
            throw Err.msg("No response from GitHub releases")
        }
        guard (200..<400).contains(http.statusCode) else {
            throw Err.msg("GitHub returned \(http.statusCode)")
        }
        let tag = finalURL.lastPathComponent // e.g. "v2.5.0"
        guard tag.contains(".") else { throw Err.msg("Unexpected releases URL") }
        let assetURL = URL(string: "https://github.com/\(Self.repo)/releases/latest/download/\(Self.assetName)")!
        return (tag, assetURL)
    }

    /// Transient failures (504s, brief offline) are common; retry a few times
    /// with short backoff before giving up so a single blip doesn't skip the
    /// whole check window.
    private func fetchLatestWithRetry(attempts: Int = 3) async -> (version: String, assetURL: URL)? {
        for i in 0..<attempts {
            do { return try await fetchLatest() }
            catch {
                if i < attempts - 1 {
                    try? await Task.sleep(nanoseconds: UInt64(i + 1) * 2_000_000_000)
                }
            }
        }
        return nil
    }

    private var retryScheduled = false
    /// After a fully-failed background check, retry in ~30 min rather than
    /// waiting the full 6-hour cadence.
    private func scheduleRetrySoon() {
        guard !retryScheduled else { return }
        retryScheduled = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1800) { [weak self] in
            self?.retryScheduled = false
            Task { await self?.check(userInitiated: false) }
        }
    }

    private func downloadVerifyStage(version: String, assetURL: URL) async {
        do {
            state = .downloading(progress: 0)
            let zipURL = try await download(assetURL)
            state = .verifying

            let stageDir = try updatesDir().appendingPathComponent(version, isDirectory: true)
            try? FileManager.default.removeItem(at: stageDir)
            try FileManager.default.createDirectory(at: stageDir, withIntermediateDirectories: true)

            // Unzip with ditto (preserves the bundle's code-signature metadata).
            let unzip = run("/usr/bin/ditto", ["-x", "-k", zipURL.path, stageDir.path])
            guard unzip.status == 0 else { throw Err.msg("Failed to expand archive") }
            try? FileManager.default.removeItem(at: zipURL)

            guard let app = findApp(in: stageDir) else { throw Err.msg("No .app in archive") }
            guard verifySignature(app) else {
                try? FileManager.default.removeItem(at: stageDir)
                throw Err.msg("Update failed signature check — not installing")
            }

            stagedApp = app
            stagedVersion = version
            state = .readyToInstall(version: version)
        } catch {
            state = .failed(short(error))
        }
    }

    private func download(_ url: URL) async throws -> URL {
        let (tempURL, resp) = try await URLSession.shared.download(from: url)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw Err.msg("Download failed")
        }
        // Move out of the system temp slot (which is reclaimed immediately).
        let dest = try updatesDir().appendingPathComponent("download.zip")
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: tempURL, to: dest)
        return dest
    }

    // MARK: - Verification (the security gate)

    /// Only install something that is (a) a structurally valid signed bundle,
    /// (b) accepted by Gatekeeper as a notarized Developer ID app, and (c)
    /// signed by *our* team id. Anything else is refused.
    private func verifySignature(_ app: URL) -> Bool {
        let cs = run("/usr/bin/codesign", ["--verify", "--deep", "--strict", "--verbose=2", app.path])
        guard cs.status == 0 else { return false }
        let gate = run("/usr/sbin/spctl", ["--assess", "--type", "execute", "--verbose=4", app.path])
        guard gate.status == 0 else { return false }
        let info = run("/usr/bin/codesign", ["-dvvv", app.path])
        return (info.out + info.err).contains("TeamIdentifier=\(Self.teamID)")
    }

    // MARK: - The swap

    /// Resolve the REAL install path. Under App Translocation `Bundle.main` is a
    /// read-only mirror, so target `/Applications/VoiceFlow.app` directly.
    private var installURL: URL {
        let path = Bundle.main.bundlePath
        if path.contains("/AppTranslocation/") {
            return URL(fileURLWithPath: "/Applications/VoiceFlow.app")
        }
        return URL(fileURLWithPath: path)
    }

    /// Run a detached shell script that waits for this process to exit, then
    /// swaps the bundle and (optionally) relaunches. It touches ONLY the app
    /// bundle path — never `~/Library` — so user data is preserved. On any
    /// failure it restores the previous app so the user is never left without one.
    private func runSwap(newApp: URL, relaunch: Bool) {
        let pid = ProcessInfo.processInfo.processIdentifier
        let install = installURL.path
        let new = newApp.path
        let script = """
        #!/bin/sh
        # VoiceFlow self-update swap — touches only the app bundle, never ~/Library.
        i=0
        while kill -0 \(pid) 2>/dev/null && [ $i -lt 150 ]; do sleep 0.2; i=$((i+1)); done
        rm -rf "\(install).old"
        if ! mv "\(install)" "\(install).old"; then
          \(relaunch ? "open \"\(install)\"" : "true")
          exit 1
        fi
        if ! mv "\(new)" "\(install)"; then
          mv "\(install).old" "\(install)"
          \(relaunch ? "open \"\(install)\"" : "true")
          exit 1
        fi
        xattr -dr com.apple.quarantine "\(install)" 2>/dev/null
        rm -rf "\(install).old"
        \(relaunch ? "open \"\(install)\"" : "true")
        exit 0
        """
        let scriptURL = (try? updatesDir())?.appendingPathComponent("swap.sh")
            ?? URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("vf-swap.sh")
        do {
            try script.write(to: scriptURL, atomically: true, encoding: .utf8)
        } catch { return }
        // Detach via nohup so the swapper survives our termination.
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/sh")
        p.arguments = ["-c", "nohup /bin/sh '\(scriptURL.path)' >/dev/null 2>&1 &"]
        try? p.run()
    }

    // MARK: - Helpers

    private func updatesDir() throws -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport
            .appendingPathComponent("com.era-laboratories.voiceflow", isDirectory: true)
            .appendingPathComponent("Updates", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func findApp(in dir: URL) -> URL? {
        let fm = FileManager.default
        let direct = dir.appendingPathComponent("VoiceFlow.app")
        if fm.fileExists(atPath: direct.path) { return direct }
        let items = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return items.first { $0.pathExtension == "app" }
    }

    private func run(_ launch: String, _ args: [String]) -> (status: Int32, out: String, err: String) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: launch)
        p.arguments = args
        let outPipe = Pipe(), errPipe = Pipe()
        p.standardOutput = outPipe
        p.standardError = errPipe
        do { try p.run() } catch { return (-1, "", "\(error)") }
        let out = outPipe.fileHandleForReading.readDataToEndOfFile()
        let err = errPipe.fileHandleForReading.readDataToEndOfFile()
        p.waitUntilExit()
        return (p.terminationStatus, String(decoding: out, as: UTF8.self), String(decoding: err, as: UTF8.self))
    }

    private func short(_ error: Error) -> String {
        if case let Err.msg(m) = error { return m }
        return (error as NSError).localizedDescription
    }

    enum Err: Error { case msg(String) }

    static func isNewer(_ remote: String, than local: String) -> Bool {
        func parts(_ s: String) -> [Int] {
            s.trimmingCharacters(in: CharacterSet(charactersIn: "vV "))
                .split(separator: ".")
                .map { Int($0.prefix(while: { $0.isNumber })) ?? 0 }
        }
        let r = parts(remote), l = parts(local)
        for i in 0..<max(r.count, l.count) {
            let rv = i < r.count ? r[i] : 0
            let lv = i < l.count ? l[i] : 0
            if rv != lv { return rv > lv }
        }
        return false
    }
}

// MARK: - Settings UI

/// The "Software Update" card body for the Settings pane.
struct UpdateSettingsView: View {
    @ObservedObject private var updater = UpdateManager.shared
    @AppStorage("autoUpdateEnabled") private var autoUpdateEnabled = true

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("VoiceFlow \(updater.currentVersion)")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.9))
                    Text(statusText)
                        .font(.system(size: 11))
                        .foregroundColor(statusColor)
                }
                Spacer()
                trailingControl
            }

            Toggle(isOn: $autoUpdateEnabled) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Install updates automatically").font(.system(size: 12))
                    Text("Downloads in the background and applies the next time you quit.")
                        .font(.system(size: 11)).foregroundColor(.white.opacity(0.45))
                }
            }
            .toggleStyle(.switch)
        }
    }

    @ViewBuilder private var trailingControl: some View {
        switch updater.state {
        case .checking:
            ProgressView().scaleEffect(0.6)
        case .downloading(let p):
            HStack(spacing: 6) {
                ProgressView(value: p).frame(width: 90)
                Text("\(Int(p * 100))%").font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.white.opacity(0.6))
            }
        case .verifying:
            Text("Verifying…").font(.system(size: 12)).foregroundColor(.white.opacity(0.6))
        case .available(let v), .readyToInstall(let v):
            Button(updater.state.isReady ? "Install & Relaunch" : "Update to \(v)") {
                Task { await updater.downloadAndInstall() }
            }
            .buttonStyle(.borderless)
            .font(.system(size: 12, weight: .semibold))
            .foregroundColor(Color(red: 0.45, green: 0.78, blue: 1.0))
        default:
            Button("Check Now") { Task { await updater.check(userInitiated: true) } }
                .buttonStyle(.borderless)
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.white.opacity(0.7))
        }
    }

    private var statusText: String {
        switch updater.state {
        case .idle:                 return "Checks automatically for new versions"
        case .checking:             return "Checking for updates…"
        case .upToDate:             return "You're up to date"
        case .available(let v):     return "Update available: \(v)"
        case .downloading:          return "Downloading update…"
        case .verifying:            return "Verifying signature…"
        case .readyToInstall(let v): return "\(v) ready — applies on quit, or install now"
        case .failed(let m):        return m
        }
    }

    private var statusColor: Color {
        switch updater.state {
        case .available, .readyToInstall: return Color(red: 0.45, green: 0.78, blue: 1.0)
        case .failed:                     return Color(red: 1.0, green: 0.5, blue: 0.45)
        default:                          return .white.opacity(0.45)
        }
    }
}

private extension UpdateManager.State {
    var isReady: Bool { if case .readyToInstall = self { return true }; return false }
}
