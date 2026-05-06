import SwiftUI
import AppKit
import Carbon
import AVFoundation
import UserNotifications
import CoreGraphics
import ApplicationServices
import Combine
import ServiceManagement
import VoiceFlowFFI

@main
struct VoiceFlowApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            SettingsView()
                .environmentObject(appDelegate.voiceFlow)
                .environmentObject(SnippetManager.shared)
        }
    }
}

// MARK: - Voice Snippets

/// A voice snippet that expands a trigger phrase into full text
struct VoiceSnippet: Codable, Identifiable, Equatable {
    let id: UUID
    var trigger: String      // What user says: "my signature"
    var expansion: String    // What it expands to: "Best regards,\nAlex"

    init(id: UUID = UUID(), trigger: String, expansion: String) {
        self.id = id
        self.trigger = trigger
        self.expansion = expansion
    }
}

/// Manages voice snippets storage and expansion
class SnippetManager: ObservableObject {
    static let shared = SnippetManager()

    @Published var snippets: [VoiceSnippet] = []

    private let storageKey = "voiceflow.snippets"

    init() {
        loadSnippets()
    }

    // MARK: - Storage

    func loadSnippets() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([VoiceSnippet].self, from: data) {
            snippets = decoded
        } else {
            // Add some default examples
            snippets = [
                VoiceSnippet(trigger: "my signature", expansion: "Best regards,\n[Your Name]"),
                VoiceSnippet(trigger: "my email", expansion: "your.email@example.com"),
            ]
            saveSnippets()
        }
    }

    func saveSnippets() {
        if let encoded = try? JSONEncoder().encode(snippets) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
    }

    // MARK: - CRUD Operations

    func addSnippet(trigger: String, expansion: String) {
        let snippet = VoiceSnippet(trigger: trigger.lowercased(), expansion: expansion)
        snippets.append(snippet)
        saveSnippets()
    }

    func updateSnippet(_ snippet: VoiceSnippet) {
        if let index = snippets.firstIndex(where: { $0.id == snippet.id }) {
            snippets[index] = snippet
            saveSnippets()
        }
    }

    func deleteSnippet(_ snippet: VoiceSnippet) {
        snippets.removeAll { $0.id == snippet.id }
        saveSnippets()
    }

    func deleteSnippets(at offsets: IndexSet) {
        snippets.remove(atOffsets: offsets)
        saveSnippets()
    }

    // MARK: - Expansion

    /// Expand any snippet triggers found in the text
    func expandSnippets(in text: String) -> String {
        var result = text
        let lowerText = text.lowercased()

        // Sort by trigger length (longest first) to avoid partial matches
        let sortedSnippets = snippets.sorted { $0.trigger.count > $1.trigger.count }

        for snippet in sortedSnippets {
            let trigger = snippet.trigger.lowercased()

            // Check for the trigger phrase (case-insensitive)
            if lowerText.contains(trigger) {
                // Find and replace (preserving surrounding text)
                result = caseInsensitiveReplace(in: result, target: trigger, replacement: snippet.expansion)
            }
        }

        return result
    }

    private func caseInsensitiveReplace(in text: String, target: String, replacement: String) -> String {
        guard let range = text.range(of: target, options: .caseInsensitive) else {
            return text
        }
        return text.replacingCharacters(in: range, with: replacement)
    }
}

// MARK: - App Profiles

/// A per-app profile for customizing dictation formatting
struct AppProfile: Codable, Identifiable, Equatable {
    let id: String           // bundle identifier
    var displayName: String  // localized app name
    var category: String     // email, slack, code, default
    var customPrompt: String? // user override, nil = use category default
    var personaId: String?   // optional: tie this app to a Persona (overrides customPrompt + category)
    let firstSeenDate: Date
}

// MARK: - Personas (named, reusable LLM context fragments)

struct Persona: Codable, Identifiable, Equatable, Hashable {
    let id: String          // UUID string, stable across renames
    var name: String        // display name e.g. "Software Engineer"
    var prompt: String      // the system-prompt fragment injected as context
    let createdAt: Date
    var isBuiltIn: Bool     // seeded defaults — UI may treat differently (e.g., disable delete)
}

class PersonaManager: ObservableObject {
    static let shared = PersonaManager()

    @Published var personas: [Persona] = []

    private let storageKey = "voiceflow.personas"

    init() {
        loadPersonas()
        // Seed starter personas on first run (when storage is empty)
        if personas.isEmpty {
            personas = Self.seedPersonas()
            savePersonas()
        }
    }

    func loadPersonas() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([Persona].self, from: data) {
            personas = decoded
        }
    }

    func savePersonas() {
        if let encoded = try? JSONEncoder().encode(personas) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
    }

    func persona(byId id: String) -> Persona? {
        personas.first(where: { $0.id == id })
    }

    func upsert(_ persona: Persona) {
        if let idx = personas.firstIndex(where: { $0.id == persona.id }) {
            personas[idx] = persona
        } else {
            personas.append(persona)
        }
        savePersonas()
    }

    func delete(_ persona: Persona) {
        personas.removeAll { $0.id == persona.id }
        savePersonas()
        // Also clear personaId from any AppProfile that referenced this persona
        let manager = AppProfileManager.shared
        for profile in manager.profiles where profile.personaId == persona.id {
            var updated = profile
            updated.personaId = nil
            manager.updateProfile(updated)
        }
    }

    /// Seed a useful starter set. Users can edit/delete/rename freely.
    static func seedPersonas() -> [Persona] {
        let now = Date()
        return [
            Persona(
                id: UUID().uuidString,
                name: "Software Engineer",
                prompt: """
                The speaker is a software engineer working with terminal commands, APIs, and cloud infrastructure. \
                Common technical terms in their dictation: kubectl, Docker, Kubernetes, GraphQL, OAuth, JWT, Postgres, \
                Redis, AWS, GCP, Terraform, Helm, gRPC, SDK, CLI, regex, JSON, YAML, TOML, repo, PR, CI/CD. \
                When a phonetically similar but semantically nonsensical word appears, prefer the technical term. \
                Examples: "cube control" → "kubectl", "oh-auth" → "OAuth", "post gress" → "Postgres", \
                "git hub" → "GitHub", "ya mall" → "YAML", "kee-ay-fka" → "Kafka". \
                Preserve code-like syntax: snake_case, camelCase, kebab-case, dotted paths. Use technical capitalization \
                for acronyms (API, URL, SQL) and product names (PostgreSQL, MongoDB).
                """,
                createdAt: now,
                isBuiltIn: true
            ),
            Persona(
                id: UUID().uuidString,
                name: "Professional Email",
                prompt: """
                The speaker is composing professional email or business correspondence. Use complete sentences, \
                formal tone, proper greetings ("Hi", "Hello", "Dear") and sign-offs ("Best", "Thanks", "Regards"). \
                Spell out numbers under 10 in prose. Avoid contractions in formal contexts unless the rest of the \
                message is conversational. Capitalize properly: people's names, company names, job titles when used \
                as titles. When the speaker dictates an email address or URL, format it cleanly with no spaces.
                """,
                createdAt: now,
                isBuiltIn: true
            ),
            Persona(
                id: UUID().uuidString,
                name: "Casual Chat",
                prompt: """
                The speaker is messaging in a casual chat context (Slack, iMessage, Discord, etc.). Contractions are \
                expected ("don't", "I'm", "we'll"). Lowercase informal acronyms ("lol", "imo", "ngl", "tbh"). Sentences \
                can be short or fragmentary. Don't over-format — match the speaker's casual register. Emoji are fine \
                when the speaker explicitly requests them ("smile emoji" → 😊).
                """,
                createdAt: now,
                isBuiltIn: true
            ),
            Persona(
                id: UUID().uuidString,
                name: "Technical Writing",
                prompt: """
                The speaker is writing technical documentation, blog posts, or long-form prose. Use full sentences \
                with clear structure. Capitalize technical terms consistently (API, JSON, HTTP, REST). When the \
                speaker references code identifiers, format them as code (`functionName`, `variable_name`). Prefer \
                precise terminology: use "function" not "method" unless the language calls them methods, \
                "endpoint" for HTTP routes, "repository" for git repos. Maintain a clear, instructive tone.
                """,
                createdAt: now,
                isBuiltIn: true
            ),
        ]
    }
}

/// Manages auto-detected per-app profiles with optional custom prompts
class AppProfileManager: ObservableObject {
    static let shared = AppProfileManager()

    @Published var profiles: [AppProfile] = []

    private let storageKey = "voiceflow.appProfiles"

    init() {
        loadProfiles()
    }

    // MARK: - Storage

    func loadProfiles() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([AppProfile].self, from: data) {
            profiles = decoded
        }
    }

    func saveProfiles() {
        if let encoded = try? JSONEncoder().encode(profiles) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
    }

    // MARK: - Auto-detect & CRUD

    /// Auto-creates a profile on first encounter, returns existing if already known
    @discardableResult
    func ensureProfile(for app: NSRunningApplication) -> AppProfile? {
        guard let bundleId = app.bundleIdentifier else { return nil }

        if let existing = profiles.first(where: { $0.id == bundleId }) {
            return existing
        }

        let appName = app.localizedName ?? bundleId
        let context = AppContextDetector.detectContext(for: bundleId, appName: appName)
        let profile = AppProfile(
            id: bundleId,
            displayName: appName,
            category: context.rawValue,
            customPrompt: nil,
            firstSeenDate: Date()
        )
        profiles.append(profile)
        saveProfiles()
        return profile
    }

    /// Browser-aware prompt resolution. If the foreground app is a browser, attempts
    /// to read the active tab's URL and match it against BrowserSiteRulesManager.
    /// Falls back to the bundle-id persona if browser/URL detection fails or no
    /// site rule matches.
    func promptForApp(_ app: NSRunningApplication?) -> String {
        guard let bundleId = app?.bundleIdentifier else { return "" }
        if BrowserContext.isBrowser(bundleId: bundleId),
           let url = BrowserContext.currentURL(forBundleId: bundleId),
           let host = BrowserContext.hostname(from: url) {
            let sitePrompt = BrowserSiteRulesManager.shared.promptForHostname(host)
            if !sitePrompt.isEmpty {
                return sitePrompt
            }
        }
        return promptForApp(bundleId: bundleId)
    }

    /// Returns the persona/custom prompt or the category default for the given app.
    /// Resolution order: persona → custom prompt → category default.
    func promptForApp(bundleId: String) -> String {
        guard let profile = profiles.first(where: { $0.id == bundleId }) else {
            return ""
        }

        // Persona takes top priority
        if let pid = profile.personaId,
           let persona = PersonaManager.shared.persona(byId: pid),
           !persona.prompt.isEmpty {
            return "\n[PERSONA: \(persona.name) — for app \(profile.displayName)]\n\(persona.prompt)"
        }

        // Custom prompt next
        if let custom = profile.customPrompt, !custom.isEmpty {
            return "\n[APPLICATION CONTEXT: \(profile.displayName)]\n\(custom)"
        }

        // Fall back to category default
        let context = AppContext(rawValue: profile.category) ?? .general
        return context.contextHint
    }

    func updateProfile(_ profile: AppProfile) {
        if let index = profiles.firstIndex(where: { $0.id == profile.id }) {
            profiles[index] = profile
            saveProfiles()
        }
    }

    func deleteProfile(_ profile: AppProfile) {
        profiles.removeAll { $0.id == profile.id }
        saveProfiles()
    }

    // MARK: - Auto-mapping installed apps to personas

    /// Curated bundle-id → persona name. Persona names must match
    /// PersonaManager.seedPersonas (or a user-renamed persona).
    private static let bundleIdToPersonaName: [String: String] = [
        // Software Engineer
        "com.cmuxterm.app": "Software Engineer",
        "com.apple.Terminal": "Software Engineer",
        "com.googlecode.iterm2": "Software Engineer",
        "com.mitchellh.ghostty": "Software Engineer",
        "dev.warp.Warp-Stable": "Software Engineer",
        "io.alacritty": "Software Engineer",
        "com.github.wez.wezterm": "Software Engineer",
        "net.kovidgoyal.kitty": "Software Engineer",
        "com.apple.dt.Xcode": "Software Engineer",
        "dev.zed.Zed": "Software Engineer",
        "com.microsoft.VSCode": "Software Engineer",
        "com.sublimetext.4": "Software Engineer",
        "com.panic.Nova": "Software Engineer",
        "com.macromates.TextMate": "Software Engineer",
        "cc.arduino.IDE2": "Software Engineer",
        "com.raspberrypi.imagingutility": "Software Engineer",
        "com.docker.docker": "Software Engineer",
        "io.tailscale.ipn.macos": "Software Engineer",
        "com.zerotier.ZeroTier-One": "Software Engineer",
        "com.linear.linear": "Software Engineer",
        "com.electron.ollama": "Software Engineer",
        "com.anthropic.claudefordesktop": "Software Engineer",
        "com.openai.codex": "Software Engineer",
        "ai.opcode.opcode": "Software Engineer",
        "ai.onemcp.onemcp": "Software Engineer",

        // Professional Email
        "com.apple.mail": "Professional Email",
        "com.microsoft.Outlook": "Professional Email",
        "it.bloop.airmail3": "Professional Email",
        "com.readdle.smartemail-Mac": "Professional Email",

        // Casual Chat
        "com.tinyspeck.slackmacgap": "Casual Chat",
        "com.apple.MobileSMS": "Casual Chat",
        "com.apple.FaceTime": "Casual Chat",
        "net.whatsapp.WhatsApp": "Casual Chat",
        "com.hnc.Discord": "Casual Chat",
        "ru.keepcoder.Telegram": "Casual Chat",
        "org.whispersystems.signal-desktop": "Casual Chat",
        "us.zoom.xos": "Casual Chat",
        "com.microsoft.teams2": "Casual Chat",
        "com.openai.atlas": "Casual Chat",

        // Technical Writing
        "notion.id": "Technical Writing",
        "md.obsidian": "Technical Writing",
        "com.apple.iWork.Pages": "Technical Writing",
        "com.literatureandlatte.scrivener3": "Technical Writing",
        "com.ulyssesapp.mac": "Technical Writing",
        "pro.writer.mac": "Technical Writing", // iA Writer
        "net.shinyfrog.bear": "Technical Writing",
    ]

    /// Display-name patterns (lowercased contains) used when bundle id isn't in the curated table.
    /// This catches new apps + variants without us having to know every bundle id.
    private static let displayNamePatterns: [(pattern: String, persona: String)] = [
        // Software Engineer
        ("terminal", "Software Engineer"),
        ("iterm", "Software Engineer"),
        ("ghostty", "Software Engineer"),
        ("warp", "Software Engineer"),
        ("alacritty", "Software Engineer"),
        ("wezterm", "Software Engineer"),
        ("kitty", "Software Engineer"),
        ("xcode", "Software Engineer"),
        ("vs code", "Software Engineer"),
        ("vscode", "Software Engineer"),
        ("zed", "Software Engineer"),
        ("intellij", "Software Engineer"),
        ("pycharm", "Software Engineer"),
        ("webstorm", "Software Engineer"),
        ("goland", "Software Engineer"),
        ("clion", "Software Engineer"),
        ("rubymine", "Software Engineer"),
        ("phpstorm", "Software Engineer"),
        ("rider", "Software Engineer"),
        ("android studio", "Software Engineer"),
        ("sublime", "Software Engineer"),
        ("textmate", "Software Engineer"),
        ("nova", "Software Engineer"),
        ("docker", "Software Engineer"),
        ("postman", "Software Engineer"),
        ("paw", "Software Engineer"),
        ("github", "Software Engineer"),
        ("gitkraken", "Software Engineer"),
        ("sourcetree", "Software Engineer"),
        ("tower", "Software Engineer"),
        ("linear", "Software Engineer"),
        ("ollama", "Software Engineer"),

        // Casual Chat
        ("slack", "Casual Chat"),
        ("messages", "Casual Chat"),
        ("imessage", "Casual Chat"),
        ("whatsapp", "Casual Chat"),
        ("messenger", "Casual Chat"),
        ("discord", "Casual Chat"),
        ("telegram", "Casual Chat"),
        ("signal", "Casual Chat"),
        ("zoom", "Casual Chat"),
        ("microsoft teams", "Casual Chat"),
        ("facetime", "Casual Chat"),

        // Professional Email
        ("mail", "Professional Email"),
        ("outlook", "Professional Email"),
        ("airmail", "Professional Email"),
        ("spark", "Professional Email"),

        // Technical Writing
        ("notion", "Technical Writing"),
        ("obsidian", "Technical Writing"),
        ("pages", "Technical Writing"),
        ("scrivener", "Technical Writing"),
        ("ulysses", "Technical Writing"),
        ("ia writer", "Technical Writing"),
        ("bear", "Technical Writing"),
    ]

    /// Scan installed apps and assign personas based on the curated tables above.
    /// Adds new profiles for installed apps; updates existing profiles only when
    /// they have no persona set (won't clobber user-customized assignments).
    /// Returns the number of profiles created/updated.
    @discardableResult
    func autoMapInstalledApps() -> Int {
        let appDirs = [
            "/Applications",
            "/Applications/Utilities",
            "/System/Applications",
            "/System/Applications/Utilities",
            NSHomeDirectory() + "/Applications",
        ]
        let personaManager = PersonaManager.shared
        var changed = 0

        for dir in appDirs {
            guard let entries = try? FileManager.default.contentsOfDirectory(atPath: dir) else { continue }
            for entry in entries where entry.hasSuffix(".app") {
                let path = dir + "/" + entry
                guard let bundle = Bundle(path: path),
                      let bundleId = bundle.bundleIdentifier else { continue }

                let displayName = (bundle.infoDictionary?["CFBundleDisplayName"] as? String)
                    ?? (bundle.infoDictionary?["CFBundleName"] as? String)
                    ?? entry.replacingOccurrences(of: ".app", with: "")

                let lcName = displayName.lowercased()
                let personaName: String? = Self.bundleIdToPersonaName[bundleId]
                    ?? Self.displayNamePatterns.first { lcName.contains($0.pattern) }?.persona

                guard let pname = personaName,
                      let persona = personaManager.personas.first(where: { $0.name == pname }) else {
                    continue
                }

                if let idx = profiles.firstIndex(where: { $0.id == bundleId }) {
                    // Only assign if user hasn't already customized this profile
                    if profiles[idx].personaId == nil
                        && (profiles[idx].customPrompt?.isEmpty ?? true) {
                        profiles[idx].personaId = persona.id
                        changed += 1
                    }
                } else {
                    profiles.append(AppProfile(
                        id: bundleId,
                        displayName: displayName,
                        category: "default",
                        customPrompt: nil,
                        personaId: persona.id,
                        firstSeenDate: Date()
                    ))
                    changed += 1
                }
            }
        }

        if changed > 0 {
            saveProfiles()
        }
        return changed
    }
}

// MARK: - Transcription Log

/// A single transcription history entry
struct TranscriptionEntry: Codable, Identifiable {
    let id: UUID
    let timestamp: Date
    let rawTranscript: String
    let formattedText: String
    let modelId: String       // LLM or VLM model ID
    let sttEngine: String     // whisper, moonshine, qwen3-asr
    let transcriptionMs: UInt64
    let llmMs: UInt64
    let totalMs: UInt64
    let targetApp: String     // frontmost app name
    var editedText: String?   // text after user correction (if detected)
    var editDetectedAt: Date? // when the edit was detected

    init(rawTranscript: String, formattedText: String, modelId: String, sttEngine: String,
         transcriptionMs: UInt64, llmMs: UInt64, totalMs: UInt64, targetApp: String) {
        self.id = UUID()
        self.timestamp = Date()
        self.rawTranscript = rawTranscript
        self.formattedText = formattedText
        self.modelId = modelId
        self.sttEngine = sttEngine
        self.transcriptionMs = transcriptionMs
        self.llmMs = llmMs
        self.totalMs = totalMs
        self.targetApp = targetApp
        self.editedText = nil
        self.editDetectedAt = nil
    }
}

/// Manages transcription history with JSONL persistence
class TranscriptionLog: ObservableObject {
    static let shared = TranscriptionLog()

    @Published var entries: [TranscriptionEntry] = []

    private let maxAgeDays: Int = 30
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    private var logFileURL: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("com.era-laboratories.voiceflow")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("history.jsonl")
    }

    init() {
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
        loadEntries()
    }

    func append(_ entry: TranscriptionEntry) {
        entries.insert(entry, at: 0)

        // Append to JSONL file
        if let data = try? encoder.encode(entry),
           let line = String(data: data, encoding: .utf8) {
            let lineWithNewline = line + "\n"
            if let fileHandle = try? FileHandle(forWritingTo: logFileURL) {
                fileHandle.seekToEndOfFile()
                fileHandle.write(lineWithNewline.data(using: .utf8)!)
                fileHandle.closeFile()
            } else {
                // File doesn't exist yet — create it
                try? lineWithNewline.data(using: .utf8)?.write(to: logFileURL)
            }
        }
    }

    func loadEntries() {
        guard FileManager.default.fileExists(atPath: logFileURL.path) else {
            entries = []
            return
        }

        guard let data = try? String(contentsOf: logFileURL, encoding: .utf8) else {
            entries = []
            return
        }

        let cutoff = Calendar.current.date(byAdding: .day, value: -maxAgeDays, to: Date()) ?? Date.distantPast
        var loaded: [TranscriptionEntry] = []

        for line in data.components(separatedBy: "\n") where !line.isEmpty {
            if let lineData = line.data(using: .utf8),
               let entry = try? decoder.decode(TranscriptionEntry.self, from: lineData) {
                if entry.timestamp >= cutoff {
                    loaded.append(entry)
                }
            }
        }

        // Sort newest first
        entries = loaded.sorted { $0.timestamp > $1.timestamp }

        // Prune old entries from file if needed
        if loaded.count < data.components(separatedBy: "\n").filter({ !$0.isEmpty }).count {
            pruneOldEntries()
        }
    }

    private func pruneOldEntries() {
        let lines = entries.reversed().compactMap { entry -> String? in
            guard let data = try? encoder.encode(entry),
                  let line = String(data: data, encoding: .utf8) else { return nil }
            return line
        }
        let content = lines.joined(separator: "\n") + (lines.isEmpty ? "" : "\n")
        try? content.data(using: .utf8)?.write(to: logFileURL)
    }

    // MARK: - Stats

    /// Total words dictated (from formatted text)
    var totalWords: Int {
        entries.reduce(0) { $0 + $1.formattedText.split(separator: " ").count }
    }

    /// Current streak days (consecutive days with at least one dictation)
    var streakDays: Int {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        var streak = 0
        var checkDate = today

        let daysWithEntries = Set(entries.map { calendar.startOfDay(for: $0.timestamp) })

        while daysWithEntries.contains(checkDate) {
            streak += 1
            guard let prev = calendar.date(byAdding: .day, value: -1, to: checkDate) else { break }
            checkDate = prev
        }
        return streak
    }

    /// Average words per minute (based on total_ms timings)
    var averageWPM: Int {
        let validEntries = entries.filter { $0.totalMs > 0 }
        guard !validEntries.isEmpty else { return 0 }
        let totalWords = validEntries.reduce(0) { $0 + $1.formattedText.split(separator: " ").count }
        let totalMinutes = validEntries.reduce(0.0) { $0 + Double($1.totalMs) / 60000.0 }
        guard totalMinutes > 0 else { return 0 }
        return Int(Double(totalWords) / totalMinutes)
    }

    /// Entries grouped by day for display
    var entriesByDay: [(String, [TranscriptionEntry])] {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let yesterday = calendar.date(byAdding: .day, value: -1, to: today)!

        var groups: [String: [TranscriptionEntry]] = [:]
        var groupOrder: [String] = []

        for entry in entries {
            let day = calendar.startOfDay(for: entry.timestamp)
            let label: String
            if day == today {
                label = "TODAY"
            } else if day == yesterday {
                label = "YESTERDAY"
            } else {
                let formatter = DateFormatter()
                formatter.dateFormat = "EEEE, MMM d"
                label = formatter.string(from: day).uppercased()
            }

            if groups[label] == nil {
                groups[label] = []
                groupOrder.append(label)
            }
            groups[label]?.append(entry)
        }

        return groupOrder.map { ($0, groups[$0]!) }
    }
}

// MARK: - Correction Learning

/// A learned correction pattern (before → after)
struct CorrectionPattern: Codable, Identifiable, Equatable {
    let id: UUID
    let original: String
    let corrected: String
    let timestamp: Date
    let targetApp: String
}

/// Manages learned correction patterns from user edits
class CorrectionManager: ObservableObject {
    static let shared = CorrectionManager()

    @Published var patterns: [CorrectionPattern] = []

    private let storageKey = "voiceflow.correctionPatterns"
    private let maxPatterns = 100

    init() {
        loadPatterns()
    }

    // MARK: - Storage

    func loadPatterns() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([CorrectionPattern].self, from: data) {
            patterns = decoded
        }
    }

    func savePatterns() {
        if let encoded = try? JSONEncoder().encode(patterns) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
    }

    // MARK: - Manual Add

    /// Add a correction manually from the Settings UI
    func addManualCorrection(original: String, corrected: String) {
        // Deduplicate
        patterns.removeAll { $0.original.lowercased() == original.lowercased() }
        let pattern = CorrectionPattern(
            id: UUID(),
            original: original,
            corrected: corrected,
            timestamp: Date(),
            targetApp: "manual"
        )
        patterns.append(pattern)
        if patterns.count > maxPatterns {
            patterns = Array(patterns.suffix(maxPatterns))
        }
        savePatterns()
        syncToRustPipeline(pattern)
    }

    // MARK: - Rust Pipeline Sync

    /// Notify the app to inject this correction into the live Rust pipeline
    func syncToRustPipeline(_ pattern: CorrectionPattern) {
        NotificationCenter.default.post(
            name: .correctionLearned,
            object: nil,
            userInfo: ["original": pattern.original, "corrected": pattern.corrected]
        )
    }

    // MARK: - Display Corrections

    /// Apply stored corrections to text for real-time display preview
    func applyStoredCorrections(_ text: String) -> String {
        guard !patterns.isEmpty else { return text }
        var result = text
        // Apply each correction (case-insensitive word replacement)
        for pattern in patterns {
            result = result.replacingOccurrencesOfWordBoundary(
                of: pattern.original, with: pattern.corrected
            )
        }
        return result
    }

    // MARK: - Delete & Clear

    func deletePattern(id: UUID) {
        // Find the pattern before removing so we can notify the Rust pipeline
        if let pattern = patterns.first(where: { $0.id == id }) {
            NotificationCenter.default.post(
                name: .correctionRemoved,
                object: nil,
                userInfo: ["original": pattern.original]
            )
        }
        patterns.removeAll { $0.id == id }
        savePatterns()
    }

    func clearAll() {
        // Notify Rust pipeline to remove each correction
        for pattern in patterns {
            NotificationCenter.default.post(
                name: .correctionRemoved,
                object: nil,
                userInfo: ["original": pattern.original]
            )
        }
        patterns.removeAll()
        savePatterns()
    }

    // MARK: - Context Injection

    /// Returns a correction history context block for the LLM
    func correctionContext(for transcript: String) -> String {
        guard !patterns.isEmpty else { return "" }

        // Use recent patterns (last 20)
        let recent = patterns.suffix(20)
        let pairs = recent.map { "'\($0.original)' -> '\($0.corrected)'" }

        return """

        [CORRECTION HISTORY]
        The user has previously corrected: \(pairs.joined(separator: ", "))
        Apply these corrections when you see the same or similar words.
        """
    }

}

// MARK: - Correction Notification

extension Notification.Name {
    static let correctionLearned = Notification.Name("voiceflow.correctionLearned")
    static let correctionRemoved = Notification.Name("voiceflow.correctionRemoved")
}

// MARK: - String Word-Boundary Replace

private extension String {
    /// Case-insensitive word-boundary replacement (mirrors Rust's `case_insensitive_replace_all`)
    func replacingOccurrencesOfWordBoundary(of target: String, with replacement: String) -> String {
        let lower = self.lowercased()
        let lowerTarget = target.lowercased()
        guard !lowerTarget.isEmpty else { return self }

        var result = ""
        var searchStart = lower.startIndex

        while let range = lower.range(of: lowerTarget, range: searchStart..<lower.endIndex) {
            // Check word boundaries
            let atWordStart = range.lowerBound == lower.startIndex
                || !self[self.index(before: range.lowerBound)].isLetter
            let atWordEnd = range.upperBound == lower.endIndex
                || !self[range.upperBound].isLetter

            if atWordStart && atWordEnd {
                result += self[searchStart..<range.lowerBound]
                result += replacement
                searchStart = range.upperBound
            } else {
                result += String(self[searchStart...range.lowerBound])
                searchStart = self.index(after: range.lowerBound)
            }
        }
        result += self[searchStart...]
        return result
    }
}

// MARK: - AI Commands

enum AIPasteMode {
    case atCursor
    case replaceAll
    case appendToEnd
}

enum AICommand {
    case reply(intent: String)
    case rewrite(style: String)
    case proofread
    case continueWriting

    var pasteMode: AIPasteMode {
        switch self {
        case .reply: return .atCursor
        case .rewrite, .proofread: return .replaceAll
        case .continueWriting: return .appendToEnd
        }
    }

    var processingLabel: String {
        switch self {
        case .reply: return "Composing reply..."
        case .rewrite(let style): return "Rewriting (\(style))..."
        case .proofread: return "Proofreading..."
        case .continueWriting: return "Continuing..."
        }
    }

    var resultTitle: String {
        switch self {
        case .reply: return "Reply"
        case .rewrite(let style): return "Rewrite (\(style))"
        case .proofread: return "Proofread"
        case .continueWriting: return "Continue"
        }
    }

    var pasteHint: String {
        switch self {
        case .reply: return "Enter to paste at cursor"
        case .rewrite, .proofread: return "Enter to replace all"
        case .continueWriting: return "Enter to append to end"
        }
    }

    static func parse(_ transcript: String) -> AICommand? {
        let text = transcript
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .trimmingCharacters(in: CharacterSet(charactersIn: "."))

        // Reply commands
        for prefix in ["reply to this saying ", "reply to this with ", "reply saying ", "reply with "] {
            if text.hasPrefix(prefix) {
                let intent = String(text.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces)
                if !intent.isEmpty {
                    return .reply(intent: intent)
                }
            }
        }

        // Rewrite commands — "make this [more] <style>"
        if text.hasPrefix("make this ") {
            var rawStyle = String(text.dropFirst("make this ".count))
            if rawStyle.hasPrefix("more ") {
                rawStyle = String(rawStyle.dropFirst("more ".count))
            }
            let trimmedStyle = rawStyle.trimmingCharacters(in: .whitespaces)
            if !trimmedStyle.isEmpty {
                return .rewrite(style: trimmedStyle)
            }
        }

        // Rewrite commands — "rewrite [this] as <style>"
        if text.hasPrefix("rewrite ") {
            var rest = String(text.dropFirst("rewrite ".count))
            if rest.hasPrefix("this ") {
                rest = String(rest.dropFirst("this ".count))
            }
            if rest.hasPrefix("as ") {
                let style = String(rest.dropFirst("as ".count)).trimmingCharacters(in: .whitespaces)
                if !style.isEmpty {
                    return .rewrite(style: style)
                }
            }
        }

        // Proofread commands
        if text == "proofread this" || text == "proofread" {
            return .proofread
        }

        // Continue commands
        if text == "continue this" || text == "continue writing" || text == "keep going" || text == "keep writing" {
            return .continueWriting
        }

        return nil
    }
}

// MARK: - Formatting Level

enum FormattingLevel: String, CaseIterable {
    case minimal = "minimal"
    case moderate = "moderate"
    case intent = "intent"
    case aggressive = "aggressive"

    var displayName: String {
        switch self {
        case .minimal: return "Minimal"
        case .moderate: return "Moderate"
        case .intent: return "Intent-Aware"
        case .aggressive: return "Aggressive"
        }
    }

    var description: String {
        switch self {
        case .minimal: return "Light cleanup, preserves original speech"
        case .moderate: return "Fix grammar, punctuation, filler words"
        case .intent: return "Fix STT mishears using sentence context"
        case .aggressive: return "Full rewrite for clarity and conciseness"
        }
    }

    var systemPrompt: String {
        switch self {
        case .minimal:
            return """
            Transcribe the speech with minimal changes. Only fix obvious typos and add basic punctuation. \
            Preserve the original wording, filler words, and speech patterns. Do not restructure sentences. \
            Capitalize proper nouns including names, places, brands, app names, and technical terms. \
            Ensure proper spacing: one space after periods, commas, and other punctuation.
            """
        case .moderate:
            return """
            Clean up the transcribed speech while preserving the speaker's intent and tone. \
            Fix grammar, add proper punctuation, remove filler words (um, uh, like), and correct minor mistakes. \
            Keep the original sentence structure where possible. \
            Capitalize proper nouns including names, places, brands, app names, UI elements, and technical terms. \
            Ensure proper spacing: one space after periods, commas, and other punctuation.
            """
        case .intent:
            return """
            [INTENT-AWARE FORMATTING — overrides the conservative \"preserve original wording\" rule above when a word is clearly misheard.]

            The input is a noisy speech-to-text transcript. STT engines sometimes mishear individual words. \
            Read each sentence as a fluent English speaker would. If a word or short phrase is grammatically out of place \
            or makes no semantic sense in context, replace it with the most plausible word the speaker intended, \
            using the surrounding sentence as context.

            How to decide if a substitution is warranted:
            1. The current word/phrase makes the sentence ungrammatical OR semantically nonsensical.
            2. There exists a phonetically similar word (similar consonants, syllable count, or stressed vowels) \
               that DOES make the sentence grammatical and meaningful.
            3. The replacement is the highest-probability word given the rest of the sentence.

            If any of those three is unclear, KEEP THE ORIGINAL. The cost of a wrong correction is higher than \
            the cost of an awkward sentence. When uncertain, prefer the literal transcript.

            Examples of corrections you SHOULD make:
            - \"let's remove all we're gonna code\" → \"let's remove all redundant code\" \
              (\"we're gonna code\" is nonsensical; \"redundant code\" is grammatical and phonetically plausible)
            - \"click the sign-in but in\" → \"click the sign-in button\"
            - \"the API endpoint is on port ate-oh-ate-oh\" → \"the API endpoint is on port 8080\"
            - \"deploy to the staging environment using cube control\" → \"...using kubectl\"

            Examples where you must NOT correct:
            - Unusual but valid word choices (\"the meeting was lugubrious\" — leave it)
            - Stylistic choices the speaker may have intended (\"I want to pivot, hard\" — leave it)
            - Anything where you cannot identify a clearly better word

            Beyond word-level fixes, also apply standard cleanup: punctuation, capitalization, filler-word removal, \
            voice-command execution, and the other rules from above. Do NOT change meaning, condense, or paraphrase.
            """
        case .aggressive:
            return """
            Transform the transcribed speech into clear, professional written text. \
            Rewrite for clarity and conciseness while preserving the core meaning. \
            Fix all grammar issues, restructure awkward sentences, remove all filler words and verbal tics, \
            and optimize for readability. The output should read as polished written communication. \
            Capitalize proper nouns including names, places, brands, app names, UI elements, and technical terms. \
            Ensure proper spacing: one space after periods, commas, and other punctuation.
            """
        }
    }
}

// MARK: - Spacing Mode

enum SpacingMode: String, CaseIterable {
    case contextAware = "contextAware"
    case smart = "smart"
    case always = "always"
    case trailing = "trailing"

    var displayName: String {
        switch self {
        case .contextAware: return "Context-Aware"
        case .smart: return "Smart"
        case .always: return "Always"
        case .trailing: return "Trailing"
        }
    }

    var description: String {
        switch self {
        case .contextAware: return "Read cursor context to determine spacing (recommended)"
        case .smart: return "Add leading space if text starts with a letter"
        case .always: return "Always add a leading space"
        case .trailing: return "Add trailing space after each transcription"
        }
    }

    /// Apply spacing to the transcribed text
    func apply(to text: String) -> String {
        switch self {
        case .contextAware:
            // Try to read character before cursor using Accessibility API
            let result = CursorContext.getCharacterBeforeCursor()

            switch result {
            case .character(let charBefore):
                // Successfully read character - add space if previous char is not whitespace
                // Skip space after dashes — em dash, en dash, hyphen concatenate with next word
                let isDash = charBefore == "\u{2014}" || charBefore == "\u{2013}" || charBefore == "-"
                if !charBefore.isWhitespace && !charBefore.isNewline && !isDash {
                    if let first = text.first, first.isLetter || first.isNumber {
                        return " " + text
                    }
                }
                return text
            case .atStart:
                // Cursor is at start of field - no space needed
                return text
            case .unavailable:
                // Can't determine context (common in browsers, web apps).
                // Don't add a leading space — the trailing space from the previous
                // dictation already ensures word separation. Adding one here would
                // cause a leading space in empty fields or double spaces after
                // consecutive dictations.
                return text
            }
        case .smart:
            // Add leading space if text starts with a letter
            if let first = text.first, first.isLetter {
                return " " + text
            }
            return text
        case .always:
            // Always add leading space
            return " " + text
        case .trailing:
            // Add trailing space
            return text + " "
        }
    }
}

// MARK: - Punctuation Options

/// Individual punctuation detection features that can be toggled
enum PunctuationOption: String, CaseIterable {
    case voiceCommands = "voiceCommands"
    case pauseAnalysis = "pauseAnalysis"
    case pitchAnalysis = "pitchAnalysis"
    case llmHints = "llmHints"

    var displayName: String {
        switch self {
        case .voiceCommands: return "Voice Commands"
        case .pauseAnalysis: return "Pause Detection"
        case .pitchAnalysis: return "Pitch Analysis"
        case .llmHints: return "LLM Hints"
        }
    }

    var description: String {
        switch self {
        case .voiceCommands: return "Say 'period', 'comma', 'question mark' for punctuation"
        case .pauseAnalysis: return "Detect pauses to infer sentence boundaries"
        case .pitchAnalysis: return "Detect rising pitch for questions"
        case .llmHints: return "Pass prosody hints to LLM for better decisions"
        }
    }

    var defaultEnabled: Bool {
        switch self {
        case .voiceCommands: return true
        case .pauseAnalysis: return true
        case .pitchAnalysis: return true
        case .llmHints: return true
        }
    }
}

// MARK: - Application Context Detection

/// Detected application context for formatting
enum AppContext: String {
    case email = "email"
    case slack = "slack"
    case code = "code"
    case general = "default"

    /// Additional context hint to append to the system prompt
    var contextHint: String {
        switch self {
        case .email:
            return """

            [APPLICATION CONTEXT: Email]
            Format for professional email communication:
            - Use proper greeting and sign-off if the content suggests a complete email
            - Use appropriate paragraph breaks between distinct topics
            - Keep tone professional but natural
            - Format lists cleanly if the speaker lists items
            """
        case .slack:
            return """

            [APPLICATION CONTEXT: Slack/Chat]
            Format for casual team communication:
            - Keep it conversational
            - Convert "[word/phrase] emoji" to Slack colon format: "thumbs up emoji" → :thumbs_up:, "fire emoji" → :fire:, "heart emoji" → :heart:
            - Common emoji mappings: thumbs up → :+1:, thumbs down → :-1:, smile/smiley → :smile:, laugh → :joy:, sad → :cry:, heart → :heart:, fire → :fire:, rocket → :rocket:, check/checkmark → :white_check_mark:, x/cross → :x:, eyes → :eyes:, thinking → :thinking_face:, party → :tada:, clap → :clap:, wave → :wave:, pray/thanks → :pray:
            - No formal greetings needed
            """
        case .code:
            return """

            [APPLICATION CONTEXT: Code Editor / Terminal]
            Format for code-related communication:
            - Be technical and precise
            - Preserve exact project names, package names, and identifiers as they appear on screen
            - When a spoken word is phonetically close to a technical term visible on screen, use the on-screen spelling
            - Variable names should be camelCase or snake_case as appropriate
            - Keep explanations concise
            """
        case .general:
            return ""
        }
    }
}

/// Detects the frontmost application and returns appropriate context
struct AppContextDetector {
    /// Map of bundle identifiers to app contexts
    private static let bundleContextMap: [String: AppContext] = [
        // Email apps
        "com.microsoft.Outlook": .email,
        "com.apple.mail": .email,
        "com.google.Chrome": .general,  // Could be Gmail, handled separately
        "com.readdle.smartemail-Mac": .email,
        "com.freron.MailMate": .email,
        "com.postbox-inc.postbox": .email,

        // Chat/Slack apps
        "com.tinyspeck.slackmacgap": .slack,
        "com.hnc.Discord": .slack,
        "com.microsoft.teams2": .slack,
        "ru.keepcoder.Telegram": .slack,
        "net.whatsapp.WhatsApp": .slack,
        "com.facebook.archon.developerID": .slack, // Messenger

        // Code editors
        "com.microsoft.VSCode": .code,
        "com.apple.dt.Xcode": .code,
        "com.sublimetext.4": .code,
        "com.jetbrains.intellij": .code,
        "com.googlecode.iterm2": .code,
        "com.apple.Terminal": .code,
        "com.cursor.Cursor": .code,
        "dev.zed.Zed": .code,
        "com.todesktop.230313mzl4w4u92": .code, // Cursor
    ]

    /// App names (partial match) to contexts - fallback for unknown bundle IDs
    private static let appNameContextMap: [(String, AppContext)] = [
        ("outlook", .email),
        ("mail", .email),
        ("gmail", .email),
        ("slack", .slack),
        ("discord", .slack),
        ("teams", .slack),
        ("telegram", .slack),
        ("whatsapp", .slack),
        ("messages", .slack),
        ("xcode", .code),
        ("code", .code),  // VS Code, Cursor
        ("terminal", .code),
        ("iterm", .code),
        ("sublime", .code),
        ("intellij", .code),
        ("pycharm", .code),
        ("webstorm", .code),
        ("cursor", .code),
        ("zed", .code),
    ]

    /// Detect the context based on the frontmost application
    static func detectContext() -> AppContext {
        guard let frontmostApp = NSWorkspace.shared.frontmostApplication else {
            return .general
        }

        let bundleID = frontmostApp.bundleIdentifier ?? ""
        let appName = frontmostApp.localizedName ?? ""
        return detectContext(for: bundleID, appName: appName)
    }

    /// Detect context for explicit bundle ID and app name (used by AppProfileManager)
    static func detectContext(for bundleID: String, appName: String) -> AppContext {
        // First try bundle identifier (most reliable)
        if let context = bundleContextMap[bundleID] {
            return context
        }

        // Fall back to app name matching
        let lowerName = appName.lowercased()
        for (namePattern, context) in appNameContextMap {
            if lowerName.contains(namePattern) {
                return context
            }
        }

        return .general
    }
}

// MARK: - Cursor Context (Accessibility API)

/// Result of attempting to read cursor context
enum CursorContextResult {
    case character(Character)  // Successfully read the character before cursor
    case atStart               // Cursor is at position 0 (start of field)
    case unavailable           // Can't determine context (no permission, unsupported app, etc.)
}

/// Helper to read cursor context using macOS Accessibility API
struct CursorContext {
    /// Get the character immediately before the cursor in the focused text field
    static func getCharacterBeforeCursor() -> CursorContextResult {
        // Check accessibility permission first
        guard AXIsProcessTrusted() else {
            return .unavailable
        }

        // Get the system-wide accessibility element
        let systemWide = AXUIElementCreateSystemWide()

        // Get the focused UI element
        var focusedElementRef: CFTypeRef?
        let focusedError = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedElementRef
        )

        guard focusedError == .success,
              let focusedElement = focusedElementRef else {
            return .unavailable
        }

        let element = focusedElement as! AXUIElement

        // Get the selected text range (this gives us cursor position)
        var selectedRangeRef: CFTypeRef?
        let rangeError = AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            &selectedRangeRef
        )

        guard rangeError == .success,
              let rangeValue = selectedRangeRef else {
            return .unavailable
        }

        // Extract the range
        var range = CFRange(location: 0, length: 0)
        guard AXValueGetValue(rangeValue as! AXValue, .cfRange, &range) else {
            return .unavailable
        }

        // Cursor position is at range.location
        let cursorPosition = range.location

        // If cursor is at the beginning, there's no character before it
        guard cursorPosition > 0 else {
            return .atStart
        }

        // Get the character before cursor using parameterized attribute
        var charRange = CFRange(location: cursorPosition - 1, length: 1)

        guard let rangeParam = AXValueCreate(.cfRange, &charRange) else {
            return .unavailable
        }

        var charRef: CFTypeRef?
        let charError = AXUIElementCopyParameterizedAttributeValue(
            element,
            kAXStringForRangeParameterizedAttribute as CFString,
            rangeParam,
            &charRef
        )

        guard charError == .success,
              let charString = charRef as? String,
              let char = charString.first else {
            return .unavailable
        }

        return .character(char)
    }

    /// Get the text content before the cursor in the focused text field (up to `maxLength` chars).
    /// Returns nil if accessibility is unavailable or no text field is focused.
    static func getTextBeforeCursor(maxLength: Int = 500) -> String? {
        guard AXIsProcessTrusted() else { return nil }

        let systemWide = AXUIElementCreateSystemWide()

        var focusedElementRef: CFTypeRef?
        let focusedError = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedElementRef
        )

        guard focusedError == .success,
              let focusedElement = focusedElementRef else {
            return nil
        }

        let element = focusedElement as! AXUIElement

        // Get cursor position from selected text range
        var selectedRangeRef: CFTypeRef?
        let rangeError = AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            &selectedRangeRef
        )

        guard rangeError == .success,
              let rangeValue = selectedRangeRef else {
            return nil
        }

        var range = CFRange(location: 0, length: 0)
        guard AXValueGetValue(rangeValue as! AXValue, .cfRange, &range) else {
            return nil
        }

        let cursorPosition = range.location
        guard cursorPosition > 0 else { return nil }

        // Read up to maxLength characters before the cursor
        let readLength = min(cursorPosition, maxLength)
        let startPos = cursorPosition - readLength
        var textRange = CFRange(location: startPos, length: readLength)

        guard let rangeParam = AXValueCreate(.cfRange, &textRange) else {
            return nil
        }

        var textRef: CFTypeRef?
        let textError = AXUIElementCopyParameterizedAttributeValue(
            element,
            kAXStringForRangeParameterizedAttribute as CFString,
            rangeParam,
            &textRef
        )

        guard textError == .success,
              let text = textRef as? String,
              !text.isEmpty else {
            return nil
        }

        return text
    }

    /// Get the cursor position (character offset) in the focused text field.
    /// Returns nil if accessibility is unavailable or no text field is focused.
    static func getCursorPosition() -> Int? {
        guard AXIsProcessTrusted() else { return nil }

        let systemWide = AXUIElementCreateSystemWide()

        var focusedElementRef: CFTypeRef?
        let focusedError = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedElementRef
        )

        guard focusedError == .success,
              let focusedElement = focusedElementRef else {
            return nil
        }

        let element = focusedElement as! AXUIElement

        var selectedRangeRef: CFTypeRef?
        let rangeError = AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            &selectedRangeRef
        )

        guard rangeError == .success,
              let rangeValue = selectedRangeRef else {
            return nil
        }

        var range = CFRange(location: 0, length: 0)
        guard AXValueGetValue(rangeValue as! AXValue, .cfRange, &range) else {
            return nil
        }

        return range.location
    }

    /// Set the text selection in the focused text field via AX API.
    /// Returns true on success.
    static func selectRange(location: Int, length: Int) -> Bool {
        guard AXIsProcessTrusted() else { return false }

        let systemWide = AXUIElementCreateSystemWide()

        var focusedElementRef: CFTypeRef?
        let focusedError = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedElementRef
        )

        guard focusedError == .success,
              let focusedElement = focusedElementRef else {
            return false
        }

        let element = focusedElement as! AXUIElement

        var range = CFRange(location: location, length: length)
        guard let rangeValue = AXValueCreate(.cfRange, &range) else {
            return false
        }

        let result = AXUIElementSetAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            rangeValue
        )

        return result == .success
    }
}

// MARK: - App Delegate

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var voiceFlow = VoiceFlowBridge()
    var audioRecorder = AudioRecorder()
    var hotkeyManager = GlobalHotkeyManager()
    var overlayPanel: NSPanel?
    var overlayHostingView: NSHostingView<RecordingOverlayView>?
    var overlayState = OverlayState()
    var settingsWindow: NSWindow?
    var wizardController: SetupWizardController?
    private var audioLevelCancellable: AnyCancellable?
    private var screenshotCapture = ScreenshotCapture()
    private var isVisualContextRecording = false

    // Moonshine streaming engine (Beta)
    var moonshineEngine = MoonshineStreamingEngine()
    private var streamingTextCancellable: AnyCancellable?
    private var isStreamingActive = false

    // Parakeet-MLX engine (batch STT alternative). When enabled via env var
    // VOICEFLOW_USE_PARAKEET=1, replaces Moonshine on the recording-release path.
    var parakeetEngine = ParakeetASREngine()
    var useParakeet: Bool {
        ProcessInfo.processInfo.environment["VOICEFLOW_USE_PARAKEET"] == "1"
    }

    // Direct-to-field streaming state
    private var lastStreamedText: String = ""
    private var dictationInsertedLength: Int = 0
    private var dictationStartPosition: Int? = nil
    private var directTypingActive: Bool = false
    private var dictationTargetApp: NSRunningApplication? = nil

    // AI Result panel
    var aiResultPanel: NSPanel?
    var aiResultHostingView: NSHostingView<AIResultView>?
    var aiResultState = AIResultState()
    private var aiResultLocalMonitor: Any?
    private var aiResultGlobalMonitor: Any?
    private var pendingAICommand: AICommand?
    private var pendingAITargetApp: NSRunningApplication?

    private var lastPastedText: String?
    private var isRecording = false
    private var recordingMenuItem: NSMenuItem?
    private var formattingMenuItems: [FormattingLevel: NSMenuItem] = [:]
    private var spacingMenuItems: [SpacingMode: NSMenuItem] = [:]
    private var punctuationMenuItems: [PunctuationOption: NSMenuItem] = [:]

    // User preference for formatting level
    var formattingLevel: FormattingLevel {
        get {
            let rawValue = UserDefaults.standard.string(forKey: "formattingLevel") ?? FormattingLevel.moderate.rawValue
            return FormattingLevel(rawValue: rawValue) ?? .moderate
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: "formattingLevel")
            updateFormattingMenuChecks()
        }
    }

    // User preference for spacing mode
    var spacingMode: SpacingMode {
        get {
            let rawValue = UserDefaults.standard.string(forKey: "spacingMode") ?? SpacingMode.contextAware.rawValue
            return SpacingMode(rawValue: rawValue) ?? .contextAware
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: "spacingMode")
            updateSpacingMenuChecks()
        }
    }

    // User preference for streaming transcription (Beta)
    var streamingEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: "streamingEnabled") }
        set { UserDefaults.standard.set(newValue, forKey: "streamingEnabled") }
    }

    var streamingModelSize: MoonshineStreamingEngine.StreamingModelSize {
        get {
            let raw = UserDefaults.standard.string(forKey: "streamingModelSize") ?? "small"
            return MoonshineStreamingEngine.StreamingModelSize(rawValue: raw) ?? .small
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: "streamingModelSize")
        }
    }

    // User preferences for punctuation options
    func isPunctuationOptionEnabled(_ option: PunctuationOption) -> Bool {
        let key = "punctuation_\(option.rawValue)"
        if UserDefaults.standard.object(forKey: key) == nil {
            return option.defaultEnabled
        }
        return UserDefaults.standard.bool(forKey: key)
    }

    func setPunctuationOption(_ option: PunctuationOption, enabled: Bool) {
        let key = "punctuation_\(option.rawValue)"
        UserDefaults.standard.set(enabled, forKey: key)
        updatePunctuationMenuChecks()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Hide dock icon - menu bar only
        NSApp.setActivationPolicy(.accessory)

        // Offer to move to /Applications if running from elsewhere (e.g. Downloads)
        offerRelocationIfNeeded()

        if !SetupWizardController.isSetupComplete {
            showSetupWizard()
        } else {
            proceedWithNormalLaunch()
        }
    }

    /// If the app is not in /Applications, offer to move it there.
    private func offerRelocationIfNeeded() {
        let appPath = Bundle.main.bundlePath
        let applicationsDir = "/Applications"

        // Already in /Applications — nothing to do
        if appPath.hasPrefix(applicationsDir) { return }

        // Don't nag on every launch — only ask once per location
        let lastAskedPath = UserDefaults.standard.string(forKey: "relocateLastAskedPath")
        if lastAskedPath == appPath { return }
        UserDefaults.standard.set(appPath, forKey: "relocateLastAskedPath")

        // Temporarily show in dock so the dialog is visible
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)

        let alert = NSAlert()
        alert.messageText = "Move to Applications?"
        alert.informativeText = "VoiceFlow works best when run from your Applications folder. Would you like to move it there now?"
        alert.alertStyle = .informational
        alert.addButton(withTitle: "Move to Applications")
        alert.addButton(withTitle: "Not Now")

        let response = alert.runModal()

        // Restore menu-bar-only mode
        NSApp.setActivationPolicy(.accessory)

        guard response == .alertFirstButtonReturn else { return }

        let appName = (appPath as NSString).lastPathComponent
        let destPath = (applicationsDir as NSString).appendingPathComponent(appName)

        do {
            let fm = FileManager.default
            // Remove existing copy if present
            if fm.fileExists(atPath: destPath) {
                try fm.removeItem(atPath: destPath)
            }
            // Copy to /Applications (source may be on a read-only volume)
            try fm.copyItem(atPath: appPath, toPath: destPath)

            // Best-effort cleanup of the original (fails silently on read-only volumes)
            try? fm.removeItem(atPath: appPath)

            // Strip quarantine xattr so macOS doesn't re-prompt
            let stripProcess = Process()
            stripProcess.executableURL = URL(fileURLWithPath: "/usr/bin/xattr")
            stripProcess.arguments = ["-dr", "com.apple.quarantine", destPath]
            try? stripProcess.run()
            stripProcess.waitUntilExit()

            // Relaunch from /Applications after a short delay to let this instance exit
            let script = """
            sleep 0.5; open "\(destPath)"
            """
            let relaunch = Process()
            relaunch.executableURL = URL(fileURLWithPath: "/bin/sh")
            relaunch.arguments = ["-c", script]
            try relaunch.run()

            NSApp.terminate(nil)
        } catch {
            // Reset the "asked" flag so the user can try again next launch
            UserDefaults.standard.removeObject(forKey: "relocateLastAskedPath")

            let errAlert = NSAlert()
            errAlert.messageText = "Could Not Move App"
            errAlert.informativeText = "Failed to move VoiceFlow to Applications: \(error.localizedDescription)\n\nYou can move it manually by dragging VoiceFlow.app into your Applications folder."
            errAlert.alertStyle = .warning
            errAlert.addButton(withTitle: "OK")
            errAlert.runModal()
        }
    }

    private func showSetupWizard() {
        let controller = SetupWizardController()
        wizardController = controller
        controller.show { [weak self] in
            self?.wizardController = nil
            self?.proceedWithNormalLaunch()
        }
    }

    private func proceedWithNormalLaunch() {
        setupStatusItem()
        setupHotkey()
        setupOverlay()
        setupAIResultPanel()

        // Request notification permission
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }

        // Check accessibility permission (required for auto-paste)
        checkAccessibilityPermission()

        // Request microphone permission (entitlements ensure the system dialog appears)
        if AVCaptureDevice.authorizationStatus(for: .audio) != .authorized {
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                if !granted {
                    DispatchQueue.main.async {
                        self.showAlert(title: "Microphone Access Required",
                                       message: "VoiceFlow needs microphone access to transcribe your speech. Please go to System Settings > Privacy & Security > Microphone and enable VoiceFlow.")
                    }
                }
            }
        }

        // Listen for learned corrections and inject into Rust pipeline
        NotificationCenter.default.addObserver(
            forName: .correctionLearned,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let self = self,
                  let info = notification.userInfo,
                  let original = info["original"] as? String,
                  let corrected = info["corrected"] as? String else { return }
            MainActor.assumeIsolated {
                self.voiceFlow.addReplacement(original: original, corrected: corrected)
            }
        }

        // Listen for removed corrections and remove from Rust pipeline
        NotificationCenter.default.addObserver(
            forName: .correctionRemoved,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let self = self,
                  let info = notification.userInfo,
                  let original = info["original"] as? String else { return }
            MainActor.assumeIsolated {
                self.voiceFlow.removeReplacement(original: original)
            }
        }

        // Auto-load Moonshine streaming model if enabled and downloaded.
        // Skipped when Parakeet is selected — Parakeet replaces both Moonshine
        // streaming and the built-in STT for the recording-release path.
        if streamingEnabled && !useParakeet {
            let size = streamingModelSize
            if MoonshineStreamingEngine.isModelDownloaded(size) {
                do {
                    try moonshineEngine.loadModel(size: size)
                    NSLog("[VoiceFlow] Streaming model loaded at launch: %@", size.rawValue)
                    // Unload the built-in STT engine — streaming replaces it
                    voiceFlow.unloadStt()
                    NSLog("[VoiceFlow] STT engine unloaded (streaming replaces it)")
                } catch {
                    NSLog("[VoiceFlow] Failed to load streaming model at launch: %@", error.localizedDescription)
                }
            }
        }

        // Auto-load Parakeet-MLX if selected (batch STT, replaces Moonshine + built-in STT)
        if useParakeet {
            voiceFlow.unloadStt()
            NSLog("[VoiceFlow] STT engine unloaded (Parakeet replaces it)")
            Task { @MainActor in
                await parakeetEngine.loadModel()
                if case .ready = parakeetEngine.state {
                    NSLog("[VoiceFlow] Parakeet model ready: %@", parakeetEngine.modelId)
                } else {
                    NSLog("[VoiceFlow] Parakeet load state: %@", String(describing: parakeetEngine.state))
                }
            }
        }

        // First-run: scan installed apps and assign personas based on the curated table.
        // Idempotent — only assigns to profiles that have no persona/custom prompt set.
        if !UserDefaults.standard.bool(forKey: "voiceflow.autoMapInstalledAppsCompleted") {
            let count = AppProfileManager.shared.autoMapInstalledApps()
            UserDefaults.standard.set(true, forKey: "voiceflow.autoMapInstalledAppsCompleted")
            NSLog("[VoiceFlow] Auto-mapped %d apps to personas", count)
        }

        // Touch BrowserSiteRulesManager so it loads/seeds on first launch (the
        // seed pass needs PersonaManager.shared to already be populated, which
        // is true after the persona-touching code above runs).
        let ruleCount = BrowserSiteRulesManager.shared.rules.count
        NSLog("[VoiceFlow] Browser site rules loaded: %d", ruleCount)
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Clean up AI result key monitors
        removeAIResultKeyMonitors()

        // Clean up streaming engine
        moonshineEngine.unloadModel()

        // Stop Parakeet daemon if it's running
        if useParakeet {
            parakeetEngine.stopDaemon()
        }

        // Clean up resources before termination to prevent memory leaks
        // This ensures the Rust side properly releases all model memory
        voiceFlow.cleanup()

        // Log final memory state for debugging
        let memUsage = VoiceFlowBridge.getMemoryUsage()
        print("VoiceFlow terminating - Final memory: \(Int(memUsage.residentMB))MB resident, \(Int(memUsage.peakMB))MB peak")
    }

    private func checkAccessibilityPermission() {
        // Check if accessibility is already granted
        let trusted = AXIsProcessTrusted()

        if !trusted {
            // Show alert explaining why we need accessibility
            let alert = NSAlert()
            alert.messageText = "Accessibility Permission Required"
            alert.informativeText = "VoiceFlow needs Accessibility permission to automatically paste transcribed text.\n\nClick 'Open Settings' to grant permission, then restart VoiceFlow."
            alert.alertStyle = .warning
            alert.addButton(withTitle: "Open Settings")
            alert.addButton(withTitle: "Later")

            let response = alert.runModal()

            if response == .alertFirstButtonReturn {
                // Open System Settings to Accessibility pane
                let prefPaneURL = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
                NSWorkspace.shared.open(prefPaneURL)
            }
        }
    }

    // MARK: - Status Bar Setup

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        updateMenuBarIcon()

        // Observe appearance changes for light/dark mode
        DistributedNotificationCenter.default.addObserver(
            self,
            selector: #selector(appearanceChanged),
            name: NSNotification.Name("AppleInterfaceThemeChangedNotification"),
            object: nil
        )

        setupMenu()
    }

    @objc private func appearanceChanged() {
        updateMenuBarIcon()
    }

    private func updateMenuBarIcon() {
        guard let button = statusItem.button else { return }

        // Use single white icon (works for both light and dark menu bars)
        if let iconPath = Bundle.main.path(forResource: "MenuBarIcon", ofType: "png"),
           let icon = NSImage(contentsOfFile: iconPath) {
            icon.size = NSSize(width: 18, height: 18)
            button.image = icon
        } else {
            // Fallback to SF Symbol
            button.image = NSImage(systemSymbolName: "mic.fill", accessibilityDescription: "VoiceFlow")
        }
    }

    private func setupMenu() {
        let menu = NSMenu()

        let statusMenuItem = NSMenuItem(title: "VoiceFlow v\(VoiceFlowBridge.version)", action: nil, keyEquivalent: "")
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)

        menu.addItem(NSMenuItem.separator())

        recordingMenuItem = NSMenuItem(title: "Hold ⌥ Space to Record", action: #selector(toggleRecording), keyEquivalent: "")
        recordingMenuItem?.target = self
        menu.addItem(recordingMenuItem!)

        menu.addItem(NSMenuItem.separator())

        // Formatting Level submenu
        let formattingMenu = NSMenu()
        let formattingMenuItem = NSMenuItem(title: "Formatting", action: nil, keyEquivalent: "")
        formattingMenuItem.submenu = formattingMenu

        for level in FormattingLevel.allCases {
            let item = NSMenuItem(
                title: level.displayName,
                action: #selector(selectFormattingLevel(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = level
            item.toolTip = level.description
            formattingMenu.addItem(item)
            formattingMenuItems[level] = item
        }

        menu.addItem(formattingMenuItem)
        updateFormattingMenuChecks()

        // Punctuation submenu
        let punctuationMenu = NSMenu()
        let punctuationMenuItem = NSMenuItem(title: "Punctuation", action: nil, keyEquivalent: "")
        punctuationMenuItem.submenu = punctuationMenu

        for option in PunctuationOption.allCases {
            let item = NSMenuItem(
                title: option.displayName,
                action: #selector(togglePunctuationOption(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = option
            item.toolTip = option.description
            punctuationMenu.addItem(item)
            punctuationMenuItems[option] = item
        }

        menu.addItem(punctuationMenuItem)
        updatePunctuationMenuChecks()

        // Spacing Mode submenu
        let spacingMenu = NSMenu()
        let spacingMenuItem = NSMenuItem(title: "Spacing", action: nil, keyEquivalent: "")
        spacingMenuItem.submenu = spacingMenu

        for mode in SpacingMode.allCases {
            let item = NSMenuItem(
                title: mode.displayName,
                action: #selector(selectSpacingMode(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = mode
            item.toolTip = mode.description
            spacingMenu.addItem(item)
            spacingMenuItems[mode] = item
        }

        menu.addItem(spacingMenuItem)
        updateSpacingMenuChecks()

        menu.addItem(NSMenuItem.separator())

        // Permissions submenu
        let permissionsMenu = NSMenu()
        let permissionsMenuItem = NSMenuItem(title: "Permissions", action: nil, keyEquivalent: "")
        permissionsMenuItem.submenu = permissionsMenu

        let accessibilityItem = NSMenuItem(
            title: "Accessibility (for paste)",
            action: #selector(openAccessibilitySettings),
            keyEquivalent: ""
        )
        accessibilityItem.target = self
        permissionsMenu.addItem(accessibilityItem)

        let microphoneItem = NSMenuItem(
            title: "Microphone (for recording)",
            action: #selector(openMicrophoneSettings),
            keyEquivalent: ""
        )
        microphoneItem.target = self
        permissionsMenu.addItem(microphoneItem)

        menu.addItem(permissionsMenuItem)

        menu.addItem(NSMenuItem.separator())

        let settingsItem = NSMenuItem(title: "Settings...", action: #selector(openSettings), keyEquivalent: ",")
        settingsItem.target = self
        menu.addItem(settingsItem)

        menu.addItem(NSMenuItem.separator())

        let quitItem = NSMenuItem(title: "Quit VoiceFlow", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem.menu = menu
    }

    @objc private func openSettings() {
        // If window already exists, just bring it to front
        if let window = settingsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        // Create settings window
        let settingsView = SettingsView()
            .environmentObject(voiceFlow)
            .environmentObject(SnippetManager.shared)

        let hostingController = NSHostingController(rootView: settingsView)

        let window = NSWindow(contentViewController: hostingController)
        window.title = "VoiceFlow"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.setContentSize(NSSize(width: 780, height: 560))
        window.minSize = NSSize(width: 680, height: 480)
        window.center()
        window.isReleasedWhenClosed = false

        settingsWindow = window
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func openAccessibilitySettings() {
        // Show Apple's native accessibility permission prompt
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
    }

    @objc private func openMicrophoneSettings() {
        // Activate so the system dialog appears on top for LSUIElement apps
        NSApp.activate(ignoringOtherApps: true)
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            if !granted {
                // If denied or dialog didn't appear, open System Settings
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                    NSWorkspace.shared.open(url)
                }
            }
        }
    }

    @objc private func selectFormattingLevel(_ sender: NSMenuItem) {
        guard let level = sender.representedObject as? FormattingLevel else { return }
        formattingLevel = level
    }

    private func updateFormattingMenuChecks() {
        for (level, item) in formattingMenuItems {
            item.state = (level == formattingLevel) ? .on : .off
        }
    }

    @objc private func selectSpacingMode(_ sender: NSMenuItem) {
        guard let mode = sender.representedObject as? SpacingMode else { return }
        spacingMode = mode
    }

    private func updateSpacingMenuChecks() {
        for (mode, item) in spacingMenuItems {
            item.state = (mode == spacingMode) ? .on : .off
        }
    }

    @objc private func togglePunctuationOption(_ sender: NSMenuItem) {
        guard let option = sender.representedObject as? PunctuationOption else { return }
        let currentlyEnabled = isPunctuationOptionEnabled(option)
        setPunctuationOption(option, enabled: !currentlyEnabled)
    }

    private func updatePunctuationMenuChecks() {
        for (option, item) in punctuationMenuItems {
            item.state = isPunctuationOptionEnabled(option) ? .on : .off
        }
    }

    // MARK: - Overlay Setup

    private func setupOverlay() {
        // Card envelope: 420 wide, up to 40% of screen height + padding for status bar
        let screen = NSScreen.main
        let screenHeight = screen?.visibleFrame.height ?? 800
        let panelWidth: CGFloat = 420
        let panelHeight: CGFloat = screenHeight * 0.4 + 80

        // Position at bottom center of screen, above the dock
        let screenFrame = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let panelX = screenFrame.midX - panelWidth / 2
        let panelY = screenFrame.minY + 80  // bottom edge 80pt above dock area

        let panel = NSPanel(
            contentRect: NSRect(x: panelX, y: panelY, width: panelWidth, height: panelHeight),
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )

        panel.level = .floating
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isMovableByWindowBackground = false
        panel.titlebarAppearsTransparent = true
        panel.titleVisibility = .hidden

        // Create SwiftUI hosting view
        let overlayView = RecordingOverlayView(state: overlayState)
        let hostingView = NSHostingView(rootView: overlayView)
        hostingView.frame = NSRect(x: 0, y: 0, width: panelWidth, height: panelHeight)

        panel.contentView = hostingView

        self.overlayPanel = panel
        self.overlayHostingView = hostingView

        // Reposition overlay when displays are added/removed
        NotificationCenter.default.addObserver(
            forName: NSApplication.didChangeScreenParametersNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self = self, let panel = self.overlayPanel, panel.isVisible else { return }
                self.repositionOverlay()
            }
        }
    }

    /// Returns the screen containing the mouse cursor, falling back to the main screen.
    private func activeScreen() -> NSScreen? {
        let mouseLocation = NSEvent.mouseLocation
        return NSScreen.screens.first { NSMouseInRect(mouseLocation, $0.frame, false) }
            ?? NSScreen.main
    }

    /// Snaps the overlay panel to bottom-center of the active screen's visible area.
    private func repositionOverlay() {
        guard let panel = overlayPanel, let screen = activeScreen() else { return }
        let screenFrame = screen.visibleFrame
        let panelFrame = panel.frame
        let x = screenFrame.midX - panelFrame.width / 2
        let y = screenFrame.minY + 80  // 80pt above dock area
        panel.setFrameOrigin(NSPoint(x: x, y: y))
    }

    // MARK: - AI Result Panel

    private func setupAIResultPanel() {
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 480, height: 300),
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        panel.level = .floating
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isMovableByWindowBackground = true
        panel.titlebarAppearsTransparent = true
        panel.titleVisibility = .hidden

        let resultView = AIResultView(state: aiResultState)
        let hostingView = NSHostingView(rootView: resultView)
        hostingView.frame = NSRect(x: 0, y: 0, width: 480, height: 300)
        panel.contentView = hostingView

        self.aiResultPanel = panel
        self.aiResultHostingView = hostingView
    }

    private func showAIResult(command: AICommand, text: String, targetApp: NSRunningApplication?) {
        pendingAICommand = command
        pendingAITargetApp = targetApp
        aiResultState.show(title: command.resultTitle, text: text, hint: command.pasteHint)

        // Center on active screen
        if let screen = activeScreen() {
            let screenFrame = screen.visibleFrame
            let panelW: CGFloat = 480
            let panelH: CGFloat = 300
            let x = screenFrame.midX - panelW / 2
            let y = screenFrame.midY - panelH / 2
            aiResultPanel?.setFrame(NSRect(x: x, y: y, width: panelW, height: panelH), display: true)
        }

        aiResultPanel?.orderFront(nil)
        installAIResultKeyMonitors()
    }

    private func hideAIResult() {
        aiResultPanel?.orderOut(nil)
        aiResultState.hide()
        removeAIResultKeyMonitors()
        pendingAICommand = nil
        pendingAITargetApp = nil
    }

    private func acceptAIResult() {
        guard let command = pendingAICommand else { return }
        let resultText = aiResultState.resultText
        let targetApp = pendingAITargetApp

        hideAIResult()

        Task { @MainActor in
            // Save clipboard
            let savedClipboard = NSPasteboard.general.string(forType: .string)

            // Activate target app
            if let app = targetApp {
                app.activate()
            }
            try? await Task.sleep(nanoseconds: 200_000_000) // 200ms

            // Put result on clipboard
            let pasteText: String
            if case .appendToEnd = command.pasteMode {
                pasteText = "\n\n" + resultText
            } else {
                pasteText = resultText
            }

            let pb = NSPasteboard.general
            pb.clearContents()
            pb.setString(pasteText, forType: .string)
            try? await Task.sleep(nanoseconds: 100_000_000) // 100ms

            // Paste based on mode
            switch command.pasteMode {
            case .atCursor:
                simulatePaste()
            case .replaceAll:
                simulateSelectAll()
                try? await Task.sleep(nanoseconds: 100_000_000)
                simulatePaste()
            case .appendToEnd:
                simulateEndOfDocument()
                try? await Task.sleep(nanoseconds: 100_000_000)
                simulatePaste()
            }

            // Restore clipboard after paste settles
            try? await Task.sleep(nanoseconds: 500_000_000) // 500ms
            let restorePb = NSPasteboard.general
            restorePb.clearContents()
            if let saved = savedClipboard {
                restorePb.setString(saved, forType: .string)
            }
        }
    }

    private func copyAIResult() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(aiResultState.resultText, forType: .string)
        showNotification(title: "Copied", body: "AI result copied to clipboard")
    }

    private func installAIResultKeyMonitors() {
        removeAIResultKeyMonitors()

        aiResultGlobalMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            MainActor.assumeIsolated {
                self?.handleAIResultKeyEvent(event)
            }
        }

        aiResultLocalMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            MainActor.assumeIsolated {
                self?.handleAIResultKeyEvent(event)
            }
            return event
        }
    }

    private func removeAIResultKeyMonitors() {
        if let monitor = aiResultGlobalMonitor {
            NSEvent.removeMonitor(monitor)
            aiResultGlobalMonitor = nil
        }
        if let monitor = aiResultLocalMonitor {
            NSEvent.removeMonitor(monitor)
            aiResultLocalMonitor = nil
        }
    }

    private func handleAIResultKeyEvent(_ event: NSEvent) {
        guard aiResultState.isVisible else { return }

        if event.keyCode == 36 { // Return
            acceptAIResult()
        } else if event.keyCode == 53 { // Escape
            hideAIResult()
        } else if event.keyCode == 8 && event.modifierFlags.contains(.command) { // Cmd+C
            copyAIResult()
        }
    }

    private func showOverlay(state: OverlayState.RecordingState) {
        overlayState.state = state
        // Resize panel based on state
        switch state {
        case .streaming, .formatting:
            // Card mode: large transparent envelope — SwiftUI handles visible size
            let maxH = (activeScreen()?.visibleFrame.height ?? 800) * 0.4 + 80
            overlayPanel?.setContentSize(NSSize(width: 420, height: maxH))
        case .aiProcessing:
            overlayPanel?.setContentSize(NSSize(width: 260, height: 70))
        default:
            overlayPanel?.setContentSize(NSSize(width: 200, height: 70))
        }
        repositionOverlay()
        overlayPanel?.orderFront(nil)
    }

    private func hideOverlay() {
        overlayPanel?.orderOut(nil)
        overlayState.state = .idle
    }

    // MARK: - Hotkey Setup

    private func setupHotkey() {
        // Option + Space: hold to record, release to stop and paste (text-only)
        hotkeyManager.register(
            id: 1,
            keyCode: UInt32(kVK_Space),
            modifiers: UInt32(optionKey),
            onPress: { [weak self] in
                DispatchQueue.main.async {
                    self?.isVisualContextRecording = false
                    self?.startRecording()
                }
            },
            onRelease: { [weak self] in
                DispatchQueue.main.async {
                    self?.stopRecordingAndPaste()
                }
            }
        )

        // Control + Option + Space: hold to record with visual context (screenshot + multimodal)
        hotkeyManager.register(
            id: 2,
            keyCode: UInt32(kVK_Space),
            modifiers: UInt32(controlKey | optionKey),
            onPress: { [weak self] in
                DispatchQueue.main.async {
                    self?.isVisualContextRecording = true
                    self?.startRecording()
                }
            },
            onRelease: { [weak self] in
                DispatchQueue.main.async {
                    self?.stopRecordingAndPaste()
                }
            }
        )
    }

    // MARK: - Recording Control

    @objc func toggleRecording() {
        if isRecording {
            stopRecordingAndPaste()
        } else {
            startRecording()
        }
    }

    private func startRecording() {
        guard !isRecording else { return }

        do {
            try audioRecorder.startRecording()
            isRecording = true
            recordingMenuItem?.title = "Recording... (release ⌥ Space)"

            // If streaming is enabled and model is loaded, start streaming session
            // (not applicable in audio-direct mode — audio goes directly to Gemma 4 on stop)
            if streamingEnabled && moonshineEngine.isLoaded && !voiceFlow.isAudioDirect {
                do {
                    // Capture cursor position and target app before starting session
                    dictationStartPosition = CursorContext.getCursorPosition()
                    dictationInsertedLength = 0
                    lastStreamedText = ""
                    dictationTargetApp = NSWorkspace.shared.frontmostApplication
                    directTypingActive = (dictationStartPosition != nil)

                    NSLog("[VoiceFlow] Direct typing: %@, cursor position: %@",
                          directTypingActive ? "active" : "inactive",
                          dictationStartPosition.map { String($0) } ?? "nil")

                    try moonshineEngine.beginSession()
                    isStreamingActive = true

                    // Wire audio chunks to streaming engine
                    // Audio chunks arrive from the audio tap thread — dispatch to main actor
                    let engine = moonshineEngine
                    audioRecorder.onAudioChunk = { chunk in
                        Task { @MainActor in
                            engine.feedAudioChunk(chunk)
                        }
                    }

                    // Subscribe to partial text updates
                    streamingTextCancellable = moonshineEngine.$partialText
                        .receive(on: DispatchQueue.main)
                        .sink { [weak self] text in
                            guard let self = self, self.isStreamingActive else { return }
                            let displayText = CorrectionManager.shared.applyStoredCorrections(text)

                            // Safety: verify focus hasn't changed
                            if self.directTypingActive {
                                if NSWorkspace.shared.frontmostApplication?.processIdentifier
                                    != self.dictationTargetApp?.processIdentifier {
                                    NSLog("[VoiceFlow] Focus changed during dictation, falling back to overlay")
                                    self.directTypingActive = false
                                    self.showOverlay(state: .streaming(displayText))
                                }
                            }

                            if self.directTypingActive {
                                let (deleteCount, insertText) = Self.computeStreamingDiff(
                                    previous: self.lastStreamedText, current: displayText)
                                if deleteCount > 0 {
                                    self.deleteBackward(count: deleteCount)
                                    self.dictationInsertedLength -= deleteCount
                                }
                                if !insertText.isEmpty {
                                    self.typeUnicodeText(insertText)
                                    self.dictationInsertedLength += insertText.count
                                }
                                self.lastStreamedText = displayText
                            } else {
                                self.overlayState.state = .streaming(displayText)
                            }
                        }

                    // Show recording pill when direct typing (text goes to field),
                    // or streaming overlay when falling back
                    showOverlay(state: directTypingActive ? .recording : .streaming(""))
                } catch {
                    NSLog("[VoiceFlow] Streaming session failed to start, falling back to normal: %@", error.localizedDescription)
                    isStreamingActive = false
                    directTypingActive = false
                    audioRecorder.onAudioChunk = nil
                    showOverlay(state: .recording)
                }
            } else {
                isStreamingActive = false
                directTypingActive = false
                showOverlay(state: .recording)
            }

            // Show overlay (already shown above based on streaming state)
            // showOverlay(state: .recording)  -- handled above

            // Subscribe to audio level changes
            audioLevelCancellable = audioRecorder.$audioLevel
                .receive(on: DispatchQueue.main)
                .sink { [weak self] level in
                    self?.overlayState.audioLevel = level
                }

            // Visual context: capture screenshot when using Control+Option+Space hotkey
            if isVisualContextRecording {
                if !ScreenshotCapture.hasPermission() {
                    let alert = NSAlert()
                    alert.messageText = "Screen Recording Permission Required"
                    alert.informativeText = "Visual context dictation requires Screen Recording permission. Please go to System Settings > Privacy & Security > Screen Recording and enable VoiceFlow."
                    alert.alertStyle = .informational
                    alert.addButton(withTitle: "Open System Settings")
                    alert.addButton(withTitle: "Continue Without")
                    let response = alert.runModal()
                    if response == .alertFirstButtonReturn {
                        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                    isVisualContextRecording = false
                }
            }
        } catch {
            showAlert(title: "Recording Error", message: error.localizedDescription)
        }
    }

    private func stopRecordingAndPaste() {
        guard isRecording else { return }

        // Cancel audio level and streaming subscriptions
        audioLevelCancellable?.cancel()
        audioLevelCancellable = nil
        streamingTextCancellable?.cancel()
        streamingTextCancellable = nil
        audioRecorder.onAudioChunk = nil

        let wasStreaming = isStreamingActive
        let wasDirectTyping = directTypingActive
        let insertedLength = dictationInsertedLength
        let startPosition = dictationStartPosition
        isStreamingActive = false

        // Reset direct typing state
        directTypingActive = false
        lastStreamedText = ""
        dictationInsertedLength = 0
        dictationStartPosition = nil
        dictationTargetApp = nil

        let audio = audioRecorder.stopRecording()
        isRecording = false
        recordingMenuItem?.title = "Hold ⌥ Space to Record"

        // Streaming path: get raw transcript from engine, then format deterministically
        if wasStreaming {
            let rawTranscript = moonshineEngine.endSession()

            guard !rawTranscript.isEmpty else {
                if wasDirectTyping && insertedLength > 0 {
                    // Clean up any partial text that was typed
                    deleteBackward(count: insertedLength)
                }
                hideOverlay()
                showNotification(title: "No Speech", body: "No speech was detected")
                return
            }

            // Streaming-release: tokens come back too fast on a fast LLM for the
            // word-by-word bar to be readable, so just show the simple processing
            // indicator until the result is ready to paste/correct.
            showOverlay(state: .processing)

            let frontmostApp = NSWorkspace.shared.frontmostApplication
            let _ = frontmostApp.flatMap { AppProfileManager.shared.ensureProfile(for: $0) }
            let appPrompt = AppProfileManager.shared.promptForApp(frontmostApp)
            let frontApp = frontmostApp?.localizedName ?? "Unknown"
            let inputFieldText = CursorContext.getTextBeforeCursor(maxLength: 500)
            let baseContext = formattingLevel.systemPrompt + appPrompt
            let currentSpacingMode = spacingMode

            // Determine mid-sentence state (same logic as non-streaming path)
            let terminalBundleIds: Set<String> = [
                "com.apple.Terminal", "com.googlecode.iterm2", "com.mitchellh.ghostty",
                "dev.warp.Warp-Stable", "io.alacritty", "com.github.wez.wezterm", "net.kovidgoyal.kitty",
            ]
            let isTerminal = terminalBundleIds.contains(frontmostApp?.bundleIdentifier ?? "")
            let isMidSentence: Bool
            if isTerminal {
                let lastOutput = self.lastPastedText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                isMidSentence = lastOutput.last != nil && !".?!".contains(String(lastOutput.last!))
            } else if let inputText = inputFieldText, !inputText.isEmpty {
                let lastChar = inputText.trimmingCharacters(in: .whitespacesAndNewlines).last
                isMidSentence = lastChar != nil && !".?!".contains(String(lastChar!))
            } else {
                isMidSentence = false
            }

            Task {
                var context = baseContext
                if let inputText = inputFieldText, !inputText.isEmpty {
                    context += """

                    [INPUT CONTEXT]
                    Text already in the input field (before the cursor):
                    \(inputText)
                    You are CONTINUING from this text. Rules:
                    - Do NOT repeat or rewrite any of the existing text.
                    - If the existing text ends mid-sentence (no final . ? ! or newline), your output must seamlessly continue the sentence. Do NOT capitalize your first word (unless it is "I" or a proper noun). Do NOT add a period at the end unless the thought is truly complete.
                    - If the existing text ends with sentence-ending punctuation (. ? !) or a newline, start a new sentence with a capital letter.
                    - Your output should read as a natural continuation — as if the existing text and your output were written together.
                    """
                }

                if isMidSentence {
                    context += "\n[MID_SENTENCE_CONTINUATION]"
                }

                let isEmptyField = inputFieldText == nil || (inputFieldText?.isEmpty ?? true)
                if isEmptyField {
                    context += """

                    [EMPTY_FIELD]
                    The input field is empty. This may be a form field (name, email, address, zip code, etc.).
                    Do NOT add a trailing period unless the user's speech clearly forms a complete sentence.
                    For short entries (names, single words, codes, numbers), output them WITHOUT trailing punctuation.
                    """
                }

                let correctionHint = CorrectionManager.shared.correctionContext(for: "")
                if !correctionHint.isEmpty {
                    context += correctionHint
                }

                // Format raw streaming transcript through LLM with token streaming.
                // Token callback is a no-op visually — UI stays in .processing until done.
                NSLog("[VoiceFlow] Streaming: formatting raw transcript (%d chars) via LLM (streaming)", rawTranscript.count)
                if let formattedText = await voiceFlow.formatTextStreaming(
                    rawTranscript, context: context,
                    onToken: { _ in }
                ) {
                    // Apply snippet expansion
                    var expandedText = SnippetManager.shared.expandSnippets(in: formattedText)

                    // Strip trailing period from short outputs in empty fields
                    if isEmptyField {
                        let trimmed = expandedText.trimmingCharacters(in: .whitespaces)
                        let wordCount = trimmed.split(separator: " ").count
                        let hasInternalPeriod = trimmed.dropLast().contains(".")
                        if wordCount <= 8 && !hasInternalPeriod && trimmed.hasSuffix(".") {
                            expandedText = String(trimmed.dropLast())
                        }
                    }

                    // Enforce mid-sentence continuation casing
                    if isMidSentence && !expandedText.isEmpty {
                        let first = expandedText.first!
                        if first.isUppercase {
                            let isStandaloneI = first == "I"
                                && (expandedText.count == 1 || !expandedText.dropFirst().first!.isLetter)
                            let isAcronym = expandedText.count >= 2
                                && expandedText.dropFirst().first!.isUppercase
                            if !isStandaloneI && !isAcronym {
                                expandedText = expandedText.prefix(1).lowercased() + expandedText.dropFirst()
                            }
                        }
                    }

                    // Apply spacing mode using pre-captured cursor context.
                    // CursorContext.getCharacterBeforeCursor() would return .unavailable
                    // here because the overlay has focus, not the original text field.
                    // Use inputFieldText (captured before the Task) to derive spacing.
                    var spacedText: String
                    if currentSpacingMode == .contextAware {
                        if let inputText = inputFieldText, !inputText.isEmpty {
                            let lastChar = inputText.last!
                            let isDash = lastChar == "\u{2014}" || lastChar == "\u{2013}" || lastChar == "-"
                            if !lastChar.isWhitespace && !lastChar.isNewline && !isDash,
                               let first = expandedText.first, first.isLetter || first.isNumber {
                                spacedText = " " + expandedText
                            } else {
                                spacedText = expandedText
                            }
                        } else {
                            spacedText = expandedText
                        }
                    } else {
                        spacedText = currentSpacingMode.apply(to: expandedText)
                    }
                    if !spacedText.isEmpty && !spacedText.hasSuffix(" ") && !spacedText.hasSuffix("\n") {
                        switch currentSpacingMode {
                        case .contextAware:
                            if inputFieldText == nil {
                                spacedText += " "
                            }
                        case .smart, .always, .trailing:
                            spacedText += " "
                        }
                    }

                    // Log transcription
                    let logEntry = TranscriptionEntry(
                        rawTranscript: rawTranscript,
                        formattedText: spacedText.trimmingCharacters(in: .whitespaces),
                        modelId: "moonshine-streaming",
                        sttEngine: "moonshine-streaming",
                        transcriptionMs: 0,  // streaming — no separate STT phase
                        llmMs: 0,
                        totalMs: 0,
                        targetApp: frontApp
                    )
                    TranscriptionLog.shared.append(logEntry)

                    if wasDirectTyping && insertedLength > 0 {
                        // Retroactive correction: select+replace the directly-typed text
                        NSLog("[VoiceFlow] Retroactive correction: replacing %d chars at position %@",
                              insertedLength, startPosition.map { String($0) } ?? "nil")
                        await MainActor.run {
                            hideOverlay()
                            performRetroactiveCorrection(
                                insertedLength: insertedLength,
                                startPosition: startPosition,
                                replacement: spacedText
                            )
                        }
                    } else {
                        // Standard overlay+paste path
                        let pasteboard = NSPasteboard.general
                        pasteboard.clearContents()
                        pasteboard.setString(spacedText, forType: .string)

                        await MainActor.run { hideOverlay() }
                        try? await Task.sleep(nanoseconds: 100_000_000)
                        await MainActor.run {
                            simulatePaste()
                        }
                    }

                    showNotification(
                        title: wasDirectTyping ? "Corrected (streaming)" : "Pasted (streaming)",
                        body: String(spacedText.prefix(100)) + (spacedText.count > 100 ? "..." : "")
                    )
                    self.lastPastedText = spacedText
                } else {
                    await MainActor.run { hideOverlay() }
                    showNotification(title: "Formatting Failed", body: voiceFlow.lastError ?? "Unknown error")
                }
            }
            return
        }

        // Parakeet-MLX path: batch STT via Python daemon, then LLM-format.
        // Activates when VOICEFLOW_USE_PARAKEET=1; replaces the in-process STT.
        if useParakeet {
            guard !audio.isEmpty else {
                hideOverlay()
                showNotification(title: "No Audio", body: "No audio was captured")
                return
            }

            showOverlay(state: .processing)

            let frontmostApp = NSWorkspace.shared.frontmostApplication
            let _ = frontmostApp.flatMap { AppProfileManager.shared.ensureProfile(for: $0) }
            let appPrompt = AppProfileManager.shared.promptForApp(frontmostApp)
            let frontApp = frontmostApp?.localizedName ?? "Unknown"
            let inputFieldText = CursorContext.getTextBeforeCursor(maxLength: 500)
            let baseContext = formattingLevel.systemPrompt + appPrompt
            let currentSpacingMode = spacingMode

            let isMidSentence: Bool
            if let inputText = inputFieldText, !inputText.isEmpty {
                let lastChar = inputText.trimmingCharacters(in: .whitespacesAndNewlines).last
                isMidSentence = lastChar != nil && !".?!".contains(String(lastChar!))
            } else {
                isMidSentence = false
            }

            Task {
                NSLog("[VoiceFlow] Parakeet: transcribing %d samples", audio.count)
                guard let rawTranscript = await parakeetEngine.transcribe(audio: audio),
                      !rawTranscript.isEmpty else {
                    await MainActor.run { hideOverlay() }
                    showNotification(title: "No Speech", body: "No speech was detected")
                    return
                }

                var context = baseContext
                if let inputText = inputFieldText, !inputText.isEmpty {
                    context += """

                    [INPUT CONTEXT]
                    Text already in the input field (before the cursor):
                    \(inputText)
                    You are CONTINUING from this text. Rules:
                    - Do NOT repeat or rewrite any of the existing text.
                    - If the existing text ends mid-sentence (no final . ? ! or newline), your output must seamlessly continue the sentence. Do NOT capitalize your first word (unless it is "I" or a proper noun). Do NOT add a period at the end unless the thought is truly complete.
                    - If the existing text ends with sentence-ending punctuation (. ? !) or a newline, start a new sentence with a capital letter.
                    - Your output should read as a natural continuation — as if the existing text and your output were written together.
                    """
                }
                if isMidSentence {
                    context += "\n[MID_SENTENCE_CONTINUATION]"
                }
                let isEmptyField = inputFieldText == nil || (inputFieldText?.isEmpty ?? true)
                if isEmptyField {
                    context += """

                    [EMPTY_FIELD]
                    The input field is empty. This may be a form field (name, email, address, zip code, etc.).
                    Do NOT add a trailing period unless the user's speech clearly forms a complete sentence.
                    For short entries (names, single words, codes, numbers), output them WITHOUT trailing punctuation.
                    """
                }
                let correctionHint = CorrectionManager.shared.correctionContext(for: "")
                if !correctionHint.isEmpty {
                    context += correctionHint
                }

                NSLog("[VoiceFlow] Parakeet → LLM: formatting %d-char transcript", rawTranscript.count)
                guard let formattedText = await voiceFlow.formatTextStreaming(
                    rawTranscript, context: context,
                    onToken: { _ in }
                ) else {
                    await MainActor.run { hideOverlay() }
                    showNotification(
                        title: "Formatting Failed",
                        body: voiceFlow.lastError ?? "Unknown error"
                    )
                    return
                }

                var expandedText = SnippetManager.shared.expandSnippets(in: formattedText)
                if isEmptyField {
                    let trimmed = expandedText.trimmingCharacters(in: .whitespaces)
                    let wordCount = trimmed.split(separator: " ").count
                    let hasInternalPeriod = trimmed.dropLast().contains(".")
                    if wordCount <= 8 && !hasInternalPeriod && trimmed.hasSuffix(".") {
                        expandedText = String(trimmed.dropLast())
                    }
                }
                if isMidSentence && !expandedText.isEmpty {
                    let first = expandedText.first!
                    if first.isUppercase {
                        let isStandaloneI = first == "I"
                            && (expandedText.count == 1 || !expandedText.dropFirst().first!.isLetter)
                        let isAcronym = expandedText.count >= 2
                            && expandedText.dropFirst().first!.isUppercase
                        if !isStandaloneI && !isAcronym {
                            expandedText = expandedText.prefix(1).lowercased() + expandedText.dropFirst()
                        }
                    }
                }
                var spacedText: String
                if currentSpacingMode == .contextAware {
                    if let inputText = inputFieldText, !inputText.isEmpty {
                        let lastChar = inputText.last!
                        let isDash = lastChar == "\u{2014}" || lastChar == "\u{2013}" || lastChar == "-"
                        if !lastChar.isWhitespace && !lastChar.isNewline && !isDash,
                           let first = expandedText.first, first.isLetter || first.isNumber {
                            spacedText = " " + expandedText
                        } else {
                            spacedText = expandedText
                        }
                    } else {
                        spacedText = expandedText
                    }
                } else {
                    spacedText = currentSpacingMode.apply(to: expandedText)
                }
                if !spacedText.isEmpty && !spacedText.hasSuffix(" ") && !spacedText.hasSuffix("\n") {
                    switch currentSpacingMode {
                    case .contextAware:
                        if inputFieldText == nil { spacedText += " " }
                    case .smart, .always, .trailing:
                        spacedText += " "
                    }
                }

                let logEntry = TranscriptionEntry(
                    rawTranscript: rawTranscript,
                    formattedText: spacedText.trimmingCharacters(in: .whitespaces),
                    modelId: parakeetEngine.modelId,
                    sttEngine: "parakeet-mlx",
                    transcriptionMs: 0,
                    llmMs: 0,
                    totalMs: 0,
                    targetApp: frontApp
                )
                TranscriptionLog.shared.append(logEntry)

                let pasteboard = NSPasteboard.general
                pasteboard.clearContents()
                pasteboard.setString(spacedText, forType: .string)

                await MainActor.run { hideOverlay() }
                try? await Task.sleep(nanoseconds: 100_000_000)
                await MainActor.run { simulatePaste() }

                showNotification(
                    title: "Pasted (parakeet)",
                    body: String(spacedText.prefix(100)) + (spacedText.count > 100 ? "..." : "")
                )
                self.lastPastedText = spacedText
            }
            return
        }

        // Audio-direct path: send raw audio to Gemma 4 (no separate STT)
        if voiceFlow.isAudioDirect {
            guard !audio.isEmpty else {
                hideOverlay()
                showNotification(title: "No Audio", body: "No audio was captured")
                return
            }

            showOverlay(state: .processing)

            let frontmostApp = NSWorkspace.shared.frontmostApplication
            let frontApp = frontmostApp?.localizedName ?? "Unknown"
            let inputFieldText = CursorContext.getTextBeforeCursor(maxLength: 500)
            let currentSpacingMode = spacingMode

            Task {
                NSLog("[VoiceFlow] Audio-direct: sending %d samples to Gemma 4", audio.count)

                guard let result = await voiceFlow.processAudioDirect(audio: audio) else {
                    await MainActor.run { hideOverlay() }
                    showNotification(title: "Audio Direct Failed", body: voiceFlow.lastError ?? "Unknown error")
                    return
                }

                var spacedText = result.formattedText

                // Apply spacing mode
                if currentSpacingMode == .contextAware {
                    if let inputText = inputFieldText, !inputText.isEmpty {
                        let lastChar = inputText.last!
                        let isDash = lastChar == "\u{2014}" || lastChar == "\u{2013}" || lastChar == "-"
                        if !lastChar.isWhitespace && !lastChar.isNewline && !isDash,
                           let first = spacedText.first, first.isLetter || first.isNumber {
                            spacedText = " " + spacedText
                        }
                    }
                } else {
                    spacedText = currentSpacingMode.apply(to: spacedText)
                }

                // Log transcription
                let logEntry = TranscriptionEntry(
                    rawTranscript: result.rawTranscript,
                    formattedText: spacedText.trimmingCharacters(in: .whitespaces),
                    modelId: "gemma4-e2b",
                    sttEngine: "audio-direct",
                    transcriptionMs: 0,
                    llmMs: result.llmMs,
                    totalMs: result.totalMs,
                    targetApp: frontApp
                )
                TranscriptionLog.shared.append(logEntry)

                // Copy to clipboard and paste
                let pasteboard = NSPasteboard.general
                pasteboard.clearContents()
                pasteboard.setString(spacedText, forType: .string)

                await MainActor.run { hideOverlay() }
                try? await Task.sleep(nanoseconds: 100_000_000)
                await MainActor.run {
                    simulatePaste()
                }

                showNotification(
                    title: "Pasted (audio-direct)",
                    body: String(spacedText.prefix(100)) + (spacedText.count > 100 ? "..." : "")
                )
                self.lastPastedText = spacedText
            }
            return
        }

        // Non-streaming path (existing flow)
        guard !audio.isEmpty else {
            hideOverlay()
            showNotification(title: "No Audio", body: "No audio was captured")
            return
        }

        // Show processing state in overlay
        showOverlay(state: .processing)

        // Detect the application context via AppProfileManager (auto-creates on first encounter)
        let frontmostApp = NSWorkspace.shared.frontmostApplication
        let _ = frontmostApp.flatMap { AppProfileManager.shared.ensureProfile(for: $0) }
        let appPrompt = AppProfileManager.shared.promptForApp(frontmostApp)
        let frontApp = frontmostApp?.localizedName ?? "Unknown"

        // Read existing text from the focused input field (must be on main thread before async)
        let inputFieldText = CursorContext.getTextBeforeCursor(maxLength: 500)

        let baseContext = formattingLevel.systemPrompt + appPrompt
        let currentSpacingMode = spacingMode
        let wantsVisualContext = isVisualContextRecording
        let capture = screenshotCapture

        Task {
            // Capture screenshot if visual context recording (Control+Option+Space)
            var screenshotData: Data? = nil
            if wantsVisualContext && ScreenshotCapture.hasPermission() {
                do {
                    screenshotData = try await capture.captureActiveWindow()
                    NSLog("[VoiceFlow] Visual context: captured screenshot (%d bytes)", screenshotData?.count ?? 0)
                } catch {
                    NSLog("[VoiceFlow] Visual context capture failed: %@", error.localizedDescription)
                }
            }

            // Combine formatting level prompt + app context + input field context
            var context = baseContext
            // Terminal apps have unreliable AX text buffers (includes shell output,
            // prompts, status bars). Only trust mid-sentence detection for standard text fields.
            let terminalBundleIds: Set<String> = [
                "com.apple.Terminal",
                "com.googlecode.iterm2",
                "com.mitchellh.ghostty",
                "dev.warp.Warp-Stable",
                "io.alacritty",
                "com.github.wez.wezterm",
                "net.kovidgoyal.kitty",
            ]
            let isTerminal = terminalBundleIds.contains(frontmostApp?.bundleIdentifier ?? "")

            // Determine mid-sentence state for continuation casing fix.
            // For standard text apps: use AX buffer (reliable).
            // For terminal apps: use last VoiceFlow output (AX buffer includes shell chrome).
            let isMidSentence: Bool
            if isTerminal {
                let lastOutput = self.lastPastedText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let lastOutputChar = lastOutput.last
                isMidSentence = lastOutputChar != nil && !".?!".contains(String(lastOutputChar!))
                NSLog("[VoiceFlow] INPUT CONTEXT: terminal mode, lastPastedText='%@', isMidSentence=%d",
                      String(lastOutput.suffix(40)),
                      isMidSentence ? 1 : 0)
            } else if let inputText = inputFieldText, !inputText.isEmpty {
                let trimmedInput = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
                let lastChar = trimmedInput.last
                isMidSentence = lastChar != nil && !".?!".contains(String(lastChar!))
                NSLog("[VoiceFlow] INPUT CONTEXT: text field, got %d chars, lastChar='%@', isMidSentence=%d",
                      inputText.count,
                      lastChar.map { String($0) } ?? "nil",
                      isMidSentence ? 1 : 0)
            } else {
                isMidSentence = false
                NSLog("[VoiceFlow] INPUT CONTEXT: no text before cursor (inputFieldText=%@)",
                      inputFieldText == nil ? "nil" : "empty")
            }

            if let inputText = inputFieldText, !inputText.isEmpty {
                context += """

                [INPUT CONTEXT]
                Text already in the input field (before the cursor):
                \(inputText)
                You are CONTINUING from this text. Rules:
                - Do NOT repeat or rewrite any of the existing text.
                - If the existing text ends mid-sentence (no final . ? ! or newline), your output must seamlessly continue the sentence. Do NOT capitalize your first word (unless it is "I" or a proper noun). Do NOT add a period at the end unless the thought is truly complete.
                - If the existing text ends with sentence-ending punctuation (. ? !) or a newline, start a new sentence with a capital letter.
                - Your output should read as a natural continuation — as if the existing text and your output were written together.
                """
            }

            if isMidSentence {
                context += "\n[MID_SENTENCE_CONTINUATION]"
            }

            // Detect likely form field entry: empty field + short dictation
            let isEmptyField = inputFieldText == nil || (inputFieldText?.isEmpty ?? true)
            if isEmptyField {
                context += """

                [EMPTY_FIELD]
                The input field is empty. This may be a form field (name, email, address, zip code, etc.).
                Do NOT add a trailing period unless the user's speech clearly forms a complete sentence.
                For short entries (names, single words, codes, numbers), output them WITHOUT trailing punctuation.
                """
            }

            // Inject correction history from previous user edits
            let correctionHint = CorrectionManager.shared.correctionContext(for: "")
            if !correctionHint.isEmpty {
                context += correctionHint
            }

            // Process audio: STT + LLM formatting (and re-format with image if visual context)
            var processResult = await voiceFlow.process(audio: audio, context: context)

            // If visual context recording with screenshot, re-format through multimodal LLM
            if let screenshot = screenshotData, let baseResult = processResult {
                NSLog("[VoiceFlow] Visual context: re-formatting with screenshot (%d bytes)", screenshot.count)
                if let imageResult = await voiceFlow.processTextWithImage(
                    baseResult.rawTranscript, context: context, imageData: screenshot
                ) {
                    processResult = imageResult
                } else {
                    NSLog("[VoiceFlow] Multimodal formatting failed, using text-only result")
                }
            }

            if let result = processResult {
                // Meta-command: "summarize this"
                let summarizeEnabled = UserDefaults.standard.bool(forKey: "aiFeatureSummarizeEnabled")
                let normalizedTranscript = result.rawTranscript.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
                if summarizeEnabled && (normalizedTranscript == "summarize this"
                    || normalizedTranscript == "summarise this"
                    || normalizedTranscript == "summarize this."
                    || normalizedTranscript == "summarise this.") {
                    await MainActor.run {
                        showOverlay(state: .aiProcessing("Reading text..."))
                        summarizeCurrentField(targetApp: frontmostApp)
                    }
                    return
                }

                // AI voice commands: reply, rewrite, proofread, continue
                if summarizeEnabled, let aiCommand = AICommand.parse(normalizedTranscript) {
                    await MainActor.run {
                        handleAICommand(aiCommand, targetApp: frontmostApp, visualDescription: nil)
                    }
                    return
                }

                // Apply snippet expansion (e.g., "my signature" → "Best regards,\nJohn")
                var expandedText = SnippetManager.shared.expandSnippets(in: result.formattedText)

                // Strip trailing period from short outputs in empty fields (form field safety net).
                // A short phrase like "John Smith." or "94117." shouldn't end with a period.
                if isEmptyField {
                    let trimmed = expandedText.trimmingCharacters(in: .whitespaces)
                    let wordCount = trimmed.split(separator: " ").count
                    let hasInternalPeriod = trimmed.dropLast().contains(".")
                    if wordCount <= 8 && !hasInternalPeriod && trimmed.hasSuffix(".") {
                        expandedText = String(trimmed.dropLast())
                        NSLog("[VoiceFlow] Stripped trailing period from short empty-field entry: \"%@\"", expandedText)
                    }
                }

                // Enforce mid-sentence continuation casing — safety net for LLM
                if isMidSentence && !expandedText.isEmpty {
                    let first = expandedText.first!
                    if first.isUppercase {
                        // Preserve standalone "I" (I, I'm, I'll, I'd, I've)
                        let isStandaloneI = first == "I"
                            && (expandedText.count == 1 || !expandedText.dropFirst().first!.isLetter)
                        // Preserve acronyms (NASA, API, etc.) — 2+ leading uppercase chars
                        let isAcronym = expandedText.count >= 2
                            && expandedText.dropFirst().first!.isUppercase
                        if !isStandaloneI && !isAcronym {
                            expandedText = expandedText.prefix(1).lowercased() + expandedText.dropFirst()
                            NSLog("[VoiceFlow] Mid-sentence: lowercased first char → \"%@\"",
                                  String(expandedText.prefix(30)))
                        }
                    }
                }

                // Apply spacing mode using pre-captured cursor context.
                // CursorContext.getCharacterBeforeCursor() would return .unavailable
                // here because the overlay has focus, not the original text field.
                // Use inputFieldText (captured before the Task) to derive spacing.
                var spacedText: String
                if currentSpacingMode == .contextAware {
                    if let inputText = inputFieldText, !inputText.isEmpty {
                        let lastChar = inputText.last!
                        let isDash = lastChar == "\u{2014}" || lastChar == "\u{2013}" || lastChar == "-"
                        if !lastChar.isWhitespace && !lastChar.isNewline && !isDash,
                           let first = expandedText.first, first.isLetter || first.isNumber {
                            spacedText = " " + expandedText
                        } else {
                            spacedText = expandedText
                        }
                    } else {
                        spacedText = expandedText
                    }
                } else {
                    spacedText = currentSpacingMode.apply(to: expandedText)
                }
                if !spacedText.isEmpty && !spacedText.hasSuffix(" ") && !spacedText.hasSuffix("\n") {
                    switch currentSpacingMode {
                    case .contextAware:
                        if inputFieldText == nil {
                            spacedText += " "
                        }
                    case .smart, .always, .trailing:
                        spacedText += " "
                    }
                }

                // Log the transcription
                let currentModel: String = {
                    if let llmPtr = voiceflow_current_model() {
                        let s = String(cString: llmPtr)
                        voiceflow_free_string(llmPtr)
                        return s
                    }
                    return "unknown"
                }()
                let currentEngine: String = {
                    if let ptr = voiceflow_current_stt_engine() {
                        let s = String(cString: ptr)
                        voiceflow_free_string(ptr)
                        return s
                    }
                    return "unknown"
                }()
                let logEntry = TranscriptionEntry(
                    rawTranscript: result.rawTranscript,
                    formattedText: spacedText.trimmingCharacters(in: .whitespaces),
                    modelId: currentModel,
                    sttEngine: currentEngine,
                    transcriptionMs: result.transcriptionMs,
                    llmMs: result.llmMs,
                    totalMs: result.totalMs,
                    targetApp: frontApp
                )
                TranscriptionLog.shared.append(logEntry)

                // Copy to clipboard
                let pasteboard = NSPasteboard.general
                pasteboard.clearContents()
                pasteboard.setString(spacedText, forType: .string)

                // Hide overlay first
                await MainActor.run {
                    hideOverlay()
                }

                // Wait for clipboard and UI to settle
                try? await Task.sleep(nanoseconds: 100_000_000)  // 100ms

                // Auto-paste with Cmd+V
                await MainActor.run {
                    simulatePaste()
                }

                showNotification(
                    title: "Pasted (\(result.totalMs)ms)",
                    body: String(spacedText.prefix(100)) + (spacedText.count > 100 ? "..." : "")
                )

                // Track last pasted text for terminal continuation detection
                self.lastPastedText = spacedText
            } else {
                await MainActor.run {
                    hideOverlay()
                }
                showNotification(title: "Processing Failed", body: voiceFlow.lastError ?? "Unknown error")
            }
        }
    }

    // MARK: - Summarize This

    /// Summarize the contents of the current text field and append the summary.
    /// `targetApp` must be the app the user was dictating into, captured BEFORE the overlay was hidden.
    private func summarizeCurrentField(targetApp: NSRunningApplication?) {
        Task { @MainActor in
            // Save current clipboard contents and mark the change count
            let savedClipboard = NSPasteboard.general.string(forType: .string)

            // Step 1: Activate the target app and wait for it to gain focus
            let appName = targetApp?.localizedName ?? "unknown"
            NSLog("[VoiceFlow Summarize] Target app: %@ (pid %d)", appName, targetApp?.processIdentifier ?? 0)

            if let app = targetApp {
                app.activate()
            }
            try? await Task.sleep(nanoseconds: 300_000_000) // 300ms for app activation

            // Step 2: Clear clipboard so we can detect when copy lands
            let pasteboard = NSPasteboard.general
            pasteboard.clearContents()
            pasteboard.setString("", forType: .string)
            let emptyChangeCount = pasteboard.changeCount
            NSLog("[VoiceFlow Summarize] Cleared clipboard, changeCount=%d", emptyChangeCount)

            // Step 3: Select All (Cmd+A)
            NSLog("[VoiceFlow Summarize] Sending Cmd+A")
            simulateSelectAll()
            try? await Task.sleep(nanoseconds: 300_000_000) // 300ms for selection

            // Step 4: Copy (Cmd+C)
            NSLog("[VoiceFlow Summarize] Sending Cmd+C")
            simulateCopy()

            // Step 5: Wait for clipboard to change from empty (poll up to 2s)
            var fieldText = ""
            var copySucceeded = false
            for i in 0..<40 {
                try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
                if pasteboard.changeCount != emptyChangeCount {
                    fieldText = pasteboard.string(forType: .string) ?? ""
                    NSLog("[VoiceFlow Summarize] Copy landed after %dms, got %d chars", (i + 1) * 50, fieldText.count)
                    copySucceeded = true
                    break
                }
            }

            if !copySucceeded {
                NSLog("[VoiceFlow Summarize] FAILED: clipboard never changed — Cmd+A/Cmd+C didn't reach %@", appName)
                hideOverlay()
                // Restore clipboard
                pasteboard.clearContents()
                if let saved = savedClipboard {
                    pasteboard.setString(saved, forType: .string)
                }
                showNotification(title: "Summarize Failed", body: "Could not read text from \(appName). Make sure the text field is focused.")
                return
            }

            // Restore original clipboard now that we have the text
            pasteboard.clearContents()
            if let saved = savedClipboard {
                pasteboard.setString(saved, forType: .string)
            }

            // Deselect: move to end of document
            simulateEndOfDocument()
            try? await Task.sleep(nanoseconds: 100_000_000) // 100ms

            guard fieldText.count >= 20 else {
                NSLog("[VoiceFlow Summarize] Not enough text: %d chars", fieldText.count)
                hideOverlay()
                showNotification(title: "Summarize", body: "Not enough text to summarize (need at least 20 characters).")
                return
            }

            showOverlay(state: .aiProcessing("Summarizing..."))
            showNotification(title: "Summarizing...", body: "Processing \(fieldText.count) characters")

            let summarizePrompt = """
            OVERRIDE: Ignore all formatting instructions above. Your task is to summarize the text below as a concise bullet-point list (3-7 bullets). Each bullet starts with "• ". Output ONLY the bullet points, nothing else.
            """

            guard let summary = await voiceFlow.formatText(fieldText, context: summarizePrompt) else {
                hideOverlay()
                showNotification(title: "Summarize Failed", body: "Could not generate summary")
                return
            }

            // Re-activate target app for pasting
            if let app = targetApp {
                app.activate()
            }
            try? await Task.sleep(nanoseconds: 200_000_000) // 200ms

            // Move to end of document
            simulateEndOfDocument()
            try? await Task.sleep(nanoseconds: 100_000_000) // 100ms

            // Put summary on clipboard and paste
            let summaryText = "\n\n---\nSummary:\n" + summary.trimmingCharacters(in: .whitespacesAndNewlines)
            let pb = NSPasteboard.general
            pb.clearContents()
            pb.setString(summaryText, forType: .string)
            try? await Task.sleep(nanoseconds: 100_000_000) // 100ms

            simulatePaste()
            hideOverlay()

            // Restore original clipboard after paste settles
            try? await Task.sleep(nanoseconds: 500_000_000) // 500ms
            let restorePb = NSPasteboard.general
            restorePb.clearContents()
            if let saved = savedClipboard {
                restorePb.setString(saved, forType: .string)
            }

            showNotification(title: "Summary Added", body: String(summary.prefix(100)))
        }
    }

    // MARK: - AI Command Helpers

    /// Capture text from the current field by selecting all and copying.
    /// Returns the captured text and the saved clipboard contents, or nil on failure.
    private func captureFieldText(targetApp: NSRunningApplication?, minLength: Int = 0) async -> (text: String, savedClipboard: String?)? {
        let savedClipboard = NSPasteboard.general.string(forType: .string)
        let appName = targetApp?.localizedName ?? "unknown"

        // Activate target app
        if let app = targetApp {
            app.activate()
        }
        try? await Task.sleep(nanoseconds: 300_000_000) // 300ms

        // Clear clipboard
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString("", forType: .string)
        let emptyChangeCount = pasteboard.changeCount

        // Select All + Copy
        simulateSelectAll()
        try? await Task.sleep(nanoseconds: 300_000_000)
        simulateCopy()

        // Poll for clipboard change (up to 2s)
        var fieldText = ""
        var copySucceeded = false
        for i in 0..<40 {
            try? await Task.sleep(nanoseconds: 50_000_000)
            if pasteboard.changeCount != emptyChangeCount {
                fieldText = pasteboard.string(forType: .string) ?? ""
                NSLog("[VoiceFlow AI] Copy landed after %dms, got %d chars", (i + 1) * 50, fieldText.count)
                copySucceeded = true
                break
            }
        }

        // Restore clipboard
        pasteboard.clearContents()
        if let saved = savedClipboard {
            pasteboard.setString(saved, forType: .string)
        }

        // Deselect: move to end of document
        simulateEndOfDocument()
        try? await Task.sleep(nanoseconds: 100_000_000)

        if !copySucceeded {
            NSLog("[VoiceFlow AI] FAILED: clipboard never changed for %@", appName)
            return nil
        }

        if fieldText.count < minLength {
            NSLog("[VoiceFlow AI] Not enough text: %d chars (min %d)", fieldText.count, minLength)
            return nil
        }

        return (text: fieldText, savedClipboard: savedClipboard)
    }

    // MARK: - AI Command Handlers

    private func handleAICommand(_ command: AICommand, targetApp: NSRunningApplication?, visualDescription: String?) {
        showOverlay(state: .aiProcessing(command.processingLabel))

        switch command {
        case .reply(let intent):
            handleReply(intent: intent, targetApp: targetApp, visualDescription: visualDescription)
        case .rewrite(let style):
            handleRewrite(style: style, targetApp: targetApp)
        case .proofread:
            handleProofread(targetApp: targetApp)
        case .continueWriting:
            handleContinue(targetApp: targetApp)
        }
    }

    private func handleReply(intent: String, targetApp: NSRunningApplication?, visualDescription: String?) {
        Task { @MainActor in
            guard let visualCtx = visualDescription, !visualCtx.isEmpty else {
                hideOverlay()
                showNotification(title: "Reply Failed", body: "Visual context required — enable VLM in settings and grant Screen Recording permission.")
                return
            }

            let prompt = """
            OVERRIDE: Ignore all formatting instructions above. You are composing a reply.

            [SCREEN CONTEXT]
            \(visualCtx)

            [USER INTENT]
            The user wants to reply saying: \(intent)

            Write a natural, contextually appropriate reply based on what's visible on screen and the user's intent. Output ONLY the reply text, nothing else. Match the tone of the conversation (casual if casual, formal if formal).
            """

            guard let reply = await voiceFlow.formatText("Reply request", context: prompt) else {
                hideOverlay()
                showNotification(title: "Reply Failed", body: "Could not generate reply")
                return
            }

            hideOverlay()
            showAIResult(command: .reply(intent: intent), text: reply.trimmingCharacters(in: .whitespacesAndNewlines), targetApp: targetApp)
        }
    }

    private func handleRewrite(style: String, targetApp: NSRunningApplication?) {
        Task { @MainActor in
            showOverlay(state: .aiProcessing("Reading text..."))

            guard let captured = await captureFieldText(targetApp: targetApp, minLength: 5) else {
                hideOverlay()
                showNotification(title: "Rewrite Failed", body: "Could not read text from the current field.")
                return
            }

            showOverlay(state: .aiProcessing("Rewriting..."))

            let prompt = """
            OVERRIDE: Ignore all formatting instructions above. Rewrite the following text to be more \(style). Preserve the core meaning. Output ONLY the rewritten text, nothing else.
            """

            guard let rewritten = await voiceFlow.formatText(captured.text, context: prompt) else {
                hideOverlay()
                showNotification(title: "Rewrite Failed", body: "Could not rewrite text")
                return
            }

            hideOverlay()
            showAIResult(command: .rewrite(style: style), text: rewritten.trimmingCharacters(in: .whitespacesAndNewlines), targetApp: targetApp)
        }
    }

    private func handleProofread(targetApp: NSRunningApplication?) {
        Task { @MainActor in
            showOverlay(state: .aiProcessing("Reading text..."))

            guard let captured = await captureFieldText(targetApp: targetApp, minLength: 5) else {
                hideOverlay()
                showNotification(title: "Proofread Failed", body: "Could not read text from the current field.")
                return
            }

            showOverlay(state: .aiProcessing("Proofreading..."))

            let prompt = """
            OVERRIDE: Ignore all formatting instructions above. Proofread the following text. Fix spelling, grammar, and punctuation errors. Make minimal changes — only correct mistakes, do not rephrase or restructure. Output ONLY the corrected text, nothing else.
            """

            guard let proofread = await voiceFlow.formatText(captured.text, context: prompt) else {
                hideOverlay()
                showNotification(title: "Proofread Failed", body: "Could not proofread text")
                return
            }

            hideOverlay()
            showAIResult(command: .proofread, text: proofread.trimmingCharacters(in: .whitespacesAndNewlines), targetApp: targetApp)
        }
    }

    private func handleContinue(targetApp: NSRunningApplication?) {
        Task { @MainActor in
            showOverlay(state: .aiProcessing("Reading text..."))

            guard let captured = await captureFieldText(targetApp: targetApp, minLength: 5) else {
                hideOverlay()
                showNotification(title: "Continue Failed", body: "Could not read text from the current field.")
                return
            }

            showOverlay(state: .aiProcessing("Continuing..."))

            let prompt = """
            OVERRIDE: Ignore all formatting instructions above. Continue writing from where the text below leaves off. Match the existing style, tone, and voice. Do NOT repeat or rewrite the original text — only output the NEW continuation. Output ONLY the continuation text, nothing else.
            """

            guard let continuation = await voiceFlow.formatText(captured.text, context: prompt) else {
                hideOverlay()
                showNotification(title: "Continue Failed", body: "Could not generate continuation")
                return
            }

            hideOverlay()
            showAIResult(command: .continueWriting, text: continuation.trimmingCharacters(in: .whitespacesAndNewlines), targetApp: targetApp)
        }
    }

    /// Simulate Cmd+V paste keystroke using AppleScript (more reliable)
    private func simulatePaste() {
        // Check accessibility first
        guard AXIsProcessTrusted() else {
            print("Accessibility not granted - cannot auto-paste")
            return
        }

        // Use AppleScript for more reliable paste
        let script = NSAppleScript(source: """
            tell application "System Events"
                keystroke "v" using command down
            end tell
        """)

        var error: NSDictionary?
        script?.executeAndReturnError(&error)

        if let error = error {
            print("AppleScript paste error: \(error)")
            // Fallback to CGEvent
            fallbackPaste()
        }
    }

    private func fallbackPaste() {
        let vKeyCode: CGKeyCode = 9

        // Create event source
        guard let source = CGEventSource(stateID: .hidSystemState) else { return }

        // Create key down event with Cmd modifier
        guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: true) else { return }
        keyDown.flags = .maskCommand

        // Create key up event with Cmd modifier
        guard let keyUp = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: false) else { return }
        keyUp.flags = .maskCommand

        // Post the events with small delay between
        keyDown.post(tap: .cgSessionEventTap)
        usleep(10000)  // 10ms delay
        keyUp.post(tap: .cgSessionEventTap)
    }

    /// Simulate Cmd+A (Select All) via CGEvent (works from background apps)
    private func simulateSelectAll() {
        simulateKeystroke(keyCode: 0, flags: .maskCommand) // 0 = 'a'
    }

    /// Simulate Cmd+C (Copy) via CGEvent (works from background apps)
    private func simulateCopy() {
        simulateKeystroke(keyCode: 8, flags: .maskCommand) // 8 = 'c'
    }

    /// Simulate Cmd+Down (End of Document) via CGEvent (works from background apps)
    private func simulateEndOfDocument() {
        simulateKeystroke(keyCode: 125, flags: .maskCommand) // 125 = down arrow
    }

    /// Post a single keystroke via CGEvent at the session level
    private func simulateKeystroke(keyCode: CGKeyCode, flags: CGEventFlags) {
        guard let source = CGEventSource(stateID: .hidSystemState) else {
            NSLog("[VoiceFlow] CGEventSource creation failed")
            return
        }
        guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true),
              let keyUp = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false) else {
            NSLog("[VoiceFlow] CGEvent creation failed for keyCode %d", keyCode)
            return
        }
        keyDown.flags = flags
        keyUp.flags = flags
        keyDown.post(tap: .cgSessionEventTap)
        usleep(10000) // 10ms between down/up
        keyUp.post(tap: .cgSessionEventTap)
    }

    // MARK: - Direct Typing Utilities

    /// Type arbitrary Unicode text via CGEvent keyboard events.
    /// Chunks into 20 UTF-16 units per event (API limit).
    private func typeUnicodeText(_ text: String) {
        guard let source = CGEventSource(stateID: .hidSystemState) else { return }
        let utf16 = Array(text.utf16)
        let chunkSize = 20

        for start in stride(from: 0, to: utf16.count, by: chunkSize) {
            let end = min(start + chunkSize, utf16.count)
            var chunk = Array(utf16[start..<end])

            guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: true),
                  let keyUp = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: false) else {
                continue
            }

            keyDown.keyboardSetUnicodeString(stringLength: chunk.count, unicodeString: &chunk)
            keyUp.keyboardSetUnicodeString(stringLength: 0, unicodeString: &chunk)

            keyDown.post(tap: .cgSessionEventTap)
            keyUp.post(tap: .cgSessionEventTap)
            usleep(2000) // 2ms between chunks
        }
    }

    /// Send `count` backspace keystrokes to delete characters before cursor.
    private func deleteBackward(count: Int) {
        guard count > 0 else { return }
        guard let source = CGEventSource(stateID: .hidSystemState) else { return }

        let backspaceKeyCode: CGKeyCode = 51
        for _ in 0..<count {
            guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: backspaceKeyCode, keyDown: true),
                  let keyUp = CGEvent(keyboardEventSource: source, virtualKey: backspaceKeyCode, keyDown: false) else {
                continue
            }
            keyDown.post(tap: .cgSessionEventTap)
            keyUp.post(tap: .cgSessionEventTap)
            usleep(1000) // 1ms between keystrokes
        }
    }

    /// Send Shift+Left arrow keystrokes to extend selection backward.
    /// Used as fallback when AX selectRange is unavailable.
    private func selectBackward(count: Int) {
        guard count > 0 else { return }
        guard let source = CGEventSource(stateID: .hidSystemState) else { return }

        let leftArrowKeyCode: CGKeyCode = 123
        for _ in 0..<count {
            guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: leftArrowKeyCode, keyDown: true),
                  let keyUp = CGEvent(keyboardEventSource: source, virtualKey: leftArrowKeyCode, keyDown: false) else {
                continue
            }
            keyDown.flags = .maskShift
            keyUp.flags = .maskShift
            keyDown.post(tap: .cgSessionEventTap)
            keyUp.post(tap: .cgSessionEventTap)
            usleep(1000) // 1ms between keystrokes
        }
    }

    /// Compute the minimal edit between previous and current streaming text.
    /// Returns (number of chars to delete from end of previous, new text to insert).
    static func computeStreamingDiff(previous: String, current: String) -> (deleteCount: Int, insertText: String) {
        // Find longest common prefix
        let prevChars = Array(previous)
        let curChars = Array(current)
        var commonLen = 0
        let minLen = min(prevChars.count, curChars.count)

        while commonLen < minLen && prevChars[commonLen] == curChars[commonLen] {
            commonLen += 1
        }

        let deleteCount = prevChars.count - commonLen
        let insertText = commonLen < curChars.count ? String(curChars[commonLen...]) : ""

        return (deleteCount, insertText)
    }

    /// Perform retroactive correction: select the inserted text and replace with normalized version.
    private func performRetroactiveCorrection(insertedLength: Int, startPosition: Int?, replacement: String) {
        // Save user's clipboard (first type+data pair from each item)
        let pasteboard = NSPasteboard.general
        let savedItems: [(NSPasteboard.PasteboardType, Data)] = pasteboard.pasteboardItems?.compactMap { item in
            for type in item.types {
                if let data = item.data(forType: type) {
                    return (type, data)
                }
            }
            return nil
        } ?? []

        // Put normalized text on clipboard
        pasteboard.clearContents()
        pasteboard.setString(replacement, forType: .string)

        // Try AX API selection first (instant)
        var selected = false
        if let start = startPosition {
            selected = CursorContext.selectRange(location: start, length: insertedLength)
        }

        // Fallback to Shift+Left arrow selection
        if !selected {
            selectBackward(count: insertedLength)
            usleep(50000) // 50ms for selection to settle
        }

        // Paste to replace selection
        usleep(50000) // 50ms before paste
        simulatePaste()

        // Restore clipboard after delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            pasteboard.clearContents()
            for (type, data) in savedItems {
                pasteboard.setData(data, forType: type)
            }
        }
    }

    // MARK: - Menu Actions

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }

    // MARK: - Notifications

    private func showNotification(title: String, body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )

        UNUserNotificationCenter.current().add(request)
    }

    private func showAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}

// MARK: - Global Hotkey Manager

final class GlobalHotkeyManager {
    private var hotKeyRefs: [UInt32: EventHotKeyRef] = [:]
    private static var eventHandlerInstalled = false

    private static var pressHandlers: [UInt32: () -> Void] = [:]
    private static var releaseHandlers: [UInt32: () -> Void] = [:]

    func register(id: UInt32, keyCode: UInt32, modifiers: UInt32, onPress: @escaping () -> Void, onRelease: @escaping () -> Void) {
        GlobalHotkeyManager.pressHandlers[id] = onPress
        GlobalHotkeyManager.releaseHandlers[id] = onRelease

        // Install the shared event handler once
        if !GlobalHotkeyManager.eventHandlerInstalled {
            var eventTypes = [
                EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
                EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))
            ]

            InstallEventHandler(
                GetApplicationEventTarget(),
                { (_, event, _) -> OSStatus in
                    // Extract the hotkey ID from the event
                    var hotkeyID = EventHotKeyID()
                    GetEventParameter(
                        event,
                        EventParamName(kEventParamDirectObject),
                        EventParamType(typeEventHotKeyID),
                        nil,
                        MemoryLayout<EventHotKeyID>.size,
                        nil,
                        &hotkeyID
                    )

                    let kind = GetEventKind(event)
                    if kind == UInt32(kEventHotKeyPressed) {
                        GlobalHotkeyManager.pressHandlers[hotkeyID.id]?()
                    } else if kind == UInt32(kEventHotKeyReleased) {
                        GlobalHotkeyManager.releaseHandlers[hotkeyID.id]?()
                    }
                    return noErr
                },
                2,
                &eventTypes,
                nil,
                nil
            )
            GlobalHotkeyManager.eventHandlerInstalled = true
        }

        var hotKeyID = EventHotKeyID()
        hotKeyID.signature = OSType(0x564643) // "VFC"
        hotKeyID.id = id

        var ref: EventHotKeyRef?
        RegisterEventHotKey(
            keyCode,
            modifiers,
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &ref
        )

        if let ref = ref {
            hotKeyRefs[id] = ref
        }
    }

    deinit {
        for (_, ref) in hotKeyRefs {
            UnregisterEventHotKey(ref)
        }
    }
}

// MARK: - Permission Buttons

struct AccessibilityPermissionButton: View {
    @State private var isGranted: Bool = AXIsProcessTrusted()

    var body: some View {
        Group {
            if isGranted {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            } else {
                Button("Grant") {
                    let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
                    AXIsProcessTrustedWithOptions(options)
                    // Check again after a short delay
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        isGranted = AXIsProcessTrusted()
                    }
                }
            }
        }
        .onAppear {
            isGranted = AXIsProcessTrusted()
        }
    }
}

struct MicrophonePermissionButton: View {
    @State private var authStatus: AVAuthorizationStatus = AVCaptureDevice.authorizationStatus(for: .audio)

    private func openMicrophonePrivacySettings() {
        // macOS 13+ (Ventura): new System Settings URL
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }

    private func requestMicAccess() {
        NSApp.activate(ignoringOtherApps: true)
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            DispatchQueue.main.async {
                authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
                if !granted && authStatus != .notDetermined {
                    openMicrophonePrivacySettings()
                }
            }
        }
    }

    var body: some View {
        Group {
            switch authStatus {
            case .authorized:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            case .notDetermined:
                Button("Grant") {
                    requestMicAccess()
                }
            case .denied, .restricted:
                Button("Open Settings") {
                    openMicrophonePrivacySettings()
                }
            @unknown default:
                Button("Grant") {
                    requestMicAccess()
                }
            }
        }
        .onAppear {
            authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        }
    }
}

struct ScreenRecordingPermissionButton: View {
    @State private var isGranted: Bool = CGPreflightScreenCaptureAccess()

    var body: some View {
        Group {
            if isGranted {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            } else {
                Button("Grant") {
                    CGRequestScreenCaptureAccess()
                    // Check again after a delay (user needs to interact with system dialog)
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                        isGranted = CGPreflightScreenCaptureAccess()
                    }
                }
            }
        }
        .onAppear {
            isGranted = CGPreflightScreenCaptureAccess()
        }
    }
}

// MARK: - Settings View

struct SettingsHeaderView: View {
    var body: some View {
        HStack(spacing: 12) {
            // App Icon
            if let logoPath = Bundle.main.path(forResource: "AppLogo", ofType: "png"),
               let logoImage = NSImage(contentsOfFile: logoPath) {
                Image(nsImage: logoImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 48, height: 48)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            } else {
                Image(systemName: "waveform.circle.fill")
                    .font(.system(size: 48))
                    .foregroundColor(.accentColor)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("VoiceFlow")
                    .font(.title2)
                    .fontWeight(.semibold)
                Text("v\(VoiceFlowBridge.version)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(.bottom, 8)
    }
}

// MARK: - Sidebar Navigation

enum SidebarPage: String, CaseIterable, Identifiable {
    case home = "Home"
    case models = "Models"
    case snippets = "Snippets"
    case style = "Style"
    case personas = "Personas"
    case appProfiles = "App Profiles"
    case browserSites = "Browser Sites"
    case aiFeatures = "AI Features"
    case settings = "Settings"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .home: return "house"
        case .models: return "cpu"
        case .snippets: return "scissors"
        case .style: return "textformat"
        case .personas: return "person.crop.circle"
        case .appProfiles: return "app.badge"
        case .browserSites: return "globe"
        case .aiFeatures: return "sparkles"
        case .settings: return "gearshape"
        }
    }

    var isBottom: Bool {
        self == .settings
    }
}

struct MainAppView: View {
    @EnvironmentObject var voiceFlow: VoiceFlowBridge
    @EnvironmentObject var snippetManager: SnippetManager
    @ObservedObject var transcriptionLog = TranscriptionLog.shared
    @StateObject private var modelManager = ModelManager()
    @State private var selectedPage: SidebarPage = .home

    var body: some View {
        NavigationSplitView {
            VStack(spacing: 0) {
                // Logo header
                HStack(spacing: 10) {
                    if let logoPath = Bundle.main.path(forResource: "AppLogo", ofType: "png"),
                       let logoImage = NSImage(contentsOfFile: logoPath) {
                        Image(nsImage: logoImage)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(width: 28, height: 28)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                    Text("VoiceFlow")
                        .font(.title3)
                        .fontWeight(.semibold)
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)
                .padding(.bottom, 16)

                // Top pages
                VStack(spacing: 2) {
                    ForEach(SidebarPage.allCases.filter { !$0.isBottom }) { page in
                        SidebarItemView(page: page, isSelected: selectedPage == page)
                            .onTapGesture { selectedPage = page }
                    }
                }
                .padding(.horizontal, 8)

                Spacer()

                // Bottom pages
                VStack(spacing: 2) {
                    ForEach(SidebarPage.allCases.filter { $0.isBottom }) { page in
                        SidebarItemView(page: page, isSelected: selectedPage == page)
                            .onTapGesture { selectedPage = page }
                    }
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 12)
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 200, max: 220)
        } detail: {
            Group {
                switch selectedPage {
                case .home:
                    HomeView()
                        .environmentObject(transcriptionLog)
                case .models:
                    ModelSettingsView()
                        .environmentObject(modelManager)
                case .snippets:
                    SnippetsSettingsView()
                        .environmentObject(snippetManager)
                case .style:
                    StyleSettingsView()
                case .personas:
                    PersonasSettingsView()
                case .appProfiles:
                    AppProfilesSettingsView()
                case .browserSites:
                    BrowserSiteRulesView()
                case .aiFeatures:
                    AIFeaturesSettingsView()
                case .settings:
                    GeneralSettingsView()
                        .environmentObject(voiceFlow)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(width: 780, height: 560)
    }
}

struct SidebarItemView: View {
    let page: SidebarPage
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: page.icon)
                .font(.system(size: 14))
                .frame(width: 20)
                .foregroundColor(isSelected ? .accentColor : .secondary)
            Text(page.rawValue)
                .font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                .foregroundColor(isSelected ? .primary : .secondary)
            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.accentColor.opacity(0.12) : Color.clear)
        )
        .contentShape(Rectangle())
    }
}

// Legacy SettingsView wrapper (for SwiftUI Settings scene)
struct SettingsView: View {
    @EnvironmentObject var voiceFlow: VoiceFlowBridge
    @EnvironmentObject var snippetManager: SnippetManager

    var body: some View {
        MainAppView()
            .environmentObject(voiceFlow)
            .environmentObject(snippetManager)
    }
}

// MARK: - Home View

struct HomeView: View {
    @EnvironmentObject var transcriptionLog: TranscriptionLog

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Welcome header
                VStack(alignment: .leading, spacing: 4) {
                    Text("Welcome back")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("Your voice-to-text activity")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.bottom, 4)

                // Stats row
                HStack(spacing: 16) {
                    StatCardView(
                        title: "Streak",
                        value: "\(transcriptionLog.streakDays)",
                        unit: transcriptionLog.streakDays == 1 ? "day" : "days",
                        icon: "flame",
                        color: .orange
                    )
                    StatCardView(
                        title: "Words",
                        value: formatNumber(transcriptionLog.totalWords),
                        unit: "total",
                        icon: "text.word.spacing",
                        color: .blue
                    )
                    StatCardView(
                        title: "Speed",
                        value: "\(transcriptionLog.averageWPM)",
                        unit: "wpm",
                        icon: "gauge.medium",
                        color: .green
                    )
                }

                // Transcription history
                VStack(alignment: .leading, spacing: 12) {
                    Text("Recent Activity")
                        .font(.headline)

                    if transcriptionLog.entries.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: "waveform")
                                .font(.system(size: 32))
                                .foregroundColor(.secondary.opacity(0.5))
                            Text("No transcriptions yet")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Text("Press ⌥ Space to start dictating")
                                .font(.caption)
                                .foregroundColor(.secondary.opacity(0.7))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 40)
                    } else {
                        LazyVStack(alignment: .leading, spacing: 0) {
                            ForEach(transcriptionLog.entriesByDay, id: \.0) { dayLabel, dayEntries in
                                // Day header
                                Text(dayLabel)
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.secondary)
                                    .padding(.top, 12)
                                    .padding(.bottom, 6)

                                ForEach(dayEntries) { entry in
                                    TranscriptionRowView(entry: entry)
                                }
                            }
                        }
                    }
                }
            }
            .padding()
        }
    }

    private func formatNumber(_ n: Int) -> String {
        if n >= 1000 {
            let k = Double(n) / 1000.0
            return String(format: "%.1fk", k)
        }
        return "\(n)"
    }
}

struct StatCardView: View {
    let title: String
    let value: String
    let unit: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12))
                    .foregroundColor(color)
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value)
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(NSColor.controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.secondary.opacity(0.15), lineWidth: 1)
        )
    }
}

struct TranscriptionRowView: View {
    let entry: TranscriptionEntry
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(entry.formattedText)
                    .font(.system(size: 13))
                    .lineLimit(2)
                Spacer()
                if entry.editedText != nil {
                    Text("Learned")
                        .font(.caption2)
                        .fontWeight(.medium)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.green.opacity(0.15))
                        .foregroundColor(.green)
                        .clipShape(Capsule())
                }
                Button(action: {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(entry.formattedText, forType: .string)
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                        copied = false
                    }
                }) {
                    Image(systemName: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 11))
                        .foregroundColor(copied ? .green : .secondary)
                }
                .buttonStyle(.plain)
                .help("Copy to clipboard")
                Text(timeString(entry.timestamp))
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            HStack(spacing: 8) {
                if !entry.targetApp.isEmpty {
                    Label(entry.targetApp, systemImage: "app")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                Text("\(entry.totalMs)ms")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 4)
        .overlay(
            Rectangle()
                .fill(Color.secondary.opacity(0.1))
                .frame(height: 1),
            alignment: .bottom
        )
    }

    private func timeString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        return formatter.string(from: date)
    }
}

// MARK: - Style Settings View

struct StyleSettingsView: View {
    @AppStorage("appProfilesEnabled") private var appProfilesEnabled = true
    @ObservedObject private var profileManager = AppProfileManager.shared
    @ObservedObject private var correctionManager = CorrectionManager.shared
    @State private var editingProfile: AppProfile? = nil
    @State private var newCorrectionOriginal = ""
    @State private var newCorrectionCorrected = ""

    @State private var formattingLevel: FormattingLevel = {
        let raw = UserDefaults.standard.string(forKey: "formattingLevel") ?? FormattingLevel.moderate.rawValue
        return FormattingLevel(rawValue: raw) ?? .moderate
    }()

    @State private var spacingMode: SpacingMode = {
        let raw = UserDefaults.standard.string(forKey: "spacingMode") ?? SpacingMode.contextAware.rawValue
        return SpacingMode(rawValue: raw) ?? .contextAware
    }()

    @State private var punctuationToggles: [PunctuationOption: Bool] = {
        var toggles: [PunctuationOption: Bool] = [:]
        for option in PunctuationOption.allCases {
            let key = "punctuation_\(option.rawValue)"
            if UserDefaults.standard.object(forKey: key) == nil {
                toggles[option] = option.defaultEnabled
            } else {
                toggles[option] = UserDefaults.standard.bool(forKey: key)
            }
        }
        return toggles
    }()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Page header
                VStack(alignment: .leading, spacing: 4) {
                    Text("Style & Formatting")
                        .font(.title2)
                        .fontWeight(.bold)
                    Text("Control how VoiceFlow formats your dictations")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.bottom, 4)

                // Formatting Level
                GroupBox {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(FormattingLevel.allCases, id: \.self) { level in
                            HStack(spacing: 10) {
                                Image(systemName: formattingLevel == level ? "largecircle.fill.circle" : "circle")
                                    .foregroundColor(formattingLevel == level ? .accentColor : .secondary)
                                    .font(.system(size: 16))
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(level.displayName)
                                        .font(.system(size: 13, weight: .medium))
                                    Text(level.description)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                            }
                            .padding(.vertical, 6)
                            .padding(.horizontal, 6)
                            .background(
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(formattingLevel == level ? Color.accentColor.opacity(0.08) : Color.clear)
                            )
                            .contentShape(Rectangle())
                            .onTapGesture {
                                formattingLevel = level
                                UserDefaults.standard.set(level.rawValue, forKey: "formattingLevel")
                            }
                        }
                    }
                    .padding(.vertical, 2)
                } label: {
                    Label("Formatting Level", systemImage: "textformat")
                        .font(.headline)
                }

                // Spacing Mode
                GroupBox {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(SpacingMode.allCases, id: \.self) { mode in
                            HStack(spacing: 10) {
                                Image(systemName: spacingMode == mode ? "largecircle.fill.circle" : "circle")
                                    .foregroundColor(spacingMode == mode ? .accentColor : .secondary)
                                    .font(.system(size: 16))
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(mode.displayName)
                                        .font(.system(size: 13, weight: .medium))
                                    Text(mode.description)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                Spacer()
                            }
                            .padding(.vertical, 6)
                            .padding(.horizontal, 6)
                            .background(
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(spacingMode == mode ? Color.accentColor.opacity(0.08) : Color.clear)
                            )
                            .contentShape(Rectangle())
                            .onTapGesture {
                                spacingMode = mode
                                UserDefaults.standard.set(mode.rawValue, forKey: "spacingMode")
                            }
                        }
                    }
                    .padding(.vertical, 2)
                } label: {
                    Label("Spacing", systemImage: "arrow.left.and.right.text.vertical")
                        .font(.headline)
                }

                // Punctuation Options
                GroupBox {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(PunctuationOption.allCases, id: \.self) { option in
                            Toggle(isOn: Binding(
                                get: { punctuationToggles[option] ?? option.defaultEnabled },
                                set: { newValue in
                                    punctuationToggles[option] = newValue
                                    UserDefaults.standard.set(newValue, forKey: "punctuation_\(option.rawValue)")
                                }
                            )) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(option.displayName)
                                        .font(.system(size: 13, weight: .medium))
                                    Text(option.description)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .toggleStyle(.switch)
                            .padding(.vertical, 2)
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Punctuation", systemImage: "textformat.abc.dottedunderline")
                        .font(.headline)
                }

                // Smart Context section header
                VStack(alignment: .leading, spacing: 2) {
                    Text("Smart Context")
                        .font(.headline)
                    Text("Context signals that feed into the LLM for better accuracy")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 4)

                // Visual Context
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 6) {
                            Image(systemName: "keyboard")
                                .font(.caption)
                                .foregroundColor(.blue)
                            Text("Hold **Control + Option + Space** to dictate with screen context.")
                                .font(.system(size: 13))
                        }
                        Text("Captures a screenshot and uses the Qwen3.5 multimodal model to incorporate on-screen terms, names, and context into your dictation. Requires Screen Recording permission.")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        if !ScreenshotCapture.hasPermission() {
                            Button("Grant Screen Recording Permission") {
                                ScreenshotCapture.requestPermission()
                            }
                            .font(.caption)
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Visual Context", systemImage: "eye")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }

                // App Profiles
                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        Toggle(isOn: $appProfilesEnabled) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Per-App Formatting")
                                    .font(.system(size: 13, weight: .medium))
                                Text("Adapt dictation style for email, Slack, code editors, and more.")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .toggleStyle(.switch)

                        if !profileManager.profiles.isEmpty {
                            Divider()

                            HStack {
                                Text("\(profileManager.profiles.count) app\(profileManager.profiles.count == 1 ? "" : "s") detected")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Spacer()
                                Button(action: {
                                    for profile in profileManager.profiles {
                                        profileManager.deleteProfile(profile)
                                    }
                                }) {
                                    Text("Clear All")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .buttonStyle(.plain)
                            }

                            VStack(spacing: 2) {
                                ForEach(Array(profileManager.profiles.sorted(by: { $0.displayName < $1.displayName }).enumerated()), id: \.element.id) { index, profile in
                                    HStack(spacing: 10) {
                                        Text(profile.displayName)
                                            .font(.system(size: 12, weight: .medium))
                                        Spacer()
                                        Text(appProfileCategoryLabel(profile.category))
                                            .font(.caption2)
                                            .fontWeight(.medium)
                                            .padding(.horizontal, 7)
                                            .padding(.vertical, 3)
                                            .background(appProfileCategoryColor(profile.category).opacity(0.12))
                                            .foregroundColor(appProfileCategoryColor(profile.category))
                                            .clipShape(Capsule())
                                        Image(systemName: "chevron.right")
                                            .font(.system(size: 10, weight: .semibold))
                                            .foregroundColor(.secondary.opacity(0.4))
                                    }
                                    .padding(.vertical, 7)
                                    .padding(.horizontal, 8)
                                    .background(
                                        RoundedRectangle(cornerRadius: 6)
                                            .fill(index % 2 == 0 ? Color(NSColor.controlBackgroundColor).opacity(0.5) : Color.clear)
                                    )
                                    .contentShape(Rectangle())
                                    .onTapGesture {
                                        editingProfile = profile
                                    }
                                }
                            }
                        } else {
                            HStack(spacing: 6) {
                                Image(systemName: "app.dashed")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text("No apps detected yet. Dictate in any app and it'll appear here.")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("App Profiles", systemImage: "app.badge")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
                .sheet(item: $editingProfile) { profile in
                    AppProfileEditView(profile: profile) { updated in
                        profileManager.updateProfile(updated)
                    }
                }

                // Correction History
                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Add word corrections to automatically fix common transcription errors.")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        // Manual quick-add form
                        HStack(spacing: 8) {
                            TextField("Wrong word", text: $newCorrectionOriginal)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                            Image(systemName: "arrow.right")
                                .foregroundColor(.secondary)
                            TextField("Right word", text: $newCorrectionCorrected)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                            Button("Add") {
                                guard !newCorrectionOriginal.isEmpty && !newCorrectionCorrected.isEmpty else { return }
                                correctionManager.addManualCorrection(
                                    original: newCorrectionOriginal,
                                    corrected: newCorrectionCorrected
                                )
                                newCorrectionOriginal = ""
                                newCorrectionCorrected = ""
                            }
                            .disabled(newCorrectionOriginal.isEmpty || newCorrectionCorrected.isEmpty)
                        }

                        if !correctionManager.patterns.isEmpty {
                            Divider()

                            HStack {
                                Text("\(correctionManager.patterns.count) correction\(correctionManager.patterns.count == 1 ? "" : "s")")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Spacer()
                                Button(action: {
                                    correctionManager.clearAll()
                                }) {
                                    Text("Clear All")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                                .buttonStyle(.plain)
                            }

                            VStack(spacing: 0) {
                                ForEach(Array(correctionManager.patterns.suffix(10).reversed().enumerated()), id: \.element.id) { index, pattern in
                                    HStack(spacing: 0) {
                                        Text(pattern.original)
                                            .font(.system(size: 12, design: .monospaced))
                                            .foregroundColor(.secondary)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                        Image(systemName: "arrow.right")
                                            .font(.system(size: 9, weight: .bold))
                                            .foregroundColor(.accentColor.opacity(0.6))
                                            .frame(width: 24)
                                        Text(pattern.corrected)
                                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                                            .foregroundColor(.primary)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                        Button(action: {
                                            withAnimation(.easeOut(duration: 0.2)) {
                                                correctionManager.deletePattern(id: pattern.id)
                                            }
                                        }) {
                                            Image(systemName: "xmark")
                                                .font(.system(size: 9, weight: .semibold))
                                                .foregroundColor(.secondary.opacity(0.4))
                                                .frame(width: 20, height: 20)
                                        }
                                        .buttonStyle(.plain)
                                    }
                                    .padding(.vertical, 6)
                                    .padding(.horizontal, 8)
                                    .background(
                                        RoundedRectangle(cornerRadius: 4)
                                            .fill(index % 2 == 0 ? Color(NSColor.controlBackgroundColor).opacity(0.5) : Color.clear)
                                    )
                                }
                            }
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                        } else {
                            HStack(spacing: 6) {
                                Image(systemName: "sparkles")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text("No corrections yet. Add one above to fix common transcription errors.")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Correction History", systemImage: "arrow.triangle.2.circlepath")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }
            }
            .padding()
        }
    }

    private func appProfileCategoryLabel(_ category: String) -> String {
        switch category {
        case "email": return "Email"
        case "slack": return "Slack/Chat"
        case "code": return "Code"
        default: return "General"
        }
    }

    private func appProfileCategoryColor(_ category: String) -> Color {
        switch category {
        case "email": return .blue
        case "slack": return .purple
        case "code": return .green
        default: return .gray
        }
    }

}

// MARK: - AI Features Settings View

struct AIFeaturesSettingsView: View {
    @AppStorage("aiFeatureSummarizeEnabled") private var summarizeEnabled = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("AI Features")
                    .font(.title2)
                    .fontWeight(.bold)

                Text("Voice-activated AI commands that go beyond dictation.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle(isOn: $summarizeEnabled) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Summarize This")
                                    .font(.system(size: 13, weight: .medium))
                                Text("Say \"summarize this\" while in a text field to append a bullet-point summary of the field's content.")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .toggleStyle(.switch)

                        Divider()

                        HStack(spacing: 6) {
                            Image(systemName: "info.circle")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text("When enabled, saying \"summarize this\" triggers summarization instead of being typed as dictation.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Summarize This", systemImage: "text.badge.star")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                }

                Spacer()
            }
            .padding(20)
        }
    }
}

// MARK: - General Settings Tab

struct GeneralSettingsView: View {
    @EnvironmentObject var voiceFlow: VoiceFlowBridge
    @State private var launchAtLogin = SMAppService.mainApp.status == .enabled

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Hotkey Section
                GroupBox {
                    HStack {
                        Label("Recording Shortcut", systemImage: "command")
                        Spacer()
                        Text("⌥ Space")
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(Color.secondary.opacity(0.15))
                            .cornerRadius(6)
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Hotkey", systemImage: "keyboard")
                        .font(.headline)
                }

                // Startup Section
                GroupBox {
                    Toggle("Launch VoiceFlow at login", isOn: $launchAtLogin)
                        .onChange(of: launchAtLogin) { newValue in
                            if newValue {
                                try? SMAppService.mainApp.register()
                            } else {
                                try? SMAppService.mainApp.unregister()
                            }
                        }
                        .padding(.vertical, 4)
                } label: {
                    Label("Startup", systemImage: "power")
                        .font(.headline)
                }

                // Status Section
                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Label("Pipeline", systemImage: "cpu")
                            Spacer()
                            if voiceFlow.isInitialized {
                                HStack(spacing: 4) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.green)
                                    Text("Ready")
                                        .foregroundColor(.green)
                                }
                            } else {
                                HStack(spacing: 6) {
                                    ProgressView()
                                        .scaleEffect(0.7)
                                    Text("Initializing...")
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Status", systemImage: "info.circle")
                        .font(.headline)
                }

                // Last Processing Section
                if let result = voiceFlow.lastResult {
                    GroupBox {
                        VStack(spacing: 8) {
                            HStack {
                                Text("Transcription")
                                Spacer()
                                Text("\(result.transcriptionMs)ms")
                                    .monospacedDigit()
                                    .foregroundColor(.secondary)
                            }
                            HStack {
                                Text("LLM Formatting")
                                Spacer()
                                Text("\(result.llmMs)ms")
                                    .monospacedDigit()
                                    .foregroundColor(.secondary)
                            }
                            Divider()
                            HStack {
                                Text("Total")
                                    .fontWeight(.medium)
                                Spacer()
                                Text("\(result.totalMs)ms")
                                    .monospacedDigit()
                                    .fontWeight(.medium)
                            }
                        }
                        .padding(.vertical, 4)
                    } label: {
                        Label("Last Processing", systemImage: "clock")
                            .font(.headline)
                    }
                }

                // Permissions Section
                GroupBox {
                    VStack(spacing: 12) {
                        HStack {
                            HStack(spacing: 8) {
                                Image(systemName: "hand.raised.fill")
                                    .foregroundColor(.blue)
                                    .frame(width: 20)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Accessibility")
                                    Text("Required for auto-paste")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            Spacer()
                            AccessibilityPermissionButton()
                        }

                        Divider()

                        HStack {
                            HStack(spacing: 8) {
                                Image(systemName: "mic.fill")
                                    .foregroundColor(.red)
                                    .frame(width: 20)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Microphone")
                                    Text("Required for recording")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            Spacer()
                            MicrophonePermissionButton()
                        }

                        Divider()

                        HStack {
                            HStack(spacing: 8) {
                                Image(systemName: "rectangle.dashed.badge.record")
                                    .foregroundColor(.purple)
                                    .frame(width: 20)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Screen Recording")
                                    Text("Required for visual context")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            Spacer()
                            ScreenRecordingPermissionButton()
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Permissions", systemImage: "lock.shield")
                        .font(.headline)
                }

                Spacer(minLength: 16)

                // Action Buttons
                HStack {
                    Button(action: restartApp) {
                        Label("Restart", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)

                    Spacer()

                    Button(action: { NSApp.terminate(nil) }) {
                        Text("Quit VoiceFlow")
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }
            }
            .padding()
        }
    }

    private func restartApp() {
        let bundlePath = Bundle.main.bundlePath

        // Use /bin/sh -c with nohup + disown so the relaunch survives our termination
        let script = "sleep 1 && open \"\(bundlePath)\" &"
        let task = Process()
        task.executableURL = URL(fileURLWithPath: "/bin/sh")
        task.arguments = ["-c", script]
        task.standardOutput = FileHandle.nullDevice
        task.standardError = FileHandle.nullDevice
        // Setting qualityOfService to .background helps the process survive parent exit
        task.qualityOfService = .background
        try? task.run()

        // Terminate after a brief delay to let the shell process detach
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            NSApp.terminate(nil)
        }
    }
}

// MARK: - Snippets Settings Tab

struct SnippetsSettingsView: View {
    @EnvironmentObject var snippetManager: SnippetManager
    @State private var showingAddSheet = false
    @State private var editingSnippet: VoiceSnippet? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Voice Snippets")
                    .font(.headline)
                Spacer()
                Button(action: { showingAddSheet = true }) {
                    Label("Add", systemImage: "plus")
                }
            }

            Text("Say a trigger phrase and it will expand to the full text.")
                .font(.caption)
                .foregroundColor(.secondary)

            if snippetManager.snippets.isEmpty {
                VStack(spacing: 8) {
                    Spacer()
                    Image(systemName: "text.badge.plus")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text("No snippets yet")
                        .foregroundColor(.secondary)
                    Text("Click + to add your first snippet")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            } else {
                List {
                    ForEach(snippetManager.snippets) { snippet in
                        SnippetRowView(snippet: snippet)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                editingSnippet = snippet
                            }
                    }
                    .onDelete(perform: snippetManager.deleteSnippets)
                }
                .listStyle(.bordered)
            }
        }
        .padding()
        .sheet(isPresented: $showingAddSheet) {
            SnippetEditView(mode: .add) { trigger, expansion in
                snippetManager.addSnippet(trigger: trigger, expansion: expansion)
            }
        }
        .sheet(item: $editingSnippet) { snippet in
            SnippetEditView(mode: .edit(snippet)) { trigger, expansion in
                var updated = snippet
                updated.trigger = trigger
                updated.expansion = expansion
                snippetManager.updateSnippet(updated)
            }
        }
    }
}

struct SnippetRowView: View {
    let snippet: VoiceSnippet

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("\"\(snippet.trigger)\"")
                    .fontWeight(.medium)
                Spacer()
                Image(systemName: "arrow.right")
                    .foregroundColor(.secondary)
                    .font(.caption)
            }
            Text(snippet.expansion.prefix(50) + (snippet.expansion.count > 50 ? "..." : ""))
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(1)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Snippet Edit Sheet

struct SnippetEditView: View {
    enum Mode {
        case add
        case edit(VoiceSnippet)
    }

    let mode: Mode
    let onSave: (String, String) -> Void

    @Environment(\.dismiss) var dismiss
    @State private var trigger: String = ""
    @State private var expansion: String = ""

    var title: String {
        switch mode {
        case .add: return "Add Snippet"
        case .edit: return "Edit Snippet"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.headline)

            VStack(alignment: .leading, spacing: 4) {
                Text("Trigger Phrase")
                    .font(.caption)
                    .foregroundColor(.secondary)
                TextField("e.g., my signature", text: $trigger)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Expands To")
                    .font(.caption)
                    .foregroundColor(.secondary)
                TextEditor(text: $expansion)
                    .font(.body)
                    .frame(height: 100)
                    .border(Color.secondary.opacity(0.3), width: 1)
            }

            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Save") {
                    onSave(trigger, expansion)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(trigger.isEmpty || expansion.isEmpty)
            }
        }
        .padding()
        .frame(width: 350, height: 280)
        .onAppear {
            if case .edit(let snippet) = mode {
                trigger = snippet.trigger
                expansion = snippet.expansion
            }
        }
    }
}

// MARK: - App Profiles Settings View

struct AppProfilesSettingsView: View {
    @ObservedObject var profileManager = AppProfileManager.shared
    @State private var editingProfile: AppProfile? = nil
    @State private var lastAutoMapCount: Int? = nil

    @State private var isClassifying: Bool = false
    @State private var classifyProgress: String = ""
    @State private var classifyResult: String? = nil
    private let classifier = PersonaClassifier()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("App Profiles")
                        .font(.headline)
                    Spacer()
                    Button {
                        Task { await runLLMClassify() }
                    } label: {
                        Label("Classify Unmapped (LLM)", systemImage: "brain")
                    }
                    .disabled(isClassifying)
                    .help("Use the local LLM to classify apps that don't have a persona yet (≥90% confidence required)")
                    Button {
                        lastAutoMapCount = profileManager.autoMapInstalledApps()
                    } label: {
                        Label("Auto-Detect", systemImage: "wand.and.stars")
                    }
                    .help("Scan installed apps and assign personas based on common bundle IDs")
                }
                Text("Apps are auto-detected when you dictate. Customize formatting per app, or tie an app to a Persona.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                if let n = lastAutoMapCount {
                    Text(n == 0
                         ? "Auto-detect: no new mappings (everything already covered)."
                         : "Auto-detect: mapped \(n) app\(n == 1 ? "" : "s") to personas.")
                        .font(.caption)
                        .foregroundColor(.accentColor)
                }
                if isClassifying {
                    HStack(spacing: 6) {
                        ProgressView().scaleEffect(0.6)
                        Text(classifyProgress).font(.caption).foregroundColor(.secondary)
                    }
                }
                if let result = classifyResult, !isClassifying {
                    Text(result).font(.caption).foregroundColor(.accentColor)
                }
            }
            .padding()

            Divider()

            if profileManager.profiles.isEmpty {
                VStack(spacing: 8) {
                    Spacer()
                    Image(systemName: "app.badge")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text("No apps detected yet").foregroundColor(.secondary)
                    Text("Start dictating and they'll appear here")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            } else {
                List {
                    ForEach(profileManager.profiles.sorted(by: { $0.displayName < $1.displayName })) { profile in
                        AppProfileRowView(profile: profile)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                editingProfile = profile
                            }
                    }
                    .onDelete { offsets in
                        let sorted = profileManager.profiles.sorted(by: { $0.displayName < $1.displayName })
                        for offset in offsets {
                            profileManager.deleteProfile(sorted[offset])
                        }
                    }
                }
                .listStyle(.inset)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .sheet(item: $editingProfile) { profile in
            AppProfileEditView(profile: profile) { updated in
                profileManager.updateProfile(updated)
            }
        }
    }

    /// Drive PersonaClassifier across every unmapped AppProfile, updating UI state
    /// as it goes. Confidence threshold defaults to 80%.
    private func runLLMClassify() async {
        isClassifying = true
        classifyResult = nil
        classifyProgress = "Starting…"
        let result = await classifier.classifyUnmappedApps { done, total, app in
            classifyProgress = "Classifying \(done)/\(total): \(app)"
        }
        isClassifying = false
        if result.considered == 0 {
            classifyResult = "Nothing to classify — every profile already has a persona or custom prompt."
        } else {
            classifyResult = "Applied \(result.applied) of \(result.considered) (≥90% confidence). Skipped \(result.considered - result.applied) low-confidence."
        }
    }
}

struct AppProfileRowView: View {
    let profile: AppProfile
    @ObservedObject private var personaManager = PersonaManager.shared

    /// Resolution: persona name > custom prompt label > category capitalized.
    private var badgeText: String {
        if let pid = profile.personaId,
           let p = personaManager.persona(byId: pid) {
            return p.name
        }
        if let custom = profile.customPrompt, !custom.isEmpty {
            return "Custom"
        }
        return profile.category.capitalized
    }

    private var badgeColor: Color {
        if profile.personaId != nil { return .accentColor }
        if profile.customPrompt?.isEmpty == false { return .orange }
        switch profile.category {
        case "email": return .blue
        case "slack": return .purple
        case "code": return .green
        default: return .gray
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(profile.displayName)
                    .fontWeight(.medium)
                Spacer()
                Text(badgeText)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(badgeColor.opacity(0.2))
                    .foregroundColor(badgeColor)
                    .clipShape(Capsule())
            }
            if let custom = profile.customPrompt, !custom.isEmpty, profile.personaId == nil {
                Text(String(custom.prefix(60)) + (custom.count > 60 ? "..." : ""))
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 4)
    }
}

struct AppProfileEditView: View {
    let profile: AppProfile
    let onSave: (AppProfile) -> Void

    @Environment(\.dismiss) var dismiss
    @ObservedObject private var personaManager = PersonaManager.shared
    @State private var category: String = "default"
    @State private var customPrompt: String = ""
    @State private var personaId: String? = nil

    private let categories = ["default", "email", "slack", "code"]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Edit \(profile.displayName)")
                .font(.headline)

            Text("Bundle ID: \(profile.id)")
                .font(.caption)
                .foregroundColor(.secondary)

            VStack(alignment: .leading, spacing: 4) {
                Text("Persona")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Tie this app to a persona. Persona prompt overrides custom prompt and category.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Picker("", selection: $personaId) {
                    Text("None (use category/custom)").tag(String?.none)
                    ForEach(personaManager.personas) { persona in
                        Text(persona.name).tag(String?.some(persona.id))
                    }
                }
                .labelsHidden()
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Category")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Picker("", selection: $category) {
                    Text("General").tag("default")
                    Text("Email").tag("email")
                    Text("Slack/Chat").tag("slack")
                    Text("Code").tag("code")
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .disabled(personaId != nil)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Custom Prompt (optional)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Overrides the category default. Leave empty to use category formatting. Ignored when a persona is selected.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                TextEditor(text: $customPrompt)
                    .font(.body)
                    .frame(height: 100)
                    .border(Color.secondary.opacity(0.3), width: 1)
                    .disabled(personaId != nil)
            }

            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Save") {
                    var updated = profile
                    updated.category = category
                    updated.customPrompt = customPrompt.isEmpty ? nil : customPrompt
                    updated.personaId = personaId
                    onSave(updated)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(width: 480, height: 440)
        .onAppear {
            category = profile.category
            customPrompt = profile.customPrompt ?? ""
            personaId = profile.personaId
        }
    }
}

// MARK: - Browser Site Rules Settings

struct BrowserSiteRulesView: View {
    @ObservedObject var manager = BrowserSiteRulesManager.shared
    @ObservedObject var personaManager = PersonaManager.shared
    @State private var editingRule: BrowserSiteRule? = nil
    @State private var showingNew = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Fixed header (outside the scrolling list so it never scrolls away)
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Browser Site Rules")
                        .font(.headline)
                    Spacer()
                    Button {
                        showingNew = true
                    } label: {
                        Label("New Rule", systemImage: "plus")
                    }
                }
                Text("When you dictate while a browser is in the foreground, VoiceFlow reads the active tab's URL and applies a persona based on these rules. Use \"*.example.com\" for subdomain wildcards. First matching rule wins (exact host beats wildcard). First use of each browser triggers a macOS Automation permission prompt — grant it.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding()

            Divider()

            // Scrolling list — owns its own height, never gets squashed by parent.
            if manager.rules.isEmpty {
                Spacer()
                Text("No rules yet")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
                Spacer()
            } else {
                List {
                    ForEach(manager.rules) { rule in
                        let personaName = personaManager.persona(byId: rule.personaId)?.name ?? "(missing persona)"
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(rule.hostnamePattern).font(.system(.body, design: .monospaced))
                                Text(personaName).font(.caption).foregroundColor(.secondary)
                            }
                            Spacer()
                            if rule.isBuiltIn {
                                Text("Built-in").font(.caption2)
                                    .padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(Color.secondary.opacity(0.2))
                                    .clipShape(Capsule())
                            }
                        }
                        .padding(.vertical, 2)
                        .contentShape(Rectangle())
                        .onTapGesture { editingRule = rule }
                    }
                    .onDelete { offsets in
                        for offset in offsets {
                            manager.delete(manager.rules[offset])
                        }
                    }
                }
                .listStyle(.inset)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .sheet(item: $editingRule) { rule in
            BrowserSiteRuleEditView(rule: rule) { updated in
                manager.upsert(updated)
            }
        }
        .sheet(isPresented: $showingNew) {
            BrowserSiteRuleEditView(
                rule: BrowserSiteRule(
                    id: UUID().uuidString,
                    hostnamePattern: "",
                    personaId: personaManager.personas.first?.id ?? "",
                    createdAt: Date(),
                    isBuiltIn: false
                )
            ) { updated in
                manager.upsert(updated)
            }
        }
    }
}

struct BrowserSiteRuleEditView: View {
    let rule: BrowserSiteRule
    let onSave: (BrowserSiteRule) -> Void

    @Environment(\.dismiss) var dismiss
    @ObservedObject private var personaManager = PersonaManager.shared
    @State private var hostnamePattern: String = ""
    @State private var personaId: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(rule.hostnamePattern.isEmpty ? "New Site Rule" : "Edit Rule")
                .font(.headline)

            VStack(alignment: .leading, spacing: 4) {
                Text("Hostname pattern").font(.caption).foregroundColor(.secondary)
                Text("Examples: \"github.com\", \"mail.google.com\", \"*.atlassian.net\"")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                TextField("github.com", text: $hostnamePattern)
                    .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled(true)
                    .font(.system(.body, design: .monospaced))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Persona").font(.caption).foregroundColor(.secondary)
                Picker("", selection: $personaId) {
                    ForEach(personaManager.personas) { p in
                        Text(p.name).tag(p.id)
                    }
                }
                .labelsHidden()
            }

            HStack {
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Spacer()
                Button("Save") {
                    var updated = rule
                    updated.hostnamePattern = hostnamePattern.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    updated.personaId = personaId
                    onSave(updated)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(hostnamePattern.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || personaId.isEmpty)
            }
        }
        .padding()
        .frame(width: 460, height: 240)
        .onAppear {
            hostnamePattern = rule.hostnamePattern
            personaId = rule.personaId.isEmpty
                ? (personaManager.personas.first?.id ?? "")
                : rule.personaId
        }
    }
}

// MARK: - Personas Settings

struct PersonasSettingsView: View {
    @ObservedObject var personaManager = PersonaManager.shared
    @State private var editingPersona: Persona? = nil
    @State private var showingNew = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Personas")
                        .font(.headline)
                    Spacer()
                    Button {
                        showingNew = true
                    } label: {
                        Label("New", systemImage: "plus")
                    }
                }
                Text("Personas are reusable LLM context fragments. Tie an app to a persona in App Profiles.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding()

            Divider()

            if personaManager.personas.isEmpty {
                Spacer()
                Text("No personas yet").foregroundColor(.secondary).frame(maxWidth: .infinity)
                Spacer()
            } else {
                List {
                    ForEach(personaManager.personas) { persona in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(persona.name).fontWeight(.medium)
                                if persona.isBuiltIn {
                                    Text("Built-in")
                                        .font(.caption2)
                                        .padding(.horizontal, 6).padding(.vertical, 2)
                                        .background(Color.secondary.opacity(0.2))
                                        .clipShape(Capsule())
                                }
                                Spacer()
                            }
                            Text(String(persona.prompt.prefix(120)) + (persona.prompt.count > 120 ? "..." : ""))
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .lineLimit(2)
                        }
                        .padding(.vertical, 4)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            editingPersona = persona
                        }
                    }
                    .onDelete { offsets in
                        for offset in offsets {
                            personaManager.delete(personaManager.personas[offset])
                        }
                    }
                }
                .listStyle(.inset)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .sheet(item: $editingPersona) { persona in
            PersonaEditView(persona: persona) { updated in
                personaManager.upsert(updated)
            }
        }
        .sheet(isPresented: $showingNew) {
            PersonaEditView(
                persona: Persona(
                    id: UUID().uuidString,
                    name: "",
                    prompt: "",
                    createdAt: Date(),
                    isBuiltIn: false
                )
            ) { updated in
                personaManager.upsert(updated)
            }
        }
    }
}

struct PersonaEditView: View {
    let persona: Persona
    let onSave: (Persona) -> Void

    @Environment(\.dismiss) var dismiss
    @State private var name: String = ""
    @State private var prompt: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(persona.name.isEmpty ? "New Persona" : "Edit \(persona.name)")
                .font(.headline)

            VStack(alignment: .leading, spacing: 4) {
                Text("Name").font(.caption).foregroundColor(.secondary)
                TextField("e.g. Software Engineer", text: $name)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Prompt").font(.caption).foregroundColor(.secondary)
                Text("This text is injected into the LLM system context whenever an app tied to this persona is the foreground app.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                TextEditor(text: $prompt)
                    .font(.body)
                    .frame(minHeight: 220)
                    .border(Color.secondary.opacity(0.3), width: 1)
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                Spacer()
                Button("Save") {
                    var updated = persona
                    updated.name = name.trimmingCharacters(in: .whitespacesAndNewlines)
                    updated.prompt = prompt
                    onSave(updated)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                          || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding()
        .frame(width: 540, height: 460)
        .onAppear {
            name = persona.name
            prompt = persona.prompt
        }
    }
}

// MARK: - Model Manager

struct LLMModel: Identifiable {
    let id: String
    let displayName: String
    let filename: String
    let sizeGB: Float
    let downloadUrl: String
    var isDownloaded: Bool
}

// MARK: - STT Engine Types

enum SttEngine: String, CaseIterable {
    case whisper = "whisper"
    case moonshine = "moonshine"
    case qwen3Asr = "qwen3-asr"

    var displayName: String {
        switch self {
        case .whisper: return "Whisper"
        case .moonshine: return "Moonshine"
        case .qwen3Asr: return "Qwen3-ASR"
        }
    }

    var description: String {
        switch self {
        case .whisper: return "OpenAI Whisper - accurate, proven technology"
        case .moonshine: return "Moonshine - 5x faster, lower memory usage"
        case .qwen3Asr: return "Qwen3-ASR - high-quality ASR with LLM formatting"
        }
    }

    /// Whether this engine is external (Python daemon, not in Rust pipeline)
    var isExternal: Bool {
        switch self {
        case .qwen3Asr: return true
        default: return false
        }
    }
}

struct MoonshineModel: Identifiable {
    let id: String
    let displayName: String
    let sizeMB: UInt32
    var isDownloaded: Bool
}

/// Pipeline mode (mirrors Rust PipelineMode enum)
enum PipelineMode: String, CaseIterable {
    case sttPlusLlm = "stt-plus-llm"
    case consolidated = "consolidated"
    case audioDirect = "audio-direct"

    var displayName: String {
        switch self {
        case .sttPlusLlm: return "STT + LLM (traditional)"
        case .consolidated: return "Consolidated (single model)"
        case .audioDirect: return "Audio Direct (Gemma 4)"
        }
    }

    var description: String {
        switch self {
        case .sttPlusLlm: return "Separate speech-to-text and language model stages"
        case .consolidated: return "Single model handles audio-to-text (Qwen3-ASR via MLX)"
        case .audioDirect: return "Audio-native LLM transcribes and formats in one pass (Gemma 4)"
        }
    }
}

/// Consolidated model info (mirrors Rust ConsolidatedModel)
struct ConsolidatedModelItem: Identifiable {
    let id: String
    let displayName: String
    let dirName: String
    let sizeGB: Float
    var isDownloaded: Bool
}

class ModelManager: ObservableObject {
    @Published var models: [LLMModel] = []
    @Published var currentModelId: String = ""
    @Published var isDownloading: Bool = false
    @Published var downloadProgress: Double = 0
    @Published var downloadError: String?
    @Published var needsRestart: Bool = false
    @Published var downloadingModelId: String?

    // STT Engine settings
    @Published var currentSttEngine: SttEngine = .whisper
    @Published var currentMoonshineModelId: String = "tiny"
    @Published var moonshineModels: [MoonshineModel] = []
    @Published var downloadingMoonshineModelId: String?
    @Published var moonshineDownloadProgress: Double = 0

    // Pipeline mode and consolidated model settings
    @Published var currentPipelineMode: PipelineMode = .sttPlusLlm
    @Published var currentConsolidatedModelId: String = "qwen3-asr-0.6b"
    @Published var consolidatedModels: [ConsolidatedModelItem] = []
    @Published var downloadingConsolidatedModelId: String?
    @Published var consolidatedDownloadProgress: Double = 0

    private var downloadTask: URLSessionDownloadTask?
    private var moonshineDownloadTasks: [URLSessionDownloadTask] = []

    /// HuggingFace repos for consolidated models
    private static let consolidatedHfRepos: [String: String] = [
        "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
        "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    ]

    /// Required files for consolidated model inference (per model)
    private static let consolidatedRequiredFiles: [String: [String]] = [
        "qwen3-asr-0.6b": [
            "config.json",
            "chat_template.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "generation_config.json",
            "model.safetensors",
        ],
        "qwen3-asr-1.7b": [
            "config.json",
            "chat_template.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    ]

    // Models directory path - ~/Library/Application Support/com.era-laboratories.voiceflow/models/
    private var modelsDir: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("com.era-laboratories.voiceflow/models")
    }

    // Config file path - ~/Library/Application Support/com.era-laboratories.voiceflow/config.toml
    private var configPath: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("com.era-laboratories.voiceflow/config.toml")
    }

    // Available models - filenames must match Rust config exactly
    private static let availableModels: [(id: String, name: String, filename: String, size: Float, repo: String, mmprojFilename: String?, mmprojHfFilename: String?, mmprojSize: Float)] = [
        ("qwen3.5-0.8b", "Qwen3.5 0.8B", "Qwen3.5-0.8B-Q4_K_M.gguf", 0.53, "unsloth/Qwen3.5-0.8B-GGUF", "mmproj-Qwen3.5-0.8B-F16.gguf", "mmproj-F16.gguf", 0.2),
        ("qwen3.5-2b", "Qwen3.5 2B", "Qwen3.5-2B-Q4_K_M.gguf", 1.28, "unsloth/Qwen3.5-2B-GGUF", "mmproj-Qwen3.5-2B-F16.gguf", "mmproj-F16.gguf", 0.65),
        ("qwen3.5-4b", "Qwen3.5 4B", "Qwen3.5-4B-Q4_K_M.gguf", 2.74, "unsloth/Qwen3.5-4B-GGUF", "mmproj-Qwen3.5-4B-F16.gguf", "mmproj-F16.gguf", 0.65),
    ]

    init() {
        loadModels()
        loadCurrentModel()
        loadSttSettings()
        loadConsolidatedSettings()
    }

    // MARK: - Pipeline Mode and Consolidated Model Management

    func loadConsolidatedSettings() {
        // Load pipeline mode
        if let modePtr = voiceflow_current_pipeline_mode() {
            let modeStr = String(cString: modePtr)
            voiceflow_free_string(modePtr)
            currentPipelineMode = PipelineMode(rawValue: modeStr) ?? .sttPlusLlm
        }

        // Load current consolidated model
        if let modelPtr = voiceflow_current_consolidated_model() {
            currentConsolidatedModelId = String(cString: modelPtr)
            voiceflow_free_string(modelPtr)
        }

        // Load consolidated model info
        let count = voiceflow_consolidated_model_count()
        var items: [ConsolidatedModelItem] = []
        for i in 0..<count {
            let info = voiceflow_consolidated_model_info(i)
            if let idPtr = info.id, let namePtr = info.display_name, let dirPtr = info.dir_name {
                let item = ConsolidatedModelItem(
                    id: String(cString: idPtr),
                    displayName: String(cString: namePtr),
                    dirName: String(cString: dirPtr),
                    sizeGB: info.size_gb,
                    isDownloaded: info.is_downloaded
                )
                items.append(item)
            }
            voiceflow_free_consolidated_model_info(info)
        }
        consolidatedModels = items
    }

    func selectPipelineMode(_ mode: PipelineMode) {
        guard mode != currentPipelineMode else { return }

        let success = mode.rawValue.withCString { cString in
            voiceflow_set_pipeline_mode(cString)
        }

        if success {
            currentPipelineMode = mode
            needsRestart = true
        } else {
            downloadError = "Failed to set pipeline mode"
        }
    }

    func selectConsolidatedModel(_ modelId: String) {
        guard modelId != currentConsolidatedModelId else { return }

        let success = modelId.withCString { cString in
            voiceflow_set_consolidated_model(cString)
        }

        if success {
            currentConsolidatedModelId = modelId
            needsRestart = true
        } else {
            downloadError = "Failed to set consolidated model"
        }
    }

    func downloadConsolidatedModel(_ modelId: String) {
        guard downloadingConsolidatedModelId == nil else { return }
        guard let model = consolidatedModels.first(where: { $0.id == modelId }) else {
            downloadError = "Consolidated model not found"
            return
        }
        guard let hfRepo = Self.consolidatedHfRepos[modelId] else {
            downloadError = "No HuggingFace repo for model \(modelId)"
            return
        }
        guard let files = Self.consolidatedRequiredFiles[modelId] else {
            downloadError = "No file list for model \(modelId)"
            return
        }

        let destDir = modelsDir.appendingPathComponent(model.dirName)

        // Create model directory
        do {
            try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            downloadError = "Failed to create model directory: \(error.localizedDescription)"
            return
        }

        downloadingConsolidatedModelId = modelId
        consolidatedDownloadProgress = 0
        downloadError = nil

        var completedFiles = 0
        let totalFiles = files.count

        func downloadNextFile(_ index: Int) {
            guard index < files.count else {
                // All done
                DispatchQueue.main.async { [weak self] in
                    self?.downloadingConsolidatedModelId = nil
                    self?.consolidatedDownloadProgress = 1.0
                    self?.loadConsolidatedSettings() // Refresh model status
                }
                return
            }

            let filename = files[index]
            let urlString = "https://huggingface.co/\(hfRepo)/resolve/main/\(filename)"
            guard let url = URL(string: urlString) else {
                DispatchQueue.main.async { [weak self] in
                    self?.downloadError = "Invalid URL for \(filename)"
                    self?.downloadingConsolidatedModelId = nil
                }
                return
            }

            let destFile = destDir.appendingPathComponent(filename)

            let task = URLSession.shared.downloadTask(with: url) { [weak self] tempURL, response, error in
                if let error = error {
                    DispatchQueue.main.async {
                        self?.downloadError = "Download failed: \(error.localizedDescription)"
                        self?.downloadingConsolidatedModelId = nil
                    }
                    return
                }

                guard let tempURL = tempURL else {
                    DispatchQueue.main.async {
                        self?.downloadError = "Download failed: no file"
                        self?.downloadingConsolidatedModelId = nil
                    }
                    return
                }

                do {
                    if FileManager.default.fileExists(atPath: destFile.path) {
                        try FileManager.default.removeItem(at: destFile)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: destFile)

                    completedFiles += 1
                    DispatchQueue.main.async {
                        self?.consolidatedDownloadProgress = Double(completedFiles) / Double(totalFiles)
                    }

                    // Download next file
                    downloadNextFile(index + 1)
                } catch {
                    DispatchQueue.main.async {
                        self?.downloadError = "Failed to save \(filename): \(error.localizedDescription)"
                        self?.downloadingConsolidatedModelId = nil
                    }
                }
            }
            task.resume()
        }

        // Start downloading first file
        downloadNextFile(0)
    }

    // MARK: - STT Engine Management

    func loadSttSettings() {
        // Load current STT engine
        if let enginePtr = voiceflow_current_stt_engine() {
            let engineStr = String(cString: enginePtr)
            voiceflow_free_string(enginePtr)
            currentSttEngine = SttEngine(rawValue: engineStr) ?? .whisper
        }

        // Load current Moonshine model
        if let modelPtr = voiceflow_current_moonshine_model() {
            currentMoonshineModelId = String(cString: modelPtr)
            voiceflow_free_string(modelPtr)
        }

        // Load Moonshine models info
        let count = voiceflow_moonshine_model_count()
        var models: [MoonshineModel] = []
        for i in 0..<count {
            let info = voiceflow_moonshine_model_info(i)
            if let idPtr = info.id, let namePtr = info.display_name {
                let model = MoonshineModel(
                    id: String(cString: idPtr),
                    displayName: String(cString: namePtr),
                    sizeMB: info.size_mb,
                    isDownloaded: info.is_downloaded
                )
                models.append(model)
                voiceflow_free_moonshine_model_info(info)
            }
        }
        moonshineModels = models
    }

    func selectSttEngine(_ engine: SttEngine) {
        guard engine != currentSttEngine else { return }

        let success = engine.rawValue.withCString { cString in
            voiceflow_set_stt_engine(cString)
        }

        if success {
            currentSttEngine = engine
            needsRestart = true
        } else {
            downloadError = "Failed to set STT engine"
        }
    }

    func selectMoonshineModel(_ modelId: String) {
        guard modelId != currentMoonshineModelId else { return }

        let success = modelId.withCString { cString in
            voiceflow_set_moonshine_model(cString)
        }

        if success {
            currentMoonshineModelId = modelId
            needsRestart = true
        } else {
            downloadError = "Failed to set Moonshine model"
        }
    }

    func downloadMoonshineModel(_ modelId: String) {
        guard downloadingMoonshineModelId == nil else { return }

        // Moonshine models are on HuggingFace: UsefulSensors/moonshine
        // Files are in onnx/tiny/ or onnx/base/ directories
        let baseUrl = "https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/\(modelId)"
        let files = ["preprocess.onnx", "encode.onnx", "uncached_decode.onnx", "cached_decode.onnx"]
        // Tokenizer is in the separate moonshine-base/moonshine-tiny repos (same tokenizer for both)
        let tokenizerUrl = "https://huggingface.co/UsefulSensors/moonshine-\(modelId)/resolve/main/tokenizer.json"

        let modelDirName = "moonshine-\(modelId)"
        let destDir = modelsDir.appendingPathComponent(modelDirName)

        // Create model directory
        do {
            try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            downloadError = "Failed to create model directory: \(error.localizedDescription)"
            return
        }

        downloadingMoonshineModelId = modelId
        moonshineDownloadProgress = 0
        downloadError = nil

        // Download all files sequentially
        let allFiles = files + ["tokenizer.json"]
        var completedFiles = 0
        let totalFiles = allFiles.count

        func downloadNextFile(_ index: Int) {
            guard index < allFiles.count else {
                // All done
                DispatchQueue.main.async { [weak self] in
                    self?.downloadingMoonshineModelId = nil
                    self?.moonshineDownloadProgress = 1.0
                    self?.loadSttSettings() // Refresh model status
                }
                return
            }

            let filename = allFiles[index]
            let urlString = index < files.count ? "\(baseUrl)/\(filename)" : tokenizerUrl
            guard let url = URL(string: urlString) else {
                DispatchQueue.main.async { [weak self] in
                    self?.downloadError = "Invalid URL for \(filename)"
                    self?.downloadingMoonshineModelId = nil
                }
                return
            }

            let destFile = destDir.appendingPathComponent(filename)

            let task = URLSession.shared.downloadTask(with: url) { [weak self] tempURL, response, error in
                if let error = error {
                    DispatchQueue.main.async {
                        self?.downloadError = "Download failed: \(error.localizedDescription)"
                        self?.downloadingMoonshineModelId = nil
                    }
                    return
                }

                guard let tempURL = tempURL else {
                    DispatchQueue.main.async {
                        self?.downloadError = "Download failed: no file"
                        self?.downloadingMoonshineModelId = nil
                    }
                    return
                }

                do {
                    // Remove existing file if present
                    if FileManager.default.fileExists(atPath: destFile.path) {
                        try FileManager.default.removeItem(at: destFile)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: destFile)

                    completedFiles += 1
                    DispatchQueue.main.async {
                        self?.moonshineDownloadProgress = Double(completedFiles) / Double(totalFiles)
                    }

                    // Download next file
                    downloadNextFile(index + 1)
                } catch {
                    DispatchQueue.main.async {
                        self?.downloadError = "Failed to save \(filename): \(error.localizedDescription)"
                        self?.downloadingMoonshineModelId = nil
                    }
                }
            }
            task.resume()
        }

        // Start downloading first file
        downloadNextFile(0)
    }

    func loadModels() {
        // Create models directory if needed
        try? FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)

        models = Self.availableModels.map { info in
            let modelPath = modelsDir.appendingPathComponent(info.filename)
            // Check file exists AND is at least 100MB (to catch corrupt/failed downloads)
            var isDownloaded = false
            if FileManager.default.fileExists(atPath: modelPath.path) {
                if let attrs = try? FileManager.default.attributesOfItem(atPath: modelPath.path),
                   let fileSize = attrs[.size] as? Int64 {
                    isDownloaded = fileSize > 100_000_000  // 100MB minimum
                }
            }
            let downloadUrl = "https://huggingface.co/\(info.repo)/resolve/main/\(info.filename)"
            // Show combined size (model + mmproj) for Qwen3.5 multimodal models
            let totalSize = info.size + info.mmprojSize
            return LLMModel(
                id: info.id,
                displayName: info.name,
                filename: info.filename,
                sizeGB: totalSize,
                downloadUrl: downloadUrl,
                isDownloaded: isDownloaded
            )
        }
    }

    func loadCurrentModel() {
        // Use FFI to get the current model ID
        if let ptr = voiceflow_current_model() {
            let modelStr = String(cString: ptr)
            voiceflow_free_string(ptr)
            // Map Rust kebab-case IDs to our display IDs
            switch modelStr {
            case "qwen3.5-0.8b", "qwen3-5-0-8-b":
                currentModelId = "qwen3.5-0.8b"
            case "qwen3.5-2b", "qwen3-5-2-b":
                currentModelId = "qwen3.5-2b"
            case "qwen3.5-4b", "qwen3-5-4-b":
                currentModelId = "qwen3.5-4b"
            default:
                currentModelId = "qwen3.5-0.8b"
            }
        } else {
            currentModelId = "qwen3.5-0.8b" // Default
        }
    }

    func selectModel(_ modelId: String) {
        guard modelId != currentModelId else { return }
        guard let model = models.first(where: { $0.id == modelId }), model.isDownloaded else { return }

        // Use FFI function to properly save config (handles TOML format correctly)
        let success = modelId.withCString { cString in
            voiceflow_set_model(cString)
        }

        if success {
            currentModelId = modelId
            needsRestart = true
        } else {
            downloadError = "Failed to save config"
        }
    }

    func downloadModel(_ modelId: String) {
        guard !isDownloading else { return }
        guard let model = models.first(where: { $0.id == modelId }) else {
            downloadError = "Model not found"
            return
        }

        guard let url = URL(string: model.downloadUrl) else {
            downloadError = "Invalid download URL"
            return
        }

        // Ensure models directory exists
        do {
            try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            downloadError = "Failed to create models directory: \(error.localizedDescription)"
            return
        }

        let destinationURL = modelsDir.appendingPathComponent(model.filename)

        // Find mmproj info for this model
        let modelInfo = Self.availableModels.first(where: { $0.id == modelId })
        let mmprojLocalFilename = modelInfo?.mmprojFilename
        let mmprojHfFilename = modelInfo?.mmprojHfFilename
        let mmprojRepo = modelInfo?.repo

        isDownloading = true
        downloadingModelId = modelId
        downloadProgress = 0
        downloadError = nil

        let session = URLSession(configuration: .default, delegate: DownloadDelegate(progress: { [weak self] progress in
            DispatchQueue.main.async {
                // Scale progress: main model is ~70%, mmproj is ~30%
                let hasMMProj = mmprojLocalFilename != nil
                self?.downloadProgress = hasMMProj ? progress * 0.7 : progress
            }
        }, completion: { [weak self] tempURL, error in
            DispatchQueue.main.async {
                if let error = error {
                    self?.isDownloading = false
                    self?.downloadingModelId = nil
                    self?.downloadError = error.localizedDescription
                    return
                }

                guard let tempURL = tempURL else {
                    self?.isDownloading = false
                    self?.downloadingModelId = nil
                    self?.downloadError = "Download failed"
                    return
                }

                do {
                    // Move downloaded file to models directory
                    if FileManager.default.fileExists(atPath: destinationURL.path) {
                        try FileManager.default.removeItem(at: destinationURL)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: destinationURL)
                } catch {
                    self?.isDownloading = false
                    self?.downloadingModelId = nil
                    self?.downloadError = "Failed to save model: \(error.localizedDescription)"
                    return
                }

                // Download mmproj file if this model has one
                if let localFile = mmprojLocalFilename, let hfFile = mmprojHfFilename, let repo = mmprojRepo {
                    self?.downloadMmproj(repo: repo, hfFilename: hfFile, localFilename: localFile, modelId: modelId)
                } else {
                    self?.isDownloading = false
                    self?.downloadingModelId = nil
                    if let index = self?.models.firstIndex(where: { $0.id == modelId }) {
                        self?.models[index].isDownloaded = true
                    }
                }
            }
        }), delegateQueue: nil)

        downloadTask = session.downloadTask(with: url)
        downloadTask?.resume()
    }

    /// Download the mmproj file for multimodal support
    private func downloadMmproj(repo: String, hfFilename: String, localFilename: String, modelId: String) {
        let urlString = "https://huggingface.co/\(repo)/resolve/main/\(hfFilename)"
        guard let url = URL(string: urlString) else {
            isDownloading = false
            downloadingModelId = nil
            downloadError = "Invalid mmproj URL"
            return
        }

        let destURL = modelsDir.appendingPathComponent(localFilename)

        let session = URLSession(configuration: .default, delegate: DownloadDelegate(progress: { [weak self] progress in
            DispatchQueue.main.async {
                self?.downloadProgress = 0.7 + progress * 0.3
            }
        }, completion: { [weak self] tempURL, error in
            DispatchQueue.main.async {
                self?.isDownloading = false
                self?.downloadingModelId = nil

                if let error = error {
                    self?.downloadError = "mmproj download failed: \(error.localizedDescription)"
                    return
                }

                guard let tempURL = tempURL else {
                    self?.downloadError = "mmproj download failed"
                    return
                }

                do {
                    if FileManager.default.fileExists(atPath: destURL.path) {
                        try FileManager.default.removeItem(at: destURL)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: destURL)

                    if let index = self?.models.firstIndex(where: { $0.id == modelId }) {
                        self?.models[index].isDownloaded = true
                    }
                } catch {
                    self?.downloadError = "Failed to save mmproj: \(error.localizedDescription)"
                }
            }
        }), delegateQueue: nil)

        downloadTask = session.downloadTask(with: url)
        downloadTask?.resume()
    }

    func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil
        isDownloading = false
        downloadingModelId = nil
        downloadProgress = 0
    }
}

// Download delegate to track progress
class DownloadDelegate: NSObject, URLSessionDownloadDelegate {
    let progressHandler: (Double) -> Void
    let completionHandler: (URL?, Error?) -> Void

    init(progress: @escaping (Double) -> Void, completion: @escaping (URL?, Error?) -> Void) {
        self.progressHandler = progress
        self.completionHandler = completion
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // URLSession deletes the temp file when this method returns, so we must copy it immediately
        let tempDir = FileManager.default.temporaryDirectory
        let permanentTemp = tempDir.appendingPathComponent(UUID().uuidString + ".gguf")

        do {
            try FileManager.default.copyItem(at: location, to: permanentTemp)
            completionHandler(permanentTemp, nil)
        } catch {
            completionHandler(nil, error)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        progressHandler(progress)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            completionHandler(nil, error)
        }
    }
}

// MARK: - Model Settings View

struct ModelSettingsView: View {
    @EnvironmentObject var modelManager: ModelManager
    @StateObject private var setup = SetupHelper()
    @StateObject private var monitor = ServiceMonitor()

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Models").font(.headline)
                    Spacer()
                    Button {
                        monitor.refresh()
                        setup.refreshInstallState()
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                }
                Text("Parakeet TDT 0.6B v2 (STT) + Bonsai-8B Q1_0 (LLM). Both run on-device on Apple Silicon.")
                    .font(.caption).foregroundColor(.secondary)
            }
            .padding()

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    modelsCard
                    servicesCard
                    setupCard
                }
                .padding()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            setup.refreshInstallState()
            monitor.start()
        }
        .onDisappear {
            monitor.stop()
        }
    }

    private var modelsCard: some View {
        GroupBox(label: Label("Installed Models", systemImage: "cpu").fontWeight(.medium)) {
            VStack(alignment: .leading, spacing: 12) {
                ModelStatusRow(
                    icon: "waveform",
                    title: "Parakeet TDT 0.6B v2",
                    subtitle: "NVIDIA — speech-to-text via parakeet-mlx (MLX-accelerated)",
                    size: fileSizeText(SetupHelper.parakeetCachePath, fallback: "~1.2 GB"),
                    installed: setup.parakeetInstalled,
                    location: SetupHelper.parakeetCachePath.path
                )
                Divider()
                ModelStatusRow(
                    icon: "brain",
                    title: "Bonsai-8B Q1_0",
                    subtitle: "PrismML — 1.125-bit GGUF, served via local llama.cpp",
                    size: fileSizeText(SetupHelper.bonsaiPath, fallback: "~1.1 GB"),
                    installed: setup.bonsaiInstalled,
                    location: SetupHelper.bonsaiPath.path
                )
            }
            .padding(.vertical, 4)
        }
    }

    private var servicesCard: some View {
        GroupBox(label: Label("Services", systemImage: "bolt.horizontal.fill").fontWeight(.medium)) {
            VStack(alignment: .leading, spacing: 12) {
                ServiceStatusRow(service: monitor.parakeet,
                                 endpointLabel: "/tmp/voiceflow_parakeet_daemon.sock",
                                 requestLabel: "transcriptions")
                Divider()
                ServiceStatusRow(service: monitor.llamaServer,
                                 endpointLabel: monitor.llamaServer.endpoint ?? "",
                                 requestLabel: "completions")
            }
            .padding(.vertical, 4)
        }
    }

    private var setupCard: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 10) {
                    if setup.isFullyInstalled {
                        Image(systemName: "checkmark.circle.fill").foregroundColor(.green)
                        Text("Both models installed and ready.").font(.callout)
                        Spacer()
                        Button {
                            Task { await setup.runSetup() }
                        } label: {
                            Label("Re-verify", systemImage: "arrow.clockwise")
                        }
                    } else {
                        Button {
                            Task { await setup.runSetup() }
                        } label: {
                            Label(setupButtonLabel, systemImage: "arrow.down.circle.fill")
                                .frame(minWidth: 140)
                        }
                        .controlSize(.large)
                        .buttonStyle(.borderedProminent)
                        .disabled(isInProgress)
                        Spacer()
                        Text("Total download: ~2.3 GB. One-time.").font(.caption).foregroundColor(.secondary)
                    }
                }

                if setup.phase != .idle {
                    if let frac = setup.progressFraction, !setup.isFullyInstalled {
                        ProgressView(value: frac).progressViewStyle(.linear)
                    } else if isInProgress {
                        ProgressView()
                    }
                    Text(setup.phaseLabel).font(.caption).foregroundColor(.secondary)
                }
                if case .failed(let msg) = setup.phase {
                    Text(msg).font(.caption).foregroundColor(.red)
                }
            }
            .padding(.vertical, 4)
        }
    }

    private var isInProgress: Bool {
        switch setup.phase {
        case .downloadingBonsai, .loadingParakeet: return true
        default: return false
        }
    }

    private var setupButtonLabel: String {
        isInProgress ? "Installing..." : "Setup"
    }

    private func fileSizeText(_ url: URL, fallback: String) -> String {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? UInt64 else {
            return directorySizeText(url) ?? fallback
        }
        let mb = Double(size) / 1_048_576.0
        if mb >= 1024 { return String(format: "%.2f GB", mb / 1024.0) }
        return String(format: "%.0f MB", mb)
    }

    private func directorySizeText(_ url: URL) -> String? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey])
        var total: UInt64 = 0
        while let f = enumerator?.nextObject() as? URL {
            if let v = try? f.resourceValues(forKeys: [.fileSizeKey]),
               let s = v.fileSize {
                total += UInt64(s)
            }
        }
        guard total > 0 else { return nil }
        let mb = Double(total) / 1_048_576.0
        if mb >= 1024 { return String(format: "%.2f GB", mb / 1024.0) }
        return String(format: "%.0f MB", mb)
    }
}

struct ServiceStatusRow: View {
    let service: ServiceMonitor.Service
    let endpointLabel: String
    let requestLabel: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Circle()
                .fill(service.isAlive ? Color.green : Color.secondary.opacity(0.4))
                .frame(width: 10, height: 10)
                .padding(.top, 5)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(service.name).fontWeight(.medium)
                    if service.isAlive, let pid = service.pid {
                        Text("PID \(pid)")
                            .font(.caption2)
                            .padding(.horizontal, 6).padding(.vertical, 1)
                            .background(Color.secondary.opacity(0.15))
                            .clipShape(Capsule())
                    } else {
                        Text("not running")
                            .font(.caption2)
                            .padding(.horizontal, 6).padding(.vertical, 1)
                            .background(Color.secondary.opacity(0.15))
                            .clipShape(Capsule())
                    }
                    Spacer()
                }
                if !endpointLabel.isEmpty {
                    Text(endpointLabel)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                if service.isAlive {
                    HStack(spacing: 16) {
                        StatChip(label: "Uptime", value: service.uptimeText)
                        StatChip(label: "Memory", value: service.memoryText)
                        if let n = service.requestCount {
                            StatChip(label: requestLabel, value: "\(n)")
                        }
                    }
                }
            }
        }
    }
}

struct StatChip: View {
    let label: String
    let value: String
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(label).font(.caption2).foregroundColor(.secondary)
            Text(value).font(.system(size: 12, weight: .medium, design: .monospaced))
        }
    }
}


struct ModelStatusRow: View {
    let icon: String
    let title: String
    let subtitle: String
    let size: String
    let installed: Bool
    let location: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 18))
                .foregroundColor(.accentColor)
                .frame(width: 28)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(title).fontWeight(.medium)
                    if installed {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .font(.system(size: 12))
                    } else {
                        Text("not installed")
                            .font(.caption2)
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background(Color.secondary.opacity(0.2))
                            .clipShape(Capsule())
                    }
                    Spacer()
                    Text(size).font(.caption).foregroundColor(.secondary)
                }
                Text(subtitle).font(.caption).foregroundColor(.secondary)
                if installed {
                    Text(location)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
        }
    }
}


struct ModelRowView: View {
    let model: LLMModel
    let isSelected: Bool
    let isDownloading: Bool
    let downloadProgress: Double
    let onSelect: () -> Void
    let onDownload: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            // Selection indicator (radio button)
            ZStack {
                Circle()
                    .stroke(isSelected ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: 2)
                    .frame(width: 20, height: 20)

                if isSelected {
                    Circle()
                        .fill(Color.accentColor)
                        .frame(width: 12, height: 12)
                }
            }

            // Model info
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(model.displayName)
                        .fontWeight(.medium)
                    if isSelected {
                        Text("Active")
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.2))
                            .cornerRadius(4)
                    }
                }

                Text(String(format: "%.1f GB", model.sizeGB))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Download/Status indicator
            if model.isDownloaded {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            } else if isDownloading {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.7)
                    Text("\(Int(downloadProgress * 100))%")
                        .font(.caption)
                        .monospacedDigit()
                }
            } else {
                Button(action: onDownload) {
                    Label("Download", systemImage: "arrow.down.circle")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            // Allow selecting downloaded models by tapping anywhere on the row
            if model.isDownloaded && !isSelected {
                onSelect()
            }
        }
        .opacity(model.isDownloaded || isDownloading ? 1.0 : 0.7)
    }
}

// MARK: - STT Engine Row View

struct SttEngineRowView: View {
    let engine: SttEngine
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            // Selection indicator (radio button)
            ZStack {
                Circle()
                    .stroke(isSelected ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: 2)
                    .frame(width: 20, height: 20)

                if isSelected {
                    Circle()
                        .fill(Color.accentColor)
                        .frame(width: 12, height: 12)
                }
            }

            // Engine info
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(engine.displayName)
                        .fontWeight(.medium)
                    if isSelected {
                        Text("Active")
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.2))
                            .cornerRadius(4)
                    }
                }

                Text(engine.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            if !isSelected {
                onSelect()
            }
        }
    }
}

// MARK: - Moonshine Model Row View

struct MoonshineModelRowView: View {
    let model: MoonshineModel
    let isSelected: Bool
    let isDownloading: Bool
    let downloadProgress: Double
    let onSelect: () -> Void
    let onDownload: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            // Selection indicator (smaller radio button)
            ZStack {
                Circle()
                    .stroke(isSelected ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: 1.5)
                    .frame(width: 16, height: 16)

                if isSelected {
                    Circle()
                        .fill(Color.accentColor)
                        .frame(width: 10, height: 10)
                }
            }

            // Model info
            VStack(alignment: .leading, spacing: 1) {
                Text(model.displayName)
                    .font(.callout)
                    .fontWeight(isSelected ? .medium : .regular)

                Text("\(model.sizeMB) MB")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Download status / button
            if model.isDownloaded {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .font(.caption)
            } else if isDownloading {
                HStack(spacing: 6) {
                    ProgressView()
                        .scaleEffect(0.6)
                    Text("\(Int(downloadProgress * 100))%")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            } else {
                Button(action: onDownload) {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.down.circle")
                        Text("Download")
                    }
                    .font(.caption)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.mini)
            }
        }
        .padding(.vertical, 2)
        .contentShape(Rectangle())
        .onTapGesture {
            if !isSelected && model.isDownloaded {
                onSelect()
            }
        }
        .opacity(model.isDownloaded || isDownloading ? 1.0 : 0.8)
    }
}

// MARK: - AI Result State

class AIResultState: ObservableObject {
    @Published var isVisible = false
    @Published var title = ""
    @Published var resultText = ""
    @Published var pasteHint = ""

    func show(title: String, text: String, hint: String) {
        self.title = title
        self.resultText = text
        self.pasteHint = hint
        self.isVisible = true
    }

    func hide() {
        self.isVisible = false
        self.title = ""
        self.resultText = ""
        self.pasteHint = ""
    }
}

// MARK: - AI Result View

struct AIResultView: View {
    @ObservedObject var state: AIResultState

    var body: some View {
        if state.isVisible {
            VStack(alignment: .leading, spacing: 0) {
                // Header
                HStack {
                    Image(systemName: "sparkles")
                        .font(.system(size: 14))
                        .foregroundColor(.purple)
                    Text(state.title)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundColor(.white)
                    Spacer()
                    Text("✕")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundColor(.gray)
                        .padding(4)
                }
                .padding(.horizontal, 16)
                .padding(.top, 14)
                .padding(.bottom, 10)

                Divider()
                    .background(Color.white.opacity(0.15))

                // Scrollable text content
                ScrollView {
                    Text(state.resultText)
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.9))
                        .lineSpacing(4)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                }
                .frame(maxHeight: 300)

                Divider()
                    .background(Color.white.opacity(0.15))

                // Bottom bar with keyboard hints
                HStack(spacing: 16) {
                    keyHint("↩", "Paste")
                    keyHint("⌘C", "Copy")
                    keyHint("esc", "Dismiss")
                    Spacer()
                    Text(state.pasteHint)
                        .font(.system(size: 11))
                        .foregroundColor(.gray)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
            }
            .frame(width: 480)
            .frame(minHeight: 160, maxHeight: 400)
            .background(Color.black.opacity(0.85))
            .clipShape(RoundedRectangle(cornerRadius: 16))
        }
    }

    private func keyHint(_ key: String, _ label: String) -> some View {
        HStack(spacing: 4) {
            Text(key)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundColor(.white.opacity(0.8))
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Color.white.opacity(0.15))
                .cornerRadius(4)
            Text(label)
                .font(.system(size: 11))
                .foregroundColor(.gray)
        }
    }
}

// MARK: - Token Accumulator

/// Mutable class wrapper for accumulating streamed LLM tokens.
/// Used as a reference type so the @MainActor onToken closure and the caller share state.
@MainActor
private final class TokenAccumulator {
    var text: String = ""
}

// MARK: - Overlay State

class OverlayState: ObservableObject {
    enum RecordingState {
        case idle
        case recording
        case streaming(String)      // partial transcript text from real-time STT
        case processing
        case formatting(String)     // LLM tokens appearing word-by-word
        case aiProcessing(String)   // label text like "Summarizing..."

        var isRecording: Bool {
            if case .recording = self { return true }
            if case .streaming = self { return true }
            return false
        }

        var isProcessing: Bool {
            if case .processing = self { return true }
            if case .formatting = self { return true }
            return false
        }

        var isStreaming: Bool {
            if case .streaming = self { return true }
            return false
        }
    }

    @Published var state: RecordingState = .idle
    @Published var audioLevel: Float = 0
}

// MARK: - Recording Overlay View

struct RecordingOverlayView: View {
    @ObservedObject var state: OverlayState
    @State private var animateGradient = false

    // Brand colors from logo
    private let gradientColors: [Color] = [
        Color(red: 0.0, green: 0.85, blue: 0.85),   // Cyan
        Color(red: 0.2, green: 0.6, blue: 0.9),    // Blue
        Color(red: 0.4, green: 0.8, blue: 0.7),    // Teal
        Color(red: 0.0, green: 0.7, blue: 0.9),    // Light blue
        Color(red: 0.0, green: 0.85, blue: 0.85),   // Cyan (loop)
    ]

    private let cardWidth: CGFloat = 420
    private let pillHeight: CGFloat = 66
    private let statusBarHeight: CGFloat = 50
    private let maxTextAreaHeight: CGFloat = 300

    // MARK: - Mode Detection

    private var isCardMode: Bool {
        switch state.state {
        case .streaming(let t) where !t.isEmpty: return true
        case .formatting: return true  // Always card mode — keeps card visible during STT→formatting transition
        default: return false
        }
    }

    private var currentText: String {
        switch state.state {
        case .streaming(let t): return t
        case .formatting(let t): return t
        default: return ""
        }
    }

    private var isFormattingPhase: Bool {
        if case .formatting = state.state { return true }
        return false
    }

    private var pillWidth: CGFloat {
        if case .aiProcessing = state.state { return 260 }
        return 200
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 0)

            if isCardMode {
                cardView
                    .transition(.opacity.combined(with: .scale(scale: 0.95, anchor: .bottom)))
            } else {
                pillView
                    .transition(.opacity.combined(with: .scale(scale: 0.95, anchor: .bottom)))
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isCardMode)
        .onAppear {
            withAnimation(.linear(duration: 3.0).repeatForever(autoreverses: false)) {
                animateGradient = true
            }
        }
    }

    // MARK: - Card View

    private var cardView: some View {
        VStack(spacing: 0) {
            if !currentText.isEmpty {
                textArea

                Rectangle()
                    .fill(Color.white.opacity(0.15))
                    .frame(height: 1)
            }

            statusBar
        }
        .frame(width: cardWidth)
        .background(cardBackground)
    }

    private var textArea: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical, showsIndicators: false) {
                Text(currentText)
                    .font(.system(size: 14, weight: .regular))
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .id("textBottom")
            }
            .onChange(of: currentText) { _ in
                withAnimation(.easeOut(duration: 0.1)) {
                    proxy.scrollTo("textBottom", anchor: .bottom)
                }
            }
        }
        .frame(maxHeight: maxTextAreaHeight)
    }

    private var statusBar: some View {
        HStack(spacing: 10) {
            if isFormattingPhase {
                Image(systemName: "sparkles")
                    .font(.system(size: 16))
                    .foregroundColor(.white)
                Text("Formatting...")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white.opacity(0.7))
                Spacer()
                PulsingDotsView()
                    .frame(width: 40, height: 16)
                    .scaleEffect(0.6)
            } else {
                logoView
                WaveformView(audioLevel: state.audioLevel)
                    .frame(width: 60, height: 30)
                Spacer()
            }
        }
        .padding(.horizontal, 16)
        .frame(height: statusBarHeight)
    }

    private var cardBackground: some View {
        ZStack {
            if !isFormattingPhase {
                AnimatedGradientBackground(animate: animateGradient, colors: gradientColors)
            } else {
                Color.black.opacity(0.85)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Pill View

    private var pillView: some View {
        HStack(spacing: 10) {
            if case .aiProcessing(let label) = state.state {
                Image(systemName: "sparkles")
                    .font(.system(size: 18))
                    .foregroundColor(.white)
                Text(label)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white)
                PulsingDotsView()
                    .frame(width: 50, height: 20)
                    .scaleEffect(0.7)
            } else if case .streaming = state.state {
                // Streaming with empty text — show waveform in pill
                logoView
                WaveformView(audioLevel: state.audioLevel)
                    .frame(width: 80, height: 45)
            } else {
                logoView
                if state.state.isRecording {
                    WaveformView(audioLevel: state.audioLevel)
                        .frame(width: 80, height: 45)
                } else if state.state.isProcessing {
                    processingView
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .frame(width: pillWidth, height: pillHeight)
        .background(pillBackground)
        .animation(.easeInOut(duration: 0.2), value: pillWidth)
    }

    // MARK: - Shared Sub-views

    @ViewBuilder
    private var logoView: some View {
        if let iconPath = Bundle.main.path(forResource: "MenuBarIcon", ofType: "png"),
           let nsImage = NSImage(contentsOfFile: iconPath) {
            Image(nsImage: nsImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 28, height: 28)
                .opacity(state.state.isRecording ? 1.0 : 0.7)
        } else {
            Image(systemName: "mic.fill")
                .font(.system(size: 24))
                .foregroundColor(.white)
        }
    }

    private var processingView: some View {
        PulsingDotsView()
            .frame(width: 80, height: 30)
    }

    private var pillBackground: some View {
        ZStack {
            if state.state.isRecording {
                AnimatedGradientBackground(animate: animateGradient, colors: gradientColors)
            } else {
                Color.black.opacity(0.85)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 30))
    }
}

// MARK: - Pulsing Dots View

struct PulsingDotsView: View {
    @State private var activeIndex = 0
    private let dotCount = 3
    private let dotSize: CGFloat = 8
    private let timer = Timer.publish(every: 0.35, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 8) {
            ForEach(0..<dotCount, id: \.self) { index in
                Circle()
                    .fill(Color.white)
                    .frame(width: dotSize, height: dotSize)
                    .opacity(activeIndex == index ? 1.0 : 0.3)
                    .scaleEffect(activeIndex == index ? 1.2 : 0.8)
                    .animation(.easeInOut(duration: 0.3), value: activeIndex)
            }
        }
        .onReceive(timer) { _ in
            activeIndex = (activeIndex + 1) % dotCount
        }
    }
}

// MARK: - Animated Gradient Background

struct AnimatedGradientBackground: View {
    let animate: Bool
    let colors: [Color]

    var body: some View {
        LinearGradient(
            colors: colors,
            startPoint: animate ? .topLeading : .bottomTrailing,
            endPoint: animate ? .bottomTrailing : .topLeading
        )
        .hueRotation(.degrees(animate ? 45 : 0))
        .opacity(0.95)
    }
}

// MARK: - Waveform View

struct WaveformView: View {
    let audioLevel: Float

    private let barCount = 5
    private let barSpacing: CGFloat = 4

    var body: some View {
        HStack(spacing: barSpacing) {
            ForEach(0..<barCount, id: \.self) { index in
                WaveformBar(audioLevel: audioLevel, index: index)
            }
        }
    }
}

struct WaveformBar: View {
    let audioLevel: Float
    let index: Int

    private let baseHeight: CGFloat = 6
    private let maxHeight: CGFloat = 40  // Much bigger amplitude

    private var computedHeight: CGFloat {
        // Amplify the audio level for more visible movement
        let amplifiedLevel = min(1.0, CGFloat(audioLevel) * 3.0)
        let variation = sin(Double(index) * 1.5) * 0.4 + 0.6
        let level = amplifiedLevel * CGFloat(variation)
        return baseHeight + level * (maxHeight - baseHeight)
    }

    var body: some View {
        RoundedRectangle(cornerRadius: 3)
            .fill(Color.white)
            .frame(width: 6, height: computedHeight)
            .animation(.easeInOut(duration: 0.08), value: audioLevel)
    }
}
