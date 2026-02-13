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
    let firstSeenDate: Date
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

    /// Returns the custom prompt or the category default for the given app
    func promptForApp(bundleId: String) -> String {
        guard let profile = profiles.first(where: { $0.id == bundleId }) else {
            return ""
        }

        // Custom prompt takes priority
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
    private let maxAgeDays = 30

    init() {
        loadPatterns()
    }

    // MARK: - Storage

    func loadPatterns() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([CorrectionPattern].self, from: data) {
            let cutoff = Calendar.current.date(byAdding: .day, value: -maxAgeDays, to: Date()) ?? Date.distantPast
            patterns = decoded.filter { $0.timestamp >= cutoff }
        }
    }

    func savePatterns() {
        if let encoded = try? JSONEncoder().encode(patterns) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
    }

    // MARK: - Detection

    /// Detect word-level corrections between original pasted text and current text
    func detectCorrections(original: String, current: String, targetApp: String) {
        let origWords = original.split(separator: " ").map(String.init)
        let currWords = current.split(separator: " ").map(String.init)

        // Simple word-level comparison: walk both arrays looking for differences
        let minCount = min(origWords.count, currWords.count)
        var newPatterns: [CorrectionPattern] = []

        for i in 0..<minCount {
            let origWord = origWords[i]
            let currWord = currWords[i]

            // Skip if identical
            guard origWord != currWord else { continue }

            // Only learn if it's a case change or small edit distance (≤2)
            let isCaseChange = origWord.lowercased() == currWord.lowercased()
            let distance = levenshtein(origWord.lowercased(), currWord.lowercased())

            if isCaseChange || distance <= 2 {
                // Check for duplicate
                let isDuplicate = patterns.contains {
                    $0.original.lowercased() == origWord.lowercased() &&
                    $0.corrected == currWord
                }

                if !isDuplicate {
                    let pattern = CorrectionPattern(
                        id: UUID(),
                        original: origWord,
                        corrected: currWord,
                        timestamp: Date(),
                        targetApp: targetApp
                    )
                    newPatterns.append(pattern)
                }
            }
        }

        if !newPatterns.isEmpty {
            patterns.append(contentsOf: newPatterns)
            // Enforce max limit (keep newest)
            if patterns.count > maxPatterns {
                patterns = Array(patterns.suffix(maxPatterns))
            }
            savePatterns()
        }
    }

    // MARK: - Delete & Clear

    func deletePattern(id: UUID) {
        patterns.removeAll { $0.id == id }
        savePatterns()
    }

    func clearAll() {
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

    // MARK: - Levenshtein Distance

    private func levenshtein(_ a: String, _ b: String) -> Int {
        let aChars = Array(a)
        let bChars = Array(b)
        let aLen = aChars.count
        let bLen = bChars.count

        if aLen == 0 { return bLen }
        if bLen == 0 { return aLen }

        var prev = Array(0...bLen)
        var curr = [Int](repeating: 0, count: bLen + 1)

        for i in 1...aLen {
            curr[0] = i
            for j in 1...bLen {
                let cost = aChars[i - 1] == bChars[j - 1] ? 0 : 1
                curr[j] = min(
                    prev[j] + 1,      // deletion
                    curr[j - 1] + 1,  // insertion
                    prev[j - 1] + cost // substitution
                )
            }
            prev = curr
        }

        return prev[bLen]
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
    case aggressive = "aggressive"

    var displayName: String {
        switch self {
        case .minimal: return "Minimal"
        case .moderate: return "Moderate"
        case .aggressive: return "Aggressive"
        }
    }

    var description: String {
        switch self {
        case .minimal: return "Light cleanup, preserves original speech"
        case .moderate: return "Fix grammar, punctuation, filler words"
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
    private var pendingVisualContext: Task<String?, Never>?

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

    // User preference for visual context
    var visualContextEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: "visualContextEnabled") }
        set { UserDefaults.standard.set(newValue, forKey: "visualContextEnabled") }
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

        let destPath = (applicationsDir as NSString).appendingPathComponent(
            (appPath as NSString).lastPathComponent
        )

        do {
            let fm = FileManager.default
            // Remove existing copy if present
            if fm.fileExists(atPath: destPath) {
                try fm.removeItem(atPath: destPath)
            }
            try fm.copyItem(atPath: appPath, toPath: destPath)

            // Relaunch from new location
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
            process.arguments = ["-n", destPath]
            try process.run()

            NSApp.terminate(nil)
        } catch {
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

        // Request microphone permission
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            if !granted {
                DispatchQueue.main.async {
                    self.showAlert(title: "Microphone Access Required",
                                   message: "VoiceFlow needs microphone access to transcribe your speech.")
                }
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        // Clean up AI result key monitors
        removeAIResultKeyMonitors()

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
        // Show Apple's native microphone permission prompt
        AVCaptureDevice.requestAccess(for: .audio) { _ in }
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
        // Create floating panel
        let panelWidth: CGFloat = 200
        let panelHeight: CGFloat = 70

        // Get main screen
        guard let screen = NSScreen.main else { return }

        // Position at bottom center of screen, above the dock
        let screenFrame = screen.visibleFrame
        let panelX = screenFrame.midX - panelWidth / 2
        let panelY = screenFrame.minY + 80  // 80pt above dock area

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
        // Resize panel for AI processing (wider to fit text)
        if case .aiProcessing = state {
            overlayPanel?.setContentSize(NSSize(width: 260, height: 70))
        } else {
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
        // Option + Space: hold to record, release to stop and paste
        hotkeyManager.register(
            keyCode: UInt32(kVK_Space),
            modifiers: UInt32(optionKey),
            onPress: { [weak self] in
                DispatchQueue.main.async {
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

    /// Determine if a transcript is complex enough to benefit from visual context (VLM).
    /// Short or trivial transcripts don't need screen context — saves VLM + LLM prompt overhead.
    private func needsVisualContext(transcript: String) -> Bool {
        let words = transcript.split(separator: " ")

        // Rule 1: Very short transcripts (< 5 words) rarely need visual context
        guard words.count >= 5 else { return false }

        // Rule 2: If transcript contains only common/stop words, skip
        // (e.g., "yes I think so too" doesn't need screen context)
        let commonWords: Set<String> = [
            "yes", "no", "ok", "okay", "sure", "thanks", "thank", "you",
            "please", "hello", "hi", "hey", "bye", "goodbye", "right",
            "i", "me", "my", "we", "the", "a", "an", "is", "are", "was",
            "it", "that", "this", "so", "and", "but", "or", "not", "just",
            "do", "did", "have", "has", "had", "will", "would", "can",
            "could", "should", "think", "know", "like", "go", "get",
            "see", "look", "make", "want", "need", "let", "say", "said",
            "well", "also", "too", "very", "really", "actually", "maybe",
            "here", "there", "now", "then", "when", "what", "how", "why",
            "who", "where", "which", "all", "some", "any", "much", "many",
            "good", "great", "fine", "new", "old", "big", "little", "more",
            "one", "two", "three", "four", "five", "first", "last", "next",
            "up", "down", "in", "on", "at", "to", "for", "of", "with",
            "from", "by", "about", "into", "over", "after", "before",
            "been", "being", "going", "come", "back", "out", "still",
            "if", "them", "they", "their", "he", "she", "his", "her",
            "its", "our", "your", "be", "am", "were", "done", "got",
        ]
        let hasUncommonWord = words.contains { word in
            !commonWords.contains(word.lowercased().trimmingCharacters(in: .punctuationCharacters))
        }

        return hasUncommonWord
    }

    private func startRecording() {
        guard !isRecording else { return }

        do {
            try audioRecorder.startRecording()
            isRecording = true
            recordingMenuItem?.title = "Recording... (release ⌥ Space)"

            // Show overlay
            showOverlay(state: .recording)

            // Subscribe to audio level changes
            audioLevelCancellable = audioRecorder.$audioLevel
                .receive(on: DispatchQueue.main)
                .sink { [weak self] level in
                    self?.overlayState.audioLevel = level
                }

            // Start visual context capture in parallel with recording
            let hasVlm: Bool = {
                if let ptr = voiceflow_current_vlm_model() {
                    voiceflow_free_string(ptr)
                    return true
                }
                return false
            }()
            if visualContextEnabled && hasVlm {
                if !ScreenshotCapture.hasPermission() {
                    // Show one-time alert directing user to grant Screen Recording
                    let alert = NSAlert()
                    alert.messageText = "Screen Recording Permission Required"
                    alert.informativeText = "Visual Context is enabled but VoiceFlow doesn't have Screen Recording permission. Please go to System Settings > Privacy & Security > Screen Recording and enable VoiceFlow, or disable Visual Context in Settings."
                    alert.alertStyle = .informational
                    alert.addButton(withTitle: "Open System Settings")
                    alert.addButton(withTitle: "Continue Without")
                    let response = alert.runModal()
                    if response == .alertFirstButtonReturn {
                        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                } else {
                    let capture = screenshotCapture
                    let engine = voiceFlow.asrEngine
                    pendingVisualContext = Task {
                        do {
                            let imageData = try await capture.captureActiveWindow()
                            return await engine?.analyzeImage(imageData: imageData)
                        } catch {
                            NSLog("[VoiceFlow] Visual context capture failed: \(error)")
                            return nil
                        }
                    }
                }
            }
        } catch {
            showAlert(title: "Recording Error", message: error.localizedDescription)
        }
    }

    private func stopRecordingAndPaste() {
        guard isRecording else { return }

        // Cancel audio level subscription
        audioLevelCancellable?.cancel()
        audioLevelCancellable = nil

        let audio = audioRecorder.stopRecording()
        isRecording = false
        recordingMenuItem?.title = "Hold ⌥ Space to Record"

        guard !audio.isEmpty else {
            hideOverlay()
            pendingVisualContext?.cancel()
            pendingVisualContext = nil
            showNotification(title: "No Audio", body: "No audio was captured")
            return
        }

        // Show processing state in overlay
        showOverlay(state: .processing)

        // Detect the application context via AppProfileManager (auto-creates on first encounter)
        let frontmostApp = NSWorkspace.shared.frontmostApplication
        let _ = frontmostApp.flatMap { AppProfileManager.shared.ensureProfile(for: $0) }
        let appPrompt = frontmostApp?.bundleIdentifier.flatMap {
            AppProfileManager.shared.promptForApp(bundleId: $0)
        } ?? ""
        let frontApp = frontmostApp?.localizedName ?? "Unknown"

        // Read existing text from the focused input field (must be on main thread before async)
        let inputFieldText = CursorContext.getTextBeforeCursor(maxLength: 500)

        let visualContextTask = pendingVisualContext
        pendingVisualContext = nil
        let baseContext = formattingLevel.systemPrompt + appPrompt
        let currentSpacingMode = spacingMode

        // Gate visual context based on recording duration (audio is 16kHz mono)
        let recordingDurationSecs = Double(audio.count) / 16000.0
        let skipVisualContext = recordingDurationSecs < 1.5
        if skipVisualContext {
            NSLog("[VoiceFlow] Short recording (%.1fs) — skipping visual context", recordingDurationSecs)
            visualContextTask?.cancel()
        }

        Task {
            // Await pending visual context (with 5s timeout) — only if recording was long enough
            let visualDescription: String? = await {
                guard !skipVisualContext, let task = visualContextTask else { return nil }
                return await withTaskGroup(of: String?.self) { group in
                    group.addTask { await task.value }
                    group.addTask {
                        try? await Task.sleep(nanoseconds: 5_000_000_000)
                        return nil
                    }
                    let result = await group.next() ?? nil
                    group.cancelAll()
                    return result
                }
            }()

            // Combine formatting level prompt + app context + visual context + input field context
            var context = baseContext
            if let visualDescription = visualDescription, !visualDescription.isEmpty {
                context += """

                [VISUAL CONTEXT — HIGH PRIORITY]
                The following was extracted from the user's active screen:
                \(visualDescription)
                CRITICAL: When the speaker says a word that sounds like a term visible on screen, \
                ALWAYS use the on-screen spelling. Screen text is ground truth — it overrides phonetic guesses. \
                Examples: if screen shows "era-code" and speaker says something like "era core", output "Era Code". \
                If screen shows "OAuth" and speaker says "oh auth", output "OAuth". \
                Use NAMES and TERMS for proper noun spelling. Use CONTEXT and NEARBY_TEXT to match tone and style.
                """
            }
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

            if let result = await voiceFlow.process(audio: audio, context: context) {
                // Log visual context decision for tuning heuristics
                let wouldNeedVLM = self.needsVisualContext(transcript: result.rawTranscript)
                if !skipVisualContext && !wouldNeedVLM {
                    NSLog("[VoiceFlow] VLM ran but transcript didn't need it: \"%@\" (%d words)",
                          result.rawTranscript,
                          result.rawTranscript.split(separator: " ").count)
                }

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
                        handleAICommand(aiCommand, targetApp: frontmostApp, visualDescription: visualDescription)
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

                // Apply spacing mode to the expanded text
                var spacedText = currentSpacingMode.apply(to: expandedText)

                // Add trailing space only when the next dictation can't determine its own leading space.
                // In contextAware mode with working AX, the next dictation reads the cursor and adds
                // a leading space itself — no trailing space needed. Without AX, add trailing as safety net.
                if !spacedText.isEmpty && !spacedText.hasSuffix(" ") && !spacedText.hasSuffix("\n") {
                    switch currentSpacingMode {
                    case .contextAware:
                        if case .unavailable = CursorContext.getCharacterBeforeCursor() {
                            spacedText += " "
                        }
                    case .smart, .always, .trailing:
                        spacedText += " "
                    }
                }

                // Log the transcription
                let currentModel: String = {
                    if let vlmPtr = voiceflow_current_vlm_model() {
                        let s = String(cString: vlmPtr)
                        voiceflow_free_string(vlmPtr)
                        return s
                    }
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

                // Post-paste correction detection: read back text after delay to detect user edits
                let pastedText = spacedText.trimmingCharacters(in: .whitespaces)
                let pastedLength = pastedText.count
                let correctionApp = frontApp
                Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 3_000_000_000)  // 3s delay
                    if let currentText = CursorContext.getTextBeforeCursor(maxLength: pastedLength + 50) {
                        let region = String(currentText.suffix(pastedLength + 10))
                        if region != pastedText && !region.isEmpty && region.count >= pastedText.count / 2 {
                            CorrectionManager.shared.detectCorrections(
                                original: pastedText, current: region, targetApp: correctionApp
                            )
                        }
                    }
                }
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
    private var hotKeyRef: EventHotKeyRef?

    private static var onPressHandler: (() -> Void)?
    private static var onReleaseHandler: (() -> Void)?

    func register(keyCode: UInt32, modifiers: UInt32, onPress: @escaping () -> Void, onRelease: @escaping () -> Void) {
        GlobalHotkeyManager.onPressHandler = onPress
        GlobalHotkeyManager.onReleaseHandler = onRelease

        var hotKeyID = EventHotKeyID()
        hotKeyID.signature = OSType(0x564643) // "VFC"
        hotKeyID.id = 1

        // Register for both press and release events
        var eventTypes = [
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))
        ]

        // Install event handler
        InstallEventHandler(
            GetApplicationEventTarget(),
            { (_, event, _) -> OSStatus in
                var eventKind: UInt32 = 0
                GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeUInt32), nil, MemoryLayout<UInt32>.size, nil, &eventKind)

                let kind = GetEventKind(event)
                if kind == UInt32(kEventHotKeyPressed) {
                    GlobalHotkeyManager.onPressHandler?()
                } else if kind == UInt32(kEventHotKeyReleased) {
                    GlobalHotkeyManager.onReleaseHandler?()
                }
                return noErr
            },
            2,
            &eventTypes,
            nil,
            nil
        )

        // Register hotkey
        RegisterEventHotKey(
            keyCode,
            modifiers,
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &hotKeyRef
        )
    }

    deinit {
        if let hotKeyRef = hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
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

    var body: some View {
        Group {
            switch authStatus {
            case .authorized:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            case .notDetermined:
                Button("Grant") {
                    AVCaptureDevice.requestAccess(for: .audio) { granted in
                        DispatchQueue.main.async {
                            authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
                        }
                    }
                }
            case .denied, .restricted:
                Button("Open Settings") {
                    if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                        NSWorkspace.shared.open(url)
                    }
                }
            @unknown default:
                Button("Grant") {
                    AVCaptureDevice.requestAccess(for: .audio) { _ in }
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
    case aiFeatures = "AI Features"
    case settings = "Settings"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .home: return "house"
        case .models: return "cpu"
        case .snippets: return "scissors"
        case .style: return "textformat"
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
    @AppStorage("visualContextEnabled") private var visualContextEnabled = false
    @AppStorage("appProfilesEnabled") private var appProfilesEnabled = true
    @ObservedObject private var profileManager = AppProfileManager.shared
    @ObservedObject private var correctionManager = CorrectionManager.shared
    @State private var editingProfile: AppProfile? = nil

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
                        Toggle(isOn: $visualContextEnabled) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Screen Capture")
                                    .font(.system(size: 13, weight: .medium))
                                Text("Extract names and terms from your active window to improve spelling.")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .toggleStyle(.switch)
                        .disabled(!hasVlmModel)
                        .onChange(of: visualContextEnabled) { newValue in
                            if newValue && !ScreenshotCapture.hasPermission() {
                                ScreenshotCapture.requestPermission()
                            }
                        }

                        if !hasVlmModel {
                            HStack(spacing: 6) {
                                Image(systemName: "info.circle")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                                Text("Download a VLM model in the Models tab to enable.")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                            }
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.orange.opacity(0.08))
                            .cornerRadius(6)
                        } else if visualContextEnabled && !ScreenshotCapture.hasPermission() {
                            HStack(spacing: 6) {
                                Image(systemName: "exclamationmark.triangle")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                                Text("Screen Recording permission required.")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                            }
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.orange.opacity(0.08))
                            .cornerRadius(6)
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
                        Text("Learns from your edits to fix recurring mistakes automatically.")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        if !correctionManager.patterns.isEmpty {
                            Divider()

                            HStack {
                                Text("\(correctionManager.patterns.count) correction\(correctionManager.patterns.count == 1 ? "" : "s") learned")
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
                                Text("No corrections learned yet. Edit text after dictating and VoiceFlow will learn your preferences.")
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

    private var hasVlmModel: Bool {
        // Check if any VLM model is downloaded (not just selected)
        let count = voiceflow_vlm_model_count()
        for i in 0..<count {
            let info = voiceflow_vlm_model_info(UInt(i))
            let downloaded = info.is_downloaded
            voiceflow_free_vlm_model_info(info)
            if downloaded { return true }
        }
        return false
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

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("App Profiles")
                    .font(.headline)
                Spacer()
            }

            Text("Apps are auto-detected when you dictate. Customize formatting per app.")
                .font(.caption)
                .foregroundColor(.secondary)

            if profileManager.profiles.isEmpty {
                VStack(spacing: 8) {
                    Spacer()
                    Image(systemName: "app.badge")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text("No apps detected yet")
                        .foregroundColor(.secondary)
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
                .listStyle(.bordered)
            }
        }
        .padding()
        .sheet(item: $editingProfile) { profile in
            AppProfileEditView(profile: profile) { updated in
                profileManager.updateProfile(updated)
            }
        }
    }
}

struct AppProfileRowView: View {
    let profile: AppProfile

    private var categoryColor: Color {
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
                Text(profile.category.capitalized)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(categoryColor.opacity(0.2))
                    .foregroundColor(categoryColor)
                    .clipShape(Capsule())
            }
            if let custom = profile.customPrompt, !custom.isEmpty {
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
    @State private var category: String = "default"
    @State private var customPrompt: String = ""

    private let categories = ["default", "email", "slack", "code"]

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Edit \(profile.displayName)")
                .font(.headline)

            Text("Bundle ID: \(profile.id)")
                .font(.caption)
                .foregroundColor(.secondary)

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
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Custom Prompt (optional)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("Overrides the category default. Leave empty to use category formatting.")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                TextEditor(text: $customPrompt)
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
                    var updated = profile
                    updated.category = category
                    updated.customPrompt = customPrompt.isEmpty ? nil : customPrompt
                    onSave(updated)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(width: 400, height: 320)
        .onAppear {
            category = profile.category
            customPrompt = profile.customPrompt ?? ""
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

    var displayName: String {
        switch self {
        case .sttPlusLlm: return "STT + LLM (traditional)"
        case .consolidated: return "Consolidated (single model)"
        }
    }

    var description: String {
        switch self {
        case .sttPlusLlm: return "Separate speech-to-text and language model stages"
        case .consolidated: return "Single model handles audio-to-text (Qwen3-ASR via MLX)"
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

/// VLM (Vision-Language Model) info (mirrors Rust VlmModel)
struct VlmModelItem: Identifiable {
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

    // VLM (Vision-Language Model) settings
    @Published var currentVlmModelId: String? = nil
    @Published var vlmModels: [VlmModelItem] = []
    @Published var downloadingVlmModelId: String?
    @Published var vlmDownloadProgress: Double = 0

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
    private static let availableModels: [(id: String, name: String, filename: String, size: Float, repo: String)] = [
        ("qwen3-1.7b", "Qwen3 1.7B", "qwen3-1.7b-q4_k_m.gguf", 1.1, "Qwen/Qwen3-1.7B-GGUF"),
        ("qwen3-4b", "Qwen3 4B", "Qwen3-4B-Q4_K_M.gguf", 2.5, "Qwen/Qwen3-4B-GGUF"),
        ("smollm3-3b", "SmolLM3 3B", "SmolLM3-Q4_K_M.gguf", 1.92, "ggml-org/SmolLM3-3B-GGUF"),
        ("gemma2-2b", "Gemma 2 2B", "gemma-2-2b-it-Q4_K_M.gguf", 1.71, "bartowski/gemma-2-2b-it-GGUF"),
    ]

    init() {
        loadModels()
        loadCurrentModel()
        loadSttSettings()
        loadConsolidatedSettings()
        loadVlmSettings()
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

    // MARK: - VLM (Vision-Language Model) Management

    func loadVlmSettings() {
        // Load current VLM model selection
        if let vlmPtr = voiceflow_current_vlm_model() {
            currentVlmModelId = String(cString: vlmPtr)
            voiceflow_free_string(vlmPtr)
        } else {
            currentVlmModelId = nil
        }

        let count = voiceflow_vlm_model_count()
        var items: [VlmModelItem] = []
        for i in 0..<count {
            let info = voiceflow_vlm_model_info(i)
            if let idPtr = info.id, let namePtr = info.display_name, let dirPtr = info.dir_name {
                let item = VlmModelItem(
                    id: String(cString: idPtr),
                    displayName: String(cString: namePtr),
                    dirName: String(cString: dirPtr),
                    sizeGB: info.size_gb,
                    isDownloaded: info.is_downloaded
                )
                items.append(item)
            }
            voiceflow_free_vlm_model_info(info)
        }
        vlmModels = items
    }

    func selectVlmModel(_ modelId: String) {
        guard modelId != currentVlmModelId else { return }
        guard let model = vlmModels.first(where: { $0.id == modelId }), model.isDownloaded else { return }

        let success = modelId.withCString { cString in
            voiceflow_set_vlm_model(cString)
        }

        if success {
            currentVlmModelId = modelId
            needsRestart = true
        } else {
            downloadError = "Failed to set VLM model"
        }
    }

    func deselectVlmModel() {
        guard currentVlmModelId != nil else { return }

        let success = voiceflow_set_vlm_model(nil)

        if success {
            currentVlmModelId = nil
            needsRestart = true
        } else {
            downloadError = "Failed to clear VLM model"
        }
    }

    func downloadVlmModel(_ modelId: String) {
        guard downloadingVlmModelId == nil else { return }
        guard let model = vlmModels.first(where: { $0.id == modelId }) else {
            downloadError = "VLM model not found"
            return
        }

        // Get HF repo from FFI
        guard let repoPtr = modelId.withCString({ voiceflow_vlm_model_hf_repo($0) }) else {
            downloadError = "No HuggingFace repo for VLM model \(modelId)"
            return
        }
        let hfRepo = String(cString: repoPtr)
        voiceflow_free_string(repoPtr)

        // Get required files from FFI
        let fileCount = modelId.withCString { voiceflow_vlm_model_file_count($0) }
        var files: [String] = []
        for i in 0..<fileCount {
            if let namePtr = modelId.withCString({ voiceflow_vlm_model_file_name($0, i) }) {
                files.append(String(cString: namePtr))
                voiceflow_free_string(namePtr)
            }
        }

        guard !files.isEmpty else {
            downloadError = "No file list for VLM model \(modelId)"
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

        downloadingVlmModelId = modelId
        vlmDownloadProgress = 0
        downloadError = nil

        var completedFiles = 0
        let totalFiles = files.count

        func downloadNextFile(_ index: Int) {
            guard index < files.count else {
                // All done
                DispatchQueue.main.async { [weak self] in
                    self?.downloadingVlmModelId = nil
                    self?.vlmDownloadProgress = 1.0
                    self?.loadVlmSettings() // Refresh model status
                }
                return
            }

            let filename = files[index]
            let urlString = "https://huggingface.co/\(hfRepo)/resolve/main/\(filename)"
            guard let url = URL(string: urlString) else {
                DispatchQueue.main.async { [weak self] in
                    self?.downloadError = "Invalid URL for \(filename)"
                    self?.downloadingVlmModelId = nil
                }
                return
            }

            let destFile = destDir.appendingPathComponent(filename)

            let task = URLSession.shared.downloadTask(with: url) { [weak self] tempURL, response, error in
                if let error = error {
                    DispatchQueue.main.async {
                        self?.downloadError = "Download failed: \(error.localizedDescription)"
                        self?.downloadingVlmModelId = nil
                    }
                    return
                }

                guard let tempURL = tempURL else {
                    DispatchQueue.main.async {
                        self?.downloadError = "Download failed: no file"
                        self?.downloadingVlmModelId = nil
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
                        self?.vlmDownloadProgress = Double(completedFiles) / Double(totalFiles)
                    }

                    // Download next file
                    downloadNextFile(index + 1)
                } catch {
                    DispatchQueue.main.async {
                        self?.downloadError = "Failed to save \(filename): \(error.localizedDescription)"
                        self?.downloadingVlmModelId = nil
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
            return LLMModel(
                id: info.id,
                displayName: info.name,
                filename: info.filename,
                sizeGB: info.size,
                downloadUrl: downloadUrl,
                isDownloaded: isDownloaded
            )
        }
    }

    func loadCurrentModel() {
        // Try to read current model from config file
        guard let configData = try? String(contentsOf: configPath, encoding: .utf8) else {
            currentModelId = "qwen3-1.7b" // Default
            return
        }

        // Simple TOML parsing for llm_model
        // Rust uses kebab-case: qwen3-4-b, qwen3-1-7-b, smol-lm3-3-b, gemma2-2-b
        if configData.contains("qwen3-4-b") {
            currentModelId = "qwen3-4b"
        } else if configData.contains("smol-lm3-3-b") {
            currentModelId = "smollm3-3b"
        } else if configData.contains("gemma2-2-b") {
            currentModelId = "gemma2-2b"
        } else {
            currentModelId = "qwen3-1.7b"
        }
    }

    func selectModel(_ modelId: String) {
        guard modelId != currentModelId || currentVlmModelId != nil else { return }
        guard let model = models.first(where: { $0.id == modelId }), model.isDownloaded else { return }

        // Use FFI function to properly save config (handles TOML format correctly)
        let success = modelId.withCString { cString in
            voiceflow_set_model(cString)
        }

        if success {
            currentModelId = modelId
            // Clear VLM selection — only one can be active
            if currentVlmModelId != nil {
                voiceflow_set_vlm_model(nil)
                currentVlmModelId = nil
            }
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

        isDownloading = true
        downloadingModelId = modelId
        downloadProgress = 0
        downloadError = nil

        let session = URLSession(configuration: .default, delegate: DownloadDelegate(progress: { [weak self] progress in
            DispatchQueue.main.async {
                self?.downloadProgress = progress
            }
        }, completion: { [weak self] tempURL, error in
            DispatchQueue.main.async {
                self?.isDownloading = false
                self?.downloadingModelId = nil

                if let error = error {
                    self?.downloadError = error.localizedDescription
                    return
                }

                guard let tempURL = tempURL else {
                    self?.downloadError = "Download failed"
                    return
                }

                do {
                    // Move downloaded file to models directory
                    if FileManager.default.fileExists(atPath: destinationURL.path) {
                        try FileManager.default.removeItem(at: destinationURL)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: destinationURL)

                    // Update model status
                    if let index = self?.models.firstIndex(where: { $0.id == modelId }) {
                        self?.models[index].isDownloaded = true
                    }
                } catch {
                    self?.downloadError = "Failed to save model: \(error.localizedDescription)"
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

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Restart banner
                if modelManager.needsRestart {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("Restart required to apply model change")
                            .font(.callout)
                        Spacer()
                        Button("Restart Now") {
                            restartApp()
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                    .padding(10)
                    .background(Color.orange.opacity(0.15))
                    .cornerRadius(8)
                }

                // Pipeline Mode Section
                GroupBox {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(PipelineMode.allCases, id: \.self) { mode in
                            HStack(spacing: 12) {
                                // Radio button
                                ZStack {
                                    Circle()
                                        .stroke(mode == modelManager.currentPipelineMode ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: 2)
                                        .frame(width: 20, height: 20)
                                    if mode == modelManager.currentPipelineMode {
                                        Circle()
                                            .fill(Color.accentColor)
                                            .frame(width: 12, height: 12)
                                    }
                                }

                                VStack(alignment: .leading, spacing: 2) {
                                    Text(mode.displayName)
                                        .fontWeight(.medium)
                                    Text(mode.description)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }

                                Spacer()
                            }
                            .contentShape(Rectangle())
                            .onTapGesture {
                                modelManager.selectPipelineMode(mode)
                            }

                            if mode != PipelineMode.allCases.last {
                                Divider()
                            }
                        }

                        // Show consolidated model selection when in consolidated mode
                        if modelManager.currentPipelineMode == .consolidated {
                            Divider()
                            Text("Consolidated Model")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .padding(.top, 4)

                            ForEach(modelManager.consolidatedModels) { model in
                                HStack(spacing: 12) {
                                    ZStack {
                                        Circle()
                                            .stroke(model.id == modelManager.currentConsolidatedModelId ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: 2)
                                            .frame(width: 18, height: 18)
                                        if model.id == modelManager.currentConsolidatedModelId {
                                            Circle()
                                                .fill(Color.accentColor)
                                                .frame(width: 10, height: 10)
                                        }
                                    }

                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(model.displayName)
                                            .fontWeight(.medium)
                                        Text(String(format: "%.1f GB", model.sizeGB))
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }

                                    Spacer()

                                    if model.isDownloaded {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundColor(.green)
                                    } else if modelManager.downloadingConsolidatedModelId == model.id {
                                        HStack(spacing: 6) {
                                            ProgressView()
                                                .scaleEffect(0.7)
                                            Text("\(Int(modelManager.consolidatedDownloadProgress * 100))%")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                    } else {
                                        Button(action: {
                                            modelManager.downloadConsolidatedModel(model.id)
                                        }) {
                                            Image(systemName: "arrow.down.circle")
                                                .foregroundColor(.accentColor)
                                        }
                                        .buttonStyle(.plain)
                                        .disabled(modelManager.downloadingConsolidatedModelId != nil)
                                    }
                                }
                                .padding(.leading, 20)
                                .contentShape(Rectangle())
                                .onTapGesture {
                                    if model.isDownloaded {
                                        modelManager.selectConsolidatedModel(model.id)
                                    }
                                }
                            }

                            HStack(spacing: 4) {
                                Image(systemName: "info.circle")
                                    .foregroundColor(.blue)
                                Text("Models are downloaded from HuggingFace (PyTorch format).")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.leading, 20)

                            if let error = modelManager.downloadError, modelManager.downloadingConsolidatedModelId == nil {
                                HStack(spacing: 4) {
                                    Image(systemName: "exclamationmark.triangle")
                                        .foregroundColor(.orange)
                                    Text(error)
                                        .font(.caption)
                                        .foregroundColor(.orange)
                                }
                                .padding(.leading, 20)
                            }
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Pipeline Mode", systemImage: "arrow.triangle.branch")
                        .font(.headline)
                }

                // STT Engine Section (only shown in traditional mode)
                if modelManager.currentPipelineMode == .sttPlusLlm {
                GroupBox {
                    VStack(alignment: .leading, spacing: 12) {
                        // Engine selection
                        ForEach(SttEngine.allCases, id: \.self) { engine in
                            SttEngineRowView(
                                engine: engine,
                                isSelected: engine == modelManager.currentSttEngine,
                                onSelect: {
                                    modelManager.selectSttEngine(engine)
                                }
                            )

                            // Show Moonshine model options when Moonshine is selected
                            if engine == .moonshine && modelManager.currentSttEngine == .moonshine {
                                VStack(spacing: 8) {
                                    ForEach(modelManager.moonshineModels) { model in
                                        MoonshineModelRowView(
                                            model: model,
                                            isSelected: model.id == modelManager.currentMoonshineModelId,
                                            isDownloading: modelManager.downloadingMoonshineModelId == model.id,
                                            downloadProgress: modelManager.moonshineDownloadProgress,
                                            onSelect: {
                                                modelManager.selectMoonshineModel(model.id)
                                            },
                                            onDownload: {
                                                modelManager.downloadMoonshineModel(model.id)
                                            }
                                        )
                                    }
                                }
                                .padding(.leading, 32)
                                .padding(.top, 4)
                            }

                            // Show Qwen3-ASR model info when selected
                            if engine == .qwen3Asr && modelManager.currentSttEngine == .qwen3Asr {
                                VStack(alignment: .leading, spacing: 6) {
                                    let hasModel = modelManager.consolidatedModels.contains(where: { $0.isDownloaded })
                                    if hasModel {
                                        HStack(spacing: 6) {
                                            Image(systemName: "checkmark.circle.fill")
                                                .foregroundColor(.green)
                                                .font(.caption)
                                            Text("Qwen3-ASR model available")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                    } else {
                                        HStack(spacing: 6) {
                                            Image(systemName: "exclamationmark.triangle.fill")
                                                .foregroundColor(.orange)
                                                .font(.caption)
                                            Text("Download a Qwen3-ASR model from Consolidated Models section")
                                                .font(.caption)
                                                .foregroundColor(.orange)
                                        }
                                    }
                                    Text("Transcription via Python daemon + LLM formatting")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                }
                                .padding(.leading, 32)
                                .padding(.top, 4)
                            }

                            if engine != SttEngine.allCases.last {
                                Divider()
                            }
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Speech-to-Text Engine", systemImage: "waveform")
                        .font(.headline)
                }

                // LLM Models Section
                GroupBox {
                    VStack(spacing: 12) {
                        ForEach(modelManager.models) { model in
                            ModelRowView(
                                model: model,
                                isSelected: model.id == modelManager.currentModelId && modelManager.currentVlmModelId == nil,
                                isDownloading: modelManager.downloadingModelId == model.id,
                                downloadProgress: modelManager.downloadProgress,
                                onSelect: {
                                    if model.isDownloaded {
                                        modelManager.selectModel(model.id)
                                    }
                                },
                                onDownload: {
                                    modelManager.downloadModel(model.id)
                                }
                            )

                            if model.id != modelManager.models.last?.id {
                                Divider()
                            }
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("LLM Models", systemImage: "cpu")
                        .font(.headline)
                }
                } // end if sttPlusLlm

                // VLM Models Section
                GroupBox {
                    VStack(spacing: 12) {
                        ForEach(modelManager.vlmModels) { model in
                            VlmModelRowView(
                                model: model,
                                isSelected: model.id == modelManager.currentVlmModelId,
                                isDownloading: modelManager.downloadingVlmModelId == model.id,
                                downloadProgress: modelManager.vlmDownloadProgress,
                                onSelect: {
                                    modelManager.selectVlmModel(model.id)
                                },
                                onDownload: {
                                    modelManager.downloadVlmModel(model.id)
                                }
                            )

                            if model.id != modelManager.vlmModels.last?.id {
                                Divider()
                            }
                        }

                        HStack(spacing: 4) {
                            Image(systemName: "info.circle")
                                .foregroundColor(.blue)
                            Text("VLM models are downloaded from HuggingFace (PyTorch safetensors format).")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Vision-Language Models", systemImage: "eye")
                        .font(.headline)
                }

                // Error display
                if let error = modelManager.downloadError {
                    HStack {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.red)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                }

                // Info section
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Larger models generally provide better formatting quality but require more memory and are slower to process.")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        HStack(spacing: 4) {
                            Image(systemName: "info.circle")
                                .foregroundColor(.blue)
                            Text("Models are downloaded from HuggingFace (Q4_K_M quantization)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Info", systemImage: "questionmark.circle")
                        .font(.headline)
                }
            }
            .padding()
        }
        .onAppear {
            modelManager.loadModels()
            modelManager.loadVlmSettings()
        }
    }

    private func restartApp() {
        let bundlePath = Bundle.main.bundlePath
        let script = """
            sleep 0.5
            open "\(bundlePath)"
            """
        let task = Process()
        task.launchPath = "/bin/bash"
        task.arguments = ["-c", script]
        try? task.run()
        // applicationWillTerminate will handle cleanup
        NSApp.terminate(nil)
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

// MARK: - VLM Model Row View

struct VlmModelRowView: View {
    let model: VlmModelItem
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
            if model.isDownloaded && !isSelected {
                onSelect()
            }
        }
        .opacity(model.isDownloaded || isDownloading ? 1.0 : 0.7)
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

// MARK: - Overlay State

class OverlayState: ObservableObject {
    enum RecordingState {
        case idle
        case recording
        case processing
        case aiProcessing(String)   // label text like "Summarizing..."

        var isRecording: Bool {
            if case .recording = self { return true }
            return false
        }

        var isProcessing: Bool {
            if case .processing = self { return true }
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

    private var pillWidth: CGFloat {
        if case .aiProcessing = state.state { return 260 }
        return 180
    }

    var body: some View {
        HStack(spacing: 10) {
            if case .aiProcessing(let label) = state.state {
                // AI processing: sparkle icon + label + pulsing dots
                Image(systemName: "sparkles")
                    .font(.system(size: 18))
                    .foregroundColor(.white)
                Text(label)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.white)
                PulsingDotsView()
                    .frame(width: 50, height: 20)
                    .scaleEffect(0.7)
            } else {
                // Logo
                logoView

                // Waveform bars or processing indicator
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
        .frame(width: pillWidth, height: 66)
        .background(backgroundView)
        .onAppear {
            withAnimation(.linear(duration: 3.0).repeatForever(autoreverses: false)) {
                animateGradient = true
            }
        }
    }

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

    private var backgroundView: some View {
        ZStack {
            if state.state.isRecording {
                // Animated gradient for recording
                AnimatedGradientBackground(animate: animateGradient, colors: gradientColors)
            } else {
                // Dark background for processing/idle
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
