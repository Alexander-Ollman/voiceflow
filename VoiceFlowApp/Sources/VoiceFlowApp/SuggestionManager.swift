import Foundation
import Combine

/// One suggested shortcut, surfaced in the Insights dashboard. Backed by
/// Bonsai's analysis of the user's recent dictation history. The user can
/// Accept (creates a real Snippet) or Dismiss (marked permanently
/// dismissed by id so it won't re-suggest).
struct Suggestion: Codable, Identifiable, Equatable {
    let id: UUID
    var trigger: String
    var expansion: String
    var reason: String
    var occurrences: Int
    /// Hash of (trigger + expansion) so we can dismiss across re-analyses
    /// (Bonsai might invent a new UUID for the same suggestion next time).
    var contentHash: String
    var createdAt: Date

    init(trigger: String, expansion: String, reason: String, occurrences: Int) {
        self.id = UUID()
        self.trigger = trigger
        self.expansion = expansion
        self.reason = reason
        self.occurrences = occurrences
        self.contentHash = Self.hash(trigger: trigger, expansion: expansion)
        self.createdAt = Date()
    }

    static func hash(trigger: String, expansion: String) -> String {
        // Cheap deterministic dedupe key. Lowercased + trimmed so trivial
        // formatting differences across analyses don't produce dupes.
        let t = trigger.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let e = expansion.trimmingCharacters(in: .whitespacesAndNewlines)
        return "\(t)\u{1}\(e.prefix(120))"
    }
}

/// Persisted-to-disk state for the suggestions feature.
private struct SuggestionStore: Codable {
    var lastAnalysis: Date?
    var suggestions: [Suggestion]
    var dismissedHashes: [String]

    static let empty = SuggestionStore(lastAnalysis: nil, suggestions: [], dismissedHashes: [])
}

/// Drives the "Suggested Shortcuts" panel in the Insights dashboard.
/// Reads the user's transcription history via TranscriptionLog.shared,
/// asks Bonsai (direct llama-server call, mirroring the post-paste edit
/// feature) to identify repeated phrases / structured data / common
/// openings, and persists results to a JSON cache so we don't re-analyze
/// on every dashboard view.
///
/// Cache invariant: if `lastAnalysis` is within `cacheLifetime` (24 h),
/// the cached suggestions are shown immediately. Older or absent → kick
/// off `analyzeHistory()` in the background.
@MainActor
final class SuggestionManager: ObservableObject {
    static let shared = SuggestionManager()

    @Published private(set) var suggestions: [Suggestion] = []
    @Published private(set) var lastAnalysis: Date?
    @Published private(set) var isAnalyzing: Bool = false
    @Published private(set) var lastError: String?

    /// 24 h — re-analyze at most once a day on dashboard view.
    static let cacheLifetime: TimeInterval = 24 * 60 * 60

    /// How many recent entries we send to Bonsai. Each entry is 1 line in
    /// the prompt; 200 entries with ~30 words avg = ~6,000 words = 8K
    /// tokens, comfortably inside Bonsai's 16K context.
    static let maxHistoryEntries: Int = 200

    private var dismissedHashes: Set<String> = []

    private var storeURL: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("com.era-laboratories.voiceflow")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("suggestions.json")
    }

    init() {
        load()
    }

    // MARK: - Persistence

    private func load() {
        guard let data = try? Data(contentsOf: storeURL),
              let store = try? JSONDecoder.iso8601().decode(SuggestionStore.self, from: data) else {
            return
        }
        self.lastAnalysis = store.lastAnalysis
        self.dismissedHashes = Set(store.dismissedHashes)
        self.suggestions = store.suggestions.filter { !dismissedHashes.contains($0.contentHash) }
    }

    private func save() {
        let store = SuggestionStore(
            lastAnalysis: lastAnalysis,
            suggestions: suggestions,
            dismissedHashes: Array(dismissedHashes)
        )
        if let data = try? JSONEncoder.iso8601().encode(store) {
            try? data.write(to: storeURL)
        }
    }

    // MARK: - Cache freshness

    var isCacheFresh: Bool {
        guard let last = lastAnalysis else { return false }
        return Date().timeIntervalSince(last) < Self.cacheLifetime
    }

    /// Kick off an analysis if the cache is stale (or empty) and we
    /// aren't already analyzing. Safe to call repeatedly — no-op when
    /// already fresh.
    func refreshIfStale() {
        guard !isAnalyzing else { return }
        guard !isCacheFresh else { return }
        Task { await analyzeHistory() }
    }

    /// User-triggered force-refresh from the dashboard "Refresh" button.
    func forceRefresh() {
        guard !isAnalyzing else { return }
        Task { await analyzeHistory() }
    }

    // MARK: - User actions

    /// Accept a suggestion → real Snippet, remove from list.
    func accept(_ suggestion: Suggestion) {
        SnippetManager.shared.addSnippet(
            trigger: suggestion.trigger,
            expansion: suggestion.expansion
        )
        // Also mark as dismissed so re-analysis doesn't re-suggest.
        dismissedHashes.insert(suggestion.contentHash)
        suggestions.removeAll { $0.id == suggestion.id }
        save()
    }

    /// Permanently dismiss a suggestion — won't re-suggest.
    func dismiss(_ suggestion: Suggestion) {
        dismissedHashes.insert(suggestion.contentHash)
        suggestions.removeAll { $0.id == suggestion.id }
        save()
    }

    // MARK: - Analysis

    func analyzeHistory() async {
        guard !isAnalyzing else { return }
        isAnalyzing = true
        lastError = nil
        defer { isAnalyzing = false }

        let entries = TranscriptionLog.shared.entries
            .prefix(Self.maxHistoryEntries)
            .map(\.formattedText)
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }

        guard entries.count >= 5 else {
            // Not enough data yet to find anything meaningful.
            lastError = "Not enough dictation history yet — keep dictating and check back."
            return
        }

        let existingTriggers = SnippetManager.shared.snippets
            .map { "\"\($0.trigger)\" → \"\($0.expansion)\"" }
            .joined(separator: "\n")

        let prompt = Self.buildPrompt(
            entries: Array(entries),
            existingSnippets: existingTriggers
        )

        do {
            let raw = try await callBonsai(prompt: prompt)
            let parsed = Self.parseSuggestions(from: raw)
                .filter { !dismissedHashes.contains($0.contentHash) }
            // Dedup against current list by content hash.
            let existingHashes = Set(self.suggestions.map(\.contentHash))
            let fresh = parsed.filter { !existingHashes.contains($0.contentHash) }
            self.suggestions = (self.suggestions + fresh)
                .sorted { $0.occurrences > $1.occurrences }
            self.lastAnalysis = Date()
            save()
        } catch {
            lastError = "Could not analyze history: \(error.localizedDescription)"
        }
    }

    private static func buildPrompt(entries: [String], existingSnippets: String) -> String {
        let history = entries.enumerated()
            .map { (i, text) in "\(i + 1). \(text)" }
            .joined(separator: "\n")
        let existing = existingSnippets.isEmpty ? "(none yet)" : existingSnippets
        return """
        You are analyzing a user's voice-dictation history to suggest shortcut \
        phrases. A shortcut is a short trigger the user can say (like "my \
        address") that expands to longer text when they dictate.

        Recent dictations (most recent last):
        \(history)

        Existing shortcuts the user already has configured:
        \(existing)

        Your task: identify patterns the user could benefit from converting \
        into shortcuts. Look for:
          1. Phrases the user repeats verbatim 3 or more times.
          2. Structured data they retype: addresses, phone numbers, email \
             addresses, sign-offs, names, dates, common URLs.
          3. Common opening or closing lines for messages.

        Return ONLY a JSON array. Each element MUST be an object with these \
        exact keys:
          {"trigger": "string (2-4 words, easy to say, unique-sounding)",
           "expansion": "string (the full text to paste in)",
           "reason": "string (one sentence explaining the pattern)",
           "occurrences": <integer count of how many times this appeared>}

        Rules:
          - Only suggest patterns that appeared at least 3 times.
          - Do NOT duplicate any existing shortcut listed above.
          - Prefer expansions that save real time (5+ words).
          - Triggers should be 2-4 words, easy to say out loud, and unlikely \
            to appear naturally in normal dictation.
          - Suggest at most 5 shortcuts. Quality over quantity.
          - Output STRICT JSON array only. No markdown, no commentary, no \
            preamble. If you see no good candidates, return: []
        """
    }

    /// POST to local llama-server, return the raw assistant content.
    private func callBonsai(prompt: String) async throws -> String {
        guard let url = URL(string: "http://127.0.0.1:8080/v1/chat/completions") else {
            throw NSError(domain: "SuggestionManager", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Invalid llama-server URL",
            ])
        }
        let body: [String: Any] = [
            "model": "default",
            "messages": [
                ["role": "system", "content": "You output ONLY valid JSON arrays. No markdown. No prose. No code fences."],
                ["role": "user", "content": prompt],
            ],
            "max_tokens": 1500,
            "temperature": 0.2,
            "stream": false,
        ]
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 90  // analysis can be slow (large prompt)
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, _) = try await URLSession.shared.data(for: request)
        guard
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let choices = json["choices"] as? [[String: Any]],
            let first = choices.first,
            let message = first["message"] as? [String: Any],
            let content = message["content"] as? String
        else {
            throw NSError(domain: "SuggestionManager", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Unexpected llama-server response shape",
            ])
        }
        return content
    }

    /// Bonsai is reliable but not perfect with JSON. Strip any prose
    /// padding around the array and try to parse what's left. Falls back
    /// to empty list on unrecoverable garbage.
    private static func parseSuggestions(from raw: String) -> [Suggestion] {
        // Strip markdown code fences if present.
        var stripped = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if stripped.hasPrefix("```") {
            // Remove leading fence (with or without language tag)
            if let firstNewline = stripped.firstIndex(of: "\n") {
                stripped = String(stripped[stripped.index(after: firstNewline)...])
            }
        }
        if stripped.hasSuffix("```") {
            stripped = String(stripped.dropLast(3)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        // Find the bounding array brackets — Bonsai sometimes adds a
        // preamble like "Here are the suggestions:" before the [.
        guard
            let start = stripped.firstIndex(of: "["),
            let end = stripped.lastIndex(of: "]"),
            start < end
        else {
            return []
        }
        let jsonText = String(stripped[start...end])
        guard let data = jsonText.data(using: .utf8) else { return [] }
        guard let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return []
        }
        return array.compactMap { dict -> Suggestion? in
            guard
                let trigger = (dict["trigger"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
                let expansion = (dict["expansion"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
                !trigger.isEmpty, !expansion.isEmpty
            else { return nil }
            let reason = (dict["reason"] as? String) ?? ""
            let occurrences: Int
            if let n = dict["occurrences"] as? Int {
                occurrences = n
            } else if let n = dict["occurrences"] as? Double {
                occurrences = Int(n)
            } else if let s = dict["occurrences"] as? String, let n = Int(s) {
                occurrences = n
            } else {
                occurrences = 0
            }
            return Suggestion(
                trigger: trigger,
                expansion: expansion,
                reason: reason,
                occurrences: occurrences
            )
        }
    }
}

private extension JSONDecoder {
    static func iso8601() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }
}

private extension JSONEncoder {
    static func iso8601() -> JSONEncoder {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }
}
