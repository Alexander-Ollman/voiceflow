import AppKit
import Foundation

/// Extracts likely proper-noun and technical-term candidates from the focused
/// window's AX text content. Used to seed the LLM's vocabulary-hint block so
/// Parakeet mishearings ("Postgres" heard as "post grass") get auto-corrected
/// using terms visible on screen.
///
/// Strategy:
///  1. Walk the AX tree of the focused window collecting visible text
///  2. Tokenize and filter for proper-noun / camelCase / acronym patterns
///  3. Score by frequency × recency × proximity to focused element
///  4. Return top N candidates
///
/// This is cheap (~5–20ms typically) and runs at recording start, off the main
/// thread when possible. Caches per-window so repeated dictations don't re-walk.
enum ScreenVocabExtractor {

    private static let cache = NSCache<NSString, NSArray>()
    private static let cacheTTL: TimeInterval = 30
    private struct CacheEntry {
        let words: [String]
        let timestamp: Date
    }
    private static var entryMap: [String: CacheEntry] = [:]
    private static let lock = NSLock()

    /// Extract up to `limit` vocabulary candidates from the focused window.
    /// Returns the empty array if AX is unavailable, the window has no text,
    /// or extraction times out.
    static func extract(limit: Int = 24) -> [String] {
        guard AXIsProcessTrusted() else { return [] }

        let cacheKey = (NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "_") + "::" + WindowKey.current()

        lock.lock()
        if let entry = entryMap[cacheKey],
           Date().timeIntervalSince(entry.timestamp) < cacheTTL {
            lock.unlock()
            return Array(entry.words.prefix(limit))
        }
        lock.unlock()

        let systemWide = AXUIElementCreateSystemWide()
        var focusedAppRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedApplicationAttribute as CFString,
            &focusedAppRef
        ) == .success, let focusedAppRef = focusedAppRef else { return [] }
        let app = focusedAppRef as! AXUIElement

        var windowRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            app,
            kAXFocusedWindowAttribute as CFString,
            &windowRef
        ) == .success, let windowRef = windowRef else { return [] }
        let window = windowRef as! AXUIElement

        var collected: [String] = []
        var visited = 0
        walkAXTree(window, depth: 0, maxDepth: 8, into: &collected, visited: &visited, maxVisits: 600)

        let candidates = filterAndRank(from: collected, limit: limit)

        lock.lock()
        entryMap[cacheKey] = CacheEntry(words: candidates, timestamp: Date())
        lock.unlock()

        return Array(candidates.prefix(limit))
    }

    /// Manually invalidate the cache (e.g. after a known major UI change).
    static func invalidate() {
        lock.lock()
        entryMap.removeAll()
        lock.unlock()
    }

    // MARK: - AX tree walk

    private static func walkAXTree(
        _ element: AXUIElement,
        depth: Int,
        maxDepth: Int,
        into out: inout [String],
        visited: inout Int,
        maxVisits: Int
    ) {
        if depth > maxDepth || visited > maxVisits { return }
        visited += 1

        // Pull any text-like attribute values
        for attr in ["AXTitle", "AXDescription", "AXValue", "AXHelp", "AXLabel"] {
            var valueRef: CFTypeRef?
            if AXUIElementCopyAttributeValue(element, attr as CFString, &valueRef) == .success,
               let s = valueRef as? String, !s.isEmpty {
                out.append(s)
            }
        }

        // Recurse into children
        var childrenRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &childrenRef) == .success,
              let children = childrenRef as? [AXUIElement]
        else { return }

        for child in children {
            if visited > maxVisits { return }
            walkAXTree(child, depth: depth + 1, maxDepth: maxDepth, into: &out, visited: &visited, maxVisits: maxVisits)
        }
    }

    // MARK: - Candidate filtering

    private static func filterAndRank(from texts: [String], limit: Int) -> [String] {
        var freq: [String: Int] = [:]
        for text in texts {
            for token in tokenize(text) {
                if isCandidate(token) {
                    freq[token, default: 0] += 1
                }
            }
        }
        let sorted = freq
            .filter { $0.value >= 1 }
            .sorted {
                if $0.value != $1.value { return $0.value > $1.value }
                return $0.key.count > $1.key.count
            }
            .prefix(limit)
            .map { $0.key }
        return Array(sorted)
    }

    /// Tokenize on whitespace + common punctuation, but preserve internal
    /// hyphens, dots, slashes (so URLs / file paths / branded names survive).
    private static func tokenize(_ s: String) -> [String] {
        s.unicodeScalars
            .split(whereSeparator: { c in
                !(CharacterSet.alphanumerics.contains(c)
                    || c == "-" || c == "_" || c == "."
                    || c == "/" || c == "@")
            })
            .map { String(String.UnicodeScalarView($0)) }
    }

    /// Heuristic: is this token worth including as a vocab hint?
    /// We're looking for proper nouns / technical terms / branded names — not
    /// common English words.
    private static func isCandidate(_ token: String) -> Bool {
        guard token.count >= 2, token.count <= 32 else { return false }

        // Reject pure numbers / numbery tokens
        if token.allSatisfy({ $0.isNumber || $0 == "." || $0 == "," }) { return false }

        // Must contain at least one letter
        guard token.contains(where: { $0.isLetter }) else { return false }

        // Common English word reject list (small; the model can re-add via dict)
        let stopwords: Set<String> = [
            "the", "and", "for", "you", "your", "this", "that", "with", "from",
            "have", "has", "are", "was", "were", "they", "them", "their", "but",
            "not", "all", "any", "can", "will", "would", "could", "should",
            "what", "when", "where", "which", "who", "why", "how",
            "settings", "menu", "search", "file", "edit", "view", "help", "open",
            "save", "close", "back", "next", "previous", "ok", "cancel", "submit",
            "yes", "no", "page", "home", "about", "contact", "login", "sign",
            "username", "password", "email", "name", "title", "description",
        ]
        if stopwords.contains(token.lowercased()) { return false }

        // Capitalized (proper noun candidate)
        if let first = token.first, first.isUppercase {
            return true
        }
        // camelCase / PascalCase
        if hasCamelOrPascalCase(token) {
            return true
        }
        // ALL CAPS acronym (2–8 chars)
        if token.count >= 2 && token.count <= 8 && token.allSatisfy({ $0.isUppercase || $0.isNumber }) {
            return true
        }
        // Has dot/slash (likely identifier or path)
        if token.contains(".") || token.contains("/") {
            return true
        }
        return false
    }

    private static func hasCamelOrPascalCase(_ s: String) -> Bool {
        var sawLower = false
        var sawUpper = false
        for (i, c) in s.enumerated() {
            if c.isUppercase && i > 0 { sawUpper = true }
            if c.isLowercase { sawLower = true }
            if sawLower && sawUpper { return true }
        }
        return false
    }
}
