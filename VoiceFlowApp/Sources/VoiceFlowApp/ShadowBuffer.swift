import AppKit
import Foundation

/// Per-target insertion history.
///
/// We always know what we pasted, when, and roughly where. The shadow buffer is
/// the safety net under AX read-back / browser bridge / Vision OCR: even when
/// every other context source fails, retroactive correction can still operate
/// on what we ourselves typed.
///
/// Sliding window: last `maxEntries` insertions, or anything younger than
/// `maxAge`. Older / overflowing entries are discarded.
///
/// In-memory only — never persisted to disk.
final class ShadowBuffer {
    static let shared = ShadowBuffer()

    struct Entry {
        let timestamp: Date
        let bundleId: String
        let windowKey: String   // window title or fallback "_"
        let cursorAtPaste: Int? // cursor offset just before paste, if known
        let inserted: String
    }

    private let queue = DispatchQueue(label: "voiceflow.shadowbuffer")
    private var entries: [Entry] = []
    private let maxEntries = 32
    private let maxAge: TimeInterval = 5 * 60 // 5 minutes

    func record(bundleId: String, windowKey: String, cursorAtPaste: Int?, inserted: String) {
        guard !inserted.isEmpty else { return }
        queue.sync {
            entries.append(Entry(
                timestamp: Date(),
                bundleId: bundleId,
                windowKey: windowKey,
                cursorAtPaste: cursorAtPaste,
                inserted: inserted
            ))
            prune()
        }
    }

    /// Recent insertions for the given target, newest last.
    func recent(bundleId: String, windowKey: String, limit: Int = 10) -> [Entry] {
        queue.sync {
            prune()
            return entries
                .filter { $0.bundleId == bundleId && $0.windowKey == windowKey }
                .suffix(limit)
                .map { $0 }
        }
    }

    /// Synthesize a plausible "field content" string from recent insertions for
    /// the given target. Used as the LLM-prompt context when AX/DOM/OCR all
    /// fail. Joins with single space — not authoritative content, just *our*
    /// record of what we inserted.
    func synthesizedContext(bundleId: String, windowKey: String, maxLength: Int = 4000) -> String {
        let entries = recent(bundleId: bundleId, windowKey: windowKey, limit: 20)
        let joined = entries.map { $0.inserted }.joined(separator: " ")
        if joined.count <= maxLength { return joined }
        // Keep the tail (most recent text near the cursor)
        let start = joined.index(joined.endIndex, offsetBy: -maxLength)
        return String(joined[start...])
    }

    /// Drop the most-recent entry (call when an Undo is detected so we don't
    /// continue believing we pasted text the user reverted).
    func popLast(bundleId: String, windowKey: String) {
        queue.sync {
            if let idx = entries.lastIndex(where: { $0.bundleId == bundleId && $0.windowKey == windowKey }) {
                entries.remove(at: idx)
            }
        }
    }

    private func prune() {
        let cutoff = Date().addingTimeInterval(-maxAge)
        entries.removeAll { $0.timestamp < cutoff }
        if entries.count > maxEntries {
            entries.removeFirst(entries.count - maxEntries)
        }
    }
}

// MARK: - Window-key resolver

/// Reads the focused window's title via AX as a stable-enough per-target key.
/// Falls back to "_" when AX isn't available (we can still scope by bundleId).
enum WindowKey {
    static func current() -> String {
        guard AXIsProcessTrusted() else { return "_" }

        let systemWide = AXUIElementCreateSystemWide()
        var focusedAppRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedApplicationAttribute as CFString,
            &focusedAppRef
        ) == .success, let focusedAppRef = focusedAppRef else { return "_" }
        let app = focusedAppRef as! AXUIElement

        var windowRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            app,
            kAXFocusedWindowAttribute as CFString,
            &windowRef
        ) == .success, let windowRef = windowRef else { return "_" }
        let window = windowRef as! AXUIElement

        var titleRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            window,
            kAXTitleAttribute as CFString,
            &titleRef
        ) == .success, let title = titleRef as? String, !title.isEmpty else {
            return "_"
        }
        return title
    }
}
