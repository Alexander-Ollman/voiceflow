import AppKit
import Foundation

/// Unified field-context resolver. Walks the layered fallback chain so the
/// rest of the app gets a single call.
///
/// Layer 1 — AX read (native fields)
/// Layer 2 — Browser JS bridge (web inputs in supported browsers)
/// Layer 3 — Shadow buffer (our own record of what we recently inserted)
/// Layer 4 — Vision OCR (universal, slowest)
///
/// `source` flags which layer answered so callers can adjust confidence
/// downstream — Bonsai's structured-edit prompt is given the source label
/// and weights the edit decision accordingly.
struct FieldContext {
    enum Source: String {
        case ax = "ax"
        case browser = "browser"
        case shadow = "shadow"
        case ocr = "ocr"
        case empty = "empty"
    }

    let text: String
    let cursor: Int          // best-effort offset; equals text.count when unknown
    let selectionLength: Int // 0 if no active selection
    let source: Source
    let bundleId: String
    let windowKey: String

    /// Resolve context for the currently-focused app/field.
    ///
    /// `requireFresh` skips the shadow buffer (only authoritative sources).
    /// `allowOCR` lets callers veto the expensive last-resort path.
    static func resolve(requireFresh: Bool = false, allowOCR: Bool = true) async -> FieldContext {
        let frontApp = NSWorkspace.shared.frontmostApplication
        let bundleId = frontApp?.bundleIdentifier ?? "unknown"
        let windowKey = WindowKey.current()

        // Layer 1: AX
        if let snap = CursorContext.getFieldContents() {
            if !snap.text.isEmpty {
                return FieldContext(
                    text: snap.text,
                    cursor: snap.cursor,
                    selectionLength: snap.selectionLength,
                    source: .ax,
                    bundleId: bundleId,
                    windowKey: windowKey
                )
            }
        }

        // Layer 2: Browser JS bridge
        if BrowserContext.isBrowser(bundleId: bundleId),
           let dom = BrowserContext.currentFieldSnapshot(forBundleId: bundleId),
           !dom.text.isEmpty {
            return FieldContext(
                text: dom.text,
                cursor: dom.cursor ?? dom.text.count,
                selectionLength: 0,
                source: .browser,
                bundleId: bundleId,
                windowKey: windowKey
            )
        }

        // Layer 3: shadow buffer (skip if caller wants only authoritative sources)
        if !requireFresh {
            let synthesized = ShadowBuffer.shared.synthesizedContext(
                bundleId: bundleId,
                windowKey: windowKey
            )
            if !synthesized.isEmpty {
                return FieldContext(
                    text: synthesized,
                    cursor: synthesized.count,
                    selectionLength: 0,
                    source: .shadow,
                    bundleId: bundleId,
                    windowKey: windowKey
                )
            }
        }

        // Layer 4: Vision OCR
        if allowOCR, let ocr = await VisionOCR.ocrFrontWindow() {
            return FieldContext(
                text: ocr.text,
                cursor: ocr.text.count,
                selectionLength: 0,
                source: .ocr,
                bundleId: bundleId,
                windowKey: windowKey
            )
        }

        return FieldContext(
            text: "",
            cursor: 0,
            selectionLength: 0,
            source: .empty,
            bundleId: bundleId,
            windowKey: windowKey
        )
    }

    /// Confidence weight for the source — passed to Bonsai so it can decline
    /// risky edits when our context is fuzzy.
    var sourceConfidence: Double {
        switch source {
        case .ax: return 1.0
        case .browser: return 0.95
        case .shadow: return 0.7
        case .ocr: return 0.55
        case .empty: return 0.0
        }
    }
}
