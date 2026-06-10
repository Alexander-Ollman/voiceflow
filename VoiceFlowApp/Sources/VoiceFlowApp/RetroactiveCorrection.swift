import AppKit
import Foundation

/// Resolves a structured `Edit` into a concrete byte range in the focused
/// field, then applies the replacement via the existing AX/paste path.
///
/// Confidence below `Self.minConfidence` is treated as abstain: the caller
/// should fall back to the default paste flow.
enum RetroactiveApplier {

    /// Minimum LLM confidence before we apply an edit silently. Below this we
    /// either abstain entirely or surface the proposal for user confirmation
    /// (TODO once confirmation UI lands).
    static let minConfidence: Float = 0.7

    enum ApplyResult {
        case applied(String)        // success — confirmation message
        case abstained(String)      // low confidence / no_op — reason
        case anchorMissing(String)  // anchor not in field — reason
        case failure(String)        // technical error
    }

    /// Apply the edit against the given field context. Side effects: writes
    /// to clipboard, simulates paste, records to shadow buffer.
    @MainActor
    static func apply(edit: Edit, context: FieldContext) -> ApplyResult {
        if edit.confidence < minConfidence || edit.action == .noOp {
            return .abstained("Confidence \(String(format: "%.2f", edit.confidence)) below threshold")
        }

        switch edit.action {
        case .replaceRange:
            return applyReplace(edit: edit, context: context)
        case .delete:
            return applyDelete(edit: edit, context: context)
        case .insert:
            return applyInsert(edit: edit, context: context)
        case .noOp:
            return .abstained("Model returned no_op")
        }
    }

    // MARK: - Replace

    @MainActor
    private static func applyReplace(edit: Edit, context: FieldContext) -> ApplyResult {
        guard !edit.anchor.isEmpty else {
            return .anchorMissing("Empty anchor for replace_range")
        }
        guard let range = locate(anchor: edit.anchor, in: context.text, occurrence: edit.occurrence) else {
            return .anchorMissing("Anchor '\(edit.anchor)' not found in field text")
        }

        // Try AX select-and-paste first.
        let pasteboard = NSPasteboard.general
        let savedItems = capturePasteboard(pasteboard)
        pasteboard.clearContents()
        pasteboard.setString(edit.replacement, forType: .string)

        // Convert character range to UTF-16 offset for AX (AX uses NSRange semantics).
        let nsRange = NSRange(range, in: context.text)
        let selected = CursorContext.selectRange(location: nsRange.location, length: nsRange.length)

        if !selected {
            restorePasteboard(pasteboard, savedItems: savedItems)
            return .failure("AX selectRange failed in source=\(context.source.rawValue)")
        }

        usleep(40_000) // 40ms for selection to settle
        AppDelegate.paste()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            restorePasteboard(pasteboard, savedItems: savedItems)
        }

        // Update the shadow buffer to reflect the new state.
        ShadowBuffer.shared.record(
            bundleId: context.bundleId,
            windowKey: context.windowKey,
            cursorAtPaste: nsRange.location,
            inserted: edit.replacement
        )

        return .applied("\(edit.anchor) → \(edit.replacement)")
    }

    // MARK: - Delete

    @MainActor
    private static func applyDelete(edit: Edit, context: FieldContext) -> ApplyResult {
        guard !edit.anchor.isEmpty else {
            return .anchorMissing("Empty anchor for delete")
        }
        guard let range = locate(anchor: edit.anchor, in: context.text, occurrence: edit.occurrence) else {
            return .anchorMissing("Anchor '\(edit.anchor)' not found")
        }

        let nsRange = NSRange(range, in: context.text)
        guard CursorContext.selectRange(location: nsRange.location, length: nsRange.length) else {
            return .failure("AX selectRange failed for delete")
        }

        // Delete the selection by sending a backspace keystroke.
        usleep(40_000)
        sendBackspace()
        return .applied("Deleted '\(edit.anchor)'")
    }

    // MARK: - Insert

    @MainActor
    private static func applyInsert(edit: Edit, context: FieldContext) -> ApplyResult {
        guard !edit.replacement.isEmpty else {
            return .abstained("Empty replacement for insert")
        }

        let pasteboard = NSPasteboard.general
        let savedItems = capturePasteboard(pasteboard)
        pasteboard.clearContents()
        pasteboard.setString(edit.replacement, forType: .string)
        AppDelegate.paste()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            restorePasteboard(pasteboard, savedItems: savedItems)
        }
        ShadowBuffer.shared.record(
            bundleId: context.bundleId,
            windowKey: context.windowKey,
            cursorAtPaste: context.cursor,
            inserted: edit.replacement
        )
        return .applied("Inserted \(edit.replacement.count) chars")
    }

    // MARK: - Anchor location

    /// Find the byte range for `anchor` in `text` according to `occurrence`.
    /// Returns nil when the anchor doesn't appear.
    static func locate(
        anchor: String,
        in text: String,
        occurrence: Occurrence
    ) -> Range<String.Index>? {
        let ranges = allRanges(of: anchor, in: text)
        if ranges.isEmpty { return nil }
        switch occurrence {
        case .first, .only: return ranges.first
        case .last: return ranges.last
        }
    }

    private static func allRanges(of needle: String, in haystack: String) -> [Range<String.Index>] {
        guard !needle.isEmpty else { return [] }
        var ranges: [Range<String.Index>] = []
        var searchStart = haystack.startIndex
        while searchStart < haystack.endIndex,
              let r = haystack.range(of: needle, range: searchStart..<haystack.endIndex) {
            ranges.append(r)
            searchStart = r.upperBound
        }
        return ranges
    }

    // MARK: - Clipboard helpers (mirror of performRetroactiveCorrection)

    private static func capturePasteboard(_ pb: NSPasteboard) -> [(NSPasteboard.PasteboardType, Data)] {
        pb.pasteboardItems?.compactMap { item in
            for type in item.types {
                if let data = item.data(forType: type) {
                    return (type, data)
                }
            }
            return nil
        } ?? []
    }

    private static func restorePasteboard(
        _ pb: NSPasteboard,
        savedItems: [(NSPasteboard.PasteboardType, Data)]
    ) {
        pb.clearContents()
        for (type, data) in savedItems {
            pb.setData(data, forType: type)
        }
    }

    @MainActor
    private static func sendBackspace() {
        let source = CGEventSource(stateID: .combinedSessionState)
        let down = CGEvent(keyboardEventSource: source, virtualKey: 51, keyDown: true) // 51 = delete
        let up = CGEvent(keyboardEventSource: source, virtualKey: 51, keyDown: false)
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)
    }
}

extension AppDelegate {

    /// Record a fresh paste into the shadow buffer so retroactive correction
    /// has fall-back context when AX/browser read-back fails. Read the cursor
    /// position immediately before paste — best-effort, OK if it's nil.
    @MainActor
    func recordPasteToShadowBuffer(_ text: String) {
        let bundleId = NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "unknown"
        let windowKey = WindowKey.current()
        let cursor = CursorContext.getCursorPosition()
        ShadowBuffer.shared.record(
            bundleId: bundleId,
            windowKey: windowKey,
            cursorAtPaste: cursor,
            inserted: text
        )
    }

    /// Try to intercept a freshly-transcribed utterance as a retroactive
    /// correction OR an AI voice command. Returns true if handled (caller
    /// should skip normal paste); false otherwise (caller falls through).
    @MainActor
    func interceptAIIntent(
        rawTranscript: String,
        voiceFlow: VoiceFlowBridge
    ) async -> Bool {
        guard let intent = VoiceFlowBridge.classifyIntent(rawTranscript) else { return false }

        switch intent.kind {
        case .retroactiveCorrection:
            return await handleRetroactive(rawTranscript: rawTranscript, voiceFlow: voiceFlow)
        case .command(let cmd):
            return await handleCommand(cmd, parameter: intent.residual, voiceFlow: voiceFlow)
        case .verbatim, .inlineCorrection:
            return false
        }
    }

    // Backwards compat alias for the earlier integration site.
    @MainActor
    func interceptRetroactiveCorrection(
        rawTranscript: String,
        voiceFlow: VoiceFlowBridge
    ) async -> Bool {
        await interceptAIIntent(rawTranscript: rawTranscript, voiceFlow: voiceFlow)
    }

    @MainActor
    private func handleRetroactive(
        rawTranscript: String,
        voiceFlow: VoiceFlowBridge
    ) async -> Bool {
        NSLog("[VoiceFlow] Retroactive intent detected for utterance: %@", rawTranscript)

        // 2. Field context
        let ctx = await FieldContext.resolve()
        guard ctx.source != .empty, !ctx.text.isEmpty else {
            NSLog("[VoiceFlow] Retroactive: no field context available, falling through")
            return false
        }
        NSLog("[VoiceFlow] Retroactive: ctx source=%@ text_len=%d cursor=%d",
              ctx.source.rawValue, ctx.text.count, ctx.cursor)

        // 3. Recent insertions from shadow buffer
        let recents = ShadowBuffer.shared.recent(
            bundleId: ctx.bundleId,
            windowKey: ctx.windowKey,
            limit: 5
        ).map { $0.inserted }

        // 4. Call Bonsai
        let input = RetroactiveInput(
            fieldText: ctx.text,
            fieldSource: ctx.source.rawValue,
            recentInsertions: recents,
            userUtterance: rawTranscript
        )
        guard let edit = await voiceFlow.retroactiveCorrect(input) else {
            NSLog("[VoiceFlow] Retroactive: LLM call failed, falling through")
            return false
        }
        NSLog("[VoiceFlow] Retroactive: LLM returned action=%@ anchor=%@ replacement=%@ confidence=%.2f",
              String(describing: edit.action), edit.anchor, edit.replacement, edit.confidence)

        // 5. Apply
        let result = RetroactiveApplier.apply(edit: edit, context: ctx)
        switch result {
        case .applied(let msg):
            RetroactiveToast.shared.show(msg, source: ctx.source)
            NSLog("[VoiceFlow] Retroactive: APPLIED %@", msg)
            return true
        case .abstained(let reason):
            NSLog("[VoiceFlow] Retroactive: abstained — %@", reason)
            return false
        case .anchorMissing(let reason):
            NSLog("[VoiceFlow] Retroactive: anchor missing — %@", reason)
            return false
        case .failure(let reason):
            NSLog("[VoiceFlow] Retroactive: failure — %@", reason)
            return false
        }
    }

    @MainActor
    private func handleCommand(
        _ command: CommandKind,
        parameter: String,
        voiceFlow: VoiceFlowBridge
    ) async -> Bool {
        NSLog("[VoiceFlow] AI command intent: %@ param=%@",
              String(describing: command), parameter)

        let ctx = await FieldContext.resolve()
        // Extract selection text (if any). AX cursor is the start; selectionLength is the length.
        let selection: String = {
            guard ctx.selectionLength > 0, !ctx.text.isEmpty else { return "" }
            let chars = Array(ctx.text)
            let start = max(0, min(ctx.cursor, chars.count))
            let end = max(start, min(start + ctx.selectionLength, chars.count))
            return String(chars[start..<end])
        }()

        let input = CommandInput(
            command: command,
            parameter: parameter,
            selection: selection,
            fieldText: ctx.text,
            fieldSource: ctx.source.rawValue
        )

        guard let result = await voiceFlow.runCommand(input) else {
            NSLog("[VoiceFlow] AI command: LLM call failed")
            return false
        }
        if result.abstained || result.output.isEmpty {
            NSLog("[VoiceFlow] AI command: model abstained")
            return false
        }

        // Paste: if a selection was active, replace it; otherwise insert at cursor.
        let pasteboard = NSPasteboard.general
        let saved = pasteboard.pasteboardItems?.compactMap { item -> (NSPasteboard.PasteboardType, Data)? in
            for type in item.types {
                if let data = item.data(forType: type) {
                    return (type, data)
                }
            }
            return nil
        } ?? []
        pasteboard.clearContents()
        pasteboard.setString(result.output, forType: .string)

        // If selection was present, ensure it's still selected (the existing
        // selection will be replaced by the paste).
        AppDelegate.paste()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            pasteboard.clearContents()
            for (t, d) in saved {
                pasteboard.setData(d, forType: t)
            }
        }

        // Update shadow buffer
        ShadowBuffer.shared.record(
            bundleId: ctx.bundleId,
            windowKey: ctx.windowKey,
            cursorAtPaste: ctx.cursor,
            inserted: result.output
        )

        RetroactiveToast.shared.show(
            "\(commandLabel(command)) applied",
            source: ctx.source
        )
        NSLog("[VoiceFlow] AI command: APPLIED (%d chars)", result.output.count)
        return true
    }

    private func commandLabel(_ c: CommandKind) -> String {
        switch c {
        case .rewrite: return "Rewrite"
        case .proofread: return "Proofread"
        case .shorten: return "Shorten"
        case .bullet: return "Bullet"
        case .continue: return "Continue"
        case .summarize: return "Summarize"
        case .reply: return "Reply"
        case .explain: return "Explain"
        case .draft: return "Draft"
        case .question: return "Answer"
        }
    }

    /// Static paste entry point so non-AppDelegate code can request a paste
    /// without taking an AppDelegate reference.
    ///
    /// Uses CGEvent directly — NOT AppleScript/System Events, which needs the
    /// Automation TCC permission onboarding never requests and fails with -1743
    /// on a clean install. CGEvent only needs Accessibility. (Mirrors
    /// AppDelegate.simulatePaste.)
    @MainActor
    static func paste() {
        let source = CGEventSource(stateID: .combinedSessionState)
        let down = CGEvent(keyboardEventSource: source, virtualKey: 9, keyDown: true)
        down?.flags = .maskCommand
        let up = CGEvent(keyboardEventSource: source, virtualKey: 9, keyDown: false)
        up?.flags = .maskCommand
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)
    }
}

// MARK: - Confirmation toast

/// Small floating notification confirming a retroactive edit. Auto-dismisses
/// after 2.5s. Click anywhere to dismiss immediately.
@MainActor
final class RetroactiveToast {
    static let shared = RetroactiveToast()

    private var window: NSPanel?
    private var dismissTimer: Timer?

    func show(_ message: String, source: FieldContext.Source) {
        // Dismiss any existing toast first
        dismiss(animated: false)

        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 320, height: 56),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.hasShadow = true
        panel.backgroundColor = .clear
        panel.isMovableByWindowBackground = false
        panel.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle]

        let container = NSView(frame: panel.contentView!.bounds)
        container.wantsLayer = true
        container.layer?.cornerRadius = 12
        container.layer?.masksToBounds = true
        container.layer?.backgroundColor = NSColor.windowBackgroundColor.withAlphaComponent(0.94).cgColor

        let visualEffect = NSVisualEffectView(frame: container.bounds)
        visualEffect.autoresizingMask = [.width, .height]
        visualEffect.material = .hudWindow
        visualEffect.blendingMode = .behindWindow
        visualEffect.state = .active
        container.addSubview(visualEffect)

        let label = NSTextField(labelWithString: "Corrected: \(message)")
        label.font = .systemFont(ofSize: 13, weight: .medium)
        label.textColor = .labelColor
        label.lineBreakMode = .byTruncatingTail
        label.frame = NSRect(x: 14, y: 24, width: 292, height: 18)
        container.addSubview(label)

        let sub = NSTextField(labelWithString: "via \(source.rawValue) · click to dismiss")
        sub.font = .systemFont(ofSize: 11)
        sub.textColor = .secondaryLabelColor
        sub.frame = NSRect(x: 14, y: 6, width: 292, height: 14)
        container.addSubview(sub)

        let click = NSClickGestureRecognizer(target: self, action: #selector(handleClick))
        container.addGestureRecognizer(click)

        panel.contentView = container

        // Position near top-center of main screen
        if let screen = NSScreen.main {
            let frame = screen.visibleFrame
            let origin = NSPoint(
                x: frame.midX - panel.frame.width / 2,
                y: frame.maxY - panel.frame.height - 80
            )
            panel.setFrameOrigin(origin)
        }

        panel.alphaValue = 0
        panel.orderFront(nil)
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.18
            panel.animator().alphaValue = 1
        }

        self.window = panel
        dismissTimer = Timer.scheduledTimer(withTimeInterval: 2.5, repeats: false) { [weak self] _ in
            DispatchQueue.main.async { self?.dismiss(animated: true) }
        }
    }

    @objc private func handleClick() {
        dismiss(animated: true)
    }

    func dismiss(animated: Bool) {
        dismissTimer?.invalidate()
        dismissTimer = nil
        guard let panel = window else { return }
        self.window = nil
        if animated {
            NSAnimationContext.runAnimationGroup({ ctx in
                ctx.duration = 0.18
                panel.animator().alphaValue = 0
            }, completionHandler: {
                panel.orderOut(nil)
            })
        } else {
            panel.orderOut(nil)
        }
    }
}
