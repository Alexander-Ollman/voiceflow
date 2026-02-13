#!/usr/bin/env swift
// Standalone test script for AICommand parsing, computed properties, and AIResultState.
// Run: swift VoiceFlowApp/Tests/test_ai_commands.swift

import Foundation

// ─── Duplicated types under test (pure logic, no FFI deps) ───

enum AIPasteMode: Equatable {
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

// ─── Minimal AIResultState (no Combine, standalone) ───

class AIResultState {
    var isVisible = false
    var title = ""
    var resultText = ""
    var pasteHint = ""

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

// ─── Test harness ───

var passed = 0
var failed = 0
var totalAssertions = 0

func assert(_ condition: Bool, _ message: String, file: String = #file, line: Int = #line) {
    totalAssertions += 1
    if condition {
        passed += 1
    } else {
        failed += 1
        print("  FAIL [\(line)]: \(message)")
    }
}

func assertEqual<T: Equatable>(_ a: T, _ b: T, _ message: String, file: String = #file, line: Int = #line) {
    totalAssertions += 1
    if a == b {
        passed += 1
    } else {
        failed += 1
        print("  FAIL [\(line)]: \(message) — got \(a), expected \(b)")
    }
}

func section(_ name: String) {
    print("--- \(name) ---")
}

// ═══════════════════════════════════════════════════════
// TEST SUITE
// ═══════════════════════════════════════════════════════

// ── 1. Reply parsing ──

section("Reply: trigger phrases")

if case .reply(let intent) = AICommand.parse("reply to this saying sounds good let's do Friday")! {
    assertEqual(intent, "sounds good let's do friday", "reply to this saying — captures intent")
} else { assert(false, "reply to this saying — should parse as .reply") }

if case .reply(let intent) = AICommand.parse("Reply to this with I'll be there at 3")! {
    assertEqual(intent, "i'll be there at 3", "reply to this with — captures intent (case-insensitive)")
} else { assert(false, "reply to this with — should parse as .reply") }

if case .reply(let intent) = AICommand.parse("reply saying thanks for the update")! {
    assertEqual(intent, "thanks for the update", "reply saying — captures intent")
} else { assert(false, "reply saying — should parse as .reply") }

if case .reply(let intent) = AICommand.parse("reply with no problem")! {
    assertEqual(intent, "no problem", "reply with — captures intent")
} else { assert(false, "reply with — should parse as .reply") }

section("Reply: trailing period stripped")

if case .reply(let intent) = AICommand.parse("reply to this saying sure thing.")! {
    assertEqual(intent, "sure thing", "reply — trailing period stripped from transcript")
} else { assert(false, "reply with period — should parse as .reply") }

section("Reply: empty intent rejected")

assert(AICommand.parse("reply to this saying ") == nil, "reply with empty intent returns nil")
assert(AICommand.parse("reply saying") == nil, "reply saying with no intent returns nil")

// ── 2. Rewrite parsing ──

section("Rewrite: 'make this' variants")

if case .rewrite(let style) = AICommand.parse("make this more formal")! {
    assertEqual(style, "formal", "make this more formal — style=formal")
} else { assert(false, "make this more formal — should parse as .rewrite") }

if case .rewrite(let style) = AICommand.parse("make this concise")! {
    assertEqual(style, "concise", "make this concise — style=concise")
} else { assert(false, "make this concise — should parse as .rewrite") }

if case .rewrite(let style) = AICommand.parse("make this more casual")! {
    assertEqual(style, "casual", "make this more casual — style=casual")
} else { assert(false, "make this more casual — should parse as .rewrite") }

if case .rewrite(let style) = AICommand.parse("Make this more professional.")! {
    assertEqual(style, "professional", "case-insensitive + period stripped")
} else { assert(false, "Make this more professional. — should parse") }

section("Rewrite: 'rewrite as' variants")

if case .rewrite(let style) = AICommand.parse("rewrite as bullet points")! {
    assertEqual(style, "bullet points", "rewrite as bullet points")
} else { assert(false, "rewrite as bullet points — should parse as .rewrite") }

if case .rewrite(let style) = AICommand.parse("rewrite this as an email")! {
    assertEqual(style, "an email", "rewrite this as an email")
} else { assert(false, "rewrite this as an email — should parse as .rewrite") }

section("Rewrite: edge cases")

assert(AICommand.parse("make this ") == nil, "make this with no style returns nil")
assert(AICommand.parse("rewrite as ") == nil, "rewrite as with no style returns nil")
assert(AICommand.parse("rewrite") == nil, "bare 'rewrite' returns nil")

// ── 3. Proofread parsing ──

section("Proofread: trigger phrases")

if case .proofread = AICommand.parse("proofread this")! {
    assert(true, "proofread this — parses")
} else { assert(false, "proofread this — should parse as .proofread") }

if case .proofread = AICommand.parse("proofread")! {
    assert(true, "proofread — parses")
} else { assert(false, "proofread — should parse as .proofread") }

if case .proofread = AICommand.parse("Proofread this.")! {
    assert(true, "Proofread this. — case + period handled")
} else { assert(false, "Proofread this. — should parse as .proofread") }

// ── 4. Continue parsing ──

section("Continue: trigger phrases")

if case .continueWriting = AICommand.parse("continue this")! {
    assert(true, "continue this")
} else { assert(false, "continue this — should parse as .continueWriting") }

if case .continueWriting = AICommand.parse("continue writing")! {
    assert(true, "continue writing")
} else { assert(false, "continue writing — should parse as .continueWriting") }

if case .continueWriting = AICommand.parse("keep going")! {
    assert(true, "keep going")
} else { assert(false, "keep going — should parse as .continueWriting") }

if case .continueWriting = AICommand.parse("keep writing")! {
    assert(true, "keep writing")
} else { assert(false, "keep writing")
}

if case .continueWriting = AICommand.parse("Continue this.")! {
    assert(true, "Continue this. — case + period handled")
} else { assert(false, "Continue this. — should parse as .continueWriting") }

// ── 5. Non-commands return nil ──

section("Non-commands return nil")

assert(AICommand.parse("hello world") == nil, "random text returns nil")
assert(AICommand.parse("summarize this") == nil, "summarize is handled separately, not here")
assert(AICommand.parse("please reply to the email") == nil, "partial match doesn't trigger")
assert(AICommand.parse("") == nil, "empty string returns nil")
assert(AICommand.parse("   ") == nil, "whitespace returns nil")
assert(AICommand.parse("I want to rewrite my essay") == nil, "'rewrite' in middle of sentence doesn't trigger")
assert(AICommand.parse("can you proofread this for me") == nil, "'proofread this' not at start")

// ── 6. Paste mode computed property ──

section("Paste modes")

assertEqual(AICommand.reply(intent: "hi").pasteMode, .atCursor, "reply → atCursor")
assertEqual(AICommand.rewrite(style: "formal").pasteMode, .replaceAll, "rewrite → replaceAll")
assertEqual(AICommand.proofread.pasteMode, .replaceAll, "proofread → replaceAll")
assertEqual(AICommand.continueWriting.pasteMode, .appendToEnd, "continue → appendToEnd")

// ── 7. Processing labels ──

section("Processing labels")

assertEqual(AICommand.reply(intent: "hi").processingLabel, "Composing reply...", "reply label")
assertEqual(AICommand.rewrite(style: "formal").processingLabel, "Rewriting (formal)...", "rewrite label")
assertEqual(AICommand.proofread.processingLabel, "Proofreading...", "proofread label")
assertEqual(AICommand.continueWriting.processingLabel, "Continuing...", "continue label")

// ── 8. Result titles ──

section("Result titles")

assertEqual(AICommand.reply(intent: "hi").resultTitle, "Reply", "reply title")
assertEqual(AICommand.rewrite(style: "concise").resultTitle, "Rewrite (concise)", "rewrite title")
assertEqual(AICommand.proofread.resultTitle, "Proofread", "proofread title")
assertEqual(AICommand.continueWriting.resultTitle, "Continue", "continue title")

// ── 9. Paste hints ──

section("Paste hints")

assertEqual(AICommand.reply(intent: "hi").pasteHint, "Enter to paste at cursor", "reply hint")
assertEqual(AICommand.rewrite(style: "formal").pasteHint, "Enter to replace all", "rewrite hint")
assertEqual(AICommand.proofread.pasteHint, "Enter to replace all", "proofread hint")
assertEqual(AICommand.continueWriting.pasteHint, "Enter to append to end", "continue hint")

// ── 10. AIResultState ──

section("AIResultState: show/hide lifecycle")

let state = AIResultState()
assert(!state.isVisible, "initial state is not visible")
assertEqual(state.title, "", "initial title is empty")
assertEqual(state.resultText, "", "initial text is empty")

state.show(title: "Reply", text: "Sounds good!", hint: "Enter to paste at cursor")
assert(state.isVisible, "after show: isVisible = true")
assertEqual(state.title, "Reply", "after show: title set")
assertEqual(state.resultText, "Sounds good!", "after show: resultText set")
assertEqual(state.pasteHint, "Enter to paste at cursor", "after show: pasteHint set")

state.hide()
assert(!state.isVisible, "after hide: isVisible = false")
assertEqual(state.title, "", "after hide: title cleared")
assertEqual(state.resultText, "", "after hide: resultText cleared")
assertEqual(state.pasteHint, "", "after hide: pasteHint cleared")

section("AIResultState: multiple show calls")

state.show(title: "A", text: "first", hint: "h1")
state.show(title: "B", text: "second", hint: "h2")
assertEqual(state.title, "B", "second show overwrites title")
assertEqual(state.resultText, "second", "second show overwrites text")

state.hide()

// ── 11. Whitespace / edge-case parsing ──

section("Whitespace and edge-case parsing")

if case .reply(let intent) = AICommand.parse("  Reply to this saying  yes  ")! {
    assertEqual(intent, "yes", "leading/trailing whitespace stripped")
} else { assert(false, "whitespace-padded reply should parse") }

if case .proofread = AICommand.parse("\n  proofread this  \n")! {
    assert(true, "newline-padded proofread parses")
} else { assert(false, "newline-padded proofread should parse") }

if case .rewrite(let style) = AICommand.parse("make this more  friendly")! {
    // "more " is stripped, leaving " friendly" which gets trimmed
    assertEqual(style, "friendly", "extra spaces in 'make this more  friendly'")
} else { assert(false, "make this more  friendly — should parse") }

// ═══════════════════════════════════════════════════════
// RESULTS
// ═══════════════════════════════════════════════════════

print("")
print("═══════════════════════════════════")
if failed == 0 {
    print("ALL \(passed) ASSERTIONS PASSED")
} else {
    print("\(failed) FAILED, \(passed) passed (of \(totalAssertions) total)")
}
print("═══════════════════════════════════")

if failed > 0 { exit(1) }
