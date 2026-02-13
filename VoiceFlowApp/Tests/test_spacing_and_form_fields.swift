#!/usr/bin/env swift
// Standalone test script for spacing mode fixes and trailing-period stripping.
// Run: swift VoiceFlowApp/Tests/test_spacing_and_form_fields.swift

import Foundation

// ─── Duplicated types under test ───

enum CursorContextResult {
    case character(Character)
    case atStart
    case unavailable
}

enum SpacingMode: String {
    case contextAware = "contextAware"
    case smart = "smart"
    case always = "always"
    case trailing = "trailing"

    /// Simulated apply() matching the real implementation after the fix
    func apply(to text: String, cursorResult: CursorContextResult = .unavailable) -> String {
        switch self {
        case .contextAware:
            switch cursorResult {
            case .character(let charBefore):
                if !charBefore.isWhitespace && !charBefore.isNewline {
                    if let first = text.first, first.isLetter || first.isNumber {
                        return " " + text
                    }
                }
                return text
            case .atStart:
                return text
            case .unavailable:
                // FIX: Don't add leading space when AX unavailable.
                // Trailing space from previous dictation handles separation.
                return text
            }
        case .smart:
            if let first = text.first, first.isLetter {
                return " " + text
            }
            return text
        case .always:
            return " " + text
        case .trailing:
            return text + " "
        }
    }
}

/// Simulates the trailing-period stripping logic from stopRecordingAndPaste
func stripTrailingPeriodIfNeeded(text: String, isEmptyField: Bool) -> String {
    guard isEmptyField else { return text }
    let trimmed = text.trimmingCharacters(in: .whitespaces)
    let wordCount = trimmed.split(separator: " ").count
    let hasInternalPeriod = trimmed.dropLast().contains(".")
    if wordCount <= 8 && !hasInternalPeriod && trimmed.hasSuffix(".") {
        return String(trimmed.dropLast())
    }
    return text
}

/// Simulates the trailing space addition from stopRecordingAndPaste
func addTrailingSpace(_ text: String) -> String {
    var result = text
    if !result.isEmpty && !result.hasSuffix(" ") && !result.hasSuffix("\n") {
        result += " "
    }
    return result
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
        print("  FAIL [\(line)]: \(message) — got \"\(a)\", expected \"\(b)\"")
    }
}

func section(_ name: String) {
    print("--- \(name) ---")
}

// ═══════════════════════════════════════════════════════
// SPACING TESTS
// ═══════════════════════════════════════════════════════

section("contextAware: .unavailable no longer adds leading space")

assertEqual(
    SpacingMode.contextAware.apply(to: "Hello", cursorResult: .unavailable),
    "Hello",
    "unavailable: no leading space on letter-starting text"
)

assertEqual(
    SpacingMode.contextAware.apply(to: "94117", cursorResult: .unavailable),
    "94117",
    "unavailable: no leading space on digit-starting text"
)

assertEqual(
    SpacingMode.contextAware.apply(to: "!important", cursorResult: .unavailable),
    "!important",
    "unavailable: no leading space on punctuation-starting text"
)

section("contextAware: .atStart still works correctly")

assertEqual(
    SpacingMode.contextAware.apply(to: "Hello", cursorResult: .atStart),
    "Hello",
    "atStart: no leading space"
)

section("contextAware: .character still adds space after non-whitespace")

assertEqual(
    SpacingMode.contextAware.apply(to: "World", cursorResult: .character("d")),
    " World",
    "after letter: leading space added"
)

assertEqual(
    SpacingMode.contextAware.apply(to: "World", cursorResult: .character(" ")),
    "World",
    "after space: no leading space"
)

assertEqual(
    SpacingMode.contextAware.apply(to: "World", cursorResult: .character("\n")),
    "World",
    "after newline: no leading space"
)

section("Consecutive dictations with trailing space (unavailable AX)")

// Simulate: empty field → first dictation → second dictation
let first = addTrailingSpace(SpacingMode.contextAware.apply(to: "Hello", cursorResult: .unavailable))
assertEqual(first, "Hello ", "first dictation: 'Hello' + trailing space")

// Second dictation — AX still unavailable, no leading space added
let second = addTrailingSpace(SpacingMode.contextAware.apply(to: "World", cursorResult: .unavailable))
assertEqual(second, "World ", "second dictation: 'World' + trailing space")

// Combined result
assertEqual(first + second, "Hello World ", "consecutive: properly separated by single space")

section("smart mode: unchanged behavior")

assertEqual(
    SpacingMode.smart.apply(to: "Hello"),
    " Hello",
    "smart: still adds leading space for letters"
)
assertEqual(
    SpacingMode.smart.apply(to: "94117"),
    "94117",
    "smart: no leading space for digits"
)

// ═══════════════════════════════════════════════════════
// TRAILING PERIOD STRIPPING
// ═══════════════════════════════════════════════════════

section("Empty field: trailing period stripped from short entries")

assertEqual(
    stripTrailingPeriodIfNeeded(text: "John Smith.", isEmptyField: true),
    "John Smith",
    "name with period → stripped"
)

assertEqual(
    stripTrailingPeriodIfNeeded(text: "94117.", isEmptyField: true),
    "94117",
    "zip code with period → stripped"
)

assertEqual(
    stripTrailingPeriodIfNeeded(text: "hello@example.com.", isEmptyField: true),
    "hello@example.com.",
    "email with internal period → NOT stripped (has internal periods)"
)

assertEqual(
    stripTrailingPeriodIfNeeded(text: "San Francisco.", isEmptyField: true),
    "San Francisco",
    "city name with period → stripped"
)

assertEqual(
    stripTrailingPeriodIfNeeded(text: "I think we should go to the store and pick up some groceries for dinner tonight.", isEmptyField: true),
    "I think we should go to the store and pick up some groceries for dinner tonight.",
    "long sentence (>8 words) → NOT stripped"
)

section("Non-empty field: period NOT stripped")

assertEqual(
    stripTrailingPeriodIfNeeded(text: "John Smith.", isEmptyField: false),
    "John Smith.",
    "non-empty field: period preserved"
)

section("Short sentences in empty fields")

assertEqual(
    stripTrailingPeriodIfNeeded(text: "Yes.", isEmptyField: true),
    "Yes",
    "single word with period → stripped"
)

assertEqual(
    stripTrailingPeriodIfNeeded(text: "Thank you very much.", isEmptyField: true),
    "Thank you very much",
    "4-word phrase with period → stripped"
)

section("No period: unchanged")

assertEqual(
    stripTrailingPeriodIfNeeded(text: "John Smith", isEmptyField: true),
    "John Smith",
    "no trailing period → unchanged"
)

assertEqual(
    stripTrailingPeriodIfNeeded(text: "Hello!", isEmptyField: true),
    "Hello!",
    "exclamation mark → unchanged (only strips periods)"
)

section("URL/dotted entries preserved")

assertEqual(
    stripTrailingPeriodIfNeeded(text: "www.google.com.", isEmptyField: true),
    "www.google.com.",
    "URL with trailing period → NOT stripped (has internal periods)"
)

assertEqual(
    stripTrailingPeriodIfNeeded(text: "v2.1.0.", isEmptyField: true),
    "v2.1.0.",
    "version with trailing period → NOT stripped (has internal periods)"
)

// ═══════════════════════════════════════════════════════
// END-TO-END FORM FIELD SIMULATION
// ═══════════════════════════════════════════════════════

section("End-to-end: empty browser form field")

// Simulate: user dictates "John Smith" into an empty browser form field
// AX unavailable (browser), field is empty, LLM outputs "John Smith."
var output = "John Smith."
output = stripTrailingPeriodIfNeeded(text: output, isEmptyField: true)
output = SpacingMode.contextAware.apply(to: output, cursorResult: .unavailable)
output = addTrailingSpace(output)
assertEqual(output, "John Smith ", "form field: no leading space, period stripped, trailing space")

// Simulate: user dictates zip code "94117" into empty field
var zipOutput = "94117."
zipOutput = stripTrailingPeriodIfNeeded(text: zipOutput, isEmptyField: true)
zipOutput = SpacingMode.contextAware.apply(to: zipOutput, cursorResult: .unavailable)
zipOutput = addTrailingSpace(zipOutput)
assertEqual(zipOutput, "94117 ", "zip code field: no leading space, period stripped")

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
