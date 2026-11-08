import AppKit
import Combine
import AVFoundation
import ApplicationServices

/// Session-only checkpoints. Never stores a recording or persists test text.
@MainActor
final class DictationDiagnostics: ObservableObject {
    static let shared = DictationDiagnostics()
    enum Stage: String, CaseIterable {
        case shortcut = "Shortcut received"
        case recording = "Recording started"
        case microphone = "Microphone signal"
        case speech = "Speech recognized"
        case formatting = "Text formatted"
        case insertion = "Test-field insertion verified"
    }
    enum Result: Equatable { case waiting, working, passed, failed(String) }
    @Published private(set) var results: [Stage: Result] = [:]
    @Published private(set) var registration: [UInt32: Int32] = [:]
    @Published var speechState = "Not started"
    @Published var formatterReady = false
    @Published var inputLevel: Float = 0
    @Published private(set) var active = false
    @Published private(set) var running = false
    @Published private(set) var hotkeyReceived = false
    @Published private(set) var verified = false
    @Published private(set) var slowStage = false
    @Published var recovering = false
    @Published var recoveryMessage: String?
    weak var field: NSTextView?
    var startRecording: (() -> Void)?
    var stopRecording: (() -> Void)?
    var cancelRecording: (() -> Void)?
    private var discardOutput = false
    var restartSpeech: (() async -> Void)?
    private var awaitingText: String?
    private var beforePaste = ""
    private var progressAt = Date()
    private var currentStage: Stage?

    var passed: Bool { hotkeyReceived && verified }
    var canRestartSpeech: Bool { !running || currentStage == .speech }
    var recording: Bool { results[.recording] == .passed && currentStage == .microphone }

    func begin() {
        active = true
        reset()
    }
    func end() {
        discardOutput = running
        if recording {
            cancelRecording?()
            running = false
        }
        active = false
        awaitingText = nil
        field = nil
    }
    func reset() {
        guard !running else { return }
        results = Dictionary(uniqueKeysWithValues: Stage.allCases.map { ($0, .waiting) })
        hotkeyReceived = false
        verified = false
        slowStage = false
        awaitingText = nil
        currentStage = nil
        recoveryMessage = nil
        discardOutput = false
    }
    func registered(id: UInt32, status: Int32) { registration[id] = status }
    func unregister(id: UInt32) { registration.removeValue(forKey: id) }
    func receivedHotkey(id: UInt32) {
        guard active, id == 1, let field, field.window?.firstResponder === field else { return }
        hotkeyReceived = true
        results[.shortcut] = .passed
    }
    func captureStarted() {
        guard active, let field, field.window?.firstResponder === field else { return }
        running = true
        verified = false
        awaitingText = nil
        for stage in Stage.allCases where stage != .shortcut { results[stage] = .waiting }
        results[.recording] = .passed
        advance(.microphone)
    }
    func audioLevel(_ level: Float) {
        guard active else { return }
        inputLevel = level.isFinite ? min(1, max(0, level)) : 0
        if running, currentStage == .microphone, inputLevel > 0.015 { results[.microphone] = .passed }
    }
    func transcribing() {
        guard active, running else { return }
        advance(.speech)
    }
    func recognized(_ text: String) {
        guard running else { return }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            fail(.speech, "No speech was recognized. Hold the shortcut longer and check the microphone meter.")
            return
        }
        results[.speech] = .passed
        advance(.formatting)
    }
    func failedSpeech(_ message: String) { fail(.speech, message) }
    func formatted(success: Bool) {
        guard running else { return }
        if success { results[.formatting] = .passed; advance(.insertion) }
        else { fail(.formatting, "Speech was recognized, but formatting failed. Check the writing service and open Logs.") }
    }
    /// Only diagnostics may paste into this specific field. A sent key event is
    /// never counted as success; the field delegate must observe the result.
    func preparePaste(_ text: String, applicationIsActive: Bool? = nil) -> Bool {
        if discardOutput {
            discardOutput = false
            running = false
            awaitingText = nil
            return false
        }
        guard active, running else { return true }
        guard currentStage == .insertion else { return false }
        guard applicationIsActive ?? NSApp.isActive, let field, field.window?.firstResponder === field else {
            fail(.insertion, "The practice field lost focus. Your result remains on the clipboard. Click the field and retry.")
            return false
        }
        beforePaste = field.string
        awaitingText = text
        advance(.insertion)
        return true
    }
    func fieldChanged(_ text: String) {
        guard active, running, let expected = awaitingText, !expected.isEmpty,
              text != beforePaste, text.contains(expected) else { return }
        results[.insertion] = .passed
        verified = true
        running = false
        awaitingText = nil
        currentStage = nil
    }
    func captureFailed(_ message: String) {
        guard active else { return }
        results[.recording] = .failed(message)
        running = false
        currentStage = nil
        awaitingText = nil
    }
    func insertionFailed(_ message: String) { fail(.insertion, message) }
    func tick(now: Date = Date()) {
        slowStage = running && now.timeIntervalSince(progressAt) > 30
    }
    func focusField() {
        guard let field, let window = field.window else { return }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        window.makeFirstResponder(field)
    }
    var summary: String {
        let lines = Stage.allCases.map { stage -> String in
            let value: String
            switch results[stage] ?? .waiting {
            case .waiting: value = "not reached"
            case .working: value = "in progress"
            case .passed: value = "passed"
            case .failed(let message): value = "failed: \(message)"
            }
            return "\(stage.rawValue): \(value)"
        }
        return (["VoiceFlow dictation check", "Accessibility: \(AXIsProcessTrusted())",
                 "Microphone authorized: \(AVCaptureDevice.authorizationStatus(for: .audio) == .authorized)",
                 "Dictate registration status: \(registration[1].map(String.init) ?? "not registered")",
                 "Speech service: \(speechState)", "Writing service ready: \(formatterReady)"] + lines).joined(separator: "\n")
    }
    private func advance(_ stage: Stage) {
        currentStage = stage
        progressAt = Date()
        slowStage = false
        if results[stage] != .passed { results[stage] = .working }
    }
    private func fail(_ stage: Stage, _ message: String) {
        guard running else { return }
        results[stage] = .failed(message)
        discardOutput = false
        running = false
        awaitingText = nil
        currentStage = nil
    }
}
