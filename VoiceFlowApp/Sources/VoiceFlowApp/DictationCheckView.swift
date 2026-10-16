import SwiftUI
import AppKit
import AVFoundation
import ApplicationServices

/// The same repeatable check is available in onboarding and Settings.
struct DictationCheckView: View {
    @ObservedObject private var diagnostics = DictationDiagnostics.shared
    @ObservedObject private var hotkeys = HotkeyStore.shared
    @State private var accessibility = false
    @State private var microphone = false
    @State private var showLogs = false
    @State private var showHotkeys = false
    var onContinue: (() -> Void)? = nil

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Test your setup").font(.title2.weight(.semibold))
                Text("Click the field, hold \(HotkeyFormatter.display(hotkeys.binding(for: .dictate))), say a short sentence, then release. Each checkpoint shows where output stops.")
                    .font(.callout).foregroundStyle(.secondary)

                HStack {
                    Label(accessibility ? "Accessibility allowed" : "Accessibility required", systemImage: accessibility ? "checkmark.circle" : "exclamationmark.triangle")
                    Spacer()
                    Label(microphone ? "Microphone allowed" : "Microphone required", systemImage: microphone ? "checkmark.circle" : "exclamationmark.triangle")
                }.font(.caption)
                Text("Speech: \(diagnostics.speechState) · Writing: \(diagnostics.formatterReady ? "Ready" : "Starting or unavailable")")
                    .font(.caption).foregroundStyle(.secondary)

                if let status = diagnostics.registration[1], status != 0 {
                    Text("Your Dictate shortcut could not be registered (status \(status)). Change it here and check ChatGPT or other apps for the same shortcut.")
                        .foregroundStyle(.orange).font(.callout)
                } else if !diagnostics.hotkeyReceived {
                    Text("A registered shortcut may still be intercepted by another app. If no first checkmark appears, choose a different shortcut in both apps.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                DiagnosticPracticeField(diagnostics: diagnostics)
                    .frame(height: 80)
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.3)))
                    .accessibilityLabel("Dictation test field")
                ProgressView(value: Double(diagnostics.inputLevel))
                    .accessibilityLabel("Live microphone level")

                ForEach(DictationDiagnostics.Stage.allCases, id: \.self) { stage in
                    HStack(alignment: .top) {
                        Image(systemName: icon(diagnostics.results[stage] ?? .waiting))
                        VStack(alignment: .leading, spacing: 2) {
                            Text(stage.rawValue)
                            if case .failed(let message) = diagnostics.results[stage] { Text(message).font(.caption).foregroundStyle(.orange) }
                        }
                        Spacer()
                    }.font(.callout)
                }
                if diagnostics.slowStage {
                    Text("This stage has taken more than 30 seconds. Open Logs or restart the speech service if recognition is stuck. Accessibility permission does not guarantee that the microphone or models are ready.")
                        .font(.caption).foregroundStyle(.orange)
                }
                HStack {
                    Button(diagnostics.recording ? "Stop microphone test" : "Test without shortcut") {
                        diagnostics.focusField()
                        if diagnostics.recording { diagnostics.stopRecording?() }
                        else { diagnostics.startRecording?() }
                    }
                    .disabled(diagnostics.recovering || !accessibility || !microphone || (diagnostics.running && !diagnostics.recording))
                    Button("Change shortcut") { showHotkeys.toggle() }.disabled(diagnostics.running)
                    Button("Open Logs") { showLogs = true }
                }
                if showHotkeys { HotkeyRecorderList() }
                HStack {
                    Button("Restart speech service") {
                        diagnostics.recovering = true
                        Task { @MainActor in
                            await diagnostics.restartSpeech?()
                            diagnostics.recovering = false
                        }
                    }.disabled(diagnostics.recovering || !diagnostics.canRestartSpeech)
                    Button("Copy check summary") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(diagnostics.summary, forType: .string)
                    }
                    Button("Reset check") { diagnostics.reset() }.disabled(diagnostics.running)
                }
                if diagnostics.recovering { ProgressView("Restarting speech service…").controlSize(.small) }
                if let message = diagnostics.recoveryMessage { Text(message).font(.caption).foregroundStyle(.secondary) }
                if !accessibility || !microphone {
                    HStack {
                        AccessibilityPermissionButton()
                        MicrophonePermissionButton()
                        Button("Open permission settings") {
                            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!)
                        }
                    }
                }
                Text("The button bypasses shortcut delivery to isolate microphone, recognition, formatting and insertion. Finishing onboarding requires a real shortcut and verified insertion. Check summaries exclude test text and audio.")
                    .font(.caption).foregroundStyle(.secondary)
                if let onContinue {
                    HStack {
                        Spacer()
                        Button("Continue") { onContinue() }
                            .buttonStyle(.borderedProminent).disabled(!diagnostics.passed)
                    }
                }
            }.padding(24)
        }
        .onAppear { diagnostics.begin() }
        .onDisappear { diagnostics.end() }
        .task {
            while !Task.isCancelled {
                accessibility = AXIsProcessTrusted()
                microphone = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
                diagnostics.tick()
                do { try await Task.sleep(nanoseconds: 500_000_000) } catch { break }
            }
        }
        .sheet(isPresented: $showLogs) { LogConsoleView() }
    }
    private func icon(_ result: DictationDiagnostics.Result) -> String {
        switch result {
        case .waiting: return "circle"
        case .working: return "ellipsis.circle"
        case .passed: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        }
    }
}

private struct DiagnosticPracticeField: NSViewRepresentable {
    let diagnostics: DictationDiagnostics
    func makeCoordinator() -> Coordinator { Coordinator(diagnostics) }
    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 540, height: 80))
        let text = NSTextView(frame: scroll.contentView.bounds)
        text.isRichText = false
        text.isAutomaticSpellingCorrectionEnabled = false
        text.font = .systemFont(ofSize: 13)
        text.textContainerInset = NSSize(width: 8, height: 8)
        text.autoresizingMask = [.width]
        text.isVerticallyResizable = true
        text.textContainer?.widthTracksTextView = true
        text.setAccessibilityIdentifier("voiceflow.dictation-check.field")
        text.delegate = context.coordinator
        scroll.documentView = text
        scroll.hasVerticalScroller = true
        diagnostics.field = text
        return scroll
    }
    func updateNSView(_ nsView: NSScrollView, context: Context) {
        if let text = nsView.documentView as? NSTextView { diagnostics.field = text }
    }
    final class Coordinator: NSObject, NSTextViewDelegate {
        let diagnostics: DictationDiagnostics
        init(_ diagnostics: DictationDiagnostics) { self.diagnostics = diagnostics }
        func textDidChange(_ notification: Notification) {
            guard let text = notification.object as? NSTextView else { return }
            diagnostics.fieldChanged(text.string)
        }
    }
}
