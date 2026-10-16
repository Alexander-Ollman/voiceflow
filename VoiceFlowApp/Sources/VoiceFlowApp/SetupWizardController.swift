import AppKit
import SwiftUI

// MARK: - Setup Wizard Controller

/// Manages the NSWindow lifecycle for the first-run setup wizard.
/// Retained by AppDelegate until wizard completion.
@MainActor
final class SetupWizardController {
    private var window: NSWindow?
    private let state = SetupWizardState()
    private var onComplete: (() -> Void)?
    private let preparePractice: () -> Void

    init(preparePractice: @escaping () -> Void = {}) {
        self.preparePractice = preparePractice
    }

    static let setupCompleteKey = "voiceflow.setupWizardCompleted"

    static var isSetupComplete: Bool {
        UserDefaults.standard.bool(forKey: setupCompleteKey)
    }

    func show(onComplete: @escaping () -> Void) {
        self.onComplete = onComplete

        let wizardView = SetupWizardView(state: state, onComplete: { [weak self] in
            self?.completeSetup()
        }, preparePractice: preparePractice)

        let hostingController = NSHostingController(rootView: wizardView)

        let window = NSWindow(contentViewController: hostingController)
        window.title = "VoiceFlow Setup"
        window.styleMask = [.titled, .fullSizeContentView]
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.isMovableByWindowBackground = true
        window.setContentSize(NSSize(width: 620, height: 700))
        window.center()
        window.isReleasedWhenClosed = false

        // Prevent close button from dismissing without completing
        window.standardWindowButton(.closeButton)?.isHidden = true
        window.standardWindowButton(.miniaturizeButton)?.isHidden = true
        window.standardWindowButton(.zoomButton)?.isHidden = true

        self.window = window

        // Switch to regular activation policy so the wizard window and any
        // system permission dialogs (microphone, etc.) can appear properly.
        // LSUIElement / .accessory apps can't present system dialogs.
        NSApp.setActivationPolicy(.regular)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func completeSetup() {
        UserDefaults.standard.set(true, forKey: Self.setupCompleteKey)
        window?.close()
        window = nil

        // Restore menu-bar-only mode after wizard closes
        ApplicationVisibility.apply()

        onComplete?()
        onComplete = nil
    }
}
