import SwiftUI
import AppKit
import AVFoundation
import AVFAudio
import ApplicationServices
import ServiceManagement

// MARK: - Wizard Step

enum WizardStep: Int, CaseIterable {
    case welcome = 0
    case permissions = 1
    case models = 2
    case done = 3
}

// MARK: - Setup Wizard View

struct SetupWizardView: View {
    @ObservedObject var state: SetupWizardState
    let onComplete: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Step indicator (hidden on welcome)
            if state.currentStep != .welcome {
                StepProgressIndicator(currentStep: state.currentStep)
                    .padding(.top, 20)
                    .padding(.bottom, 12)
            }

            // Step content
            Group {
                switch state.currentStep {
                case .welcome:
                    WelcomeStepView {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            state.currentStep = .permissions
                        }
                    }
                case .permissions:
                    PermissionsStepView {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            state.currentStep = .models
                        }
                    }
                case .models:
                    ModelDownloadStepView(modelManager: state.modelManager) {
                        withAnimation(.easeInOut(duration: 0.3)) {
                            state.currentStep = .done
                        }
                    }
                case .done:
                    DoneStepView(onComplete: onComplete)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(width: 620, height: 700)
        .background(Color(NSColor.windowBackgroundColor))
    }
}

// MARK: - Step Progress Indicator

struct StepProgressIndicator: View {
    let currentStep: WizardStep

    private let steps: [WizardStep] = [.permissions, .models, .done]

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(steps.enumerated()), id: \.element) { index, step in
                if index > 0 {
                    Rectangle()
                        .fill(step.rawValue <= currentStep.rawValue ? Color.accentColor : Color.secondary.opacity(0.3))
                        .frame(height: 2)
                        .frame(maxWidth: 60)
                }

                ZStack {
                    if step.rawValue < currentStep.rawValue {
                        Circle()
                            .fill(Color.accentColor)
                            .frame(width: 20, height: 20)
                        Image(systemName: "checkmark")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(.white)
                    } else if step == currentStep {
                        Circle()
                            .fill(Color.accentColor)
                            .frame(width: 20, height: 20)
                        Circle()
                            .fill(Color.white)
                            .frame(width: 8, height: 8)
                    } else {
                        Circle()
                            .stroke(Color.secondary.opacity(0.4), lineWidth: 2)
                            .frame(width: 20, height: 20)
                    }
                }
            }
        }
        .padding(.horizontal, 40)
    }
}

// MARK: - Step 1: Welcome

struct WelcomeStepView: View {
    let onNext: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            if let logoPath = Bundle.main.path(forResource: "AppLogo", ofType: "png"),
               let logoImage = NSImage(contentsOfFile: logoPath) {
                Image(nsImage: logoImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 96, height: 96)
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .shadow(color: .black.opacity(0.1), radius: 8, y: 4)
            } else {
                Image(systemName: "waveform.circle.fill")
                    .font(.system(size: 80))
                    .foregroundColor(.accentColor)
            }

            VStack(spacing: 8) {
                Text("Welcome to VoiceFlow")
                    .font(.system(size: 28, weight: .bold))

                Text("Dictate anywhere on your Mac with local AI")
                    .font(.body)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Button(action: onNext) {
                Text("Get Started")
                    .font(.headline)
                    .frame(maxWidth: 200)
                    .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            Spacer()
                .frame(height: 40)
        }
        .padding(.horizontal, 40)
    }
}

// MARK: - Step 2: Permissions

struct PermissionsStepView: View {
    let onNext: () -> Void

    @State private var micGranted: Bool = false
    @State private var accessibilityGranted: Bool = false
    @State private var pollTimer: Timer?

    var body: some View {
        VStack(spacing: 20) {
            Text("Permissions")
                .font(.title2.bold())
                .padding(.top, 16)

            Text("VoiceFlow needs these permissions to work properly.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            VStack(spacing: 12) {
                PermissionCard(
                    icon: "mic.fill",
                    iconColor: .red,
                    title: "Microphone",
                    description: "Required for recording your voice",
                    isGranted: micGranted,
                    action: {
                        NSApp.activate(ignoringOtherApps: true)
                        // Request via AVCaptureDevice first — with entitlements this
                        // triggers the native Allow/Cancel dialog and waits for the
                        // user's response before calling the completion handler.
                        AVCaptureDevice.requestAccess(for: .audio) { granted in
                            DispatchQueue.main.async {
                                micGranted = granted
                            }
                        }
                    }
                )

                PermissionCard(
                    icon: "hand.raised.fill",
                    iconColor: .blue,
                    title: "Accessibility",
                    description: "Required for auto-paste into any app",
                    isGranted: accessibilityGranted,
                    action: {
                        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
                        AXIsProcessTrustedWithOptions(options)
                    }
                )
            }
            .padding(.horizontal, 24)

            if !micGranted || !accessibilityGranted {
                HStack(spacing: 8) {
                    Image(systemName: "info.circle.fill")
                        .foregroundColor(.orange)
                    Text("You can grant these later from Settings")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(8)
            }

            Spacer()

            Button(action: onNext) {
                Text("Continue")
                    .font(.headline)
                    .frame(maxWidth: 200)
                    .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            Spacer()
                .frame(height: 24)
        }
        .padding(.horizontal, 24)
        .onAppear {
            checkPermissions()
            pollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                DispatchQueue.main.async {
                    checkPermissions()
                }
            }
        }
        .onDisappear {
            pollTimer?.invalidate()
            pollTimer = nil
        }
    }

    private func checkPermissions() {
        micGranted = AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
        accessibilityGranted = AXIsProcessTrusted()
    }
}

// MARK: - Permission Card

struct PermissionCard: View {
    let icon: String
    let iconColor: Color
    let title: String
    let description: String
    let isGranted: Bool
    let action: () -> Void

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 22))
                .foregroundColor(iconColor)
                .frame(width: 40, height: 40)
                .background(iconColor.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 10))

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.headline)
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            if isGranted {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 22))
                    .foregroundColor(.green)
            } else {
                Button("Grant", action: action)
                    .buttonStyle(.bordered)
            }
        }
        .padding(14)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(12)
    }
}

// MARK: - Step 3: Model Download (Profile-Based)

/// A user-facing preset that maps to concrete STT + LLM model choices behind the scenes.
struct ModelProfile: Identifiable {
    let id: String
    let name: String
    let icon: String
    let description: String
    let downloadSize: String
    let bestFor: String
    let sttId: String       // internal: moonshine model id or consolidated model id
    let sttType: SttType    // internal: which download path to use
    let llmId: String       // internal: LLM model id
    let requiresPython: Bool
    let minRAMGB: Int

    enum SttType {
        case moonshine(modelId: String) // e.g. "tiny"
        case consolidated(modelId: String) // e.g. "qwen3-asr-0.6b"
    }
}

struct ModelDownloadStepView: View {
    @ObservedObject var modelManager: ModelManager
    let onNext: () -> Void

    // SetupHelper drives the actual install — both onboarding wizard and the
    // Models settings page share this code path.
    @StateObject private var setup = SetupHelper()

    var body: some View {
        Group {
            if isInProgress {
                installingView
            } else {
                introView
            }
        }
        .padding(.top, 8)
        .onAppear { setup.refreshInstallState() }
    }

    // MARK: While downloading — slim status strip + a contained tips card.

    private var installingView: some View {
        VStack(spacing: 18) {
            // Zone 1 — status strip: what's happening, compact and top-anchored.
            VStack(spacing: 10) {
                HStack(spacing: 10) {
                    Image(systemName: "arrow.down.circle")
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.accentColor)
                    Text("Installing models")
                        .font(.headline)
                    Spacer()
                }
                if let frac = setup.progressFraction {
                    ProgressView(value: frac).progressViewStyle(.linear)
                } else {
                    ProgressView().progressViewStyle(.linear)
                }
                HStack {
                    Text(setup.phaseLabel)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(NSColor.controlBackgroundColor))
            )
            .padding(.horizontal, 32)
            .padding(.top, 8)

            // Zone 2 — tips card: what to read while you wait. Contained so it
            // reads as its own module and cycling never shifts the layout.
            QuickTipsCarousel(autoAdvance: true)
                .padding(.horizontal, 16)
                .padding(.vertical, 16)
                .background(
                    RoundedRectangle(cornerRadius: 16)
                        .fill(Color.primary.opacity(0.04))
                        .overlay(
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                        )
                )
                .padding(.horizontal, 32)

            Spacer(minLength: 0)

            // Quiet busy indicator — nothing is actionable while installing, so
            // just a spinner, no competing button or label.
            ProgressView()
                .controlSize(.small)
                .padding(.bottom, 28)
        }
    }

    // MARK: Pre-download / complete — header, rundown, and the action button.

    private var introView: some View {
        VStack(spacing: 20) {
            // Header
            VStack(spacing: 8) {
                Image(systemName: "shippingbox.and.arrow.backward")
                    .font(.system(size: 38))
                    .foregroundColor(.accentColor)
                Text("Install models")
                    .font(.title2.weight(.semibold))
                Text("VoiceFlow uses two on-device models — both run locally on Apple Silicon.")
                    .font(.callout)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, 8)

            // Rundown
            VStack(alignment: .leading, spacing: 10) {
                ModelRundownRow(
                    icon: "waveform",
                    title: "Speech model",
                    detail: "Transcribes your voice on-device",
                    size: "~1.2 GB",
                    installed: setup.parakeetInstalled
                )
                ModelRundownRow(
                    icon: "brain",
                    title: "Language model",
                    detail: "Formats and punctuates your dictation on-device",
                    size: "~1.1 GB",
                    installed: setup.bonsaiInstalled
                )
                Divider()
                Text("Total: about 2.3 GB. One-time download. Everything stays on your machine.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(NSColor.controlBackgroundColor))
            )
            .padding(.horizontal, 32)

            if case .failed(let msg) = setup.phase {
                Text(msg).font(.caption).foregroundColor(.red)
            }

            Spacer(minLength: 0)

            // Action
            HStack {
                Spacer()
                if setup.isFullyInstalled || setup.phase == .complete {
                    Button(action: onNext) {
                        Text("Continue")
                            .frame(minWidth: 140)
                    }
                    .keyboardShortcut(.defaultAction)
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
                } else {
                    Button(action: { Task { await setup.runSetup() } }) {
                        Text("Setup")
                            .frame(minWidth: 140)
                    }
                    .keyboardShortcut(.defaultAction)
                    .controlSize(.large)
                    .buttonStyle(.borderedProminent)
                }
                Spacer()
            }
            .padding(.bottom, 28)
        }
    }

    private var isInProgress: Bool {
        switch setup.phase {
        case .downloadingBonsai, .loadingParakeet: return true
        default: return false
        }
    }
}

struct ModelRundownRow: View {
    let icon: String
    let title: String
    let detail: String
    let size: String
    let installed: Bool

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 18))
                .foregroundColor(.accentColor)
                .frame(width: 30)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(title).font(.callout).fontWeight(.medium)
                    if installed {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .font(.system(size: 12))
                    }
                }
                Text(detail).font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            Text(size).font(.caption).foregroundColor(.secondary)
        }
    }
}


// MARK: - Profile Card

struct ProfileCard: View {
    let profile: ModelProfile
    let isSelected: Bool
    let isAvailable: Bool
    let unavailableReason: String?
    let isDownloaded: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 14) {
                // Icon
                Image(systemName: profile.icon)
                    .font(.system(size: 20))
                    .foregroundColor(.accentColor)
                    .frame(width: 40, height: 40)
                    .background(Color.accentColor.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                // Text
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(profile.name)
                            .font(.headline)
                            .foregroundColor(.primary)

                        if isDownloaded {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 12))
                                .foregroundColor(.green)
                        }
                    }
                    Text(profile.description)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                    HStack(spacing: 12) {
                        Label(profile.downloadSize, systemImage: "arrow.down.circle")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text(profile.bestFor)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }

                Spacer()

                // Selection indicator
                ZStack {
                    Circle()
                        .stroke(isSelected ? Color.accentColor : Color.secondary.opacity(0.4), lineWidth: 2)
                        .frame(width: 20, height: 20)
                    if isSelected {
                        Circle()
                            .fill(Color.accentColor)
                            .frame(width: 12, height: 12)
                    }
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(NSColor.controlBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Step 4: Done

struct DoneStepView: View {
    let onComplete: () -> Void

    @State private var showCheckmark = false
    @State private var launchAtLogin = true

    var body: some View {
        VStack(spacing: 12) {
            Spacer()

            ZStack {
                Circle()
                    .fill(Color.green.opacity(0.15))
                    .frame(width: 60, height: 60)
                    .scaleEffect(showCheckmark ? 1.0 : 0.5)

                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 40))
                    .foregroundColor(.green)
                    .scaleEffect(showCheckmark ? 1.0 : 0.0)
            }
            .animation(.spring(response: 0.5, dampingFraction: 0.6), value: showCheckmark)

            VStack(spacing: 6) {
                Text("Talk this way.")
                    .font(.system(size: 24, weight: .bold))

                Text("Hold ⌥ Space, speak, and you're off.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            // Where to find it from now on.
            HStack(spacing: 12) {
                Image(systemName: "menubar.arrow.up.rectangle")
                    .font(.system(size: 22))
                    .foregroundColor(.accentColor)
                    .frame(width: 32)
                Text("VoiceFlow lives in your menu bar — click the icon at the top of your screen anytime for settings, insights, and help.")
                    .font(.callout)
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(NSColor.controlBackgroundColor))
            )
            .padding(.horizontal, 40)
            .padding(.top, 8)

            Toggle("Launch VoiceFlow when you log in", isOn: $launchAtLogin)
                .padding(.horizontal, 60)
                .onChange(of: launchAtLogin) { newValue in
                    if newValue {
                        try? SMAppService.mainApp.register()
                    } else {
                        try? SMAppService.mainApp.unregister()
                    }
                }

            Spacer()

            Button(action: onComplete) {
                Text("Start Using VoiceFlow")
                    .font(.headline)
                    .frame(maxWidth: 240)
                    .padding(.vertical, 8)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)

            Spacer()
                .frame(height: 20)
        }
        .onAppear {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                showCheckmark = true
            }
            // Default to enabled — register on appear
            try? SMAppService.mainApp.register()
        }
    }
}

// MARK: - Usage Tip Row

struct UsageTipRow: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 14))
                .foregroundColor(.accentColor)
                .frame(width: 24)
            Text(text)
                .font(.callout)
            Spacer()
        }
    }
}

// MARK: - Wizard State

class SetupWizardState: ObservableObject {
    @Published var currentStep: WizardStep = .welcome
    let modelManager = ModelManager()
}
