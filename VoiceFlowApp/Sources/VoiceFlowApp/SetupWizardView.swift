import SwiftUI
import AppKit
import AVFoundation
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
        .frame(width: 560, height: 480)
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
                        // Activate so the system dialog appears on top (LSUIElement apps
                        // aren't frontmost, so the permission dialog gets lost otherwise)
                        NSApp.activate(ignoringOtherApps: true)
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

    @State private var selectedProfileId: String = ""
    @State private var isDownloading = false
    @State private var downloadPhase: DownloadPhase = .idle
    @State private var downloadError: String?
    @State private var pythonAvailable: Bool = true
    @State private var showPythonAlert: Bool = false
    @State private var showAdvanced: Bool = false

    // Advanced overrides (only used when showAdvanced is true)
    @State private var advancedSttId: String = ""
    @State private var advancedLlmId: String = ""

    enum DownloadPhase: Equatable {
        case idle
        case downloadingStep1
        case downloadingStep2
        case complete
    }

    private var systemRAMGB: Int {
        Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024))
    }

    // MARK: - Profiles

    private var profiles: [ModelProfile] {
        [
            ModelProfile(
                id: "lightweight",
                name: "Lightweight",
                icon: "hare",
                description: "Fastest performance, smallest download",
                downloadSize: "~1.3 GB",
                bestFor: "Best for 8 GB Macs or quick setup",
                sttId: "moonshine-tiny",
                sttType: .moonshine(modelId: "tiny"),
                llmId: "qwen3-1.7b",
                requiresPython: false,
                minRAMGB: 8
            ),
            ModelProfile(
                id: "recommended",
                name: "Recommended",
                icon: "star",
                description: "Great balance of speed and accuracy",
                downloadSize: "~2.3 GB",
                bestFor: "Best for most Macs with 16 GB+ RAM",
                sttId: "qwen3-asr-0.6b",
                sttType: .consolidated(modelId: "qwen3-asr-0.6b"),
                llmId: "qwen3-1.7b",
                requiresPython: true,
                minRAMGB: 16
            ),
            ModelProfile(
                id: "quality",
                name: "Higher Quality",
                icon: "dial.high",
                description: "Most accurate transcription and formatting",
                downloadSize: "~3.7 GB",
                bestFor: "Best for 24 GB+ Macs",
                sttId: "qwen3-asr-0.6b",
                sttType: .consolidated(modelId: "qwen3-asr-0.6b"),
                llmId: "qwen3-4b",
                requiresPython: true,
                minRAMGB: 24
            ),
        ]
    }

    private var selectedProfile: ModelProfile? {
        profiles.first(where: { $0.id == selectedProfileId })
    }

    // Effective model IDs (advanced overrides or profile defaults)
    private var effectiveSttId: String {
        showAdvanced ? advancedSttId : (selectedProfile?.sttId ?? "")
    }

    private var effectiveLlmId: String {
        showAdvanced ? advancedLlmId : (selectedProfile?.llmId ?? "")
    }

    private var sttIsDownloaded: Bool {
        let sttId = effectiveSttId
        if sttId == "moonshine-tiny" {
            return modelManager.moonshineModels.first(where: { $0.id == "tiny" })?.isDownloaded ?? false
        } else if sttId.hasPrefix("qwen3-asr") {
            return modelManager.consolidatedModels.first(where: { $0.id == sttId })?.isDownloaded ?? false
        }
        return false
    }

    private var llmIsDownloaded: Bool {
        modelManager.models.first(where: { $0.id == effectiveLlmId })?.isDownloaded ?? false
    }

    private var allDownloaded: Bool {
        sttIsDownloaded && llmIsDownloaded
    }

    // Advanced dropdown options
    private var sttOptions: [(id: String, label: String, size: String)] {
        var opts: [(id: String, label: String, size: String)] = [
            ("moonshine-tiny", "Moonshine Tiny", "~190 MB"),
        ]
        if pythonAvailable {
            opts.append(("qwen3-asr-0.6b", "Qwen3-ASR 0.6B", "~1.2 GB"))
        }
        return opts
    }

    private var llmOptions: [(id: String, label: String, size: String)] {
        [
            ("qwen3-1.7b", "Qwen3 1.7B", "~1.1 GB"),
            ("qwen3-4b", "Qwen3 4B", "~2.5 GB"),
            ("smollm3-3b", "SmolLM3 3B", "~1.9 GB"),
            ("gemma2-2b", "Gemma 2 2B", "~1.7 GB"),
        ]
    }

    // MARK: - Body

    var body: some View {
        VStack(spacing: 14) {
            Text("Download AI Models")
                .font(.title2.bold())
                .padding(.top, 8)

            Text("VoiceFlow runs entirely on your Mac. Choose a setup to download.")
                .font(.callout)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)

            ScrollView {
                VStack(spacing: 10) {
                    // Profile cards
                    ForEach(profiles) { profile in
                        ProfileCard(
                            profile: profile,
                            isSelected: selectedProfileId == profile.id && !showAdvanced,
                            isAvailable: true,
                            unavailableReason: nil,
                            isDownloaded: profileIsDownloaded(profile)
                        ) {
                            showAdvanced = false
                            selectedProfileId = profile.id
                            syncAdvancedToProfile(profile)
                        }

                    }

                    // Advanced Settings
                    Button(action: {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            showAdvanced.toggle()
                        }
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: showAdvanced ? "chevron.down" : "chevron.right")
                                .font(.caption2)
                            Text("Advanced Settings")
                                .font(.callout)
                        }
                        .foregroundColor(.accentColor)
                    }
                    .buttonStyle(.plain)
                    .padding(.horizontal, 4)
                    .padding(.top, 4)

                    if showAdvanced {
                        VStack(spacing: 12) {
                            // Speech recognition picker
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Speech Recognition")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Picker("", selection: $advancedSttId) {
                                    ForEach(sttOptions, id: \.id) { option in
                                        Text("\(option.label) (\(option.size))")
                                            .tag(option.id)
                                    }
                                }
                                .labelsHidden()
                                .disabled(isDownloading)
                            }

                            // Text formatting picker
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Text Formatting")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Picker("", selection: $advancedLlmId) {
                                    ForEach(llmOptions, id: \.id) { option in
                                        Text("\(option.label) (\(option.size))")
                                            .tag(option.id)
                                    }
                                }
                                .labelsHidden()
                                .disabled(isDownloading)
                            }
                        }
                        .padding(.top, 8)
                    }

                    // Download progress
                    if isDownloading {
                        downloadProgressView
                    }

                    // Error display
                    if let error = downloadError ?? modelManager.downloadError {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.circle.fill")
                                .foregroundColor(.red)
                            Text(error)
                                .font(.caption)
                                .foregroundColor(.red)
                                .lineLimit(2)
                            Spacer()
                            Button("Retry") {
                                downloadError = nil
                                modelManager.downloadError = nil
                                startDownloads()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                        .padding(10)
                        .background(Color.red.opacity(0.08))
                        .cornerRadius(8)
                    }
                }
                .padding(.horizontal, 24)
            }

            Spacer(minLength: 0)

            // Action buttons
            VStack(spacing: 8) {
                if allDownloaded || downloadPhase == .complete {
                    Button(action: onNext) {
                        Text("Continue")
                            .font(.headline)
                            .frame(maxWidth: 200)
                            .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                } else if isDownloading {
                    // Progress is shown inline above
                } else {
                    Button(action: handleDownloadPressed) {
                        Text("Download")
                            .font(.headline)
                            .frame(maxWidth: 200)
                            .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(effectiveSttId.isEmpty || effectiveLlmId.isEmpty)
                }

                if !isDownloading && !allDownloaded && downloadPhase != .complete {
                    Button(action: onNext) {
                        Text("Skip for now")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Models are needed before recording works. You can download them later from Settings.")
                }
            }
            .padding(.bottom, 16)
        }
        .onAppear {
            checkPythonAvailability()
            modelManager.loadModels()
            modelManager.loadSttSettings()
            modelManager.loadConsolidatedSettings()
            autoSelectProfile()
        }
        .alert("Python 3.10+ Required", isPresented: $showPythonAlert) {
            Button("Install Python") {
                NSWorkspace.shared.open(URL(string: "https://www.python.org/downloads/")!)
            }
            Button("Continue Anyway", role: .destructive) {
                startDownloads()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The selected configuration uses Qwen3-ASR, which requires Python 3.10 or later.\n\nPython was not detected on this Mac. You can install it now, or continue anyway and set it up later.")
        }
    }

    // MARK: - Download Progress View

    @ViewBuilder
    private var downloadProgressView: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                ProgressView()
                    .controlSize(.small)
                Text(downloadPhaseLabel)
                    .font(.callout)
                    .foregroundColor(.secondary)
                Spacer()
            }
            ProgressView(value: currentProgress)
                .progressViewStyle(.linear)
            Text("\(Int(currentProgress * 100))%")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(12)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(10)
    }

    private var downloadPhaseLabel: String {
        switch downloadPhase {
        case .downloadingStep1:
            return "Downloading speech recognition model..."
        case .downloadingStep2:
            return "Downloading text formatting model..."
        default:
            return "Preparing..."
        }
    }

    private var currentProgress: Double {
        switch downloadPhase {
        case .downloadingStep1:
            return sttDownloadProgress
        case .downloadingStep2:
            return modelManager.downloadProgress
        default:
            return 0
        }
    }

    private var sttDownloadProgress: Double {
        let sttId = effectiveSttId
        if sttId == "moonshine-tiny" {
            return modelManager.moonshineDownloadProgress
        } else if sttId.hasPrefix("qwen3-asr") {
            return modelManager.consolidatedDownloadProgress
        }
        return 0
    }

    // MARK: - Profile Helpers

    private func profileIsAvailable(_ profile: ModelProfile) -> Bool {
        // All profiles are always selectable — warnings are shown instead of disabling
        return true
    }

    private func profileWarning(_ profile: ModelProfile) -> String? {
        if profile.requiresPython && !pythonAvailable {
            return "Python 3.10+ required — install it before using this mode"
        }
        return nil
    }

    private func profileUnavailableReason(_ profile: ModelProfile) -> String? {
        return nil
    }

    private func profileIsDownloaded(_ profile: ModelProfile) -> Bool {
        let sttDownloaded: Bool
        switch profile.sttType {
        case .moonshine(let modelId):
            sttDownloaded = modelManager.moonshineModels.first(where: { $0.id == modelId })?.isDownloaded ?? false
        case .consolidated(let modelId):
            sttDownloaded = modelManager.consolidatedModels.first(where: { $0.id == modelId })?.isDownloaded ?? false
        }
        let llmDownloaded = modelManager.models.first(where: { $0.id == profile.llmId })?.isDownloaded ?? false
        return sttDownloaded && llmDownloaded
    }

    private func autoSelectProfile() {
        // Pick the best profile for this Mac's RAM — always selectable
        if systemRAMGB >= 24 {
            selectedProfileId = "quality"
        } else if systemRAMGB >= 16 {
            selectedProfileId = "recommended"
        } else {
            selectedProfileId = "lightweight"
        }
        if let profile = selectedProfile {
            syncAdvancedToProfile(profile)
        }
    }

    private func syncAdvancedToProfile(_ profile: ModelProfile) {
        advancedSttId = profile.sttId
        advancedLlmId = profile.llmId
    }

    // MARK: - Python Check

    private func checkPythonAvailability() {
        // macOS GUI apps don't inherit the user's shell PATH, so /usr/bin/env may not
        // find Homebrew Python. Check multiple common paths.
        let candidates = [
            "/opt/homebrew/bin/python3",        // Apple Silicon Homebrew
            "/usr/local/bin/python3",           // Intel Homebrew
            "/usr/bin/python3",                 // Xcode CLT / system
        ]

        for candidate in candidates {
            if FileManager.default.fileExists(atPath: candidate) {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: candidate)
                process.arguments = ["--version"]
                let pipe = Pipe()
                process.standardOutput = pipe
                process.standardError = pipe

                do {
                    try process.run()
                    process.waitUntilExit()
                    if process.terminationStatus == 0 {
                        let data = pipe.fileHandleForReading.readDataToEndOfFile()
                        let output = String(data: data, encoding: .utf8) ?? ""
                        if let versionStr = output.trimmingCharacters(in: .whitespacesAndNewlines)
                            .components(separatedBy: " ").last {
                            let parts = versionStr.components(separatedBy: ".")
                            if parts.count >= 2,
                               let major = Int(parts[0]),
                               let minor = Int(parts[1]),
                               major >= 3 && minor >= 10 {
                                pythonAvailable = true
                                return
                            }
                        }
                    }
                } catch {
                    continue
                }
            }
        }

        pythonAvailable = false
    }

    // MARK: - Download Logic

    private func handleDownloadPressed() {
        // Check if the selected config requires Python and it's not available
        let needsPython: Bool
        if showAdvanced {
            needsPython = effectiveSttId.hasPrefix("qwen3-asr")
        } else {
            needsPython = selectedProfile?.requiresPython ?? false
        }

        if needsPython && !pythonAvailable {
            showPythonAlert = true
        } else {
            startDownloads()
        }
    }

    private func startDownloads() {
        isDownloading = true
        downloadError = nil
        modelManager.downloadError = nil
        downloadPhase = .idle
        downloadSttModel()
    }

    private func downloadSttModel() {
        if sttIsDownloaded {
            downloadLlmModel()
            return
        }

        downloadPhase = .downloadingStep1
        let sttId = effectiveSttId

        if sttId == "moonshine-tiny" {
            modelManager.downloadMoonshineModel("tiny")
            pollMoonshineCompletion()
        } else if sttId.hasPrefix("qwen3-asr") {
            modelManager.downloadConsolidatedModel(sttId)
            pollConsolidatedCompletion()
        }
    }

    private func pollMoonshineCompletion() {
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { timer in
            DispatchQueue.main.async {
                if modelManager.downloadingMoonshineModelId == nil {
                    timer.invalidate()
                    if modelManager.downloadError != nil {
                        isDownloading = false
                        downloadPhase = .idle
                    } else {
                        modelManager.loadSttSettings()
                        downloadLlmModel()
                    }
                }
            }
        }
    }

    private func pollConsolidatedCompletion() {
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { timer in
            DispatchQueue.main.async {
                if modelManager.downloadingConsolidatedModelId == nil {
                    timer.invalidate()
                    if modelManager.downloadError != nil {
                        isDownloading = false
                        downloadPhase = .idle
                    } else {
                        modelManager.loadConsolidatedSettings()
                        downloadLlmModel()
                    }
                }
            }
        }
    }

    private func downloadLlmModel() {
        if llmIsDownloaded {
            finishDownloads()
            return
        }

        downloadPhase = .downloadingStep2
        modelManager.downloadModel(effectiveLlmId)

        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { timer in
            DispatchQueue.main.async {
                if !modelManager.isDownloading {
                    timer.invalidate()
                    if modelManager.downloadError != nil {
                        isDownloading = false
                        downloadPhase = .idle
                    } else {
                        modelManager.loadModels()
                        finishDownloads()
                    }
                }
            }
        }
    }

    private func finishDownloads() {
        isDownloading = false
        downloadPhase = .complete
        configureSelectedModels()
    }

    private func configureSelectedModels() {
        let sttId = effectiveSttId
        let llmId = effectiveLlmId

        if sttId == "moonshine-tiny" {
            modelManager.selectSttEngine(.moonshine)
            modelManager.selectMoonshineModel("tiny")
            modelManager.selectPipelineMode(.sttPlusLlm)
        } else if sttId.hasPrefix("qwen3-asr") {
            modelManager.selectSttEngine(.qwen3Asr)
            modelManager.selectConsolidatedModel(sttId)
            modelManager.selectPipelineMode(.sttPlusLlm)
        }

        if modelManager.models.first(where: { $0.id == llmId })?.isDownloaded == true {
            modelManager.selectModel(llmId)
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
        VStack(spacing: 24) {
            Spacer()

            ZStack {
                Circle()
                    .fill(Color.green.opacity(0.15))
                    .frame(width: 100, height: 100)
                    .scaleEffect(showCheckmark ? 1.0 : 0.5)

                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 64))
                    .foregroundColor(.green)
                    .scaleEffect(showCheckmark ? 1.0 : 0.0)
            }
            .animation(.spring(response: 0.5, dampingFraction: 0.6), value: showCheckmark)

            VStack(spacing: 8) {
                Text("You're All Set!")
                    .font(.system(size: 28, weight: .bold))

                Text("VoiceFlow is ready to use")
                    .font(.body)
                    .foregroundColor(.secondary)
            }

            VStack(spacing: 10) {
                UsageTipRow(icon: "option", text: "Hold \u{2325} Space to record, release to paste")
                UsageTipRow(icon: "menubar.rectangle", text: "Access settings from the menu bar icon")
            }
            .padding(.horizontal, 40)
            .padding(.vertical, 16)
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(12)
            .padding(.horizontal, 40)

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
                .frame(height: 40)
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
