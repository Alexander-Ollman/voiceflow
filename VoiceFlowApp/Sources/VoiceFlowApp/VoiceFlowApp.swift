import SwiftUI
import AppKit
import Carbon
import AVFoundation
import UserNotifications
import CoreGraphics
import ApplicationServices
import Combine
import VoiceFlowFFI

@main
struct VoiceFlowApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            SettingsView()
                .environmentObject(appDelegate.voiceFlow)
                .environmentObject(SnippetManager.shared)
        }
    }
}

// MARK: - Voice Snippets

/// A voice snippet that expands a trigger phrase into full text
struct VoiceSnippet: Codable, Identifiable, Equatable {
    let id: UUID
    var trigger: String      // What user says: "my signature"
    var expansion: String    // What it expands to: "Best regards,\nAlex"

    init(id: UUID = UUID(), trigger: String, expansion: String) {
        self.id = id
        self.trigger = trigger
        self.expansion = expansion
    }
}

/// Manages voice snippets storage and expansion
class SnippetManager: ObservableObject {
    static let shared = SnippetManager()

    @Published var snippets: [VoiceSnippet] = []

    private let storageKey = "voiceflow.snippets"

    init() {
        loadSnippets()
    }

    // MARK: - Storage

    func loadSnippets() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([VoiceSnippet].self, from: data) {
            snippets = decoded
        } else {
            // Add some default examples
            snippets = [
                VoiceSnippet(trigger: "my signature", expansion: "Best regards,\n[Your Name]"),
                VoiceSnippet(trigger: "my email", expansion: "your.email@example.com"),
            ]
            saveSnippets()
        }
    }

    func saveSnippets() {
        if let encoded = try? JSONEncoder().encode(snippets) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
    }

    // MARK: - CRUD Operations

    func addSnippet(trigger: String, expansion: String) {
        let snippet = VoiceSnippet(trigger: trigger.lowercased(), expansion: expansion)
        snippets.append(snippet)
        saveSnippets()
    }

    func updateSnippet(_ snippet: VoiceSnippet) {
        if let index = snippets.firstIndex(where: { $0.id == snippet.id }) {
            snippets[index] = snippet
            saveSnippets()
        }
    }

    func deleteSnippet(_ snippet: VoiceSnippet) {
        snippets.removeAll { $0.id == snippet.id }
        saveSnippets()
    }

    func deleteSnippets(at offsets: IndexSet) {
        snippets.remove(atOffsets: offsets)
        saveSnippets()
    }

    // MARK: - Expansion

    /// Expand any snippet triggers found in the text
    func expandSnippets(in text: String) -> String {
        var result = text
        let lowerText = text.lowercased()

        // Sort by trigger length (longest first) to avoid partial matches
        let sortedSnippets = snippets.sorted { $0.trigger.count > $1.trigger.count }

        for snippet in sortedSnippets {
            let trigger = snippet.trigger.lowercased()

            // Check for the trigger phrase (case-insensitive)
            if lowerText.contains(trigger) {
                // Find and replace (preserving surrounding text)
                result = caseInsensitiveReplace(in: result, target: trigger, replacement: snippet.expansion)
            }
        }

        return result
    }

    private func caseInsensitiveReplace(in text: String, target: String, replacement: String) -> String {
        guard let range = text.range(of: target, options: .caseInsensitive) else {
            return text
        }
        return text.replacingCharacters(in: range, with: replacement)
    }
}

// MARK: - Formatting Level

enum FormattingLevel: String, CaseIterable {
    case minimal = "minimal"
    case moderate = "moderate"
    case aggressive = "aggressive"

    var displayName: String {
        switch self {
        case .minimal: return "Minimal"
        case .moderate: return "Moderate"
        case .aggressive: return "Aggressive"
        }
    }

    var description: String {
        switch self {
        case .minimal: return "Light cleanup, preserves original speech"
        case .moderate: return "Fix grammar, punctuation, filler words"
        case .aggressive: return "Full rewrite for clarity and conciseness"
        }
    }

    var systemPrompt: String {
        switch self {
        case .minimal:
            return """
            Transcribe the speech with minimal changes. Only fix obvious typos and add basic punctuation. \
            Preserve the original wording, filler words, and speech patterns. Do not restructure sentences. \
            Capitalize proper nouns including names, places, brands, app names, and technical terms. \
            Ensure proper spacing: one space after periods, commas, and other punctuation.
            """
        case .moderate:
            return """
            Clean up the transcribed speech while preserving the speaker's intent and tone. \
            Fix grammar, add proper punctuation, remove filler words (um, uh, like), and correct minor mistakes. \
            Keep the original sentence structure where possible. \
            Capitalize proper nouns including names, places, brands, app names, UI elements, and technical terms. \
            Ensure proper spacing: one space after periods, commas, and other punctuation.
            """
        case .aggressive:
            return """
            Transform the transcribed speech into clear, professional written text. \
            Rewrite for clarity and conciseness while preserving the core meaning. \
            Fix all grammar issues, restructure awkward sentences, remove all filler words and verbal tics, \
            and optimize for readability. The output should read as polished written communication. \
            Capitalize proper nouns including names, places, brands, app names, UI elements, and technical terms. \
            Ensure proper spacing: one space after periods, commas, and other punctuation.
            """
        }
    }
}

// MARK: - Spacing Mode

enum SpacingMode: String, CaseIterable {
    case contextAware = "contextAware"
    case smart = "smart"
    case always = "always"
    case trailing = "trailing"

    var displayName: String {
        switch self {
        case .contextAware: return "Context-Aware"
        case .smart: return "Smart"
        case .always: return "Always"
        case .trailing: return "Trailing"
        }
    }

    var description: String {
        switch self {
        case .contextAware: return "Read cursor context to determine spacing (recommended)"
        case .smart: return "Add leading space if text starts with a letter"
        case .always: return "Always add a leading space"
        case .trailing: return "Add trailing space after each transcription"
        }
    }

    /// Apply spacing to the transcribed text
    func apply(to text: String) -> String {
        switch self {
        case .contextAware:
            // Try to read character before cursor using Accessibility API
            let result = CursorContext.getCharacterBeforeCursor()

            switch result {
            case .character(let charBefore):
                // Successfully read character - add space if previous char is not whitespace
                if !charBefore.isWhitespace && !charBefore.isNewline {
                    if let first = text.first, first.isLetter || first.isNumber {
                        return " " + text
                    }
                }
                return text
            case .atStart:
                // Cursor is at start of field - no space needed
                return text
            case .unavailable:
                // Can't determine context - be conservative, no space
                return text
            }
        case .smart:
            // Add leading space if text starts with a letter
            if let first = text.first, first.isLetter {
                return " " + text
            }
            return text
        case .always:
            // Always add leading space
            return " " + text
        case .trailing:
            // Add trailing space
            return text + " "
        }
    }
}

// MARK: - Punctuation Options

/// Individual punctuation detection features that can be toggled
enum PunctuationOption: String, CaseIterable {
    case voiceCommands = "voiceCommands"
    case pauseAnalysis = "pauseAnalysis"
    case pitchAnalysis = "pitchAnalysis"
    case llmHints = "llmHints"

    var displayName: String {
        switch self {
        case .voiceCommands: return "Voice Commands"
        case .pauseAnalysis: return "Pause Detection"
        case .pitchAnalysis: return "Pitch Analysis"
        case .llmHints: return "LLM Hints"
        }
    }

    var description: String {
        switch self {
        case .voiceCommands: return "Say 'period', 'comma', 'question mark' for punctuation"
        case .pauseAnalysis: return "Detect pauses to infer sentence boundaries"
        case .pitchAnalysis: return "Detect rising pitch for questions"
        case .llmHints: return "Pass prosody hints to LLM for better decisions"
        }
    }

    var defaultEnabled: Bool {
        switch self {
        case .voiceCommands: return true
        case .pauseAnalysis: return true
        case .pitchAnalysis: return true
        case .llmHints: return true
        }
    }
}

// MARK: - Application Context Detection

/// Detected application context for formatting
enum AppContext: String {
    case email = "email"
    case slack = "slack"
    case code = "code"
    case general = "default"

    /// Additional context hint to append to the system prompt
    var contextHint: String {
        switch self {
        case .email:
            return """

            [APPLICATION CONTEXT: Email]
            Format for professional email communication:
            - Use proper greeting and sign-off if the content suggests a complete email
            - Use appropriate paragraph breaks between distinct topics
            - Keep tone professional but natural
            - Format lists cleanly if the speaker lists items
            """
        case .slack:
            return """

            [APPLICATION CONTEXT: Slack/Chat]
            Format for casual team communication:
            - Keep it conversational
            - Convert "[word/phrase] emoji" to Slack colon format: "thumbs up emoji" → :thumbs_up:, "fire emoji" → :fire:, "heart emoji" → :heart:
            - Common emoji mappings: thumbs up → :+1:, thumbs down → :-1:, smile/smiley → :smile:, laugh → :joy:, sad → :cry:, heart → :heart:, fire → :fire:, rocket → :rocket:, check/checkmark → :white_check_mark:, x/cross → :x:, eyes → :eyes:, thinking → :thinking_face:, party → :tada:, clap → :clap:, wave → :wave:, pray/thanks → :pray:
            - No formal greetings needed
            """
        case .code:
            return """

            [APPLICATION CONTEXT: Code Editor]
            Format for code comments or documentation:
            - Be technical and precise
            - Use appropriate comment syntax if dictating a comment
            - Variable names should be camelCase or snake_case as appropriate
            - Keep explanations concise
            """
        case .general:
            return ""
        }
    }
}

/// Detects the frontmost application and returns appropriate context
struct AppContextDetector {
    /// Map of bundle identifiers to app contexts
    private static let bundleContextMap: [String: AppContext] = [
        // Email apps
        "com.microsoft.Outlook": .email,
        "com.apple.mail": .email,
        "com.google.Chrome": .general,  // Could be Gmail, handled separately
        "com.readdle.smartemail-Mac": .email,
        "com.freron.MailMate": .email,
        "com.postbox-inc.postbox": .email,

        // Chat/Slack apps
        "com.tinyspeck.slackmacgap": .slack,
        "com.hnc.Discord": .slack,
        "com.microsoft.teams2": .slack,
        "ru.keepcoder.Telegram": .slack,
        "net.whatsapp.WhatsApp": .slack,
        "com.facebook.archon.developerID": .slack, // Messenger

        // Code editors
        "com.microsoft.VSCode": .code,
        "com.apple.dt.Xcode": .code,
        "com.sublimetext.4": .code,
        "com.jetbrains.intellij": .code,
        "com.googlecode.iterm2": .code,
        "com.apple.Terminal": .code,
        "com.cursor.Cursor": .code,
        "dev.zed.Zed": .code,
        "com.todesktop.230313mzl4w4u92": .code, // Cursor
    ]

    /// App names (partial match) to contexts - fallback for unknown bundle IDs
    private static let appNameContextMap: [(String, AppContext)] = [
        ("outlook", .email),
        ("mail", .email),
        ("gmail", .email),
        ("slack", .slack),
        ("discord", .slack),
        ("teams", .slack),
        ("telegram", .slack),
        ("whatsapp", .slack),
        ("messages", .slack),
        ("xcode", .code),
        ("code", .code),  // VS Code, Cursor
        ("terminal", .code),
        ("iterm", .code),
        ("sublime", .code),
        ("intellij", .code),
        ("pycharm", .code),
        ("webstorm", .code),
        ("cursor", .code),
        ("zed", .code),
    ]

    /// Detect the context based on the frontmost application
    static func detectContext() -> AppContext {
        guard let frontmostApp = NSWorkspace.shared.frontmostApplication else {
            return .general
        }

        // First try bundle identifier (most reliable)
        if let bundleID = frontmostApp.bundleIdentifier,
           let context = bundleContextMap[bundleID] {
            return context
        }

        // Fall back to app name matching
        let appName = frontmostApp.localizedName?.lowercased() ?? ""
        for (namePattern, context) in appNameContextMap {
            if appName.contains(namePattern) {
                return context
            }
        }

        return .general
    }
}

// MARK: - Cursor Context (Accessibility API)

/// Result of attempting to read cursor context
enum CursorContextResult {
    case character(Character)  // Successfully read the character before cursor
    case atStart               // Cursor is at position 0 (start of field)
    case unavailable           // Can't determine context (no permission, unsupported app, etc.)
}

/// Helper to read cursor context using macOS Accessibility API
struct CursorContext {
    /// Get the character immediately before the cursor in the focused text field
    static func getCharacterBeforeCursor() -> CursorContextResult {
        // Check accessibility permission first
        guard AXIsProcessTrusted() else {
            return .unavailable
        }

        // Get the system-wide accessibility element
        let systemWide = AXUIElementCreateSystemWide()

        // Get the focused UI element
        var focusedElementRef: CFTypeRef?
        let focusedError = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedElementRef
        )

        guard focusedError == .success,
              let focusedElement = focusedElementRef else {
            return .unavailable
        }

        let element = focusedElement as! AXUIElement

        // Get the selected text range (this gives us cursor position)
        var selectedRangeRef: CFTypeRef?
        let rangeError = AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            &selectedRangeRef
        )

        guard rangeError == .success,
              let rangeValue = selectedRangeRef else {
            return .unavailable
        }

        // Extract the range
        var range = CFRange(location: 0, length: 0)
        guard AXValueGetValue(rangeValue as! AXValue, .cfRange, &range) else {
            return .unavailable
        }

        // Cursor position is at range.location
        let cursorPosition = range.location

        // If cursor is at the beginning, there's no character before it
        guard cursorPosition > 0 else {
            return .atStart
        }

        // Get the character before cursor using parameterized attribute
        var charRange = CFRange(location: cursorPosition - 1, length: 1)

        guard let rangeParam = AXValueCreate(.cfRange, &charRange) else {
            return .unavailable
        }

        var charRef: CFTypeRef?
        let charError = AXUIElementCopyParameterizedAttributeValue(
            element,
            kAXStringForRangeParameterizedAttribute as CFString,
            rangeParam,
            &charRef
        )

        guard charError == .success,
              let charString = charRef as? String,
              let char = charString.first else {
            return .unavailable
        }

        return .character(char)
    }
}

// MARK: - App Delegate

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem!
    var voiceFlow = VoiceFlowBridge()
    var audioRecorder = AudioRecorder()
    var hotkeyManager = GlobalHotkeyManager()
    var overlayPanel: NSPanel?
    var overlayHostingView: NSHostingView<RecordingOverlayView>?
    var overlayState = OverlayState()
    var settingsWindow: NSWindow?
    private var audioLevelCancellable: AnyCancellable?

    private var isRecording = false
    private var recordingMenuItem: NSMenuItem?
    private var formattingMenuItems: [FormattingLevel: NSMenuItem] = [:]
    private var spacingMenuItems: [SpacingMode: NSMenuItem] = [:]
    private var punctuationMenuItems: [PunctuationOption: NSMenuItem] = [:]

    // User preference for formatting level
    var formattingLevel: FormattingLevel {
        get {
            let rawValue = UserDefaults.standard.string(forKey: "formattingLevel") ?? FormattingLevel.moderate.rawValue
            return FormattingLevel(rawValue: rawValue) ?? .moderate
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: "formattingLevel")
            updateFormattingMenuChecks()
        }
    }

    // User preference for spacing mode
    var spacingMode: SpacingMode {
        get {
            let rawValue = UserDefaults.standard.string(forKey: "spacingMode") ?? SpacingMode.contextAware.rawValue
            return SpacingMode(rawValue: rawValue) ?? .contextAware
        }
        set {
            UserDefaults.standard.set(newValue.rawValue, forKey: "spacingMode")
            updateSpacingMenuChecks()
        }
    }

    // User preferences for punctuation options
    func isPunctuationOptionEnabled(_ option: PunctuationOption) -> Bool {
        let key = "punctuation_\(option.rawValue)"
        if UserDefaults.standard.object(forKey: key) == nil {
            return option.defaultEnabled
        }
        return UserDefaults.standard.bool(forKey: key)
    }

    func setPunctuationOption(_ option: PunctuationOption, enabled: Bool) {
        let key = "punctuation_\(option.rawValue)"
        UserDefaults.standard.set(enabled, forKey: key)
        updatePunctuationMenuChecks()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Hide dock icon - menu bar only
        NSApp.setActivationPolicy(.accessory)

        setupStatusItem()
        setupHotkey()
        setupOverlay()

        // Request notification permission
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }

        // Check accessibility permission (required for auto-paste)
        checkAccessibilityPermission()

        // Request microphone permission
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            if !granted {
                DispatchQueue.main.async {
                    self.showAlert(title: "Microphone Access Required",
                                   message: "VoiceFlow needs microphone access to transcribe your speech.")
                }
            }
        }
    }

    private func checkAccessibilityPermission() {
        // Check if accessibility is already granted
        let trusted = AXIsProcessTrusted()

        if !trusted {
            // Show alert explaining why we need accessibility
            let alert = NSAlert()
            alert.messageText = "Accessibility Permission Required"
            alert.informativeText = "VoiceFlow needs Accessibility permission to automatically paste transcribed text.\n\nClick 'Open Settings' to grant permission, then restart VoiceFlow."
            alert.alertStyle = .warning
            alert.addButton(withTitle: "Open Settings")
            alert.addButton(withTitle: "Later")

            let response = alert.runModal()

            if response == .alertFirstButtonReturn {
                // Open System Settings to Accessibility pane
                let prefPaneURL = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
                NSWorkspace.shared.open(prefPaneURL)
            }
        }
    }

    // MARK: - Status Bar Setup

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        updateMenuBarIcon()

        // Observe appearance changes for light/dark mode
        DistributedNotificationCenter.default.addObserver(
            self,
            selector: #selector(appearanceChanged),
            name: NSNotification.Name("AppleInterfaceThemeChangedNotification"),
            object: nil
        )

        setupMenu()
    }

    @objc private func appearanceChanged() {
        updateMenuBarIcon()
    }

    private func updateMenuBarIcon() {
        guard let button = statusItem.button else { return }

        // Use single white icon (works for both light and dark menu bars)
        if let iconPath = Bundle.main.path(forResource: "MenuBarIcon", ofType: "png"),
           let icon = NSImage(contentsOfFile: iconPath) {
            icon.size = NSSize(width: 18, height: 18)
            button.image = icon
        } else {
            // Fallback to SF Symbol
            button.image = NSImage(systemSymbolName: "mic.fill", accessibilityDescription: "VoiceFlow")
        }
    }

    private func setupMenu() {
        let menu = NSMenu()

        let statusMenuItem = NSMenuItem(title: "VoiceFlow v\(VoiceFlowBridge.version)", action: nil, keyEquivalent: "")
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)

        menu.addItem(NSMenuItem.separator())

        recordingMenuItem = NSMenuItem(title: "Hold ⌥ Space to Record", action: #selector(toggleRecording), keyEquivalent: "")
        recordingMenuItem?.target = self
        menu.addItem(recordingMenuItem!)

        menu.addItem(NSMenuItem.separator())

        // Formatting Level submenu
        let formattingMenu = NSMenu()
        let formattingMenuItem = NSMenuItem(title: "Formatting", action: nil, keyEquivalent: "")
        formattingMenuItem.submenu = formattingMenu

        for level in FormattingLevel.allCases {
            let item = NSMenuItem(
                title: level.displayName,
                action: #selector(selectFormattingLevel(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = level
            item.toolTip = level.description
            formattingMenu.addItem(item)
            formattingMenuItems[level] = item
        }

        menu.addItem(formattingMenuItem)
        updateFormattingMenuChecks()

        // Punctuation submenu
        let punctuationMenu = NSMenu()
        let punctuationMenuItem = NSMenuItem(title: "Punctuation", action: nil, keyEquivalent: "")
        punctuationMenuItem.submenu = punctuationMenu

        for option in PunctuationOption.allCases {
            let item = NSMenuItem(
                title: option.displayName,
                action: #selector(togglePunctuationOption(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = option
            item.toolTip = option.description
            punctuationMenu.addItem(item)
            punctuationMenuItems[option] = item
        }

        menu.addItem(punctuationMenuItem)
        updatePunctuationMenuChecks()

        // Spacing Mode submenu
        let spacingMenu = NSMenu()
        let spacingMenuItem = NSMenuItem(title: "Spacing", action: nil, keyEquivalent: "")
        spacingMenuItem.submenu = spacingMenu

        for mode in SpacingMode.allCases {
            let item = NSMenuItem(
                title: mode.displayName,
                action: #selector(selectSpacingMode(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = mode
            item.toolTip = mode.description
            spacingMenu.addItem(item)
            spacingMenuItems[mode] = item
        }

        menu.addItem(spacingMenuItem)
        updateSpacingMenuChecks()

        menu.addItem(NSMenuItem.separator())

        // Permissions submenu
        let permissionsMenu = NSMenu()
        let permissionsMenuItem = NSMenuItem(title: "Permissions", action: nil, keyEquivalent: "")
        permissionsMenuItem.submenu = permissionsMenu

        let accessibilityItem = NSMenuItem(
            title: "Accessibility (for paste)",
            action: #selector(openAccessibilitySettings),
            keyEquivalent: ""
        )
        accessibilityItem.target = self
        permissionsMenu.addItem(accessibilityItem)

        let microphoneItem = NSMenuItem(
            title: "Microphone (for recording)",
            action: #selector(openMicrophoneSettings),
            keyEquivalent: ""
        )
        microphoneItem.target = self
        permissionsMenu.addItem(microphoneItem)

        menu.addItem(permissionsMenuItem)

        menu.addItem(NSMenuItem.separator())

        let settingsItem = NSMenuItem(title: "Settings...", action: #selector(openSettings), keyEquivalent: ",")
        settingsItem.target = self
        menu.addItem(settingsItem)

        menu.addItem(NSMenuItem.separator())

        let quitItem = NSMenuItem(title: "Quit VoiceFlow", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem.menu = menu
    }

    @objc private func openSettings() {
        // If window already exists, just bring it to front
        if let window = settingsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        // Create settings window
        let settingsView = SettingsView()
            .environmentObject(voiceFlow)
            .environmentObject(SnippetManager.shared)

        let hostingController = NSHostingController(rootView: settingsView)

        let window = NSWindow(contentViewController: hostingController)
        window.title = "VoiceFlow Settings"
        window.styleMask = [.titled, .closable, .miniaturizable]
        window.setContentSize(NSSize(width: 450, height: 500))
        window.center()
        window.isReleasedWhenClosed = false

        settingsWindow = window
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func openAccessibilitySettings() {
        // Show Apple's native accessibility permission prompt
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
    }

    @objc private func openMicrophoneSettings() {
        // Show Apple's native microphone permission prompt
        AVCaptureDevice.requestAccess(for: .audio) { _ in }
    }

    @objc private func selectFormattingLevel(_ sender: NSMenuItem) {
        guard let level = sender.representedObject as? FormattingLevel else { return }
        formattingLevel = level
    }

    private func updateFormattingMenuChecks() {
        for (level, item) in formattingMenuItems {
            item.state = (level == formattingLevel) ? .on : .off
        }
    }

    @objc private func selectSpacingMode(_ sender: NSMenuItem) {
        guard let mode = sender.representedObject as? SpacingMode else { return }
        spacingMode = mode
    }

    private func updateSpacingMenuChecks() {
        for (mode, item) in spacingMenuItems {
            item.state = (mode == spacingMode) ? .on : .off
        }
    }

    @objc private func togglePunctuationOption(_ sender: NSMenuItem) {
        guard let option = sender.representedObject as? PunctuationOption else { return }
        let currentlyEnabled = isPunctuationOptionEnabled(option)
        setPunctuationOption(option, enabled: !currentlyEnabled)
    }

    private func updatePunctuationMenuChecks() {
        for (option, item) in punctuationMenuItems {
            item.state = isPunctuationOptionEnabled(option) ? .on : .off
        }
    }

    // MARK: - Overlay Setup

    private func setupOverlay() {
        // Create floating panel
        let panelWidth: CGFloat = 200
        let panelHeight: CGFloat = 60

        // Get main screen
        guard let screen = NSScreen.main else { return }

        // Position at bottom center of screen, above the dock
        let screenFrame = screen.visibleFrame
        let panelX = screenFrame.midX - panelWidth / 2
        let panelY = screenFrame.minY + 80  // 80pt above dock area

        let panel = NSPanel(
            contentRect: NSRect(x: panelX, y: panelY, width: panelWidth, height: panelHeight),
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )

        panel.level = .floating
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isMovableByWindowBackground = false
        panel.titlebarAppearsTransparent = true
        panel.titleVisibility = .hidden

        // Create SwiftUI hosting view
        let overlayView = RecordingOverlayView(state: overlayState)
        let hostingView = NSHostingView(rootView: overlayView)
        hostingView.frame = NSRect(x: 0, y: 0, width: panelWidth, height: panelHeight)

        panel.contentView = hostingView

        self.overlayPanel = panel
        self.overlayHostingView = hostingView
    }

    private func showOverlay(state: OverlayState.RecordingState) {
        overlayState.state = state
        overlayPanel?.orderFront(nil)
    }

    private func hideOverlay() {
        overlayPanel?.orderOut(nil)
        overlayState.state = .idle
    }

    // MARK: - Hotkey Setup

    private func setupHotkey() {
        // Option + Space: hold to record, release to stop and paste
        hotkeyManager.register(
            keyCode: UInt32(kVK_Space),
            modifiers: UInt32(optionKey),
            onPress: { [weak self] in
                DispatchQueue.main.async {
                    self?.startRecording()
                }
            },
            onRelease: { [weak self] in
                DispatchQueue.main.async {
                    self?.stopRecordingAndPaste()
                }
            }
        )
    }

    // MARK: - Recording Control

    @objc func toggleRecording() {
        if isRecording {
            stopRecordingAndPaste()
        } else {
            startRecording()
        }
    }

    private func startRecording() {
        guard !isRecording else { return }

        do {
            try audioRecorder.startRecording()
            isRecording = true
            recordingMenuItem?.title = "Recording... (release ⌥ Space)"

            // Show overlay
            showOverlay(state: .recording)

            // Subscribe to audio level changes
            audioLevelCancellable = audioRecorder.$audioLevel
                .receive(on: DispatchQueue.main)
                .sink { [weak self] level in
                    self?.overlayState.audioLevel = level
                }
        } catch {
            showAlert(title: "Recording Error", message: error.localizedDescription)
        }
    }

    private func stopRecordingAndPaste() {
        guard isRecording else { return }

        // Cancel audio level subscription
        audioLevelCancellable?.cancel()
        audioLevelCancellable = nil

        let audio = audioRecorder.stopRecording()
        isRecording = false
        recordingMenuItem?.title = "Hold ⌥ Space to Record"

        guard !audio.isEmpty else {
            hideOverlay()
            showNotification(title: "No Audio", body: "No audio was captured")
            return
        }

        // Show processing state in overlay
        showOverlay(state: .processing)

        // Detect the application context (email, slack, code, etc.)
        let appContext = AppContextDetector.detectContext()

        // Combine formatting level prompt with app-specific context
        let context = formattingLevel.systemPrompt + appContext.contextHint
        let currentSpacingMode = spacingMode

        Task {
            if let result = await voiceFlow.process(audio: audio, context: context) {
                // Apply snippet expansion (e.g., "my signature" → "Best regards,\nJohn")
                let expandedText = SnippetManager.shared.expandSnippets(in: result.formattedText)

                // Apply spacing mode to the expanded text
                let spacedText = currentSpacingMode.apply(to: expandedText)

                // Copy to clipboard
                let pasteboard = NSPasteboard.general
                pasteboard.clearContents()
                pasteboard.setString(spacedText, forType: .string)

                // Hide overlay first
                await MainActor.run {
                    hideOverlay()
                }

                // Wait for clipboard and UI to settle
                try? await Task.sleep(nanoseconds: 100_000_000)  // 100ms

                // Auto-paste with Cmd+V
                await MainActor.run {
                    simulatePaste()
                }

                showNotification(
                    title: "Pasted (\(result.totalMs)ms)",
                    body: String(spacedText.prefix(100)) + (spacedText.count > 100 ? "..." : "")
                )
            } else {
                await MainActor.run {
                    hideOverlay()
                }
                showNotification(title: "Processing Failed", body: voiceFlow.lastError ?? "Unknown error")
            }
        }
    }

    /// Simulate Cmd+V paste keystroke using AppleScript (more reliable)
    private func simulatePaste() {
        // Check accessibility first
        guard AXIsProcessTrusted() else {
            print("Accessibility not granted - cannot auto-paste")
            return
        }

        // Use AppleScript for more reliable paste
        let script = NSAppleScript(source: """
            tell application "System Events"
                keystroke "v" using command down
            end tell
        """)

        var error: NSDictionary?
        script?.executeAndReturnError(&error)

        if let error = error {
            print("AppleScript paste error: \(error)")
            // Fallback to CGEvent
            fallbackPaste()
        }
    }

    private func fallbackPaste() {
        let vKeyCode: CGKeyCode = 9

        // Create event source
        guard let source = CGEventSource(stateID: .hidSystemState) else { return }

        // Create key down event with Cmd modifier
        guard let keyDown = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: true) else { return }
        keyDown.flags = .maskCommand

        // Create key up event with Cmd modifier
        guard let keyUp = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: false) else { return }
        keyUp.flags = .maskCommand

        // Post the events with small delay between
        keyDown.post(tap: .cgSessionEventTap)
        usleep(10000)  // 10ms delay
        keyUp.post(tap: .cgSessionEventTap)
    }

    // MARK: - Menu Actions

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }

    // MARK: - Notifications

    private func showNotification(title: String, body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default

        let request = UNNotificationRequest(
            identifier: UUID().uuidString,
            content: content,
            trigger: nil
        )

        UNUserNotificationCenter.current().add(request)
    }

    private func showAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}

// MARK: - Global Hotkey Manager

final class GlobalHotkeyManager {
    private var hotKeyRef: EventHotKeyRef?

    private static var onPressHandler: (() -> Void)?
    private static var onReleaseHandler: (() -> Void)?

    func register(keyCode: UInt32, modifiers: UInt32, onPress: @escaping () -> Void, onRelease: @escaping () -> Void) {
        GlobalHotkeyManager.onPressHandler = onPress
        GlobalHotkeyManager.onReleaseHandler = onRelease

        var hotKeyID = EventHotKeyID()
        hotKeyID.signature = OSType(0x564643) // "VFC"
        hotKeyID.id = 1

        // Register for both press and release events
        var eventTypes = [
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))
        ]

        // Install event handler
        InstallEventHandler(
            GetApplicationEventTarget(),
            { (_, event, _) -> OSStatus in
                var eventKind: UInt32 = 0
                GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeUInt32), nil, MemoryLayout<UInt32>.size, nil, &eventKind)

                let kind = GetEventKind(event)
                if kind == UInt32(kEventHotKeyPressed) {
                    GlobalHotkeyManager.onPressHandler?()
                } else if kind == UInt32(kEventHotKeyReleased) {
                    GlobalHotkeyManager.onReleaseHandler?()
                }
                return noErr
            },
            2,
            &eventTypes,
            nil,
            nil
        )

        // Register hotkey
        RegisterEventHotKey(
            keyCode,
            modifiers,
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &hotKeyRef
        )
    }

    deinit {
        if let hotKeyRef = hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
        }
    }
}

// MARK: - Permission Buttons

struct AccessibilityPermissionButton: View {
    @State private var isGranted: Bool = AXIsProcessTrusted()

    var body: some View {
        Group {
            if isGranted {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            } else {
                Button("Grant") {
                    let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
                    AXIsProcessTrustedWithOptions(options)
                    // Check again after a short delay
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        isGranted = AXIsProcessTrusted()
                    }
                }
            }
        }
        .onAppear {
            isGranted = AXIsProcessTrusted()
        }
    }
}

struct MicrophonePermissionButton: View {
    @State private var authStatus: AVAuthorizationStatus = AVCaptureDevice.authorizationStatus(for: .audio)

    var body: some View {
        Group {
            switch authStatus {
            case .authorized:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            case .notDetermined:
                Button("Grant") {
                    AVCaptureDevice.requestAccess(for: .audio) { granted in
                        DispatchQueue.main.async {
                            authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
                        }
                    }
                }
            case .denied, .restricted:
                Button("Open Settings") {
                    if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                        NSWorkspace.shared.open(url)
                    }
                }
            @unknown default:
                Button("Grant") {
                    AVCaptureDevice.requestAccess(for: .audio) { _ in }
                }
            }
        }
        .onAppear {
            authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        }
    }
}

// MARK: - Settings View

struct SettingsHeaderView: View {
    var body: some View {
        HStack(spacing: 12) {
            // App Icon
            if let logoPath = Bundle.main.path(forResource: "AppLogo", ofType: "png"),
               let logoImage = NSImage(contentsOfFile: logoPath) {
                Image(nsImage: logoImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 48, height: 48)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            } else {
                Image(systemName: "waveform.circle.fill")
                    .font(.system(size: 48))
                    .foregroundColor(.accentColor)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text("VoiceFlow")
                    .font(.title2)
                    .fontWeight(.semibold)
                Text("v\(VoiceFlowBridge.version)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(.bottom, 8)
    }
}

struct SettingsView: View {
    @EnvironmentObject var voiceFlow: VoiceFlowBridge
    @EnvironmentObject var snippetManager: SnippetManager
    @StateObject private var modelManager = ModelManager()

    var body: some View {
        VStack(spacing: 0) {
            // Header with logo
            SettingsHeaderView()
                .padding(.horizontal, 20)
                .padding(.top, 16)

            Divider()
                .padding(.horizontal, 16)

            // Tabs
            TabView {
                GeneralSettingsView()
                    .environmentObject(voiceFlow)
                    .tabItem {
                        Label("General", systemImage: "gear")
                    }

                ModelSettingsView()
                    .environmentObject(modelManager)
                    .tabItem {
                        Label("Models", systemImage: "cpu")
                    }

                SnippetsSettingsView()
                    .environmentObject(snippetManager)
                    .tabItem {
                        Label("Snippets", systemImage: "text.badge.plus")
                    }
            }
        }
        .frame(width: 520, height: 560)
    }
}

// MARK: - General Settings Tab

struct GeneralSettingsView: View {
    @EnvironmentObject var voiceFlow: VoiceFlowBridge

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Hotkey Section
                GroupBox {
                    HStack {
                        Label("Recording Shortcut", systemImage: "command")
                        Spacer()
                        Text("⌥ Space")
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(Color.secondary.opacity(0.15))
                            .cornerRadius(6)
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Hotkey", systemImage: "keyboard")
                        .font(.headline)
                }

                // Status Section
                GroupBox {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Label("Pipeline", systemImage: "cpu")
                            Spacer()
                            if voiceFlow.isInitialized {
                                HStack(spacing: 4) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(.green)
                                    Text("Ready")
                                        .foregroundColor(.green)
                                }
                            } else {
                                HStack(spacing: 6) {
                                    ProgressView()
                                        .scaleEffect(0.7)
                                    Text("Initializing...")
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Status", systemImage: "info.circle")
                        .font(.headline)
                }

                // Last Processing Section
                if let result = voiceFlow.lastResult {
                    GroupBox {
                        VStack(spacing: 8) {
                            HStack {
                                Text("Transcription")
                                Spacer()
                                Text("\(result.transcriptionMs)ms")
                                    .monospacedDigit()
                                    .foregroundColor(.secondary)
                            }
                            HStack {
                                Text("LLM Formatting")
                                Spacer()
                                Text("\(result.llmMs)ms")
                                    .monospacedDigit()
                                    .foregroundColor(.secondary)
                            }
                            Divider()
                            HStack {
                                Text("Total")
                                    .fontWeight(.medium)
                                Spacer()
                                Text("\(result.totalMs)ms")
                                    .monospacedDigit()
                                    .fontWeight(.medium)
                            }
                        }
                        .padding(.vertical, 4)
                    } label: {
                        Label("Last Processing", systemImage: "clock")
                            .font(.headline)
                    }
                }

                // Permissions Section
                GroupBox {
                    VStack(spacing: 12) {
                        HStack {
                            HStack(spacing: 8) {
                                Image(systemName: "hand.raised.fill")
                                    .foregroundColor(.blue)
                                    .frame(width: 20)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Accessibility")
                                    Text("Required for auto-paste")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            Spacer()
                            AccessibilityPermissionButton()
                        }

                        Divider()

                        HStack {
                            HStack(spacing: 8) {
                                Image(systemName: "mic.fill")
                                    .foregroundColor(.red)
                                    .frame(width: 20)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Microphone")
                                    Text("Required for recording")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            Spacer()
                            MicrophonePermissionButton()
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Permissions", systemImage: "lock.shield")
                        .font(.headline)
                }

                Spacer(minLength: 16)

                // Action Buttons
                HStack {
                    Button(action: restartApp) {
                        Label("Restart", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)

                    Spacer()

                    Button(action: { NSApp.terminate(nil) }) {
                        Text("Quit VoiceFlow")
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }
            }
            .padding()
        }
    }

    private func restartApp() {
        // Get the path to the current app bundle
        let bundlePath = Bundle.main.bundlePath

        // Create a shell script that waits briefly then relaunches
        let script = """
            sleep 0.5
            open "\(bundlePath)"
            """

        // Run the script in background
        let task = Process()
        task.launchPath = "/bin/bash"
        task.arguments = ["-c", script]
        try? task.run()

        // Terminate the current instance
        NSApp.terminate(nil)
    }
}

// MARK: - Snippets Settings Tab

struct SnippetsSettingsView: View {
    @EnvironmentObject var snippetManager: SnippetManager
    @State private var showingAddSheet = false
    @State private var editingSnippet: VoiceSnippet? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Voice Snippets")
                    .font(.headline)
                Spacer()
                Button(action: { showingAddSheet = true }) {
                    Label("Add", systemImage: "plus")
                }
            }

            Text("Say a trigger phrase and it will expand to the full text.")
                .font(.caption)
                .foregroundColor(.secondary)

            if snippetManager.snippets.isEmpty {
                VStack(spacing: 8) {
                    Spacer()
                    Image(systemName: "text.badge.plus")
                        .font(.system(size: 40))
                        .foregroundColor(.secondary)
                    Text("No snippets yet")
                        .foregroundColor(.secondary)
                    Text("Click + to add your first snippet")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .frame(maxWidth: .infinity)
            } else {
                List {
                    ForEach(snippetManager.snippets) { snippet in
                        SnippetRowView(snippet: snippet)
                            .contentShape(Rectangle())
                            .onTapGesture {
                                editingSnippet = snippet
                            }
                    }
                    .onDelete(perform: snippetManager.deleteSnippets)
                }
                .listStyle(.bordered)
            }
        }
        .padding()
        .sheet(isPresented: $showingAddSheet) {
            SnippetEditView(mode: .add) { trigger, expansion in
                snippetManager.addSnippet(trigger: trigger, expansion: expansion)
            }
        }
        .sheet(item: $editingSnippet) { snippet in
            SnippetEditView(mode: .edit(snippet)) { trigger, expansion in
                var updated = snippet
                updated.trigger = trigger
                updated.expansion = expansion
                snippetManager.updateSnippet(updated)
            }
        }
    }
}

struct SnippetRowView: View {
    let snippet: VoiceSnippet

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("\"\(snippet.trigger)\"")
                    .fontWeight(.medium)
                Spacer()
                Image(systemName: "arrow.right")
                    .foregroundColor(.secondary)
                    .font(.caption)
            }
            Text(snippet.expansion.prefix(50) + (snippet.expansion.count > 50 ? "..." : ""))
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(1)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Snippet Edit Sheet

struct SnippetEditView: View {
    enum Mode {
        case add
        case edit(VoiceSnippet)
    }

    let mode: Mode
    let onSave: (String, String) -> Void

    @Environment(\.dismiss) var dismiss
    @State private var trigger: String = ""
    @State private var expansion: String = ""

    var title: String {
        switch mode {
        case .add: return "Add Snippet"
        case .edit: return "Edit Snippet"
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title)
                .font(.headline)

            VStack(alignment: .leading, spacing: 4) {
                Text("Trigger Phrase")
                    .font(.caption)
                    .foregroundColor(.secondary)
                TextField("e.g., my signature", text: $trigger)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Expands To")
                    .font(.caption)
                    .foregroundColor(.secondary)
                TextEditor(text: $expansion)
                    .font(.body)
                    .frame(height: 100)
                    .border(Color.secondary.opacity(0.3), width: 1)
            }

            HStack {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Save") {
                    onSave(trigger, expansion)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(trigger.isEmpty || expansion.isEmpty)
            }
        }
        .padding()
        .frame(width: 350, height: 280)
        .onAppear {
            if case .edit(let snippet) = mode {
                trigger = snippet.trigger
                expansion = snippet.expansion
            }
        }
    }
}

// MARK: - Model Manager

struct LLMModel: Identifiable {
    let id: String
    let displayName: String
    let filename: String
    let sizeGB: Float
    let downloadUrl: String
    var isDownloaded: Bool
}

// MARK: - STT Engine Types

enum SttEngine: String, CaseIterable {
    case whisper = "whisper"
    case moonshine = "moonshine"

    var displayName: String {
        switch self {
        case .whisper: return "Whisper"
        case .moonshine: return "Moonshine"
        }
    }

    var description: String {
        switch self {
        case .whisper: return "OpenAI Whisper - accurate, proven technology"
        case .moonshine: return "Moonshine - 5x faster, lower memory usage"
        }
    }
}

struct MoonshineModel: Identifiable {
    let id: String
    let displayName: String
    let sizeMB: UInt32
    var isDownloaded: Bool
}

class ModelManager: ObservableObject {
    @Published var models: [LLMModel] = []
    @Published var currentModelId: String = ""
    @Published var isDownloading: Bool = false
    @Published var downloadProgress: Double = 0
    @Published var downloadError: String?
    @Published var needsRestart: Bool = false
    @Published var downloadingModelId: String?

    // STT Engine settings
    @Published var currentSttEngine: SttEngine = .whisper
    @Published var currentMoonshineModelId: String = "tiny"
    @Published var moonshineModels: [MoonshineModel] = []
    @Published var downloadingMoonshineModelId: String?
    @Published var moonshineDownloadProgress: Double = 0

    private var downloadTask: URLSessionDownloadTask?
    private var moonshineDownloadTasks: [URLSessionDownloadTask] = []

    // Models directory path - ~/Library/Application Support/com.era-laboratories.voiceflow/models/
    private var modelsDir: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("com.era-laboratories.voiceflow/models")
    }

    // Config file path - ~/Library/Application Support/com.era-laboratories.voiceflow/config.toml
    private var configPath: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("com.era-laboratories.voiceflow/config.toml")
    }

    // Available models - filenames must match Rust config exactly
    private static let availableModels: [(id: String, name: String, filename: String, size: Float, repo: String)] = [
        ("qwen3-1.7b", "Qwen3 1.7B", "qwen3-1.7b-q4_k_m.gguf", 1.1, "Qwen/Qwen3-1.7B-GGUF"),
        ("qwen3-4b", "Qwen3 4B", "Qwen3-4B-Q4_K_M.gguf", 2.5, "Qwen/Qwen3-4B-GGUF"),
        ("smollm3-3b", "SmolLM3 3B", "SmolLM3-Q4_K_M.gguf", 1.92, "ggml-org/SmolLM3-3B-GGUF"),
        ("gemma2-2b", "Gemma 2 2B", "gemma-2-2b-it-Q4_K_M.gguf", 1.71, "bartowski/gemma-2-2b-it-GGUF"),
    ]

    init() {
        loadModels()
        loadCurrentModel()
        loadSttSettings()
    }

    // MARK: - STT Engine Management

    func loadSttSettings() {
        // Load current STT engine
        if let enginePtr = voiceflow_current_stt_engine() {
            let engineStr = String(cString: enginePtr)
            voiceflow_free_string(enginePtr)
            currentSttEngine = SttEngine(rawValue: engineStr) ?? .whisper
        }

        // Load current Moonshine model
        if let modelPtr = voiceflow_current_moonshine_model() {
            currentMoonshineModelId = String(cString: modelPtr)
            voiceflow_free_string(modelPtr)
        }

        // Load Moonshine models info
        let count = voiceflow_moonshine_model_count()
        var models: [MoonshineModel] = []
        for i in 0..<count {
            let info = voiceflow_moonshine_model_info(i)
            if let idPtr = info.id, let namePtr = info.display_name {
                let model = MoonshineModel(
                    id: String(cString: idPtr),
                    displayName: String(cString: namePtr),
                    sizeMB: info.size_mb,
                    isDownloaded: info.is_downloaded
                )
                models.append(model)
                voiceflow_free_moonshine_model_info(info)
            }
        }
        moonshineModels = models
    }

    func selectSttEngine(_ engine: SttEngine) {
        guard engine != currentSttEngine else { return }

        let success = engine.rawValue.withCString { cString in
            voiceflow_set_stt_engine(cString)
        }

        if success {
            currentSttEngine = engine
            needsRestart = true
        } else {
            downloadError = "Failed to set STT engine"
        }
    }

    func selectMoonshineModel(_ modelId: String) {
        guard modelId != currentMoonshineModelId else { return }

        let success = modelId.withCString { cString in
            voiceflow_set_moonshine_model(cString)
        }

        if success {
            currentMoonshineModelId = modelId
            needsRestart = true
        } else {
            downloadError = "Failed to set Moonshine model"
        }
    }

    func downloadMoonshineModel(_ modelId: String) {
        guard downloadingMoonshineModelId == nil else { return }

        // Moonshine models are on HuggingFace: UsefulSensors/moonshine
        // Files are in onnx/tiny/ or onnx/base/ directories
        let baseUrl = "https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/\(modelId)"
        let files = ["preprocess.onnx", "encode.onnx", "uncached_decode.onnx", "cached_decode.onnx"]
        // Tokenizer is in the separate moonshine-base/moonshine-tiny repos (same tokenizer for both)
        let tokenizerUrl = "https://huggingface.co/UsefulSensors/moonshine-\(modelId)/resolve/main/tokenizer.json"

        let modelDirName = "moonshine-\(modelId)"
        let destDir = modelsDir.appendingPathComponent(modelDirName)

        // Create model directory
        do {
            try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            downloadError = "Failed to create model directory: \(error.localizedDescription)"
            return
        }

        downloadingMoonshineModelId = modelId
        moonshineDownloadProgress = 0
        downloadError = nil

        // Download all files sequentially
        let allFiles = files + ["tokenizer.json"]
        var completedFiles = 0
        let totalFiles = allFiles.count

        func downloadNextFile(_ index: Int) {
            guard index < allFiles.count else {
                // All done
                DispatchQueue.main.async { [weak self] in
                    self?.downloadingMoonshineModelId = nil
                    self?.moonshineDownloadProgress = 1.0
                    self?.loadSttSettings() // Refresh model status
                }
                return
            }

            let filename = allFiles[index]
            let urlString = index < files.count ? "\(baseUrl)/\(filename)" : tokenizerUrl
            guard let url = URL(string: urlString) else {
                DispatchQueue.main.async { [weak self] in
                    self?.downloadError = "Invalid URL for \(filename)"
                    self?.downloadingMoonshineModelId = nil
                }
                return
            }

            let destFile = destDir.appendingPathComponent(filename)

            let task = URLSession.shared.downloadTask(with: url) { [weak self] tempURL, response, error in
                if let error = error {
                    DispatchQueue.main.async {
                        self?.downloadError = "Download failed: \(error.localizedDescription)"
                        self?.downloadingMoonshineModelId = nil
                    }
                    return
                }

                guard let tempURL = tempURL else {
                    DispatchQueue.main.async {
                        self?.downloadError = "Download failed: no file"
                        self?.downloadingMoonshineModelId = nil
                    }
                    return
                }

                do {
                    // Remove existing file if present
                    if FileManager.default.fileExists(atPath: destFile.path) {
                        try FileManager.default.removeItem(at: destFile)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: destFile)

                    completedFiles += 1
                    DispatchQueue.main.async {
                        self?.moonshineDownloadProgress = Double(completedFiles) / Double(totalFiles)
                    }

                    // Download next file
                    downloadNextFile(index + 1)
                } catch {
                    DispatchQueue.main.async {
                        self?.downloadError = "Failed to save \(filename): \(error.localizedDescription)"
                        self?.downloadingMoonshineModelId = nil
                    }
                }
            }
            task.resume()
        }

        // Start downloading first file
        downloadNextFile(0)
    }

    func loadModels() {
        // Create models directory if needed
        try? FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)

        models = Self.availableModels.map { info in
            let modelPath = modelsDir.appendingPathComponent(info.filename)
            // Check file exists AND is at least 100MB (to catch corrupt/failed downloads)
            var isDownloaded = false
            if FileManager.default.fileExists(atPath: modelPath.path) {
                if let attrs = try? FileManager.default.attributesOfItem(atPath: modelPath.path),
                   let fileSize = attrs[.size] as? Int64 {
                    isDownloaded = fileSize > 100_000_000  // 100MB minimum
                }
            }
            let downloadUrl = "https://huggingface.co/\(info.repo)/resolve/main/\(info.filename)"
            return LLMModel(
                id: info.id,
                displayName: info.name,
                filename: info.filename,
                sizeGB: info.size,
                downloadUrl: downloadUrl,
                isDownloaded: isDownloaded
            )
        }
    }

    func loadCurrentModel() {
        // Try to read current model from config file
        guard let configData = try? String(contentsOf: configPath, encoding: .utf8) else {
            currentModelId = "qwen3-1.7b" // Default
            return
        }

        // Simple TOML parsing for llm_model
        // Rust uses kebab-case: qwen3-4-b, qwen3-1-7-b, smol-lm3-3-b, gemma2-2-b
        if configData.contains("qwen3-4-b") {
            currentModelId = "qwen3-4b"
        } else if configData.contains("smol-lm3-3-b") {
            currentModelId = "smollm3-3b"
        } else if configData.contains("gemma2-2-b") {
            currentModelId = "gemma2-2b"
        } else {
            currentModelId = "qwen3-1.7b"
        }
    }

    func selectModel(_ modelId: String) {
        guard modelId != currentModelId else { return }
        guard let model = models.first(where: { $0.id == modelId }), model.isDownloaded else { return }

        // Use FFI function to properly save config (handles TOML format correctly)
        let success = modelId.withCString { cString in
            voiceflow_set_model(cString)
        }

        if success {
            currentModelId = modelId
            needsRestart = true
        } else {
            downloadError = "Failed to save config"
        }
    }

    func downloadModel(_ modelId: String) {
        guard !isDownloading else { return }
        guard let model = models.first(where: { $0.id == modelId }) else {
            downloadError = "Model not found"
            return
        }

        guard let url = URL(string: model.downloadUrl) else {
            downloadError = "Invalid download URL"
            return
        }

        // Ensure models directory exists
        do {
            try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            downloadError = "Failed to create models directory: \(error.localizedDescription)"
            return
        }

        let destinationURL = modelsDir.appendingPathComponent(model.filename)

        isDownloading = true
        downloadingModelId = modelId
        downloadProgress = 0
        downloadError = nil

        let session = URLSession(configuration: .default, delegate: DownloadDelegate(progress: { [weak self] progress in
            DispatchQueue.main.async {
                self?.downloadProgress = progress
            }
        }, completion: { [weak self] tempURL, error in
            DispatchQueue.main.async {
                self?.isDownloading = false
                self?.downloadingModelId = nil

                if let error = error {
                    self?.downloadError = error.localizedDescription
                    return
                }

                guard let tempURL = tempURL else {
                    self?.downloadError = "Download failed"
                    return
                }

                do {
                    // Move downloaded file to models directory
                    if FileManager.default.fileExists(atPath: destinationURL.path) {
                        try FileManager.default.removeItem(at: destinationURL)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: destinationURL)

                    // Update model status
                    if let index = self?.models.firstIndex(where: { $0.id == modelId }) {
                        self?.models[index].isDownloaded = true
                    }
                } catch {
                    self?.downloadError = "Failed to save model: \(error.localizedDescription)"
                }
            }
        }), delegateQueue: nil)

        downloadTask = session.downloadTask(with: url)
        downloadTask?.resume()
    }

    func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil
        isDownloading = false
        downloadingModelId = nil
        downloadProgress = 0
    }
}

// Download delegate to track progress
class DownloadDelegate: NSObject, URLSessionDownloadDelegate {
    let progressHandler: (Double) -> Void
    let completionHandler: (URL?, Error?) -> Void

    init(progress: @escaping (Double) -> Void, completion: @escaping (URL?, Error?) -> Void) {
        self.progressHandler = progress
        self.completionHandler = completion
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // URLSession deletes the temp file when this method returns, so we must copy it immediately
        let tempDir = FileManager.default.temporaryDirectory
        let permanentTemp = tempDir.appendingPathComponent(UUID().uuidString + ".gguf")

        do {
            try FileManager.default.copyItem(at: location, to: permanentTemp)
            completionHandler(permanentTemp, nil)
        } catch {
            completionHandler(nil, error)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
        progressHandler(progress)
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            completionHandler(nil, error)
        }
    }
}

// MARK: - Model Settings View

struct ModelSettingsView: View {
    @EnvironmentObject var modelManager: ModelManager

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Restart banner
                if modelManager.needsRestart {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("Restart required to apply model change")
                            .font(.callout)
                        Spacer()
                        Button("Restart Now") {
                            restartApp()
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }
                    .padding(10)
                    .background(Color.orange.opacity(0.15))
                    .cornerRadius(8)
                }

                // STT Engine Section
                GroupBox {
                    VStack(alignment: .leading, spacing: 12) {
                        // Engine selection
                        ForEach(SttEngine.allCases, id: \.self) { engine in
                            SttEngineRowView(
                                engine: engine,
                                isSelected: engine == modelManager.currentSttEngine,
                                onSelect: {
                                    modelManager.selectSttEngine(engine)
                                }
                            )

                            // Show Moonshine model options when Moonshine is selected
                            if engine == .moonshine && modelManager.currentSttEngine == .moonshine {
                                VStack(spacing: 8) {
                                    ForEach(modelManager.moonshineModels) { model in
                                        MoonshineModelRowView(
                                            model: model,
                                            isSelected: model.id == modelManager.currentMoonshineModelId,
                                            isDownloading: modelManager.downloadingMoonshineModelId == model.id,
                                            downloadProgress: modelManager.moonshineDownloadProgress,
                                            onSelect: {
                                                modelManager.selectMoonshineModel(model.id)
                                            },
                                            onDownload: {
                                                modelManager.downloadMoonshineModel(model.id)
                                            }
                                        )
                                    }
                                }
                                .padding(.leading, 32)
                                .padding(.top, 4)
                            }

                            if engine != SttEngine.allCases.last {
                                Divider()
                            }
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Speech-to-Text Engine", systemImage: "waveform")
                        .font(.headline)
                }

                // LLM Models Section
                GroupBox {
                    VStack(spacing: 12) {
                        ForEach(modelManager.models) { model in
                            ModelRowView(
                                model: model,
                                isSelected: model.id == modelManager.currentModelId,
                                isDownloading: modelManager.downloadingModelId == model.id,
                                downloadProgress: modelManager.downloadProgress,
                                onSelect: {
                                    if model.isDownloaded {
                                        modelManager.selectModel(model.id)
                                    }
                                },
                                onDownload: {
                                    modelManager.downloadModel(model.id)
                                }
                            )

                            if model.id != modelManager.models.last?.id {
                                Divider()
                            }
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("LLM Models", systemImage: "cpu")
                        .font(.headline)
                }

                // Error display
                if let error = modelManager.downloadError {
                    HStack {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundColor(.red)
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                }

                // Info section
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Larger models generally provide better formatting quality but require more memory and are slower to process.")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        HStack(spacing: 4) {
                            Image(systemName: "info.circle")
                                .foregroundColor(.blue)
                            Text("Models are downloaded from HuggingFace (Q4_K_M quantization)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.vertical, 4)
                } label: {
                    Label("Info", systemImage: "questionmark.circle")
                        .font(.headline)
                }
            }
            .padding()
        }
        .onAppear {
            modelManager.loadModels()
        }
    }

    private func restartApp() {
        let bundlePath = Bundle.main.bundlePath
        let script = """
            sleep 0.5
            open "\(bundlePath)"
            """
        let task = Process()
        task.launchPath = "/bin/bash"
        task.arguments = ["-c", script]
        try? task.run()
        NSApp.terminate(nil)
    }
}

struct ModelRowView: View {
    let model: LLMModel
    let isSelected: Bool
    let isDownloading: Bool
    let downloadProgress: Double
    let onSelect: () -> Void
    let onDownload: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            // Selection indicator (radio button)
            ZStack {
                Circle()
                    .stroke(isSelected ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: 2)
                    .frame(width: 20, height: 20)

                if isSelected {
                    Circle()
                        .fill(Color.accentColor)
                        .frame(width: 12, height: 12)
                }
            }

            // Model info
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(model.displayName)
                        .fontWeight(.medium)
                    if isSelected {
                        Text("Active")
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.2))
                            .cornerRadius(4)
                    }
                }

                Text(String(format: "%.1f GB", model.sizeGB))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Download/Status indicator
            if model.isDownloaded {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            } else if isDownloading {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.7)
                    Text("\(Int(downloadProgress * 100))%")
                        .font(.caption)
                        .monospacedDigit()
                }
            } else {
                Button(action: onDownload) {
                    Label("Download", systemImage: "arrow.down.circle")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            // Allow selecting downloaded models by tapping anywhere on the row
            if model.isDownloaded && !isSelected {
                onSelect()
            }
        }
        .opacity(model.isDownloaded || isDownloading ? 1.0 : 0.7)
    }
}

// MARK: - STT Engine Row View

struct SttEngineRowView: View {
    let engine: SttEngine
    let isSelected: Bool
    let onSelect: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            // Selection indicator (radio button)
            ZStack {
                Circle()
                    .stroke(isSelected ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: 2)
                    .frame(width: 20, height: 20)

                if isSelected {
                    Circle()
                        .fill(Color.accentColor)
                        .frame(width: 12, height: 12)
                }
            }

            // Engine info
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(engine.displayName)
                        .fontWeight(.medium)
                    if isSelected {
                        Text("Active")
                            .font(.caption2)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.2))
                            .cornerRadius(4)
                    }
                }

                Text(engine.description)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture {
            if !isSelected {
                onSelect()
            }
        }
    }
}

// MARK: - Moonshine Model Row View

struct MoonshineModelRowView: View {
    let model: MoonshineModel
    let isSelected: Bool
    let isDownloading: Bool
    let downloadProgress: Double
    let onSelect: () -> Void
    let onDownload: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            // Selection indicator (smaller radio button)
            ZStack {
                Circle()
                    .stroke(isSelected ? Color.accentColor : Color.secondary.opacity(0.3), lineWidth: 1.5)
                    .frame(width: 16, height: 16)

                if isSelected {
                    Circle()
                        .fill(Color.accentColor)
                        .frame(width: 10, height: 10)
                }
            }

            // Model info
            VStack(alignment: .leading, spacing: 1) {
                Text(model.displayName)
                    .font(.callout)
                    .fontWeight(isSelected ? .medium : .regular)

                Text("\(model.sizeMB) MB")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Download status / button
            if model.isDownloaded {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .font(.caption)
            } else if isDownloading {
                HStack(spacing: 6) {
                    ProgressView()
                        .scaleEffect(0.6)
                    Text("\(Int(downloadProgress * 100))%")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            } else {
                Button(action: onDownload) {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.down.circle")
                        Text("Download")
                    }
                    .font(.caption)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.mini)
            }
        }
        .padding(.vertical, 2)
        .contentShape(Rectangle())
        .onTapGesture {
            if !isSelected && model.isDownloaded {
                onSelect()
            }
        }
        .opacity(model.isDownloaded || isDownloading ? 1.0 : 0.8)
    }
}

// MARK: - Overlay State

class OverlayState: ObservableObject {
    enum RecordingState {
        case idle
        case recording
        case processing
    }

    @Published var state: RecordingState = .idle
    @Published var audioLevel: Float = 0
}

// MARK: - Recording Overlay View

struct RecordingOverlayView: View {
    @ObservedObject var state: OverlayState
    @State private var animateGradient = false

    // Brand colors from logo
    private let gradientColors: [Color] = [
        Color(red: 0.0, green: 0.85, blue: 0.85),   // Cyan
        Color(red: 0.2, green: 0.6, blue: 0.9),    // Blue
        Color(red: 0.4, green: 0.8, blue: 0.7),    // Teal
        Color(red: 0.0, green: 0.7, blue: 0.9),    // Light blue
        Color(red: 0.0, green: 0.85, blue: 0.85),   // Cyan (loop)
    ]

    var body: some View {
        HStack(spacing: 12) {
            // Logo
            logoView

            // Waveform bars or processing indicator
            if state.state == .recording {
                WaveformView(audioLevel: state.audioLevel)
                    .frame(width: 100, height: 45)
            } else if state.state == .processing {
                processingView
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(backgroundView)
        .onAppear {
            withAnimation(.linear(duration: 3.0).repeatForever(autoreverses: false)) {
                animateGradient = true
            }
        }
    }

    @ViewBuilder
    private var logoView: some View {
        if let iconPath = Bundle.main.path(forResource: "MenuBarIcon", ofType: "png"),
           let nsImage = NSImage(contentsOfFile: iconPath) {
            Image(nsImage: nsImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 28, height: 28)
                .opacity(state.state == .recording ? 1.0 : 0.7)
        } else {
            Image(systemName: "mic.fill")
                .font(.system(size: 24))
                .foregroundColor(.white)
        }
    }

    private var processingView: some View {
        HStack(spacing: 8) {
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                .scaleEffect(0.8)
            Text("Processing...")
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(.white.opacity(0.9))
        }
    }

    private var backgroundView: some View {
        ZStack {
            if state.state == .recording {
                // Animated gradient for recording
                AnimatedGradientBackground(animate: animateGradient, colors: gradientColors)
            } else {
                // Dark background for processing/idle
                Color.black.opacity(0.85)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 30))
        .shadow(color: .black.opacity(0.3), radius: 10, x: 0, y: 5)
    }
}

// MARK: - Animated Gradient Background

struct AnimatedGradientBackground: View {
    let animate: Bool
    let colors: [Color]

    var body: some View {
        LinearGradient(
            colors: colors,
            startPoint: animate ? .topLeading : .bottomTrailing,
            endPoint: animate ? .bottomTrailing : .topLeading
        )
        .hueRotation(.degrees(animate ? 45 : 0))
        .opacity(0.95)
    }
}

// MARK: - Waveform View

struct WaveformView: View {
    let audioLevel: Float

    private let barCount = 5
    private let barSpacing: CGFloat = 4

    var body: some View {
        HStack(spacing: barSpacing) {
            ForEach(0..<barCount, id: \.self) { index in
                WaveformBar(audioLevel: audioLevel, index: index)
            }
        }
    }
}

struct WaveformBar: View {
    let audioLevel: Float
    let index: Int

    private let baseHeight: CGFloat = 6
    private let maxHeight: CGFloat = 40  // Much bigger amplitude

    private var computedHeight: CGFloat {
        // Amplify the audio level for more visible movement
        let amplifiedLevel = min(1.0, CGFloat(audioLevel) * 3.0)
        let variation = sin(Double(index) * 1.5) * 0.4 + 0.6
        let level = amplifiedLevel * CGFloat(variation)
        return baseHeight + level * (maxHeight - baseHeight)
    }

    var body: some View {
        RoundedRectangle(cornerRadius: 3)
            .fill(Color.white)
            .frame(width: 6, height: computedHeight)
            .animation(.easeInOut(duration: 0.08), value: audioLevel)
    }
}
