import AppKit
import Carbon.HIToolbox
import SwiftUI

// MARK: - Model

/// A bindable global hotkey action. The `hotkeyID` matches the numeric id used
/// by `GlobalHotkeyManager` / `setupHotkey()`.
enum HotkeyAction: String, CaseIterable, Identifiable {
    case dictate
    case visualDictate
    case edit
    case cycleFormatting

    var id: String { rawValue }

    var hotkeyID: UInt32 {
        switch self {
        case .dictate:         return 1
        case .visualDictate:   return 2
        case .edit:            return 3
        case .cycleFormatting: return 4
        }
    }

    var title: String {
        switch self {
        case .dictate:         return "Dictate"
        case .visualDictate:   return "Dictate with screen context"
        case .edit:            return "Edit / repeat last dictation"
        case .cycleFormatting: return "Cycle formatting level"
        }
    }

    var subtitle: String {
        switch self {
        case .dictate:         return "Hold to record, release to transcribe and paste"
        case .visualDictate:   return "Hold to capture your screen + record for context-aware text"
        case .edit:            return "Hold and re-say to replace, or speak an instruction"
        case .cycleFormatting: return "Tap to cycle Minimal → Moderate → Intent-Aware → Aggressive"
        }
    }

    /// The factory binding, used as the fallback and for "Reset".
    var defaultBinding: HotkeyBinding {
        switch self {
        case .dictate:
            return HotkeyBinding(keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey))
        case .visualDictate:
            return HotkeyBinding(keyCode: UInt32(kVK_Space), modifiers: UInt32(controlKey | optionKey))
        case .edit:
            return HotkeyBinding(keyCode: UInt32(kVK_Space), modifiers: UInt32(optionKey | shiftKey))
        case .cycleFormatting:
            return HotkeyBinding(keyCode: UInt32(kVK_Space), modifiers: UInt32(controlKey | optionKey | shiftKey))
        }
    }
}

/// A key + Carbon modifier mask. `modifiers` uses Carbon constants
/// (`optionKey` / `shiftKey` / `controlKey` / `cmdKey`) so it can be passed
/// straight to `RegisterEventHotKey`.
struct HotkeyBinding: Codable, Equatable {
    var keyCode: UInt32
    var modifiers: UInt32
}

// MARK: - Store

/// Persisted, observable store of the global hotkey bindings. `AppDelegate`
/// reads it when registering hotkeys and re-registers on `changedNotification`.
/// While the user is recording a new shortcut it broadcasts begin/end so the
/// app can suspend the live hotkeys (otherwise pressing the current shortcut
/// would fire its action instead of being captured).
final class HotkeyStore: ObservableObject {
    static let shared = HotkeyStore()

    static let changedNotification = Notification.Name("VoiceFlow.hotkeysChanged")
    static let recordingBeganNotification = Notification.Name("VoiceFlow.hotkeyRecordingBegan")
    static let recordingEndedNotification = Notification.Name("VoiceFlow.hotkeyRecordingEnded")

    @Published private(set) var bindings: [String: HotkeyBinding] = [:]

    private init() {
        for action in HotkeyAction.allCases {
            bindings[action.id] = Self.load(action) ?? action.defaultBinding
        }
    }

    func binding(for action: HotkeyAction) -> HotkeyBinding {
        bindings[action.id] ?? action.defaultBinding
    }

    func setBinding(_ binding: HotkeyBinding, for action: HotkeyAction) {
        bindings[action.id] = binding
        Self.save(binding, action)
        NotificationCenter.default.post(name: Self.changedNotification, object: nil)
    }

    func reset(_ action: HotkeyAction) {
        setBinding(action.defaultBinding, for: action)
    }

    func resetAll() {
        for action in HotkeyAction.allCases {
            bindings[action.id] = action.defaultBinding
            Self.save(action.defaultBinding, action)
        }
        NotificationCenter.default.post(name: Self.changedNotification, object: nil)
    }

    /// Another action (if any) already bound to the exact same key+modifiers.
    func conflict(for binding: HotkeyBinding, excluding: HotkeyAction) -> HotkeyAction? {
        HotkeyAction.allCases.first { $0 != excluding && self.binding(for: $0) == binding }
    }

    // Persistence — one JSON blob per action under "hotkey.<action>".
    private static func key(_ action: HotkeyAction) -> String { "hotkey.\(action.rawValue)" }

    private static func load(_ action: HotkeyAction) -> HotkeyBinding? {
        guard let data = UserDefaults.standard.data(forKey: key(action)) else { return nil }
        return try? JSONDecoder().decode(HotkeyBinding.self, from: data)
    }

    private static func save(_ binding: HotkeyBinding, _ action: HotkeyAction) {
        if let data = try? JSONEncoder().encode(binding) {
            UserDefaults.standard.set(data, forKey: key(action))
        }
    }
}

// MARK: - Formatting + validation

enum HotkeyFormatter {

    /// Carbon modifier mask → glyphs in the canonical macOS order ⌃⌥⇧⌘.
    static func modifierGlyphs(_ modifiers: UInt32) -> String {
        var s = ""
        if modifiers & UInt32(controlKey) != 0 { s += "⌃" }
        if modifiers & UInt32(optionKey)  != 0 { s += "⌥" }
        if modifiers & UInt32(shiftKey)   != 0 { s += "⇧" }
        if modifiers & UInt32(cmdKey)     != 0 { s += "⌘" }
        return s
    }

    static func display(_ binding: HotkeyBinding) -> String {
        let mods = modifierGlyphs(binding.modifiers)
        let key = keyName(binding.keyCode)
        return mods.isEmpty ? key : "\(mods) \(key)"
    }

    static func carbonModifiers(from flags: NSEvent.ModifierFlags) -> UInt32 {
        var m: UInt32 = 0
        if flags.contains(.command) { m |= UInt32(cmdKey) }
        if flags.contains(.option)  { m |= UInt32(optionKey) }
        if flags.contains(.control) { m |= UInt32(controlKey) }
        if flags.contains(.shift)   { m |= UInt32(shiftKey) }
        return m
    }

    static func hasCommandControlOption(_ modifiers: UInt32) -> Bool {
        modifiers & (UInt32(controlKey) | UInt32(optionKey) | UInt32(cmdKey)) != 0
    }

    /// Typing keys (letters, digits, space, punctuation, return, etc.) must
    /// carry at least one of ⌃⌥⌘ or, as a global hotkey, they'd swallow normal
    /// input everywhere. Function keys and the like may be bound bare.
    static func requiresModifier(_ keyCode: UInt32) -> Bool {
        !bareAllowedKeyCodes.contains(keyCode)
    }

    static func isValid(_ binding: HotkeyBinding) -> Bool {
        requiresModifier(binding.keyCode) ? hasCommandControlOption(binding.modifiers) : true
    }

    /// Function keys can stand alone; everything else needs a real modifier.
    private static let bareAllowedKeyCodes: Set<UInt32> = [
        UInt32(kVK_F1), UInt32(kVK_F2), UInt32(kVK_F3), UInt32(kVK_F4),
        UInt32(kVK_F5), UInt32(kVK_F6), UInt32(kVK_F7), UInt32(kVK_F8),
        UInt32(kVK_F9), UInt32(kVK_F10), UInt32(kVK_F11), UInt32(kVK_F12),
        UInt32(kVK_F13), UInt32(kVK_F14), UInt32(kVK_F15), UInt32(kVK_F16),
        UInt32(kVK_F17), UInt32(kVK_F18), UInt32(kVK_F19), UInt32(kVK_F20),
    ]

    /// Carbon virtual key code → human-readable name.
    static func keyName(_ keyCode: UInt32) -> String {
        if let name = specialNames[Int(keyCode)] { return name }
        if let ansi = ansiNames[Int(keyCode)] { return ansi }
        return "Key \(keyCode)"
    }

    private static let specialNames: [Int: String] = [
        kVK_Space: "Space", kVK_Return: "Return", kVK_Tab: "Tab",
        kVK_Delete: "Delete", kVK_ForwardDelete: "⌦", kVK_Escape: "Esc",
        kVK_LeftArrow: "←", kVK_RightArrow: "→", kVK_UpArrow: "↑", kVK_DownArrow: "↓",
        kVK_Home: "Home", kVK_End: "End", kVK_PageUp: "Page Up", kVK_PageDown: "Page Down",
        kVK_Help: "Help", kVK_ANSI_KeypadEnter: "Enter",
        kVK_F1: "F1", kVK_F2: "F2", kVK_F3: "F3", kVK_F4: "F4", kVK_F5: "F5",
        kVK_F6: "F6", kVK_F7: "F7", kVK_F8: "F8", kVK_F9: "F9", kVK_F10: "F10",
        kVK_F11: "F11", kVK_F12: "F12", kVK_F13: "F13", kVK_F14: "F14", kVK_F15: "F15",
        kVK_F16: "F16", kVK_F17: "F17", kVK_F18: "F18", kVK_F19: "F19", kVK_F20: "F20",
    ]

    private static let ansiNames: [Int: String] = [
        kVK_ANSI_A: "A", kVK_ANSI_B: "B", kVK_ANSI_C: "C", kVK_ANSI_D: "D",
        kVK_ANSI_E: "E", kVK_ANSI_F: "F", kVK_ANSI_G: "G", kVK_ANSI_H: "H",
        kVK_ANSI_I: "I", kVK_ANSI_J: "J", kVK_ANSI_K: "K", kVK_ANSI_L: "L",
        kVK_ANSI_M: "M", kVK_ANSI_N: "N", kVK_ANSI_O: "O", kVK_ANSI_P: "P",
        kVK_ANSI_Q: "Q", kVK_ANSI_R: "R", kVK_ANSI_S: "S", kVK_ANSI_T: "T",
        kVK_ANSI_U: "U", kVK_ANSI_V: "V", kVK_ANSI_W: "W", kVK_ANSI_X: "X",
        kVK_ANSI_Y: "Y", kVK_ANSI_Z: "Z",
        kVK_ANSI_0: "0", kVK_ANSI_1: "1", kVK_ANSI_2: "2", kVK_ANSI_3: "3",
        kVK_ANSI_4: "4", kVK_ANSI_5: "5", kVK_ANSI_6: "6", kVK_ANSI_7: "7",
        kVK_ANSI_8: "8", kVK_ANSI_9: "9",
        kVK_ANSI_Minus: "-", kVK_ANSI_Equal: "=", kVK_ANSI_LeftBracket: "[",
        kVK_ANSI_RightBracket: "]", kVK_ANSI_Backslash: "\\", kVK_ANSI_Semicolon: ";",
        kVK_ANSI_Quote: "'", kVK_ANSI_Comma: ",", kVK_ANSI_Period: ".",
        kVK_ANSI_Slash: "/", kVK_ANSI_Grave: "`",
    ]
}

// MARK: - Recorder UI

/// One row in the Hotkey settings card: shows the action, its current binding,
/// and a Record button that captures the next keystroke (any key + any
/// combination of ⌃⌥⇧⌘). Esc cancels recording.
struct HotkeyRecorderRow: View {
    let action: HotkeyAction
    @ObservedObject private var store = HotkeyStore.shared
    @State private var recording = false
    @State private var monitor: Any?
    @State private var error: String?

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(action.title)
                    .font(.system(size: 13))
                    .foregroundColor(.white.opacity(0.9))
                Text(error ?? action.subtitle)
                    .font(.system(size: 11))
                    .foregroundColor(error == nil ? .white.opacity(0.45)
                                                  : Color(red: 1.0, green: 0.5, blue: 0.45))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 8)

            Text(recording ? "Press keys…" : HotkeyFormatter.display(store.binding(for: action)))
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Capsule().fill(Color.white.opacity(recording ? 0.18 : 0.08)))
                .overlay(
                    Capsule().stroke(recording ? Color(red: 0.45, green: 0.78, blue: 1.0).opacity(0.7)
                                               : Color.clear, lineWidth: 1)
                )
                .foregroundColor(.white.opacity(0.9))
                .frame(minWidth: 86)

            Button(recording ? "Cancel" : "Record") {
                recording ? stop() : start()
            }
            .buttonStyle(.borderless)
            .font(.system(size: 12, weight: .medium))
            .foregroundColor(recording ? Color(red: 1.0, green: 0.5, blue: 0.45)
                                       : Color(red: 0.45, green: 0.78, blue: 1.0))

            if store.binding(for: action) != action.defaultBinding {
                Button {
                    store.reset(action)
                } label: {
                    Image(systemName: "arrow.uturn.backward")
                        .font(.system(size: 11))
                        .foregroundColor(.white.opacity(0.5))
                }
                .buttonStyle(.borderless)
                .help("Reset to default")
            }
        }
        .padding(.vertical, 4)
        .onDisappear { stop() }
    }

    private func start() {
        error = nil
        recording = true
        NotificationCenter.default.post(name: HotkeyStore.recordingBeganNotification, object: nil)
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown]) { event in
            handle(event)
            return nil // swallow — don't let the keystroke reach the UI
        }
    }

    private func stop() {
        if let m = monitor { NSEvent.removeMonitor(m) }
        monitor = nil
        if recording {
            recording = false
            NotificationCenter.default.post(name: HotkeyStore.recordingEndedNotification, object: nil)
        }
    }

    private func handle(_ event: NSEvent) {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        // Bare Esc cancels.
        if event.keyCode == UInt16(kVK_Escape) && !flags.contains(.command)
            && !flags.contains(.option) && !flags.contains(.control) && !flags.contains(.shift) {
            stop()
            return
        }

        let binding = HotkeyBinding(
            keyCode: UInt32(event.keyCode),
            modifiers: HotkeyFormatter.carbonModifiers(from: flags)
        )

        guard HotkeyFormatter.isValid(binding) else {
            error = "Add at least one of ⌃ ⌥ ⌘ to this key."
            return // keep recording
        }
        if let other = store.conflict(for: binding, excluding: action) {
            error = "Already used by “\(other.title)”."
            return
        }
        error = nil
        store.setBinding(binding, for: action)
        stop()
    }
}

/// The full Hotkey settings card body: one recorder row per action plus a
/// reset-all control and a hint.
struct HotkeyRecorderList: View {
    @ObservedObject private var store = HotkeyStore.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(HotkeyAction.allCases) { action in
                HotkeyRecorderRow(action: action)
                if action != HotkeyAction.allCases.last {
                    Divider().overlay(Color.white.opacity(0.06))
                }
            }
            HStack {
                Text("Tip: combine ⌃ ⌥ ⇧ ⌘ with any key. Press Esc to cancel recording.")
                    .font(.system(size: 11))
                    .foregroundColor(.white.opacity(0.4))
                Spacer()
                if HotkeyAction.allCases.contains(where: { store.binding(for: $0) != $0.defaultBinding }) {
                    Button("Reset all") { store.resetAll() }
                        .buttonStyle(.borderless)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(.white.opacity(0.6))
                }
            }
            .padding(.top, 6)
        }
    }
}
