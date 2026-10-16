import AppKit
import Foundation

/// Keeps macOS's own Globe/fn-key action out of the way while a VoiceFlow
/// hotkey is bound to the bare 🌐 fn key.
///
/// macOS gives the Globe key a system action — Emoji & Symbols, Dictation, or
/// input-source switching — configured in System Settings › Keyboard as
/// "Press 🌐 key to" and stored as `AppleFnUsageType` in the
/// `com.apple.HIToolbox` domain. With that set, every press-and-release of
/// the key would also pop the emoji picker (or start Apple's dictation) on
/// top of VoiceFlow's hold-to-dictate. So while any binding uses the bare fn
/// key we switch the system action to "Do Nothing", remember what it was,
/// and put it back when the binding goes away or the app quits.
///
/// The live switch goes through HIToolbox's `TISUpdateFnUsageType` — the
/// same call System Settings makes, resolved at runtime with `dlsym` so a
/// missing symbol degrades to a plain preference write plus a hint in the
/// Settings UI instead of a crash.
@MainActor
final class GlobeKeyController {
    static let shared = GlobeKeyController()

    /// Values of `AppleFnUsageType`, matching the "Press 🌐 key to" menu.
    enum FnUsage: Int {
        case doNothing = 0
        case changeInputSource = 1
        case showEmoji = 2
        case startDictation = 3

        var title: String {
            switch self {
            case .doNothing:         return "Do Nothing"
            case .changeInputSource: return "Change Input Source"
            case .showEmoji:         return "Show Emoji & Symbols"
            case .startDictation:    return "Start Dictation"
            }
        }
    }

    /// UserDefaults key holding the user's original Globe action while we
    /// hold it at Do Nothing. Survives a crash, so the next launch still
    /// knows what to restore.
    static let originalUsageKey = "globeKey.originalFnUsageType"

    private let defaults: UserDefaults
    private let readUsage: (() -> FnUsage?)?
    private let writeUsage: ((FnUsage) -> Void)?

    private typealias GetFn = @convention(c) () -> Int
    private typealias UpdateFn = @convention(c) (Int) -> Int
    private let getFn: GetFn?
    private let updateFn: UpdateFn?

    private init() {
        defaults = .standard
        readUsage = nil
        writeUsage = nil
        let path = "/System/Library/Frameworks/Carbon.framework/Frameworks/HIToolbox.framework/HIToolbox"
        if let handle = dlopen(path, RTLD_NOW) {
            getFn = dlsym(handle, "TISGetFnUsageType").map { unsafeBitCast($0, to: GetFn.self) }
            updateFn = dlsym(handle, "TISUpdateFnUsageType").map { unsafeBitCast($0, to: UpdateFn.self) }
        } else {
            getFn = nil
            updateFn = nil
        }
    }

    /// Isolated preference storage and a simulated system action for lifecycle tests.
    init(defaults: UserDefaults, readUsage: @escaping () -> FnUsage?, writeUsage: @escaping (FnUsage) -> Void) {
        self.defaults = defaults
        self.readUsage = readUsage
        self.writeUsage = writeUsage
        getFn = nil
        updateFn = nil
    }

    // MARK: - State

    /// What the Globe key currently does system-wide, or nil if unknown.
    var currentUsage: FnUsage? {
        if let readUsage { return readUsage() }
        if let getFn {
            return FnUsage(rawValue: getFn() & 0xFFFF_FFFF)
        }
        let value = CFPreferencesCopyAppValue(
            "AppleFnUsageType" as CFString, "com.apple.HIToolbox" as CFString
        ) as? Int
        return value.flatMap(FnUsage.init(rawValue:))
    }

    /// The action we replaced, if we're currently holding the key at Do Nothing.
    var originalUsage: FnUsage? {
        guard let raw = defaults.object(forKey: Self.originalUsageKey) as? Int else { return nil }
        return FnUsage(rawValue: raw)
    }

    /// True when a bare-fn hotkey would collide with the system's Globe action
    /// right now (i.e. our switch didn't take, or someone flipped it back).
    var systemActionConflicts: Bool {
        guard let usage = currentUsage else { return false }
        return usage != .doNothing
    }

    // MARK: - Claim / restore

    /// Re-evaluate after the hotkey bindings change: hold the Globe action at
    /// Do Nothing while any binding is the bare fn key, otherwise give it back.
    func sync(bindings: [HotkeyBinding]) {
        if bindings.contains(where: { $0.modifierOnly == .function }) {
            claim()
        } else {
            restore()
        }
    }

    /// Make sure the system Globe action is Do Nothing, remembering the
    /// previous value the first time. Cheap enough to call on every fn press,
    /// which is how we cope with another app (or the user) flipping the
    /// setting back mid-session.
    func claim() {
        guard let current = currentUsage, current != .doNothing else { return }
        // A nonzero action while claimed is a newer external choice. Restore
        // that choice, rather than overwriting it with the first value we saw.
        defaults.set(current.rawValue, forKey: Self.originalUsageKey)
        apply(.doNothing)
        NSLog("[VoiceFlow] Globe key: system action was '%@', set to Do Nothing while fn is a VoiceFlow hotkey", current.title)
    }

    /// Put the user's original Globe action back — but only if it's still the
    /// Do Nothing we set, so we never clobber a change they made meanwhile.
    func restore() {
        guard let original = originalUsage else { return }
        guard let current = currentUsage else { return }
        guard current == .doNothing, original != .doNothing else {
            defaults.removeObject(forKey: Self.originalUsageKey)
            return
        }
        apply(original)
        // Keep the recovery record if the system write failed, so a later
        // restore or next launch can retry instead of losing the user's choice.
        guard currentUsage == original else { return }
        defaults.removeObject(forKey: Self.originalUsageKey)
        NSLog("[VoiceFlow] Globe key: restored system action to '%@'", original.title)
    }

    private func apply(_ usage: FnUsage) {
        if let writeUsage {
            writeUsage(usage)
            return
        }
        // TISUpdateFnUsageType flips the live setting in every process (it's
        // what System Settings calls). Also persist the preference so the
        // value survives even if the private call is unavailable.
        if let updateFn {
            _ = updateFn(usage.rawValue)
        }
        CFPreferencesSetAppValue(
            "AppleFnUsageType" as CFString, usage.rawValue as CFNumber, "com.apple.HIToolbox" as CFString
        )
        CFPreferencesAppSynchronize("com.apple.HIToolbox" as CFString)
    }

    /// System Settings › Keyboard, where "Press 🌐 key to" lives.
    static func openKeyboardSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.Keyboard-Settings.extension") {
            NSWorkspace.shared.open(url)
        }
    }
}
