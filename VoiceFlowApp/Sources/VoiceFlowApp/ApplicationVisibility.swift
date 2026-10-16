import AppKit

@MainActor
enum ApplicationVisibility {
    static let preferenceKey = "alwaysShowInDock"
    static let changed = Notification.Name("VoiceFlow.applicationVisibilityChanged")
    static var alwaysShow: Bool {
        UserDefaults.standard.object(forKey: preferenceKey) == nil
            || UserDefaults.standard.bool(forKey: preferenceKey)
    }
    static func apply() {
        let showWithWindows = UserDefaults.standard.object(forKey: "dockIconWhileSettingsOpen") == nil
            || UserDefaults.standard.bool(forKey: "dockIconWhileSettingsOpen")
        let hasWindow = NSApp.windows.contains { $0.isVisible && $0.canBecomeMain && !($0 is NSPanel) }
        NSApp.setActivationPolicy(alwaysShow || (showWithWindows && hasWindow) ? .regular : .accessory)
    }
    static func installMenuIfNeeded(open: AnyObject, action: Selector) {
        // Settings-only SwiftUI apps do not always create a useful app menu.
        guard NSApp.mainMenu == nil || NSApp.mainMenu?.items.isEmpty == true else { return }
        let menu = NSMenu()
        let root = NSMenuItem()
        let appMenu = NSMenu(title: "VoiceFlow")
        root.submenu = appMenu
        menu.addItem(root)
        appMenu.addItem(withTitle: "About VoiceFlow", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        let settings = appMenu.addItem(withTitle: "Open VoiceFlow…", action: action, keyEquivalent: ",")
        settings.target = open
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit VoiceFlow", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let editRoot = NSMenuItem()
        editRoot.submenu = NSMenu(title: "Edit")
        menu.addItem(editRoot)
        for (title, selector, key) in [("Undo", "undo:", "z"), ("Cut", "cut:", "x"), ("Copy", "copy:", "c"), ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")] {
            editRoot.submenu?.addItem(withTitle: title, action: Selector(selector), keyEquivalent: key)
        }
        NSApp.mainMenu = menu
    }
}
