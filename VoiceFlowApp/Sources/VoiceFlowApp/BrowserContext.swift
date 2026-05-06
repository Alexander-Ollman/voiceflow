import AppKit
import Foundation

/// Extracts the URL of the active tab in the frontmost browser via AppleScript.
/// Used to look up site-specific persona overrides (e.g. gmail.com → Professional Email).
///
/// First call against a given browser triggers macOS's Automation permission prompt.
/// If the user denies, calls return nil and the dictation flow falls back to the
/// bundle-id persona.
enum BrowserContext {
    /// Bundle IDs of supported browsers and the AppleScript app name + tab accessor
    /// they expect. All Chromium-based browsers share Chrome's script syntax.
    private struct BrowserDriver {
        let appName: String
        let isChromium: Bool   // true → "active tab of front window", false (Safari) → "current tab"
    }

    private static let drivers: [String: BrowserDriver] = [
        "com.apple.Safari": .init(appName: "Safari", isChromium: false),
        "com.google.Chrome": .init(appName: "Google Chrome", isChromium: true),
        "com.google.Chrome.canary": .init(appName: "Google Chrome Canary", isChromium: true),
        "com.brave.Browser": .init(appName: "Brave Browser", isChromium: true),
        "com.brave.Browser.beta": .init(appName: "Brave Browser Beta", isChromium: true),
        "com.vivaldi.Vivaldi": .init(appName: "Vivaldi", isChromium: true),
        "com.microsoft.edgemac": .init(appName: "Microsoft Edge", isChromium: true),
        "com.thebrowser.Browser": .init(appName: "Arc", isChromium: true),  // Arc uses Chromium-style
        "company.thebrowser.Browser": .init(appName: "Arc", isChromium: true),  // alt id
    ]

    static func isBrowser(bundleId: String) -> Bool {
        drivers[bundleId] != nil
    }

    /// Returns the URL of the active tab of the given browser, or nil if it can't
    /// be retrieved (browser not running, no window, automation denied, etc.).
    static func currentURL(forBundleId bundleId: String) -> URL? {
        guard let driver = drivers[bundleId] else { return nil }

        let tabAccessor = driver.isChromium ? "active tab of front window" : "current tab of front window"
        let script = """
        tell application "\(driver.appName)"
            if (count of windows) is 0 then return ""
            return URL of \(tabAccessor)
        end tell
        """

        var error: NSDictionary?
        guard let appleScript = NSAppleScript(source: script) else { return nil }
        let result = appleScript.executeAndReturnError(&error)

        if let err = error {
            // -1743 = automation not authorized; -609/-1712 = app not running / timeout
            let code = (err["NSAppleScriptErrorNumber"] as? Int) ?? 0
            if code != -1728 { // -1728 is "no window" — silent
                NSLog("[BrowserContext] AppleScript error %d: %@",
                      code, (err["NSAppleScriptErrorMessage"] as? String) ?? "")
            }
            return nil
        }

        guard let urlString = result.stringValue, !urlString.isEmpty else { return nil }
        return URL(string: urlString)
    }

    /// Extract a normalized hostname (lowercased, leading "www." stripped) from a URL.
    static func hostname(from url: URL) -> String? {
        guard var host = url.host?.lowercased() else { return nil }
        if host.hasPrefix("www.") {
            host = String(host.dropFirst(4))
        }
        return host
    }
}

// MARK: - Browser site rules (hostname → persona)

struct BrowserSiteRule: Codable, Identifiable, Equatable, Hashable {
    let id: String              // UUID
    var hostnamePattern: String // exact host or "*.suffix" wildcard
    var personaId: String       // references Persona.id
    let createdAt: Date
    var isBuiltIn: Bool

    /// Match a hostname against this rule's pattern.
    func matches(_ hostname: String) -> Bool {
        let pat = hostnamePattern.lowercased()
        let host = hostname.lowercased()
        if pat.hasPrefix("*.") {
            let suffix = String(pat.dropFirst(2))
            return host == suffix || host.hasSuffix("." + suffix)
        }
        return host == pat
    }
}

class BrowserSiteRulesManager: ObservableObject {
    static let shared = BrowserSiteRulesManager()

    @Published var rules: [BrowserSiteRule] = []

    private let storageKey = "voiceflow.browserSiteRules"

    init() {
        load()
        if rules.isEmpty {
            rules = Self.seedRules()
            save()
        }
    }

    func load() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([BrowserSiteRule].self, from: data) {
            rules = decoded
        }
    }

    func save() {
        if let encoded = try? JSONEncoder().encode(rules) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
    }

    func upsert(_ rule: BrowserSiteRule) {
        if let idx = rules.firstIndex(where: { $0.id == rule.id }) {
            rules[idx] = rule
        } else {
            rules.append(rule)
        }
        save()
    }

    func delete(_ rule: BrowserSiteRule) {
        rules.removeAll { $0.id == rule.id }
        save()
    }

    /// Find the highest-priority rule that matches a hostname.
    /// Exact-match rules win over wildcards; first match otherwise.
    func match(hostname: String) -> BrowserSiteRule? {
        let exact = rules.first { !$0.hostnamePattern.contains("*") && $0.matches(hostname) }
        if let exact = exact { return exact }
        return rules.first { $0.matches(hostname) }
    }

    /// Resolve a hostname to a persona-prompt fragment, or empty if no rule matches.
    func promptForHostname(_ hostname: String) -> String {
        guard let rule = match(hostname: hostname),
              let persona = PersonaManager.shared.persona(byId: rule.personaId),
              !persona.prompt.isEmpty else {
            return ""
        }
        return "\n[PERSONA: \(persona.name) — for site \(hostname)]\n\(persona.prompt)"
    }

    /// Seeded starter rules. Looked up against PersonaManager by name; if a
    /// persona doesn't exist (renamed), the rule is dropped on first match.
    static func seedRules() -> [BrowserSiteRule] {
        let now = Date()
        let pm = PersonaManager.shared
        func id(of name: String) -> String? { pm.personas.first { $0.name == name }?.id }

        var seeds: [BrowserSiteRule] = []
        func add(_ host: String, _ personaName: String) {
            guard let pid = id(of: personaName) else { return }
            seeds.append(BrowserSiteRule(
                id: UUID().uuidString,
                hostnamePattern: host,
                personaId: pid,
                createdAt: now,
                isBuiltIn: true
            ))
        }

        // Software Engineer
        add("github.com", "Software Engineer")
        add("gitlab.com", "Software Engineer")
        add("bitbucket.org", "Software Engineer")
        add("stackoverflow.com", "Software Engineer")
        add("*.atlassian.net", "Software Engineer")
        add("linear.app", "Software Engineer")
        add("vercel.com", "Software Engineer")
        add("railway.app", "Software Engineer")
        add("fly.io", "Software Engineer")
        add("aws.amazon.com", "Software Engineer")
        add("console.cloud.google.com", "Software Engineer")
        add("portal.azure.com", "Software Engineer")
        add("huggingface.co", "Software Engineer")
        add("*.notion.site", "Software Engineer")  // dev docs are often on notion.site
        add("*.readthedocs.io", "Software Engineer")
        add("docs.rs", "Software Engineer")

        // Professional Email (mail webapps)
        add("mail.google.com", "Professional Email")
        add("outlook.live.com", "Professional Email")
        add("outlook.office.com", "Professional Email")
        add("outlook.office365.com", "Professional Email")
        add("mail.proton.me", "Professional Email")
        add("app.fastmail.com", "Professional Email")
        add("hey.com", "Professional Email")
        add("superhuman.com", "Professional Email")

        // Casual Chat
        add("twitter.com", "Casual Chat")
        add("x.com", "Casual Chat")
        add("bsky.app", "Casual Chat")
        add("threads.net", "Casual Chat")
        add("reddit.com", "Casual Chat")
        add("discord.com", "Casual Chat")
        add("*.slack.com", "Casual Chat")
        add("web.whatsapp.com", "Casual Chat")
        add("messages.google.com", "Casual Chat")
        add("messenger.com", "Casual Chat")
        add("chatgpt.com", "Casual Chat")
        add("claude.ai", "Casual Chat")
        add("perplexity.ai", "Casual Chat")
        add("gemini.google.com", "Casual Chat")

        // Technical Writing
        add("notion.so", "Technical Writing")
        add("www.notion.so", "Technical Writing")
        add("docs.google.com", "Technical Writing")
        add("substack.com", "Technical Writing")
        add("medium.com", "Technical Writing")
        add("dev.to", "Technical Writing")
        add("publish.obsidian.md", "Technical Writing")

        return seeds
    }
}
