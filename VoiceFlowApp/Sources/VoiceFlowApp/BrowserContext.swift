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

    // MARK: - Active-field DOM read

    /// Snapshot of the focused DOM element in the active browser tab.
    struct DOMFieldSnapshot {
        let text: String
        let cursor: Int?    // selectionStart, if applicable
        let tag: String     // INPUT / TEXTAREA / DIV / etc.
        let role: String?   // ARIA role if present
    }

    /// Run JS in the active tab of the given browser to read the focused field.
    /// Returns nil if browser isn't running, automation is denied, no field is focused,
    /// or (Safari) "Allow JavaScript from Apple Events" is off.
    ///
    /// Coverage: Gmail compose, Slack/Discord web, Notion web, Linear, ChatGPT/Claude,
    /// any contenteditable or input/textarea. Recipient pills are read separately by
    /// the recipient-context flow.
    static func currentFieldSnapshot(forBundleId bundleId: String) -> DOMFieldSnapshot? {
        guard let driver = drivers[bundleId] else { return nil }

        // Single-line JS — AppleScript chokes on multi-line strings.
        // Returns JSON: {text, cursor, tag, role} or {err}.
        let js = "(function(){var e=document.activeElement;if(!e||e===document.body)return JSON.stringify({err:'no-focus'});var v=e.value;if(v===undefined||v===null){v=e.innerText||e.textContent||'';}var s=null;try{s=e.selectionStart;}catch(_){}return JSON.stringify({text:String(v),cursor:s,tag:e.tagName||'',role:e.getAttribute&&e.getAttribute('role')||null});})()"

        let script: String
        if driver.isChromium {
            // execute … javascript returns the JS result directly as a string.
            script = """
            tell application "\(driver.appName)"
                if (count of windows) is 0 then return ""
                return execute active tab of front window javascript "\(js.replacingOccurrences(of: "\"", with: "\\\""))"
            end tell
            """
        } else {
            // Safari: do JavaScript … in current tab. Requires Develop → Allow JavaScript from Apple Events.
            script = """
            tell application "\(driver.appName)"
                if (count of windows) is 0 then return ""
                return (do JavaScript "\(js.replacingOccurrences(of: "\"", with: "\\\""))" in current tab of front window) as string
            end tell
            """
        }

        var error: NSDictionary?
        guard let appleScript = NSAppleScript(source: script) else { return nil }
        let result = appleScript.executeAndReturnError(&error)

        if let err = error {
            let code = (err["NSAppleScriptErrorNumber"] as? Int) ?? 0
            if code != -1728 { // -1728 "no window" is normal
                NSLog("[BrowserContext] JS-bridge error %d: %@",
                      code, (err["NSAppleScriptErrorMessage"] as? String) ?? "")
            }
            return nil
        }

        guard let jsonString = result.stringValue, !jsonString.isEmpty,
              let data = jsonString.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }

        if obj["err"] != nil { return nil }

        let text = (obj["text"] as? String) ?? ""
        let cursor = obj["cursor"] as? Int
        let tag = (obj["tag"] as? String) ?? ""
        let role = obj["role"] as? String

        return DOMFieldSnapshot(text: text, cursor: cursor, tag: tag, role: role)
    }

    // MARK: - Recipient extraction (Phase 4)

    /// Information about the message-target that's active in the browser.
    /// E.g. Gmail compose "To:" pills, Slack channel name, Discord DM target.
    struct RecipientHint {
        let target: String          // "Sarah Chen" / "#engineering" / "@ops"
        let channel: String?        // "Gmail" / "Slack" / "Discord" / "Linear"
        let role: String?           // "recipient" / "channel" / "team"
    }

    /// Extract a recipient hint from the active browser tab. Returns nil if
    /// no recognized compose surface is on screen.
    ///
    /// Coverage:
    /// - Gmail compose: reads the `To:` field pills
    /// - Slack DM/channel: reads the active conversation header
    /// - Discord: reads the channel header
    /// - Linear: reads the assignee field
    static func currentRecipientHint(forBundleId bundleId: String) -> RecipientHint? {
        guard let driver = drivers[bundleId] else { return nil }

        // Single-line JS that detects each surface in turn and returns the first hit.
        let js = """
        (function(){
          var loc=location.hostname||'';
          function txt(sel){var e=document.querySelector(sel);return e?(e.innerText||e.textContent||'').trim():'';}
          function pills(sel){var nl=document.querySelectorAll(sel);var out=[];for(var i=0;i<nl.length;i++){var t=(nl[i].innerText||nl[i].textContent||'').trim();if(t)out.push(t);}return out;}
          // Gmail
          if(loc.indexOf('mail.google.com')>=0){var to=pills('[aria-label*=\\"To\\"][role=\\"option\\"], .vR .vN');if(to.length)return JSON.stringify({target:to.join(', '),channel:'Gmail',role:'recipient'});}
          // Slack web
          if(loc.indexOf('slack.com')>=0){var ch=txt('[data-qa=\\"channel_name_button_text\\"]')||txt('[data-qa=\\"channel_name\\"]')||txt('h2[data-qa=\\"channel_name\\"]');if(ch)return JSON.stringify({target:ch,channel:'Slack',role:'channel'});}
          // Discord
          if(loc.indexOf('discord.com')>=0){var dc=txt('[class*=\\"title-\\"] h3, [class*=\\"channelName\\"]')||txt('[class*=\\"name-\\"]');if(dc)return JSON.stringify({target:dc,channel:'Discord',role:'channel'});}
          // Linear
          if(loc.indexOf('linear.app')>=0){var ass=txt('[aria-label*=\\"Assignee\\"]')||txt('button[aria-label*=\\"assignee\\"]');if(ass)return JSON.stringify({target:ass,channel:'Linear',role:'assignee'});}
          return '';
        })()
        """.replacingOccurrences(of: "\n", with: " ")

        let script: String
        if driver.isChromium {
            script = """
            tell application "\(driver.appName)"
                if (count of windows) is 0 then return ""
                return execute active tab of front window javascript "\(js.replacingOccurrences(of: "\"", with: "\\\""))"
            end tell
            """
        } else {
            script = """
            tell application "\(driver.appName)"
                if (count of windows) is 0 then return ""
                return (do JavaScript "\(js.replacingOccurrences(of: "\"", with: "\\\""))" in current tab of front window) as string
            end tell
            """
        }

        var error: NSDictionary?
        guard let appleScript = NSAppleScript(source: script) else { return nil }
        let result = appleScript.executeAndReturnError(&error)

        if error != nil { return nil }
        guard let jsonString = result.stringValue, !jsonString.isEmpty,
              let data = jsonString.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let target = obj["target"] as? String, !target.isEmpty
        else { return nil }

        return RecipientHint(
            target: target,
            channel: obj["channel"] as? String,
            role: obj["role"] as? String
        )
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
