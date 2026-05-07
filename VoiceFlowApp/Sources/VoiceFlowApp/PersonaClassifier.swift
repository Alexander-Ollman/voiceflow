import Foundation

/// Uses the local LLM (already running for dictation formatting) to auto-classify
/// installed apps and browser hostnames into personas. Calls the OpenAI-compatible
/// server endpoint configured via VOICEFLOW_LLM_SERVER_ENDPOINT, asks for JSON
/// output with a confidence score, and applies the classification only when the
/// model is at least minConfidence sure.
///
/// Threshold defaults to 0.80 — the LLM volunteers a confidence and we trust its
/// self-report (Qwen/Bonsai are decent at calibration on classification tasks).
/// Default confidence threshold (nonisolated so it can be referenced as a default
/// argument value without crossing actor boundaries). Raised from 0.80 → 0.90
/// after observing small models (Bonsai-8B) emit confident misclassifications
/// based on superficial reasoning ("it's a macOS app, therefore Software Engineer").
let kPersonaClassifierDefaultMinConfidence: Double = 0.90

@MainActor
final class PersonaClassifier {
    static let defaultMinConfidence: Double = kPersonaClassifierDefaultMinConfidence

    struct Classification {
        let personaName: String?  // nil = "no good fit"
        let confidence: Double    // 0.0 - 1.0
        let reason: String?
    }

    /// Classify a single app. Returns nil if the LLM can't decide or HTTP fails.
    func classifyApp(displayName: String, bundleId: String) async -> Classification? {
        let personas = PersonaManager.shared.personas
        guard !personas.isEmpty else { return nil }

        let userBlock = """
        APP_NAME: \(displayName)
        BUNDLE_ID: \(bundleId)
        """
        return await classify(userBlock: userBlock, kind: "macOS application")
    }

    /// Classify a browser hostname (optionally with the page title for richer context).
    func classifyHostname(_ hostname: String, pageTitle: String? = nil) async -> Classification? {
        let personas = PersonaManager.shared.personas
        guard !personas.isEmpty else { return nil }

        var userBlock = "HOSTNAME: \(hostname)"
        if let title = pageTitle, !title.isEmpty {
            userBlock += "\nPAGE_TITLE: \(title)"
        }
        return await classify(userBlock: userBlock, kind: "website")
    }

    // MARK: - Core HTTP request

    private func classify(userBlock: String, kind: String) async -> Classification? {
        let env = ProcessInfo.processInfo.environment
        let endpoint = env["VOICEFLOW_LLM_SERVER_ENDPOINT"] ?? "http://127.0.0.1:8080"
        let model = env["VOICEFLOW_LLM_SERVER_MODEL"] ?? "default"

        let personaCatalog = PersonaManager.shared.personas
            .map { p in
                "- \(p.name): \(p.prompt.prefix(180))\(p.prompt.count > 180 ? "..." : "")"
            }
            .joined(separator: "\n")

        let systemInstructions = """
        You are classifying a \(kind) by what KIND OF WRITING the user does there. Return null \
        unless you can identify the app/site's actual purpose AND match it to a clearly-fitting \
        persona.

        ## Persona Catalog
        \(personaCatalog)

        ## How to think about this

        Step 1: From the name (and bundle id / hostname), identify what the app or site DOES. \
        If the name doesn't tell you anything specific (e.g. just "TextEdit", "Notes", "Stickies", \
        "Finder", "System Settings") then the writing context is genuinely ambiguous — return null.

        Step 2: Browsers (Safari, Chrome, Firefox, Vivaldi, Brave, Edge, Arc) are ALWAYS ambiguous \
        because content varies tab-to-tab. Return null with confidence near 0 — site rules cover them.

        Step 3: System utilities, file managers, media players, and generic note-takers are \
        ambiguous. Return null.

        Step 4: Only when you can identify a SPECIFIC use case (e.g. Slack → Casual Chat, Xcode → \
        Software Engineer, Mail → Professional Email, Notion → Technical Writing) should you pick \
        a persona.

        Step 5: Confidence must reflect how certain you are about the SPECIFIC USE CASE, not how \
        certain you are that "it's a macOS app". Generic reasoning like "this is a macOS app with \
        a bundle id" earns confidence 0.

        ## Output rules
        - Only confident, specific classifications get applied (threshold 0.90).
        - If the reason field amounts to "it's a macOS app" or "the bundle id suggests it could be" \
          or "this is typical for X", set confidence ≤ 0.5.
        - If you cannot name what specific writing the user actually does in this app or site, \
          return persona=null.
        - Return ONLY a JSON object — no prose, no markdown, no code fences.

        ## Output schema
        {"persona": "<exact persona name from catalog or null>", "confidence": <0.0-1.0>, "reason": "<one short sentence naming the SPECIFIC use case>"}

        ## Examples (illustrative, not exhaustive)
        - "Xcode" → {"persona": "Software Engineer", "confidence": 0.97, "reason": "Apple's IDE for Swift/Objective-C development"}
        - "Slack" → {"persona": "Casual Chat", "confidence": 0.95, "reason": "team chat with informal register"}
        - "Mail" → {"persona": "Professional Email", "confidence": 0.95, "reason": "Apple's email client, used for correspondence"}
        - "Safari" → {"persona": null, "confidence": 0.05, "reason": "browser — context varies per site"}
        - "Notes" → {"persona": null, "confidence": 0.30, "reason": "general-purpose notes, mixed casual and technical content"}
        - "System Settings" → {"persona": null, "confidence": 0.05, "reason": "configuration UI, no real writing happens here"}
        - "Finder" → {"persona": null, "confidence": 0.10, "reason": "file manager, search input only"}
        """

        // Build a Qwen chat-template prompt and use the /v1/completions raw-prompt endpoint
        // (matches what VoiceFlow does for dictation formatting — no double template wrap).
        let prompt = """
        <|im_start|>system
        \(systemInstructions)<|im_end|>
        <|im_start|>user
        \(userBlock)<|im_end|>
        <|im_start|>assistant
        <think>

        </think>


        """

        // Temperature is intentionally non-zero — the self-consistency vote in
        // classifyAppWithSelfConsistency relies on independent samples to
        // surface uncertainty. With T=0 every trial would be identical.
        let body: [String: Any] = [
            "model": model,
            "prompt": prompt,
            "max_tokens": 120,
            "temperature": 0.5,
            "top_p": 0.9,
            "stream": false,
        ]

        guard let url = URL(string: endpoint.trimmingCharacters(in: CharacterSet(charactersIn: "/")) + "/v1/completions") else {
            return nil
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 20
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)

        do {
            let (data, _) = try await URLSession.shared.data(for: req)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let text = choices.first?["text"] as? String else {
                return nil
            }
            return parseClassification(text)
        } catch {
            NSLog("[PersonaClassifier] HTTP error: %@", error.localizedDescription)
            return nil
        }
    }

    /// Extract the JSON object from the LLM's reply (which may be padded with whitespace).
    private func parseClassification(_ raw: String) -> Classification? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // Locate the first {...} substring — tolerant of leading/trailing prose.
        guard let start = trimmed.firstIndex(of: "{"),
              let end = trimmed.lastIndex(of: "}"),
              start < end else {
            NSLog("[PersonaClassifier] No JSON object in reply: %@", String(trimmed.prefix(120)))
            return nil
        }
        let jsonStr = String(trimmed[start...end])
        guard let data = jsonStr.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            NSLog("[PersonaClassifier] Bad JSON: %@", jsonStr)
            return nil
        }

        let personaName: String?
        if let name = obj["persona"] as? String, !name.lowercased().contains("null") {
            personaName = name
        } else {
            personaName = nil
        }

        let confidence: Double
        if let n = obj["confidence"] as? Double { confidence = n }
        else if let n = obj["confidence"] as? Int { confidence = Double(n) }
        else { confidence = 0.0 }

        let reason = obj["reason"] as? String

        return Classification(
            personaName: personaName,
            confidence: max(0.0, min(1.0, confidence)),
            reason: reason
        )
    }

    // MARK: - Batch classify all unmapped apps

    /// Iterate over every AppProfile that has no persona/customPrompt and classify each.
    /// Applies any classification with confidence >= minConfidence.
    /// Returns (applied, considered) counts.
    @discardableResult
    func classifyUnmappedApps(
        minConfidence: Double = kPersonaClassifierDefaultMinConfidence,
        progress: ((_ done: Int, _ total: Int, _ currentApp: String) -> Void)? = nil
    ) async -> (applied: Int, considered: Int) {
        let manager = AppProfileManager.shared
        let unmapped = manager.profiles.filter {
            $0.personaId == nil && ($0.customPrompt?.isEmpty ?? true)
        }
        var applied = 0
        for (index, profile) in unmapped.enumerated() {
            await MainActor.run {
                progress?(index + 1, unmapped.count, profile.displayName)
            }

            // Self-consistency: ask the model 3 times, only apply if a single
            // (non-null) persona wins ≥2 votes AND every vote that picked it
            // was confident enough. Cuts the rate of confident-but-wrong picks
            // that small models like Bonsai produce on ambiguous inputs.
            let cls = await classifyAppWithSelfConsistency(
                displayName: profile.displayName,
                bundleId: profile.id,
                trials: 3,
                minConfidence: minConfidence
            )

            guard let cls = cls,
                  let pname = cls.personaName,
                  let persona = PersonaManager.shared.personas.first(where: {
                      $0.name.lowercased() == pname.lowercased()
                  })
            else { continue }

            var updated = profile
            updated.personaId = persona.id
            manager.updateProfile(updated)
            applied += 1
        }
        return (applied: applied, considered: unmapped.count)
    }

    /// Run `trials` independent classifications and return the winner only when
    /// a single (non-null) persona is picked ≥2 times AND every vote for it is
    /// at confidence ≥ minConfidence. Returns nil otherwise — the user can
    /// always assign manually.
    private func classifyAppWithSelfConsistency(
        displayName: String,
        bundleId: String,
        trials: Int,
        minConfidence: Double
    ) async -> Classification? {
        var results: [Classification] = []
        for _ in 0..<trials {
            if let cls = await classifyApp(displayName: displayName, bundleId: bundleId) {
                results.append(cls)
            }
        }

        let voteSummary = results.map { c in
            "\(c.personaName ?? "(null)")@\(String(format: "%.2f", c.confidence))"
        }.joined(separator: ", ")
        NSLog("[PersonaClassifier] %@ votes: [%@]", displayName, voteSummary)

        // Tally non-null persona votes (case-insensitive).
        var byPersona: [String: [Classification]] = [:]
        for r in results {
            guard let p = r.personaName else { continue }
            byPersona[p.lowercased(), default: []].append(r)
        }
        // Find the winning persona — most votes; ties → no apply (ambiguous).
        let sorted = byPersona.sorted { $0.value.count > $1.value.count }
        guard let (winnerKey, winnerVotes) = sorted.first,
              winnerVotes.count >= 2 else {
            return nil
        }
        // Tie-break check: if a second persona also has ≥2 votes, drop.
        if sorted.count > 1, sorted[1].value.count >= 2 {
            return nil
        }
        // Confidence floor: every vote for the winner must clear the threshold.
        guard winnerVotes.allSatisfy({ $0.confidence >= minConfidence }) else {
            return nil
        }

        let canonicalName = winnerVotes.first?.personaName ?? winnerKey
        let avgConf = winnerVotes.map(\.confidence).reduce(0, +) / Double(winnerVotes.count)
        NSLog("[PersonaClassifier] %@ → %@ (avg %.2f, %d/%d votes) — applied",
              displayName, canonicalName, avgConf, winnerVotes.count, results.count)
        return Classification(
            personaName: canonicalName,
            confidence: avgConf,
            reason: winnerVotes.first?.reason
        )
    }
}
