//! Edit-intent classifier.
//!
//! Runs BEFORE the rest of the normalization pipeline. Decides whether the
//! current utterance is:
//!
//! - `Verbatim`             — paste as-is (after normalization)
//! - `InlineCorrection`     — same-utterance self-correction (handled by
//!                            `apply_correction_commands` later in the pipeline)
//! - `RetroactiveCorrection` — refers to text outside this utterance; needs
//!                            field-context read-back + structured LLM edit
//! - `Command`              — AI command ("rewrite this", "proofread this",
//!                            "summarize this", "reply to this", "continue
//!                            this", "explain this", "draft …", etc.)
//!
//! This is a cheap rules-based pre-filter — the goal is to avoid paying
//! Bonsai latency on every dictation, only invoking the structured-edit path
//! when there's strong signal. False positives degrade UX (we ask Bonsai to
//! find an anchor that doesn't exist); false negatives just mean the user
//! has to repeat themselves more verbatim. We err toward false negatives.

/// What kind of utterance is this?
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "command", rename_all = "snake_case")]
pub enum IntentKind {
    Verbatim,
    InlineCorrection,
    RetroactiveCorrection,
    Command(CommandKind),
}

/// Recognized AI command. The string parameter is the residual utterance after
/// the trigger phrase is stripped (e.g. "make this more formal" → param "more formal").
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandKind {
    /// "rewrite this [as X]", "make this [X]"
    Rewrite,
    /// "proofread this", "fix the grammar"
    Proofread,
    /// "shorten this", "make this shorter", "make this concise"
    Shorten,
    /// "make this a list", "convert to bullets", "bullet this"
    Bullet,
    /// "continue this", "keep going", "extend this"
    Continue,
    /// "summarize this", "tl;dr this", "give me the gist"
    Summarize,
    /// "reply to this [saying …]", "draft a reply [saying …]"
    Reply,
    /// "explain this", "what does this mean"
    Explain,
    /// "draft [an email / a message / …]"
    Draft,
    /// "what's …", "how do I …", general factual / unscoped Q&A
    Question,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntentResult {
    pub kind: IntentKind,
    /// Residual user content after stripping the trigger phrase.
    /// For RetroactiveCorrection this is the full utterance (the LLM needs
    /// everything); for Command this is the parameter text.
    pub residual: String,
    /// Optional anchor + replacement extracted from explicit "change X to Y" /
    /// "X, not Y" patterns. Lets us bypass the LLM call when intent is
    /// unambiguous (saves ~200ms).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor_hint: Option<AnchorHint>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnchorHint {
    pub find: String,
    pub replace: String,
}

/// Classify the intent of a freshly-transcribed utterance.
pub fn classify_intent(text: &str) -> IntentResult {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return IntentResult {
            kind: IntentKind::Verbatim,
            residual: String::new(),
            anchor_hint: None,
        };
    }

    let lower = trimmed.to_lowercase();

    // Order matters: check commands first (they're explicit), then retroactive
    // (specific patterns), then fall through to verbatim/inline.

    if let Some(cmd) = match_command(&lower, trimmed) {
        return cmd;
    }

    if let Some(retro) = match_retroactive(&lower, trimmed) {
        return retro;
    }

    if has_inline_correction(&lower) {
        return IntentResult {
            kind: IntentKind::InlineCorrection,
            residual: trimmed.to_string(),
            anchor_hint: None,
        };
    }

    IntentResult {
        kind: IntentKind::Verbatim,
        residual: trimmed.to_string(),
        anchor_hint: None,
    }
}

// -----------------------------------------------------------------------------
// Retroactive correction detection
// -----------------------------------------------------------------------------

fn match_retroactive(lower: &str, original: &str) -> Option<IntentResult> {
    // Pattern A: explicit "X, not Y" / "I meant X, not Y" / "not Y, X"
    if let Some(hint) = extract_not_pattern(original) {
        return Some(IntentResult {
            kind: IntentKind::RetroactiveCorrection,
            residual: original.to_string(),
            anchor_hint: Some(hint),
        });
    }

    // Pattern B: "change X to Y" / "replace X with Y"
    if let Some(hint) = extract_change_pattern(lower, original) {
        return Some(IntentResult {
            kind: IntentKind::RetroactiveCorrection,
            residual: original.to_string(),
            anchor_hint: Some(hint),
        });
    }

    // Pattern C: anaphoric references that imply prior text without explicit
    // anchor — let the LLM figure it out.
    //
    // Markers must be strong enough not to fire on ordinary dictation:
    // "go back and …", "earlier I said …", "previously …", "fix the …".
    const ANAPHORIC_TRIGGERS: &[&str] = &[
        "go back and",
        "earlier i said",
        "earlier i typed",
        "i said earlier",
        "previously i said",
        "previously i wrote",
        "fix the ",
        "correct the ",
        "in the previous",
    ];
    for trig in ANAPHORIC_TRIGGERS {
        if lower.contains(trig) {
            return Some(IntentResult {
                kind: IntentKind::RetroactiveCorrection,
                residual: original.to_string(),
                anchor_hint: None,
            });
        }
    }

    None
}

/// Extract "I meant X, not Y" / "X, not Y" patterns.
///
/// "I meant" / "I mean" prefix wins when present — `last_noun_phrase` can't
/// distinguish the verb from a content noun, so handle the explicit form first.
fn extract_not_pattern(text: &str) -> Option<AnchorHint> {
    let lower = text.to_lowercase();

    // Pattern A: explicit "I meant X (,) not Y"
    for marker in &["i meant ", "i mean "] {
        if let Some(start) = lower.find(marker) {
            let rest = &text[start + marker.len()..];
            let rest_lower = rest.to_lowercase();
            let not_hit = rest_lower
                .find(", not ")
                .map(|p| (p, 6))
                .or_else(|| rest_lower.find(" not ").map(|p| (p, 5)));
            if let Some((np, np_len)) = not_hit {
                let x = rest[..np]
                    .trim()
                    .trim_matches(|c: char| c == ',' || c == '"' || c == '\'');
                let y_raw = rest[np + np_len..].trim();
                let y = match first_noun_phrase(y_raw) {
                    Some(y) => y,
                    None => continue,
                };
                if !x.is_empty() && !y.is_empty() && x.to_lowercase() != y.to_lowercase() {
                    return Some(AnchorHint {
                        find: y,
                        replace: x.to_string(),
                    });
                }
            }
        }
    }

    // Pattern B: bare "X, not Y" (no "I meant" prefix). Comma is required to
    // avoid false positives like "do this, not that way" — actually that
    // example IS a retroactive intent, so a leading comma + " not " is enough.
    if let Some(pos) = lower.find(", not ") {
        let before = text[..pos].trim_end();
        let after = text[pos + 6..].trim();
        if let (Some(x), Some(y)) = (last_noun_phrase(before), first_noun_phrase(after)) {
            if x.to_lowercase() != y.to_lowercase() {
                return Some(AnchorHint { find: y, replace: x });
            }
        }
    }

    None
}

/// Extract "change X to Y" / "replace X with Y" patterns.
fn extract_change_pattern(lower: &str, original: &str) -> Option<AnchorHint> {
    const PATTERNS: &[(&str, &str)] = &[
        ("change ", " to "),
        ("replace ", " with "),
        ("swap ", " for "),
    ];
    for (trig, sep) in PATTERNS {
        if let Some(t_start) = lower.find(trig) {
            let after_trig = &original[t_start + trig.len()..];
            let after_lower = &lower[t_start + trig.len()..];
            if let Some(sep_pos) = after_lower.find(sep) {
                let x = after_trig[..sep_pos].trim().trim_matches('"').trim_matches('\'');
                let y_raw = after_trig[sep_pos + sep.len()..].trim();
                // Y ends at the next punctuation or end of utterance
                let y_end = y_raw
                    .find(|c: char| matches!(c, '.' | ',' | ';' | '!' | '?'))
                    .unwrap_or(y_raw.len());
                let y = y_raw[..y_end].trim().trim_matches('"').trim_matches('\'');
                if !x.is_empty() && !y.is_empty() && x.to_lowercase() != y.to_lowercase() {
                    return Some(AnchorHint {
                        find: x.to_string(),
                        replace: y.to_string(),
                    });
                }
            }
        }
    }
    None
}

/// The last 1–3 word phrase before a marker. Strips leading articles.
fn last_noun_phrase(text: &str) -> Option<String> {
    let trimmed = text.trim_end_matches(|c: char| c.is_ascii_punctuation()).trim();
    let words: Vec<&str> = trimmed.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }
    // Take the last 1–3 words; stop at common clause-boundary words.
    let mut start = words.len().saturating_sub(3);
    for (i, w) in words.iter().enumerate().skip(start) {
        let wl = w.to_lowercase();
        if matches!(
            wl.as_str(),
            "and" | "or" | "but" | "so" | "because" | "that" | "which" | "the" | "a" | "an"
        ) && i + 1 < words.len()
        {
            start = i + 1;
        }
    }
    Some(words[start..].join(" "))
}

/// The first 1–3 word phrase from `text`, stopping at clause-boundary words or
/// punctuation.
fn first_noun_phrase(text: &str) -> Option<String> {
    let trimmed = text.trim_start_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace());
    let mut out = Vec::new();
    for w in trimmed.split_whitespace() {
        let cleaned = w.trim_matches(|c: char| c.is_ascii_punctuation());
        if cleaned.is_empty() {
            break;
        }
        out.push(cleaned);
        if out.len() >= 3 {
            break;
        }
        // Stop at clause-boundary if we have at least one content word
        let next_lower = cleaned.to_lowercase();
        if matches!(next_lower.as_str(), "and" | "or" | "but" | "so" | "because") {
            out.pop();
            break;
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out.join(" "))
    }
}

/// Same-utterance corrections: "I mean X", "wait no X", etc.
/// Reuses the trigger list from `correction_commands.rs` (subset).
fn has_inline_correction(lower: &str) -> bool {
    const TRIGGERS: &[&str] = &[
        "scratch that",
        "delete that",
        "never mind",
        "nevermind",
        " i mean ",
        " or rather ",
        " wait no ",
    ];
    TRIGGERS.iter().any(|t| lower.contains(t))
}

// -----------------------------------------------------------------------------
// AI command detection
// -----------------------------------------------------------------------------

fn match_command(lower: &str, original: &str) -> Option<IntentResult> {
    // Order: longer / more-specific triggers first so "reply to this" beats
    // "reply" and "make this a list" beats "make this …".

    let triggers: &[(&str, CommandKind)] = &[
        // List/bullet first — "make this a list" must beat "make this …"
        ("make this a list", CommandKind::Bullet),
        ("make this into a list", CommandKind::Bullet),
        ("convert this to bullets", CommandKind::Bullet),
        ("convert to bullets", CommandKind::Bullet),
        ("bullet this", CommandKind::Bullet),
        ("turn this into bullets", CommandKind::Bullet),
        // Reply
        ("reply to this", CommandKind::Reply),
        ("draft a reply", CommandKind::Reply),
        ("write a reply", CommandKind::Reply),
        // Continue
        ("continue this", CommandKind::Continue),
        ("keep going", CommandKind::Continue),
        ("extend this", CommandKind::Continue),
        // Summarize
        ("summarize this", CommandKind::Summarize),
        ("tldr this", CommandKind::Summarize),
        ("tl;dr this", CommandKind::Summarize),
        ("give me the gist", CommandKind::Summarize),
        // Proofread
        ("proofread this", CommandKind::Proofread),
        ("fix the grammar", CommandKind::Proofread),
        ("fix grammar", CommandKind::Proofread),
        ("check the grammar", CommandKind::Proofread),
        // Shorten
        ("shorten this", CommandKind::Shorten),
        ("make this shorter", CommandKind::Shorten),
        ("make this concise", CommandKind::Shorten),
        ("tighten this up", CommandKind::Shorten),
        // Rewrite (catch-all "make this …" — keep generic so residual captures the full param)
        ("rewrite this", CommandKind::Rewrite),
        ("rewrite as", CommandKind::Rewrite),
        ("make this ", CommandKind::Rewrite),
        // Explain
        ("explain this", CommandKind::Explain),
        ("what does this mean", CommandKind::Explain),
        ("what does this say", CommandKind::Explain),
        // Draft
        ("draft an email", CommandKind::Draft),
        ("draft a message", CommandKind::Draft),
        ("draft a ", CommandKind::Draft),
        ("write an email", CommandKind::Draft),
        ("write a message", CommandKind::Draft),
    ];

    for (trig, kind) in triggers {
        if let Some(pos) = lower.find(trig) {
            // Trigger should be at start or after sentence boundary — avoids
            // matching "I thought we should rewrite this paragraph" as a command.
            if pos == 0 || is_after_boundary(lower, pos) {
                let residual_start = pos + trig.len();
                let residual = original[residual_start..]
                    .trim_start_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
                    .to_string();
                return Some(IntentResult {
                    kind: IntentKind::Command(kind.clone()),
                    residual,
                    anchor_hint: None,
                });
            }
        }
    }

    // Question-form fallback: "what's …", "how do I …", "what is …"
    const Q_PREFIXES: &[&str] = &[
        "what's ", "what is ", "how do i ", "how do you ", "tell me ",
        "answer ", "how can i ",
    ];
    for prefix in Q_PREFIXES {
        if lower.starts_with(prefix) {
            return Some(IntentResult {
                kind: IntentKind::Command(CommandKind::Question),
                residual: original.to_string(),
                anchor_hint: None,
            });
        }
    }

    None
}

fn is_after_boundary(text: &str, pos: usize) -> bool {
    // Find the previous non-whitespace char before `pos`
    let before = &text[..pos];
    let last_char = before
        .chars()
        .rev()
        .find(|c| !c.is_whitespace());
    matches!(last_char, None | Some('.') | Some('!') | Some('?'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent(s: &str) -> IntentKind {
        classify_intent(s).kind
    }

    #[test]
    fn verbatim_passthrough() {
        assert_eq!(intent("hello world"), IntentKind::Verbatim);
        assert_eq!(intent("the quick brown fox"), IntentKind::Verbatim);
    }

    #[test]
    fn detects_not_pattern() {
        let r = classify_intent("I meant pears, not bananas");
        assert_eq!(r.kind, IntentKind::RetroactiveCorrection);
        let hint = r.anchor_hint.unwrap();
        assert_eq!(hint.find.to_lowercase(), "bananas");
        assert_eq!(hint.replace.to_lowercase(), "pears");
    }

    #[test]
    fn detects_change_to_pattern() {
        let r = classify_intent("change bananas to pears");
        assert_eq!(r.kind, IntentKind::RetroactiveCorrection);
        let hint = r.anchor_hint.unwrap();
        assert_eq!(hint.find.to_lowercase(), "bananas");
        assert_eq!(hint.replace.to_lowercase(), "pears");
    }

    #[test]
    fn detects_replace_with_pattern() {
        let r = classify_intent("replace the price with twelve dollars");
        assert_eq!(r.kind, IntentKind::RetroactiveCorrection);
        let hint = r.anchor_hint.unwrap();
        assert!(hint.find.to_lowercase().contains("price"));
    }

    #[test]
    fn anaphoric_retroactive() {
        assert_eq!(
            intent("go back and fix the second paragraph"),
            IntentKind::RetroactiveCorrection
        );
        assert_eq!(
            intent("earlier I said pears but I meant bananas"),
            IntentKind::RetroactiveCorrection
        );
    }

    #[test]
    fn inline_correction_not_retroactive() {
        // Same-utterance correction — handled later in the pipeline
        assert_eq!(
            intent("buy apples I mean oranges"),
            IntentKind::InlineCorrection
        );
        assert_eq!(
            intent("meet at three wait no four"),
            IntentKind::InlineCorrection
        );
    }

    #[test]
    fn detects_commands() {
        assert_eq!(intent("rewrite this more formally"), IntentKind::Command(CommandKind::Rewrite));
        assert_eq!(intent("proofread this"), IntentKind::Command(CommandKind::Proofread));
        assert_eq!(intent("make this shorter"), IntentKind::Command(CommandKind::Shorten));
        assert_eq!(intent("make this a list"), IntentKind::Command(CommandKind::Bullet));
        assert_eq!(intent("continue this"), IntentKind::Command(CommandKind::Continue));
        assert_eq!(intent("summarize this"), IntentKind::Command(CommandKind::Summarize));
        assert_eq!(intent("reply to this saying I agree"), IntentKind::Command(CommandKind::Reply));
        assert_eq!(intent("explain this"), IntentKind::Command(CommandKind::Explain));
        assert_eq!(intent("draft an email to Sarah"), IntentKind::Command(CommandKind::Draft));
    }

    #[test]
    fn question_form_routes_to_qa() {
        assert_eq!(intent("what's the capital of Peru"), IntentKind::Command(CommandKind::Question));
        assert_eq!(intent("how do I undo a git rebase"), IntentKind::Command(CommandKind::Question));
    }

    #[test]
    fn command_extracts_residual() {
        let r = classify_intent("make this more formal");
        assert_eq!(r.kind, IntentKind::Command(CommandKind::Rewrite));
        assert_eq!(r.residual, "more formal");

        let r = classify_intent("reply to this saying I agree but let's push to next week");
        assert_eq!(r.kind, IntentKind::Command(CommandKind::Reply));
        assert!(r.residual.starts_with("saying I agree"));
    }

    #[test]
    fn command_inside_sentence_is_not_command() {
        // The trigger appears mid-sentence — should NOT route to command
        let r = classify_intent("I think we should rewrite this paragraph next week");
        assert_eq!(r.kind, IntentKind::Verbatim);
    }
}
