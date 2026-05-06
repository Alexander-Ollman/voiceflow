//! Spoken self-correction command handling
//!
//! Handles patterns like:
//! - Deletion: "scratch that", "delete that", "never mind"
//! - Inline correction: "I mean X", "wait no X", "actually X", "or rather X"

/// Apply spoken correction commands to text.
///
/// Processes right-to-left so chained corrections work:
/// "A I mean B I mean C" → "C"
pub fn apply_correction_commands(text: &str) -> String {
    let mut result = text.to_string();

    // Process deletion commands first
    result = apply_deletion_commands(&result);

    // Process inline corrections (right-to-left)
    result = apply_inline_corrections(&result);

    // Clean up any double spaces left behind
    collapse_spaces(&result)
}

/// Deletion triggers remove everything from the preceding clause boundary through the trigger.
/// "buy milk. And eggs scratch that" → "buy milk."
fn apply_deletion_commands(text: &str) -> String {
    const DELETION_TRIGGERS: &[&str] = &[
        "scratch that",
        "delete that",
        "never mind",
        "nevermind",
    ];

    let mut result = text.to_string();
    let lower = result.to_lowercase();

    for trigger in DELETION_TRIGGERS {
        if let Some(trigger_pos) = lower.rfind(trigger) {
            // Find the preceding clause boundary (sentence-ending punctuation or start of text)
            let before = &result[..trigger_pos];
            let clause_start = find_clause_boundary(before);

            // Remove from clause_start through end of trigger
            let trigger_end = trigger_pos + trigger.len();
            // Also consume trailing punctuation/space after the trigger
            let after_trigger = result[trigger_end..].trim_start();
            let consumed_ws = result.len() - trigger_end - after_trigger.len();

            result = format!(
                "{}{}",
                result[..clause_start].trim_end(),
                &result[trigger_end + consumed_ws..]
            );

            // Recurse to handle multiple deletion commands
            return apply_deletion_commands(&result);
        }
    }

    result
}

/// Find the start of the current clause by looking backward for sentence-ending punctuation.
/// Returns the byte offset where the clause starts (after the punctuation + space).
fn find_clause_boundary(text: &str) -> usize {
    let bytes = text.as_bytes();

    // Walk backward from the end of text
    let mut i = bytes.len();
    while i > 0 {
        i -= 1;
        let c = bytes[i] as char;

        // Sentence-ending punctuation marks a clause boundary
        if matches!(c, '.' | '!' | '?') {
            // The clause starts after the punctuation (and any following whitespace)
            let after = &text[i + 1..];
            let trimmed = after.trim_start();
            let ws_len = after.len() - trimmed.len();
            return i + 1 + ws_len;
        }
    }

    // No boundary found — the whole text is one clause
    0
}

/// Inline correction triggers: "I mean X", "wait no X", "actually X", "or rather X"
///
/// Strategy: find the trigger, take the replacement text after it, and replace the
/// matching-length span of words before it.
fn apply_inline_corrections(text: &str) -> String {
    const CORRECTION_TRIGGERS: &[&str] = &[
        "or rather",
        "wait no",
        "I mean",
        "i mean",
        "actually",
    ];

    let mut result = text.to_string();

    // Process right-to-left: find the rightmost trigger first
    loop {
        let mut best_pos: Option<(usize, &str)> = None;

        let lower = result.to_lowercase();
        for trigger in CORRECTION_TRIGGERS {
            let search = trigger.to_lowercase();
            if let Some(pos) = lower.rfind(&search) {
                // Guard against false positives
                if is_false_positive_correction(&result, pos, trigger) {
                    continue;
                }

                if best_pos.map_or(true, |(bp, _)| pos > bp) {
                    best_pos = Some((pos, trigger));
                }
            }
        }

        let (trigger_pos, trigger) = match best_pos {
            Some(t) => t,
            None => break,
        };

        let trigger_end = trigger_pos + trigger.len();

        // Get the replacement text (words after the trigger)
        let after = result[trigger_end..].trim_start();
        if after.is_empty() {
            // No replacement text — remove the trigger itself
            result = result[..trigger_pos].trim_end().to_string();
            continue;
        }

        // Count replacement words
        let replacement_words: Vec<&str> = after.split_whitespace().collect();
        let replacement_count = replacement_words.len();

        // Get the text to replace: same number of words before the trigger
        let before = result[..trigger_pos].trim_end();
        let before_words: Vec<&str> = before.split_whitespace().collect();

        if before_words.len() < replacement_count {
            // Not enough words before the trigger — just remove the trigger
            let replacement_text = replacement_words.join(" ");
            result = format!("{} {}", before, replacement_text);
            continue;
        }

        // Replace the last N words before the trigger with the replacement words
        let keep_count = before_words.len() - replacement_count;
        let kept: String = if keep_count > 0 {
            before_words[..keep_count].join(" ")
        } else {
            String::new()
        };

        let replacement_text = replacement_words.join(" ");

        result = if kept.is_empty() {
            replacement_text
        } else {
            format!("{} {}", kept, replacement_text)
        };
    }

    result
}

/// Check if a correction trigger is a false positive (discourse marker, not correction).
fn is_false_positive_correction(text: &str, trigger_pos: usize, trigger: &str) -> bool {
    let trigger_lower = trigger.to_lowercase();

    // "I mean" at sentence start = discourse marker
    if trigger_lower == "i mean" {
        let before = text[..trigger_pos].trim();
        // At the very start of text
        if before.is_empty() {
            return true;
        }
        // After sentence-ending punctuation (start of new sentence)
        if before.ends_with('.') || before.ends_with('!') || before.ends_with('?') {
            return true;
        }
    }

    // "actually" at sentence start = emphasis, not correction
    if trigger_lower == "actually" {
        let before = text[..trigger_pos].trim();
        if before.is_empty() {
            return true;
        }
        if before.ends_with('.') || before.ends_with('!') || before.ends_with('?') {
            return true;
        }
    }

    // Trigger followed by comma = discourse marker ("I mean, it's fine")
    let trigger_end = trigger_pos + trigger.len();
    let after = text[trigger_end..].trim_start();
    if after.starts_with(',') {
        return true;
    }

    false
}

/// Collapse multiple spaces into single spaces
fn collapse_spaces(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_space = false;

    for c in text.chars() {
        if c == ' ' {
            if !last_was_space {
                result.push(' ');
            }
            last_was_space = true;
        } else {
            result.push(c);
            last_was_space = false;
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Deletion commands
    // ========================================================================

    #[test]
    fn test_scratch_that_removes_clause() {
        assert_eq!(
            apply_correction_commands("buy milk. And eggs scratch that"),
            "buy milk."
        );
    }

    #[test]
    fn test_scratch_that_entire_text() {
        assert_eq!(
            apply_correction_commands("buy eggs scratch that"),
            ""
        );
    }

    #[test]
    fn test_delete_that() {
        assert_eq!(
            apply_correction_commands("first sentence. second sentence delete that"),
            "first sentence."
        );
    }

    #[test]
    fn test_never_mind() {
        assert_eq!(
            apply_correction_commands("send the email. wait never mind"),
            "send the email."
        );
    }

    // ========================================================================
    // Inline corrections
    // ========================================================================

    #[test]
    fn test_i_mean_simple() {
        assert_eq!(
            apply_correction_commands("buy apples I mean oranges"),
            "buy oranges"
        );
    }

    #[test]
    fn test_wait_no() {
        assert_eq!(
            apply_correction_commands("meet at three wait no four"),
            "meet at four"
        );
    }

    #[test]
    fn test_or_rather() {
        assert_eq!(
            apply_correction_commands("the red one or rather blue one"),
            "the blue one"
        );
    }

    #[test]
    fn test_actually_correction() {
        // "actually" mid-sentence = correction
        assert_eq!(
            apply_correction_commands("send it Monday actually Tuesday"),
            "send it Tuesday"
        );
    }

    #[test]
    fn test_chained_corrections() {
        assert_eq!(
            apply_correction_commands("A I mean B I mean C"),
            "C"
        );
    }

    // ========================================================================
    // False positive guards
    // ========================================================================

    #[test]
    fn test_i_mean_at_start_is_discourse_marker() {
        assert_eq!(
            apply_correction_commands("I mean it's really great"),
            "I mean it's really great"
        );
    }

    #[test]
    fn test_i_mean_after_period_is_discourse_marker() {
        assert_eq!(
            apply_correction_commands("Yes. I mean it's obvious"),
            "Yes. I mean it's obvious"
        );
    }

    #[test]
    fn test_i_mean_with_comma_is_discourse_marker() {
        assert_eq!(
            apply_correction_commands("so I mean, it could work"),
            "so I mean, it could work"
        );
    }

    #[test]
    fn test_actually_at_start_is_emphasis() {
        assert_eq!(
            apply_correction_commands("Actually that sounds good"),
            "Actually that sounds good"
        );
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    #[test]
    fn test_empty_string() {
        assert_eq!(apply_correction_commands(""), "");
    }

    #[test]
    fn test_no_corrections() {
        assert_eq!(
            apply_correction_commands("hello world"),
            "hello world"
        );
    }
}
