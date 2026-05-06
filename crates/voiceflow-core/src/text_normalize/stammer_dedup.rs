//! Stammer and repetition deduplication
//!
//! Two patterns:
//! 1. Exact repeat: "the the meeting" → "the meeting"
//! 2. Prefix stammer: "infra infrastructure" → "infrastructure"

/// Words exempt from deduplication (intentional repetition).
const EXEMPT_WORDS: &[&str] = &[
    "very", "really", "so", "that", "had", "do", "no", "bye",
    "go", "now", "wait",
];

/// Remove stammers and accidental word repetitions.
pub fn dedup_stammers(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 2 {
        return text.to_string();
    }

    let mut result: Vec<&str> = Vec::with_capacity(words.len());
    let mut i = 0;

    while i < words.len() {
        let word = words[i];
        let word_lower = word.to_lowercase();

        if i + 1 < words.len() {
            let next = words[i + 1];
            let next_lower = next.to_lowercase();

            // Check exemption
            if !is_exempt(&word_lower) {
                // Pattern 1: Exact repeat (case-insensitive)
                if word_lower == next_lower {
                    // Keep the second occurrence (may have better casing)
                    i += 1;
                    continue;
                }

                // Pattern 2: Prefix stammer — word is a ≥3-char prefix of the next word
                if word_lower.len() >= 3
                    && next_lower.len() > word_lower.len()
                    && next_lower.starts_with(&word_lower)
                {
                    // Skip the prefix stammer, the next word is the intended one
                    i += 1;
                    continue;
                }
            }
        }

        result.push(word);
        i += 1;
    }

    result.join(" ")
}

fn is_exempt(word: &str) -> bool {
    EXEMPT_WORDS.contains(&word)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Exact repeats
    // ========================================================================

    #[test]
    fn test_exact_repeat() {
        assert_eq!(dedup_stammers("the the meeting"), "the meeting");
    }

    #[test]
    fn test_exact_repeat_case_insensitive() {
        assert_eq!(dedup_stammers("The the meeting"), "the meeting");
    }

    #[test]
    fn test_multiple_repeats() {
        assert_eq!(
            dedup_stammers("I I went to the the store"),
            "I went to the store"
        );
    }

    // ========================================================================
    // Prefix stammers
    // ========================================================================

    #[test]
    fn test_prefix_stammer() {
        assert_eq!(
            dedup_stammers("infra infrastructure"),
            "infrastructure"
        );
    }

    #[test]
    fn test_prefix_stammer_short() {
        assert_eq!(
            dedup_stammers("app application"),
            "application"
        );
    }

    #[test]
    fn test_prefix_too_short() {
        // 2-char prefix should NOT be treated as stammer
        assert_eq!(
            dedup_stammers("in information"),
            "in information"
        );
    }

    // ========================================================================
    // Exemptions
    // ========================================================================

    #[test]
    fn test_exempt_very() {
        assert_eq!(
            dedup_stammers("very very good"),
            "very very good"
        );
    }

    #[test]
    fn test_exempt_really() {
        assert_eq!(
            dedup_stammers("really really fast"),
            "really really fast"
        );
    }

    #[test]
    fn test_exempt_no() {
        assert_eq!(
            dedup_stammers("no no that's wrong"),
            "no no that's wrong"
        );
    }

    #[test]
    fn test_exempt_bye() {
        assert_eq!(dedup_stammers("bye bye"), "bye bye");
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    #[test]
    fn test_empty() {
        assert_eq!(dedup_stammers(""), "");
    }

    #[test]
    fn test_single_word() {
        assert_eq!(dedup_stammers("hello"), "hello");
    }

    #[test]
    fn test_no_stammers() {
        assert_eq!(
            dedup_stammers("this is a normal sentence"),
            "this is a normal sentence"
        );
    }
}
