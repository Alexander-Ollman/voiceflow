//! Filler word removal for speech transcription
//!
//! Removes common speech disfluencies like "um", "uh", "ah", "hmm", etc.
//! These are typical in spontaneous speech but unwanted in final output.

/// Remove common filler words from transcribed text.
///
/// Handles standalone fillers (word boundaries respected) to avoid
/// corrupting real words that contain filler substrings.
pub fn remove_filler_words(text: &str) -> String {
    // Filler words to remove (must match as whole words)
    const FILLERS: &[&str] = &[
        "um", "umm", "ummm",
        "uh", "uhh", "uhhh",
        "ah", "ahh", "ahhh",
        "hmm", "hmmm", "hm",
        "er", "erm",
        "eh",
        "mhm",
        "mm", "mmm",
    ];

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let filtered: Vec<&str> = words
        .into_iter()
        .filter(|word| {
            // Strip trailing punctuation/comma for comparison
            let stripped = word.trim_end_matches(|c: char| c == ',' || c == '.' || c == ';');
            let lower = stripped.to_lowercase();
            !FILLERS.contains(&lower.as_str())
        })
        .collect();

    let result = filtered.join(" ");

    // Clean up any double spaces
    let mut clean = result;
    while clean.contains("  ") {
        clean = clean.replace("  ", " ");
    }

    clean.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_um() {
        assert_eq!(
            remove_filler_words("So um I was thinking about it"),
            "So I was thinking about it"
        );
    }

    #[test]
    fn test_remove_uh() {
        assert_eq!(
            remove_filler_words("I uh want to add bananas"),
            "I want to add bananas"
        );
    }

    #[test]
    fn test_remove_multiple_fillers() {
        assert_eq!(
            remove_filler_words("So um I was uh thinking um about it"),
            "So I was thinking about it"
        );
    }

    #[test]
    fn test_remove_hmm() {
        assert_eq!(
            remove_filler_words("Hmm let me think about that"),
            "let me think about that"
        );
    }

    #[test]
    fn test_filler_with_comma() {
        assert_eq!(
            remove_filler_words("So, um, I was thinking"),
            "So, I was thinking"
        );
    }

    #[test]
    fn test_no_false_positives() {
        // "umbrella" contains "um" but should NOT be stripped
        assert_eq!(
            remove_filler_words("I need an umbrella"),
            "I need an umbrella"
        );
        // "human" contains "um" but should NOT be stripped
        assert_eq!(
            remove_filler_words("That is a human"),
            "That is a human"
        );
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(
            remove_filler_words("Um I was thinking Uh about it"),
            "I was thinking about it"
        );
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(remove_filler_words(""), "");
    }

    #[test]
    fn test_only_fillers() {
        assert_eq!(remove_filler_words("um uh ah"), "");
    }

    #[test]
    fn test_er_not_in_words() {
        // "er" filler should be removed, but "error" should not
        assert_eq!(
            remove_filler_words("er there was an error"),
            "there was an error"
        );
    }
}
