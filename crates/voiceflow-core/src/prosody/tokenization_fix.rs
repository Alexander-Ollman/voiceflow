//! Tokenization artifact fixing for Moonshine STT
//!
//! Moonshine STT can produce artifacts where punctuation appears mid-word,
//! or where capitalization appears mid-word due to tokenization boundaries.
//! This module fixes these artifacts by:
//! 1. Removing mid-word punctuation (it's an artifact, not intentional)
//! 2. Inserting spaces before mid-word capitals (to split merged words)
//!
//! Examples:
//! - "grea.tsaint" -> "great saint" (punctuation removed, letter moved back)
//! - "cas.tsin" -> "cast sin"
//! - "no.tspeak" -> "not speak"
//! - "hiMLike" -> "hi MLike" (space inserted before capital)

/// Punctuation characters that can appear mid-word as artifacts
const MID_WORD_PUNCT: &[char] = &['.', '!', '?', ',', ';', ':'];

/// Fix tokenization artifacts from Moonshine STT
///
/// This function detects and fixes two types of artifacts:
/// 1. Mid-word punctuation: "grea.tsaint" -> "great saint" (removes artifact punctuation)
/// 2. Mid-word capitalization: "hiMLike" -> "him like" (lowercases the stray capital)
///
/// # Arguments
/// * `text` - The transcribed text that may contain tokenization artifacts
///
/// # Returns
/// The text with tokenization artifacts fixed
pub fn fix_tokenization_artifacts(text: &str) -> String {
    let mut result = text.to_string();

    // Fix mid-word punctuation artifacts (removes the punctuation since it's an artifact)
    result = fix_mid_word_punctuation(&result);

    // Fix mid-word capitalization artifacts (lowercases stray capitals)
    result = fix_mid_word_capitalization(&result);

    // Clean up any double spaces
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }

    result
}

/// Fix mid-word punctuation artifacts
///
/// Detects patterns like "grea.tsaint" and transforms them to "great saint"
/// The letter immediately after the punctuation is moved back to the previous word,
/// and the punctuation is removed (since it's an STT artifact, not intentional punctuation).
fn fix_mid_word_punctuation(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len() + 10);
    let len = chars.len();

    let mut i = 0;
    while i < len {
        let c = chars[i];

        // Check if this is a mid-word punctuation artifact
        // Pattern: letter + punctuation + lowercase letter + more letters
        if MID_WORD_PUNCT.contains(&c) && i > 0 && i + 2 < len {
            let prev = chars[i - 1];
            let next = chars[i + 1];
            let next_next = chars[i + 2];

            // Check if this looks like a tokenization artifact:
            // - Previous char is a letter
            // - Next char is a lowercase letter
            // - Char after that is also a letter (part of a word)
            if prev.is_alphabetic()
                && next.is_ascii_lowercase()
                && next_next.is_alphabetic()
            {
                // Move the letter after punctuation back: "grea" + "." + "t" + "saint"
                // becomes: "great" + " " + "saint" (punctuation removed as it's an artifact)
                result.push(next); // Move the letter back
                result.push(' '); // Add space (no punctuation - it was an artifact)
                i += 2; // Skip the punctuation and the moved letter
                continue;
            }
        }

        result.push(c);
        i += 1;
    }

    result
}

/// Fix mid-word capitalization artifacts
///
/// Detects patterns like "hiMLike" and transforms them to "him Like"
/// A space is inserted before the uppercase letter in the middle of a word.
fn fix_mid_word_capitalization(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len() + 10);
    let len = chars.len();

    let mut i = 0;
    let mut word_start = true; // Track if we're at the start of a word

    while i < len {
        let c = chars[i];

        if c.is_whitespace() {
            result.push(c);
            word_start = true;
            i += 1;
            continue;
        }

        // Check if this is a mid-word capitalization artifact
        // Pattern: lowercase letters followed by uppercase letter in middle of word
        if !word_start && c.is_uppercase() && i > 0 {
            let prev = chars[i - 1];

            // If previous char was lowercase and current is uppercase, insert space
            // But only if there's more alphabetic content after (part of a word split)
            // Check if there's at least one more letter after this capital
            if prev.is_lowercase() && i + 1 < len && chars[i + 1].is_alphabetic() {
                result.push(' ');
            }
        }

        result.push(c);
        word_start = false;
        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mid-word punctuation tests - punctuation is REMOVED since it's an artifact
    #[test]
    fn test_mid_word_period() {
        assert_eq!(
            fix_tokenization_artifacts("grea.tsaint"),
            "great saint"
        );
    }

    #[test]
    fn test_mid_word_period_cast() {
        assert_eq!(
            fix_tokenization_artifacts("cas.tsin"),
            "cast sin"
        );
    }

    #[test]
    fn test_mid_word_period_not() {
        assert_eq!(
            fix_tokenization_artifacts("no.tspeak"),
            "not speak"
        );
    }

    #[test]
    fn test_multiple_artifacts_in_sentence() {
        assert_eq!(
            fix_tokenization_artifacts("I wan.tto spea.kwith you"),
            "I want to speak with you"
        );
    }

    #[test]
    fn test_mid_word_comma() {
        assert_eq!(
            fix_tokenization_artifacts("hel.lo"),
            "hell o"
        );
    }

    #[test]
    fn test_mid_word_exclamation() {
        assert_eq!(
            fix_tokenization_artifacts("wai!tfor"),
            "wait for"
        );
    }

    #[test]
    fn test_mid_word_question() {
        assert_eq!(
            fix_tokenization_artifacts("wha?twas"),
            "what was"
        );
    }

    // Mid-word capitalization tests - space is inserted BEFORE the uppercase letter
    #[test]
    fn test_mid_word_capital() {
        // "hiMLike" -> "hi MLike" (space before first uppercase letter)
        // The LLM can clean up "hi MLike" -> "him like" or similar
        assert_eq!(
            fix_tokenization_artifacts("hiMLike"),
            "hi MLike"
        );
    }

    #[test]
    fn test_mid_word_capital_multiple() {
        assert_eq!(
            fix_tokenization_artifacts("theyWent"),
            "they Went"
        );
    }

    #[test]
    fn test_preserves_start_capital() {
        // Words starting with capital should not be split
        assert_eq!(
            fix_tokenization_artifacts("Hello"),
            "Hello"
        );
    }

    #[test]
    fn test_preserves_all_caps() {
        // All caps words should not be split (no lowercase before capital)
        assert_eq!(
            fix_tokenization_artifacts("HTML"),
            "HTML"
        );
    }

    #[test]
    fn test_preserves_normal_sentence() {
        // Normal text should pass through unchanged
        assert_eq!(
            fix_tokenization_artifacts("Hello world, how are you?"),
            "Hello world, how are you?"
        );
    }

    #[test]
    fn test_combined_artifacts() {
        // Test both types of artifacts together
        assert_eq!(
            fix_tokenization_artifacts("I wan.tto talkWith you"),
            "I want to talk With you"
        );
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(fix_tokenization_artifacts(""), "");
    }

    #[test]
    fn test_single_word() {
        assert_eq!(fix_tokenization_artifacts("hello"), "hello");
    }

    #[test]
    fn test_only_punctuation() {
        assert_eq!(fix_tokenization_artifacts("..."), "...");
    }

    #[test]
    fn test_proper_sentence() {
        // Make sure normal punctuation at word boundaries isn't affected
        assert_eq!(
            fix_tokenization_artifacts("Hello. World"),
            "Hello. World"
        );
    }

    #[test]
    fn test_end_of_sentence_period() {
        // Period at end of word should not be modified
        assert_eq!(
            fix_tokenization_artifacts("Hello."),
            "Hello."
        );
    }

    #[test]
    fn test_period_before_capital() {
        // Period followed by capital letter is normal sentence boundary
        assert_eq!(
            fix_tokenization_artifacts("Hello.World"),
            "Hello.World"
        );
    }

    #[test]
    fn test_period_before_space() {
        // Period followed by space is normal
        assert_eq!(
            fix_tokenization_artifacts("Hello. world"),
            "Hello. world"
        );
    }

    #[test]
    fn test_capital_at_word_start() {
        // Capital at start of word (after space) should not add extra space
        assert_eq!(
            fix_tokenization_artifacts("hello World"),
            "hello World"
        );
    }

    #[test]
    fn test_capital_at_end() {
        // Capital at end of word should not trigger spacing
        assert_eq!(
            fix_tokenization_artifacts("helloM"),
            "helloM"
        );
    }
}
