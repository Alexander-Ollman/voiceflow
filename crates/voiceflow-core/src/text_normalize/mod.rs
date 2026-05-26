//! Deterministic text normalization pipeline
//!
//! Replaces the LLM post-processing step with a fast, deterministic chain of
//! transforms. All existing pre/post-processing functions plus two new ones
//! (correction commands + stammer dedup).
//!
//! Target latency: <5ms for typical transcripts.

mod correction_commands;
mod intent_classifier;
mod stammer_dedup;

use crate::llm::numbers::{fix_abbreviations, normalize_numbers};
use crate::llm::prompts::{
    fix_punctuation_spacing, fix_url_spacing, lowercase_urls, normalize_ellipses,
    strip_trailing_ellipsis,
};
use crate::prosody::{
    concatenate_spelled_words_aggressive, fix_tokenization_artifacts, remove_filler_words,
    replace_voice_commands, ReplacementDictionary,
};

pub use correction_commands::apply_correction_commands;
pub use intent_classifier::{classify_intent, AnchorHint, CommandKind, IntentKind, IntentResult};
pub use stammer_dedup::dedup_stammers;

/// Result of deterministic normalization
#[derive(Debug, Clone)]
pub struct NormalizeResult {
    /// The fully normalized text
    pub text: String,
    /// Number of transforms applied (for diagnostics)
    pub transforms_applied: u32,
}

/// Run the full deterministic normalization pipeline.
///
/// Pipeline order (15 steps):
///  1. fix_tokenization_artifacts  — fix mid-word punctuation/caps from STT
///  2. remove_filler_words         — strip um, uh, ah, hmm
///  3. apply_correction_commands   — "scratch that", "I mean X"  (NEW)
///  4. dedup_stammers              — "the the" → "the"            (NEW)
///  5. replace_voice_commands      — "period" → "."
///  6. concatenate_spelled_words   — "S M O L L M" → "SMOLLM"
///  7. replacements.apply          — user/built-in word replacements
///  8. fix_url_spacing             — "google. Com" → "google.com"
///  9. lowercase_urls              — "www.Google.Com" → "www.google.com"
/// 10. fix_punctuation_spacing     — "Hello.World" → "Hello. World"
/// 11. normalize_ellipses          — "wait....." → "wait..."
/// 12. strip_trailing_ellipsis     — "Hello world..." → "Hello world."
/// 13. normalize_numbers           — "fifty thousand" → "50,000"
/// 14. fix_abbreviations           — "doctor Smith" → "Dr. Smith"
/// 15. Final trim
pub fn normalize(text: &str, replacements: &ReplacementDictionary) -> NormalizeResult {
    if text.trim().is_empty() {
        return NormalizeResult {
            text: String::new(),
            transforms_applied: 0,
        };
    }

    let mut s = text.to_string();
    let mut count = 0u32;

    // 1. Fix tokenization artifacts
    s = fix_tokenization_artifacts(&s);
    count += 1;

    // 2. Remove filler words
    s = remove_filler_words(&s);
    count += 1;

    // 3. Correction commands (NEW) — needs raw words before voice commands convert them
    s = apply_correction_commands(&s);
    count += 1;

    // 4. Stammer dedup (NEW) — after filler removal ("the um the" → "the the" → "the")
    s = dedup_stammers(&s);
    count += 1;

    // 5. Voice commands — "period" → ".", "new line" → "\n"
    s = replace_voice_commands(&s);
    count += 1;

    // 6. Concatenate spelled words — "S M O L L M" → "SMOLLM"
    s = concatenate_spelled_words_aggressive(&s);
    count += 1;

    // 7. User/built-in replacements
    s = replacements.apply(&s);
    count += 1;

    // 8. Fix URL spacing
    s = fix_url_spacing(&s);
    count += 1;

    // 9. Lowercase URLs
    s = lowercase_urls(&s);
    count += 1;

    // 10. Fix punctuation spacing
    s = fix_punctuation_spacing(&s);
    count += 1;

    // 11. Normalize ellipses
    s = normalize_ellipses(&s);
    count += 1;

    // 12. Strip trailing ellipsis
    s = strip_trailing_ellipsis(&s);
    count += 1;

    // 13. Normalize numbers
    s = normalize_numbers(&s);
    count += 1;

    // 14. Fix abbreviations
    s = fix_abbreviations(&s);
    count += 1;

    // 15. Final trim
    let text = s.trim().to_string();
    count += 1;

    NormalizeResult {
        text,
        transforms_applied: count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize_text(text: &str) -> String {
        let replacements = ReplacementDictionary::load_default();
        normalize(text, &replacements).text
    }

    #[test]
    fn test_empty() {
        assert_eq!(normalize_text(""), "");
        assert_eq!(normalize_text("   "), "");
    }

    #[test]
    fn test_passthrough() {
        assert_eq!(normalize_text("Hello world."), "Hello world.");
    }

    #[test]
    fn test_filler_removal() {
        let result = normalize_text("Um hello uh world");
        // voice_commands capitalizes first letter, so "hello" → "Hello"
        let lower = result.to_lowercase();
        assert!(lower.contains("hello"), "result was: {}", result);
        assert!(!lower.contains("um "), "result was: {}", result);
        assert!(!lower.contains(" uh "), "result was: {}", result);
    }

    #[test]
    fn test_stammer_dedup() {
        let result = normalize_text("the the meeting");
        // voice_commands capitalizes first letter → "The meeting"
        let lower = result.to_lowercase();
        assert!(lower.contains("the meeting"), "result was: {}", result);
        // Should not have "the the" anymore
        assert!(!lower.contains("the the"), "result was: {}", result);
    }

    #[test]
    fn test_correction_command() {
        let result = normalize_text("buy apples I mean oranges");
        let lower = result.to_lowercase();
        assert!(lower.contains("oranges"), "result was: {}", result);
        assert!(!lower.contains("apples"), "result was: {}", result);
    }

    #[test]
    fn test_number_normalization() {
        let result = normalize_text("there are fifty thousand items");
        assert!(result.contains("50,000"), "result was: {}", result);
    }

    #[test]
    fn test_url_handling() {
        // fix_url_spacing collapses ". Com" → ".com" (safe TLD pattern)
        // but lowercase_urls only lowercases URLs with www/http prefix
        let result = normalize_text("visit www. Google. Com");
        assert!(result.contains("www.google.com"), "result was: {}", result);
    }

    #[test]
    fn test_transforms_count() {
        let replacements = ReplacementDictionary::load_default();
        let result = normalize("hello world", &replacements);
        assert_eq!(result.transforms_applied, 15);
    }

    #[test]
    fn test_pipeline_order_correction_before_voice_commands() {
        // "scratch that" should remove the clause before voice commands run.
        // "hello. world scratch that" → deletion removes "world scratch that" → "hello."
        // voice_commands capitalizes first letter → "Hello."
        let result = normalize_text("hello. world scratch that");
        assert_eq!(result, "Hello.");
    }
}
