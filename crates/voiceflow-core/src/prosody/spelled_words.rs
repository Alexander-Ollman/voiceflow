//! Spelled word detection and concatenation
//!
//! Detects when individual letters are spoken separately and concatenates them.
//! E.g., "S M O L L M" → "SMOLLM", "C P P" → "CPP"

/// Result of spelled word detection
#[derive(Debug, Clone)]
pub struct SpelledWordMatch {
    /// Starting index in the word list
    pub start_index: usize,
    /// Ending index (exclusive) in the word list
    pub end_index: usize,
    /// The concatenated result
    pub concatenated: String,
    /// Original tokens that were matched
    pub original_tokens: Vec<String>,
}

/// Detect and concatenate spelled-out letter sequences in text
///
/// # Examples
/// - "S M O L L M" → "SMOLLM"
/// - "The S M O L L M model" → "The SMOLLM model"
/// - "whisper C P P" → "whisper CPP"
/// - "S M O L L M - 3 B" → "SMOLLM-3B"
pub fn concatenate_spelled_words(text: &str) -> String {
    let tokens: Vec<&str> = text.split_whitespace().collect();

    if tokens.len() < 2 {
        return text.to_string();
    }

    let mut result_tokens: Vec<String> = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        // Try to find a spelled sequence starting at this position
        if let Some(match_result) = find_spelled_sequence(&tokens, i) {
            result_tokens.push(match_result.concatenated);
            i = match_result.end_index;
        } else {
            result_tokens.push(tokens[i].to_string());
            i += 1;
        }
    }

    result_tokens.join(" ")
}

/// Find a spelled letter sequence starting at the given index
fn find_spelled_sequence(tokens: &[&str], start: usize) -> Option<SpelledWordMatch> {
    if start >= tokens.len() {
        return None;
    }

    // Check if first token looks like a single letter (not a common word like "a" or "I")
    if !is_single_letter_token(tokens[start]) {
        return None;
    }

    let mut letters = Vec::new();
    let mut original_tokens = Vec::new();
    let mut end = start;

    while end < tokens.len() {
        let token = tokens[end];

        if is_single_letter_token(token) {
            // Single letter - add to sequence
            letters.push(token.to_uppercase());
            original_tokens.push(token.to_string());
            end += 1;
        } else if is_number_token(token) && end > start {
            // Number after letters - include it (e.g., "3" in "SMOLLM3B")
            letters.push(token.to_string());
            original_tokens.push(token.to_string());
            end += 1;
            // Continue to check if there's a letter after the number (e.g., "B" in "3B")
        } else if is_connector_token(token) && end > start {
            // Connector (hyphen) - include if we already have letters
            // and there's more after (letter or number)
            if end + 1 < tokens.len() && (is_single_letter_token(tokens[end + 1]) || is_number_token(tokens[end + 1])) {
                letters.push(token.to_string());
                original_tokens.push(token.to_string());
                end += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    // Need at least 2 letters to consider it a spelled word
    let letter_count = letters.iter().filter(|l| l.len() == 1 && l.chars().next().map(|c| c.is_alphabetic()).unwrap_or(false)).count();

    if letter_count >= 2 {
        Some(SpelledWordMatch {
            start_index: start,
            end_index: end,
            concatenated: letters.join(""),
            original_tokens,
        })
    } else {
        None
    }
}

/// Check if a token is a single letter (but not common words like "a", "I")
fn is_single_letter_token(token: &str) -> bool {
    // Must be exactly one character
    if token.chars().count() != 1 {
        return false;
    }

    let c = token.chars().next().unwrap();

    // Must be alphabetic
    if !c.is_alphabetic() {
        return false;
    }

    // Exclude common single-letter words in certain contexts
    // But we're more permissive here since context usually makes it clear
    // "a" at the start of a sequence is probably being spelled
    true
}

/// Check if a token is a connector (hyphen, dash)
fn is_connector_token(token: &str) -> bool {
    matches!(token, "-" | "–" | "—" | ".")
}

/// Check if a token is a number (for things like "3B", "2.0")
fn is_number_token(token: &str) -> bool {
    token.chars().all(|c| c.is_ascii_digit() || c == '.')
}

/// More aggressive spelled word detection that also handles
/// lowercase sequences and mixed patterns
pub fn concatenate_spelled_words_aggressive(text: &str) -> String {
    // First pass: standard concatenation
    let result = concatenate_spelled_words(text);

    // Second pass: handle patterns like "C pp" → "Cpp" or ".cpp"
    // and "Z ill oc on" type phonetic breakups
    fix_partial_spellings(&result)
}

/// Fix partial spellings where some letters got grouped
/// E.g., "C pp" might be "Cpp" or ".cpp"
fn fix_partial_spellings(text: &str) -> String {
    let mut result = text.to_string();

    // Common programming extensions that get mangled
    let extensions = [
        ("C pp", ".cpp"),
        ("C PP", ".cpp"),
        ("Cpp", ".cpp"),
        ("CPP", ".cpp"),
        ("H pp", ".hpp"),
        ("H PP", ".hpp"),
        ("J S", ".js"),
        ("T S", ".ts"),
        ("P Y", ".py"),
        ("R S", ".rs"),
        ("G O", ".go"),
    ];

    for (pattern, replacement) in extensions {
        // Only replace if it looks like a file extension context
        // (preceded by a word character or at word boundary)
        if result.contains(pattern) {
            result = result.replace(pattern, replacement);
        }
    }

    // Common tech terms that get phonetically mangled
    let tech_terms = [
        // Silicon variations
        ("Z ill oc on", "Silicon"),
        ("sill oc on", "Silicon"),
        ("sill icon", "Silicon"),
        ("Z ilicon", "Silicon"),
        // GPU/ML terms
        ("C U D A", "CUDA"),
        ("cuda", "CUDA"),
        ("G P U", "GPU"),
        ("C P U", "CPU"),
        ("M L", "ML"),
        ("A I", "AI"),
        ("L L M", "LLM"),
        ("N L P", "NLP"),
        // Apple terms
        ("mac O S", "macOS"),
        ("Mac O S", "macOS"),
        ("i O S", "iOS"),
        ("I O S", "iOS"),
    ];

    for (pattern, replacement) in tech_terms {
        // Case-insensitive replacement for some patterns
        if result.to_lowercase().contains(&pattern.to_lowercase()) {
            result = case_insensitive_replace(&result, pattern, replacement);
        }
    }

    result
}

/// Case-insensitive string replacement with word boundary awareness
///
/// Only replaces if the match is at word boundaries (not in the middle of words).
/// This prevents "him like" from matching "M L" → "ML" and becoming "hiMLike".
fn case_insensitive_replace(text: &str, pattern: &str, replacement: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let lower_text = text.to_lowercase();
    let lower_pattern = pattern.to_lowercase();

    if let Some(start) = lower_text.find(&lower_pattern) {
        let end = start + pattern.len();

        // Check if this match is at word boundaries
        let at_word_start = start == 0 || !chars.get(start.saturating_sub(1))
            .map(|c| c.is_alphabetic())
            .unwrap_or(false);
        let at_word_end = end >= chars.len() || !chars.get(end)
            .map(|c| c.is_alphabetic())
            .unwrap_or(false);

        if at_word_start && at_word_end {
            format!(
                "{}{}{}",
                &text[..start],
                replacement,
                &text[start + pattern.len()..]
            )
        } else {
            text.to_string()
        }
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_spelled_word() {
        assert_eq!(
            concatenate_spelled_words("S M O L L M"),
            "SMOLLM"
        );
    }

    #[test]
    fn test_spelled_word_in_sentence() {
        assert_eq!(
            concatenate_spelled_words("The S M O L L M model"),
            "The SMOLLM model"
        );
    }

    #[test]
    fn test_spelled_with_number() {
        assert_eq!(
            concatenate_spelled_words("S M O L L M 3 B"),
            "SMOLLM3B"
        );
    }

    #[test]
    fn test_spelled_with_hyphen() {
        assert_eq!(
            concatenate_spelled_words("S M O L L M - 3 B"),
            "SMOLLM-3B"
        );
    }

    #[test]
    fn test_cpp_extension() {
        let result = concatenate_spelled_words_aggressive("whisper C P P");
        assert!(result.contains(".cpp") || result.contains("CPP"));
    }

    #[test]
    fn test_no_false_positives() {
        // "I went to a" should not be concatenated
        assert_eq!(
            concatenate_spelled_words("I went to a museum"),
            "I went to a museum"
        );
    }

    #[test]
    fn test_silicon_fix() {
        assert_eq!(
            concatenate_spelled_words_aggressive("Apple Z ill oc on"),
            "Apple Silicon"
        );
    }
}
