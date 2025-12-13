//! Voice command detection for explicit punctuation
//!
//! Detects spoken punctuation commands like "period", "comma", "question mark"
//! and replaces them with the actual punctuation characters.

use std::collections::HashMap;

/// Replace spoken punctuation commands with actual punctuation
///
/// Handles various forms like "period", "full stop", "comma", etc.
/// Also handles "new line", "new paragraph" for formatting.
pub fn replace_voice_commands(text: &str) -> String {
    let mut result = text.to_string();

    // Build replacement map with all variations
    let replacements = build_replacement_map();

    // Sort by length (longest first) to avoid partial replacements
    let mut sorted_keys: Vec<_> = replacements.keys().collect();
    sorted_keys.sort_by(|a, b| b.len().cmp(&a.len()));

    // Apply replacements (case-insensitive)
    for key in sorted_keys {
        let replacement = replacements.get(*key).unwrap();
        result = replace_case_insensitive(&result, key, replacement);
    }

    // Clean up any double spaces created by replacements
    while result.contains("  ") {
        result = result.replace("  ", " ");
    }

    // Clean up space before punctuation
    result = cleanup_punctuation_spacing(&result);

    result
}

/// Build the map of voice commands to punctuation
fn build_replacement_map() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();

    // Period / Full stop
    map.insert(" period", ".");
    map.insert(" full stop", ".");
    map.insert(" dot", ".");
    map.insert(" end sentence", ".");

    // Comma
    map.insert(" comma", ",");

    // Question mark
    map.insert(" question mark", "?");
    map.insert(" question", "?"); // Be careful - might be part of actual word

    // Exclamation
    map.insert(" exclamation mark", "!");
    map.insert(" exclamation point", "!");
    map.insert(" exclamation", "!");
    map.insert(" bang", "!");

    // Colon and semicolon
    map.insert(" colon", ":");
    map.insert(" semicolon", ";");
    map.insert(" semi colon", ";");

    // Quotes
    map.insert(" open quote", "\"");
    map.insert(" close quote", "\"");
    map.insert(" quote", "\"");
    map.insert(" open single quote", "'");
    map.insert(" close single quote", "'");
    map.insert(" apostrophe", "'");

    // Parentheses and brackets
    map.insert(" open paren", "(");
    map.insert(" close paren", ")");
    map.insert(" open parenthesis", "(");
    map.insert(" close parenthesis", ")");
    map.insert(" left paren", "(");
    map.insert(" right paren", ")");
    map.insert(" open bracket", "[");
    map.insert(" close bracket", "]");
    map.insert(" open brace", "{");
    map.insert(" close brace", "}");

    // Dashes and hyphens
    map.insert(" dash", "—");
    map.insert(" em dash", "—");
    map.insert(" en dash", "–");
    map.insert(" hyphen", "-");

    // New lines and paragraphs
    map.insert(" new line", "\n");
    map.insert(" newline", "\n");
    map.insert(" line break", "\n");
    map.insert(" new paragraph", "\n\n");
    map.insert(" paragraph", "\n\n");
    map.insert(" next paragraph", "\n\n");

    // Ellipsis
    map.insert(" ellipsis", "...");
    map.insert(" dot dot dot", "...");

    // Special characters
    map.insert(" ampersand", "&");
    map.insert(" at sign", "@");
    map.insert(" at symbol", "@");
    map.insert(" hashtag", "#");
    map.insert(" hash", "#");
    map.insert(" dollar sign", "$");
    map.insert(" percent", "%");
    map.insert(" percent sign", "%");
    map.insert(" asterisk", "*");
    map.insert(" star", "*");
    map.insert(" underscore", "_");
    map.insert(" slash", "/");
    map.insert(" forward slash", "/");
    map.insert(" backslash", "\\");
    map.insert(" back slash", "\\");

    // Programming-specific
    map.insert(" equals", "=");
    map.insert(" plus", "+");
    map.insert(" minus", "-");
    map.insert(" greater than", ">");
    map.insert(" less than", "<");
    map.insert(" pipe", "|");
    map.insert(" tilde", "~");
    map.insert(" caret", "^");

    map
}

/// Case-insensitive replacement
fn replace_case_insensitive(text: &str, pattern: &str, replacement: &str) -> String {
    let lower_text = text.to_lowercase();
    let lower_pattern = pattern.to_lowercase();

    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;

    for (start, _) in lower_text.match_indices(&lower_pattern) {
        // Add text before this match
        result.push_str(&text[last_end..start]);
        // Add replacement
        result.push_str(replacement);
        last_end = start + pattern.len();
    }

    // Add remaining text
    result.push_str(&text[last_end..]);

    result
}

/// Clean up spacing around punctuation
fn cleanup_punctuation_spacing(text: &str) -> String {
    let mut result = text.to_string();

    // Remove space before punctuation
    let punctuation = ['.', ',', '?', '!', ':', ';', ')', ']', '}', '"', '\'', '—', '–', '-'];
    for p in punctuation {
        result = result.replace(&format!(" {}", p), &p.to_string());
    }

    // Remove space after newlines (but keep the newline)
    while result.contains("\n ") {
        result = result.replace("\n ", "\n");
    }

    // Ensure space after sentence-ending punctuation (if followed by letter)
    let chars: Vec<char> = result.chars().collect();
    let mut final_result = String::with_capacity(result.len());

    for i in 0..chars.len() {
        final_result.push(chars[i]);

        // After sentence-ending punctuation, ensure space before next letter
        if matches!(chars[i], '.' | '?' | '!' | ':') {
            if let Some(&next) = chars.get(i + 1) {
                if next.is_alphabetic() {
                    final_result.push(' ');
                }
            }
        }
    }

    // Capitalize after sentence-ending punctuation
    capitalize_after_punctuation(&final_result)
}

/// Capitalize the first letter after sentence-ending punctuation
fn capitalize_after_punctuation(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut capitalize_next = true;

    for c in text.chars() {
        if capitalize_next && c.is_alphabetic() {
            result.push(c.to_uppercase().next().unwrap_or(c));
            capitalize_next = false;
        } else {
            result.push(c);
        }

        // Capitalize after sentence endings
        if matches!(c, '.' | '?' | '!' | '\n') {
            capitalize_next = true;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_punctuation() {
        assert_eq!(
            replace_voice_commands("Hello period How are you question mark"),
            "Hello. How are you?"
        );
    }

    #[test]
    fn test_comma() {
        assert_eq!(
            replace_voice_commands("Well comma I think so"),
            "Well, I think so"
        );
    }

    #[test]
    fn test_new_line() {
        assert_eq!(
            replace_voice_commands("First line new line Second line"),
            "First line\nSecond line"
        );
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(
            replace_voice_commands("Hello PERIOD goodbye"),
            "Hello. Goodbye"
        );
    }

    #[test]
    fn test_exclamation() {
        assert_eq!(
            replace_voice_commands("Wow exclamation mark That's amazing"),
            "Wow! That's amazing"
        );
    }
}
