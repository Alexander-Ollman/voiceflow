//! Prompt formatting and output post-processing utilities

use super::numbers::{fix_abbreviations, normalize_numbers};
use crate::config::Config;

/// Format a prompt template with the transcript and config
pub fn format_prompt(template: &str, transcript: &str, config: &Config) -> String {
    let mut prompt = template.replace("{transcript}", transcript);

    // Add personal dictionary if present
    if !config.personal_dictionary.is_empty() {
        let dict_str = config.personal_dictionary.join(", ");
        prompt = prompt.replace(
            "{personal_dictionary}",
            &format!("\nPersonal vocabulary: {}", dict_str),
        );
    } else {
        prompt = prompt.replace("{personal_dictionary}", "");
    }

    // Add /no_think for Qwen3 models when thinking is disabled (faster inference).
    // Insert before "## Input" so it's treated as an instruction, not output content.
    if !config.llm_options.enable_thinking {
        if let Some(pos) = prompt.find("## Input") {
            prompt.insert_str(pos, "/no_think\n\n");
        } else {
            // No ## Input marker — place before the last line
            prompt.push_str(" /no_think");
        }
    }

    prompt
}

/// Build a chat-formatted prompt for Qwen3/SmolLM3
#[allow(dead_code)]
pub fn build_chat_prompt(system: &str, user: &str, enable_thinking: bool) -> String {
    if enable_thinking {
        // With thinking enabled (slower but more thoughtful)
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user
        )
    } else {
        // Direct response (faster) - add /no_think for Qwen3
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{} /no_think<|im_end|>\n<|im_start|>assistant\n",
            system, user
        )
    }
}

//=============================================================================
// Output Post-Processing
//=============================================================================

/// Apply all post-processing to LLM output
pub fn post_process_output(text: &str) -> String {
    let trimmed = text.trim();
    let urls_fixed = fix_url_spacing(trimmed);
    let urls_lowercased = lowercase_urls(&urls_fixed);
    let fixed = fix_punctuation_spacing(&urls_lowercased);
    let ellipses = normalize_ellipses(&fixed);
    let numbers = normalize_numbers(&ellipses);
    fix_abbreviations(&numbers)
}

/// Lowercase URLs - "www.Google.Com" → "www.google.com"
fn lowercase_urls(text: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Check if we're at the start of a URL
        let remaining: String = chars[i..].iter().collect();
        let remaining_lower = remaining.to_lowercase();

        if remaining_lower.starts_with("www.") ||
           remaining_lower.starts_with("http://") ||
           remaining_lower.starts_with("https://") {
            // Find the end of the URL (space or end of string)
            let url_end = remaining.find(|c: char| c.is_whitespace()).unwrap_or(remaining.len());
            let url = &remaining[..url_end];

            // Lowercase the entire URL
            result.push_str(&url.to_lowercase());
            i += url_end;
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Fix spacing in URLs - remove spaces around dots in URL patterns
/// "www. google. com" → "www.google.com"
fn fix_url_spacing(text: &str) -> String {
    let mut result = text.to_string();

    // Common TLDs to detect URLs
    const TLDS: &[&str] = &[
        "com", "org", "net", "edu", "gov", "io", "co", "dev", "app", "ai", "ly", "me",
        "uk", "us", "ca", "de", "fr", "jp", "au", "in", "ru", "br", "it", "es", "nl",
    ];

    // Fix ". tld" patterns (space before TLD)
    for tld in TLDS {
        // ". com" → ".com"
        result = result.replace(&format!(". {}", tld), &format!(".{}", tld));
        // ". Com" → ".com" (case insensitive)
        let tld_cap: String = tld.chars().enumerate()
            .map(|(i, c)| if i == 0 { c.to_ascii_uppercase() } else { c })
            .collect();
        result = result.replace(&format!(". {}", tld_cap), &format!(".{}", tld));
        // ". COM" → ".com"
        result = result.replace(&format!(". {}", tld.to_uppercase()), &format!(".{}", tld));
    }

    // Fix "www. " pattern
    result = result.replace("www. ", "www.");
    result = result.replace("Www. ", "www.");
    result = result.replace("WWW. ", "www.");

    // Fix "http:// " and "https:// " patterns
    result = result.replace("http:// ", "http://");
    result = result.replace("https:// ", "https://");
    result = result.replace("Http:// ", "http://");
    result = result.replace("Https:// ", "https://");

    // Now handle middle parts of URLs (e.g., "google. " when followed by a TLD)
    // This is trickier - we need to find patterns like "word. word. tld"
    for tld in TLDS {
        // Look for patterns like "something. tld" that we haven't caught yet
        let pattern = format!(". {}", tld);
        if result.contains(&pattern) {
            result = result.replace(&pattern, &format!(".{}", tld));
        }
    }

    // Clean up any remaining "word. word" patterns that are clearly part of URLs
    // by checking if they're near www or end with a TLD
    let words: Vec<&str> = result.split_whitespace().collect();
    if words.len() >= 2 {
        let mut new_result = String::new();
        let mut i = 0;
        let chars: Vec<char> = result.chars().collect();

        while i < chars.len() {
            new_result.push(chars[i]);

            // If we just pushed a dot followed by space, check if we should remove the space
            if chars[i] == '.' && i + 1 < chars.len() && chars[i + 1] == ' ' {
                // Look ahead to see if this could be a URL
                let remaining: String = chars[i + 2..].iter().take(20).collect();
                let next_word: String = remaining.chars().take_while(|c| c.is_alphabetic()).collect();
                let next_lower = next_word.to_lowercase();

                // Check if next word is followed by a TLD pattern
                let has_tld_after = TLDS.iter().any(|tld| {
                    remaining.to_lowercase().contains(&format!(".{}", tld)) ||
                    next_lower == *tld
                });

                // Check if we recently had "www"
                let lookback: String = new_result.chars().rev().take(20).collect::<String>().chars().rev().collect();
                let has_www = lookback.to_lowercase().contains("www");

                if has_tld_after || has_www {
                    // Skip the space
                    i += 1;
                }
            }

            i += 1;
        }
        result = new_result;
    }

    result
}

/// Check if we're in a URL context (should not add space after dot)
fn is_url_context(chars: &[char], dot_pos: usize) -> bool {
    const TLDS: &[&str] = &[
        "com", "org", "net", "edu", "gov", "io", "co", "dev", "app", "ai", "ly", "me",
        "uk", "us", "ca", "de", "fr", "jp", "au", "in", "ru", "br", "it", "es", "nl",
    ];

    // Get word after the dot
    let after: String = chars[dot_pos + 1..].iter().take(15).collect();
    let word: String = after.chars().take_while(|c| c.is_alphabetic()).collect();
    let word_lower = word.to_lowercase();

    // If next word is a TLD, it's a URL
    if TLDS.contains(&word_lower.as_str()) {
        return true;
    }

    // Check if "www" or "http" appears before this dot
    if dot_pos >= 3 {
        let before: String = chars[dot_pos.saturating_sub(20)..dot_pos].iter().collect();
        let before_lower = before.to_lowercase();
        if before_lower.contains("www") || before_lower.contains("http") {
            return true;
        }
    }

    // Check if there's a TLD coming up (e.g., we're at "google" in "www.google.com")
    for tld in TLDS {
        if after.to_lowercase().contains(&format!(".{}", tld)) {
            return true;
        }
    }

    false
}

/// Ensure proper spacing after punctuation marks
/// Adds a space after . ? ! , ; : when followed directly by a letter or number
/// BUT NOT for URLs (www.google.com should stay together)
fn fix_punctuation_spacing(text: &str) -> String {
    let mut result = String::with_capacity(text.len() + 50);
    let chars: Vec<char> = text.chars().collect();

    for i in 0..chars.len() {
        let c = chars[i];
        result.push(c);

        // Check if this is sentence-ending punctuation
        if matches!(c, '.' | '?' | '!' | ',' | ';' | ':') {
            // Check if there's a next character and it's alphanumeric (no space between)
            if let Some(&next) = chars.get(i + 1) {
                if next.is_alphanumeric() {
                    // Don't add space if this looks like a URL
                    if c == '.' && is_url_context(&chars, i) {
                        // Skip - this is part of a URL
                    } else {
                        result.push(' ');
                    }
                }
            }
        }
    }

    // Apply UI path quoting
    wrap_ui_paths(&result)
}

/// Wrap UI navigation paths in single quotes
/// Detects patterns like "go to Settings" → "go to 'Settings'"
pub fn wrap_ui_paths(text: &str) -> String {
    let mut result = text.to_string();

    // Navigation trigger phrases (case-insensitive matching, but we preserve original case)
    let triggers = [
        "go to ",
        "Go to ",
        "going to ",
        "Going to ",
        "navigate to ",
        "Navigate to ",
        "open ",
        "Open ",
        "click ",
        "Click ",
        "click on ",
        "Click on ",
        "select ",
        "Select ",
        "tap ",
        "Tap ",
        "tap on ",
        "Tap on ",
        "choose ",
        "Choose ",
        "find ",
        "Find ",
        "under ",
        "Under ",
        "in the ",
        "In the ",
        "into ",
        "Into ",
    ];

    // Words that indicate the end of a UI element name
    let stop_words = [
        " and ", " or ", " to ", " in ", " on ", " for ", " the ", " a ", " an ",
        " then ", " next ", " after ", " before ", " where ", " when ", " if ",
        " but ", " so ", " because ", " which ", " that ", " with ", " from ",
        " menu", " button", " option", " setting", " tab", " section", " page",
        " screen", " window", " dialog", " panel", " pane",
        ".", ",", ";", ":", "!", "?", "\n",
    ];

    for trigger in &triggers {
        let mut search_start = 0;

        while let Some(pos) = result[search_start..].find(trigger) {
            let abs_pos = search_start + pos;
            let after_trigger = abs_pos + trigger.len();

            if after_trigger >= result.len() {
                break;
            }

            // Check if the next character starts a capitalized word (likely UI element)
            let remaining = &result[after_trigger..];
            let first_char = remaining.chars().next();

            if let Some(c) = first_char {
                // Skip if already quoted
                if c == '\'' || c == '"' {
                    search_start = after_trigger;
                    continue;
                }

                // Only process if it starts with uppercase (likely a proper noun/UI element)
                if c.is_uppercase() {
                    // Find the end of the UI element name
                    let mut end_offset = 0;
                    let mut found_stop = false;

                    for stop in &stop_words {
                        if let Some(stop_pos) = remaining.to_lowercase().find(&stop.to_lowercase()) {
                            if stop_pos > 0 && (end_offset == 0 || stop_pos < end_offset) {
                                end_offset = stop_pos;
                                found_stop = true;
                            }
                        }
                    }

                    // If no stop word found, take until end of line or reasonable length
                    if !found_stop {
                        end_offset = remaining.find('\n').unwrap_or(remaining.len().min(50));
                    }

                    if end_offset > 0 {
                        let ui_element = remaining[..end_offset].trim_end();

                        // Only quote if it's not empty and looks like a UI path
                        // (contains at least one capital letter and isn't too long)
                        if !ui_element.is_empty()
                            && ui_element.len() <= 40
                            && ui_element.chars().any(|c| c.is_uppercase())
                            && !ui_element.contains('\'')
                        {
                            let quoted = format!("'{}'", ui_element);
                            let replace_start = after_trigger;
                            let replace_end = after_trigger + ui_element.len();

                            result = format!(
                                "{}{}{}",
                                &result[..replace_start],
                                quoted,
                                &result[replace_end..]
                            );

                            // Move past the quoted text
                            search_start = replace_start + quoted.len();
                            continue;
                        }
                    }
                }
            }

            search_start = after_trigger;
        }
    }

    result
}

/// Normalize ellipses to exactly 3 dots maximum
/// Collapses any sequence of 3+ dots into exactly "..."
/// Also converts the Unicode ellipsis character (…) to three dots
pub fn normalize_ellipses(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut dot_count = 0;

    for c in text.chars() {
        if c == '.' {
            dot_count += 1;
        } else if c == '…' {
            // Unicode ellipsis - treat as 3 dots
            if dot_count > 0 {
                // Already had dots, just continue the sequence
                dot_count = 3.max(dot_count);
            } else {
                dot_count = 3;
            }
        } else {
            // Non-dot character - flush any accumulated dots
            if dot_count > 0 {
                // Cap at 3 dots for ellipsis
                let dots_to_add = dot_count.min(3);
                for _ in 0..dots_to_add {
                    result.push('.');
                }
                dot_count = 0;
            }
            result.push(c);
        }
    }

    // Flush any remaining dots at end of string
    if dot_count > 0 {
        let dots_to_add = dot_count.min(3);
        for _ in 0..dots_to_add {
            result.push('.');
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_ui_paths_go_to() {
        let result = wrap_ui_paths("go to Settings");
        assert_eq!(result, "go to 'Settings'");
    }

    #[test]
    fn test_wrap_ui_paths_click() {
        let result = wrap_ui_paths("click Submit");
        assert_eq!(result, "click 'Submit'");
    }

    #[test]
    fn test_wrap_ui_paths_already_quoted() {
        let result = wrap_ui_paths("go to 'Settings'");
        assert_eq!(result, "go to 'Settings'");
    }

    #[test]
    fn test_normalize_ellipses_three_dots() {
        assert_eq!(normalize_ellipses("wait..."), "wait...");
    }

    #[test]
    fn test_normalize_ellipses_too_many() {
        assert_eq!(normalize_ellipses("wait....."), "wait...");
    }

    #[test]
    fn test_normalize_ellipses_unicode() {
        assert_eq!(normalize_ellipses("wait…"), "wait...");
    }

    #[test]
    fn test_punctuation_spacing() {
        assert_eq!(fix_punctuation_spacing("Hello.World"), "Hello. World");
    }

    #[test]
    fn test_post_process_output() {
        let result = post_process_output("  go to Settings.Click Submit  ");
        assert!(result.contains("'Settings'"));
        assert!(result.contains("'Submit'"));
    }
}
