//! Prompt formatting and output post-processing utilities

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

    // Add /no_think for Qwen3 models when thinking is disabled (faster inference)
    if !config.llm_options.enable_thinking {
        prompt.push_str(" /no_think");
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
    let fixed = fix_punctuation_spacing(trimmed);
    normalize_ellipses(&fixed)
}

/// Ensure proper spacing after punctuation marks
/// Adds a space after . ? ! , ; : when followed directly by a letter or number
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
                    result.push(' ');
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
