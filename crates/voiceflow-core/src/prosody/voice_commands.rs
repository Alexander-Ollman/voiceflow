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

    // Apply wrapping commands FIRST so "in quotes" isn't partially consumed
    // by the " quote" → '"' simple substitution. Simple subs then handle
    // punctuation ("question mark" → "?") inside the wrapped text.
    result = apply_wrapping_commands(&result);

    // Apply simple substitutions
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

/// Case-insensitive replacement with word boundary protection.
/// Won't match if the character after the pattern is alphabetic,
/// preventing e.g. " period" from matching inside " periods".
fn replace_case_insensitive(text: &str, pattern: &str, replacement: &str) -> String {
    let lower_text = text.to_lowercase();
    let lower_pattern = pattern.to_lowercase();

    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;

    for (start, _) in lower_text.match_indices(&lower_pattern) {
        // Skip if this match overlaps with a previous replacement
        if start < last_end {
            continue;
        }
        // Word boundary check: skip if the next char after the match is alphabetic
        let end = start + pattern.len();
        if let Some(next_char) = text[end..].chars().next() {
            if next_char.is_alphabetic() {
                continue;
            }
        }
        // Add text before this match
        result.push_str(&text[last_end..start]);
        // Add replacement
        result.push_str(replacement);
        last_end = end;
    }

    // Add remaining text
    result.push_str(&text[last_end..]);

    result
}

/// Check if text starting at position looks like a URL component
/// Only matches unambiguous signals: www/http prefix or safe TLDs that
/// won't collide with English words (e.g. .computer, .com, .org).
fn is_url_context(chars: &[char], pos: usize) -> bool {
    // Safe TLDs: long or uncommon enough to never appear as English words after a period
    const SAFE_TLDS: &[&str] = &[
        "com", "org", "net", "edu", "gov", "dev", "app", "io",
        "computer", "website", "online", "tech", "cloud", "digital",
    ];

    // Check if the word at this position matches a safe TLD
    let remaining: String = chars[pos..].iter().take(30).collect();
    let word: String = remaining.chars().take_while(|c| c.is_alphabetic()).collect();
    let word_lower = word.to_lowercase();
    if SAFE_TLDS.contains(&word_lower.as_str()) {
        return true;
    }

    // Check if a safe TLD appears ahead (e.g., "era" in "demo.era.computer")
    // Normalize: strip spaces and lowercase, since ASR may produce "Era. Computer"
    let remaining_normalized: String = remaining.to_lowercase().chars().filter(|c| *c != ' ').collect();
    for tld in SAFE_TLDS {
        if remaining_normalized.contains(&format!(".{}", tld)) {
            return true;
        }
    }

    // Check if "www" or "http" appears earlier in the text (within ~50 chars)
    if pos >= 3 {
        let lookback: String = chars[pos.saturating_sub(50)..pos].iter().collect();
        let lower = lookback.to_lowercase();
        if lower.contains("www") || lower.contains("http") {
            return true;
        }
    }

    false
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
    // BUT NOT for URLs (e.g., "www.google.com" should stay together)
    let chars: Vec<char> = result.chars().collect();
    let mut final_result = String::with_capacity(result.len());

    for i in 0..chars.len() {
        final_result.push(chars[i]);

        // After sentence-ending punctuation, ensure space before next letter
        // Skip this for periods that look like they're part of a URL
        if matches!(chars[i], '.' | '?' | '!' | ':') {
            if let Some(&next) = chars.get(i + 1) {
                if next.is_alphabetic() {
                    // Don't add space if this looks like a URL
                    if chars[i] == '.' && is_url_context(&chars, i + 1) {
                        // Skip adding space - this is likely a URL
                    } else {
                        final_result.push(' ');
                    }
                }
            }
        }
    }

    // Capitalize after sentence-ending punctuation
    capitalize_after_punctuation(&final_result)
}

/// Check if text at the given position looks like the start of a URL
fn starts_with_url(chars: &[char], pos: usize) -> bool {
    let remaining: String = chars[pos..].iter().take(10).collect();
    let lower = remaining.to_lowercase();
    lower.starts_with("www.") || lower.starts_with("http://") || lower.starts_with("https://")
}

/// Capitalize the first letter after sentence-ending punctuation
/// But NOT inside URLs (don't capitalize "google" in "www.google.com")
fn capitalize_after_punctuation(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut capitalize_next = true;
    let chars: Vec<char> = text.chars().collect();

    for (i, &c) in chars.iter().enumerate() {
        if capitalize_next && c.is_alphabetic() {
            // Check if we're in a URL context
            let in_url = if i > 0 && chars[i - 1] == '.' {
                // After a dot - check URL context
                is_url_context(&chars, i)
            } else if i == 0 || (i > 0 && chars[i - 1] == ' ') {
                // Start of text or after space - check if URL starts here
                starts_with_url(&chars, i)
            } else {
                false
            };

            if in_url {
                // Don't capitalize inside URLs
                result.push(c);
            } else {
                result.push(c.to_uppercase().next().unwrap_or(c));
            }
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

/// Returns the end marker phrases for a given delimiter pair.
fn end_markers_for(open: &str, close: &str) -> &'static [&'static str] {
    match (open, close) {
        ("(", ")") => &["end parentheses", "end parenthesis", "end parens", "end brackets"],
        ("[", "]") => &["end square brackets"],
        ("\"", "\"") => &["end quotes", "end quote"],
        ("{", "}") => &["end curly braces", "end braces"],
        _ => &[],
    }
}

/// Find the rightmost clause boundary in text, returning (byte_offset, token_byte_len).
/// Scans for both punctuation characters and voice command words.
fn find_last_clause_boundary(text: &str) -> Option<(usize, usize)> {
    let punct_chars = [',', '.', '!', '?', ':', ';'];
    let voice_words: &[&str] = &[
        "comma", "period", "colon", "semicolon",
        "question mark", "exclamation mark", "exclamation point", "full stop",
    ];

    let mut best: Option<(usize, usize)> = None;

    // Check punctuation chars — find last occurrence
    for &p in &punct_chars {
        if let Some(pos) = text.rfind(p) {
            if best.is_none() || pos > best.unwrap().0 {
                best = Some((pos, p.len_utf8()));
            }
        }
    }

    // Check voice command words — find last occurrence with word boundaries
    let lower = text.to_lowercase();
    for &word in voice_words {
        let mut search_from = 0;
        let mut last_valid = None;
        while search_from < lower.len() {
            if let Some(rel_pos) = lower[search_from..].find(word) {
                let abs_pos = search_from + rel_pos;
                let end = abs_pos + word.len();
                let start_ok =
                    abs_pos == 0 || !lower.as_bytes()[abs_pos - 1].is_ascii_alphabetic();
                let end_ok =
                    end >= lower.len() || !lower.as_bytes()[end].is_ascii_alphabetic();
                if start_ok && end_ok {
                    last_valid = Some(abs_pos);
                }
                search_from = abs_pos + 1;
            } else {
                break;
            }
        }
        if let Some(pos) = last_valid {
            if best.is_none() || pos > best.unwrap().0 {
                best = Some((pos, word.len()));
            }
        }
    }

    best
}

/// Apply wrapping voice commands: "in parentheses X" → "(X)", "in brackets X" → "(X)", etc.
///
/// Wraps all text from the trigger to the end of the string (or until an explicit
/// close delimiter already present from simple substitution).
/// Handles STT-inserted punctuation after the trigger (e.g., "in parentheses, X").
fn apply_wrapping_commands(text: &str) -> String {
    // Triggers WITHOUT trailing space — we handle separator matching ourselves
    let triggers: &[(&str, &str, &str)] = &[
        ("in parentheses", "(", ")"),
        ("in parenthesis", "(", ")"),
        ("in parens", "(", ")"),
        ("in brackets", "(", ")"),
        ("in square brackets", "[", "]"),
        ("in quotes", "\"", "\""),
        ("in curly braces", "{", "}"),
    ];

    let lower = text.to_lowercase();

    // Find the first (longest) matching trigger
    let mut best: Option<(usize, usize, &str, &str)> = None; // (pos, trigger_end, open, close)
    for &(trigger, open, close) in triggers {
        if let Some(pos) = lower.find(trigger) {
            let after_trigger = pos + trigger.len();

            // Verify this isn't a partial word match (e.g., "in parenthesized")
            let next_char = lower[after_trigger..].chars().next();
            if let Some(c) = next_char {
                if c.is_alphanumeric() {
                    continue; // partial word match, skip
                }
            }

            // Skip any punctuation and whitespace between trigger and content
            // (STT may insert ", " or ": " or just " " after the trigger phrase)
            let content_start = after_trigger
                + lower[after_trigger..]
                    .chars()
                    .take_while(|c| c.is_whitespace() || matches!(c, ',' | ':' | ';' | '-'))
                    .map(|c| c.len_utf8())
                    .sum::<usize>();

            if best.is_none()
                || pos < best.unwrap().0
                || (pos == best.unwrap().0 && trigger.len() > (best.unwrap().1 - best.unwrap().0))
            {
                best = Some((pos, content_start, open, close));
            }
        }
    }

    let (pos, content_start, open, close) = match best {
        Some(b) => b,
        None => return text.to_string(),
    };

    let before = text[..pos].trim_end();
    let content = text[content_start..].trim();

    // ── Feature 1: End delimiters ──────────────────────────────────────────
    if !content.is_empty() {
        let end_markers = end_markers_for(open, close);
        let content_lower = content.to_lowercase();
        // Find the longest matching end marker with word boundaries
        let mut best_marker: Option<(usize, usize)> = None;
        for &marker in end_markers {
            if let Some(mpos) = content_lower.find(marker) {
                let mend = mpos + marker.len();
                let start_ok = mpos == 0
                    || !content_lower.as_bytes()[mpos - 1].is_ascii_alphabetic();
                let end_ok = mend >= content_lower.len()
                    || !content_lower.as_bytes()[mend].is_ascii_alphabetic();
                if start_ok && end_ok {
                    if best_marker.is_none() || marker.len() > best_marker.unwrap().1 {
                        best_marker = Some((mpos, marker.len()));
                    }
                }
            }
        }

        if let Some((marker_pos, marker_len)) = best_marker {
            let wrapped_content = content[..marker_pos].trim();
            let after_marker = marker_pos + marker_len;
            let trailing_start = after_marker
                + content[after_marker..]
                    .chars()
                    .take_while(|c| c.is_whitespace() || matches!(c, ',' | ':' | ';' | '-'))
                    .map(|c| c.len_utf8())
                    .sum::<usize>();
            let trailing = content[trailing_start..].trim();

            return match (before.is_empty(), trailing.is_empty()) {
                (true, true) => format!("{}{}{}", open, wrapped_content, close),
                (true, false) => format!("{}{}{} {}", open, wrapped_content, close, trailing),
                (false, true) => format!("{} {}{}{}", before, open, wrapped_content, close),
                (false, false) => {
                    format!("{} {}{}{} {}", before, open, wrapped_content, close, trailing)
                }
            };
        }
        // No end marker found — fall through to wrap-to-end (step 5)
    }

    // ── Feature 2: Postfix wrapping ────────────────────────────────────────
    if content.is_empty() {
        if before.is_empty() {
            return text.to_string();
        }
        return if let Some((boundary_offset, token_len)) = find_last_clause_boundary(before) {
            let split_at = boundary_offset + token_len;
            let prefix = &before[..split_at];
            let to_wrap = before[split_at..].trim();
            if to_wrap.is_empty() {
                return text.to_string();
            }
            format!("{} {}{}{}", prefix, open, to_wrap, close)
        } else {
            format!("{}{}{}", open, before, close)
        };
    }

    // ── Step 5: Existing wrap-to-end with dedup ────────────────────────────
    let content = if close == ")" && content.ends_with(')') {
        content[..content.len() - 1].trim_end()
    } else if close == "]" && content.ends_with(']') {
        content[..content.len() - 1].trim_end()
    } else if close == "\"" && content.ends_with('"') {
        content[..content.len() - 1].trim_end()
    } else {
        content
    };

    if before.is_empty() {
        format!("{}{}{}", open, content, close)
    } else {
        format!("{} {}{}{}", before, open, content, close)
    }
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

    #[test]
    fn test_url_no_spacing() {
        // URLs with www prefix should not get spaces added after dots
        assert_eq!(
            cleanup_punctuation_spacing("www.google.com"),
            "www.google.com"
        );
        assert_eq!(
            cleanup_punctuation_spacing("visit www.google.com today"),
            "Visit www.google.com today"
        );
        // Safe TLDs (.io, .com, .computer, etc.) are detected even without www
        assert_eq!(
            cleanup_punctuation_spacing("go to example.io"),
            "Go to example.io"
        );
        assert_eq!(
            cleanup_punctuation_spacing("visit demo.era.computer"),
            "Visit demo.era.computer"
        );
    }

    #[test]
    fn test_url_vs_sentence() {
        // Regular sentences should still get spacing
        assert_eq!(
            cleanup_punctuation_spacing("Hello.World"),
            "Hello. World"
        );
        // Safe TLDs (.com) are detected as URL even without www
        assert_eq!(
            cleanup_punctuation_spacing("Hello.com"),
            "Hello.com"
        );
        // Dangerous TLDs (.it, .in, .us, .me) are NOT matched — avoids false positives
        assert_eq!(
            cleanup_punctuation_spacing("Never mind.It appears"),
            "Never mind. It appears"
        );
        assert_eq!(
            cleanup_punctuation_spacing("Call me.In the morning"),
            "Call me. In the morning"
        );
    }

    #[test]
    fn test_word_boundary_period_vs_periods() {
        // "periods" should NOT be converted — word boundary protection
        let result = replace_voice_commands("If there are multiple periods then");
        assert!(result.contains("periods"), "Expected 'periods' to be preserved, got: {}", result);
        // But standalone "period" at end should still convert
        let result2 = replace_voice_commands("end of sentence period");
        assert!(result2.contains('.'), "Expected period punctuation, got: {}", result2);
    }

    #[test]
    fn test_url_lookahead_with_spaces() {
        // The is_url_context lookahead should detect .computer even with spaces
        // Note: cleanup_punctuation_spacing handles the dot-letter joining,
        // but collapsing ". Word" (with space) into ".word" is done by fix_url_spacing in prompts.rs
        // Here we just verify that dots followed directly by safe TLDs are preserved
        assert_eq!(
            cleanup_punctuation_spacing("demo.era.computer"),
            "Demo.era.computer"
        );
    }

    // ========================================================================
    // Wrapping commands — unit tests for apply_wrapping_commands
    // ========================================================================

    #[test]
    fn test_wrap_in_parentheses_basic() {
        assert_eq!(
            apply_wrapping_commands("not quite in parentheses see?"),
            "not quite (see?)"
        );
    }

    #[test]
    fn test_wrap_in_parentheses_with_comma() {
        // STT often inserts a comma after "in parentheses"
        assert_eq!(
            apply_wrapping_commands("not quite in parentheses, see?"),
            "not quite (see?)"
        );
    }

    #[test]
    fn test_wrap_in_parentheses_with_colon() {
        assert_eq!(
            apply_wrapping_commands("in parentheses: about 50%"),
            "(about 50%)"
        );
    }

    #[test]
    fn test_wrap_in_brackets() {
        assert_eq!(
            apply_wrapping_commands("hello in brackets world"),
            "hello (world)"
        );
    }

    #[test]
    fn test_wrap_in_brackets_with_comma() {
        assert_eq!(
            apply_wrapping_commands("hello in brackets, world"),
            "hello (world)"
        );
    }

    #[test]
    fn test_wrap_in_quotes() {
        assert_eq!(
            apply_wrapping_commands("he said in quotes hello world"),
            "he said \"hello world\""
        );
    }

    #[test]
    fn test_wrap_in_quotes_with_comma() {
        assert_eq!(
            apply_wrapping_commands("he said in quotes, hello world"),
            "he said \"hello world\""
        );
    }

    #[test]
    fn test_wrap_in_parens() {
        assert_eq!(
            apply_wrapping_commands("check in parens see above"),
            "check (see above)"
        );
    }

    #[test]
    fn test_wrap_in_square_brackets() {
        assert_eq!(
            apply_wrapping_commands("add in square brackets citation needed"),
            "add [citation needed]"
        );
    }

    #[test]
    fn test_wrap_in_curly_braces() {
        assert_eq!(
            apply_wrapping_commands("in curly braces name"),
            "{name}"
        );
    }

    #[test]
    fn test_wrap_at_start() {
        assert_eq!(
            apply_wrapping_commands("in parentheses see?"),
            "(see?)"
        );
    }

    #[test]
    fn test_wrap_at_start_with_comma() {
        assert_eq!(
            apply_wrapping_commands("In parentheses, are you sure?"),
            "(are you sure?)"
        );
    }

    #[test]
    fn test_wrap_no_double_close() {
        // If "close parenthesis" was already substituted to ")", don't double it
        assert_eq!(
            apply_wrapping_commands("in parentheses see?)"),
            "(see?)"
        );
    }

    #[test]
    fn test_wrap_case_insensitive() {
        assert_eq!(
            apply_wrapping_commands("In Parentheses hello"),
            "(hello)"
        );
        assert_eq!(
            apply_wrapping_commands("IN BRACKETS world"),
            "(world)"
        );
    }

    #[test]
    fn test_wrap_no_match() {
        // "parentheses" alone without "in" shouldn't trigger
        assert_eq!(
            apply_wrapping_commands("use parentheses for grouping"),
            "use parentheses for grouping"
        );
    }

    #[test]
    fn test_wrap_empty_content() {
        // Trigger with no content after → leave unchanged
        assert_eq!(
            apply_wrapping_commands("in parentheses"),
            "in parentheses"
        );
        assert_eq!(
            apply_wrapping_commands("in parentheses, "),
            "in parentheses, "
        );
    }

    #[test]
    fn test_wrap_no_partial_word_match() {
        // "in parenthesized" should NOT trigger
        assert_eq!(
            apply_wrapping_commands("in parenthesized form"),
            "in parenthesized form"
        );
    }

    // ========================================================================
    // Wrapping commands — full pipeline (voice commands + wrapping)
    // ========================================================================

    #[test]
    fn test_wrap_full_pipeline_question_mark() {
        assert_eq!(
            replace_voice_commands("not quite in parentheses see question mark"),
            "Not quite (see?)"
        );
    }

    #[test]
    fn test_wrap_full_pipeline_with_comma_from_stt() {
        // Simulates STT producing "In parentheses, are you sure question mark"
        assert_eq!(
            replace_voice_commands("In parentheses, are you sure question mark"),
            "(Are you sure?)"
        );
    }

    #[test]
    fn test_wrap_full_pipeline_period() {
        assert_eq!(
            replace_voice_commands("in brackets see above period"),
            "(See above.)"
        );
    }

    #[test]
    fn test_wrap_full_pipeline_exclamation() {
        assert_eq!(
            replace_voice_commands("in parentheses wow exclamation mark"),
            "(Wow!)"
        );
    }

    #[test]
    fn test_wrap_full_pipeline_quotes_with_comma() {
        let result = replace_voice_commands("he said in quotes, hello world");
        // cleanup_punctuation_spacing strips space before " (pre-existing behavior)
        assert!(result.contains("\"hello world\""), "got: {}", result);
    }

    // ========================================================================
    // Standard voice commands — comprehensive punctuation tests
    // ========================================================================

    #[test]
    fn test_colon() {
        assert_eq!(
            replace_voice_commands("Dear sir colon"),
            "Dear sir:"
        );
    }

    #[test]
    fn test_semicolon() {
        assert_eq!(
            replace_voice_commands("first item semicolon second item"),
            "First item; second item"
        );
    }

    #[test]
    fn test_open_close_parenthesis() {
        let result = replace_voice_commands("see open parenthesis above close parenthesis");
        assert!(result.contains("(") && result.contains(")") && result.contains("above"));
    }

    #[test]
    fn test_open_close_bracket() {
        let result = replace_voice_commands("citation open bracket 1 close bracket");
        assert!(result.contains("[") && result.contains("]") && result.contains("1"));
    }

    #[test]
    fn test_em_dash() {
        let result = replace_voice_commands("well em dash I think so");
        assert!(result.contains("—") && result.to_lowercase().contains("think"));
    }

    #[test]
    fn test_ellipsis() {
        let result = replace_voice_commands("wait ellipsis okay");
        assert!(result.contains("...") && result.to_lowercase().contains("okay"));
    }

    #[test]
    fn test_multiple_punctuation() {
        assert_eq!(
            replace_voice_commands("Hello comma world period How are you question mark"),
            "Hello, world. How are you?"
        );
    }

    #[test]
    fn test_new_paragraph() {
        let result = replace_voice_commands("End of first new paragraph Start of second");
        assert!(result.contains("\n\n"));
    }

    // ========================================================================
    // End delimiter tests — apply_wrapping_commands
    // ========================================================================

    #[test]
    fn test_end_delimiter_basic() {
        assert_eq!(
            apply_wrapping_commands("in parentheses see above end parentheses and then continue"),
            "(see above) and then continue"
        );
    }

    #[test]
    fn test_end_delimiter_with_before() {
        assert_eq!(
            apply_wrapping_commands("say in brackets hello end brackets then goodbye"),
            "say (hello) then goodbye"
        );
    }

    #[test]
    fn test_end_delimiter_no_trailing() {
        assert_eq!(
            apply_wrapping_commands("in parentheses one two three end parentheses"),
            "(one two three)"
        );
    }

    #[test]
    fn test_end_delimiter_quotes() {
        assert_eq!(
            apply_wrapping_commands("in quotes note end quotes rest"),
            "\"note\" rest"
        );
    }

    #[test]
    fn test_end_delimiter_curly_braces() {
        assert_eq!(
            apply_wrapping_commands("in curly braces name end curly braces rest"),
            "{name} rest"
        );
    }

    #[test]
    fn test_end_delimiter_no_open_trigger() {
        assert_eq!(
            apply_wrapping_commands("hello end parentheses world"),
            "hello end parentheses world"
        );
    }

    #[test]
    fn test_end_delimiter_stt_comma() {
        assert_eq!(
            apply_wrapping_commands("in parentheses see above end parentheses, and continue"),
            "(see above) and continue"
        );
    }

    #[test]
    fn test_end_delimiter_synonym() {
        assert_eq!(
            apply_wrapping_commands("in parentheses hello end parens world"),
            "(hello) world"
        );
    }

    // ========================================================================
    // Postfix wrapping tests — apply_wrapping_commands
    // ========================================================================

    #[test]
    fn test_postfix_basic() {
        assert_eq!(
            apply_wrapping_commands("see page 3 in parentheses"),
            "(see page 3)"
        );
    }

    #[test]
    fn test_postfix_brackets() {
        assert_eq!(
            apply_wrapping_commands("see page 3 in brackets"),
            "(see page 3)"
        );
    }

    #[test]
    fn test_postfix_with_voice_boundary() {
        assert_eq!(
            apply_wrapping_commands("hello comma see above in parentheses"),
            "hello comma (see above)"
        );
    }

    #[test]
    fn test_postfix_with_punct_boundary() {
        assert_eq!(
            apply_wrapping_commands("hello, see above in parentheses"),
            "hello, (see above)"
        );
    }

    #[test]
    fn test_postfix_no_boundary() {
        assert_eq!(
            apply_wrapping_commands("see above in parentheses"),
            "(see above)"
        );
    }

    #[test]
    fn test_postfix_nothing_before_or_after() {
        assert_eq!(
            apply_wrapping_commands("in parentheses"),
            "in parentheses"
        );
    }

    #[test]
    fn test_postfix_prefix_mode_unchanged() {
        assert_eq!(
            apply_wrapping_commands("in parentheses hello"),
            "(hello)"
        );
    }

    // ========================================================================
    // Full pipeline tests — end delimiter + postfix through replace_voice_commands
    // ========================================================================

    #[test]
    fn test_full_pipeline_end_delimiter_quotes() {
        assert_eq!(
            replace_voice_commands("in quotes note end quotes period"),
            "\"Note\"."
        );
    }

    #[test]
    fn test_full_pipeline_postfix() {
        assert_eq!(
            replace_voice_commands("hello comma see above in parentheses"),
            "Hello, (see above)"
        );
    }

    #[test]
    fn test_full_pipeline_end_delimiter_with_trailing() {
        assert_eq!(
            replace_voice_commands("in parentheses see above end parentheses period and continue"),
            "(See above). And continue"
        );
    }
}
