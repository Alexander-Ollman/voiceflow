//! Pause-based punctuation detection
//!
//! Analyzes gaps between words to infer punctuation.
//! Uses word-level timestamps from Whisper.

/// Threshold for a comma-level pause (milliseconds)
const COMMA_PAUSE_MS: i64 = 150;

/// Threshold for a period-level pause (milliseconds)
const PERIOD_PAUSE_MS: i64 = 400;

/// Threshold for a paragraph-level pause (milliseconds)
const PARAGRAPH_PAUSE_MS: i64 = 1000;

/// A hint about punctuation based on pause duration
#[derive(Debug, Clone)]
pub struct PauseHint {
    /// Word index after which the pause occurs
    pub after_word_index: usize,
    /// Duration of the pause in milliseconds
    pub duration_ms: i64,
    /// Suggested punctuation based on pause duration
    pub suggested_punctuation: SuggestedPunctuation,
    /// The word before the pause
    pub word_before: String,
    /// The word after the pause (if any)
    pub word_after: Option<String>,
}

/// Suggested punctuation based on prosodic analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestedPunctuation {
    /// No punctuation needed
    None,
    /// Comma (short pause)
    Comma,
    /// Period (longer pause)
    Period,
    /// Paragraph break (very long pause)
    ParagraphBreak,
}

impl SuggestedPunctuation {
    /// Convert to punctuation string
    pub fn to_str(&self) -> &'static str {
        match self {
            SuggestedPunctuation::None => "",
            SuggestedPunctuation::Comma => ",",
            SuggestedPunctuation::Period => ".",
            SuggestedPunctuation::ParagraphBreak => ".\n\n",
        }
    }
}

/// Analyze word timestamps to detect pauses
///
/// # Arguments
/// * `word_timestamps` - Vec of (word, start_ms, end_ms) tuples
///
/// # Returns
/// Vector of pause hints with suggested punctuation
pub fn analyze_pauses(word_timestamps: &[(String, i64, i64)]) -> Vec<PauseHint> {
    let mut hints = Vec::new();

    if word_timestamps.len() < 2 {
        return hints;
    }

    for i in 0..word_timestamps.len() - 1 {
        let (word_before, _, end_time) = &word_timestamps[i];
        let (word_after, start_time, _) = &word_timestamps[i + 1];

        // Calculate pause duration
        let pause_ms = start_time - end_time;

        // Skip if negative or zero (overlapping timestamps)
        if pause_ms <= 0 {
            continue;
        }

        // Determine suggested punctuation based on pause duration
        let suggested = if pause_ms >= PARAGRAPH_PAUSE_MS {
            SuggestedPunctuation::ParagraphBreak
        } else if pause_ms >= PERIOD_PAUSE_MS {
            SuggestedPunctuation::Period
        } else if pause_ms >= COMMA_PAUSE_MS {
            SuggestedPunctuation::Comma
        } else {
            SuggestedPunctuation::None
        };

        // Only add hint if we're suggesting punctuation
        if suggested != SuggestedPunctuation::None {
            hints.push(PauseHint {
                after_word_index: i,
                duration_ms: pause_ms,
                suggested_punctuation: suggested,
                word_before: word_before.clone(),
                word_after: Some(word_after.clone()),
            });
        }
    }

    hints
}

/// Apply pause-based punctuation hints to text
///
/// This inserts punctuation based on detected pauses if the text
/// doesn't already have punctuation at those positions.
pub fn apply_pause_hints(text: &str, hints: &[PauseHint]) -> String {
    if hints.is_empty() {
        return text.to_string();
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut result = String::with_capacity(text.len() + hints.len() * 2);

    for (i, word) in words.iter().enumerate() {
        result.push_str(word);

        // Check if there's a pause hint after this word
        if let Some(hint) = hints.iter().find(|h| h.after_word_index == i) {
            // Only add punctuation if word doesn't already end with punctuation
            let last_char = word.chars().last();
            if !last_char.map(|c| c.is_ascii_punctuation()).unwrap_or(false) {
                result.push_str(hint.suggested_punctuation.to_str());
            }
        }

        // Add space before next word (unless it's the last word or we added a paragraph break)
        if i < words.len() - 1 {
            let added_paragraph = hints
                .iter()
                .any(|h| h.after_word_index == i && h.suggested_punctuation == SuggestedPunctuation::ParagraphBreak);

            if !added_paragraph {
                result.push(' ');
            }
        }
    }

    result
}

/// Merge pause hints with existing text, respecting existing punctuation
///
/// This is smarter than `apply_pause_hints` - it doesn't override existing
/// punctuation and can upgrade weak punctuation (comma → period).
pub fn merge_pause_hints_smart(text: &str, hints: &[PauseHint]) -> String {
    if hints.is_empty() {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len() + hints.len() * 2);
    let chars: Vec<char> = text.chars().collect();

    // Create a map of word positions to pause hints
    // We'll track word boundaries as we iterate
    let mut word_count = 0;
    let mut in_word = false;

    for (_i, c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            if in_word {
                // End of word - check for pause hint
                if let Some(hint) = hints.iter().find(|h| h.after_word_index == word_count) {
                    // Check if previous char is already punctuation
                    let prev_char = result.chars().last();
                    let has_punct = prev_char.map(|p| p.is_ascii_punctuation()).unwrap_or(false);

                    if !has_punct {
                        result.push_str(hint.suggested_punctuation.to_str());
                    } else if prev_char == Some(',') && hint.suggested_punctuation == SuggestedPunctuation::Period {
                        // Upgrade comma to period if pause is long enough
                        result.pop();
                        result.push('.');
                    }
                }
                word_count += 1;
                in_word = false;
            }
            result.push(*c);
        } else {
            in_word = true;
            result.push(*c);
        }
    }

    // Handle last word
    if in_word {
        if let Some(hint) = hints.iter().find(|h| h.after_word_index == word_count) {
            let prev_char = result.chars().last();
            let has_punct = prev_char.map(|p| p.is_ascii_punctuation()).unwrap_or(false);

            if !has_punct {
                result.push_str(hint.suggested_punctuation.to_str());
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_pauses_empty() {
        let timestamps: Vec<(String, i64, i64)> = vec![];
        let hints = analyze_pauses(&timestamps);
        assert!(hints.is_empty());
    }

    #[test]
    fn test_analyze_pauses_comma() {
        let timestamps = vec![
            ("Hello".to_string(), 0, 500),
            ("world".to_string(), 700, 1200), // 200ms pause - comma
        ];
        let hints = analyze_pauses(&timestamps);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].suggested_punctuation, SuggestedPunctuation::Comma);
    }

    #[test]
    fn test_analyze_pauses_period() {
        let timestamps = vec![
            ("Hello".to_string(), 0, 500),
            ("World".to_string(), 1000, 1500), // 500ms pause - period
        ];
        let hints = analyze_pauses(&timestamps);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].suggested_punctuation, SuggestedPunctuation::Period);
    }

    #[test]
    fn test_apply_pause_hints() {
        let text = "Hello world";
        let hints = vec![PauseHint {
            after_word_index: 0,
            duration_ms: 500,
            suggested_punctuation: SuggestedPunctuation::Period,
            word_before: "Hello".to_string(),
            word_after: Some("world".to_string()),
        }];
        assert_eq!(apply_pause_hints(text, &hints), "Hello. world");
    }
}
