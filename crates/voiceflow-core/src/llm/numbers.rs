//! Number normalization post-processing
//!
//! Converts spelled-out numbers to digits in contextually safe patterns.
//! The LLM understands these should be numbers but can't execute the conversion,
//! so this acts as a safety net.

/// A pending replacement: byte range in original text → replacement string
struct Replacement {
    start: usize,
    end: usize,
    text: String,
}

/// Word with its byte position in the original string
struct Word {
    text: String,
    lower: String,
    start: usize,
    end: usize,
}

fn split_words(text: &str) -> Vec<Word> {
    let mut words = Vec::new();
    let mut chars = text.char_indices().peekable();

    while let Some(&(start, c)) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        let mut end = start;
        while let Some(&(i, ch)) = chars.peek() {
            if ch.is_whitespace() {
                break;
            }
            end = i + ch.len_utf8();
            chars.next();
        }
        let t = &text[start..end];
        words.push(Word {
            text: t.to_string(),
            lower: t.to_lowercase(),
            start,
            end,
        });
    }
    words
}

/// Strip trailing punctuation from a word for matching, return (clean, punct)
fn strip_trailing_punct(w: &str) -> (&str, &str) {
    let trimmed = w.trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | ':' | '!' | '?'));
    let punct = &w[trimmed.len()..];
    (trimmed, punct)
}

//=============================================================================
// Core Number Parser
//=============================================================================

const ONES: &[(&str, u64)] = &[
    ("zero", 0),
    ("one", 1),
    ("two", 2),
    ("three", 3),
    ("four", 4),
    ("five", 5),
    ("six", 6),
    ("seven", 7),
    ("eight", 8),
    ("nine", 9),
    ("ten", 10),
    ("eleven", 11),
    ("twelve", 12),
    ("thirteen", 13),
    ("fourteen", 14),
    ("fifteen", 15),
    ("sixteen", 16),
    ("seventeen", 17),
    ("eighteen", 18),
    ("nineteen", 19),
];

const TENS: &[(&str, u64)] = &[
    ("twenty", 20),
    ("thirty", 30),
    ("forty", 40),
    ("fifty", 50),
    ("sixty", 60),
    ("seventy", 70),
    ("eighty", 80),
    ("ninety", 90),
];

const ORDINALS: &[(&str, u64)] = &[
    ("first", 1),
    ("second", 2),
    ("third", 3),
    ("fourth", 4),
    ("fifth", 5),
    ("sixth", 6),
    ("seventh", 7),
    ("eighth", 8),
    ("ninth", 9),
    ("tenth", 10),
    ("eleventh", 11),
    ("twelfth", 12),
    ("thirteenth", 13),
    ("fourteenth", 14),
    ("fifteenth", 15),
    ("sixteenth", 16),
    ("seventeenth", 17),
    ("eighteenth", 18),
    ("nineteenth", 19),
    ("twentieth", 20),
    ("thirtieth", 30),
];

fn parse_ones(w: &str) -> Option<u64> {
    ONES.iter().find(|(name, _)| *name == w).map(|(_, v)| *v)
}

fn parse_tens(w: &str) -> Option<u64> {
    TENS.iter().find(|(name, _)| *name == w).map(|(_, v)| *v)
}

fn parse_ordinal(w: &str) -> Option<u64> {
    ORDINALS.iter().find(|(name, _)| *name == w).map(|(_, v)| *v)
}

/// Check if a word is a hyphenated compound like "twenty-three"
fn parse_hyphenated(w: &str) -> Option<u64> {
    if let Some(pos) = w.find('-') {
        let left = &w[..pos];
        let right = &w[pos + 1..];
        let tens = parse_tens(left)?;
        let ones = parse_ones(right)?;
        if ones >= 1 && ones <= 9 {
            return Some(tens + ones);
        }
    }
    None
}

/// Parse a group value: ones, tens, or compound (twenty three / twenty-three)
/// Returns (value, words_consumed)
fn parse_small(words: &[Word], start: usize) -> Option<(u64, usize)> {
    if start >= words.len() {
        return None;
    }
    let (clean, _) = strip_trailing_punct(&words[start].lower);

    // Try hyphenated first: "twenty-three"
    if let Some(v) = parse_hyphenated(clean) {
        return Some((v, 1));
    }

    // Try tens
    if let Some(tens) = parse_tens(clean) {
        // Look for following ones: "twenty three"
        if start + 1 < words.len() {
            let (next_clean, _) = strip_trailing_punct(&words[start + 1].lower);
            if let Some(ones) = parse_ones(next_clean) {
                if ones >= 1 && ones <= 9 {
                    return Some((tens + ones, 2));
                }
            }
        }
        return Some((tens, 1));
    }

    // Try ones/teens
    if let Some(v) = parse_ones(clean) {
        return Some((v, 1));
    }

    None
}

/// Parse a full number: supports hundreds, thousands, millions
/// e.g., "three hundred and forty two thousand five hundred" = 342,500
/// Returns (value, words_consumed)
fn parse_number_words(words: &[Word], start: usize) -> Option<(u64, usize)> {
    if start >= words.len() {
        return None;
    }

    let mut total: u64 = 0;
    let mut current: u64 = 0;
    let mut consumed = 0;
    let mut i = start;
    let mut found_any = false;

    while i < words.len() {
        let (clean, _) = strip_trailing_punct(&words[i].lower);

        // Skip "and" between groups
        if clean == "and" && found_any && i + 1 < words.len() {
            // Peek ahead to see if the next word is a number
            let (next_clean, _) = strip_trailing_punct(&words[i + 1].lower);
            if parse_ones(next_clean).is_some()
                || parse_tens(next_clean).is_some()
                || parse_hyphenated(next_clean).is_some()
            {
                i += 1;
                consumed += 1;
                continue;
            } else {
                break;
            }
        }

        // Check for a magnitude word directly (e.g., "thousand" after "three hundred")
        let (mag_clean, _) = strip_trailing_punct(&words[i].lower);
        let magnitude_handled = if found_any && current > 0 {
            match mag_clean {
                "thousand" => {
                    total += current * 1_000;
                    current = 0;
                    consumed += 1;
                    i += 1;
                    true
                }
                "million" => {
                    total += current * 1_000_000;
                    current = 0;
                    consumed += 1;
                    i += 1;
                    true
                }
                "billion" => {
                    total += current * 1_000_000_000;
                    current = 0;
                    consumed += 1;
                    i += 1;
                    true
                }
                _ => false,
            }
        } else {
            false
        };

        if magnitude_handled {
            continue;
        }

        // Try to parse a small number at position i
        if let Some((val, wc)) = parse_small(words, i) {
            found_any = true;
            current += val;
            consumed += wc;
            i += wc;

            // Check for magnitude word after the small number
            if i < words.len() {
                let (next_mag, _) = strip_trailing_punct(&words[i].lower);
                match next_mag {
                    "hundred" => {
                        current *= 100;
                        consumed += 1;
                        i += 1;
                    }
                    "thousand" => {
                        if current == 0 {
                            current = 1;
                        }
                        total += current * 1_000;
                        current = 0;
                        consumed += 1;
                        i += 1;
                    }
                    "million" => {
                        if current == 0 {
                            current = 1;
                        }
                        total += current * 1_000_000;
                        current = 0;
                        consumed += 1;
                        i += 1;
                    }
                    "billion" => {
                        if current == 0 {
                            current = 1;
                        }
                        total += current * 1_000_000_000;
                        current = 0;
                        consumed += 1;
                        i += 1;
                    }
                    _ => {}
                }
            }
        } else {
            break;
        }
    }

    if !found_any {
        return None;
    }

    total += current;
    Some((total, consumed))
}

/// Format a number with commas: 50000 → "50,000"
fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    if s.len() <= 3 {
        return s;
    }
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

//=============================================================================
// Pattern Matchers
//=============================================================================

/// "fifty thousand dollars" → "$50,000"
fn replace_currency(words: &[Word]) -> Vec<Replacement> {
    let mut replacements = Vec::new();
    let currency_words = ["dollar", "dollars", "bucks"];

    let mut i = 0;
    while i < words.len() {
        if let Some((value, wc)) = parse_number_words(words, i) {
            let end_idx = i + wc;
            // Check if next word is a currency word
            if end_idx < words.len() {
                let (clean, punct) = strip_trailing_punct(&words[end_idx].lower);
                if currency_words.contains(&clean) {
                    // Check for "and" + cents pattern
                    let mut cents: Option<u64> = None;
                    let mut total_end = end_idx + 1;

                    if end_idx + 2 < words.len() {
                        let (and_clean, _) = strip_trailing_punct(&words[end_idx + 1].lower);
                        if and_clean == "and" {
                            if let Some((cv, cc)) = parse_number_words(words, end_idx + 2) {
                                let cents_end = end_idx + 2 + cc;
                                if cents_end < words.len() {
                                    let (cents_word, _) =
                                        strip_trailing_punct(&words[cents_end].lower);
                                    if cents_word == "cents" && cv < 100 {
                                        cents = Some(cv);
                                        total_end = cents_end + 1;
                                    }
                                }
                            }
                        }
                    }

                    let formatted = if let Some(c) = cents {
                        format!("${}.{:02}{}", format_with_commas(value), c, punct)
                    } else {
                        format!("${}{}", format_with_commas(value), punct)
                    };
                    replacements.push(Replacement {
                        start: words[i].start,
                        end: words[total_end - 1].end,
                        text: formatted,
                    });
                    i = total_end;
                    continue;
                }
            }
        }
        i += 1;
    }
    replacements
}

/// "twenty five percent" → "25%"
fn replace_percentages(words: &[Word]) -> Vec<Replacement> {
    let mut replacements = Vec::new();
    let mut i = 0;

    while i < words.len() {
        if let Some((value, wc)) = parse_number_words(words, i) {
            let end_idx = i + wc;
            if end_idx < words.len() {
                let (clean, punct) = strip_trailing_punct(&words[end_idx].lower);
                if clean == "percent" {
                    replacements.push(Replacement {
                        start: words[i].start,
                        end: words[end_idx].end,
                        text: format!("{}%{}", value, punct),
                    });
                    i = end_idx + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    replacements
}

/// "three thirty pm" → "3:30 PM", "five o'clock" → "5:00"
fn replace_times(words: &[Word]) -> Vec<Replacement> {
    let mut replacements = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let (clean_i, _) = strip_trailing_punct(&words[i].lower);

        // Parse the hour
        if let Some(hour) = parse_ones(clean_i).or_else(|| parse_tens(clean_i)) {
            if hour >= 1 && hour <= 12 {
                // Check for "three thirty pm" pattern: hour + tens-value + am/pm
                if i + 2 < words.len() {
                    let (next_clean, _) = strip_trailing_punct(&words[i + 1].lower);
                    let (marker_clean, punct) = strip_trailing_punct(&words[i + 2].lower);

                    if let Some(minutes) = parse_tens(next_clean)
                        .or_else(|| parse_ones(next_clean))
                        .or_else(|| parse_hyphenated(next_clean))
                    {
                        if matches!(marker_clean, "am" | "pm" | "a.m" | "p.m" | "a.m." | "p.m.")
                        {
                            let period = if marker_clean.starts_with('a') {
                                "AM"
                            } else {
                                "PM"
                            };
                            replacements.push(Replacement {
                                start: words[i].start,
                                end: words[i + 2].end,
                                text: format!("{}:{:02} {}{}", hour, minutes, period, punct),
                            });
                            i += 3;
                            continue;
                        }
                    }
                }

                // Check for "three pm" pattern: hour + am/pm
                if i + 1 < words.len() {
                    let (marker_clean, punct) = strip_trailing_punct(&words[i + 1].lower);
                    if matches!(marker_clean, "am" | "pm" | "a.m" | "p.m" | "a.m." | "p.m.") {
                        let period = if marker_clean.starts_with('a') {
                            "AM"
                        } else {
                            "PM"
                        };
                        replacements.push(Replacement {
                            start: words[i].start,
                            end: words[i + 1].end,
                            text: format!("{}:00 {}{}", hour, period, punct),
                        });
                        i += 2;
                        continue;
                    }

                    // "five o'clock"
                    if marker_clean == "o'clock" {
                        replacements.push(Replacement {
                            start: words[i].start,
                            end: words[i + 1].end,
                            text: format!("{}:00{}", hour, punct),
                        });
                        i += 2;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    replacements
}

/// Detects exactly 7 or 10 consecutive single-digit number words → phone number
/// "five five five one two three four five six seven" → "555-123-4567"
fn replace_phone_numbers(words: &[Word]) -> Vec<Replacement> {
    let mut replacements = Vec::new();
    let mut i = 0;

    while i < words.len() {
        // Collect consecutive single digits (0-9)
        let mut digits = Vec::new();
        let mut j = i;

        while j < words.len() {
            let (clean, _) = strip_trailing_punct(&words[j].lower);
            if let Some(d) = parse_ones(clean) {
                if d <= 9 {
                    digits.push(d);
                    j += 1;
                } else {
                    break;
                }
            } else {
                // Also handle "oh" as zero in phone context
                if clean == "oh" || clean == "o" {
                    digits.push(0);
                    j += 1;
                } else {
                    break;
                }
            }
        }

        if digits.len() == 10 {
            let phone = format!(
                "{}{}{}-{}{}{}-{}{}{}{}",
                digits[0],
                digits[1],
                digits[2],
                digits[3],
                digits[4],
                digits[5],
                digits[6],
                digits[7],
                digits[8],
                digits[9]
            );
            let (_, punct) = strip_trailing_punct(&words[j - 1].text);
            replacements.push(Replacement {
                start: words[i].start,
                end: words[j - 1].end,
                text: format!("{}{}", phone, punct),
            });
            i = j;
        } else if digits.len() == 7 {
            let phone = format!(
                "{}{}{}-{}{}{}{}",
                digits[0], digits[1], digits[2], digits[3], digits[4], digits[5], digits[6]
            );
            let (_, punct) = strip_trailing_punct(&words[j - 1].text);
            replacements.push(Replacement {
                start: words[i].start,
                end: words[j - 1].end,
                text: format!("{}{}", phone, punct),
            });
            i = j;
        } else {
            i += 1;
        }
    }
    replacements
}

/// Detects 3-6 consecutive single-digit number words → concatenated digits.
/// Catches zip codes (5 digits: "nine four one one seven" → "94117"),
/// PINs (4 digits), verification codes (6 digits), area codes (3 digits).
/// Skips sequences already handled by phone numbers (7 or 10 digits).
fn replace_digit_sequences(words: &[Word], covered: &[(usize, usize)]) -> Vec<Replacement> {
    let mut replacements = Vec::new();
    let mut i = 0;

    while i < words.len() {
        // Skip if already covered by phone numbers or other patterns
        let byte_start = words[i].start;
        if covered
            .iter()
            .any(|&(cs, ce)| byte_start >= cs && byte_start < ce)
        {
            i += 1;
            continue;
        }

        // Collect consecutive single digits (0-9)
        let mut digits = Vec::new();
        let mut j = i;

        while j < words.len() {
            let (clean, _) = strip_trailing_punct(&words[j].lower);
            if let Some(d) = parse_ones(clean) {
                if d <= 9 {
                    digits.push(d);
                    j += 1;
                } else {
                    break;
                }
            } else if clean == "oh" || clean == "o" {
                // "oh" / "o" as zero (common in codes)
                digits.push(0);
                j += 1;
            } else {
                break;
            }
        }

        // Only match 3-6 consecutive digits (not 7 or 10, which are phone numbers)
        if digits.len() >= 3 && digits.len() <= 6 {
            let code: String = digits.iter().map(|d| d.to_string()).collect();
            let (_, punct) = strip_trailing_punct(&words[j - 1].text);
            replacements.push(Replacement {
                start: words[i].start,
                end: words[j - 1].end,
                text: format!("{}{}", code, punct),
            });
            i = j;
        } else {
            i += 1;
        }
    }
    replacements
}

const MONTHS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// "january fifteenth" → "January 15"
fn replace_ordinal_dates(words: &[Word]) -> Vec<Replacement> {
    let mut replacements = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let (clean, _) = strip_trailing_punct(&words[i].lower);

        if MONTHS.contains(&clean) {
            // Check next word for ordinal
            if i + 1 < words.len() {
                let (next_clean, punct) = strip_trailing_punct(&words[i + 1].lower);
                if let Some(day) = parse_ordinal(next_clean) {
                    if day >= 1 && day <= 31 {
                        // Capitalize month
                        let month = capitalize_first(&words[i].text);
                        replacements.push(Replacement {
                            start: words[i].start,
                            end: words[i + 1].end,
                            text: format!("{} {}{}", month, day, punct),
                        });
                        i += 2;
                        continue;
                    }
                }
                // Also handle cardinal: "january fifteen" → "January 15"
                if let Some((day, wc)) = parse_number_words(words, i + 1) {
                    if day >= 1 && day <= 31 {
                        let last_idx = i + wc; // last number word index
                        let (_, punct) = strip_trailing_punct(&words[last_idx].lower);
                        let month = capitalize_first(&words[i].text);
                        replacements.push(Replacement {
                            start: words[i].start,
                            end: words[last_idx].end,
                            text: format!("{} {}{}", month, day, punct),
                        });
                        i += wc + 1;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    replacements
}

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// "port eight thousand" → "port 8000"
fn replace_keyword_numbers(words: &[Word]) -> Vec<Replacement> {
    let mut replacements = Vec::new();
    let keywords = [
        "port",
        "version",
        "error",
        "code",
        "page",
        "chapter",
        "step",
        "level",
        "grade",
        "route",
        "highway",
        "flight",
        "room",
        "apartment",
        "suite",
        "unit",
        "model",
        "episode",
        "season",
        "track",
        "item",
        "number",
    ];

    let mut i = 0;
    while i < words.len() {
        let (clean, _) = strip_trailing_punct(&words[i].lower);

        if keywords.contains(&clean) && i + 1 < words.len() {
            if let Some((value, wc)) = parse_number_words(words, i + 1) {
                let last_idx = i + wc;
                let (_, punct) = strip_trailing_punct(&words[last_idx].text);
                replacements.push(Replacement {
                    start: words[i + 1].start,
                    end: words[last_idx].end,
                    text: format!("{}{}", value, punct),
                });
                i = last_idx + 1;
                continue;
            }
        }
        i += 1;
    }
    replacements
}

/// Idioms that should NOT be converted
const IDIOMS: &[&str] = &[
    "fifty fifty",
    "twenty twenty",
    "one on one",
    "one by one",
    "two by two",
    "nine to five",
    "twenty four seven",
    "four by four",
];

/// Check if position i starts an idiom. Returns the number of words in the idiom if found.
fn idiom_length(words: &[Word], i: usize) -> Option<usize> {
    for idiom in IDIOMS {
        let idiom_words: Vec<&str> = idiom.split_whitespace().collect();
        if i + idiom_words.len() <= words.len() {
            let matches = idiom_words.iter().enumerate().all(|(j, iw)| {
                let (clean, _) = strip_trailing_punct(&words[i + j].lower);
                clean == *iw
            });
            if matches {
                return Some(idiom_words.len());
            }
        }
    }
    None
}

/// General fallback: convert spelled-out numbers >= 10 that weren't caught by specific patterns.
/// Skips values 1-9 (per style guide: spell out one through nine).
fn replace_large_numbers(words: &[Word], covered: &[(usize, usize)]) -> Vec<Replacement> {
    let mut replacements = Vec::new();
    let mut i = 0;

    while i < words.len() {
        // Skip if this position is already covered by a specific pattern
        let byte_start = words[i].start;
        if covered
            .iter()
            .any(|&(cs, ce)| byte_start >= cs && byte_start < ce)
        {
            i += 1;
            continue;
        }

        // Skip idioms
        if let Some(idiom_len) = idiom_length(words, i) {
            i += idiom_len;
            continue;
        }

        if let Some((value, wc)) = parse_number_words(words, i) {
            // Only convert values >= 10 (spell out one through nine)
            if value >= 10 && wc >= 1 {
                // Grammar guard: single "ones" word (two, three, etc.) should not be converted
                // unless it's part of a larger number (two hundred, etc.)
                if wc == 1 && value <= 9 {
                    i += 1;
                    continue;
                }

                let last_idx = i + wc - 1;

                // Skip if last word in the number has trailing punctuation that's part of a contraction
                let last_word = &words[last_idx].text;
                if last_word.contains('\'') && !last_word.ends_with('\'') {
                    i += 1;
                    continue;
                }

                let (_, punct) = strip_trailing_punct(&words[last_idx].text);
                let formatted = if value >= 1000 {
                    format_with_commas(value)
                } else {
                    value.to_string()
                };
                replacements.push(Replacement {
                    start: words[i].start,
                    end: words[last_idx].end,
                    text: format!("{}{}", formatted, punct),
                });
                i += wc;
                continue;
            }
        }
        i += 1;
    }
    replacements
}

//=============================================================================
// Orchestration
//=============================================================================

/// Main entry point: normalize spelled-out numbers in text
pub fn normalize_numbers(text: &str) -> String {
    let words = split_words(text);
    if words.is_empty() {
        return text.to_string();
    }

    // Collect all replacements from specific patterns
    let mut all_replacements = Vec::new();

    all_replacements.extend(replace_currency(&words));
    all_replacements.extend(replace_percentages(&words));
    all_replacements.extend(replace_times(&words));
    all_replacements.extend(replace_phone_numbers(&words));
    all_replacements.extend(replace_ordinal_dates(&words));
    all_replacements.extend(replace_keyword_numbers(&words));

    // Build covered ranges from specific patterns (phone, currency, times, dates, keywords)
    let covered_before_digits: Vec<(usize, usize)> = all_replacements
        .iter()
        .map(|r| (r.start, r.end))
        .collect();

    // Digit sequences (3-6 consecutive single digits) for zip codes, PINs, etc.
    all_replacements.extend(replace_digit_sequences(&words, &covered_before_digits));

    // Build full covered ranges including digit sequences
    let covered: Vec<(usize, usize)> = all_replacements
        .iter()
        .map(|r| (r.start, r.end))
        .collect();

    // Run general large-number replacement, skipping covered ranges
    all_replacements.extend(replace_large_numbers(&words, &covered));

    if all_replacements.is_empty() {
        return text.to_string();
    }

    // Sort by start position descending so we can apply from end-to-start
    all_replacements.sort_by(|a, b| b.start.cmp(&a.start));

    let mut result = text.to_string();
    for r in &all_replacements {
        result.replace_range(r.start..r.end, &r.text);
    }

    result
}

/// Fix common abbreviation patterns
/// "doctor" → "Dr." when before a capitalized name
/// "follow up" → "follow-up" before a noun (approximated by checking next word)
pub fn fix_abbreviations(text: &str) -> String {
    let words = split_words(text);
    if words.is_empty() {
        return text.to_string();
    }

    let mut replacements = Vec::new();

    for i in 0..words.len() {
        let (clean, _punct) = strip_trailing_punct(&words[i].lower);

        // "doctor" → "Dr." when followed by a capitalized word
        if clean == "doctor" && i + 1 < words.len() {
            let next = &words[i + 1].text;
            if next.chars().next().map_or(false, |c| c.is_uppercase()) {
                replacements.push(Replacement {
                    start: words[i].start,
                    end: words[i].end,
                    text: "Dr.".to_string(),
                });
            }
        }

        // "follow up" → "follow-up" when used as compound modifier/noun
        if clean == "follow" && i + 1 < words.len() {
            let (next_clean, next_punct) = strip_trailing_punct(&words[i + 1].lower);
            if next_clean == "up" {
                // Check if it's followed by a noun-like word (not end of sentence, not a verb)
                let is_compound = if i + 2 < words.len() {
                    let after = &words[i + 2].lower;
                    // Common patterns where "follow-up" is correct
                    ["appointment", "email", "message", "call", "meeting", "visit",
                     "question", "questions", "note", "notes", "action", "actions",
                     "item", "items", "task", "tasks", "on", "with", "to"]
                        .iter()
                        .any(|w| after.starts_with(w))
                } else {
                    // End of text — likely a noun ("schedule a follow-up")
                    !next_punct.is_empty() || i + 2 >= words.len()
                };

                if is_compound {
                    replacements.push(Replacement {
                        start: words[i].start,
                        end: words[i + 1].end,
                        text: format!("follow-up{}", next_punct),
                    });
                }
            }
        }
    }

    if replacements.is_empty() {
        return text.to_string();
    }

    replacements.sort_by(|a, b| b.start.cmp(&a.start));
    let mut result = text.to_string();
    for r in &replacements {
        result.replace_range(r.start..r.end, &r.text);
    }
    result
}

//=============================================================================
// Tests
//=============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Core parser ---

    #[test]
    fn test_parse_ones() {
        assert_eq!(parse_ones("five"), Some(5));
        assert_eq!(parse_ones("nineteen"), Some(19));
        assert_eq!(parse_ones("hello"), None);
    }

    #[test]
    fn test_parse_tens() {
        assert_eq!(parse_tens("twenty"), Some(20));
        assert_eq!(parse_tens("ninety"), Some(90));
    }

    #[test]
    fn test_parse_hyphenated() {
        assert_eq!(parse_hyphenated("twenty-three"), Some(23));
        assert_eq!(parse_hyphenated("fifty-nine"), Some(59));
        assert_eq!(parse_hyphenated("hello-world"), None);
    }

    #[test]
    fn test_parse_number_words_simple() {
        let words = split_words("forty two");
        assert_eq!(parse_number_words(&words, 0), Some((42, 2)));
    }

    #[test]
    fn test_parse_number_words_hundred() {
        let words = split_words("three hundred");
        assert_eq!(parse_number_words(&words, 0), Some((300, 2)));
    }

    #[test]
    fn test_parse_number_words_complex() {
        let words = split_words("three hundred and forty two");
        assert_eq!(parse_number_words(&words, 0), Some((342, 5)));
    }

    #[test]
    fn test_parse_number_words_thousand() {
        let words = split_words("fifty thousand");
        assert_eq!(parse_number_words(&words, 0), Some((50_000, 2)));
    }

    #[test]
    fn test_parse_number_words_large() {
        let words = split_words("two million three hundred thousand");
        assert_eq!(parse_number_words(&words, 0), Some((2_300_000, 5)));
    }

    // --- Currency ---

    #[test]
    fn test_currency_simple() {
        assert_eq!(
            normalize_numbers("fifty thousand dollars"),
            "$50,000"
        );
    }

    #[test]
    fn test_currency_with_cents() {
        assert_eq!(
            normalize_numbers("twenty dollars and fifty cents"),
            "$20.50"
        );
    }

    #[test]
    fn test_currency_bucks() {
        assert_eq!(normalize_numbers("ten bucks"), "$10");
    }

    // --- Percentages ---

    #[test]
    fn test_percentage() {
        assert_eq!(normalize_numbers("twenty five percent"), "25%");
    }

    #[test]
    fn test_percentage_large() {
        assert_eq!(normalize_numbers("one hundred percent"), "100%");
    }

    // --- Times ---

    #[test]
    fn test_time_with_minutes() {
        assert_eq!(normalize_numbers("three thirty pm"), "3:30 PM");
    }

    #[test]
    fn test_time_hour_only() {
        assert_eq!(normalize_numbers("five pm"), "5:00 PM");
    }

    #[test]
    fn test_time_oclock() {
        assert_eq!(normalize_numbers("five o'clock"), "5:00");
    }

    // --- Phone numbers ---

    #[test]
    fn test_phone_10_digits() {
        assert_eq!(
            normalize_numbers("five five five one two three four five six seven"),
            "555-123-4567"
        );
    }

    #[test]
    fn test_phone_7_digits() {
        assert_eq!(
            normalize_numbers("five five five one two three four"),
            "555-1234"
        );
    }

    // --- Digit sequences (zip codes, PINs, etc.) ---

    #[test]
    fn test_zip_code_5_digits() {
        assert_eq!(
            normalize_numbers("nine four one one seven"),
            "94117"
        );
    }

    #[test]
    fn test_zip_code_with_context() {
        assert_eq!(
            normalize_numbers("my zip code is nine four one one seven"),
            "my zip code is 94117"
        );
    }

    #[test]
    fn test_pin_4_digits() {
        assert_eq!(
            normalize_numbers("one two three four"),
            "1234"
        );
    }

    #[test]
    fn test_verification_code_6_digits() {
        assert_eq!(
            normalize_numbers("the code is five oh three nine two one"),
            "the code is 503921"
        );
    }

    #[test]
    fn test_area_code_3_digits() {
        assert_eq!(
            normalize_numbers("four one five"),
            "415"
        );
    }

    #[test]
    fn test_digit_sequence_with_oh() {
        assert_eq!(
            normalize_numbers("nine oh two one oh"),
            "90210"
        );
    }

    #[test]
    fn test_two_digits_not_converted() {
        // Only 2 consecutive digits — should NOT be converted
        assert_eq!(
            normalize_numbers("I have two three options"),
            "I have two three options"
        );
    }

    // --- Dates ---

    #[test]
    fn test_ordinal_date() {
        assert_eq!(normalize_numbers("january fifteenth"), "January 15");
    }

    #[test]
    fn test_cardinal_date() {
        assert_eq!(normalize_numbers("march fifteen"), "March 15");
    }

    // --- Keyword numbers ---

    #[test]
    fn test_keyword_port() {
        assert_eq!(normalize_numbers("port eight thousand"), "port 8000");
    }

    #[test]
    fn test_keyword_version() {
        assert_eq!(normalize_numbers("version twelve"), "version 12");
    }

    #[test]
    fn test_keyword_page() {
        assert_eq!(normalize_numbers("page forty two"), "page 42");
    }

    // --- General large numbers ---

    #[test]
    fn test_large_number_standalone() {
        assert_eq!(
            normalize_numbers("there were fifty thousand people"),
            "there were 50,000 people"
        );
    }

    #[test]
    fn test_large_number_hundred() {
        assert_eq!(
            normalize_numbers("about three hundred items"),
            "about 300 items"
        );
    }

    // --- Safety guards ---

    #[test]
    fn test_small_numbers_unchanged() {
        assert_eq!(
            normalize_numbers("I have two dogs"),
            "I have two dogs"
        );
    }

    #[test]
    fn test_idiom_fifty_fifty() {
        assert_eq!(
            normalize_numbers("fifty fifty chance"),
            "fifty fifty chance"
        );
    }

    #[test]
    fn test_idiom_twenty_twenty() {
        assert_eq!(
            normalize_numbers("twenty twenty vision"),
            "twenty twenty vision"
        );
    }

    #[test]
    fn test_no_false_positive_simple_text() {
        let text = "the cat sat on the mat";
        assert_eq!(normalize_numbers(text), text);
    }

    // --- Abbreviations ---

    #[test]
    fn test_doctor_abbreviation() {
        assert_eq!(fix_abbreviations("doctor Smith"), "Dr. Smith");
    }

    #[test]
    fn test_doctor_no_name() {
        assert_eq!(fix_abbreviations("see the doctor today"), "see the doctor today");
    }

    #[test]
    fn test_follow_up_noun() {
        assert_eq!(
            fix_abbreviations("schedule a follow up appointment"),
            "schedule a follow-up appointment"
        );
    }

    #[test]
    fn test_follow_up_end_of_sentence() {
        assert_eq!(
            fix_abbreviations("schedule a follow up."),
            "schedule a follow-up."
        );
    }

    // --- Format with commas ---

    #[test]
    fn test_format_commas() {
        assert_eq!(format_with_commas(1000), "1,000");
        assert_eq!(format_with_commas(50000), "50,000");
        assert_eq!(format_with_commas(1000000), "1,000,000");
        assert_eq!(format_with_commas(999), "999");
        assert_eq!(format_with_commas(42), "42");
    }

    // --- Integration: numbers in sentences ---

    #[test]
    fn test_sentence_with_currency() {
        assert_eq!(
            normalize_numbers("The total is fifty thousand dollars for the project."),
            "The total is $50,000 for the project."
        );
    }

    #[test]
    fn test_sentence_with_percentage() {
        assert_eq!(
            normalize_numbers("We achieved ninety five percent accuracy."),
            "We achieved 95% accuracy."
        );
    }

    #[test]
    fn test_mixed_safe_and_unsafe() {
        assert_eq!(
            normalize_numbers("I have two dogs and fifty thousand dollars"),
            "I have two dogs and $50,000"
        );
    }

    #[test]
    fn test_number_with_trailing_punctuation() {
        assert_eq!(
            normalize_numbers("about three hundred."),
            "about 300."
        );
    }

    #[test]
    fn test_ten_converts() {
        assert_eq!(
            normalize_numbers("there are ten items"),
            "there are 10 items"
        );
    }

    #[test]
    fn test_twelve_converts() {
        assert_eq!(
            normalize_numbers("she has twelve cats"),
            "she has 12 cats"
        );
    }
}
