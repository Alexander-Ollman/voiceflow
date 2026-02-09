//! Evaluation command - run transcription quality benchmarks against standard datasets
//!
//! Supports LibriSpeech test-clean for WER (Word Error Rate) measurement

use anyhow::{Context, Result};
use console::{style, Term};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write as IoWrite};
use std::path::{Path, PathBuf};
use voiceflow_core::config::{LlmModel, MoonshineModel, SttEngine, WhisperModel};
use voiceflow_core::{Config, Pipeline};

/// Error categories for detailed analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// STT heard a completely different word
    SttMistranscription,
    /// STT missed a word entirely
    SttDeletion,
    /// STT added a word that wasn't spoken
    SttInsertion,
    /// LLM merged words together (e.g., "face fight" → "facefight")
    LlmWordMerge,
    /// LLM added words not in the original
    LlmAddedWords,
    /// LLM removed words from the original
    LlmRemovedWords,
    /// LLM modernized archaic language (e.g., "yea" → "yes")
    LlmModernized,
    /// LLM changed word form (e.g., "awoke" → "awaked")
    LlmWordFormChange,
    /// Number not formatted (e.g., "twenty" vs "20")
    NumberNotFormatted,
    /// Abbreviation mismatch (e.g., "mister" vs "mr")
    AbbreviationMismatch,
    /// Other/uncategorized error
    Other,
}

impl ErrorCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCategory::SttMistranscription => "STT Mistranscription",
            ErrorCategory::SttDeletion => "STT Deletion",
            ErrorCategory::SttInsertion => "STT Insertion",
            ErrorCategory::LlmWordMerge => "LLM Word Merge",
            ErrorCategory::LlmAddedWords => "LLM Added Words",
            ErrorCategory::LlmRemovedWords => "LLM Removed Words",
            ErrorCategory::LlmModernized => "LLM Modernized",
            ErrorCategory::LlmWordFormChange => "LLM Word Form Change",
            ErrorCategory::NumberNotFormatted => "Number Not Formatted",
            ErrorCategory::AbbreviationMismatch => "Abbreviation Mismatch",
            ErrorCategory::Other => "Other",
        }
    }

    /// Which tier can potentially fix this error
    pub fn fixable_by(&self) -> &'static str {
        match self {
            ErrorCategory::SttMistranscription => "Unfixable (STT limit)",
            ErrorCategory::SttDeletion => "Unfixable (STT limit)",
            ErrorCategory::SttInsertion => "Unfixable (STT limit)",
            ErrorCategory::LlmWordMerge => "Prompt fix",
            ErrorCategory::LlmAddedWords => "Prompt fix",
            ErrorCategory::LlmRemovedWords => "Prompt fix",
            ErrorCategory::LlmModernized => "Prompt fix",
            ErrorCategory::LlmWordFormChange => "Prompt fix",
            ErrorCategory::NumberNotFormatted => "Tier 1 (rules)",
            ErrorCategory::AbbreviationMismatch => "Tier 1 (rules)",
            ErrorCategory::Other => "Unknown",
        }
    }
}

/// A single categorized error instance
#[derive(Debug, Clone)]
pub struct CategorizedError {
    pub category: ErrorCategory,
    pub reference_word: String,
    pub raw_word: String,
    pub llm_word: String,
    pub sample_id: String,
}

/// Error analysis results
#[derive(Debug, Default)]
pub struct ErrorAnalysis {
    pub errors_by_category: HashMap<ErrorCategory, Vec<CategorizedError>>,
    pub total_errors: usize,
}

/// LibriSpeech test-clean URL (smallest test set, ~350MB compressed)
const LIBRISPEECH_TEST_CLEAN_URL: &str =
    "https://www.openslr.org/resources/12/test-clean.tar.gz";

/// Dataset directory name
const DATASET_DIR: &str = "datasets";

/// Evaluation results for a single sample
#[derive(Debug)]
struct SampleResult {
    audio_id: String,
    reference: String,
    hypothesis: String,
    raw_transcript: String,
    wer: f32,
    transcription_ms: u64,
    llm_ms: u64,
}

/// Aggregate evaluation results
#[derive(Debug, Default)]
struct EvalResults {
    total_samples: usize,
    total_wer: f32,
    total_words: usize,
    total_errors: usize,
    // LLM-formatted WER tracking
    total_formatted_errors: usize,
    total_transcription_ms: u64,
    total_llm_ms: u64,
    samples: Vec<SampleResult>,
    // Error analysis
    error_analysis: ErrorAnalysis,
}

impl EvalResults {
    fn average_wer(&self) -> f32 {
        if self.total_words == 0 {
            0.0
        } else {
            self.total_errors as f32 / self.total_words as f32 * 100.0
        }
    }

    fn average_formatted_wer(&self) -> f32 {
        if self.total_words == 0 {
            0.0
        } else {
            self.total_formatted_errors as f32 / self.total_words as f32 * 100.0
        }
    }

    fn average_transcription_ms(&self) -> u64 {
        if self.total_samples == 0 {
            0
        } else {
            self.total_transcription_ms / self.total_samples as u64
        }
    }

    fn average_llm_ms(&self) -> u64 {
        if self.total_samples == 0 {
            0
        } else {
            self.total_llm_ms / self.total_samples as u64
        }
    }
}

/// Calculate Word Error Rate between reference and hypothesis
/// Returns (wer, word_count, error_count)
fn calculate_wer(reference: &str, hypothesis: &str) -> (f32, usize, usize) {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

    let ref_len = ref_words.len();
    let hyp_len = hyp_words.len();

    if ref_len == 0 {
        return if hyp_len == 0 {
            (0.0, 0, 0)
        } else {
            (100.0, 0, hyp_len)
        };
    }

    // Levenshtein distance with dynamic programming
    let mut dp = vec![vec![0usize; hyp_len + 1]; ref_len + 1];

    for i in 0..=ref_len {
        dp[i][0] = i;
    }
    for j in 0..=hyp_len {
        dp[0][j] = j;
    }

    for i in 1..=ref_len {
        for j in 1..=hyp_len {
            let cost = if ref_words[i - 1].to_lowercase() == hyp_words[j - 1].to_lowercase() {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1) // deletion
                .min(dp[i][j - 1] + 1) // insertion
                .min(dp[i - 1][j - 1] + cost); // substitution
        }
    }

    let errors = dp[ref_len][hyp_len];
    let wer = errors as f32 / ref_len as f32 * 100.0;

    (wer, ref_len, errors)
}

/// Normalize text for WER comparison
/// Removes punctuation, lowercases, normalizes whitespace
fn normalize_for_wer(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Common abbreviation mappings
fn get_abbreviation_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        ("mister", "mr"),
        ("missus", "mrs"),
        ("doctor", "dr"),
        ("saint", "st"),
        ("versus", "vs"),
        ("et cetera", "etc"),
        ("professor", "prof"),
        ("captain", "capt"),
        ("sergeant", "sgt"),
        ("lieutenant", "lt"),
    ]
}

/// Common archaic to modern word mappings
fn get_archaic_modern_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        ("yea", "yes"),
        ("ye", "you"),
        ("thy", "your"),
        ("thee", "you"),
        ("thou", "you"),
        ("hath", "has"),
        ("doth", "does"),
        ("wherefore", "why"),
        ("whence", "from where"),
        ("thence", "from there"),
        ("ere", "before"),
        ("twas", "it was"),
        ("tis", "it is"),
    ]
}

/// Check if two words are an abbreviation pair
fn is_abbreviation_pair(word1: &str, word2: &str) -> bool {
    let w1 = word1.to_lowercase();
    let w2 = word2.to_lowercase();

    for (full, abbrev) in get_abbreviation_pairs() {
        if (w1 == full && w2 == abbrev) || (w1 == abbrev && w2 == full) {
            return true;
        }
    }
    false
}

/// Check if a word change is archaic modernization
fn is_archaic_modernization(original: &str, changed: &str) -> bool {
    let orig = original.to_lowercase();
    let new = changed.to_lowercase();

    for (archaic, modern) in get_archaic_modern_pairs() {
        if orig == archaic && new == modern {
            return true;
        }
    }
    false
}

/// Check if the LLM merged words (e.g., "face fight" → "facefight")
fn detect_word_merge(raw_words: &[&str], llm_words: &[&str], raw_idx: usize, llm_idx: usize) -> bool {
    if llm_idx >= llm_words.len() || raw_idx + 1 >= raw_words.len() {
        return false;
    }

    let llm_word = llm_words[llm_idx].to_lowercase();
    let merged = format!("{}{}", raw_words[raw_idx].to_lowercase(), raw_words[raw_idx + 1].to_lowercase());

    llm_word == merged
}

/// Categorize errors between reference, raw STT output, and LLM output
pub fn categorize_errors(
    reference: &str,
    raw: &str,
    llm: &str,
    sample_id: &str,
) -> Vec<CategorizedError> {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let raw_words: Vec<&str> = raw.split_whitespace().collect();
    let llm_words: Vec<&str> = llm.split_whitespace().collect();

    let mut errors = Vec::new();

    // Use alignment to find specific differences
    let ref_raw_alignment = align_words(&ref_words, &raw_words);
    let raw_llm_alignment = align_words(&raw_words, &llm_words);
    let ref_llm_alignment = align_words(&ref_words, &llm_words);

    // Analyze STT errors (reference vs raw)
    for (ref_idx, raw_idx) in &ref_raw_alignment {
        match (ref_idx, raw_idx) {
            (Some(ri), Some(rai)) => {
                let ref_word = ref_words[*ri].to_lowercase();
                let raw_word = raw_words[*rai].to_lowercase();
                if ref_word != raw_word {
                    // Check if it's an abbreviation difference
                    let category = if is_abbreviation_pair(&ref_word, &raw_word) {
                        ErrorCategory::AbbreviationMismatch
                    } else {
                        ErrorCategory::SttMistranscription
                    };

                    errors.push(CategorizedError {
                        category,
                        reference_word: ref_words[*ri].to_string(),
                        raw_word: raw_words[*rai].to_string(),
                        llm_word: String::new(),
                        sample_id: sample_id.to_string(),
                    });
                }
            }
            (Some(ri), None) => {
                // Word in reference but not in raw = STT deletion
                errors.push(CategorizedError {
                    category: ErrorCategory::SttDeletion,
                    reference_word: ref_words[*ri].to_string(),
                    raw_word: String::new(),
                    llm_word: String::new(),
                    sample_id: sample_id.to_string(),
                });
            }
            (None, Some(rai)) => {
                // Word in raw but not in reference = STT insertion
                errors.push(CategorizedError {
                    category: ErrorCategory::SttInsertion,
                    reference_word: String::new(),
                    raw_word: raw_words[*rai].to_string(),
                    llm_word: String::new(),
                    sample_id: sample_id.to_string(),
                });
            }
            _ => {}
        }
    }

    // Analyze LLM errors (raw vs llm) - only where LLM changed things
    for (raw_idx, llm_idx) in &raw_llm_alignment {
        match (raw_idx, llm_idx) {
            (Some(rai), Some(li)) => {
                let raw_word = raw_words[*rai].to_lowercase();
                let llm_word = llm_words[*li].to_lowercase();
                if raw_word != llm_word {
                    // Determine what kind of LLM change this is
                    let category = if is_archaic_modernization(&raw_word, &llm_word) {
                        ErrorCategory::LlmModernized
                    } else if llm_word.contains(&raw_word) && llm_word.len() > raw_word.len() {
                        // LLM might have merged this word with next
                        ErrorCategory::LlmWordMerge
                    } else if raw_word.ends_with("ed") != llm_word.ends_with("ed")
                        || raw_word.ends_with("ing") != llm_word.ends_with("ing")
                        || raw_word.ends_with("s") != llm_word.ends_with("s")
                    {
                        ErrorCategory::LlmWordFormChange
                    } else {
                        ErrorCategory::Other
                    };

                    errors.push(CategorizedError {
                        category,
                        reference_word: String::new(),
                        raw_word: raw_words[*rai].to_string(),
                        llm_word: llm_words[*li].to_string(),
                        sample_id: sample_id.to_string(),
                    });
                }
            }
            (Some(rai), None) => {
                // Word in raw but not in LLM output = LLM removed it
                errors.push(CategorizedError {
                    category: ErrorCategory::LlmRemovedWords,
                    reference_word: String::new(),
                    raw_word: raw_words[*rai].to_string(),
                    llm_word: String::new(),
                    sample_id: sample_id.to_string(),
                });
            }
            (None, Some(li)) => {
                // Word in LLM but not in raw = LLM added it
                errors.push(CategorizedError {
                    category: ErrorCategory::LlmAddedWords,
                    reference_word: String::new(),
                    raw_word: String::new(),
                    llm_word: llm_words[*li].to_string(),
                    sample_id: sample_id.to_string(),
                });
            }
            _ => {}
        }
    }

    errors
}

/// Align two word sequences using edit distance backtracking
/// Returns pairs of (source_idx, target_idx) where None means insertion/deletion
fn align_words(source: &[&str], target: &[&str]) -> Vec<(Option<usize>, Option<usize>)> {
    let m = source.len();
    let n = target.len();

    // Build DP table
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if source[i - 1].to_lowercase() == target[j - 1].to_lowercase() {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    // Backtrack to find alignment
    let mut alignment = Vec::new();
    let mut i = m;
    let mut j = n;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let cost = if source[i - 1].to_lowercase() == target[j - 1].to_lowercase() {
                0
            } else {
                1
            };
            if dp[i][j] == dp[i - 1][j - 1] + cost {
                // Match or substitution
                alignment.push((Some(i - 1), Some(j - 1)));
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            // Deletion from source
            alignment.push((Some(i - 1), None));
            i -= 1;
        } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            // Insertion in target
            alignment.push((None, Some(j - 1)));
            j -= 1;
        } else {
            break;
        }
    }

    alignment.reverse();
    alignment
}

/// Generate a summary report of error analysis
pub fn generate_error_report(analysis: &ErrorAnalysis, total_words: usize) -> String {
    let mut report = String::new();

    report.push_str("═".repeat(60).as_str());
    report.push_str("\nVOICEFLOW ERROR ANALYSIS REPORT\n");
    report.push_str("═".repeat(60).as_str());
    report.push_str("\n\n");

    report.push_str(&format!("Total Errors Analyzed: {}\n", analysis.total_errors));
    report.push_str(&format!("Total Words: {}\n\n", total_words));

    report.push_str("ERROR DISTRIBUTION:\n");
    report.push_str("─".repeat(60).as_str());
    report.push('\n');

    // Sort categories by count
    let mut categories: Vec<_> = analysis.errors_by_category.iter().collect();
    categories.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    let mut tier1_fixable = 0;
    let mut prompt_fixable = 0;
    let mut unfixable = 0;

    for (category, errors) in &categories {
        let count = errors.len();
        let percentage = if analysis.total_errors > 0 {
            count as f32 / analysis.total_errors as f32 * 100.0
        } else {
            0.0
        };

        report.push_str(&format!(
            "  {:25} {:4} ({:5.1}%)  ← {}\n",
            category.as_str(),
            count,
            percentage,
            category.fixable_by()
        ));

        // Accumulate fixability stats
        match category.fixable_by() {
            "Tier 1 (rules)" => tier1_fixable += count,
            "Prompt fix" => prompt_fixable += count,
            s if s.contains("Unfixable") => unfixable += count,
            _ => {}
        }
    }

    report.push_str("\n");
    report.push_str("IMPROVEMENT POTENTIAL:\n");
    report.push_str("─".repeat(60).as_str());
    report.push('\n');

    let tier1_wer_impact = tier1_fixable as f32 / total_words as f32 * 100.0;
    let prompt_wer_impact = prompt_fixable as f32 / total_words as f32 * 100.0;
    let unfixable_wer = unfixable as f32 / total_words as f32 * 100.0;

    report.push_str(&format!(
        "  Fixable by Tier 1 (rules):  {:4} errors → ~{:.2}% WER improvement\n",
        tier1_fixable, tier1_wer_impact
    ));
    report.push_str(&format!(
        "  Fixable by prompt changes:  {:4} errors → ~{:.2}% WER improvement\n",
        prompt_fixable, prompt_wer_impact
    ));
    report.push_str(&format!(
        "  Unfixable (STT limit):      {:4} errors → {:.2}% floor\n",
        unfixable, unfixable_wer
    ));

    report.push_str("\n");
    report.push_str("SAMPLE ERRORS BY CATEGORY:\n");
    report.push_str("─".repeat(60).as_str());
    report.push('\n');

    // Show up to 3 examples per category
    for (category, errors) in &categories {
        if errors.is_empty() {
            continue;
        }

        report.push_str(&format!("\n{}:\n", category.as_str()));
        for error in errors.iter().take(3) {
            if !error.reference_word.is_empty() && !error.raw_word.is_empty() {
                report.push_str(&format!(
                    "  REF: '{}' → RAW: '{}'\n",
                    error.reference_word, error.raw_word
                ));
            } else if !error.raw_word.is_empty() && !error.llm_word.is_empty() {
                report.push_str(&format!(
                    "  RAW: '{}' → LLM: '{}'\n",
                    error.raw_word, error.llm_word
                ));
            } else if !error.reference_word.is_empty() {
                report.push_str(&format!("  Missing: '{}'\n", error.reference_word));
            } else if !error.llm_word.is_empty() {
                report.push_str(&format!("  Added: '{}'\n", error.llm_word));
            }
        }
        if errors.len() > 3 {
            report.push_str(&format!("  ... and {} more\n", errors.len() - 3));
        }
    }

    report
}

/// Get the cache directory for datasets
fn get_cache_dir() -> Result<PathBuf> {
    let cache_dir = directories::ProjectDirs::from("com", "voiceflow", "voiceflow")
        .map(|dirs| dirs.cache_dir().to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".cache"));

    let dataset_dir = cache_dir.join(DATASET_DIR);
    fs::create_dir_all(&dataset_dir)?;
    Ok(dataset_dir)
}

/// Download and extract LibriSpeech test-clean dataset
fn download_librispeech(term: &Term) -> Result<PathBuf> {
    let cache_dir = get_cache_dir()?;
    let extract_dir = cache_dir.join("LibriSpeech");
    let test_clean_dir = extract_dir.join("test-clean");

    // Check if already downloaded
    if test_clean_dir.exists() {
        term.write_line(&format!(
            "{} LibriSpeech test-clean already downloaded",
            style("✓").green()
        ))?;
        return Ok(test_clean_dir);
    }

    term.write_line(&format!(
        "{} Downloading LibriSpeech test-clean (~350MB)...",
        style("↓").cyan()
    ))?;

    // Download with progress
    let tar_gz_path = cache_dir.join("test-clean.tar.gz");

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()?;

    let response = client
        .get(LIBRISPEECH_TEST_CLEAN_URL)
        .send()
        .context("Failed to download LibriSpeech")?;

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"),
    );

    let mut file = File::create(&tar_gz_path)?;
    let mut downloaded = 0u64;

    let mut response = response;
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = response.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;
        pb.set_position(downloaded);
    }
    pb.finish_with_message("Download complete");

    // Extract
    term.write_line(&format!(
        "{} Extracting archive...",
        style("📦").cyan()
    ))?;

    let tar_gz = File::open(&tar_gz_path)?;
    let tar = flate2::read::GzDecoder::new(tar_gz);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(&cache_dir)?;

    // Cleanup tar.gz
    fs::remove_file(&tar_gz_path).ok();

    term.write_line(&format!(
        "{} LibriSpeech test-clean ready",
        style("✓").green()
    ))?;

    Ok(test_clean_dir)
}

/// Parse LibriSpeech transcription files
/// Format: UTTERANCE-ID TRANSCRIPTION
fn parse_transcripts(dataset_dir: &Path) -> Result<HashMap<String, String>> {
    let mut transcripts = HashMap::new();

    // Walk through all speaker/chapter directories
    for speaker_entry in fs::read_dir(dataset_dir)? {
        let speaker_dir = speaker_entry?.path();
        if !speaker_dir.is_dir() {
            continue;
        }

        for chapter_entry in fs::read_dir(&speaker_dir)? {
            let chapter_dir = chapter_entry?.path();
            if !chapter_dir.is_dir() {
                continue;
            }

            // Find .trans.txt file
            for file_entry in fs::read_dir(&chapter_dir)? {
                let file_path = file_entry?.path();
                if file_path.extension().map_or(false, |e| e == "txt")
                    && file_path
                        .file_name()
                        .map_or(false, |n| n.to_string_lossy().contains(".trans."))
                {
                    let file = File::open(&file_path)?;
                    let reader = BufReader::new(file);

                    for line in reader.lines() {
                        let line = line?;
                        if let Some((id, text)) = line.split_once(' ') {
                            transcripts.insert(id.to_string(), text.to_string());
                        }
                    }
                }
            }
        }
    }

    Ok(transcripts)
}

/// Find all FLAC audio files in the dataset
fn find_audio_files(dataset_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for speaker_entry in fs::read_dir(dataset_dir)? {
        let speaker_dir = speaker_entry?.path();
        if !speaker_dir.is_dir() {
            continue;
        }

        for chapter_entry in fs::read_dir(&speaker_dir)? {
            let chapter_dir = chapter_entry?.path();
            if !chapter_dir.is_dir() {
                continue;
            }

            for file_entry in fs::read_dir(&chapter_dir)? {
                let file_path = file_entry?.path();
                if file_path.extension().map_or(false, |e| e == "flac") {
                    files.push(file_path);
                }
            }
        }
    }

    files.sort();
    Ok(files)
}

/// Load FLAC audio file and convert to f32 samples at 16kHz
fn load_flac_audio(path: &Path) -> Result<Vec<f32>> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .context("Failed to probe FLAC file")?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .context("No audio track found")?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .context("Failed to create decoder")?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(16000);
    let mut samples = Vec::new();

    loop {
        match format.next_packet() {
            Ok(packet) => {
                let decoded = decoder.decode(&packet)?;
                let spec = *decoded.spec();
                let mut sample_buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(decoded);

                let buf_samples = sample_buf.samples();

                // Convert to mono if stereo
                if spec.channels.count() == 2 {
                    for chunk in buf_samples.chunks(2) {
                        if chunk.len() == 2 {
                            samples.push((chunk[0] + chunk[1]) / 2.0);
                        }
                    }
                } else {
                    samples.extend_from_slice(buf_samples);
                }
            }
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        }
    }

    // LibriSpeech is already 16kHz, but verify and resample if needed
    if sample_rate != 16000 {
        samples = voiceflow_core::audio::resample_to_16khz(&samples, sample_rate)?;
    }

    Ok(samples)
}

/// Parse STT model from string
fn parse_stt_model(name: &str) -> Option<(SttEngine, Option<WhisperModel>, Option<MoonshineModel>)> {
    match name.to_lowercase().replace("-", "_").as_str() {
        "moonshine_base" | "moonshine" => Some((SttEngine::Moonshine, None, Some(MoonshineModel::Base))),
        "moonshine_tiny" => Some((SttEngine::Moonshine, None, Some(MoonshineModel::Tiny))),
        "whisper_base" | "whisper" => Some((SttEngine::Whisper, Some(WhisperModel::Base), None)),
        "whisper_tiny" => Some((SttEngine::Whisper, Some(WhisperModel::Tiny), None)),
        "whisper_small" => Some((SttEngine::Whisper, Some(WhisperModel::Small), None)),
        "whisper_medium" => Some((SttEngine::Whisper, Some(WhisperModel::Medium), None)),
        "whisper_turbo" | "whisper_large_v3_turbo" | "turbo" => Some((SttEngine::Whisper, Some(WhisperModel::LargeV3Turbo), None)),
        "whisper_distil" | "distil" | "distil_large_v3" => Some((SttEngine::Whisper, Some(WhisperModel::DistilLargeV3), None)),
        "whisper_large_v3" | "large_v3" => Some((SttEngine::Whisper, Some(WhisperModel::LargeV3), None)),
        _ => None,
    }
}

/// Parse LLM model from string
fn parse_llm_model(name: &str) -> Option<LlmModel> {
    match name.to_lowercase().replace("-", "_").as_str() {
        "qwen3_1_7b" | "qwen3_1.7b" | "qwen_1.7b" => Some(LlmModel::Qwen3_1_7B),
        "qwen3_4b" | "qwen3" | "qwen" => Some(LlmModel::Qwen3_4B),
        "smollm3_3b" | "smollm3" | "smollm" => Some(LlmModel::SmolLM3_3B),
        "gemma2_2b" | "gemma2" => Some(LlmModel::Gemma2_2B),
        "gemma3n_e2b" | "gemma3n_2b" | "gemma3n" => Some(LlmModel::Gemma3nE2B),
        "gemma3n_e4b" | "gemma3n_4b" => Some(LlmModel::Gemma3nE4B),
        "phi4_mini" | "phi4" => Some(LlmModel::Phi4Mini),
        "phi2" => Some(LlmModel::Phi2),
        _ => None,
    }
}

/// Run evaluation on LibriSpeech dataset
pub async fn run(
    config: &Config,
    limit: Option<usize>,
    show_samples: bool,
    skip_llm: bool,
    analyze_errors: bool,
    report_path: Option<&str>,
    stt_override: Option<&str>,
    llm_override: Option<&str>,
) -> Result<()> {
    let term = Term::stdout();

    // Apply model overrides if specified
    let mut config = config.clone();

    if let Some(stt_name) = stt_override {
        if let Some((engine, whisper, moonshine)) = parse_stt_model(stt_name) {
            config.stt_engine = engine;
            if let Some(w) = whisper {
                config.whisper_model = w;
            }
            if let Some(m) = moonshine {
                config.moonshine_model = m;
            }
        } else {
            anyhow::bail!("Unknown STT model: {}. Valid options: moonshine-base, moonshine-tiny, whisper-base, whisper-turbo, whisper-distil", stt_name);
        }
    }

    if let Some(llm_name) = llm_override {
        if let Some(llm) = parse_llm_model(llm_name) {
            config.llm_model = llm;
        } else {
            anyhow::bail!("Unknown LLM model: {}. Valid options: qwen3-1.7b, qwen3-4b, smollm3, gemma3n, phi4", llm_name);
        }
    }

    term.write_line(&format!(
        "{} VoiceFlow Evaluation",
        style("📊").cyan()
    ))?;
    term.write_line("")?;

    // Download/locate dataset
    let dataset_dir = download_librispeech(&term)?;

    // Parse transcripts
    term.write_line("Loading transcripts...")?;
    let transcripts = parse_transcripts(&dataset_dir)?;
    term.write_line(&format!(
        "Found {} transcripts",
        style(transcripts.len()).cyan()
    ))?;

    // Find audio files
    let audio_files = find_audio_files(&dataset_dir)?;
    let total_files = audio_files.len();
    term.write_line(&format!(
        "Found {} audio files",
        style(total_files).cyan()
    ))?;

    // Limit samples if requested
    let files_to_process: Vec<_> = if let Some(limit) = limit {
        audio_files.into_iter().take(limit).collect()
    } else {
        audio_files
    };

    let process_count = files_to_process.len();
    term.write_line(&format!(
        "Processing {} samples{}",
        style(process_count).cyan(),
        if skip_llm { " (transcription only)" } else { "" }
    ))?;
    term.write_line("")?;

    // Initialize pipeline
    term.write_line("Initializing pipeline...")?;
    let mut pipeline = Pipeline::new(&config)?;
    term.write_line(&format!("{} Pipeline ready", style("✓").green()))?;
    term.write_line("")?;

    // Progress bar
    let pb = ProgressBar::new(process_count as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")?
            .progress_chars("#>-"),
    );

    let mut results = EvalResults::default();

    for audio_path in files_to_process {
        let audio_id = audio_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        pb.set_message(format!("Processing {}", &audio_id));

        // Get reference transcript
        let reference = match transcripts.get(&audio_id) {
            Some(t) => t.clone(),
            None => {
                pb.inc(1);
                continue;
            }
        };

        // Load and process audio
        let samples = match load_flac_audio(&audio_path) {
            Ok(s) => s,
            Err(e) => {
                pb.println(format!("Failed to load {}: {}", audio_id, e));
                pb.inc(1);
                continue;
            }
        };

        // Run through pipeline
        let result = if skip_llm {
            match pipeline.transcribe_only(&samples) {
                Ok(r) => r,
                Err(e) => {
                    pb.println(format!("Failed to transcribe {}: {}", audio_id, e));
                    pb.inc(1);
                    continue;
                }
            }
        } else {
            match pipeline.process(&samples, None) {
                Ok(r) => r,
                Err(e) => {
                    pb.println(format!("Failed to process {}: {}", audio_id, e));
                    pb.inc(1);
                    continue;
                }
            }
        };

        // Get hypothesis (use raw transcript for WER, formatted for display)
        let raw_transcript = result.raw_transcript.clone();

        // Debug: trace specific samples
        if raw_transcript.contains("MLike") || raw_transcript.contains("hiM") {
            eprintln!("DEBUG: Sample {} has mid-word cap", audio_id);
            eprintln!("  raw_transcript: '{}'", raw_transcript);
        }

        let hypothesis = if skip_llm {
            raw_transcript.clone()
        } else {
            result.formatted_text.clone()
        };

        // Calculate WER on normalized text
        let norm_ref = normalize_for_wer(&reference);
        let norm_raw = normalize_for_wer(&raw_transcript);
        let norm_formatted = normalize_for_wer(&hypothesis);

        let (wer, word_count, error_count) = calculate_wer(&norm_ref, &norm_raw);
        let (_, _, formatted_error_count) = calculate_wer(&norm_ref, &norm_formatted);

        // Log cases where LLM made it WORSE
        if formatted_error_count > error_count {
            eprintln!("LLM_REGRESSION [{}]: {} errors → {} errors (+{})",
                audio_id, error_count, formatted_error_count, formatted_error_count - error_count);
            eprintln!("  REF: {}", norm_ref);
            eprintln!("  RAW: {}", norm_raw);
            eprintln!("  LLM: {}", norm_formatted);
            eprintln!("");
        }

        results.total_samples += 1;
        results.total_words += word_count;
        results.total_errors += error_count;
        results.total_formatted_errors += formatted_error_count;
        results.total_wer += wer;
        results.total_transcription_ms += result.timings.transcription_ms;
        results.total_llm_ms += result.timings.llm_formatting_ms;

        // Categorize errors if analysis is enabled
        if analyze_errors {
            let errors = categorize_errors(&norm_ref, &norm_raw, &norm_formatted, &audio_id);
            for error in errors {
                results.error_analysis.total_errors += 1;
                results
                    .error_analysis
                    .errors_by_category
                    .entry(error.category)
                    .or_insert_with(Vec::new)
                    .push(error);
            }
        }

        if show_samples {
            results.samples.push(SampleResult {
                audio_id,
                reference,
                hypothesis,
                raw_transcript,
                wer,
                transcription_ms: result.timings.transcription_ms,
                llm_ms: result.timings.llm_formatting_ms,
            });
        }

        pb.inc(1);
    }

    pb.finish_and_clear();

    // Print results
    term.write_line("")?;
    term.write_line(&format!("{}", style("═".repeat(60)).dim()))?;
    term.write_line(&format!("{}", style("EVALUATION RESULTS").bold()))?;
    term.write_line(&format!("{}", style("═".repeat(60)).dim()))?;
    term.write_line("")?;

    term.write_line(&format!(
        "Samples processed:    {}",
        style(results.total_samples).cyan()
    ))?;
    term.write_line(&format!(
        "Total words:          {}",
        style(results.total_words).cyan()
    ))?;
    term.write_line(&format!(
        "Total errors:         {}",
        style(results.total_errors).yellow()
    ))?;
    term.write_line("")?;

    let avg_wer = results.average_wer();
    let avg_formatted_wer = results.average_formatted_wer();

    let wer_color = if avg_wer < 5.0 {
        style(format!("{:.2}%", avg_wer)).green()
    } else if avg_wer < 10.0 {
        style(format!("{:.2}%", avg_wer)).yellow()
    } else {
        style(format!("{:.2}%", avg_wer)).red()
    };

    let formatted_wer_color = if avg_formatted_wer < 5.0 {
        style(format!("{:.2}%", avg_formatted_wer)).green()
    } else if avg_formatted_wer < 10.0 {
        style(format!("{:.2}%", avg_formatted_wer)).yellow()
    } else {
        style(format!("{:.2}%", avg_formatted_wer)).red()
    };

    term.write_line(&format!("Raw WER:              {}", wer_color))?;
    term.write_line(&format!("LLM-Formatted WER:    {}", formatted_wer_color))?;

    // Show improvement/regression
    let delta = avg_wer - avg_formatted_wer;
    if delta.abs() > 0.01 {
        let delta_str = if delta > 0.0 {
            style(format!("↓ {:.2}% improvement", delta)).green()
        } else {
            style(format!("↑ {:.2}% regression", -delta)).red()
        };
        term.write_line(&format!("LLM Impact:           {}", delta_str))?;
    }

    term.write_line("")?;
    term.write_line(&format!(
        "Avg transcription:    {}ms",
        style(results.average_transcription_ms()).cyan()
    ))?;
    term.write_line(&format!(
        "Avg LLM formatting:   {}ms",
        style(results.average_llm_ms()).cyan()
    ))?;
    term.write_line("")?;

    // Show worst samples if requested
    if show_samples && !results.samples.is_empty() {
        term.write_line(&format!("{}", style("─".repeat(60)).dim()))?;
        term.write_line(&format!("{}", style("SAMPLE DETAILS (sorted by WER)").bold()))?;
        term.write_line(&format!("{}", style("─".repeat(60)).dim()))?;

        // Sort by WER descending
        results.samples.sort_by(|a, b| {
            b.wer.partial_cmp(&a.wer).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, sample) in results.samples.iter().take(10).enumerate() {
            term.write_line("")?;
            term.write_line(&format!(
                "{}. {} (WER: {:.1}%)",
                i + 1,
                style(&sample.audio_id).cyan(),
                sample.wer
            ))?;
            term.write_line(&format!(
                "   REF: {}",
                truncate_string(&sample.reference, 70)
            ))?;
            term.write_line(&format!(
                "   RAW: {}",
                truncate_string(&sample.raw_transcript, 70)
            ))?;
            // Show formatted output if different from raw
            if sample.hypothesis != sample.raw_transcript {
                term.write_line(&format!(
                    "   LLM: {}",
                    truncate_string(&sample.hypothesis, 70)
                ))?;
            }
        }
    }

    // Print error analysis report if enabled
    if analyze_errors && results.error_analysis.total_errors > 0 {
        term.write_line("")?;
        let report = generate_error_report(&results.error_analysis, results.total_words);
        term.write_line(&report)?;

        // Save to file if path provided
        if let Some(path) = report_path {
            let mut file = File::create(path)?;
            file.write_all(report.as_bytes())?;
            term.write_line(&format!(
                "\n{} Error report saved to: {}",
                style("📄").cyan(),
                path
            ))?;
        }
    }

    term.write_line("")?;

    Ok(())
}

fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Benchmark result for a single model combination
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub stt_name: String,
    pub llm_name: String,
    pub raw_wer: f32,
    pub formatted_wer: f32,
    pub llm_impact: f32,
    pub avg_stt_ms: u64,
    pub avg_llm_ms: u64,
    pub samples: usize,
}

/// Run comprehensive benchmark across all downloaded model combinations
pub async fn run_benchmark(samples_per_combo: usize) -> Result<()> {
    let term = Term::stdout();

    term.write_line(&format!(
        "{} VoiceFlow Model Benchmark Matrix",
        style("🔬").cyan()
    ))?;
    term.write_line("")?;

    // Get models directory and check which models are downloaded
    let models_dir = Config::models_dir()?;

    // Define STT models to test (only downloaded ones)
    let mut stt_models: Vec<(&str, SttEngine, Option<WhisperModel>, Option<MoonshineModel>)> = vec![];

    // Check Moonshine models
    let config = Config::default();
    if config.moonshine_model_downloaded_for(&MoonshineModel::Base) {
        stt_models.push(("Moonshine Base", SttEngine::Moonshine, None, Some(MoonshineModel::Base)));
    }
    if config.moonshine_model_downloaded_for(&MoonshineModel::Tiny) {
        stt_models.push(("Moonshine Tiny", SttEngine::Moonshine, None, Some(MoonshineModel::Tiny)));
    }

    // Check Whisper models (filter out corrupted downloads by size)
    for model in WhisperModel::all_models() {
        let path = models_dir.join(model.filename());
        if let Ok(metadata) = std::fs::metadata(&path) {
            // Expected size varies: tiny ~39MB, base ~147MB, small ~483MB, large ~3GB
            // Filter out corrupted downloads (less than 10MB)
            if metadata.len() > 10_000_000 {
                let name = model.display_name();
                stt_models.push((Box::leak(name.to_string().into_boxed_str()), SttEngine::Whisper, Some(model), None));
            }
        }
    }

    // Define LLM models to test (only downloaded ones, excluding known-unsupported models)
    let mut llm_models: Vec<(&str, LlmModel)> = vec![];
    for model in LlmModel::benchmark_models() {
        let path = models_dir.join(model.filename());
        // Check if file exists and is reasonably sized (not corrupted)
        if let Ok(metadata) = std::fs::metadata(&path) {
            if metadata.len() > 1_000_000 {
                // At least 1MB - not a corrupted download
                let name = model.display_name();
                llm_models.push((Box::leak(name.to_string().into_boxed_str()), model));
            }
        }
    }

    term.write_line(&format!(
        "Found {} STT models and {} LLM models",
        style(stt_models.len()).cyan(),
        style(llm_models.len()).cyan()
    ))?;
    term.write_line("")?;

    term.write_line(&format!("STT Models: {}", stt_models.iter().map(|(n, _, _, _)| *n).collect::<Vec<_>>().join(", ")))?;
    term.write_line(&format!("LLM Models: {}", llm_models.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")))?;
    term.write_line("")?;

    let total_combos = stt_models.len() * llm_models.len();
    term.write_line(&format!(
        "Running {} combinations with {} samples each",
        style(total_combos).cyan(),
        style(samples_per_combo).cyan()
    ))?;
    term.write_line("")?;

    // Download dataset first
    let dataset_dir = download_librispeech(&term)?;
    let transcripts = parse_transcripts(&dataset_dir)?;
    let audio_files: Vec<_> = find_audio_files(&dataset_dir)?
        .into_iter()
        .take(samples_per_combo)
        .collect();

    term.write_line(&format!("Using {} samples for each combination", audio_files.len()))?;
    term.write_line("")?;

    let mut results: Vec<BenchmarkResult> = vec![];

    // First, run raw STT benchmarks (no LLM)
    term.write_line(&format!("{}", style("═".repeat(70)).dim()))?;
    term.write_line(&format!("{} Phase 1: Raw STT Performance", style("📊").cyan()))?;
    term.write_line(&format!("{}", style("═".repeat(70)).dim()))?;
    term.write_line("")?;

    let mut stt_raw_wer: HashMap<String, (f32, u64)> = HashMap::new();

    for (stt_name, stt_engine, whisper_model, moonshine_model) in &stt_models {
        term.write_line(&format!("Testing {} (raw)...", style(*stt_name).yellow()))?;

        let mut config = Config::default();
        config.stt_engine = stt_engine.clone();
        if let Some(w) = whisper_model {
            config.whisper_model = w.clone();
        }
        if let Some(m) = moonshine_model {
            config.moonshine_model = m.clone();
        }

        let mut pipeline = match Pipeline::new(&config) {
            Ok(p) => p,
            Err(e) => {
                term.write_line(&format!("  {} Failed to load: {}", style("✗").red(), e))?;
                continue;
            }
        };

        let mut total_errors = 0usize;
        let mut total_words = 0usize;
        let mut total_stt_ms = 0u64;
        let mut processed = 0usize;

        for audio_path in &audio_files {
            let audio_id = audio_path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
            let reference = match transcripts.get(audio_id) {
                Some(t) => t,
                None => continue,
            };

            let samples = match load_flac_audio(audio_path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let start = std::time::Instant::now();
            let result = match pipeline.transcribe_only(&samples) {
                Ok(r) => r,
                Err(_) => continue,
            };
            total_stt_ms += start.elapsed().as_millis() as u64;

            // Normalize text for WER comparison (same as main eval)
            let norm_ref = normalize_for_wer(reference);
            let norm_raw = normalize_for_wer(&result.raw_transcript);
            let (_, word_count, error_count) = calculate_wer(&norm_ref, &norm_raw);
            total_errors += error_count;
            total_words += word_count;
            processed += 1;
        }

        if processed > 0 && total_words > 0 {
            let raw_wer = (total_errors as f32 / total_words as f32) * 100.0;
            let avg_stt_ms = total_stt_ms / processed as u64;
            stt_raw_wer.insert(stt_name.to_string(), (raw_wer, avg_stt_ms));
            term.write_line(&format!(
                "  {} Raw WER: {:.2}%  Avg latency: {}ms",
                style("✓").green(),
                raw_wer,
                avg_stt_ms
            ))?;
        }
    }

    term.write_line("")?;

    // Now run with LLM formatting
    term.write_line(&format!("{}", style("═".repeat(70)).dim()))?;
    term.write_line(&format!("{} Phase 2: STT + LLM Combinations", style("📊").cyan()))?;
    term.write_line(&format!("{}", style("═".repeat(70)).dim()))?;
    term.write_line("")?;

    let pb = ProgressBar::new(total_combos as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")?
            .progress_chars("#>-"),
    );

    for (stt_name, stt_engine, whisper_model, moonshine_model) in &stt_models {
        for (llm_name, llm_model) in &llm_models {
            pb.set_message(format!("{} + {}", stt_name, llm_name));

            let mut config = Config::default();
            config.stt_engine = stt_engine.clone();
            if let Some(w) = whisper_model {
                config.whisper_model = w.clone();
            }
            if let Some(m) = moonshine_model {
                config.moonshine_model = m.clone();
            }
            config.llm_model = llm_model.clone();

            let mut pipeline = match Pipeline::new(&config) {
                Ok(p) => p,
                Err(_) => {
                    pb.inc(1);
                    continue;
                }
            };

            let mut total_raw_errors = 0usize;
            let mut total_fmt_errors = 0usize;
            let mut total_words = 0usize;
            let mut total_stt_ms = 0u64;
            let mut total_llm_ms = 0u64;
            let mut processed = 0usize;

            for audio_path in &audio_files {
                let audio_id = audio_path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
                let reference = match transcripts.get(audio_id) {
                    Some(t) => t,
                    None => continue,
                };

                let samples = match load_flac_audio(audio_path) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                // Use full pipeline to get both raw and formatted, plus timings
                let result = match pipeline.process(&samples, None) {
                    Ok(r) => r,
                    Err(_) => continue,
                };

                total_stt_ms += result.timings.transcription_ms;
                total_llm_ms += result.timings.llm_formatting_ms;

                // Normalize text for WER comparison (same as main eval)
                let norm_ref = normalize_for_wer(reference);
                let norm_raw = normalize_for_wer(&result.raw_transcript);
                let norm_fmt = normalize_for_wer(&result.formatted_text);

                let (_, word_count, raw_err) = calculate_wer(&norm_ref, &norm_raw);
                let (_, _, fmt_err) = calculate_wer(&norm_ref, &norm_fmt);

                total_raw_errors += raw_err;
                total_fmt_errors += fmt_err;
                total_words += word_count;
                processed += 1;
            }

            if processed > 0 && total_words > 0 {
                let raw_wer = (total_raw_errors as f32 / total_words as f32) * 100.0;
                let formatted_wer = (total_fmt_errors as f32 / total_words as f32) * 100.0;
                let llm_impact = formatted_wer - raw_wer;

                results.push(BenchmarkResult {
                    stt_name: stt_name.to_string(),
                    llm_name: llm_name.to_string(),
                    raw_wer,
                    formatted_wer,
                    llm_impact,
                    avg_stt_ms: total_stt_ms / processed as u64,
                    avg_llm_ms: total_llm_ms / processed as u64,
                    samples: processed,
                });
            }

            pb.inc(1);
        }
    }

    pb.finish_and_clear();

    // Print results matrix
    term.write_line("")?;
    term.write_line(&format!("{}", style("═".repeat(70)).dim()))?;
    term.write_line(&format!("{} BENCHMARK RESULTS", style("📊").cyan().bold()))?;
    term.write_line(&format!("{}", style("═".repeat(70)).dim()))?;
    term.write_line("")?;

    // Print Raw STT results
    term.write_line(&format!("{}", style("Raw STT Performance (no LLM):").bold().underlined()))?;
    term.write_line("")?;
    term.write_line(&format!(
        "{:30} {:>10} {:>12}",
        style("Model").bold(),
        style("WER").bold(),
        style("Latency").bold()
    ))?;
    term.write_line(&format!("{}", style("─".repeat(55)).dim()))?;

    let mut raw_results: Vec<_> = stt_raw_wer.iter().collect();
    raw_results.sort_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap());

    for (name, (wer, latency)) in &raw_results {
        let wer_style = if *wer < 3.5 {
            style(format!("{:.2}%", wer)).green()
        } else if *wer < 5.0 {
            style(format!("{:.2}%", wer)).yellow()
        } else {
            style(format!("{:.2}%", wer)).red()
        };
        term.write_line(&format!(
            "{:30} {:>10} {:>10}ms",
            name,
            wer_style,
            latency
        ))?;
    }

    term.write_line("")?;

    // Print LLM impact matrix
    term.write_line(&format!("{}", style("LLM Impact Matrix (Formatted WER - Raw WER):").bold().underlined()))?;
    term.write_line("")?;

    // Build matrix
    let unique_stt: Vec<_> = stt_models.iter().map(|(n, _, _, _)| n.to_string()).collect();
    let unique_llm: Vec<_> = llm_models.iter().map(|(n, _)| n.to_string()).collect();

    // Print header
    let mut header = format!("{:20}", "");
    for llm in &unique_llm {
        header.push_str(&format!("{:>12}", truncate_string(llm, 10)));
    }
    term.write_line(&format!("{}", style(header).bold()))?;
    term.write_line(&format!("{}", style("─".repeat(20 + unique_llm.len() * 12)).dim()))?;

    // Print rows
    for stt in &unique_stt {
        let mut row = format!("{:20}", truncate_string(stt, 18));
        for llm in &unique_llm {
            let result = results.iter().find(|r| r.stt_name == *stt && r.llm_name == *llm);
            if let Some(r) = result {
                let impact_str = if r.llm_impact > 0.0 {
                    style(format!("+{:.2}%", r.llm_impact)).red()
                } else if r.llm_impact < -0.1 {
                    style(format!("{:.2}%", r.llm_impact)).green()
                } else {
                    style(format!("{:.2}%", r.llm_impact)).yellow()
                };
                row.push_str(&format!("{:>12}", impact_str));
            } else {
                row.push_str(&format!("{:>12}", "-"));
            }
        }
        term.write_line(&row)?;
    }

    term.write_line("")?;

    // Print best combinations
    term.write_line(&format!("{}", style("Best Combinations (by formatted WER):").bold().underlined()))?;
    term.write_line("")?;

    let mut sorted_results = results.clone();
    sorted_results.sort_by(|a, b| a.formatted_wer.partial_cmp(&b.formatted_wer).unwrap());

    term.write_line(&format!(
        "{:20} {:15} {:>10} {:>10} {:>10} {:>12}",
        style("STT").bold(),
        style("LLM").bold(),
        style("Raw").bold(),
        style("Formatted").bold(),
        style("Impact").bold(),
        style("Total ms").bold()
    ))?;
    term.write_line(&format!("{}", style("─".repeat(80)).dim()))?;

    for (i, r) in sorted_results.iter().take(10).enumerate() {
        let rank = if i == 0 { "🥇" } else if i == 1 { "🥈" } else if i == 2 { "🥉" } else { "  " };
        let impact_style = if r.llm_impact > 0.0 {
            style(format!("+{:.2}%", r.llm_impact)).red()
        } else {
            style(format!("{:.2}%", r.llm_impact)).green()
        };
        term.write_line(&format!(
            "{} {:18} {:15} {:>9.2}% {:>9.2}% {:>10} {:>10}ms",
            rank,
            truncate_string(&r.stt_name, 16),
            truncate_string(&r.llm_name, 13),
            r.raw_wer,
            r.formatted_wer,
            impact_style,
            r.avg_stt_ms + r.avg_llm_ms
        ))?;
    }

    term.write_line("")?;
    term.write_line(&format!(
        "{} Benchmark complete. {} combinations tested with {} samples each.",
        style("✓").green(),
        results.len(),
        samples_per_combo
    ))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wer_identical() {
        let (wer, words, errors) = calculate_wer("hello world", "hello world");
        assert_eq!(wer, 0.0);
        assert_eq!(words, 2);
        assert_eq!(errors, 0);
    }

    #[test]
    fn test_wer_one_error() {
        let (wer, words, errors) = calculate_wer("hello world", "hello word");
        assert_eq!(words, 2);
        assert_eq!(errors, 1);
        assert!((wer - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_wer_case_insensitive() {
        let (wer, _, _) = calculate_wer("Hello World", "hello world");
        assert_eq!(wer, 0.0);
    }

    #[test]
    fn test_normalize_for_wer() {
        assert_eq!(
            normalize_for_wer("Hello, World! How are you?"),
            "hello world how are you"
        );
    }

    #[test]
    fn test_categorize_stt_mistranscription() {
        let errors = categorize_errors(
            "hotel a place",
            "how tell a place",
            "how tell a place",
            "test-1"
        );

        // Should detect STT mistranscription (hotel -> how tell)
        assert!(!errors.is_empty());
        let stt_errors: Vec<_> = errors.iter()
            .filter(|e| matches!(e.category, ErrorCategory::SttMistranscription | ErrorCategory::SttInsertion))
            .collect();
        assert!(!stt_errors.is_empty());
    }

    #[test]
    fn test_categorize_llm_modernization() {
        let errors = categorize_errors(
            "yea his honourable worship",
            "yea his honourable worship",
            "yes his honourable worship",
            "test-2"
        );

        // Should detect LLM modernized (yea -> yes)
        let modernized: Vec<_> = errors.iter()
            .filter(|e| matches!(e.category, ErrorCategory::LlmModernized))
            .collect();
        assert!(!modernized.is_empty());
    }

    #[test]
    fn test_categorize_llm_removed_words() {
        let errors = categorize_errors(
            "you left him in a chair you say",
            "you left him in a chair you say",
            "you left him in a chair",
            "test-3"
        );

        // Should detect LLM removed words (you say)
        let removed: Vec<_> = errors.iter()
            .filter(|e| matches!(e.category, ErrorCategory::LlmRemovedWords))
            .collect();
        assert!(!removed.is_empty());
    }

    #[test]
    fn test_categorize_abbreviation_mismatch() {
        let errors = categorize_errors(
            "mister smith",
            "mr smith",
            "mr smith",
            "test-4"
        );

        // Should detect abbreviation mismatch (mister -> mr)
        let abbrev: Vec<_> = errors.iter()
            .filter(|e| matches!(e.category, ErrorCategory::AbbreviationMismatch))
            .collect();
        assert!(!abbrev.is_empty());
    }

    #[test]
    fn test_is_archaic_modernization() {
        assert!(is_archaic_modernization("yea", "yes"));
        assert!(is_archaic_modernization("ye", "you"));
        assert!(is_archaic_modernization("hath", "has"));
        assert!(!is_archaic_modernization("hello", "world"));
    }

    #[test]
    fn test_is_abbreviation_pair() {
        assert!(is_abbreviation_pair("mister", "mr"));
        assert!(is_abbreviation_pair("mr", "mister"));
        assert!(is_abbreviation_pair("doctor", "dr"));
        assert!(!is_abbreviation_pair("hello", "world"));
    }

    #[test]
    fn test_error_report_generation() {
        let mut analysis = ErrorAnalysis::default();
        analysis.total_errors = 10;

        analysis.errors_by_category.insert(
            ErrorCategory::SttMistranscription,
            vec![
                CategorizedError {
                    category: ErrorCategory::SttMistranscription,
                    reference_word: "hotel".to_string(),
                    raw_word: "how tell".to_string(),
                    llm_word: String::new(),
                    sample_id: "test-1".to_string(),
                }
            ]
        );

        analysis.errors_by_category.insert(
            ErrorCategory::LlmModernized,
            vec![
                CategorizedError {
                    category: ErrorCategory::LlmModernized,
                    reference_word: String::new(),
                    raw_word: "yea".to_string(),
                    llm_word: "yes".to_string(),
                    sample_id: "test-2".to_string(),
                }
            ]
        );

        let report = generate_error_report(&analysis, 1000);

        // Report should contain key sections
        assert!(report.contains("ERROR DISTRIBUTION"));
        assert!(report.contains("IMPROVEMENT POTENTIAL"));
        assert!(report.contains("STT Mistranscription"));
        assert!(report.contains("LLM Modernized"));
    }
}
