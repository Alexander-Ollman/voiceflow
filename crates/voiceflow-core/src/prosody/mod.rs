//! Prosody analysis for punctuation detection
//!
//! This module provides four methods for detecting punctuation from speech:
//! 1. Voice commands - explicit spoken punctuation ("period", "comma", etc.)
//! 2. Pause analysis - detect pauses between words to infer punctuation
//! 3. Pitch contour - detect rising/falling pitch for questions vs statements
//! 4. Combined hints - pass prosodic hints to the LLM for better decisions

mod voice_commands;
mod pause_analysis;
mod pitch_analysis;
mod spelled_words;
mod replacements;
mod tokenization_fix;
mod filler_words;

pub use voice_commands::replace_voice_commands;
pub use pause_analysis::{PauseHint, analyze_pauses};
pub use pitch_analysis::{PitchContour, analyze_pitch_contour};
pub use spelled_words::{concatenate_spelled_words, concatenate_spelled_words_aggressive};
pub use replacements::ReplacementDictionary;
pub use tokenization_fix::fix_tokenization_artifacts;
pub use filler_words::remove_filler_words;

/// Combined prosody analysis result
#[derive(Debug, Clone, Default)]
pub struct ProsodyHints {
    /// Detected pauses with their suggested punctuation
    pub pause_hints: Vec<PauseHint>,
    /// Overall pitch contour (for question detection)
    pub pitch_contour: PitchContour,
    /// Confidence score (0.0 - 1.0) for the analysis
    pub confidence: f32,
}

impl ProsodyHints {
    /// Format hints as a string for LLM context
    pub fn to_llm_context(&self) -> String {
        let mut hints = Vec::new();

        // Add pitch hint
        match self.pitch_contour {
            PitchContour::Rising => {
                hints.push("The speaker's pitch rose at the end, suggesting a question.".to_string());
            }
            PitchContour::Falling => {
                hints.push("The speaker's pitch fell at the end, suggesting a statement.".to_string());
            }
            PitchContour::Emphatic => {
                hints.push("The speaker used emphatic intonation, possibly an exclamation.".to_string());
            }
            PitchContour::Neutral => {}
        }

        // Add pause hints (only significant ones)
        let long_pauses: Vec<_> = self.pause_hints.iter()
            .filter(|p| p.duration_ms > 300)
            .collect();

        if !long_pauses.is_empty() {
            hints.push(format!(
                "Detected {} significant pause(s) that may indicate sentence boundaries.",
                long_pauses.len()
            ));
        }

        if hints.is_empty() {
            String::new()
        } else {
            format!("\n[Prosody hints: {}]", hints.join(" "))
        }
    }
}

/// Analyze audio for prosodic features
///
/// # Arguments
/// * `audio` - PCM f32 samples at 16kHz
/// * `word_timestamps` - Optional word-level timestamps from Whisper
pub fn analyze_prosody(
    audio: &[f32],
    word_timestamps: Option<&[(String, i64, i64)]>,
) -> ProsodyHints {
    // Analyze pauses if we have timestamps
    let pause_hints = if let Some(timestamps) = word_timestamps {
        analyze_pauses(timestamps)
    } else {
        Vec::new()
    };

    // Analyze pitch contour
    let pitch_contour = analyze_pitch_contour(audio);

    // Calculate confidence based on audio quality
    let confidence = calculate_confidence(audio);

    ProsodyHints {
        pause_hints,
        pitch_contour,
        confidence,
    }
}

/// Calculate confidence score based on audio characteristics
fn calculate_confidence(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }

    // Calculate RMS energy
    let rms: f32 = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len() as f32).sqrt();

    // Good audio should have reasonable energy (not too quiet, not clipping)
    let energy_score = if rms < 0.01 {
        0.3 // Too quiet
    } else if rms > 0.9 {
        0.5 // Possibly clipping
    } else {
        1.0 // Good range
    };

    // Check for enough samples (at least 0.5 seconds at 16kHz)
    let length_score = if audio.len() < 8000 {
        0.5 // Short audio
    } else {
        1.0
    };

    energy_score * length_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_confidence_empty() {
        assert_eq!(calculate_confidence(&[]), 0.0);
    }

    #[test]
    fn test_calculate_confidence_quiet_audio() {
        let audio = vec![0.001f32; 16000]; // 1 second of very quiet audio
        let confidence = calculate_confidence(&audio);
        assert!(confidence < 0.5, "Quiet audio should have low confidence");
    }

    #[test]
    fn test_calculate_confidence_good_audio() {
        let audio = vec![0.3f32; 16000]; // 1 second of reasonable audio
        let confidence = calculate_confidence(&audio);
        assert_eq!(confidence, 1.0, "Good audio should have full confidence");
    }

    #[test]
    fn test_calculate_confidence_short_audio() {
        let audio = vec![0.3f32; 4000]; // 0.25 seconds
        let confidence = calculate_confidence(&audio);
        assert!(confidence < 1.0, "Short audio should have reduced confidence");
    }

    #[test]
    fn test_calculate_confidence_clipping() {
        let audio = vec![0.95f32; 16000]; // Clipping audio
        let confidence = calculate_confidence(&audio);
        assert!(confidence < 1.0, "Clipping audio should have reduced confidence");
    }

    #[test]
    fn test_prosody_hints_to_llm_context_empty() {
        let hints = ProsodyHints::default();
        assert_eq!(hints.to_llm_context(), "");
    }

    #[test]
    fn test_prosody_hints_to_llm_context_rising_pitch() {
        let hints = ProsodyHints {
            pause_hints: vec![],
            pitch_contour: PitchContour::Rising,
            confidence: 0.8,
        };
        let context = hints.to_llm_context();
        assert!(context.contains("question"));
    }

    #[test]
    fn test_prosody_hints_to_llm_context_falling_pitch() {
        let hints = ProsodyHints {
            pause_hints: vec![],
            pitch_contour: PitchContour::Falling,
            confidence: 0.8,
        };
        let context = hints.to_llm_context();
        assert!(context.contains("statement"));
    }

    #[test]
    fn test_prosody_hints_to_llm_context_emphatic() {
        let hints = ProsodyHints {
            pause_hints: vec![],
            pitch_contour: PitchContour::Emphatic,
            confidence: 0.8,
        };
        let context = hints.to_llm_context();
        assert!(context.contains("exclamation"));
    }

    #[test]
    fn test_prosody_hints_to_llm_context_with_pauses() {
        let hints = ProsodyHints {
            pause_hints: vec![
                PauseHint {
                    after_word_index: 2,
                    duration_ms: 500,
                    suggested_punctuation: pause_analysis::SuggestedPunctuation::Period,
                    word_before: "hello".to_string(),
                    word_after: Some("world".to_string()),
                },
            ],
            pitch_contour: PitchContour::Neutral,
            confidence: 0.8,
        };
        let context = hints.to_llm_context();
        assert!(context.contains("pause"));
    }

    #[test]
    fn test_analyze_prosody_no_timestamps() {
        let audio = vec![0.3f32; 16000];
        let hints = analyze_prosody(&audio, None);
        assert!(hints.pause_hints.is_empty());
    }

    #[test]
    fn test_analyze_prosody_with_timestamps() {
        let audio = vec![0.3f32; 16000];
        let timestamps = vec![
            ("Hello".to_string(), 0i64, 500i64),
            ("world".to_string(), 1000i64, 1500i64), // 500ms pause
        ];
        let hints = analyze_prosody(&audio, Some(&timestamps));
        assert!(!hints.pause_hints.is_empty());
    }
}
