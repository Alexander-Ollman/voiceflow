//! Pitch contour analysis for question/exclamation detection
//!
//! Analyzes the fundamental frequency (F0) contour of speech to detect:
//! - Rising pitch → likely a question
//! - Falling pitch → likely a statement
//! - Emphatic pitch → likely an exclamation

use pitch_detection::detector::mcleod::McLeodDetector;
use pitch_detection::detector::PitchDetector;

/// Sample rate for analysis (must match input audio)
const SAMPLE_RATE: usize = 16000;

/// Window size for pitch detection (in samples)
/// ~50ms window for good frequency resolution at speech frequencies
const WINDOW_SIZE: usize = 800;

/// Hop size between windows (in samples)
/// ~25ms hop for good temporal resolution
const HOP_SIZE: usize = 400;

/// Minimum pitch to consider (Hz) - below typical speech
const MIN_PITCH_HZ: f32 = 50.0;

/// Maximum pitch to consider (Hz) - above typical speech
const MAX_PITCH_HZ: f32 = 500.0;

/// Pitch contour classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PitchContour {
    /// Pitch rises at the end (typical for questions)
    Rising,
    /// Pitch falls at the end (typical for statements)
    Falling,
    /// High energy/emphatic pitch (typical for exclamations)
    Emphatic,
    /// No clear pattern detected
    #[default]
    Neutral,
}

impl PitchContour {
    /// Suggest punctuation based on pitch contour
    pub fn suggest_punctuation(&self) -> Option<char> {
        match self {
            PitchContour::Rising => Some('?'),
            PitchContour::Emphatic => Some('!'),
            PitchContour::Falling => Some('.'),
            PitchContour::Neutral => None,
        }
    }
}

/// Detailed pitch analysis result
#[derive(Debug, Clone)]
pub struct PitchAnalysis {
    /// Classified pitch contour
    pub contour: PitchContour,
    /// Average pitch in Hz (0 if undetectable)
    pub average_pitch_hz: f32,
    /// Pitch at start of utterance (Hz)
    pub start_pitch_hz: f32,
    /// Pitch at end of utterance (Hz)
    pub end_pitch_hz: f32,
    /// Pitch change (end - start, in semitones)
    pub pitch_change_semitones: f32,
    /// Maximum energy level
    pub max_energy: f32,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

impl Default for PitchAnalysis {
    fn default() -> Self {
        Self {
            contour: PitchContour::Neutral,
            average_pitch_hz: 0.0,
            start_pitch_hz: 0.0,
            end_pitch_hz: 0.0,
            pitch_change_semitones: 0.0,
            max_energy: 0.0,
            confidence: 0.0,
        }
    }
}

/// Analyze the pitch contour of audio
///
/// # Arguments
/// * `audio` - PCM f32 samples at 16kHz
///
/// # Returns
/// Classified pitch contour (Rising, Falling, Emphatic, or Neutral)
pub fn analyze_pitch_contour(audio: &[f32]) -> PitchContour {
    analyze_pitch_detailed(audio).contour
}

/// Perform detailed pitch analysis
///
/// # Arguments
/// * `audio` - PCM f32 samples at 16kHz
///
/// # Returns
/// Detailed pitch analysis including contour, frequencies, and confidence
pub fn analyze_pitch_detailed(audio: &[f32]) -> PitchAnalysis {
    if audio.len() < WINDOW_SIZE * 2 {
        return PitchAnalysis::default();
    }

    // Extract pitch values using McLeod pitch detection
    let pitches = extract_pitches(audio);

    if pitches.is_empty() {
        return PitchAnalysis::default();
    }

    // Filter out outliers and invalid pitches
    let valid_pitches: Vec<f32> = pitches
        .iter()
        .filter(|&&p| p >= MIN_PITCH_HZ && p <= MAX_PITCH_HZ)
        .copied()
        .collect();

    if valid_pitches.len() < 3 {
        return PitchAnalysis::default();
    }

    // Calculate statistics
    let average_pitch = valid_pitches.iter().sum::<f32>() / valid_pitches.len() as f32;

    // Get pitch at start (first 20% of valid pitches)
    let start_count = (valid_pitches.len() / 5).max(1);
    let start_pitch: f32 = valid_pitches[..start_count].iter().sum::<f32>() / start_count as f32;

    // Get pitch at end (last 20% of valid pitches)
    let end_start = valid_pitches.len() - start_count;
    let end_pitch: f32 = valid_pitches[end_start..].iter().sum::<f32>() / start_count as f32;

    // Calculate pitch change in semitones
    let pitch_change_semitones = if start_pitch > 0.0 && end_pitch > 0.0 {
        12.0 * (end_pitch / start_pitch).log2()
    } else {
        0.0
    };

    // Calculate max energy
    let max_energy = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

    // Classify contour based on pitch change
    let contour = classify_contour(pitch_change_semitones, max_energy, &valid_pitches);

    // Calculate confidence based on consistency of pitch values
    let confidence = calculate_pitch_confidence(&valid_pitches, pitches.len());

    PitchAnalysis {
        contour,
        average_pitch_hz: average_pitch,
        start_pitch_hz: start_pitch,
        end_pitch_hz: end_pitch,
        pitch_change_semitones,
        max_energy,
        confidence,
    }
}

/// Extract pitch values from audio using McLeod algorithm
fn extract_pitches(audio: &[f32]) -> Vec<f32> {
    let mut detector = McLeodDetector::new(WINDOW_SIZE, WINDOW_SIZE / 2);
    let mut pitches = Vec::new();

    // Process audio in overlapping windows
    let num_windows = (audio.len() - WINDOW_SIZE) / HOP_SIZE + 1;

    for i in 0..num_windows {
        let start = i * HOP_SIZE;
        let end = start + WINDOW_SIZE;

        if end > audio.len() {
            break;
        }

        let window = &audio[start..end];

        // Check if window has sufficient energy
        let energy: f32 = window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32;
        if energy < 0.0001 {
            continue; // Skip silent windows
        }

        // Detect pitch
        if let Some(pitch) = detector.get_pitch(window, SAMPLE_RATE, 0.1, 0.1) {
            if pitch.frequency >= MIN_PITCH_HZ && pitch.frequency <= MAX_PITCH_HZ {
                pitches.push(pitch.frequency as f32);
            }
        }
    }

    pitches
}

/// Classify pitch contour based on pitch change and energy
fn classify_contour(pitch_change_semitones: f32, max_energy: f32, pitches: &[f32]) -> PitchContour {
    // Threshold for significant pitch change (in semitones)
    // A semitone is ~6% change in frequency
    const RISING_THRESHOLD: f32 = 2.0;   // ~12% rise
    const FALLING_THRESHOLD: f32 = -2.0; // ~12% fall

    // Threshold for emphatic speech (high energy)
    const EMPHATIC_ENERGY_THRESHOLD: f32 = 0.5;

    // Check for emphatic speech (high energy + variable pitch)
    if max_energy > EMPHATIC_ENERGY_THRESHOLD {
        // Calculate pitch variance
        if pitches.len() > 1 {
            let mean = pitches.iter().sum::<f32>() / pitches.len() as f32;
            let variance = pitches.iter()
                .map(|p| (p - mean).powi(2))
                .sum::<f32>() / pitches.len() as f32;
            let std_dev = variance.sqrt();

            // High variance + high energy = emphatic
            if std_dev > 30.0 {
                return PitchContour::Emphatic;
            }
        }
    }

    // Check for rising/falling based on pitch change
    if pitch_change_semitones > RISING_THRESHOLD {
        PitchContour::Rising
    } else if pitch_change_semitones < FALLING_THRESHOLD {
        PitchContour::Falling
    } else {
        PitchContour::Neutral
    }
}

/// Calculate confidence in pitch detection
fn calculate_pitch_confidence(valid_pitches: &[f32], total_windows: usize) -> f32 {
    if total_windows == 0 {
        return 0.0;
    }

    // Confidence based on:
    // 1. How many windows had detectable pitch
    let detection_rate = valid_pitches.len() as f32 / total_windows as f32;

    // 2. Consistency of detected pitches (lower variance = higher confidence)
    let consistency = if valid_pitches.len() > 1 {
        let mean = valid_pitches.iter().sum::<f32>() / valid_pitches.len() as f32;
        let variance = valid_pitches.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f32>() / valid_pitches.len() as f32;
        let coefficient_of_variation = variance.sqrt() / mean;

        // Lower CV = higher confidence (max out at CV of 0.5)
        (1.0 - coefficient_of_variation.min(0.5) * 2.0).max(0.0)
    } else {
        0.5
    };

    // Combine factors
    (detection_rate * 0.5 + consistency * 0.5).min(1.0)
}

/// Analyze pitch for the final portion of audio (for question detection)
///
/// Questions typically have rising pitch in the last 200-500ms
pub fn analyze_final_pitch(audio: &[f32]) -> PitchContour {
    // Analyze last 500ms (8000 samples at 16kHz)
    let final_samples = 8000.min(audio.len());
    let start = audio.len().saturating_sub(final_samples);

    analyze_pitch_contour(&audio[start..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_audio() {
        let audio: Vec<f32> = vec![];
        assert_eq!(analyze_pitch_contour(&audio), PitchContour::Neutral);
    }

    #[test]
    fn test_silent_audio() {
        let audio = vec![0.0f32; WINDOW_SIZE * 4];
        assert_eq!(analyze_pitch_contour(&audio), PitchContour::Neutral);
    }

    #[test]
    fn test_pitch_analysis_structure() {
        // Generate a simple sine wave (440 Hz = A4)
        let freq = 440.0;
        let duration_samples = SAMPLE_RATE * 2; // 2 seconds
        let audio: Vec<f32> = (0..duration_samples)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
            })
            .collect();

        let analysis = analyze_pitch_detailed(&audio);
        assert!(analysis.average_pitch_hz > 0.0);
    }
}
