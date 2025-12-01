//! Audio resampling utilities

use anyhow::Result;
use rubato::{FftFixedIn, Resampler};

/// Resample audio to 16kHz mono (required by Whisper)
pub fn resample_to_16khz(samples: &[f32], input_sample_rate: u32) -> Result<Vec<f32>> {
    const TARGET_RATE: u32 = 16000;

    if input_sample_rate == TARGET_RATE {
        return Ok(samples.to_vec());
    }

    tracing::debug!(
        "Resampling from {} Hz to {} Hz",
        input_sample_rate,
        TARGET_RATE
    );

    let mut resampler = FftFixedIn::<f32>::new(
        input_sample_rate as usize,
        TARGET_RATE as usize,
        1024, // chunk size
        1,    // sub chunks
        1,    // channels
    )?;

    let input_frames = resampler.input_frames_next();
    let mut output = Vec::new();

    // Process in chunks
    for chunk in samples.chunks(input_frames) {
        // Pad last chunk if needed
        let input = if chunk.len() < input_frames {
            let mut padded = chunk.to_vec();
            padded.resize(input_frames, 0.0);
            vec![padded]
        } else {
            vec![chunk.to_vec()]
        };

        let resampled = resampler.process(&input, None)?;
        if let Some(channel) = resampled.first() {
            output.extend_from_slice(channel);
        }
    }

    tracing::debug!(
        "Resampled {} samples to {} samples",
        samples.len(),
        output.len()
    );

    Ok(output)
}

/// Convert stereo to mono by averaging channels
pub fn stereo_to_mono(samples: &[f32]) -> Vec<f32> {
    samples
        .chunks(2)
        .map(|chunk| {
            if chunk.len() == 2 {
                (chunk[0] + chunk[1]) / 2.0
            } else {
                chunk[0]
            }
        })
        .collect()
}
