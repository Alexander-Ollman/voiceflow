//! File command - transcribe audio file

use anyhow::{Context, Result};
use console::{style, Term};
use hound::WavReader;
use std::path::Path;
use voiceflow_core::audio::resample_to_16khz;
use voiceflow_core::{Config, Pipeline};

pub async fn run(
    config: &Config,
    path: &str,
    context: Option<&str>,
    raw: bool,
) -> Result<()> {
    let term = Term::stdout();
    let file_path = Path::new(path);

    if !file_path.exists() {
        anyhow::bail!("File not found: {}", path);
    }

    term.write_line(&format!(
        "{} Loading audio file: {}",
        style("📁").cyan(),
        path
    ))?;

    // Read WAV file
    let reader = WavReader::open(file_path)
        .with_context(|| format!("Failed to open WAV file: {}", path))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    term.write_line(&format!(
        "  Sample rate: {} Hz, Channels: {}, Bits: {}",
        sample_rate, spec.channels, spec.bits_per_sample
    ))?;

    // Read samples
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(Result::ok)
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert to mono if stereo
    let samples = if spec.channels == 2 {
        voiceflow_core::audio::stereo_to_mono(&samples)
    } else {
        samples
    };

    // Resample if needed
    let samples = resample_to_16khz(&samples, sample_rate)?;

    let duration_secs = samples.len() as f32 / 16000.0;
    term.write_line(&format!(
        "  Duration: {:.1}s ({} samples at 16kHz)",
        duration_secs,
        samples.len()
    ))?;

    // Process
    term.write_line(&format!("{} Processing...", style("⚙").cyan()))?;

    let mut pipeline = Pipeline::new(config)?;

    let result = if raw {
        pipeline.transcribe_only(&samples)?
    } else {
        pipeline.process(&samples, context)?
    };

    // Output
    term.write_line("")?;
    term.write_line(&format!("{}", style("Output:").bold()))?;
    term.write_line(&result.formatted_text)?;
    term.write_line("")?;

    term.write_line(&format!(
        "{} Transcription: {}ms | LLM: {}ms | Total: {}ms",
        style("⏱").dim(),
        result.timings.transcription_ms,
        result.timings.llm_formatting_ms,
        result.timings.total_ms
    ))?;

    Ok(())
}
