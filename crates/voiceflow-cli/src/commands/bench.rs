//! Bench command - run performance benchmarks

use anyhow::Result;
use console::{style, Term};
use voiceflow_core::{Config, Pipeline};

pub async fn run(config: &Config, iterations: u32, file: Option<&str>) -> Result<()> {
    let term = Term::stdout();

    term.write_line(&format!(
        "{} VoiceFlow Benchmark",
        style("⚡").yellow()
    ))?;
    term.write_line("")?;

    // Generate or load test audio
    let samples = if let Some(path) = file {
        // Load from file
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();

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

        // Resample to 16kHz
        voiceflow_core::audio::resample_to_16khz(&samples, spec.sample_rate)?
    } else {
        // Generate 5 seconds of silence (for testing pipeline overhead)
        term.write_line("Using synthetic audio (5s of low noise)")?;
        let sample_rate = 16000;
        let duration = 5.0;
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.001) // Very quiet noise
            .collect();
        samples
    };

    let duration_secs = samples.len() as f32 / 16000.0;
    term.write_line(&format!(
        "Audio duration: {:.1}s ({} samples)",
        duration_secs,
        samples.len()
    ))?;
    term.write_line(&format!("Iterations: {}", iterations))?;
    term.write_line("")?;

    // Initialize pipeline (warm up)
    term.write_line("Initializing pipeline...")?;
    let mut pipeline = Pipeline::new(config)?;

    // Run benchmarks
    let mut transcription_times = Vec::new();
    let mut llm_times = Vec::new();
    let mut total_times = Vec::new();

    for i in 0..iterations {
        term.write_line(&format!("Run {}/{}...", i + 1, iterations))?;

        let result = pipeline.process(&samples, None)?;

        transcription_times.push(result.timings.transcription_ms);
        llm_times.push(result.timings.llm_formatting_ms);
        total_times.push(result.timings.total_ms);
    }

    // Calculate stats
    term.write_line("")?;
    term.write_line(&format!("{}", style("Results:").bold()))?;
    term.write_line("")?;

    let avg_transcription: u64 = transcription_times.iter().sum::<u64>() / iterations as u64;
    let avg_llm: u64 = llm_times.iter().sum::<u64>() / iterations as u64;
    let avg_total: u64 = total_times.iter().sum::<u64>() / iterations as u64;

    let min_total = *total_times.iter().min().unwrap_or(&0);
    let max_total = *total_times.iter().max().unwrap_or(&0);

    term.write_line(&format!(
        "Transcription:  avg {}ms",
        style(avg_transcription).cyan()
    ))?;
    term.write_line(&format!(
        "LLM Formatting: avg {}ms",
        style(avg_llm).cyan()
    ))?;
    term.write_line(&format!(
        "Total:          avg {}ms (min: {}, max: {})",
        style(avg_total).green(),
        min_total,
        max_total
    ))?;

    let rtf = avg_total as f32 / (duration_secs * 1000.0);
    term.write_line(&format!(
        "Real-time factor: {:.2}x",
        style(format!("{:.2}", 1.0 / rtf)).green()
    ))?;

    Ok(())
}

mod rand {
    pub fn random<T>() -> f32 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        nanos as f32 / u32::MAX as f32
    }
}
