//! Record command - capture audio from microphone

use anyhow::Result;
use arboard::Clipboard;
use console::{style, Term};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use voiceflow_core::audio::AudioCapture;
use voiceflow_core::{Config, Pipeline};

pub async fn run(
    config: &Config,
    clipboard: bool,
    context: Option<&str>,
    raw: bool,
) -> Result<()> {
    let term = Term::stdout();

    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    // Start audio capture
    let mut capture = AudioCapture::new()?;

    term.write_line(&format!(
        "{} Recording... (press {} to stop)",
        style("🎙").green(),
        style("Ctrl+C").cyan()
    ))?;

    capture.start()?;

    // Show recording indicator
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
            .template("{spinner:.green} {msg}")?,
    );
    pb.set_message("Recording...");

    // Wait for Ctrl+C
    while running.load(Ordering::SeqCst) {
        pb.tick();
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    pb.finish_and_clear();

    // Stop capture and get audio
    let samples = capture.stop()?;
    let duration_secs = samples.len() as f32 / capture.sample_rate() as f32;

    term.write_line(&format!(
        "{} Captured {:.1}s of audio ({} samples)",
        style("✓").green(),
        duration_secs,
        samples.len()
    ))?;

    if samples.is_empty() {
        term.write_line(&format!("{} No audio captured", style("⚠").yellow()))?;
        return Ok(());
    }

    // Process audio
    term.write_line(&format!("{} Processing...", style("⚙").cyan()))?;

    let mut pipeline = Pipeline::new(config)?;

    let result = if raw {
        pipeline.transcribe_only(&samples)?
    } else {
        pipeline.process(&samples, context)?
    };

    // Output result
    term.write_line("")?;

    if raw {
        term.write_line(&format!("{}", style("Raw transcript:").bold()))?;
    } else {
        term.write_line(&format!("{}", style("Formatted text:").bold()))?;
    }

    term.write_line(&result.formatted_text)?;
    term.write_line("")?;

    // Show timings
    term.write_line(&format!(
        "{} Transcription: {}ms | LLM: {}ms | Total: {}ms",
        style("⏱").dim(),
        result.timings.transcription_ms,
        result.timings.llm_formatting_ms,
        result.timings.total_ms
    ))?;

    // Copy to clipboard if requested
    if clipboard {
        let mut cb = Clipboard::new()?;
        cb.set_text(&result.formatted_text)?;
        term.write_line(&format!(
            "{} Copied to clipboard",
            style("📋").green()
        ))?;
    }

    Ok(())
}
