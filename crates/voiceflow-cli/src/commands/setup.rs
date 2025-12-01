//! Setup command - download models

use anyhow::Result;
use console::{style, Term};
use indicatif::{ProgressBar, ProgressStyle};
use std::process::Command;
use voiceflow_core::config::{LlmModel, WhisperModel};
use voiceflow_core::Config;

pub async fn run(whisper: &str, llm: &str) -> Result<()> {
    let term = Term::stdout();

    term.write_line(&format!(
        "{} VoiceFlow Setup",
        style("🚀").green()
    ))?;
    term.write_line("")?;

    let models_dir = Config::models_dir()?;
    term.write_line(&format!("Models directory: {:?}", models_dir))?;
    term.write_line("")?;

    // Parse model choices
    let whisper_model = match whisper.to_lowercase().as_str() {
        "tiny" => WhisperModel::Tiny,
        "base" => WhisperModel::Base,
        "small" => WhisperModel::Small,
        "medium" => WhisperModel::Medium,
        _ => {
            term.write_line(&format!(
                "{} Unknown whisper model '{}', using 'base'",
                style("⚠").yellow(),
                whisper
            ))?;
            WhisperModel::Base
        }
    };

    let llm_model = match llm.to_lowercase().replace("-", "_").as_str() {
        "qwen3_1_7b" | "qwen3" | "qwen" => LlmModel::Qwen3_1_7B,
        "smollm3_3b" | "smollm3" | "smollm" => LlmModel::SmolLM3_3B,
        "gemma2_2b" | "gemma2" | "gemma" => LlmModel::Gemma2_2B,
        _ => {
            term.write_line(&format!(
                "{} Unknown LLM model '{}', using 'qwen3-1.7b'",
                style("⚠").yellow(),
                llm
            ))?;
            LlmModel::Qwen3_1_7B
        }
    };

    // Download Whisper model
    let whisper_path = models_dir.join(whisper_model.filename());
    if whisper_path.exists() {
        term.write_line(&format!(
            "{} Whisper {} already downloaded",
            style("✓").green(),
            whisper
        ))?;
    } else {
        term.write_line(&format!(
            "{} Downloading Whisper {}...",
            style("⬇").cyan(),
            whisper
        ))?;

        download_file(whisper_model.url(), &whisper_path)?;

        term.write_line(&format!(
            "{} Whisper {} downloaded",
            style("✓").green(),
            whisper
        ))?;
    }

    // Download LLM model
    let llm_path = models_dir.join(llm_model.filename());
    if llm_path.exists() {
        term.write_line(&format!(
            "{} {} already downloaded",
            style("✓").green(),
            llm_model.display_name()
        ))?;
    } else {
        if let Some(repo) = llm_model.hf_repo() {
            term.write_line(&format!(
                "{} Downloading {}...",
                style("⬇").cyan(),
                llm_model.display_name()
            ))?;
            term.write_line(&format!("  From: {}", repo))?;

            // Use huggingface-cli if available, otherwise curl
            let hf_url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                repo,
                llm_model.filename()
            );

            download_file(&hf_url, &llm_path)?;

            term.write_line(&format!(
                "{} {} downloaded",
                style("✓").green(),
                llm_model.display_name()
            ))?;
        } else {
            term.write_line(&format!(
                "{} Cannot download custom model, please place it at {:?}",
                style("⚠").yellow(),
                llm_path
            ))?;
        }
    }

    // Save config with selected models
    let mut config = Config::load(None).unwrap_or_default();
    config.whisper_model = whisper_model;
    config.llm_model = llm_model;
    config.save(None)?;

    term.write_line("")?;
    term.write_line(&format!(
        "{} Setup complete! Run {} to start recording.",
        style("✓").green(),
        style("voiceflow record").cyan()
    ))?;

    Ok(())
}

fn download_file(url: &str, path: &std::path::Path) -> Result<()> {
    // Create progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"),
    );

    // Use curl for download with progress
    let output = Command::new("curl")
        .args([
            "-L",
            "-o",
            path.to_str().unwrap(),
            "--progress-bar",
            url,
        ])
        .output()?;

    pb.finish();

    if !output.status.success() {
        anyhow::bail!(
            "Download failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(())
}
