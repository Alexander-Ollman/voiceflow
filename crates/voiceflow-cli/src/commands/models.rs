//! Models command - list available models

use anyhow::Result;
use console::{style, Term};
use voiceflow_core::config::{LlmModel, WhisperModel};
use voiceflow_core::Config;

pub fn list() -> Result<()> {
    let term = Term::stdout();

    term.write_line(&format!("{}", style("Available Models").bold()))?;
    term.write_line("")?;

    // Whisper models
    term.write_line(&format!("{}", style("Whisper (Speech-to-Text):").underlined()))?;
    term.write_line("")?;

    let whisper_models = [
        (WhisperModel::Tiny, "~75MB", "Fastest, lower accuracy"),
        (WhisperModel::Base, "~150MB", "Good balance (recommended)"),
        (WhisperModel::Small, "~500MB", "Better accuracy"),
        (WhisperModel::Medium, "~1.5GB", "Best accuracy, slower"),
    ];

    let models_dir = Config::models_dir()?;

    for (model, size, desc) in whisper_models {
        let path = models_dir.join(model.filename());
        let installed = if path.exists() {
            style("✓").green()
        } else {
            style("○").dim()
        };

        term.write_line(&format!(
            "  {} {:8} {:10} {}",
            installed,
            format!("{:?}", model).to_lowercase(),
            style(size).dim(),
            desc
        ))?;
    }

    term.write_line("")?;

    // LLM models
    term.write_line(&format!(
        "{}",
        style("LLM (Text Formatting) - Non-Meta, Permissive Licenses:").underlined()
    ))?;
    term.write_line("")?;

    let llm_models = [
        (LlmModel::Qwen3_1_7B, "~1.2GB", "Fast, multilingual (Apache 2.0) - default"),
        (LlmModel::SmolLM3_3B, "~2.0GB", "Efficient, long context (Apache 2.0)"),
        (LlmModel::Gemma2_2B, "~1.5GB", "Google's compact model (Gemma license)"),
    ];

    for (model, size, desc) in llm_models {
        let path = models_dir.join(model.filename());
        let installed = if path.exists() {
            style("✓").green()
        } else {
            style("○").dim()
        };

        term.write_line(&format!(
            "  {} {:12} {:10} {}",
            installed,
            model.display_name(),
            style(size).dim(),
            desc
        ))?;
    }

    term.write_line("")?;
    term.write_line(&format!(
        "Run {} to download models",
        style("voiceflow setup").cyan()
    ))?;
    term.write_line(&format!(
        "Run {} to switch models",
        style("voiceflow config set-model <name>").cyan()
    ))?;

    Ok(())
}
