//! Config command - manage configuration

use anyhow::Result;
use console::{style, Term};
use voiceflow_core::config::{LlmModel, WhisperModel};
use voiceflow_core::Config;

pub fn show(config: &Config) -> Result<()> {
    let term = Term::stdout();

    term.write_line(&format!("{}", style("VoiceFlow Configuration").bold()))?;
    term.write_line("")?;

    term.write_line(&format!(
        "Whisper model:    {}",
        style(format!("{:?}", config.whisper_model)).cyan()
    ))?;
    term.write_line(&format!(
        "LLM model:        {}",
        style(config.llm_model.display_name()).cyan()
    ))?;
    term.write_line(&format!(
        "Default context:  {}",
        style(&config.default_context).cyan()
    ))?;
    term.write_line(&format!(
        "Auto clipboard:   {}",
        style(config.auto_clipboard).cyan()
    ))?;

    term.write_line("")?;
    term.write_line(&format!("{}", style("LLM Options:").dim()))?;
    term.write_line(&format!(
        "  Temperature:    {}",
        config.llm_options.temperature
    ))?;
    term.write_line(&format!(
        "  Max tokens:     {}",
        config.llm_options.max_tokens
    ))?;
    term.write_line(&format!(
        "  GPU layers:     {}",
        config.llm_options.n_gpu_layers
    ))?;
    term.write_line(&format!(
        "  Enable thinking: {}",
        config.llm_options.enable_thinking
    ))?;

    if !config.personal_dictionary.is_empty() {
        term.write_line("")?;
        term.write_line(&format!("{}", style("Personal Dictionary:").dim()))?;
        for word in &config.personal_dictionary {
            term.write_line(&format!("  - {}", word))?;
        }
    }

    Ok(())
}

pub fn set_model(config: &mut Config, model: &str) -> Result<()> {
    let term = Term::stdout();

    let llm_model = match model.to_lowercase().replace("-", "_").as_str() {
        "qwen3_1_7b" | "qwen3" | "qwen" => LlmModel::Qwen3_1_7B,
        "smollm3_3b" | "smollm3" | "smollm" => LlmModel::SmolLM3_3B,
        "gemma2_2b" | "gemma2" | "gemma" => LlmModel::Gemma2_2B,
        _ => {
            term.write_line(&format!(
                "{} Unknown model '{}'. Available: qwen3-1.7b, smollm3-3b, gemma2-2b",
                style("✗").red(),
                model
            ))?;
            return Ok(());
        }
    };

    config.llm_model = llm_model.clone();
    config.save(None)?;

    term.write_line(&format!(
        "{} LLM model set to: {}",
        style("✓").green(),
        llm_model.display_name()
    ))?;

    // Check if model is downloaded
    let model_path = config.llm_model_path()?;
    if !model_path.exists() {
        term.write_line(&format!(
            "{} Model not downloaded. Run: voiceflow setup --llm {}",
            style("⚠").yellow(),
            model
        ))?;
    }

    Ok(())
}

pub fn set_whisper(config: &mut Config, size: &str) -> Result<()> {
    let term = Term::stdout();

    let whisper_model = match size.to_lowercase().as_str() {
        "tiny" => WhisperModel::Tiny,
        "base" => WhisperModel::Base,
        "small" => WhisperModel::Small,
        "medium" => WhisperModel::Medium,
        _ => {
            term.write_line(&format!(
                "{} Unknown size '{}'. Available: tiny, base, small, medium",
                style("✗").red(),
                size
            ))?;
            return Ok(());
        }
    };

    config.whisper_model = whisper_model.clone();
    config.save(None)?;

    term.write_line(&format!(
        "{} Whisper model set to: {:?}",
        style("✓").green(),
        whisper_model
    ))?;

    // Check if model is downloaded
    let model_path = config.whisper_model_path()?;
    if !model_path.exists() {
        term.write_line(&format!(
            "{} Model not downloaded. Run: voiceflow setup --whisper {}",
            style("⚠").yellow(),
            size
        ))?;
    }

    Ok(())
}

pub fn add_word(config: &mut Config, word: &str) -> Result<()> {
    let term = Term::stdout();

    if config.personal_dictionary.contains(&word.to_string()) {
        term.write_line(&format!(
            "{} '{}' already in dictionary",
            style("ℹ").blue(),
            word
        ))?;
        return Ok(());
    }

    config.personal_dictionary.push(word.to_string());
    config.save(None)?;

    term.write_line(&format!(
        "{} Added '{}' to personal dictionary",
        style("✓").green(),
        word
    ))?;

    Ok(())
}

pub fn show_path() -> Result<()> {
    let term = Term::stdout();
    let config_path = Config::default_config_path()?;

    term.write_line(&format!("Config file: {:?}", config_path))?;

    if config_path.exists() {
        term.write_line(&format!("{} File exists", style("✓").green()))?;
    } else {
        term.write_line(&format!(
            "{} File does not exist (using defaults)",
            style("ℹ").blue()
        ))?;
    }

    Ok(())
}
