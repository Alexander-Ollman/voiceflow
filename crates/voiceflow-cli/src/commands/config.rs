//! Config command - manage configuration

use anyhow::Result;
use console::{style, Term};
use voiceflow_core::config::{LlmModel, WhisperModel, PipelineMode, ConsolidatedModel};
use voiceflow_core::Config;

pub fn show(config: &Config) -> Result<()> {
    let term = Term::stdout();

    term.write_line(&format!("{}", style("VoiceFlow Configuration").bold()))?;
    term.write_line("")?;

    term.write_line(&format!(
        "Pipeline mode:    {}",
        style(config.pipeline_mode.display_name()).cyan()
    ))?;

    if config.is_consolidated_mode() {
        term.write_line(&format!(
            "Consolidated:     {}",
            style(config.consolidated_model.display_name()).cyan()
        ))?;
        let downloaded = config.consolidated_model_downloaded();
        let status = if downloaded {
            style("downloaded").green()
        } else {
            style("not downloaded").red()
        };
        term.write_line(&format!(
            "  Status:         {}",
            status
        ))?;
    }

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

    // Normalize model name: replace - and . with _ for matching
    let normalized = model.to_lowercase().replace(['-', '.'], "_");
    let llm_model = match normalized.as_str() {
        "qwen3_1_7b" | "qwen3" | "qwen" => LlmModel::Qwen3_1_7B,
        "qwen3_4b" => LlmModel::Qwen3_4B,
        "smollm3_3b" | "smollm3" | "smollm" => LlmModel::SmolLM3_3B,
        "gemma2_2b" | "gemma2" | "gemma" => LlmModel::Gemma2_2B,
        "gemma3n_e2b" | "gemma3n" => LlmModel::Gemma3nE2B,
        "gemma3n_e4b" => LlmModel::Gemma3nE4B,
        "phi4_mini" | "phi4" => LlmModel::Phi4Mini,
        "phi_2" | "phi2" => LlmModel::Phi2,
        _ => {
            term.write_line(&format!(
                "{} Unknown model '{}'. Available: qwen3-1.7b, qwen3-4b, smollm3-3b, gemma2-2b, gemma3n-e2b, gemma3n-e4b, phi4-mini, phi-2",
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

pub fn set_mode(config: &mut Config, mode: &str) -> Result<()> {
    let term = Term::stdout();

    let pipeline_mode = match mode.to_lowercase().replace("-", "_").as_str() {
        "stt_plus_llm" | "traditional" | "default" => PipelineMode::SttPlusLlm,
        "consolidated" => PipelineMode::Consolidated,
        _ => {
            term.write_line(&format!(
                "{} Unknown mode '{}'. Available: stt-plus-llm, consolidated",
                style("✗").red(),
                mode
            ))?;
            return Ok(());
        }
    };

    config.pipeline_mode = pipeline_mode.clone();
    config.save(None)?;

    term.write_line(&format!(
        "{} Pipeline mode set to: {}",
        style("✓").green(),
        pipeline_mode.display_name()
    ))?;

    if matches!(pipeline_mode, PipelineMode::Consolidated) && !config.consolidated_model_downloaded() {
        term.write_line(&format!(
            "{} Consolidated model ({}) not downloaded. Run: voiceflow setup --consolidated {}",
            style("⚠").yellow(),
            config.consolidated_model.display_name(),
            config.consolidated_model.dir_name()
        ))?;
    }

    Ok(())
}

pub fn set_consolidated_model(config: &mut Config, model: &str) -> Result<()> {
    let term = Term::stdout();

    let normalized = model.to_lowercase().replace(['-', '.'], "_");
    let consolidated_model = match normalized.as_str() {
        "qwen3_asr_0_6b" | "qwen3_asr_0.6b" | "0_6b" | "0.6b" => ConsolidatedModel::Qwen3Asr0_6B,
        "qwen3_asr_1_7b" | "qwen3_asr_1.7b" | "1_7b" | "1.7b" => ConsolidatedModel::Qwen3Asr1_7B,
        _ => {
            term.write_line(&format!(
                "{} Unknown consolidated model '{}'. Available: qwen3-asr-0.6b, qwen3-asr-1.7b",
                style("✗").red(),
                model
            ))?;
            return Ok(());
        }
    };

    config.consolidated_model = consolidated_model.clone();
    config.save(None)?;

    term.write_line(&format!(
        "{} Consolidated model set to: {}",
        style("✓").green(),
        consolidated_model.display_name()
    ))?;

    if !config.consolidated_model_downloaded() {
        term.write_line(&format!(
            "{} Model not downloaded yet. Download will be available once MLX Swift integration is complete.",
            style("⚠").yellow()
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
