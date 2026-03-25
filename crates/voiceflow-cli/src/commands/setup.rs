//! Setup command - download models

use anyhow::Result;
use console::{style, Term};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::process::Command;
use voiceflow_core::config::{ConsolidatedModel, LlmModel, MoonshineModel, WhisperModel};
use voiceflow_core::Config;

/// Download a single STT model (Whisper or Moonshine)
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
    let whisper_model = parse_whisper_model(whisper);
    let llm_model = parse_llm_model(llm);

    // Download Whisper model
    download_whisper_model(&term, &whisper_model)?;

    // Download LLM model
    download_llm_model(&term, &llm_model)?;

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

/// Download all benchmark models for comprehensive evaluation
pub async fn run_benchmark() -> Result<()> {
    let term = Term::stdout();

    term.write_line(&format!(
        "{} VoiceFlow Benchmark Setup",
        style("🔬").green()
    ))?;
    term.write_line("")?;

    let models_dir = Config::models_dir()?;
    term.write_line(&format!("Models directory: {:?}", models_dir))?;
    term.write_line("")?;

    // Calculate total download size
    let whisper_models = WhisperModel::benchmark_models();
    let moonshine_models = MoonshineModel::all_models();
    let llm_models = LlmModel::benchmark_models();

    let total_whisper_mb: u32 = whisper_models.iter().map(|m| m.size_mb()).sum();
    let total_moonshine_mb: u32 = moonshine_models.iter().map(|m| m.size_mb()).sum();
    let total_llm_gb: f32 = llm_models.iter().map(|m| m.size_gb()).sum();

    term.write_line(&format!(
        "{} Benchmark models to download:",
        style("📦").cyan()
    ))?;
    term.write_line("")?;

    // List STT models
    term.write_line(&format!("  {} STT Models (Whisper):", style("Speech-to-Text").bold()))?;
    for model in &whisper_models {
        let path = models_dir.join(model.filename());
        let status = if path.exists() {
            style("✓ downloaded").green()
        } else {
            style("○ pending").dim()
        };
        term.write_line(&format!(
            "    {} {} ({} MB) {}",
            status,
            model.display_name(),
            model.size_mb(),
            style(format!("WER ~{:.1}%", model.expected_wer())).dim()
        ))?;
    }
    term.write_line("")?;

    // List Moonshine models
    term.write_line(&format!("  {} STT Models (Moonshine):", style("Speech-to-Text").bold()))?;
    for model in &moonshine_models {
        let config = Config::default();
        let downloaded = config.moonshine_model_downloaded_for(model);
        let status = if downloaded {
            style("✓ downloaded").green()
        } else {
            style("○ pending").dim()
        };
        term.write_line(&format!(
            "    {} {} ({} MB)",
            status,
            model.display_name(),
            model.size_mb()
        ))?;
    }
    term.write_line("")?;

    // List LLM models
    term.write_line(&format!("  {} LLM Models:", style("Language Models").bold()))?;
    for model in &llm_models {
        let path = models_dir.join(model.filename());
        let status = if path.exists() {
            style("✓ downloaded").green()
        } else {
            style("○ pending").dim()
        };
        let multimodal = if model.is_multimodal() {
            style(" [multimodal]").magenta()
        } else {
            style("").dim()
        };
        term.write_line(&format!(
            "    {} {} ({:.1} GB){}",
            status,
            model.display_name(),
            model.size_gb(),
            multimodal
        ))?;
    }
    term.write_line("")?;

    term.write_line(&format!(
        "  {} Total: ~{} MB STT + ~{:.1} GB LLM",
        style("💾").cyan(),
        total_whisper_mb + total_moonshine_mb,
        total_llm_gb
    ))?;
    term.write_line("")?;

    // Download all models
    term.write_line(&format!("{} Downloading models...", style("⬇").cyan()))?;
    term.write_line("")?;

    // Download Whisper benchmark models
    for model in &whisper_models {
        download_whisper_model(&term, model)?;
    }

    // Download Moonshine models
    for model in &moonshine_models {
        download_moonshine_model(&term, model).await?;
    }

    // Download LLM benchmark models
    for model in &llm_models {
        download_llm_model(&term, model)?;
    }

    term.write_line("")?;
    term.write_line(&format!(
        "{} Benchmark setup complete!",
        style("✓").green()
    ))?;
    term.write_line("")?;
    term.write_line(&format!(
        "Run {} to evaluate models.",
        style("voiceflow eval --benchmark").cyan()
    ))?;

    Ok(())
}

/// List all available models and their download status
pub fn list_models() -> Result<()> {
    let term = Term::stdout();
    let models_dir = Config::models_dir()?;

    term.write_line(&format!(
        "{} Available Models",
        style("📋").green()
    ))?;
    term.write_line("")?;

    // Whisper models
    term.write_line(&format!("{}:", style("Whisper STT").bold().underlined()))?;
    for model in WhisperModel::all_models() {
        let path = models_dir.join(model.filename());
        let status = if path.exists() {
            style("✓").green()
        } else {
            style("○").dim()
        };
        let benchmark = if WhisperModel::benchmark_models().contains(&model) {
            style(" [benchmark]").cyan()
        } else {
            style("").dim()
        };
        term.write_line(&format!(
            "  {} {:30} {:>6} MB  WER ~{:.1}%{}",
            status,
            model.display_name(),
            model.size_mb(),
            model.expected_wer(),
            benchmark
        ))?;
    }
    term.write_line("")?;

    // Moonshine models
    term.write_line(&format!("{}:", style("Moonshine STT").bold().underlined()))?;
    let config = Config::default();
    for model in MoonshineModel::all_models() {
        let downloaded = config.moonshine_model_downloaded_for(&model);
        let status = if downloaded {
            style("✓").green()
        } else {
            style("○").dim()
        };
        term.write_line(&format!(
            "  {} {:30} {:>6} MB",
            status,
            model.display_name(),
            model.size_mb()
        ))?;
    }
    term.write_line("")?;

    // LLM models
    term.write_line(&format!("{}:", style("LLM Models").bold().underlined()))?;
    for model in LlmModel::all_models() {
        let path = models_dir.join(model.filename());
        let status = if path.exists() {
            style("✓").green()
        } else {
            style("○").dim()
        };
        let benchmark = if LlmModel::benchmark_models().contains(&model) {
            style(" [benchmark]").cyan()
        } else {
            style("").dim()
        };
        let multimodal = if model.is_multimodal() {
            style(" [multimodal]").magenta()
        } else {
            style("").dim()
        };
        term.write_line(&format!(
            "  {} {:30} {:>6.1} GB{}{}",
            status,
            model.display_name(),
            model.size_gb(),
            multimodal,
            benchmark
        ))?;
    }
    term.write_line("")?;

    // Consolidated models
    term.write_line(&format!("{}:", style("Consolidated Models (ASR)").bold().underlined()))?;
    let config = Config::load(None).unwrap_or_default();
    for model in ConsolidatedModel::all_models() {
        let downloaded = config.consolidated_model_downloaded_for(&model);
        let status = if downloaded {
            style("✓").green()
        } else {
            style("○").dim()
        };
        term.write_line(&format!(
            "  {} {:30} {:>6.1} GB",
            status,
            model.display_name(),
            model.size_gb()
        ))?;
    }
    term.write_line("")?;

    term.write_line(&format!(
        "Use {} to download benchmark models",
        style("voiceflow setup --benchmark").cyan()
    ))?;

    Ok(())
}

fn parse_whisper_model(name: &str) -> WhisperModel {
    match name.to_lowercase().replace("-", "_").as_str() {
        "tiny" => WhisperModel::Tiny,
        "base" => WhisperModel::Base,
        "small" => WhisperModel::Small,
        "medium" => WhisperModel::Medium,
        "large_v3" | "largev3" | "large" => WhisperModel::LargeV3,
        "large_v3_turbo" | "largev3turbo" | "turbo" => WhisperModel::LargeV3Turbo,
        "distil_large_v3" | "distillargev3" | "distil" => WhisperModel::DistilLargeV3,
        _ => {
            eprintln!("Unknown whisper model '{}', using 'base'", name);
            WhisperModel::Base
        }
    }
}

fn parse_llm_model(name: &str) -> LlmModel {
    match name.to_lowercase().replace("-", "_").as_str() {
        "qwen3.5_0_8b" | "qwen3.5_0.8b" | "qwen3.5_0.8b" => LlmModel::Qwen3_5_0_8B,
        "qwen3.5_2b" | "qwen3.5_2b" => LlmModel::Qwen3_5_2B,
        "qwen3.5_4b" | "qwen3.5_4b" => LlmModel::Qwen3_5_4B,
        _ => {
            eprintln!("Unknown LLM model '{}', using 'qwen3.5-0.8b'", name);
            LlmModel::Qwen3_5_0_8B
        }
    }
}

fn download_whisper_model(term: &Term, model: &WhisperModel) -> Result<()> {
    let models_dir = Config::models_dir()?;
    let path = models_dir.join(model.filename());

    if path.exists() {
        term.write_line(&format!(
            "  {} {} already downloaded",
            style("✓").green(),
            model.display_name()
        ))?;
    } else {
        term.write_line(&format!(
            "  {} Downloading {}...",
            style("⬇").cyan(),
            model.display_name()
        ))?;

        download_file(model.url(), &path)?;

        term.write_line(&format!(
            "  {} {} downloaded",
            style("✓").green(),
            model.display_name()
        ))?;
    }

    Ok(())
}

fn download_llm_model(term: &Term, model: &LlmModel) -> Result<()> {
    let models_dir = Config::models_dir()?;
    let path = models_dir.join(model.filename());

    if path.exists() {
        term.write_line(&format!(
            "  {} {} already downloaded",
            style("✓").green(),
            model.display_name()
        ))?;
    } else {
        if let Some(repo) = model.hf_repo() {
            term.write_line(&format!(
                "  {} Downloading {}...",
                style("⬇").cyan(),
                model.display_name()
            ))?;

            let hf_url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                repo,
                model.filename()
            );

            download_file(&hf_url, &path)?;

            term.write_line(&format!(
                "  {} {} downloaded",
                style("✓").green(),
                model.display_name()
            ))?;
        } else {
            term.write_line(&format!(
                "  {} Cannot download custom model, place at {:?}",
                style("⚠").yellow(),
                path
            ))?;
        }
    }

    Ok(())
}

async fn download_moonshine_model(term: &Term, model: &MoonshineModel) -> Result<()> {
    let models_dir = Config::models_dir()?;
    let model_dir = models_dir.join(model.dir_name());

    // Check if all files exist
    let all_exist = model.required_files().iter().all(|f| model_dir.join(f).exists());

    // Also need tokenizer
    let tokenizer_exists = model_dir.join("tokenizer.json").exists();

    if all_exist && tokenizer_exists {
        term.write_line(&format!(
            "  {} {} already downloaded",
            style("✓").green(),
            model.display_name()
        ))?;
        return Ok(());
    }

    term.write_line(&format!(
        "  {} Downloading {}...",
        style("⬇").cyan(),
        model.display_name()
    ))?;

    // Create model directory
    std::fs::create_dir_all(&model_dir)?;

    // Download ONNX files
    let base_url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        model.hf_repo(),
        model.onnx_path()
    );

    for file in model.required_files() {
        let file_path = model_dir.join(file);
        if !file_path.exists() {
            let url = format!("{}/{}", base_url, file);
            download_file(&url, &file_path)?;
        }
    }

    // Download tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        let tokenizer_url = format!(
            "https://huggingface.co/{}/resolve/main/tokenizer.json",
            model.hf_repo()
        );
        download_file(&tokenizer_url, &tokenizer_path)?;
    }

    term.write_line(&format!(
        "  {} {} downloaded",
        style("✓").green(),
        model.display_name()
    ))?;

    Ok(())
}

fn download_file(url: &str, path: &Path) -> Result<()> {
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
