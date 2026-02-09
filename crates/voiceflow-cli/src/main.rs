//! VoiceFlow CLI - Voice-to-text with AI formatting

use anyhow::Result;
use clap::{Parser, Subcommand};
use voiceflow_core::Config;

mod commands;

#[derive(Parser)]
#[command(name = "voiceflow")]
#[command(author = "Era Laboratories")]
#[command(version)]
#[command(about = "Voice-to-text with AI formatting", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Path to config file
    #[arg(short, long, global = true)]
    config: Option<String>,

    /// Verbose output (show timings and debug info)
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Record audio and transcribe (press Ctrl+C to stop)
    Record {
        /// Copy output to clipboard instead of stdout
        #[arg(short = 'c', long)]
        clipboard: bool,

        /// Context hint (email, slack, code, default)
        #[arg(long)]
        context: Option<String>,

        /// Skip LLM formatting, output raw transcript
        #[arg(long)]
        raw: bool,
    },

    /// Transcribe an existing audio file
    File {
        /// Path to audio file (WAV, 16kHz mono preferred)
        path: String,

        /// Context hint
        #[arg(long)]
        context: Option<String>,

        /// Skip LLM formatting
        #[arg(long)]
        raw: bool,
    },

    /// Download required models
    Setup {
        /// Whisper model size (tiny, base, small, medium, large-v3, turbo, distil)
        #[arg(long, default_value = "base")]
        whisper: String,

        /// LLM model (qwen3-1.7b, qwen3-4b, smollm3-3b, gemma2-2b, gemma3n, phi4)
        #[arg(long, default_value = "qwen3-1.7b")]
        llm: String,

        /// Download all benchmark models for comprehensive evaluation
        #[arg(long)]
        benchmark: bool,
    },

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Run benchmark on sample audio
    Bench {
        /// Number of iterations
        #[arg(short, long, default_value = "3")]
        iterations: u32,

        /// Path to test audio file
        #[arg(short, long)]
        file: Option<String>,
    },

    /// Evaluate transcription quality against LibriSpeech test-clean
    Eval {
        /// Limit number of samples to process
        #[arg(short, long)]
        limit: Option<usize>,

        /// Show sample-by-sample results (sorted by WER)
        #[arg(short, long)]
        samples: bool,

        /// Skip LLM formatting (evaluate raw transcription only)
        #[arg(long)]
        raw: bool,

        /// Analyze and categorize errors by type
        #[arg(short, long)]
        analyze: bool,

        /// Save error report to file
        #[arg(long)]
        report: Option<String>,

        /// STT model to use (moonshine-base, moonshine-tiny, whisper-base, whisper-turbo, whisper-distil)
        #[arg(long)]
        stt: Option<String>,

        /// LLM model to use (qwen3-1.7b, qwen3-4b, smollm3, gemma3n, phi4)
        #[arg(long)]
        llm: Option<String>,

        /// Run full benchmark matrix across all downloaded models
        #[arg(long)]
        benchmark: bool,
    },

    /// List available models
    Models,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,

    /// Set the LLM model
    SetModel {
        /// Model name (qwen3-1.7b, smollm3-3b, gemma2-2b)
        model: String,
    },

    /// Set the Whisper model size
    SetWhisper {
        /// Model size (tiny, base, small, medium)
        size: String,
    },

    /// Set the pipeline mode (stt-plus-llm or consolidated)
    SetMode {
        /// Pipeline mode (stt-plus-llm, consolidated)
        mode: String,
    },

    /// Set the consolidated model (used in consolidated mode)
    SetConsolidatedModel {
        /// Model name (qwen3-asr-0.6b, qwen3-asr-1.7b)
        model: String,
    },

    /// Add word to personal dictionary
    AddWord {
        /// Word to add
        word: String,
    },

    /// Show config file path
    Path,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .with_target(false)
        .init();

    // Load configuration
    let mut config = Config::load(cli.config.as_deref())?;

    match cli.command {
        Commands::Record {
            clipboard,
            context,
            raw,
        } => {
            commands::record::run(&config, clipboard, context.as_deref(), raw).await
        }

        Commands::File { path, context, raw } => {
            commands::file::run(&config, &path, context.as_deref(), raw).await
        }

        Commands::Setup { whisper, llm, benchmark } => {
            if benchmark {
                commands::setup::run_benchmark().await
            } else {
                commands::setup::run(&whisper, &llm).await
            }
        }

        Commands::Config { action } => match action {
            ConfigAction::Show => {
                commands::config::show(&config)
            }
            ConfigAction::SetModel { model } => {
                commands::config::set_model(&mut config, &model)
            }
            ConfigAction::SetWhisper { size } => {
                commands::config::set_whisper(&mut config, &size)
            }
            ConfigAction::SetMode { mode } => {
                commands::config::set_mode(&mut config, &mode)
            }
            ConfigAction::SetConsolidatedModel { model } => {
                commands::config::set_consolidated_model(&mut config, &model)
            }
            ConfigAction::AddWord { word } => {
                commands::config::add_word(&mut config, &word)
            }
            ConfigAction::Path => {
                commands::config::show_path()
            }
        },

        Commands::Bench { iterations, file } => {
            commands::bench::run(&config, iterations, file.as_deref()).await
        }

        Commands::Eval { limit, samples, raw, analyze, report, stt, llm, benchmark } => {
            if benchmark {
                commands::eval::run_benchmark(limit.unwrap_or(50)).await
            } else {
                commands::eval::run(&config, limit, samples, raw, analyze, report.as_deref(), stt.as_deref(), llm.as_deref()).await
            }
        }

        Commands::Models => {
            commands::setup::list_models()
        }
    }
}
