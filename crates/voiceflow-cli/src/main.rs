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
        /// Whisper model size (tiny, base, small, medium)
        #[arg(long, default_value = "base")]
        whisper: String,

        /// LLM model (qwen3-1.7b, smollm3-3b, gemma2-2b)
        #[arg(long, default_value = "qwen3-1.7b")]
        llm: String,
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

        Commands::Setup { whisper, llm } => {
            commands::setup::run(&whisper, &llm).await
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

        Commands::Models => {
            commands::models::list()
        }
    }
}
