//! mistral.rs backend for LLM inference (Candle-based, no GGML)
//!
//! Supports Gemma 4 E2B/E4B with native audio input.
//! Uses HuggingFace hub for model downloads and ISQ quantization at load time.
//! Entire module gated behind `#[cfg(feature = "mistralrs")]`.

use crate::config::Config;
use crate::llm::backend::{LlmBackendTrait, TokenCallback};
use crate::runtime;
use anyhow::{Context, Result};
use mistralrs::{
    AudioInput, ChatCompletionChunkResponse, ChunkChoice, Delta, IsqBits, Model,
    MultimodalMessages, MultimodalModelBuilder, Response, TextMessageRole, TextMessages,
    RequestBuilder,
};
use std::io::Write as _;
use std::sync::Arc;

/// Maximum audio duration in seconds (Gemma 4 limitation)
const MAX_AUDIO_DURATION_SECS: f32 = 30.0;

/// mistral.rs backend implementation
pub struct MistralRsBackend {
    model: Arc<Model>,
}

impl MistralRsBackend {
    /// Create a new mistral.rs backend
    ///
    /// Downloads the model from HuggingFace hub on first run and applies ISQ4 quantization.
    /// Subsequent runs use the cached model.
    pub fn new(config: &Config) -> Result<Self> {
        let model_id = config
            .llm_model
            .hf_model_id()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Model {} does not have a HuggingFace model ID for mistral.rs",
                    config.llm_model.display_name()
                )
            })?
            .to_string();

        tracing::info!(
            "Loading model {} with mistral.rs (ISQ4 quantization)",
            config.llm_model.display_name()
        );

        // Build the multimodal model (Gemma 4 supports text, vision, and audio)
        // ISQ4 quantization is applied at load time for memory efficiency
        let model = runtime::block_on(async {
            MultimodalModelBuilder::new(&model_id)
                .with_auto_isq(IsqBits::Four)
                .with_logging()
                .build()
                .await
        })
        .context("Failed to build mistral.rs multimodal model")?;

        tracing::info!(
            "mistral.rs backend loaded: {} (ISQ4)",
            config.llm_model.display_name()
        );

        Ok(Self {
            model: Arc::new(model),
        })
    }

    /// Encode PCM f32 samples as an in-memory WAV buffer
    /// mistral.rs AudioInput::from_bytes expects WAV/MP3/FLAC format
    fn pcm_to_wav(audio_pcm: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
        let mut cursor = std::io::Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .context("Failed to create WAV writer")?;
        for &sample in audio_pcm {
            writer
                .write_sample(sample)
                .context("Failed to write WAV sample")?;
        }
        writer.finalize().context("Failed to finalize WAV")?;
        Ok(cursor.into_inner())
    }

    /// Build a RequestBuilder with standard sampling parameters
    fn build_request(
        &self,
        messages: impl Into<RequestBuilder>,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> RequestBuilder {
        let request: RequestBuilder = messages.into();
        request
            .set_sampler_temperature(temperature as f64)
            .set_sampler_topp(top_p as f64)
            .set_sampler_max_len(max_tokens as usize)
            .enable_thinking(false)
    }

    /// Send a non-streaming chat request
    fn send_request(&self, request: RequestBuilder) -> Result<String> {
        let model = Arc::clone(&self.model);
        let response = runtime::block_on(async move {
            model.send_chat_request(request).await
        })
        .context("mistral.rs chat request failed")?;

        response.choices[0]
            .message
            .content
            .as_ref()
            .map(|s| s.trim().to_string())
            .ok_or_else(|| anyhow::anyhow!("mistral.rs returned empty response"))
    }

    /// Send a streaming chat request, forwarding tokens through an mpsc channel
    /// so the non-Send callback can consume them on the calling thread.
    fn send_request_streaming(
        &self,
        request: RequestBuilder,
        on_token: TokenCallback<'_>,
    ) -> Result<String> {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let model = Arc::clone(&self.model);

        // Spawn the async stream consumer on the runtime
        let join_handle = std::thread::spawn(move || -> Result<()> {
            runtime::block_on(async move {
                let mut stream = model
                    .stream_chat_request(request)
                    .await
                    .context("Failed to start streaming")?;

                while let Some(chunk) = stream.next().await {
                    if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = chunk {
                        if let Some(ChunkChoice {
                            delta: Delta {
                                content: Some(content),
                                ..
                            },
                            ..
                        }) = choices.first()
                        {
                            if tx.send(content.clone()).is_err() {
                                break; // Receiver dropped (callback returned false)
                            }
                        }
                    }
                }
                Ok(())
            })
        });

        // Consume tokens on the calling thread where the callback lives
        let mut full_output = String::new();
        for token in rx {
            full_output.push_str(&token);
            if !on_token(&token) {
                break;
            }
        }

        // Wait for the async thread to finish
        join_handle
            .join()
            .map_err(|_| anyhow::anyhow!("Streaming thread panicked"))??;

        Ok(full_output.trim().to_string())
    }
}

impl LlmBackendTrait for MistralRsBackend {
    fn generate(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        let messages = TextMessages::new()
            .add_message(TextMessageRole::User, prompt);
        let request = self.build_request(messages, max_tokens, temperature, top_p);
        self.send_request(request)
    }

    fn name(&self) -> &'static str {
        "mistral.rs"
    }

    fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        on_token: TokenCallback<'_>,
    ) -> Result<String> {
        let messages = TextMessages::new()
            .add_message(TextMessageRole::User, prompt);
        let request = self.build_request(messages, max_tokens, temperature, top_p);
        self.send_request_streaming(request, on_token)
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn generate_with_image(
        &self,
        prompt: &str,
        image: &[u8],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        let img = image::load_from_memory(image)
            .context("Failed to decode image for mistral.rs")?;
        let messages = MultimodalMessages::new().add_multimodal_message(
            TextMessageRole::User,
            prompt,
            vec![img],
            vec![],
            vec![],
        );
        let request = self.build_request(messages, max_tokens, temperature, top_p);
        self.send_request(request)
    }

    fn generate_with_image_streaming(
        &self,
        prompt: &str,
        image: &[u8],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        on_token: TokenCallback<'_>,
    ) -> Result<String> {
        let img = image::load_from_memory(image)
            .context("Failed to decode image for mistral.rs")?;
        let messages = MultimodalMessages::new().add_multimodal_message(
            TextMessageRole::User,
            prompt,
            vec![img],
            vec![],
            vec![],
        );
        let request = self.build_request(messages, max_tokens, temperature, top_p);
        self.send_request_streaming(request, on_token)
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    fn generate_from_audio(
        &self,
        audio_pcm: &[f32],
        sample_rate: u32,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        let duration_secs = audio_pcm.len() as f32 / sample_rate as f32;

        // Truncate if exceeding Gemma 4's 30s audio cap
        let audio = if duration_secs > MAX_AUDIO_DURATION_SECS {
            tracing::warn!(
                "Audio duration {:.1}s exceeds {:.0}s limit, truncating",
                duration_secs,
                MAX_AUDIO_DURATION_SECS
            );
            let max_samples = (MAX_AUDIO_DURATION_SECS * sample_rate as f32) as usize;
            &audio_pcm[..max_samples]
        } else {
            audio_pcm
        };

        let wav_bytes = Self::pcm_to_wav(audio, sample_rate)?;
        let audio_input = AudioInput::from_bytes(&wav_bytes)
            .context("Failed to create AudioInput from WAV bytes")?;

        let messages = MultimodalMessages::new().add_audio_message(
            TextMessageRole::User,
            prompt,
            vec![audio_input],
        );
        let request = self.build_request(messages, max_tokens, temperature, top_p);
        self.send_request(request)
    }

    fn generate_from_audio_streaming(
        &self,
        audio_pcm: &[f32],
        sample_rate: u32,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        on_token: TokenCallback<'_>,
    ) -> Result<String> {
        let duration_secs = audio_pcm.len() as f32 / sample_rate as f32;

        let audio = if duration_secs > MAX_AUDIO_DURATION_SECS {
            tracing::warn!(
                "Audio duration {:.1}s exceeds {:.0}s limit, truncating",
                duration_secs,
                MAX_AUDIO_DURATION_SECS
            );
            let max_samples = (MAX_AUDIO_DURATION_SECS * sample_rate as f32) as usize;
            &audio_pcm[..max_samples]
        } else {
            audio_pcm
        };

        let wav_bytes = Self::pcm_to_wav(audio, sample_rate)?;
        let audio_input = AudioInput::from_bytes(&wav_bytes)
            .context("Failed to create AudioInput from WAV bytes")?;

        let messages = MultimodalMessages::new().add_audio_message(
            TextMessageRole::User,
            prompt,
            vec![audio_input],
        );
        let request = self.build_request(messages, max_tokens, temperature, top_p);
        self.send_request_streaming(request, on_token)
    }

    fn supports_audio_direct(&self) -> bool {
        true
    }
}
