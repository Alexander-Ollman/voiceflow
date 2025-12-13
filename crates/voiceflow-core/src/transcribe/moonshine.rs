//! Moonshine speech-to-text engine using ONNX Runtime

use crate::config::Config;
use crate::transcribe::whisper::TranscriptionResult;
use anyhow::{Context, Result};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::collections::HashMap;
use std::path::Path;

/// Moonshine ONNX-based speech-to-text engine
pub struct MoonshineEngine {
    preprocess: Session,
    encode: Session,
    uncached_decode: Session,
    cached_decode: Session,
    tokenizer: Tokenizer,
}

/// Simple tokenizer for Moonshine (vocab.json based)
struct Tokenizer {
    id_to_token: HashMap<i64, String>,
    eos_token_id: i64,
    sos_token_id: i64,
}

impl Tokenizer {
    fn load(model_dir: &Path) -> Result<Self> {
        let vocab_path = model_dir.join("tokenizer.json");

        // Try to load tokenizer.json, fall back to creating default vocab
        let id_to_token = if vocab_path.exists() {
            let content = std::fs::read_to_string(&vocab_path)
                .context("Failed to read tokenizer.json")?;
            let json: serde_json::Value = serde_json::from_str(&content)
                .context("Failed to parse tokenizer.json")?;

            let mut map = HashMap::new();
            if let Some(vocab) = json.get("model").and_then(|m| m.get("vocab")).and_then(|v| v.as_object()) {
                for (token, id) in vocab {
                    if let Some(id_num) = id.as_i64() {
                        map.insert(id_num, token.clone());
                    }
                }
            } else if let Some(vocab) = json.get("vocab").and_then(|v| v.as_object()) {
                for (token, id) in vocab {
                    if let Some(id_num) = id.as_i64() {
                        map.insert(id_num, token.clone());
                    }
                }
            }
            map
        } else {
            // Create minimal fallback vocab
            tracing::warn!("tokenizer.json not found, using fallback vocabulary");
            HashMap::new()
        };

        Ok(Self {
            id_to_token,
            eos_token_id: 2, // Standard EOS token
            sos_token_id: 1, // Standard SOS/BOS token
        })
    }

    fn decode(&self, token_ids: &[i64]) -> String {
        let mut result = String::new();
        for id in token_ids {
            if *id == self.eos_token_id || *id == self.sos_token_id {
                continue;
            }
            if let Some(token) = self.id_to_token.get(id) {
                // Replace sentencepiece space marker with actual space
                let token = token.replace('▁', " ");
                result.push_str(&token);
            }
        }
        result.trim().to_string()
    }
}

impl MoonshineEngine {
    /// Create a new Moonshine engine from the configured model directory
    pub fn new(config: &Config) -> Result<Self> {
        let model_dir = config.moonshine_model_dir()?;

        if !model_dir.exists() {
            anyhow::bail!(
                "Moonshine model directory not found at {:?}. Download the model first.",
                model_dir
            );
        }

        tracing::info!("Loading Moonshine models from {:?}", model_dir);

        // Load all four ONNX models
        let preprocess = Self::load_session(&model_dir, "preprocess.onnx")?;
        let encode = Self::load_session(&model_dir, "encode.onnx")?;
        let uncached_decode = Self::load_session(&model_dir, "uncached_decode.onnx")?;
        let cached_decode = Self::load_session(&model_dir, "cached_decode.onnx")?;

        // Load tokenizer
        let tokenizer = Tokenizer::load(&model_dir)?;

        Ok(Self {
            preprocess,
            encode,
            uncached_decode,
            cached_decode,
            tokenizer,
        })
    }

    fn load_session(model_dir: &Path, filename: &str) -> Result<Session> {
        let path = model_dir.join(filename);
        if !path.exists() {
            anyhow::bail!("Model file not found: {:?}", path);
        }

        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(std::thread::available_parallelism()?.get())?
            .commit_from_file(&path)
            .with_context(|| format!("Failed to load ONNX model: {:?}", path))
    }

    /// Transcribe audio samples to text
    ///
    /// # Arguments
    /// * `audio` - PCM f32 samples at 16kHz (caller must resample first)
    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        Ok(self.transcribe_with_timestamps(audio, false)?.text)
    }

    /// Transcribe audio samples with optional word-level timestamps
    ///
    /// Note: Moonshine doesn't provide word-level timestamps natively,
    /// so this returns an empty timestamps vector.
    pub fn transcribe_with_timestamps(
        &mut self,
        audio: &[f32],
        _enable_timestamps: bool,
    ) -> Result<TranscriptionResult> {
        if audio.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                word_timestamps: vec![],
            });
        }

        // Calculate max tokens based on audio duration (6 tokens per second is typical)
        let duration_secs = audio.len() as f32 / 16000.0;
        let max_tokens = ((duration_secs * 6.0) as usize).max(10).min(448);

        // Step 1: Preprocess audio - shape [1, audio_len]
        let audio_tensor = Tensor::from_array(([1usize, audio.len()], audio.to_vec()))?;

        let preprocess_outputs = self.preprocess.run(ort::inputs!["args_0" => audio_tensor])?;
        let features_value = &preprocess_outputs["sequential"];
        let (features_shape, features_data) = features_value.try_extract_tensor::<f32>()?;

        let seq_len = features_shape[1] as i32;

        // Step 2: Encode
        let features_tensor = Tensor::from_array((features_shape.to_vec(), features_data.to_vec()))?;
        let seq_len_tensor = Tensor::from_array(([1usize], vec![seq_len]))?;

        let encode_outputs = self.encode.run(ort::inputs![
            "args_0" => features_tensor,
            "args_1" => seq_len_tensor
        ])?;

        // Use index-based access since output name differs between tiny/base models
        let context_value = encode_outputs.iter().next()
            .ok_or_else(|| anyhow::anyhow!("No output from encode model"))?
            .1;
        let (context_shape, context_data) = context_value.try_extract_tensor::<f32>()?;

        // Step 3: Uncached decode (first token)
        // IMPORTANT: Model expects int32 tensors, not int64
        let initial_token = Tensor::from_array(([1usize, 1], vec![self.tokenizer.sos_token_id as i32]))?;
        let context_tensor = Tensor::from_array((context_shape.to_vec(), context_data.to_vec()))?;
        let seq_len_decode = Tensor::from_array(([1usize], vec![1i32]))?;

        let uncached_outputs = self.uncached_decode.run(ort::inputs![
            "args_0" => initial_token,
            "args_1" => context_tensor,
            "args_2" => seq_len_decode
        ])?;

        // Collect all outputs - first is logits, rest are cache tensors
        let output_names: Vec<_> = uncached_outputs.iter().map(|(name, _)| name.to_string()).collect();

        // Get logits (first output)
        let logits_value = uncached_outputs.iter().next()
            .ok_or_else(|| anyhow::anyhow!("No output from uncached_decode model"))?
            .1;
        let (logits_shape, logits_data) = logits_value.try_extract_tensor::<f32>()?;

        let mut tokens = Vec::new();
        let first_token = Self::argmax(logits_data);

        eprintln!("Moonshine: first token = {}, EOS = {}", first_token, self.tokenizer.eos_token_id);
        eprintln!("Moonshine: vocab size = {}", self.tokenizer.id_to_token.len());

        if first_token == self.tokenizer.eos_token_id {
            eprintln!("Moonshine: first token is EOS, returning empty");
            return Ok(TranscriptionResult {
                text: String::new(),
                word_timestamps: vec![],
            });
        }
        tokens.push(first_token);

        // Extract cache data as owned vectors (shape + data) to avoid ONNX Runtime mutex issues
        let mut cache_data: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();
        for (_name, value) in uncached_outputs.iter().skip(1) {
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            cache_data.push((shape.to_vec(), data.to_vec()));
        }
        // Drop uncached_outputs to release ONNX Runtime resources
        drop(uncached_outputs);

        // Step 4: Cached decode loop for remaining tokens
        let mut current_token = first_token;

        for step in 2..=max_tokens {
            // Build inputs for cached_decode
            let token_tensor = Tensor::from_array(([1usize, 1], vec![current_token as i32]))?;
            let pos_tensor = Tensor::from_array(([1usize], vec![step as i32]))?;

            // Create input map with token, context, position, and cache tensors
            let mut inputs: Vec<(std::borrow::Cow<str>, ort::value::DynValue)> = Vec::new();
            inputs.push(("args_0".into(), token_tensor.into()));
            inputs.push(("args_1".into(), Tensor::from_array((context_shape.to_vec(), context_data.to_vec()))?.into()));
            inputs.push(("args_2".into(), pos_tensor.into()));

            // Add cache tensors from owned data
            for (i, (shape, data)) in cache_data.iter().enumerate() {
                let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
                let tensor = Tensor::from_array((shape_usize, data.clone()))?;
                inputs.push((format!("args_{}", i + 3).into(), tensor.into()));
            }

            let cached_outputs = self.cached_decode.run(inputs)?;

            // Get logits from first output
            let logits_value = cached_outputs.iter().next()
                .ok_or_else(|| anyhow::anyhow!("No output from cached_decode"))?
                .1;
            let (_, logits_data) = logits_value.try_extract_tensor::<f32>()?;

            let next_token = Self::argmax(logits_data);

            if next_token == self.tokenizer.eos_token_id {
                break;
            }

            tokens.push(next_token);
            current_token = next_token;

            // Update cache data for next iteration (extract to owned vectors)
            cache_data.clear();
            for (_, value) in cached_outputs.iter().skip(1) {
                let (shape, data) = value.try_extract_tensor::<f32>()?;
                cache_data.push((shape.to_vec(), data.to_vec()));
            }
            // Drop cached_outputs to release ONNX Runtime resources
            drop(cached_outputs);
        }

        eprintln!("Moonshine: generated {} tokens: {:?}", tokens.len(), &tokens[..tokens.len().min(20)]);
        let text = self.tokenizer.decode(&tokens);
        eprintln!("Moonshine: decoded text = '{}'", text);

        Ok(TranscriptionResult {
            text,
            word_timestamps: vec![], // Moonshine doesn't provide timestamps
        })
    }

    /// Find the index of the maximum value in a slice
    fn argmax(slice: &[f32]) -> i64 {
        slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as i64)
            .unwrap_or(0)
    }
}
