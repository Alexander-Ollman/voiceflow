//! OpenAI-compatible HTTP server backend.
//!
//! Talks SSE to a local server (llama-server, mlx_lm.server, vLLM, Ollama with
//! OAI shim, etc.) at /v1/chat/completions. Lets us hot-swap serving frameworks
//! without rebuilding the Rust pipeline — the only contract is the HTTP API.
//!
//! Audio-direct is intentionally unsupported (the OAI standard has no audio
//! input field); that path stays in-process behind mistral.rs/llama.cpp.

use crate::config::LlmServerConfig;
use crate::llm::backend::{LlmBackendTrait, TokenCallback};
use anyhow::{anyhow, Context, Result};
use base64::Engine as _;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::time::Duration;

pub struct OpenAIServerBackend {
    server: LlmServerConfig,
    agent: ureq::Agent,
}

impl OpenAIServerBackend {
    pub fn new(server: LlmServerConfig) -> Result<Self> {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(5))
            .timeout(Duration::from_secs(server.timeout_secs))
            .build();

        // Best-effort health check — fail loudly if the server isn't reachable so
        // users don't think their input was lost into a void.
        let probe_url = format!("{}/v1/models", server.endpoint.trim_end_matches('/'));
        match agent.get(&probe_url).call() {
            Ok(_) => tracing::info!("OpenAI-compatible server reachable at {}", server.endpoint),
            Err(e) => tracing::warn!(
                "OpenAI-compatible server probe failed at {}: {}. Generation calls may fail.",
                server.endpoint,
                e
            ),
        }

        Ok(Self { server, agent })
    }

    fn chat_url(&self) -> String {
        format!(
            "{}/v1/chat/completions",
            self.server.endpoint.trim_end_matches('/')
        )
    }

    fn completions_url(&self) -> String {
        format!(
            "{}/v1/completions",
            self.server.endpoint.trim_end_matches('/')
        )
    }

    /// Build a /v1/completions body. VoiceFlow's prompt module produces a fully-baked
    /// chat-template string (Qwen `<|im_start|>` markers), so we want raw text in,
    /// raw text out — using /v1/completions avoids double-wrapping by the server.
    fn build_completion_body(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        stream: bool,
    ) -> Value {
        json!({
            "model": self.server.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        })
    }

    /// Build a /v1/chat/completions body. Used for vision — `image_url` content
    /// blocks are only valid in the chat schema.
    fn build_chat_body(
        &self,
        messages: Value,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        stream: bool,
    ) -> Value {
        json!({
            "model": self.server.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        })
    }

    /// Build a /v1/chat/completions body with JSON-schema constraint.
    ///
    /// Uses the chat endpoint so llama-server applies the model's own template
    /// (Bonsai needs this — its baked template generates Claude-style "Human:"
    /// output when fed raw text through /v1/completions). The structured
    /// caller splits the prompt into a system block + user block.
    fn build_structured_chat_body(
        &self,
        system: &str,
        user: &str,
        schema: &Value,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Value {
        json!({
            "model": self.server.model,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": false,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": true,
                    "schema": schema,
                }
            }
        })
    }

    fn post_request(&self, url: &str, body: Value) -> Result<ureq::Response> {
        let mut req = self.agent.post(url)
            .set("Content-Type", "application/json");
        if let Some(key) = self.server.api_key.as_deref() {
            req = req.set("Authorization", &format!("Bearer {}", key));
        }
        req.send_json(body)
            .map_err(|e| anyhow!("OpenAI server request failed: {}", e))
    }

    /// Parse a non-streaming /v1/completions response: choices[0].text
    fn parse_completion_response(resp: ureq::Response) -> Result<String> {
        let v: Value = resp.into_json().context("OpenAI server returned non-JSON body")?;
        v.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("OpenAI server /v1/completions missing choices[0].text: {}", v))
    }

    /// Parse a non-streaming /v1/chat/completions response: choices[0].message.content
    fn parse_chat_response(resp: ureq::Response) -> Result<String> {
        let v: Value = resp.into_json().context("OpenAI server returned non-JSON body")?;
        v.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow!("OpenAI server /v1/chat/completions missing choices[0].message.content: {}", v))
    }

    /// Read SSE stream from either endpoint. Handles both completion (`text`) and
    /// chat (`delta.content`) frame shapes — server picks one based on the URL.
    fn read_sse_stream(
        resp: ureq::Response,
        on_token: TokenCallback<'_>,
    ) -> Result<String> {
        let reader = BufReader::new(resp.into_reader());
        let mut full = String::new();

        for line in reader.lines() {
            let line = line.context("Failed to read SSE line")?;
            let payload = match line.strip_prefix("data:") {
                Some(p) => p.trim(),
                None => continue, // blank line between events, or comment
            };
            if payload.is_empty() || payload == "[DONE]" {
                continue;
            }
            let v: Value = match serde_json::from_str(payload) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("Skipping malformed SSE frame: {} (payload: {})", e, payload);
                    continue;
                }
            };
            let token_str = v
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| {
                    // Try chat shape first
                    c.get("delta")
                        .and_then(|d| d.get("content"))
                        .and_then(|s| s.as_str())
                        // Fall back to completions shape
                        .or_else(|| c.get("text").and_then(|s| s.as_str()))
                });
            if let Some(token) = token_str {
                if !token.is_empty() {
                    full.push_str(token);
                    if !on_token(token) {
                        break;
                    }
                }
            }
        }

        Ok(full)
    }
}

impl LlmBackendTrait for OpenAIServerBackend {
    fn name(&self) -> &'static str {
        "openai-server"
    }

    fn generate(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        let body = self.build_completion_body(prompt, max_tokens, temperature, top_p, false);
        let resp = self.post_request(&self.completions_url(), body)?;
        Self::parse_completion_response(resp)
    }

    fn generate_streaming(
        &self,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        on_token: TokenCallback<'_>,
    ) -> Result<String> {
        let body = self.build_completion_body(prompt, max_tokens, temperature, top_p, true);
        let resp = self.post_request(&self.completions_url(), body)?;
        Self::read_sse_stream(resp, on_token)
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
        let messages = build_vision_messages(prompt, image);
        let body = self.build_chat_body(messages, max_tokens, temperature, top_p, false);
        let resp = self.post_request(&self.chat_url(), body)?;
        Self::parse_chat_response(resp)
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
        let messages = build_vision_messages(prompt, image);
        let body = self.build_chat_body(messages, max_tokens, temperature, top_p, true);
        let resp = self.post_request(&self.chat_url(), body)?;
        Self::read_sse_stream(resp, on_token)
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    fn generate_structured(
        &self,
        prompt: &str,
        schema: &Value,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        // Split prompt into system + user at the "## Input" boundary so the
        // chat template puts the bulky data block in the user turn (where the
        // model expects it) and instructions in the system turn.
        let (system, user) = if let Some(pos) = prompt.find("## Input") {
            (prompt[..pos].trim_end().to_string(), prompt[pos..].to_string())
        } else {
            (prompt.to_string(), String::new())
        };

        let body = self.build_structured_chat_body(
            &system, &user, schema, max_tokens, temperature, top_p,
        );
        let resp = self.post_request(&self.chat_url(), body)?;
        let raw = Self::parse_chat_response(resp)?;

        // Strip Qwen-style thinking blocks if the model emitted any despite our
        // chat-template hints. llama-server's chat template usually handles this,
        // but Bonsai's template doesn't always pre-fill <think></think>.
        let cleaned = strip_thinking_prefix(&raw);

        // Validate JSON; raise a useful error when the model strayed.
        serde_json::from_str::<Value>(cleaned).with_context(|| {
            format!(
                "Structured generation returned non-JSON output: {}",
                cleaned.chars().take(200).collect::<String>()
            )
        })?;
        Ok(cleaned.to_string())
    }

    fn supports_structured(&self) -> bool {
        true
    }

    fn generate_chat(
        &self,
        system: &str,
        user: &str,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        stop: &[&str],
    ) -> Result<String> {
        let messages = json!([
            { "role": "system", "content": system },
            { "role": "user", "content": user }
        ]);
        let mut body = self.build_chat_body(messages, max_tokens, temperature, top_p, false);
        if !stop.is_empty() {
            body["stop"] = json!(stop);
        }
        let resp = self.post_request(&self.chat_url(), body)?;
        let raw = Self::parse_chat_response(resp)?;
        Ok(strip_thinking_prefix(&raw).to_string())
    }

    fn supports_chat(&self) -> bool {
        true
    }
}

fn build_vision_messages(prompt: &str, image: &[u8]) -> Value {
    let mime = sniff_image_mime(image);
    let b64 = base64::engine::general_purpose::STANDARD.encode(image);
    let data_url = format!("data:{};base64,{}", mime, b64);
    json!([{
        "role": "user",
        "content": [
            { "type": "text", "text": prompt },
            { "type": "image_url", "image_url": { "url": data_url } }
        ]
    }])
}

/// Strip a leading `<think>...</think>` block (or `</think>` alone if llama-server
/// pre-filled an empty thinking turn) from raw model output. Returns the rest
/// trimmed of leading whitespace.
fn strip_thinking_prefix(s: &str) -> &str {
    let trimmed = s.trim_start();
    if let Some(rest) = trimmed.strip_prefix("<think>") {
        if let Some(close) = rest.find("</think>") {
            return rest[close + "</think>".len()..].trim_start();
        }
    }
    if let Some(rest) = trimmed.strip_prefix("</think>") {
        return rest.trim_start();
    }
    trimmed
}

fn sniff_image_mime(image: &[u8]) -> &'static str {
    if image.starts_with(&[0x89, b'P', b'N', b'G']) {
        "image/png"
    } else if image.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg"
    } else if image.starts_with(b"GIF8") {
        "image/gif"
    } else if image.len() >= 12 && &image[8..12] == b"WEBP" {
        "image/webp"
    } else {
        "application/octet-stream"
    }
}
