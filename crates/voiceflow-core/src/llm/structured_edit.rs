//! Structured-edit dispatch: retroactive correction + AI voice commands.
//!
//! Wraps the LLM backend's `generate_structured` path with prompt + schema
//! pairs for each task. Callers pass already-collected context (field text,
//! recent insertions, user utterance) and get back a parsed `Edit` they can
//! apply via the platform-specific accessibility layer.

use crate::llm::backend::LlmBackendTrait;
use crate::text_normalize::CommandKind;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const RETROACTIVE_PROMPT: &str = include_str!("../../../../prompts/retroactive_correction.txt");
const AI_COMMAND_PROMPT: &str = include_str!("../../../../prompts/ai_command.txt");
const TRANSLATE_PROMPT: &str = include_str!("../../../../prompts/translate.txt");

/// What action the LLM decided on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EditAction {
    ReplaceRange,
    Insert,
    Delete,
    NoOp,
}

/// Which occurrence of `anchor` to target when it appears more than once.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Occurrence {
    First,
    Last,
    Only,
}

impl Default for Occurrence {
    fn default() -> Self {
        Occurrence::Last
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edit {
    pub action: EditAction,
    pub anchor: String,
    #[serde(default)]
    pub occurrence: Occurrence,
    pub replacement: String,
    pub confidence: f32,
    pub explanation: String,
}

/// Inputs for retroactive correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetroactiveInput {
    pub field_text: String,
    pub field_source: String,           // "ax" / "browser" / "shadow" / "ocr"
    pub recent_insertions: Vec<String>, // oldest first
    pub user_utterance: String,
}

/// JSON schema enforced on the LLM output. Matches `Edit`.
fn edit_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["action", "anchor", "occurrence", "replacement", "confidence", "explanation"],
        "properties": {
            "action": { "type": "string", "enum": ["replace_range", "insert", "delete", "no_op"] },
            "anchor": { "type": "string" },
            "occurrence": { "type": "string", "enum": ["first", "last", "only"] },
            "replacement": { "type": "string" },
            "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "explanation": { "type": "string" }
        }
    })
}

/// Run retroactive correction against the given backend.
///
/// Returns an `Edit` even on abstain — callers should check `confidence` and
/// `action == NoOp` before applying. Confidence below the caller's threshold
/// (typically 0.7) should fall through to a normal paste.
pub fn retroactive_correct(
    backend: &dyn LlmBackendTrait,
    input: &RetroactiveInput,
) -> Result<Edit> {
    let recent_block = if input.recent_insertions.is_empty() {
        "(none recorded)".to_string()
    } else {
        input
            .recent_insertions
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{}. {}", i + 1, s))
            .collect::<Vec<_>>()
            .join("\n")
    };

    let prompt = RETROACTIVE_PROMPT
        .replace("{field_source}", &input.field_source)
        .replace("{field_text}", &input.field_text)
        .replace("{recent_insertions}", &recent_block)
        .replace("{user_utterance}", &input.user_utterance);

    let raw = if backend.supports_structured() {
        backend
            .generate_structured(&prompt, &edit_schema(), 256, 0.1, 0.9)
            .context("LLM structured generation failed")?
    } else {
        // Best-effort: prompt asks for JSON; we'll validate by parsing.
        backend
            .generate(&prompt, 256, 0.1, 0.9)
            .context("LLM generation failed")?
    };

    // Some backends emit JSON wrapped in code fences or with trailing prose
    // when structured-output isn't enforced — try to be tolerant.
    let cleaned = extract_json_object(&raw);

    serde_json::from_str::<Edit>(cleaned)
        .with_context(|| format!("Failed to parse Edit from LLM output: {}", cleaned))
}

/// Pull the first balanced `{...}` substring out of a string. Returns the input
/// unchanged when no braces are found — the JSON parse will then surface a
/// useful error.
fn extract_json_object(s: &str) -> &str {
    let bytes = s.as_bytes();
    let Some(start) = bytes.iter().position(|&b| b == b'{') else {
        return s;
    };
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if escape {
            escape = false;
            continue;
        }
        match b {
            b'\\' if in_string => escape = true,
            b'"' => in_string = !in_string,
            b'{' if !in_string => depth += 1,
            b'}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return &s[start..=i];
                }
            }
            _ => {}
        }
    }
    s
}

// =============================================================================
// Phase 2: AI voice commands
// =============================================================================

/// Inputs for an AI voice command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandInput {
    pub command: CommandKind,
    /// Residual user utterance after the command trigger ("more formal", etc.)
    pub parameter: String,
    /// Active selection in the focused field. Empty when nothing is selected.
    pub selection: String,
    /// Full focused-field text. Empty when not readable.
    pub field_text: String,
    pub field_source: String,
}

/// Result of running a command. `output` is the raw text the LLM produced;
/// callers paste it (replacing selection if any).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandOutput {
    pub output: String,
    /// True if the LLM emitted `[abstain]` or `[unsure]`. Caller should not paste.
    pub abstained: bool,
}

/// Run an AI voice command against the given backend.
pub fn run_command(
    backend: &dyn LlmBackendTrait,
    input: &CommandInput,
) -> Result<CommandOutput> {
    let command_str = match input.command {
        CommandKind::Rewrite => "rewrite",
        CommandKind::Proofread => "proofread",
        CommandKind::Shorten => "shorten",
        CommandKind::Bullet => "bullet",
        CommandKind::Continue => "continue",
        CommandKind::Summarize => "summarize",
        CommandKind::Reply => "reply",
        CommandKind::Explain => "explain",
        CommandKind::Draft => "draft",
        CommandKind::Question => "question",
    };

    let prompt = AI_COMMAND_PROMPT
        .replace("{command}", command_str)
        .replace("{parameter}", input.parameter.trim())
        .replace("{selection}", &input.selection)
        .replace("{field_text}", &input.field_text)
        .replace("{field_source}", &input.field_source);

    // Split at "## Input" so the bulky data block goes in the user turn
    let (system, user) = if let Some(pos) = prompt.find("## Input") {
        (prompt[..pos].trim_end().to_string(), prompt[pos..].to_string())
    } else {
        (prompt.clone(), String::new())
    };

    // Stop sequences to prevent the model from continuing into chat-style
    // "Human:" / "AI:" rambling after the answer.
    let stop = &["Human:", "AI:", "Assistant:", "\nUser:", "</s>"];

    let raw = if backend.supports_chat() {
        backend
            .generate_chat(&system, &user, 512, 0.3, 0.95, stop)
            .context("LLM chat command generation failed")?
    } else {
        // Fallback path: relies on the prompt template's instructions.
        backend
            .generate(&prompt, 512, 0.3, 0.95)
            .context("LLM command generation failed")?
    };

    let cleaned = clean_command_output(&raw);

    // Sentinel handling
    if cleaned == "[abstain]" || cleaned == "[unsure]" {
        return Ok(CommandOutput {
            output: String::new(),
            abstained: true,
        });
    }

    Ok(CommandOutput {
        output: cleaned,
        abstained: false,
    })
}

/// Strip thinking artifacts, Human:/AI: chat markers, and code fences from
/// raw LLM output. Defense in depth — stop tokens should catch most of this,
/// but small / heavily-quantized models occasionally slip through.
fn clean_command_output(s: &str) -> String {
    let mut result = s.trim().to_string();

    // Strip leading <think>...</think> or bare </think>
    if let Some(rest) = result.strip_prefix("<think>") {
        if let Some(close) = rest.find("</think>") {
            result = rest[close + "</think>".len()..].trim_start().to_string();
        }
    } else if let Some(rest) = result.strip_prefix("</think>") {
        result = rest.trim_start().to_string();
    }

    // Strip "Human:" / "AI:" / "Assistant:" prefix if it leaked through
    for prefix in &["Human:", "AI:", "Assistant:", "User:"] {
        if let Some(rest) = result.strip_prefix(prefix) {
            result = rest.trim_start().to_string();
            break;
        }
    }

    // Truncate at the FIRST appearance of chat-style continuation markers
    // anywhere in the body (these mean the model started a new turn).
    for marker in &["\nHuman:", "\nUser:", "\nAI:", "\nAssistant:", "\n</think>"] {
        if let Some(pos) = result.find(marker) {
            result.truncate(pos);
        }
    }

    let result = result.trim().to_string();
    strip_code_fences(&result)
}

/// Strip leading ```lang … trailing ``` if the entire output is one fenced block.
fn strip_code_fences(s: &str) -> String {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        // skip optional language tag up to first newline
        let rest = rest.split_once('\n').map(|(_, body)| body).unwrap_or(rest);
        if let Some(body) = rest.strip_suffix("```") {
            return body.trim().to_string();
        }
    }
    trimmed.to_string()
}

// =============================================================================
// Phase 9: Translation
// =============================================================================

/// Translate spoken text from one language to another via the LLM.
///
/// `source_language` and `target_language` are human-readable names ("English",
/// "Spanish", "Mandarin", etc.) — the LLM does its own language resolution.
/// Returns the translation; an empty string when the LLM abstains.
pub fn translate(
    backend: &dyn LlmBackendTrait,
    spoken_text: &str,
    source_language: &str,
    target_language: &str,
) -> Result<String> {
    let prompt = TRANSLATE_PROMPT
        .replace("{spoken_text}", spoken_text)
        .replace("{source_language}", source_language)
        .replace("{target_language}", target_language);

    let (system, user) = if let Some(pos) = prompt.find("## Input") {
        (prompt[..pos].trim_end().to_string(), prompt[pos..].to_string())
    } else {
        (prompt.clone(), String::new())
    };

    let stop = &["Human:", "AI:", "Assistant:", "\nUser:", "</s>"];

    let raw = if backend.supports_chat() {
        backend
            .generate_chat(&system, &user, 512, 0.2, 0.95, stop)
            .context("LLM translate generation failed")?
    } else {
        backend.generate(&prompt, 512, 0.2, 0.95)
            .context("LLM translate generation failed")?
    };

    Ok(clean_command_output(&raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_object_strips_fences() {
        let input = "```json\n{\"a\":1,\"b\":\"x\"}\n```";
        assert_eq!(extract_json_object(input), "{\"a\":1,\"b\":\"x\"}");
    }

    #[test]
    fn extract_json_object_nested() {
        let input = "prose {\"a\":{\"b\":2}} trailing";
        assert_eq!(extract_json_object(input), "{\"a\":{\"b\":2}}");
    }

    #[test]
    fn extract_json_object_handles_braces_in_strings() {
        let input = "{\"text\":\"a } in string\"}";
        assert_eq!(extract_json_object(input), input);
    }

    #[test]
    fn edit_round_trips_json() {
        let edit = Edit {
            action: EditAction::ReplaceRange,
            anchor: "bananas".into(),
            occurrence: Occurrence::Last,
            replacement: "pears".into(),
            confidence: 0.95,
            explanation: "fixed".into(),
        };
        let json = serde_json::to_string(&edit).unwrap();
        let parsed: Edit = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.action, EditAction::ReplaceRange);
        assert_eq!(parsed.anchor, "bananas");
        assert_eq!(parsed.replacement, "pears");
    }
}
