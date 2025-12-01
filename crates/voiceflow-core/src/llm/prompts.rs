//! Prompt formatting utilities

use crate::config::Config;

/// Format a prompt template with the transcript and config
pub fn format_prompt(template: &str, transcript: &str, config: &Config) -> String {
    let mut prompt = template.replace("{transcript}", transcript);

    // Add personal dictionary if present
    if !config.personal_dictionary.is_empty() {
        let dict_str = config.personal_dictionary.join(", ");
        prompt = prompt.replace(
            "{personal_dictionary}",
            &format!("\nPersonal vocabulary: {}", dict_str),
        );
    } else {
        prompt = prompt.replace("{personal_dictionary}", "");
    }

    prompt
}

/// Build a chat-formatted prompt for Qwen3/SmolLM3
#[allow(dead_code)]
pub fn build_chat_prompt(system: &str, user: &str, enable_thinking: bool) -> String {
    if enable_thinking {
        // With thinking enabled (slower but more thoughtful)
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user
        )
    } else {
        // Direct response (faster) - add /no_think for Qwen3
        format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{} /no_think<|im_end|>\n<|im_start|>assistant\n",
            system, user
        )
    }
}
