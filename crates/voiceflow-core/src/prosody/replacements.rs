//! User-defined replacement dictionary
//!
//! Loads replacements from a TOML file and applies them to transcriptions.

use std::collections::HashMap;
use std::path::Path;

/// A dictionary of text replacements
#[derive(Debug, Clone, Default)]
pub struct ReplacementDictionary {
    /// Map of "what Whisper outputs" → "correct version"
    replacements: Vec<(String, String)>,
}

impl ReplacementDictionary {
    /// Create an empty dictionary
    pub fn new() -> Self {
        Self::default()
    }

    /// Load replacements from a TOML file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let contents = std::fs::read_to_string(path)?;
        Ok(Self::parse_toml(&contents))
    }

    /// Load from the default prompts/replacements.toml location
    pub fn load_default() -> Self {
        // Try to load from the prompts directory
        if let Ok(prompts_dir) = crate::config::Config::prompts_dir() {
            let replacements_path = prompts_dir.join("replacements.toml");
            if replacements_path.exists() {
                if let Ok(dict) = Self::load_from_file(&replacements_path) {
                    tracing::info!("Loaded {} replacements from {:?}", dict.replacements.len(), replacements_path);
                    return dict;
                }
            }
        }

        // Fall back to built-in replacements
        Self::builtin()
    }

    /// Parse TOML content into replacements
    fn parse_toml(contents: &str) -> Self {
        let mut replacements = Vec::new();

        // Parse TOML
        if let Ok(value) = contents.parse::<toml::Value>() {
            if let Some(table) = value.as_table() {
                // Iterate through all sections
                for (_section_name, section_value) in table {
                    if let Some(section_table) = section_value.as_table() {
                        for (key, val) in section_table {
                            if let Some(replacement) = val.as_str() {
                                replacements.push((key.clone(), replacement.to_string()));
                            }
                        }
                    }
                }
            }
        }

        // Sort by length (longest first) to avoid partial replacements
        replacements.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Self { replacements }
    }

    /// Built-in common replacements (fallback if no file exists)
    fn builtin() -> Self {
        let mut replacements = vec![
            // ML/AI terms
            ("S M O L L M", "SmolLM"),
            ("G P T", "GPT"),
            ("L L M", "LLM"),
            ("C U D A", "CUDA"),
            ("G P U", "GPU"),
            ("C P U", "CPU"),
            ("M L", "ML"),
            ("A I", "AI"),
            // Programming
            ("A P I", "API"),
            ("U R L", "URL"),
            ("J S O N", "JSON"),
            ("H T T P", "HTTP"),
            // File extensions
            ("C P P", ".cpp"),
            ("dot cpp", ".cpp"),
            ("J S", ".js"),
            ("T S", ".ts"),
            ("P Y", ".py"),
            ("R S", ".rs"),
            // Apple
            ("Z ill oc on", "Silicon"),
            ("sill icon", "Silicon"),
            ("mac O S", "macOS"),
            ("i O S", "iOS"),
            // Tools
            ("whisper. cpp", "whisper.cpp"),
            ("whisper C P P", "whisper.cpp"),
        ];

        // Sort by length (longest first)
        replacements.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Self {
            replacements: replacements
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    /// Apply all replacements to the given text
    pub fn apply(&self, text: &str) -> String {
        let mut result = text.to_string();

        for (pattern, replacement) in &self.replacements {
            // Case-insensitive replacement
            result = case_insensitive_replace_all(&result, pattern, replacement);
        }

        result
    }

    /// Get the number of replacements loaded
    pub fn len(&self) -> usize {
        self.replacements.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.replacements.is_empty()
    }
}

/// Case-insensitive replace all occurrences
fn case_insensitive_replace_all(text: &str, pattern: &str, replacement: &str) -> String {
    let lower_text = text.to_lowercase();
    let lower_pattern = pattern.to_lowercase();

    let mut result = String::with_capacity(text.len());
    let mut last_end = 0;

    for (start, _) in lower_text.match_indices(&lower_pattern) {
        // Add text before this match
        result.push_str(&text[last_end..start]);
        // Add replacement
        result.push_str(replacement);
        last_end = start + pattern.len();
    }

    // Add remaining text
    result.push_str(&text[last_end..]);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_replacements() {
        let dict = ReplacementDictionary::builtin();
        assert!(!dict.is_empty());

        let result = dict.apply("Using G P T and C U D A");
        assert_eq!(result, "Using GPT and CUDA");
    }

    #[test]
    fn test_case_insensitive() {
        let dict = ReplacementDictionary::builtin();

        let result = dict.apply("using g p t");
        assert_eq!(result, "using GPT");
    }

    #[test]
    fn test_whisper_cpp() {
        let dict = ReplacementDictionary::builtin();

        let result = dict.apply("whisper. cpp is great");
        assert_eq!(result, "whisper.cpp is great");
    }
}
