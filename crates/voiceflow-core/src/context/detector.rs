//! Active application detection

/// Detected application context
#[derive(Debug, Clone)]
pub struct AppContext {
    pub app_name: String,
    pub suggested_context: String,
}

/// Detect the currently active application (macOS only)
#[cfg(target_os = "macos")]
pub fn detect_active_app() -> Option<AppContext> {
    // TODO: Implement using accessibility APIs or AppleScript
    // For now, return None
    None
}

/// Detect the currently active application (Linux/other)
#[cfg(not(target_os = "macos"))]
pub fn detect_active_app() -> Option<AppContext> {
    // Not implemented for other platforms yet
    None
}

/// Map application name to suggested context
#[allow(dead_code)]
pub fn app_to_context(app_name: &str) -> &'static str {
    let name_lower = app_name.to_lowercase();

    if name_lower.contains("mail") || name_lower.contains("outlook") || name_lower.contains("gmail")
    {
        "email"
    } else if name_lower.contains("slack")
        || name_lower.contains("discord")
        || name_lower.contains("teams")
    {
        "slack"
    } else if name_lower.contains("code")
        || name_lower.contains("xcode")
        || name_lower.contains("intellij")
        || name_lower.contains("vim")
        || name_lower.contains("neovim")
    {
        "code"
    } else if name_lower.contains("notes") || name_lower.contains("notion") {
        "notes"
    } else {
        "default"
    }
}
