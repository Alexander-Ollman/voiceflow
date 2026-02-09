//! Evaluate input context continuation approaches for capitalization accuracy
//!
//! Tests two approaches for handling input field context:
//! - Approach A: Prompt-based (add smart continuation hint to context)
//! - Approach B: Structural (concatenate input text with transcript in ## Input)
//!
//! Run: cargo run --release -p voiceflow-core --example eval_context

use std::time::Instant;
use voiceflow_core::config::{Config, LlmModel};
use voiceflow_core::llm::{format_prompt, LlmEngine};

/// A single test case
struct TestCase {
    /// Text already in the input field (before cursor). Empty = fresh field.
    input_context: &'static str,
    /// Raw STT transcript (what the user just said)
    transcript: &'static str,
    /// Expected formatted output
    expected: &'static str,
    /// Category for grouping results
    category: &'static str,
}

fn test_cases() -> Vec<TestCase> {
    vec![
        // ===== CATEGORY: Fresh field (no input context) =====
        TestCase {
            input_context: "",
            transcript: "hello world",
            expected: "Hello world",
            category: "fresh",
        },
        TestCase {
            input_context: "",
            transcript: "this is a test of the dictation system",
            expected: "This is a test of the dictation system.",
            category: "fresh",
        },
        TestCase {
            input_context: "",
            transcript: "can you send me the report by friday",
            expected: "Can you send me the report by Friday?",
            category: "fresh",
        },
        TestCase {
            input_context: "",
            transcript: "the meeting is at three thirty pm tomorrow",
            expected: "The meeting is at 3:30 PM tomorrow.",
            category: "fresh",
        },
        TestCase {
            input_context: "",
            transcript: "please call doctor smith at his office",
            expected: "Please call Dr. Smith at his office.",
            category: "fresh",
        },
        TestCase {
            input_context: "",
            transcript: "i need to buy milk eggs and bread",
            expected: "I need to buy milk, eggs, and bread.",
            category: "fresh",
        },
        TestCase {
            input_context: "",
            transcript: "what time does the store close",
            expected: "What time does the store close?",
            category: "fresh",
        },
        TestCase {
            input_context: "",
            transcript: "the quick brown fox jumps over the lazy dog",
            expected: "The quick brown fox jumps over the lazy dog.",
            category: "fresh",
        },
        TestCase {
            input_context: "",
            transcript: "we should discuss this at the next team standup",
            expected: "We should discuss this at the next team standup.",
            category: "fresh",
        },
        TestCase {
            input_context: "",
            transcript: "i think the API endpoint needs to be updated",
            expected: "I think the API endpoint needs to be updated.",
            category: "fresh",
        },

        // ===== CATEGORY: After period (should capitalize) =====
        TestCase {
            input_context: "The first task is done.",
            transcript: "now we need to move on to the second one",
            expected: "Now we need to move on to the second one.",
            category: "after_period",
        },
        TestCase {
            input_context: "I finished the report.",
            transcript: "please review it when you get a chance",
            expected: "Please review it when you get a chance.",
            category: "after_period",
        },
        TestCase {
            input_context: "That sounds great.",
            transcript: "let me know if you need anything else",
            expected: "Let me know if you need anything else.",
            category: "after_period",
        },
        TestCase {
            input_context: "Done.",
            transcript: "what should we work on next",
            expected: "What should we work on next?",
            category: "after_period",
        },
        TestCase {
            input_context: "The build passed.",
            transcript: "we can deploy to production now",
            expected: "We can deploy to production now.",
            category: "after_period",
        },
        TestCase {
            input_context: "I agree with the plan.",
            transcript: "however we should also consider the budget",
            expected: "However, we should also consider the budget.",
            category: "after_period",
        },
        TestCase {
            input_context: "Meeting ended early.",
            transcript: "everyone was aligned on the approach",
            expected: "Everyone was aligned on the approach.",
            category: "after_period",
        },
        TestCase {
            input_context: "The PR was merged.",
            transcript: "there were no conflicts",
            expected: "There were no conflicts.",
            category: "after_period",
        },
        TestCase {
            input_context: "That's correct.",
            transcript: "the server is running on port eight thousand",
            expected: "The server is running on port 8000.",
            category: "after_period",
        },
        TestCase {
            input_context: "Sounds good.",
            transcript: "i will send the invite shortly",
            expected: "I will send the invite shortly.",
            category: "after_period",
        },

        // ===== CATEGORY: After question mark (should capitalize) =====
        TestCase {
            input_context: "Did you finish the task?",
            transcript: "yes i completed it this morning",
            expected: "Yes, I completed it this morning.",
            category: "after_question",
        },
        TestCase {
            input_context: "What do you think?",
            transcript: "i think we should go with option two",
            expected: "I think we should go with option two.",
            category: "after_question",
        },
        TestCase {
            input_context: "Are we ready to ship?",
            transcript: "not yet we still need to fix the login bug",
            expected: "Not yet, we still need to fix the login bug.",
            category: "after_question",
        },
        TestCase {
            input_context: "Can you check on that?",
            transcript: "sure i will look into it right away",
            expected: "Sure, I will look into it right away.",
            category: "after_question",
        },
        TestCase {
            input_context: "When is the deadline?",
            transcript: "it is due next friday",
            expected: "It is due next Friday.",
            category: "after_question",
        },

        // ===== CATEGORY: Mid-sentence continuation (should NOT capitalize) =====
        TestCase {
            input_context: "I was thinking we could",
            transcript: "maybe try a different approach to this problem",
            expected: "maybe try a different approach to this problem.",
            category: "mid_sentence",
        },
        TestCase {
            input_context: "The reason I asked is",
            transcript: "because the current implementation has a bug",
            expected: "because the current implementation has a bug.",
            category: "mid_sentence",
        },
        TestCase {
            input_context: "We should probably",
            transcript: "update the documentation before the release",
            expected: "update the documentation before the release.",
            category: "mid_sentence",
        },
        TestCase {
            input_context: "The main issue is that",
            transcript: "the database queries are too slow",
            expected: "the database queries are too slow.",
            category: "mid_sentence",
        },
        TestCase {
            input_context: "I need to",
            transcript: "finish this before the end of the day",
            expected: "finish this before the end of the day.",
            category: "mid_sentence",
        },
        TestCase {
            input_context: "Let me explain why",
            transcript: "this matters for the project timeline",
            expected: "this matters for the project timeline.",
            category: "mid_sentence",
        },
        TestCase {
            input_context: "He said that",
            transcript: "the deployment would be delayed by a week",
            expected: "the deployment would be delayed by a week.",
            category: "mid_sentence",
        },
        TestCase {
            input_context: "The problem with",
            transcript: "the current setup is that it doesn't scale",
            expected: "the current setup is that it doesn't scale.",
            category: "mid_sentence",
        },
        TestCase {
            input_context: "I wanted to mention that",
            transcript: "we have a new team member starting monday",
            expected: "we have a new team member starting Monday.",
            category: "mid_sentence",
        },
        TestCase {
            input_context: "Please make sure to",
            transcript: "test everything before pushing to main",
            expected: "test everything before pushing to main.",
            category: "mid_sentence",
        },

        // ===== CATEGORY: After comma (should NOT capitalize) =====
        TestCase {
            input_context: "First,",
            transcript: "we need to set up the development environment",
            expected: "we need to set up the development environment.",
            category: "after_comma",
        },
        TestCase {
            input_context: "In addition,",
            transcript: "we should add more unit tests",
            expected: "we should add more unit tests.",
            category: "after_comma",
        },
        TestCase {
            input_context: "On the other hand,",
            transcript: "the existing solution works fine for now",
            expected: "the existing solution works fine for now.",
            category: "after_comma",
        },
        TestCase {
            input_context: "If that's the case,",
            transcript: "then we should postpone the migration",
            expected: "then we should postpone the migration.",
            category: "after_comma",
        },
        TestCase {
            input_context: "As I mentioned before,",
            transcript: "the server needs to be restarted",
            expected: "the server needs to be restarted.",
            category: "after_comma",
        },

        // ===== CATEGORY: After colon (context-dependent) =====
        TestCase {
            input_context: "Here's the plan:",
            transcript: "we deploy on tuesday and monitor for a week",
            expected: "we deploy on Tuesday and monitor for a week.",
            category: "after_colon",
        },
        TestCase {
            input_context: "Action items:",
            transcript: "update the docs fix the bug and write tests",
            expected: "update the docs, fix the bug, and write tests.",
            category: "after_colon",
        },
        TestCase {
            input_context: "Note:",
            transcript: "this only applies to the staging environment",
            expected: "this only applies to the staging environment.",
            category: "after_colon",
        },

        // ===== CATEGORY: Proper nouns after mid-sentence (should capitalize proper nouns only) =====
        TestCase {
            input_context: "I was talking to",
            transcript: "john about the project and he agreed",
            expected: "John about the project and he agreed.",
            category: "proper_noun_mid",
        },
        TestCase {
            input_context: "We should deploy to",
            transcript: "amazon web services using the new pipeline",
            expected: "Amazon Web Services using the new pipeline.",
            category: "proper_noun_mid",
        },
        TestCase {
            input_context: "The ticket was assigned to",
            transcript: "sarah from the backend team",
            expected: "Sarah from the backend team.",
            category: "proper_noun_mid",
        },
        TestCase {
            input_context: "Can you send it to",
            transcript: "michael and jennifer on the design team",
            expected: "Michael and Jennifer on the design team.",
            category: "proper_noun_mid",
        },
        TestCase {
            input_context: "This is similar to how",
            transcript: "google handles their search indexing",
            expected: "Google handles their search indexing.",
            category: "proper_noun_mid",
        },

        // ===== CATEGORY: After exclamation (should capitalize) =====
        TestCase {
            input_context: "Great news!",
            transcript: "the feature is ready for testing",
            expected: "The feature is ready for testing.",
            category: "after_exclamation",
        },
        TestCase {
            input_context: "Watch out!",
            transcript: "that endpoint is deprecated",
            expected: "That endpoint is deprecated.",
            category: "after_exclamation",
        },
        TestCase {
            input_context: "Nice work!",
            transcript: "the performance improved by fifty percent",
            expected: "The performance improved by 50%.",
            category: "after_exclamation",
        },

        // ===== CATEGORY: Multi-sentence input context =====
        TestCase {
            input_context: "I reviewed the PR. There are a few issues.",
            transcript: "the main one is the error handling in the auth module",
            expected: "The main one is the error handling in the auth module.",
            category: "multi_sentence",
        },
        TestCase {
            input_context: "The build failed again. I checked the logs.",
            transcript: "it looks like a dependency conflict",
            expected: "It looks like a dependency conflict.",
            category: "multi_sentence",
        },
        TestCase {
            input_context: "We had a good meeting. Everyone agreed on the direction. However,",
            transcript: "we still need to finalize the timeline",
            expected: "we still need to finalize the timeline.",
            category: "multi_sentence",
        },
        TestCase {
            input_context: "The migration is complete. All data was transferred. Next,",
            transcript: "we need to verify the integrity of the records",
            expected: "we need to verify the integrity of the records.",
            category: "multi_sentence",
        },
        TestCase {
            input_context: "Step one is done. Step two is in progress. For step three,",
            transcript: "we will need the credentials from the ops team",
            expected: "we will need the credentials from the ops team.",
            category: "multi_sentence",
        },

        // ===== CATEGORY: Technical terms / acronyms =====
        TestCase {
            input_context: "",
            transcript: "the API returns a JSON response with the user ID",
            expected: "The API returns a JSON response with the user ID.",
            category: "technical",
        },
        TestCase {
            input_context: "For the OAuth flow,",
            transcript: "we need to configure the redirect URL in the dashboard",
            expected: "we need to configure the redirect URL in the dashboard.",
            category: "technical",
        },
        TestCase {
            input_context: "",
            transcript: "run npm install and then npm run build",
            expected: "Run npm install and then npm run build.",
            category: "technical",
        },
        TestCase {
            input_context: "The CI pipeline",
            transcript: "uses github actions to run the tests automatically",
            expected: "uses GitHub Actions to run the tests automatically.",
            category: "technical",
        },
        TestCase {
            input_context: "",
            transcript: "we should switch from REST to GraphQL for this endpoint",
            expected: "We should switch from REST to GraphQL for this endpoint.",
            category: "technical",
        },

        // ===== CATEGORY: Numbers and formatting =====
        TestCase {
            input_context: "",
            transcript: "the budget is fifty thousand dollars",
            expected: "The budget is $50,000.",
            category: "numbers",
        },
        TestCase {
            input_context: "The meeting is scheduled for",
            transcript: "january fifteenth at two thirty pm",
            expected: "January 15 at 2:30 PM.",
            category: "numbers",
        },
        TestCase {
            input_context: "",
            transcript: "we have about twenty five percent capacity remaining",
            expected: "We have about 25% capacity remaining.",
            category: "numbers",
        },
        TestCase {
            input_context: "There are",
            transcript: "three hundred and forty two items in the queue",
            expected: "342 items in the queue.",
            category: "numbers",
        },
        TestCase {
            input_context: "Call me at",
            transcript: "five five five one two three four five six seven",
            expected: "555-123-4567.",
            category: "numbers",
        },

        // ===== CATEGORY: Short utterances (fresh) =====
        TestCase {
            input_context: "",
            transcript: "yes",
            expected: "Yes",
            category: "short",
        },
        TestCase {
            input_context: "",
            transcript: "sounds good",
            expected: "Sounds good.",
            category: "short",
        },
        TestCase {
            input_context: "",
            transcript: "thank you",
            expected: "Thank you.",
            category: "short",
        },
        TestCase {
            input_context: "",
            transcript: "no problem",
            expected: "No problem.",
            category: "short",
        },
        TestCase {
            input_context: "",
            transcript: "got it",
            expected: "Got it.",
            category: "short",
        },

        // ===== CATEGORY: Short utterances (continuation) =====
        TestCase {
            input_context: "I think we should",
            transcript: "wait",
            expected: "wait.",
            category: "short_continuation",
        },
        TestCase {
            input_context: "That's",
            transcript: "correct",
            expected: "correct.",
            category: "short_continuation",
        },
        TestCase {
            input_context: "The answer is",
            transcript: "no",
            expected: "no.",
            category: "short_continuation",
        },
        TestCase {
            input_context: "Please",
            transcript: "stop",
            expected: "stop.",
            category: "short_continuation",
        },
        TestCase {
            input_context: "It was",
            transcript: "fine actually",
            expected: "fine, actually.",
            category: "short_continuation",
        },

        // ===== CATEGORY: Tricky cases (and/but/or starting after period) =====
        TestCase {
            input_context: "The first option didn't work.",
            transcript: "and the second one was even worse",
            expected: "And the second one was even worse.",
            category: "conjunction_after_period",
        },
        TestCase {
            input_context: "I tried everything.",
            transcript: "but nothing seemed to fix the issue",
            expected: "But nothing seemed to fix the issue.",
            category: "conjunction_after_period",
        },
        TestCase {
            input_context: "We can go with plan A.",
            transcript: "or we could try a completely different approach",
            expected: "Or we could try a completely different approach.",
            category: "conjunction_after_period",
        },
        TestCase {
            input_context: "That's not ideal.",
            transcript: "however it is the best we can do right now",
            expected: "However, it is the best we can do right now.",
            category: "conjunction_after_period",
        },

        // ===== CATEGORY: Conjunction mid-sentence (should NOT capitalize) =====
        TestCase {
            input_context: "I finished the frontend",
            transcript: "and now i am working on the backend",
            expected: "and now I am working on the backend.",
            category: "conjunction_mid",
        },
        TestCase {
            input_context: "The tests passed",
            transcript: "but we still need code review",
            expected: "but we still need code review.",
            category: "conjunction_mid",
        },
        TestCase {
            input_context: "We can use postgres",
            transcript: "or switch to mongodb if needed",
            expected: "or switch to MongoDB if needed.",
            category: "conjunction_mid",
        },
        TestCase {
            input_context: "The server is up",
            transcript: "and running smoothly",
            expected: "and running smoothly.",
            category: "conjunction_mid",
        },
        TestCase {
            input_context: "I read the spec",
            transcript: "and it looks good to me",
            expected: "and it looks good to me.",
            category: "conjunction_mid",
        },

        // ===== CATEGORY: Em-dash / interruption =====
        TestCase {
            input_context: "The main reason —",
            transcript: "well actually there are several reasons",
            expected: "well, actually there are several reasons.",
            category: "after_dash",
        },
        TestCase {
            input_context: "I was going to say —",
            transcript: "never mind it doesn't matter",
            expected: "never mind, it doesn't matter.",
            category: "after_dash",
        },

        // ===== CATEGORY: Email / formal =====
        TestCase {
            input_context: "Hi team,",
            transcript: "i wanted to share an update on the project",
            expected: "I wanted to share an update on the project.",
            category: "email",
        },
        TestCase {
            input_context: "Dear Mr. Johnson,",
            transcript: "thank you for your prompt response",
            expected: "Thank you for your prompt response.",
            category: "email",
        },
        TestCase {
            input_context: "Best regards,",
            transcript: "alexander",
            expected: "Alexander",
            category: "email",
        },
        TestCase {
            input_context: "Hi Sarah,\n\nJust a quick note.",
            transcript: "the deadline has been moved to next week",
            expected: "The deadline has been moved to next week.",
            category: "email",
        },
        TestCase {
            input_context: "Thanks for the update.\n\n",
            transcript: "i have a few follow up questions",
            expected: "I have a few follow-up questions.",
            category: "email",
        },
    ]
}

/// Build the base prompt template (same as what the pipeline uses)
fn get_base_prompt(config: &Config) -> String {
    config.get_prompt_for_context(None)
}

/// Approach A: Prompt-based — add smart continuation hint in context section
fn build_prompt_approach_a(base_prompt: &str, input_context: &str, transcript: &str, config: &Config) -> String {
    let mut prompt = base_prompt.to_string();

    if !input_context.is_empty() {
        // Inject context hint before ## Input
        let hint = format!(
            "\n[INPUT CONTEXT]\n\
             Text already in the input field (before the cursor):\n\
             {}\n\
             You are CONTINUING from this text. Capitalization rule:\n\
             - If the existing text ends with sentence-ending punctuation (. ? ! or newline), capitalize your first word.\n\
             - If the existing text ends mid-sentence (no final punctuation, or ends with , ; : —), do NOT capitalize your first word (unless it is a proper noun like a name, place, or brand).\n\
             Normal capitalization rules apply to all OTHER words in your output.\n",
            input_context
        );

        if let Some(pos) = prompt.find("## Input") {
            prompt.insert_str(pos, &hint);
        } else {
            prompt.push_str(&hint);
        }
    }

    // Replace {transcript} and {personal_dictionary}
    format_prompt(&prompt, transcript, config)
}

/// Approach B: Structural — concatenate input text with transcript in the ## Input section
fn build_prompt_approach_b(base_prompt: &str, input_context: &str, transcript: &str, config: &Config) -> String {
    let mut prompt = base_prompt.to_string();

    if !input_context.is_empty() {
        // Replace the {transcript} placeholder with a combined format
        let combined = format!(
            "[EXISTING TEXT]{}\n[NEW DICTATION — format only this part]{}",
            input_context, transcript
        );
        prompt = prompt.replace("{transcript}", &combined);

        // Inject a brief note before ## Input explaining the format
        let note = "\nThe transcript below has two parts: [EXISTING TEXT] is already typed (do not output it), \
                     and [NEW DICTATION] is what the user just said (output only this part, properly formatted). \
                     Continue naturally from the existing text — match its casing flow.\n\n";

        if let Some(pos) = prompt.find("## Input") {
            prompt.insert_str(pos, note);
        }

        // Handle personal dictionary
        if !config.personal_dictionary.is_empty() {
            let dict_str = config.personal_dictionary.join(", ");
            prompt = prompt.replace(
                "{personal_dictionary}",
                &format!("\nPersonal vocabulary: {}", dict_str),
            );
        } else {
            prompt = prompt.replace("{personal_dictionary}", "");
        }

        if !config.llm_options.enable_thinking {
            if let Some(pos) = prompt.find("## Input") {
                prompt.insert_str(pos, "/no_think\n\n");
            }
        }

        return prompt;
    }

    // No input context — standard formatting
    format_prompt(&prompt, transcript, config)
}

/// Approach C (baseline): No input context provided at all (current behavior without the feature)
fn build_prompt_baseline(base_prompt: &str, transcript: &str, config: &Config) -> String {
    format_prompt(base_prompt, transcript, config)
}

/// Calculate word-level accuracy between hypothesis and reference
fn word_accuracy(hypothesis: &str, reference: &str) -> (f32, usize, Vec<String>) {
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();
    let ref_words: Vec<&str> = reference.split_whitespace().collect();

    if ref_words.is_empty() && hyp_words.is_empty() {
        return (100.0, 0, vec![]);
    }
    if ref_words.is_empty() {
        return (0.0, hyp_words.len(), vec![format!("added {} words", hyp_words.len())]);
    }

    // Simple word error rate using Levenshtein on words
    let n = ref_words.len();
    let m = hyp_words.len();
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = if normalize_word(ref_words[i - 1]) == normalize_word(hyp_words[j - 1]) {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1) // deletion
                .min(dp[i][j - 1] + 1)     // insertion
                .min(dp[i - 1][j - 1] + cost); // substitution
        }
    }

    let errors = dp[n][m];
    let wer = errors as f32 / n.max(1) as f32;
    let accuracy = ((1.0 - wer) * 100.0).max(0.0);

    // Collect specific differences
    let mut diffs = vec![];
    let mut i = n;
    let mut j = m;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && normalize_word(ref_words[i - 1]) == normalize_word(hyp_words[j - 1]) {
            i -= 1;
            j -= 1;
        } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
            diffs.push(format!("'{}' → '{}'", ref_words[i - 1], hyp_words[j - 1]));
            i -= 1;
            j -= 1;
        } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            diffs.push(format!("+'{}'", hyp_words[j - 1]));
            j -= 1;
        } else if i > 0 {
            diffs.push(format!("-'{}'", ref_words[i - 1]));
            i -= 1;
        } else {
            break;
        }
    }
    diffs.reverse();

    (accuracy, errors, diffs)
}

/// Normalize a word for comparison (strip punctuation from edges, lowercase)
fn normalize_word(word: &str) -> String {
    word.trim_matches(|c: char| c.is_ascii_punctuation())
        .to_lowercase()
}

/// Check if the first word capitalization is correct
fn first_word_cap_correct(hypothesis: &str, reference: &str) -> Option<bool> {
    let hyp_first = hypothesis.split_whitespace().next()?;
    let ref_first = reference.split_whitespace().next()?;

    // Compare just the first character's case
    let hyp_cap = hyp_first.chars().next()?.is_uppercase();
    let ref_cap = ref_first.chars().next()?.is_uppercase();

    Some(hyp_cap == ref_cap)
}

/// Parse CLI args to get model selection
fn parse_model_arg() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--model" {
            if let Some(m) = args.get(i + 1) {
                return Some(m.clone());
            }
        }
    }
    None
}

/// Get the list of models to test
fn get_models_to_test(filter: Option<&str>) -> Vec<LlmModel> {
    let all_models = vec![
        LlmModel::Qwen3_1_7B,
        LlmModel::Qwen3_4B,
        LlmModel::SmolLM3_3B,
        LlmModel::Phi4Mini,
        LlmModel::Gemma3nE2B,
    ];

    if let Some(name) = filter {
        match name.to_lowercase().as_str() {
            "qwen3-1.7b" | "qwen3_1_7b" | "qwen3-1.7" => vec![LlmModel::Qwen3_1_7B],
            "qwen3-4b" | "qwen3_4b" => vec![LlmModel::Qwen3_4B],
            "smollm3" | "smollm3-3b" => vec![LlmModel::SmolLM3_3B],
            "phi4" | "phi4-mini" | "phi-4" => vec![LlmModel::Phi4Mini],
            "gemma3n" | "gemma-3n" | "gemma3n-e2b" => vec![LlmModel::Gemma3nE2B],
            "all" => all_models,
            _ => {
                eprintln!("Unknown model: {}. Valid: qwen3-1.7b, qwen3-4b, smollm3, phi4, gemma3n, all", name);
                std::process::exit(1);
            }
        }
    } else {
        // Default: just the current model
        vec![LlmModel::Qwen3_1_7B]
    }
}

/// Run evaluation for a single model, returns (model_name, accuracy, wer, first_word_pct, errors, avg_ms, failures)
fn run_eval_for_model(model: &LlmModel) -> Option<ModelResult> {
    let mut config = Config::default();
    config.llm_model = model.clone();

    // Check if model file exists
    let model_path = match config.llm_model_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  Skipping {} — cannot resolve path: {}", model.display_name(), e);
            return None;
        }
    };
    if !model_path.exists() {
        eprintln!("  Skipping {} — model file not found at {:?}", model.display_name(), model_path);
        return None;
    }

    println!("\n  Loading {} ({})...", model.display_name(), model.filename());
    let llm = match LlmEngine::new(&config) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("  Skipping {} — failed to load: {}", model.display_name(), e);
            return None;
        }
    };

    let base_prompt = get_base_prompt(&config);
    let cases = test_cases();
    let total = cases.len();

    let mut total_accuracy: f32 = 0.0;
    let mut total_errors: usize = 0;
    let mut total_words: usize = 0;
    let mut first_word_correct: usize = 0;
    let mut first_word_total: usize = 0;
    let mut total_ms: u128 = 0;
    let mut category_accuracy: std::collections::HashMap<String, (f32, usize)> = std::collections::HashMap::new();
    let mut failures: Vec<(usize, String, String, String, String)> = vec![];

    for (i, case) in cases.iter().enumerate() {
        print!("\r  [{:3}/{}] {:<25}", i + 1, total, case.category);

        // Only test Approach A (the winner) for multi-model comparison
        let prompt = build_prompt_approach_a(&base_prompt, case.input_context, case.transcript, &config);

        let start = Instant::now();
        let output = match llm.format(case.transcript, &prompt) {
            Ok(text) => text.trim().to_string(),
            Err(e) => {
                eprintln!("\n    LLM error on case {}: {}", i + 1, e);
                continue;
            }
        };
        let elapsed = start.elapsed().as_millis();

        let output = output
            .trim_matches('"')
            .trim_matches('`')
            .trim()
            .to_string();

        let (accuracy, errors, diffs) = word_accuracy(&output, case.expected);
        let ref_words = case.expected.split_whitespace().count();

        total_accuracy += accuracy;
        total_errors += errors;
        total_words += ref_words;
        total_ms += elapsed;

        if let Some(correct) = first_word_cap_correct(&output, case.expected) {
            first_word_total += 1;
            if correct {
                first_word_correct += 1;
            }
        }

        let cat_entry = category_accuracy
            .entry(case.category.to_string())
            .or_insert((0.0, 0));
        cat_entry.0 += accuracy;
        cat_entry.1 += 1;

        if errors > 0 {
            failures.push((
                i + 1,
                case.category.to_string(),
                case.expected.to_string(),
                output.clone(),
                diffs.join(", "),
            ));
        }
    }
    println!("\r  {}", " ".repeat(60));

    Some(ModelResult {
        name: model.display_name().to_string(),
        total,
        total_accuracy,
        total_errors,
        total_words,
        first_word_correct,
        first_word_total,
        total_ms,
        category_accuracy,
        failures,
    })
}

struct ModelResult {
    name: String,
    total: usize,
    total_accuracy: f32,
    total_errors: usize,
    total_words: usize,
    first_word_correct: usize,
    first_word_total: usize,
    total_ms: u128,
    category_accuracy: std::collections::HashMap<String, (f32, usize)>,
    failures: Vec<(usize, String, String, String, String)>,
}

fn main() {
    let model_arg = parse_model_arg();
    let models = get_models_to_test(model_arg.as_deref());
    let multi_model = models.len() > 1;

    println!("═══════════════════════════════════════════════════════════════");
    if multi_model {
        println!("  VoiceFlow Multi-Model Evaluation (Approach A only)          ");
    } else {
        println!("  VoiceFlow Input Context Continuation — Approach Evaluation  ");
    }
    println!("═══════════════════════════════════════════════════════════════\n");

    if multi_model {
        println!("Models to test: {}", models.iter().map(|m| m.display_name()).collect::<Vec<_>>().join(", "));
    }

    // If single model and no --model flag, run the original 3-approach comparison
    if !multi_model {
        let mut config = Config::default();
        config.llm_model = models[0].clone();

        println!("Loading LLM engine...");
        let llm = match LlmEngine::new(&config) {
            Ok(engine) => engine,
            Err(e) => {
                eprintln!("Failed to load LLM: {}", e);
                eprintln!("Make sure models are downloaded: cargo run -p voiceflow-cli -- setup");
                std::process::exit(1);
            }
        };
        println!("LLM loaded: {}\n", config.llm_model.display_name());

        let base_prompt = get_base_prompt(&config);
        let cases = test_cases();
        let total = cases.len();

        println!("Running {} test cases across 3 approaches...\n", total);

        struct ApproachResults {
            name: String,
            total_accuracy: f32,
            total_errors: usize,
            total_words: usize,
            first_word_correct: usize,
            first_word_total: usize,
            category_accuracy: std::collections::HashMap<String, (f32, usize)>,
            failures: Vec<(usize, String, String, String, String)>,
            total_ms: u128,
        }

        let mut results = vec![
            ApproachResults { name: "A: Prompt-based".into(), total_accuracy: 0.0, total_errors: 0, total_words: 0, first_word_correct: 0, first_word_total: 0, category_accuracy: std::collections::HashMap::new(), failures: vec![], total_ms: 0 },
            ApproachResults { name: "B: Structural".into(), total_accuracy: 0.0, total_errors: 0, total_words: 0, first_word_correct: 0, first_word_total: 0, category_accuracy: std::collections::HashMap::new(), failures: vec![], total_ms: 0 },
            ApproachResults { name: "C: Baseline (no ctx)".into(), total_accuracy: 0.0, total_errors: 0, total_words: 0, first_word_correct: 0, first_word_total: 0, category_accuracy: std::collections::HashMap::new(), failures: vec![], total_ms: 0 },
        ];

        for (i, case) in cases.iter().enumerate() {
            let ctx_display = if case.input_context.is_empty() {
                "(empty)".to_string()
            } else if case.input_context.len() > 40 {
                format!("\"{}...\"", &case.input_context[..37])
            } else {
                format!("\"{}\"", case.input_context)
            };

            print!("\r[{:3}/{}] {} | {} ", i + 1, total, case.category, ctx_display);

            let prompts = [
                build_prompt_approach_a(&base_prompt, case.input_context, case.transcript, &config),
                build_prompt_approach_b(&base_prompt, case.input_context, case.transcript, &config),
                build_prompt_baseline(&base_prompt, case.transcript, &config),
            ];

            for (approach_idx, prompt) in prompts.iter().enumerate() {
                let start = Instant::now();
                let output = match llm.format(case.transcript, prompt) {
                    Ok(text) => text.trim().to_string(),
                    Err(e) => { eprintln!("\n  LLM error on case {}: {}", i + 1, e); continue; }
                };
                let elapsed = start.elapsed().as_millis();

                let output = output.trim_matches('"').trim_matches('`').trim().to_string();
                let (accuracy, errors, diffs) = word_accuracy(&output, case.expected);
                let ref_words = case.expected.split_whitespace().count();

                let r = &mut results[approach_idx];
                r.total_accuracy += accuracy;
                r.total_errors += errors;
                r.total_words += ref_words;
                r.total_ms += elapsed;

                if let Some(correct) = first_word_cap_correct(&output, case.expected) {
                    r.first_word_total += 1;
                    if correct { r.first_word_correct += 1; }
                }

                let cat_entry = r.category_accuracy.entry(case.category.to_string()).or_insert((0.0, 0));
                cat_entry.0 += accuracy;
                cat_entry.1 += 1;

                if errors > 0 {
                    r.failures.push((i + 1, case.category.to_string(), case.expected.to_string(), output.clone(), diffs.join(", ")));
                }
            }
        }

        println!("\r{}\n", " ".repeat(80));

        println!("═══════════════════════════════════════════════════════════════");
        println!("                        SUMMARY                               ");
        println!("═══════════════════════════════════════════════════════════════\n");

        println!("{:<25} {:>10} {:>10} {:>12} {:>10} {:>10}", "Approach", "Accuracy", "WER", "1st-Word Cap", "Errors", "Avg ms");
        println!("{}", "─".repeat(80));

        for r in &results {
            let avg_accuracy = r.total_accuracy / total as f32;
            let wer = if r.total_words > 0 { r.total_errors as f32 / r.total_words as f32 * 100.0 } else { 0.0 };
            let first_word_pct = if r.first_word_total > 0 { r.first_word_correct as f32 / r.first_word_total as f32 * 100.0 } else { 0.0 };
            let avg_ms = r.total_ms / total as u128;
            println!("{:<25} {:>9.1}% {:>9.1}% {:>11.1}% {:>10} {:>8}ms", r.name, avg_accuracy, wer, first_word_pct, r.total_errors, avg_ms);
        }

        // Per-category breakdown
        println!("\n\n═══════════════════════════════════════════════════════════════");
        println!("                   PER-CATEGORY BREAKDOWN                      ");
        println!("═══════════════════════════════════════════════════════════════\n");

        let mut categories: Vec<String> = results[0].category_accuracy.keys().cloned().collect();
        categories.sort();

        println!("{:<25} {:>15} {:>15} {:>15}", "Category", "A: Prompt", "B: Structural", "C: Baseline");
        println!("{}", "─".repeat(72));

        for cat in &categories {
            let scores: Vec<String> = results.iter().map(|r| {
                if let Some((acc, count)) = r.category_accuracy.get(cat) { format!("{:.1}%", acc / *count as f32) } else { "N/A".to_string() }
            }).collect();
            println!("{:<25} {:>15} {:>15} {:>15}", cat, scores[0], scores[1], scores[2]);
        }

        // First-word capitalization
        println!("\n\n═══════════════════════════════════════════════════════════════");
        println!("              FIRST-WORD CAPITALIZATION ACCURACY                ");
        println!("═══════════════════════════════════════════════════════════════\n");

        println!("{:<25} {:>15} {:>15} {:>15}", "Approach", "Correct", "Total", "Accuracy");
        println!("{}", "─".repeat(72));
        for r in &results {
            let pct = if r.first_word_total > 0 { r.first_word_correct as f32 / r.first_word_total as f32 * 100.0 } else { 0.0 };
            println!("{:<25} {:>15} {:>15} {:>14.1}%", r.name, r.first_word_correct, r.first_word_total, pct);
        }

        // Show failures
        for r in &results {
            if r.failures.is_empty() { continue; }
            println!("\n\n═══════════════════════════════════════════════════════════════");
            println!("  FAILURES: {}  ({} errors)", r.name, r.failures.len());
            println!("═══════════════════════════════════════════════════════════════\n");
            for (idx, cat, expected, got, diffs) in &r.failures {
                println!("  #{} [{}]", idx, cat);
                println!("    Expected: {}", expected);
                println!("    Got:      {}", got);
                println!("    Diffs:    {}", diffs);
                println!();
            }
        }

        println!("\nDone. {} test cases × 3 approaches = {} LLM calls.", total, total * 3);
        return;
    }

    // ===== Multi-model mode: test Approach A across all models =====
    let mut all_results: Vec<ModelResult> = vec![];

    for model in &models {
        println!("\n───────────────────────────────────────────────────────────────");
        println!("  Testing: {} ({})", model.display_name(), model.filename());
        println!("───────────────────────────────────────────────────────────────");

        if let Some(result) = run_eval_for_model(model) {
            all_results.push(result);
        }
    }

    // Print multi-model comparison
    println!("\n\n═══════════════════════════════════════════════════════════════════════════════");
    println!("                    MULTI-MODEL COMPARISON (Approach A)                        ");
    println!("═══════════════════════════════════════════════════════════════════════════════\n");

    println!("{:<22} {:>10} {:>10} {:>12} {:>8} {:>8}", "Model", "Accuracy", "WER", "1st-Word", "Errors", "Avg ms");
    println!("{}", "─".repeat(75));

    for r in &all_results {
        let avg_accuracy = r.total_accuracy / r.total as f32;
        let wer = if r.total_words > 0 { r.total_errors as f32 / r.total_words as f32 * 100.0 } else { 0.0 };
        let first_word_pct = if r.first_word_total > 0 { r.first_word_correct as f32 / r.first_word_total as f32 * 100.0 } else { 0.0 };
        let avg_ms = r.total_ms / r.total as u128;
        println!("{:<22} {:>9.1}% {:>9.1}% {:>11.1}% {:>8} {:>6}ms", r.name, avg_accuracy, wer, first_word_pct, r.total_errors, avg_ms);
    }

    // Per-category breakdown for each model
    println!("\n\n═══════════════════════════════════════════════════════════════════════════════");
    println!("                      PER-CATEGORY BREAKDOWN                                   ");
    println!("═══════════════════════════════════════════════════════════════════════════════\n");

    if let Some(first) = all_results.first() {
        let mut categories: Vec<String> = first.category_accuracy.keys().cloned().collect();
        categories.sort();

        // Header
        print!("{:<22}", "Category");
        for r in &all_results {
            let short_name: String = r.name.chars().take(14).collect();
            print!(" {:>14}", short_name);
        }
        println!();
        println!("{}", "─".repeat(22 + 15 * all_results.len()));

        for cat in &categories {
            print!("{:<22}", cat);
            for r in &all_results {
                if let Some((acc, count)) = r.category_accuracy.get(cat) {
                    print!(" {:>13.1}%", acc / *count as f32);
                } else {
                    print!(" {:>14}", "N/A");
                }
            }
            println!();
        }
    }

    // Show failures per model
    for r in &all_results {
        if r.failures.is_empty() { continue; }
        println!("\n\n═══════════════════════════════════════════════════════════════");
        println!("  FAILURES: {} ({} errors)", r.name, r.failures.len());
        println!("═══════════════════════════════════════════════════════════════\n");
        for (idx, cat, expected, got, diffs) in &r.failures {
            println!("  #{} [{}]", idx, cat);
            println!("    Expected: {}", expected);
            println!("    Got:      {}", got);
            println!("    Diffs:    {}", diffs);
            println!();
        }
    }

    let total_cases: usize = all_results.iter().map(|r| r.total).sum();
    println!("\nDone. {} models × {} test cases = {} LLM calls.", all_results.len(), all_results.first().map(|r| r.total).unwrap_or(0), total_cases);
}
