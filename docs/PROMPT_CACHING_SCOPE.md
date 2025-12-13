# Prompt Caching Implementation Scope

## Problem Statement

Currently, the LLM engine formats a new prompt on every transcription request. Since context-specific prompts (email, slack, code, default) are largely static, we can cache the tokenized prompts to reduce latency.

## Current Flow

```
User speaks → Audio → Whisper → Raw transcript → Format prompt (string concat) → LLM tokenizes → LLM generates → Output
```

The "Format prompt" and "LLM tokenizes" steps happen on every request, even when using the same context.

## Proposed Solution

### Phase 1: Prompt Template Caching

Cache the formatted prompt templates (before transcript insertion):

```rust
pub struct PromptCache {
    // Pre-formatted templates by context
    templates: HashMap<String, String>,
    // Timestamp of last cache refresh
    last_refresh: Instant,
    // TTL for cache entries
    ttl: Duration,
}

impl PromptCache {
    pub fn get_or_format(&mut self, context: &str, config: &Config) -> &str {
        // Return cached template or format and cache
    }

    pub fn invalidate(&mut self) {
        // Clear cache when config changes
    }
}
```

**Expected Improvement**: Minor - saves string allocation (~1-5ms)

### Phase 2: Tokenized Prompt Caching (Requires mistral.rs support)

Cache the tokenized system prompt portion:

```rust
pub struct TokenizedPromptCache {
    // Pre-tokenized system portions by context
    system_tokens: HashMap<String, Vec<u32>>,
}
```

**Expected Improvement**: Significant - saves tokenization (~50-100ms on first token)

**Dependency**: Requires mistral.rs to expose tokenization API separately from inference.

### Phase 3: KV Cache Persistence (Advanced)

If mistral.rs supports it, persist the KV cache for the system prompt prefix:

- Pre-compute attention for system prompt
- Store KV states
- Restore on each request

**Expected Improvement**: Major - saves forward pass on system tokens (~200-500ms)

**Dependency**: Requires mistral.rs KV cache export/import API.

## Implementation Priority

| Phase | Effort | Impact | Recommendation |
|-------|--------|--------|----------------|
| Phase 1 | Low | Low | Implement now |
| Phase 2 | Medium | Medium | Wait for mistral.rs API |
| Phase 3 | High | High | Future consideration |

## Phase 1 Implementation Details

### Files to Modify

1. `crates/voiceflow-core/src/llm/prompts.rs` - Add PromptCache struct
2. `crates/voiceflow-core/src/llm/engine.rs` - Use cached prompts
3. `crates/voiceflow-core/src/pipeline.rs` - Wire up cache lifecycle

### API Changes

```rust
// In LlmEngine
pub fn set_prompt_cache(&mut self, cache: Arc<Mutex<PromptCache>>);

// In Pipeline
pub fn preload_prompts(&mut self) -> Result<()>;
pub fn invalidate_prompt_cache(&mut self);
```

### Cache Invalidation Triggers

- Config reload
- Manual `invalidate_prompt_cache()` call
- TTL expiration (default: 5 minutes)

## Metrics to Track

- `prompt_format_ms` - Time to format prompt
- `prompt_cache_hits` - Number of cache hits
- `prompt_cache_misses` - Number of cache misses

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Stale prompts after config change | Invalidate on config reload |
| Memory growth with many contexts | LRU eviction policy |
| Thread safety | Use `Arc<Mutex<PromptCache>>` |

## Estimated Implementation Time

- Phase 1: 2-4 hours
- Phase 2: Dependent on mistral.rs
- Phase 3: Dependent on mistral.rs
