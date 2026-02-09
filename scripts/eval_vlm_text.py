#!/usr/bin/env python3
"""Evaluate VLM models on text-only dictation formatting.

Runs the same test cases as eval_context.rs but using Python/transformers
for safetensors-based VLM models (Qwen3-VL, Jina-VLM).

Usage:
    .venv/bin/python3 scripts/eval_vlm_text.py --model qwen3-vl
    .venv/bin/python3 scripts/eval_vlm_text.py --model jina-vlm
    .venv/bin/python3 scripts/eval_vlm_text.py --model all
"""

import argparse
import os
import sys
import time
from pathlib import Path

MODELS_DIR = Path.home() / "Library/Application Support/com.era-laboratories.voiceflow/models"
PROMPT_FILE = Path(__file__).parent.parent / "prompts/default.txt"

# Same test cases as eval_context.rs (Approach A only — the winner)
TEST_CASES = [
    # (input_context, transcript, expected, category)
    # Fresh field
    ("", "hello world", "Hello world", "fresh"),
    ("", "this is a test of the dictation system", "This is a test of the dictation system.", "fresh"),
    ("", "can you send me the report by friday", "Can you send me the report by Friday?", "fresh"),
    ("", "the meeting is at three thirty pm tomorrow", "The meeting is at 3:30 PM tomorrow.", "fresh"),
    ("", "please call doctor smith at his office", "Please call Dr. Smith at his office.", "fresh"),
    ("", "i need to buy milk eggs and bread", "I need to buy milk, eggs, and bread.", "fresh"),
    ("", "what time does the store close", "What time does the store close?", "fresh"),
    ("", "the quick brown fox jumps over the lazy dog", "The quick brown fox jumps over the lazy dog.", "fresh"),
    ("", "we should discuss this at the next team standup", "We should discuss this at the next team standup.", "fresh"),
    ("", "i think the API endpoint needs to be updated", "I think the API endpoint needs to be updated.", "fresh"),
    # After period
    ("The first task is done.", "now we need to move on to the second one", "Now we need to move on to the second one.", "after_period"),
    ("I finished the report.", "please review it when you get a chance", "Please review it when you get a chance.", "after_period"),
    ("That sounds great.", "let me know if you need anything else", "Let me know if you need anything else.", "after_period"),
    ("Done.", "what should we work on next", "What should we work on next?", "after_period"),
    ("The build passed.", "we can deploy to production now", "We can deploy to production now.", "after_period"),
    ("I agree with the plan.", "however we should also consider the budget", "However, we should also consider the budget.", "after_period"),
    ("Meeting ended early.", "everyone was aligned on the approach", "Everyone was aligned on the approach.", "after_period"),
    ("The PR was merged.", "there were no conflicts", "There were no conflicts.", "after_period"),
    ("That's correct.", "the server is running on port eight thousand", "The server is running on port 8000.", "after_period"),
    ("Sounds good.", "i will send the invite shortly", "I will send the invite shortly.", "after_period"),
    # After question
    ("Did you finish the task?", "yes i completed it this morning", "Yes, I completed it this morning.", "after_question"),
    ("What do you think?", "i think we should go with option two", "I think we should go with option two.", "after_question"),
    ("Are we ready to ship?", "not yet we still need to fix the login bug", "Not yet, we still need to fix the login bug.", "after_question"),
    ("Can you check on that?", "sure i will look into it right away", "Sure, I will look into it right away.", "after_question"),
    ("When is the deadline?", "it is due next friday", "It is due next Friday.", "after_question"),
    # Mid-sentence
    ("I was thinking we could", "maybe try a different approach to this problem", "maybe try a different approach to this problem.", "mid_sentence"),
    ("The reason I asked is", "because the current implementation has a bug", "because the current implementation has a bug.", "mid_sentence"),
    ("We should probably", "update the documentation before the release", "update the documentation before the release.", "mid_sentence"),
    ("The main issue is that", "the database queries are too slow", "the database queries are too slow.", "mid_sentence"),
    ("I need to", "finish this before the end of the day", "finish this before the end of the day.", "mid_sentence"),
    ("Let me explain why", "this matters for the project timeline", "this matters for the project timeline.", "mid_sentence"),
    ("He said that", "the deployment would be delayed by a week", "the deployment would be delayed by a week.", "mid_sentence"),
    ("The problem with", "the current setup is that it doesn't scale", "the current setup is that it doesn't scale.", "mid_sentence"),
    ("I wanted to mention that", "we have a new team member starting monday", "we have a new team member starting Monday.", "mid_sentence"),
    ("Please make sure to", "test everything before pushing to main", "test everything before pushing to main.", "mid_sentence"),
    # After comma
    ("First,", "we need to set up the development environment", "we need to set up the development environment.", "after_comma"),
    ("In addition,", "we should add more unit tests", "we should add more unit tests.", "after_comma"),
    ("On the other hand,", "the existing solution works fine for now", "the existing solution works fine for now.", "after_comma"),
    ("If that's the case,", "then we should postpone the migration", "then we should postpone the migration.", "after_comma"),
    ("As I mentioned before,", "the server needs to be restarted", "the server needs to be restarted.", "after_comma"),
    # After colon
    ("Here's the plan:", "we deploy on tuesday and monitor for a week", "we deploy on Tuesday and monitor for a week.", "after_colon"),
    ("Action items:", "update the docs fix the bug and write tests", "update the docs, fix the bug, and write tests.", "after_colon"),
    ("Note:", "this only applies to the staging environment", "this only applies to the staging environment.", "after_colon"),
    # Proper noun mid
    ("I was talking to", "john about the project and he agreed", "John about the project and he agreed.", "proper_noun_mid"),
    ("We should deploy to", "amazon web services using the new pipeline", "Amazon Web Services using the new pipeline.", "proper_noun_mid"),
    ("The ticket was assigned to", "sarah from the backend team", "Sarah from the backend team.", "proper_noun_mid"),
    ("Can you send it to", "michael and jennifer on the design team", "Michael and Jennifer on the design team.", "proper_noun_mid"),
    ("This is similar to how", "google handles their search indexing", "Google handles their search indexing.", "proper_noun_mid"),
    # After exclamation
    ("Great news!", "the feature is ready for testing", "The feature is ready for testing.", "after_exclamation"),
    ("Watch out!", "that endpoint is deprecated", "That endpoint is deprecated.", "after_exclamation"),
    ("Nice work!", "the performance improved by fifty percent", "The performance improved by 50%.", "after_exclamation"),
    # Multi-sentence
    ("I reviewed the PR. There are a few issues.", "the main one is the error handling in the auth module", "The main one is the error handling in the auth module.", "multi_sentence"),
    ("The build failed again. I checked the logs.", "it looks like a dependency conflict", "It looks like a dependency conflict.", "multi_sentence"),
    ("We had a good meeting. Everyone agreed on the direction. However,", "we still need to finalize the timeline", "we still need to finalize the timeline.", "multi_sentence"),
    ("The migration is complete. All data was transferred. Next,", "we need to verify the integrity of the records", "we need to verify the integrity of the records.", "multi_sentence"),
    ("Step one is done. Step two is in progress. For step three,", "we will need the credentials from the ops team", "we will need the credentials from the ops team.", "multi_sentence"),
    # Technical
    ("", "the API returns a JSON response with the user ID", "The API returns a JSON response with the user ID.", "technical"),
    ("For the OAuth flow,", "we need to configure the redirect URL in the dashboard", "we need to configure the redirect URL in the dashboard.", "technical"),
    ("", "run npm install and then npm run build", "Run npm install and then npm run build.", "technical"),
    ("The CI pipeline", "uses github actions to run the tests automatically", "uses GitHub Actions to run the tests automatically.", "technical"),
    ("", "we should switch from REST to GraphQL for this endpoint", "We should switch from REST to GraphQL for this endpoint.", "technical"),
    # Numbers
    ("", "the budget is fifty thousand dollars", "The budget is $50,000.", "numbers"),
    ("The meeting is scheduled for", "january fifteenth at two thirty pm", "January 15 at 2:30 PM.", "numbers"),
    ("", "we have about twenty five percent capacity remaining", "We have about 25% capacity remaining.", "numbers"),
    ("There are", "three hundred and forty two items in the queue", "342 items in the queue.", "numbers"),
    ("Call me at", "five five five one two three four five six seven", "555-123-4567.", "numbers"),
    # Short
    ("", "yes", "Yes", "short"),
    ("", "sounds good", "Sounds good.", "short"),
    ("", "thank you", "Thank you.", "short"),
    ("", "no problem", "No problem.", "short"),
    ("", "got it", "Got it.", "short"),
    # Short continuation
    ("I think we should", "wait", "wait.", "short_continuation"),
    ("That's", "correct", "correct.", "short_continuation"),
    ("The answer is", "no", "no.", "short_continuation"),
    ("Please", "stop", "stop.", "short_continuation"),
    ("It was", "fine actually", "fine, actually.", "short_continuation"),
    # Conjunction after period
    ("The first option didn't work.", "and the second one was even worse", "And the second one was even worse.", "conjunction_after_period"),
    ("I tried everything.", "but nothing seemed to fix the issue", "But nothing seemed to fix the issue.", "conjunction_after_period"),
    ("We can go with plan A.", "or we could try a completely different approach", "Or we could try a completely different approach.", "conjunction_after_period"),
    ("That's not ideal.", "however it is the best we can do right now", "However, it is the best we can do right now.", "conjunction_after_period"),
    # Conjunction mid
    ("I finished the frontend", "and now i am working on the backend", "and now I am working on the backend.", "conjunction_mid"),
    ("The tests passed", "but we still need code review", "but we still need code review.", "conjunction_mid"),
    ("We can use postgres", "or switch to mongodb if needed", "or switch to MongoDB if needed.", "conjunction_mid"),
    ("The server is up", "and running smoothly", "and running smoothly.", "conjunction_mid"),
    ("I read the spec", "and it looks good to me", "and it looks good to me.", "conjunction_mid"),
    # After dash
    ("The main reason \u2014", "well actually there are several reasons", "well, actually there are several reasons.", "after_dash"),
    ("I was going to say \u2014", "never mind it doesn't matter", "never mind, it doesn't matter.", "after_dash"),
    # Email
    ("Hi team,", "i wanted to share an update on the project", "I wanted to share an update on the project.", "email"),
    ("Dear Mr. Johnson,", "thank you for your prompt response", "Thank you for your prompt response.", "email"),
    ("Best regards,", "alexander", "Alexander", "email"),
    ("Hi Sarah,\n\nJust a quick note.", "the deadline has been moved to next week", "The deadline has been moved to next week.", "email"),
    ("Thanks for the update.\n\n", "i have a few follow up questions", "I have a few follow-up questions.", "email"),
]


def load_prompt_template():
    with open(PROMPT_FILE) as f:
        return f.read()


def build_approach_a_prompt(base_prompt, input_context, transcript):
    """Build Approach A prompt (same as Rust eval)."""
    prompt = base_prompt

    if input_context:
        hint = f"""
[INPUT CONTEXT]
Text already in the input field (before the cursor):
{input_context}
You are CONTINUING from this text. Capitalization rule:
- If the existing text ends with sentence-ending punctuation (. ? ! or newline), capitalize your first word.
- If the existing text ends mid-sentence (no final punctuation, or ends with , ; : \u2014), do NOT capitalize your first word (unless it is a proper noun like a name, place, or brand).
Normal capitalization rules apply to all OTHER words in your output.
Do not repeat or rewrite the existing text.
"""
        if "## Input" in prompt:
            pos = prompt.index("## Input")
            prompt = prompt[:pos] + hint + "\n" + prompt[pos:]
        else:
            prompt += hint

    # Insert /no_think before ## Input
    if "## Input" in prompt:
        pos = prompt.index("## Input")
        prompt = prompt[:pos] + "/no_think\n\n" + prompt[pos:]

    prompt = prompt.replace("{transcript}", transcript)
    prompt = prompt.replace("{personal_dictionary}", "")
    return prompt


def word_accuracy(hypothesis, reference):
    """Word-level accuracy using Levenshtein distance on words."""
    hyp_words = hypothesis.split()
    ref_words = reference.split()

    if not ref_words and not hyp_words:
        return 100.0, 0, []
    if not ref_words:
        return 0.0, len(hyp_words), [f"+'{w}'" for w in hyp_words]

    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if normalize_word(ref_words[i-1]) == normalize_word(hyp_words[j-1]) else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)

    errors = dp[n][m]
    wer = errors / max(n, 1)
    accuracy = max(0.0, (1.0 - wer) * 100.0)

    # Backtrace for diffs
    diffs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and normalize_word(ref_words[i-1]) == normalize_word(hyp_words[j-1]):
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            diffs.append(f"'{ref_words[i-1]}' -> '{hyp_words[j-1]}'")
            i -= 1; j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            diffs.append(f"+'{hyp_words[j-1]}'")
            j -= 1
        elif i > 0:
            diffs.append(f"-'{ref_words[i-1]}'")
            i -= 1
        else:
            break
    diffs.reverse()
    return accuracy, errors, diffs


def normalize_word(w):
    return w.strip(".,!?;:\"'`()[]{}").lower()


def first_word_cap_correct(hyp, ref):
    hw = hyp.split()
    rw = ref.split()
    if not hw or not rw:
        return None
    return hw[0][0].isupper() == rw[0][0].isupper()


def run_eval(model_name, model_path, base_prompt):
    """Run eval for a model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

    print(f"\n  Loading {model_name} from {model_path}...")
    start_load = time.time()

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

        # Load model based on architecture
        if "qwen3-vl" in str(model_path).lower() or "qwen3_vl" in str(model_path).lower():
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path),
                dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            # Jina-VLM and other custom architectures
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        model.eval()
    except Exception as e:
        print(f"  Failed to load {model_name}: {e}")
        import traceback; traceback.print_exc()
        return None

    load_time = time.time() - start_load
    print(f"  Loaded in {load_time:.1f}s")

    total = len(TEST_CASES)
    total_accuracy = 0.0
    total_errors = 0
    total_words = 0
    first_word_correct = 0
    first_word_total = 0
    total_ms = 0
    category_accuracy = {}
    failures = []

    for i, (input_ctx, transcript, expected, category) in enumerate(TEST_CASES):
        print(f"\r  [{i+1:3}/{total}] {category:<25}", end="", flush=True)

        prompt = build_approach_a_prompt(base_prompt, input_ctx, transcript)

        start = time.time()
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = outputs[0][input_len:]
            output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        except Exception as e:
            print(f"\n    Error on case {i+1}: {e}")
            continue

        elapsed_ms = int((time.time() - start) * 1000)

        # Strip quotes/backticks
        output = output.strip('"').strip('`').strip()
        # Take only first line (model might generate extra)
        if '\n' in output:
            output = output.split('\n')[0].strip()

        accuracy, errors, diffs = word_accuracy(output, expected)
        ref_words = len(expected.split())

        total_accuracy += accuracy
        total_errors += errors
        total_words += ref_words
        total_ms += elapsed_ms

        cap_correct = first_word_cap_correct(output, expected)
        if cap_correct is not None:
            first_word_total += 1
            if cap_correct:
                first_word_correct += 1

        if category not in category_accuracy:
            category_accuracy[category] = [0.0, 0]
        category_accuracy[category][0] += accuracy
        category_accuracy[category][1] += 1

        if errors > 0:
            failures.append((i + 1, category, expected, output, ", ".join(diffs)))

    print(f"\r  {'':60}")

    # Clean up GPU memory
    del model
    del tokenizer
    import gc; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return {
        "name": model_name,
        "total": total,
        "total_accuracy": total_accuracy,
        "total_errors": total_errors,
        "total_words": total_words,
        "first_word_correct": first_word_correct,
        "first_word_total": first_word_total,
        "total_ms": total_ms,
        "category_accuracy": category_accuracy,
        "failures": failures,
    }


def print_results(results):
    """Print comparison table."""
    print("\n" + "=" * 75)
    print("              VLM TEXT-ONLY EVALUATION (Approach A)                ")
    print("=" * 75 + "\n")

    print(f"{'Model':<25} {'Accuracy':>10} {'WER':>10} {'1st-Word':>10} {'Errors':>8} {'Avg ms':>8}")
    print("-" * 75)

    for r in results:
        avg_acc = r["total_accuracy"] / r["total"]
        wer = r["total_errors"] / max(r["total_words"], 1) * 100
        fw_pct = r["first_word_correct"] / max(r["first_word_total"], 1) * 100
        avg_ms = r["total_ms"] // r["total"]
        print(f"{r['name']:<25} {avg_acc:>9.1f}% {wer:>9.1f}% {fw_pct:>9.1f}% {r['total_errors']:>8} {avg_ms:>6}ms")

    # Per-category
    print("\n" + "=" * 75)
    print("                    PER-CATEGORY BREAKDOWN                        ")
    print("=" * 75 + "\n")

    all_cats = sorted(set(c for r in results for c in r["category_accuracy"]))
    header = f"{'Category':<22}"
    for r in results:
        header += f" {r['name'][:14]:>14}"
    print(header)
    print("-" * (22 + 15 * len(results)))

    for cat in all_cats:
        row = f"{cat:<22}"
        for r in results:
            if cat in r["category_accuracy"]:
                acc, cnt = r["category_accuracy"][cat]
                row += f" {acc/cnt:>13.1f}%"
            else:
                row += f" {'N/A':>14}"
        print(row)

    # Failures
    for r in results:
        if not r["failures"]:
            continue
        print(f"\n{'=' * 75}")
        print(f"  FAILURES: {r['name']} ({len(r['failures'])} errors)")
        print(f"{'=' * 75}\n")
        for idx, cat, expected, got, diffs in r["failures"]:
            print(f"  #{idx} [{cat}]")
            print(f"    Expected: {expected}")
            print(f"    Got:      {got}")
            print(f"    Diffs:    {diffs}")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", help="qwen3-vl, jina-vlm, or all")
    args = parser.parse_args()

    base_prompt = load_prompt_template()
    results = []

    models_to_test = []
    if args.model in ("qwen3-vl", "all"):
        models_to_test.append(("Qwen3-VL 2B", MODELS_DIR / "qwen3-vl-2b-instruct"))
    if args.model in ("jina-vlm", "all"):
        models_to_test.append(("Jina-VLM 4B", MODELS_DIR / "jina-vlm"))

    if not models_to_test:
        print(f"Unknown model: {args.model}. Use qwen3-vl, jina-vlm, or all")
        sys.exit(1)

    for name, path in models_to_test:
        if not path.exists():
            print(f"Skipping {name} - not found at {path}")
            continue
        result = run_eval(name, path, base_prompt)
        if result:
            results.append(result)

    if results:
        print_results(results)
    else:
        print("No models completed successfully.")


if __name__ == "__main__":
    main()
