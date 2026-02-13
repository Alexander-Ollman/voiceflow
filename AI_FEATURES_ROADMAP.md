# AI Features Roadmap

Voice-triggered AI features powered by local LLM and VLM, accessible from any app via the existing push-to-talk interface.

---

## Tier 1: Text Field AI Commands

Natural extensions of "summarize this" — read from and write to the active text field using the existing overlay pill for status.

### Reply to This
Say **"reply to this"** followed by your intent (e.g., *"reply to this saying I agree but let's push to next week"*). The VLM reads the message on screen, the LLM generates a contextual reply matching the conversation's tone. A pop-up modal shows the draft — Enter to paste, Escape to discard.

### Rewrite This As...
Say **"make this more formal"**, **"make this concise"**, **"rewrite as bullet points"**, **"make this friendlier"**, etc. Reads the current field content, transforms it, and replaces in-place. Overlay pill shows "Rewriting..." during LLM generation.

### Proofread This
Say **"proofread this"**. Reads the field, fixes grammar/spelling/punctuation, and pastes the corrected version back. The pop-up modal variant shows a diff (original vs. corrected) so you can review changes before accepting.

### Continue This
Say **"continue this"**. The LLM reads the existing field content and extends it in the same voice/style. Pop-up modal shows the continuation — Enter to append.

---

## Tier 2: Pop-up Modal as Primary Surface

Features that don't require a text field — the pop-up modal is the UI.

### Draft From Voice
Say **"draft an email to Sarah about rescheduling Thursday's standup to Friday"**. Pure generation from a voice prompt. Pop-up modal shows the generated text. Copy, paste, or dismiss.

### What Does This Say?
Say **"what does this say?"** while looking at a dense PDF, screenshot, wall of text, or foreign-language content. VLM captures the screen, LLM produces a digestible summary in the pop-up. Works on anything visible — no text field needed.

### Explain This
Say **"explain this"** while looking at code, an error message, a config file, or a chart. VLM captures the screen, LLM explains it in plain language in the pop-up.

### Quick Question Mode
Say **"what's 15% of 347?"**, **"how do I undo a git rebase?"**, **"what's the Swift syntax for guard let?"**. Pop-up shows the answer. A local, private, voice-triggered Spotlight replacement — no browser, no context switch.

---

## Tier 3: VLM + Voice Compound Features

Features that exploit the combination of screen vision and voice input.

### Extract From Screen
Say **"extract the phone numbers from this"**, **"pull the action items from this"**, **"get the URLs from this page"**. VLM reads the screen, LLM extracts structured data, pop-up shows results with one-click copy.

### Fill This In
Say **"fill this in"** while looking at a form, then dictate field values. VLM identifies form fields, LLM maps dictated info to the right fields. Targets simple web forms, PDFs, and applications.

### Compare With Clipboard
Say **"compare this with what I copied"**. VLM reads the screen, clipboard provides the other text, pop-up shows a comparison or diff. Useful for code review, document versions, and side-by-side evaluation.

---

## Shared Infrastructure

### Pop-up Modal
A floating results panel used by features that produce output (replies, drafts, explanations, answers):

- Appears center-screen, floating above all windows
- Enter pastes into the active field and dismisses
- Cmd+C copies to clipboard without pasting
- Escape dismisses
- The overlay pill transitions into the modal when the LLM finishes

### AI Command Detection
Voice commands are detected after transcription, similar to the existing "summarize this" pattern. Commands are parsed to extract:
- The action (reply, rewrite, proofread, continue, explain, draft, extract, etc.)
- Any parameters (tone, recipient, topic, question, etc.)

### Overlay Pill States
The existing `aiProcessing(String)` state handles status for all AI features:
- "Reading screen..." (VLM capture)
- "Generating reply..." / "Rewriting..." / "Proofreading..." / etc.
- Transitions to pop-up modal or hides after paste
