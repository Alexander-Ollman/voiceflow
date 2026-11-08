---
title: Onboarding and Restart Troubleshooting
icon: 🛠️
order: 32
tags: [review, onboarding, diagnostics, reliability]
---

# Onboarding and restart troubleshooting

This extends draft PR #9 with the additional tester feedback. It remains a
review build; the reported failures on the other user's Mac have not been
reproduced, and their logs were not available.

## User-visible changes

- A required **Test your setup** step follows downloads. It initializes the real
  runtime while displaying readiness instead of waiting invisibly on the model
  page. The same check is available under **Settings → Diagnostics**.
- A dedicated native text field, live microphone meter and six checkpoints show
  shortcut receipt, capture start, audio signal, recognition, formatting and
  verified field insertion. Continue requires an actual Dictate event and a
  matching change in the focused practice field. A saved/registered shortcut or
  a sent Cmd+V event alone never passes the check.
- **Test without shortcut** uses the same capture/transcription/format/paste path
  while bypassing hotkey delivery. If it succeeds and the shortcut does not,
  inspect conflicting shortcuts in other apps, including ChatGPT.
- Live permission state is separate from model readiness. Registration failures
  are shown; a successfully registered shortcut is explicitly not a guarantee
  that another app cannot intercept it.
- Change the shortcut, open Logs, restart speech recognition, reset the check, or
  copy a checkpoint summary from this screen. The summary contains status and
  failure details without test transcript or audio. Existing application logs
  may still contain transcripts; this change does not alter their privacy policy.
- VoiceFlow appears in the Dock and app switcher by default, with an opt-out in
  Startup settings. This makes it easier to reopen, pin in the Dock, and find in
  Force Quit. A fallback application menu supplies Open and Quit when SwiftUI has
  not installed an app menu. Wizard completion honors the visibility preference.

The native setup-check preview is [available here](./setup-check-preview.png).
Its readiness labels were supplied for rendering; it is not evidence of a live
end-to-end dictation on the other user's machine.

## Concrete failure paths addressed

The earlier Parakeet shutdown sent terminate and then called `waitUntilExit()`
on the main actor without a deadline. A Python daemon busy in native inference,
or still waiting on a client socket, could prevent that return. Llama shutdown
also used synchronous waits. AppKit now delays termination explicitly, yields
while owned helpers exit, and escalates TERM-resistant processes to SIGKILL
within a bounded interval. Rust cleanup runs in the background with a separate
completion deadline, and native initialization no longer runs on the main actor.
Restart waits for the old app PID to disappear before reopening the bundle.

Parakeet health/control probes now use a one-second socket budget, transcription
uses 30 seconds, and first model preload retains its 300-second budget. A
nonresponsive owned daemon is not replaced by spawning another copy. Model-load
attempt identities prevent shutdown/restart from accepting stale completion.
Unexpected exits surface an engine error rather than an indefinitely Ready label.
Sockets suppress SIGPIPE, and daemon logs are created/appended before spawning.

Helpers launched by the combined branch now have the existing process-identity
watchdog used in the local development checkout, so an abnormal app exit can
also retire its own children. It does not kill arbitrary third-party processes.

This Mac's log also contained a very-short-capture Metal allocation failure.
Both the Swift client and daemon now reject clips shorter than 200 ms before
STFT/model inference. That protects the short-buffer path; it does not prove the
root cause of every Metal error or the other user's report.

During a setup test, output is authorized only for its practice field. Closing
an in-flight check suppresses late test output. The ordinary application's
verified-target safety remains owned by the existing runtime in the local
migration; this draft does not import that entire migration into `main`.

## Validation and remaining work

- Release Swift builds for both the combined PR branch and the local migration.
- 24 checks against production diagnostics, process-shutdown and socket code:
  field verification, hotkey versus button-only checks, silence, failed stages,
  privacy of summaries, late-output suppression, TERM-resistant process cleanup,
  main-actor responsiveness and a real stalled Unix socket with a short deadline.
- Existing 20 PR lifecycle checks pass.
- Two Python tests, including four short-length subcases, verify that invalid
  short clips do not enter Metal/STFT and a missing model stays an error.
- The watchdog test confirms child cleanup after a simulated parent exit.
- Native SwiftUI rendering was visually inspected; shell and whitespace checks
  pass. Existing compiler warnings remain.

Physical shortcut capture, live TCC permission changes, real microphone input,
Dock/Force Quit behavior, and a full installed-bundle restart still need manual
validation. No new model weights, native Rust rebuild, daemon repackaging,
signing, installation or release is performed in this review. A shipped build
must regenerate the frozen daemon because its Python source changed.
