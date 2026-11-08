import SwiftUI

/// Final onboarding playground, shown after the setup check and before the
/// main window. Everything is live: dictation pastes into this field through
/// the normal pipeline (the setup check has ended, so no diagnostics gating).
/// Nothing here is scored or required; it is a place to play, then continue.
struct TryItOutStepView: View {
    @ObservedObject private var hotkeys = HotkeyStore.shared
    let onContinue: () -> Void

    @State private var text = ""
    @FocusState private var editorFocused: Bool

    private func key(_ action: HotkeyAction) -> String {
        HotkeyFormatter.display(hotkeys.binding(for: action))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: VF.Space.l) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Try it out")
                    .font(.system(size: 24, weight: .bold))
                Text("Everything is live. Click the page below and talk, just as you will in any app.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            editor

            VStack(spacing: VF.Space.s) {
                suggestion(
                    number: 1, key: key(.dictate), title: "Dictate",
                    say: "Um, so I think we should, uh, meet on Thursday at three thirty.",
                    result: "Fillers disappear; punctuation and the time are cleaned up.",
                    done: !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                suggestion(
                    number: 2, key: key(.edit), title: "Fix it by voice",
                    say: "Actually, make it Friday.",
                    result: "Hold within 10 seconds of a dictation to revise it in place.",
                    done: false)
                suggestion(
                    number: 3, key: key(.cycleFormatting), title: "Change the style",
                    say: nil,
                    result: "Tap to cycle Minimal → Moderate → Intent-Aware → Aggressive, then dictate again.",
                    done: false)
            }

            HStack {
                Button("Clear") { text = ""; editorFocused = true }
                    .buttonStyle(VFGhostPillStyle())
                    .disabled(text.isEmpty)
                Spacer()
                Button(action: onContinue) {
                    Text("Continue").frame(minWidth: 140)
                }
                .keyboardShortcut(.defaultAction)
                .controlSize(.large)
                .vfPrimaryButton()
            }
        }
        .padding(.horizontal, 40)
        .padding(.vertical, 24)
        .onAppear {
            // Make the wizard key so the first dictation lands here.
            NSApp.activate(ignoringOtherApps: true)
            DispatchQueue.main.async { editorFocused = true }
        }
    }

    private var editor: some View {
        ZStack(alignment: .topLeading) {
            TextEditor(text: $text)
                .font(VF.Font.transcript)
                .scrollContentBackground(.hidden)
                .focused($editorFocused)
                .padding(VF.Space.m)
            if text.isEmpty {
                Text("Hold \(key(.dictate)) and start talking…")
                    .font(VF.Font.transcript)
                    .foregroundStyle(.tertiary)
                    .padding(.horizontal, VF.Space.m + 5)
                    .padding(.vertical, VF.Space.m)
                    .allowsHitTesting(false)
            }
        }
        .frame(minHeight: 150, maxHeight: .infinity)
        .background(
            RoundedRectangle(cornerRadius: VF.Radius.card, style: .continuous).fill(VF.surface))
        .overlay(
            RoundedRectangle(cornerRadius: VF.Radius.card, style: .continuous)
                .strokeBorder(editorFocused ? VF.accent.opacity(0.6) : VF.surfaceStroke, lineWidth: 1))
        .onTapGesture { editorFocused = true }
    }

    private func suggestion(number: Int, key: String, title: String, say: String?, result: String, done: Bool) -> some View {
        HStack(alignment: .top, spacing: VF.Space.m) {
            ZStack {
                Circle().fill(done ? VF.Palette.success.opacity(0.2) : VF.surface).frame(width: 24, height: 24)
                if done {
                    Image(systemName: "checkmark").font(.system(size: 11, weight: .bold)).foregroundStyle(VF.Palette.success)
                } else {
                    Text("\(number)").font(.system(size: 11, weight: .semibold)).foregroundStyle(.secondary)
                }
            }
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(title).font(VF.Font.heading)
                    VFKeycap(key: key)
                }
                if let say {
                    Text("Say “\(say)”").font(.callout).foregroundStyle(.primary.opacity(0.85))
                }
                Text(result).font(VF.Font.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(VF.Space.m)
        .background(RoundedRectangle(cornerRadius: VF.Radius.field, style: .continuous).fill(VF.surface.opacity(0.6)))
    }
}
