import SwiftUI
import AppKit

// MARK: - Quick Tips carousel
//
// Five looping, vector-only "how to use VoiceFlow" animations shown on the
// final onboarding step and replayable from Settings. Every visual is driven
// off a single `TimelineView(.animation)` clock and pure math — no GIF/MP4
// assets, so it stays crisp on Retina and adds ~0 KB to the bundle.
//
// The user advances manually (‹ / › / dots). The animation *within* each card
// loops continuously while it's on screen. Pass `autoAdvance: true` (used during
// the model-download wait) to also cycle between cards automatically.

enum QuickTipKind: Int, CaseIterable, Identifiable {
    case holdSpace, correct, menuBar, cleanup, adapt
    var id: Int { rawValue }

    var title: String {
        switch self {
        case .holdSpace: return "Hold ⌥ Space to talk"
        case .correct:   return "Fix it by voice"
        case .menuBar:   return "Lives in your menu bar"
        case .cleanup:   return "It cleans up as you speak"
        case .adapt:     return "Fits the app you're in"
        }
    }

    var caption: String {
        switch self {
        case .holdSpace: return "Hold ⌥ Space, speak, then release — VoiceFlow types it for you."
        case .correct:   return "Just dictated? Hold ⌥⇧Space and say the change — it rewrites the last paste."
        case .menuBar:   return "Find VoiceFlow's icon up top. Click it for settings & insights."
        case .cleanup:   return "Ums, repeats and stumbles are removed. Punctuation added for you."
        case .adapt:     return "The tone adapts per app — casual in chat, polished in email."
        }
    }
}

/// Shared accent used across the playful bits. Falls back gracefully in light mode.
private enum Tip {
    static let accent = Color(red: 0.40, green: 0.46, blue: 1.0)
    static let accent2 = Color(red: 0.62, green: 0.40, blue: 1.0)
    static let gradient = LinearGradient(
        colors: [accent, accent2],
        startPoint: .leading, endPoint: .trailing
    )
    /// 0..1 position within a repeating period for the given date.
    static func cycle(_ date: Date, period: Double) -> Double {
        let s = date.timeIntervalSinceReferenceDate
        return (s.truncatingRemainder(dividingBy: period)) / period
    }
    /// Smoothstep ease for 0..1.
    static func ease(_ x: Double) -> Double {
        let c = min(max(x, 0), 1)
        return c * c * (3 - 2 * c)
    }
    /// Maps `x` in [a,b] to 0..1 (clamped), 0 outside.
    static func ramp(_ x: Double, _ a: Double, _ b: Double) -> Double {
        guard b > a else { return x >= b ? 1 : 0 }
        return min(max((x - a) / (b - a), 0), 1)
    }
}

struct QuickTipsCarousel: View {
    /// When true, the carousel cycles through tips on its own (used during the
    /// model-download wait so the user passively reads them). Manual ‹ / › / dots
    /// still work and the auto-timer just picks up from wherever they left off.
    var autoAdvance: Bool = false

    @State private var index = 0
    private let kinds = QuickTipKind.allCases
    private let autoTimer = Timer.publish(every: 4.5, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 14) {
            GeometryReader { geo in
                let w = geo.size.width
                HStack(spacing: 0) {
                    ForEach(kinds) { kind in
                        QuickTipCard(kind: kind).frame(width: w)
                    }
                }
                .offset(x: -CGFloat(index) * w)
                .animation(.spring(response: 0.5, dampingFraction: 0.86), value: index)
            }
            .frame(height: 258)
            .clipped()

            controls
        }
        .onReceive(autoTimer) { _ in
            guard autoAdvance else { return }
            withAnimation(.spring(response: 0.5, dampingFraction: 0.86)) {
                index = (index + 1) % kinds.count
            }
        }
    }

    private var controls: some View {
        HStack(spacing: 26) {
            Button { step(-1) } label: {
                Image(systemName: "chevron.left")
                    .font(.system(size: 20, weight: .semibold))
            }
            .buttonStyle(.plain)
            .disabled(index == 0)
            .opacity(index == 0 ? 0.3 : 1)

            HStack(spacing: 11) {
                ForEach(kinds) { kind in
                    Capsule()
                        .fill(kind.rawValue == index ? AnyShapeStyle(Tip.gradient)
                                                      : AnyShapeStyle(Color.secondary.opacity(0.3)))
                        .frame(width: kind.rawValue == index ? 28 : 10, height: 10)
                        .animation(.spring(response: 0.4, dampingFraction: 0.8), value: index)
                        .onTapGesture { withAnimation { index = kind.rawValue } }
                }
            }

            Button { step(1) } label: {
                Image(systemName: "chevron.right")
                    .font(.system(size: 20, weight: .semibold))
            }
            .buttonStyle(.plain)
            .disabled(index == kinds.count - 1)
            .opacity(index == kinds.count - 1 ? 0.3 : 1)
        }
        .foregroundColor(.secondary)
    }

    private func step(_ d: Int) {
        let next = index + d
        guard next >= 0, next < kinds.count else { return }
        withAnimation { index = next }
    }
}

// MARK: - One card (title + animated visual + caption)

struct QuickTipCard: View {
    let kind: QuickTipKind

    var body: some View {
        VStack(spacing: 12) {
            Text(kind.title)
                .font(.system(size: 22, weight: .bold))
                .foregroundColor(.primary)

            visual
                .frame(maxWidth: .infinity)
                .frame(height: 150)

            Text(kind.caption)
                .font(.system(size: 15))
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
                .frame(height: 44)
        }
        .padding(.horizontal, 44)
    }

    @ViewBuilder private var visual: some View {
        switch kind {
        case .holdSpace: HoldSpaceVisual()
        case .correct:   CorrectVisual()
        case .menuBar:   MenuBarVisual()
        case .cleanup:   CleanupVisual()
        case .adapt:     AdaptVisual()
        }
    }
}

// MARK: - Shared keycap

private struct KeyCap: View {
    let label: String
    let width: CGFloat
    let pressed: Bool
    var fontSize: CGFloat = 17

    var body: some View {
        Text(label)
            .font(.system(size: fontSize, weight: .medium, design: .rounded))
            .foregroundColor(pressed ? .white : .primary.opacity(0.8))
            .frame(width: width, height: 46)
            .background(
                RoundedRectangle(cornerRadius: 11, style: .continuous)
                    .fill(pressed ? AnyShapeStyle(Tip.gradient)
                                  : AnyShapeStyle(Color.primary.opacity(0.06)))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 11, style: .continuous)
                    .stroke(Color.primary.opacity(0.12), lineWidth: 1)
            )
            .shadow(color: pressed ? Tip.accent.opacity(0.55) : .clear, radius: 11, y: 3)
            .scaleEffect(pressed ? 0.96 : 1.0)
            .offset(y: pressed ? 3 : 0)
            .animation(.spring(response: 0.25, dampingFraction: 0.6), value: pressed)
    }
}

// MARK: - ① Hold ⌥ Space → paste

private struct HoldSpaceVisual: View {
    var body: some View {
        TimelineView(.animation) { tl in
            let p = Tip.cycle(tl.date, period: 7.2)
            let secs = tl.date.timeIntervalSinceReferenceDate
            HoldSpaceFrame(p: p, secs: secs)
        }
    }
}

private struct HoldSpaceFrame: View {
    let p: Double      // 0..1 through the loop
    let secs: Double   // absolute clock for the waveform

    private var pressed: Bool { p >= 0.12 && p < 0.62 }
    private var recording: Double { Tip.ramp(p, 0.20, 0.26) - Tip.ramp(p, 0.58, 0.62) }
    private var transcribing: Bool { p >= 0.62 && p < 0.74 }
    private var pasteReveal: Double { Tip.ease(Tip.ramp(p, 0.74, 0.92)) }
    private var fade: Double { 1 - Tip.ramp(p, 0.95, 1.0) }

    private var status: (String, Color) {
        if p < 0.12 { return ("Hold to talk", .secondary) }
        if recording > 0.3 { return ("● Listening", Tip.accent) }
        if transcribing { return ("Transcribing…", .secondary) }
        if pasteReveal > 0.1 { return ("✓ Pasted", .green) }
        return ("Hold to talk", .secondary)
    }

    var body: some View {
        VStack(spacing: 16) {
            Text(status.0)
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(status.1)
                .frame(height: 18)
                .animation(.easeInOut(duration: 0.2), value: status.0)

            ZStack {
                Waveform(secs: secs, level: recording)
                    .opacity(recording)

                if transcribing {
                    DotsBounce(secs: secs).transition(.opacity)
                }

                PasteField(reveal: pasteReveal)
                    .opacity(pasteReveal > 0.02 ? 1 : 0)
            }
            .frame(height: 52)

            HStack(spacing: 11) {
                KeyCap(label: "⌥", width: 56, pressed: pressed)
                KeyCap(label: "space", width: 190, pressed: pressed)
            }
        }
        .opacity(fade)
    }
}

private struct Waveform: View {
    let secs: Double
    let level: Double
    private let bars = 17

    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<bars, id: \.self) { i in
                let phase = secs * 4.5 + Double(i) * 0.7
                let h = (0.35 + 0.65 * (0.5 + 0.5 * sin(phase))) * 50 * max(level, 0.12)
                Capsule()
                    .fill(Tip.gradient)
                    .frame(width: 6, height: max(6, h))
            }
        }
    }
}

private struct DotsBounce: View {
    let secs: Double
    var body: some View {
        HStack(spacing: 9) {
            ForEach(0..<3, id: \.self) { i in
                let y = sin(secs * 4 + Double(i) * 0.9) * 4
                Circle()
                    .fill(Color.secondary)
                    .frame(width: 9, height: 9)
                    .offset(y: y)
            }
        }
    }
}

private struct PasteField: View {
    let reveal: Double
    private let text = "Send me the Q3 deck by Friday."

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "envelope.fill")
                .font(.system(size: 12))
                .foregroundColor(.secondary)
            Text(text)
                .font(.system(size: 15, weight: .medium))
                .foregroundColor(.primary)
                .lineLimit(1)
                .mask(alignment: .leading) {
                    GeometryReader { g in
                        Rectangle().frame(width: g.size.width * reveal)
                    }
                }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.primary.opacity(0.05))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Tip.accent.opacity(0.3), lineWidth: 1)
                )
        )
    }
}

// MARK: - ② Fix it by voice (⌥⇧Space correction)

private struct CorrectVisual: View {
    var body: some View {
        TimelineView(.animation) { tl in
            let p = Tip.cycle(tl.date, period: 8.0)
            CorrectFrame(p: p)
        }
    }
}

private struct CorrectFrame: View {
    let p: Double

    private var pressed: Bool { p >= 0.14 && p < 0.56 }
    private var instr: Double { Tip.ease(Tip.ramp(p, 0.20, 0.44)) }
    private var updating: Bool { p >= 0.56 && p < 0.66 }
    private var edit: Double { Tip.ease(Tip.ramp(p, 0.66, 0.84)) }
    private var highlight: Double { edit * (1 - Tip.ramp(p, 0.90, 1.0)) }
    private var fade: Double { 1 - Tip.ramp(p, 0.96, 1.0) }

    private var status: (String, Color) {
        if pressed { return ("● Listening", Tip.accent) }
        if updating { return ("Updating…", .secondary) }
        if edit > 0.1 { return ("✓ Updated", .green) }
        return ("Hold to fix the last paste", .secondary)
    }

    var body: some View {
        VStack(spacing: 13) {
            Text(status.0)
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(status.1)
                .frame(height: 18)
                .animation(.easeInOut(duration: 0.2), value: status.0)

            // The edited line: "Friday." → "Monday."
            HStack(spacing: 0) {
                Text("Send him the report by ")
                    .font(.system(size: 16))
                    .foregroundColor(.primary)
                ZStack(alignment: .leading) {
                    Text("Friday.")
                        .font(.system(size: 16))
                        .foregroundColor(.secondary)
                        .strikethrough(edit > 0.05, color: .secondary)
                        .opacity(1 - edit)
                    Text("Monday.")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.primary)
                        .padding(.horizontal, 4)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(Tip.accent.opacity(0.22 * highlight))
                        )
                        .opacity(edit)
                }
            }
            .frame(height: 22)

            // Spoken instruction chip (types in while listening)
            HStack(spacing: 8) {
                Image(systemName: "mic.fill")
                    .font(.system(size: 12))
                    .foregroundColor(Tip.accent)
                Text("change Friday to Monday")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundColor(.primary)
                    .lineLimit(1)
                    .mask(alignment: .leading) {
                        GeometryReader { g in
                            Rectangle().frame(width: g.size.width * instr)
                        }
                    }
            }
            .padding(.horizontal, 13)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Color.primary.opacity(0.05))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .stroke(Tip.accent.opacity(0.3), lineWidth: 1)
                    )
            )
            .opacity(instr > 0.02 ? 1 : 0)
            .frame(height: 32)

            // Keycaps ⌥ ⇧ space
            HStack(spacing: 9) {
                KeyCap(label: "⌥", width: 50, pressed: pressed)
                KeyCap(label: "⇧", width: 50, pressed: pressed)
                KeyCap(label: "space", width: 150, pressed: pressed)
            }
        }
        .opacity(fade)
    }
}

// MARK: - ③ Menu-bar pulse

private struct MenuBarVisual: View {
    var body: some View {
        TimelineView(.animation) { tl in
            let secs = tl.date.timeIntervalSinceReferenceDate
            ZStack(alignment: .trailing) {
                HStack(spacing: 0) {
                    Text("VoiceFlow   File   Edit   View")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(.secondary)
                        .padding(.leading, 16)
                    Spacer()
                    HStack(spacing: 16) {
                        glyph(secs: secs)
                        Image(systemName: "wifi").font(.system(size: 13))
                        Image(systemName: "battery.100").font(.system(size: 13))
                        Text("9:41").font(.system(size: 13, weight: .medium))
                    }
                    .foregroundColor(.secondary)
                    .padding(.trailing, 16)
                }
                .frame(height: 38)
                .background(
                    RoundedRectangle(cornerRadius: 9)
                        .fill(Color.primary.opacity(0.06))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 9)
                        .stroke(Color.primary.opacity(0.1), lineWidth: 1)
                )
            }
            .padding(.horizontal, 10)
        }
    }

    private func glyph(secs: Double) -> some View {
        ZStack {
            ForEach(0..<3, id: \.self) { i in
                let rp = ((secs / 3.4 + Double(i) / 3.0).truncatingRemainder(dividingBy: 1))
                Circle()
                    .stroke(Tip.accent, lineWidth: 2)
                    .frame(width: 22, height: 22)
                    .scaleEffect(0.5 + rp * 1.9)
                    .opacity((1 - rp) * 0.7)
            }
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 20))
                .foregroundStyle(Tip.gradient)
        }
        .frame(width: 22, height: 22)
    }
}

// MARK: - ④ Clean-up before/after

private struct CleanupVisual: View {
    var body: some View {
        TimelineView(.animation) { tl in
            let p = Tip.cycle(tl.date, period: 7.6)
            let reveal = Tip.ease(Tip.ramp(p, 0.30, 0.62))
            CleanupFrame(reveal: reveal)
        }
    }
}

private struct CleanupFrame: View {
    let reveal: Double
    private let messy = "um send him the the report uh by friday"
    private let clean = "Send him the report by Friday."

    var body: some View {
        VStack(spacing: 18) {
            HStack(spacing: 8) {
                Text("you said")
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Capsule().fill(Color.primary.opacity(0.06)))
                Text(messy)
                    .font(.system(size: 14))
                    .foregroundColor(.secondary)
                    .strikethrough(reveal > 0.5, color: .secondary.opacity(0.6))
            }
            .opacity(1 - reveal * 0.65)

            ZStack(alignment: .leading) {
                HStack(spacing: 8) {
                    Image(systemName: "sparkles")
                        .font(.system(size: 14))
                        .foregroundStyle(Tip.gradient)
                    Text(clean)
                        .font(.system(size: 18, weight: .semibold))
                        .foregroundColor(.primary)
                }
                .mask(alignment: .leading) {
                    GeometryReader { g in
                        Rectangle().frame(width: g.size.width * reveal)
                    }
                }

                GeometryReader { g in
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Tip.gradient)
                        .frame(width: 3.5, height: 28)
                        .opacity(reveal > 0.02 && reveal < 0.99 ? 0.9 : 0)
                        .offset(x: g.size.width * reveal - 1.5, y: 1)
                        .blendMode(.plusLighter)
                }
            }
            .frame(height: 30)
        }
    }
}

// MARK: - ⑤ Adapts per app

private struct AdaptVisual: View {
    var body: some View {
        TimelineView(.animation) { tl in
            let p = Tip.cycle(tl.date, period: 8.0)
            let toMail = Tip.ease(Tip.ramp(p, 0.42, 0.55)) - Tip.ease(Tip.ramp(p, 0.92, 1.0))
            ZStack {
                appChip(app: "Slack", icon: "message.fill",
                        tone: "casual", tint: Color(red: 0.36, green: 0.7, blue: 0.5),
                        text: "yeah sounds good — ship it 🚀")
                    .opacity(1 - toMail)
                    .scaleEffect(0.96 + 0.04 * (1 - toMail))

                appChip(app: "Mail", icon: "envelope.fill",
                        tone: "professional", tint: Tip.accent,
                        text: "That sounds good. Let's proceed.")
                    .opacity(toMail)
                    .scaleEffect(0.96 + 0.04 * toMail)
            }
        }
    }

    private func appChip(app: String, icon: String, tone: String, tint: Color, text: String) -> some View {
        VStack(spacing: 12) {
            HStack(spacing: 9) {
                Image(systemName: icon)
                    .font(.system(size: 15))
                    .foregroundColor(tint)
                Text(app)
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundColor(.primary)
                Spacer()
                Text(tone)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(tint)
                    .padding(.horizontal, 8).padding(.vertical, 3)
                    .background(Capsule().fill(tint.opacity(0.15)))
            }
            Text(text)
                .font(.system(size: 16))
                .foregroundColor(.primary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(16)
        .frame(width: 380)
        .background(
            RoundedRectangle(cornerRadius: 13, style: .continuous)
                .fill(Color.primary.opacity(0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 13, style: .continuous)
                .stroke(tint.opacity(0.3), lineWidth: 1)
        )
    }
}

// MARK: - Replay sheet (Settings)

struct QuickTipsSheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 24) {
            HStack {
                Text("How to use VoiceFlow")
                    .font(.system(size: 20, weight: .bold))
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.plain)
                    .foregroundColor(.secondary)
                    .font(.system(size: 15))
            }
            QuickTipsCarousel()
            Spacer(minLength: 0)
        }
        .padding(32)
        .frame(width: 760, height: 520)
        .background(VisualEffectBackground(material: .hudWindow, blending: .behindWindow).ignoresSafeArea())
    }
}
