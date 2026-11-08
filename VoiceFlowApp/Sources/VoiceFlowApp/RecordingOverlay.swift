import SwiftUI

// MARK: - Overlay State

class OverlayState: ObservableObject {
    enum RecordingState {
        case idle
        case recording
        case streaming(String)      // partial transcript text from real-time STT
        case processing
        case formatting(String)     // LLM tokens appearing word-by-word
        case aiProcessing(String)   // label text like "Summarizing..."
        case succeeded              // text landed: brief green ripple, then hide
        case failed                 // work ended without output: brief amber ripple

        var isWorking: Bool {
            switch self {
            case .processing, .formatting, .aiProcessing: return true
            default: return false
            }
        }

        var isRecording: Bool {
            if case .recording = self { return true }
            if case .streaming = self { return true }
            return false
        }

        var isProcessing: Bool {
            if case .processing = self { return true }
            if case .formatting = self { return true }
            return false
        }

        var isStreaming: Bool {
            if case .streaming = self { return true }
            return false
        }
    }

    let waveform = RecordingWaveformHistory()
    @Published var state: RecordingState = .idle {
        didSet {
            if state.isRecording && !oldValue.isRecording { waveform.reset() }
        }
    }
    @Published var audioLevel: Float = 0 {
        didSet { waveform.append(audioLevel) }
    }
}

// MARK: - Recording Overlay View

/// The voice overlay: one Liquid Glass object that listens, then reads back.
///
/// - Listening: a glass capsule holding a fluid waveform driven by the mic.
/// - Live text: the same object grows into a card; text above, waveform below.
/// - Working (recognizing / formatting / AI command): the waveform settles into
///   a slow shimmer with a short label. No spinners, no colour flips.
struct RecordingOverlayView: View {
    @ObservedObject var state: OverlayState

    static let cardWidth: CGFloat = 440
    static let maxTextHeight: CGFloat = 280
    /// Transparent margin around the glass so its shadow and morph never clip.
    static let envelopeMargin: CGFloat = 24

    private var text: String {
        switch state.state {
        case .streaming(let t), .formatting(let t): return t
        default: return ""
        }
    }

    private var isCard: Bool { !text.isEmpty }

    private var label: String? {
        switch state.state {
        // Recognition and formatting need no words: the waveform's working
        // motion says it. Only AI commands name what they're doing.
        case .aiProcessing(let l): return l.trimmingCharacters(in: CharacterSet(charactersIn: ". …"))
        default: return nil
        }
    }

    private var waveMode: LiquidWaveformView.Mode {
        switch state.state {
        case .recording, .streaming: return .listening
        case .succeeded: return .success
        case .failed: return .failure
        default: return .thinking
        }
    }

    /// The waveform is a fixed small object in every state.
    static let waveSize = CGSize(width: 110, height: 18)

    /// Animate the morph on layout changes only, never on every token.
    private var layoutKey: String { "\(isCard)-\(label ?? "")" }

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 0)
            glassObject
        }
        .padding(Self.envelopeMargin)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
        .environment(\.colorScheme, .dark)
        .animation(VF.Motion.morph, value: layoutKey)
    }

    private var glassObject: some View {
        let shape = RoundedRectangle(cornerRadius: isCard ? VF.Radius.panel : 17.5, style: .continuous)
        return VStack(spacing: 0) {
            if isCard {
                transcript
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
            footer
        }
        .frame(width: isCard ? Self.cardWidth : nil)
        .fixedSize(horizontal: !isCard, vertical: false)
        .vfGlass(in: shape, tint: Color.black.opacity(isCard ? 0.55 : 0.30))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityText)
    }

    /// Newest words stay pinned to the bottom; older lines slide up and fade
    /// out under a soft top edge. No scroll view, no scrolling animation.
    private var transcript: some View {
        BottomClampLayout(maxHeight: Self.maxTextHeight) {
            Text(text)
                .font(VF.Font.transcript)
                .foregroundStyle(.white.opacity(0.95))
                .lineSpacing(3)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        // Overflow rises into the top padding, which is the fade zone, so the
        // fade only ever touches text that has actually scrolled away.
        .padding(.top, 18)
        .clipped()
        .mask(
            VStack(spacing: 0) {
                LinearGradient(colors: [.clear, .black], startPoint: .top, endPoint: .bottom).frame(height: 18)
                Color.black
            }
        )
        .padding(.horizontal, 20)
        .padding(.bottom, 2)
    }

    private var footer: some View {
        HStack(spacing: 8) {
            if let label {
                Image(systemName: "sparkles")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(VF.glowGradient)
                Text(label)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.white.opacity(0.75))
                    .lineLimit(1)
                    .fixedSize()
                    .transition(.opacity)
            }
            LiquidWaveformView(history: state.waveform, mode: waveMode)
                .frame(width: Self.waveSize.width, height: Self.waveSize.height)
        }
        .padding(.horizontal, 17)
        .frame(maxWidth: isCard ? .infinity : nil)
        .frame(height: isCard ? 40 : 35)
    }

    private var accessibilityText: String {
        switch state.state {
        case .idle: return ""
        case .recording: return "VoiceFlow is listening"
        case .streaming(let t): return t.isEmpty ? "VoiceFlow is listening" : "Listening: \(t)"
        case .processing: return "Working"
        case .formatting(let t): return t
        case .aiProcessing(let l): return l
        case .succeeded: return "Done"
        case .failed: return "Nothing was inserted"
        }
    }
}

/// Sizes to its child's natural height up to `maxHeight`, and pins the child's
/// bottom edge to its own, so overflow leaves through the top (to be clipped).
struct BottomClampLayout: Layout {
    let maxHeight: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        guard let child = subviews.first else { return .zero }
        let natural = child.sizeThatFits(ProposedViewSize(width: proposal.width, height: nil))
        return CGSize(width: proposal.width ?? natural.width, height: min(natural.height, maxHeight))
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        guard let child = subviews.first else { return }
        let natural = child.sizeThatFits(ProposedViewSize(width: bounds.width, height: nil))
        child.place(at: CGPoint(x: bounds.minX, y: bounds.maxY),
                    anchor: .bottomLeading,
                    proposal: ProposedViewSize(width: bounds.width, height: natural.height))
    }
}
