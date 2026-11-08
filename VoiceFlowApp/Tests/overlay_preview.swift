import AppKit
import SwiftUI

/// Live preview of the voice overlay over whatever is on screen.
/// Loops: listen → release (flare, glide) → done (green); live text →
/// formatting → nothing inserted (amber); AI command → done.
/// Pass --palette glacier|lagoon|aurora|ember|mono to pick a swatch.
/// Pass --mic to drive the waveform from the real microphone instead of a
/// synthetic speech envelope. Quit with Ctrl-C.
@MainActor final class OverlayPreview {
    let state = OverlayState()
    var panel: NSPanel!
    var tick = 0
    let useMic = CommandLine.arguments.contains("--mic")
    let recorder = AVAudioLevelTap()
    let words = "Hey team, quick update on the release. The onboarding check is merged and the new overlay is ready for review, so please try it on your machines today and tell me what feels off".split(separator: " ")

    func start() {
        if let i = CommandLine.arguments.firstIndex(of: "--palette"), i + 1 < CommandLine.arguments.count {
            UserDefaults.standard.set(CommandLine.arguments[i + 1], forKey: WaveformPalette.preferenceKey)
        }
        let screen = NSScreen.main!.visibleFrame
        let size = NSSize(width: RecordingOverlayView.cardWidth + RecordingOverlayView.envelopeMargin * 2,
                          height: RecordingOverlayView.maxTextHeight + 46 + 32 + RecordingOverlayView.envelopeMargin * 2)
        panel = NSPanel(contentRect: NSRect(x: screen.midX - size.width / 2, y: screen.minY + 80 - RecordingOverlayView.envelopeMargin,
                                            width: size.width, height: size.height),
                        styleMask: [.nonactivatingPanel, .fullSizeContentView], backing: .buffered, defer: false)
        panel.level = .floating
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = false
        panel.ignoresMouseEvents = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.contentView = NSHostingView(rootView: RecordingOverlayView(state: state))
        panel.orderFrontRegardless()
        if useMic { recorder.start() }
        // ~47 Hz, matching AudioRecorder.meterLevels (4 slices per 4096-frame
        // tap at 48 kHz), so the preview moves exactly as the app does.
        Timer.scheduledTimer(withTimeInterval: 1.0 / 47.0, repeats: true) { _ in
            MainActor.assumeIsolated { self.step() }
        }
    }

    func syntheticLevel(_ t: Double) -> Float {
        let phrase = sin(t * 0.9) > -0.6 ? 1.0 : 0.05
        let syllable = max(0, sin(t * 11.0)) * (0.6 + 0.4 * sin(t * 2.3))
        return Float(0.02 + 0.22 * phrase * syllable + Double.random(in: 0...0.015))
    }

    func step() {
        tick += 1
        let t = Double(tick % 752) / 47.0   // 16s loop
        let level = useMic ? recorder.level : syntheticLevel(Double(tick) / 47.0)
        switch t {
        case ..<3: state.state = .recording; state.audioLevel = level           // speak
        case ..<4.6: state.state = .processing                                  // release → flare → glide
        case ..<5.2: state.state = .succeeded                                   // pasted → green ripple
        case ..<6.2: state.state = .idle; panel.orderOut(nil); return
        case ..<10:
            let n = min(words.count, Int((t - 6.2) * 5))
            state.state = .streaming(words.prefix(n).joined(separator: " ")); state.audioLevel = level
        case ..<12: state.state = .formatting("Hey team, quick update on the release: the onboarding check is merged, and the new overlay is ready for review.")
        case ..<12.6: state.state = .failed                                     // nothing inserted → amber
        case ..<13.6: state.state = .idle; panel.orderOut(nil); return
        case ..<15.4: state.state = .aiProcessing("Summarizing...")
        default: state.state = .succeeded
        }
        if !panel.isVisible { panel.orderFrontRegardless() }
    }
}

import AVFoundation

/// Minimal mic RMS tap, same scaling as AudioRecorder (rms * 10, clamped).
final class AVAudioLevelTap {
    private let engine = AVAudioEngine()
    private(set) var level: Float = 0
    func start() {
        let input = engine.inputNode
        input.installTap(onBus: 0, bufferSize: 1024, format: input.outputFormat(forBus: 0)) { [weak self] buffer, _ in
            guard let data = buffer.floatChannelData?[0] else { return }
            let n = Int(buffer.frameLength)
            var sum: Float = 0
            for i in 0..<n { sum += data[i] * data[i] }
            self?.level = min(1, sqrt(sum / Float(max(n, 1))) * 10)
        }
        try? engine.start()
    }
}

@main enum OverlayPreviewMain {
    static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        let preview = MainActor.assumeIsolated { OverlayPreview() }
        MainActor.assumeIsolated { preview.start() }
        app.run()
    }
}
