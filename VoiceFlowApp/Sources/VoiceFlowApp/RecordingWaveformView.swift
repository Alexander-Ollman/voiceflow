import Combine
import SwiftUI

/// A bounded microphone history. Rendering is driven by incoming audio values;
/// there is no idle timer, synthetic motion, or per-bar animation task.
final class RecordingWaveformHistory: ObservableObject {
    @Published private(set) var levels = Array(repeating: Float.zero, count: 37)

    func append(_ level: Float) {
        var next = Array(levels.dropFirst())
        next.append(level.isFinite ? min(1, max(0, level)) : 0)
        levels = next
    }

    func reset() {
        levels = Array(repeating: .zero, count: 37)
    }
}

struct LiveRecordingWaveformView: View {
    @ObservedObject var history: RecordingWaveformHistory

    var body: some View {
        RecordingWaveformView(levels: history.levels)
    }
}

/// One canvas keeps the waveform inexpensive even at the recorder's cadence.
/// Silence is a quiet baseline; only microphone energy gives the bars height.
struct RecordingWaveformView: View {
    let levels: [Float]

    var body: some View {
        Canvas { context, size in
            guard !levels.isEmpty else { return }
            let step = size.width / CGFloat(levels.count)
            let width = max(1, step * 0.52)
            var bars = Path()
            for (index, level) in levels.enumerated() {
                let finiteLevel = level.isFinite ? Double(level) : 0
                let energy = min(1, max(0, finiteLevel) * 3.6)
                let height = 2 + (size.height - 2) * pow(energy, 0.72)
                let rect = CGRect(
                    x: CGFloat(index) * step + (step - width) / 2,
                    y: (size.height - height) / 2,
                    width: width,
                    height: height
                )
                bars.addRoundedRect(in: rect, cornerSize: CGSize(width: width / 2, height: width / 2))
            }
            context.fill(bars, with: .linearGradient(
                Gradient(colors: [Color(red: 0.24, green: 0.72, blue: 0.88),
                                  Color(red: 0.65, green: 0.95, blue: 0.96),
                                  Color(red: 0.24, green: 0.72, blue: 0.88)]),
                startPoint: .zero,
                endPoint: CGPoint(x: 0, y: size.height)
            ))
        }
        .transaction { $0.animation = nil }
        .accessibilityHidden(true)
    }
}
