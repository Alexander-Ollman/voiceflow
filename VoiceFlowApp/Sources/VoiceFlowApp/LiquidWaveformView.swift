import SwiftUI

/// Colour swatches for the voice waveform. One gradient runs across the bars
/// (edges → center → edges); the center stop is the brightest.
enum WaveformPalette: String, CaseIterable, Identifiable {
    case glacier, lagoon, aurora, ember, mono

    var id: String { rawValue }

    /// Edge colour, center colour.
    var stops: (edge: Color, center: Color) {
        switch self {
        case .glacier: return (Color(red: 0.45, green: 0.68, blue: 1.00), Color(red: 0.88, green: 0.96, blue: 1.00))
        case .lagoon:  return (Color(red: 0.05, green: 0.62, blue: 0.70), Color(red: 0.55, green: 1.00, blue: 0.86))
        case .aurora:  return (Color(red: 0.36, green: 0.86, blue: 0.80), Color(red: 0.78, green: 0.70, blue: 1.00))
        case .ember:   return (Color(red: 1.00, green: 0.45, blue: 0.40), Color(red: 1.00, green: 0.84, blue: 0.55))
        case .mono:    return (Color.white.opacity(0.55), Color.white)
        }
    }

    static let preferenceKey = "waveformPalette"
    static let `default`: WaveformPalette = .glacier
}

/// The voice overlay's waveform: a small row of fine rounded bars that rise
/// from the center line. Each bar is a critically damped spring chasing a
/// smoothed target height, so motion is quick but never bouncy.
///
/// - Listening: bar heights come from real microphone history (~47 Hz):
///   the newest level drives the center bars, slightly older levels the outer
///   bars, so speech visibly pushes outward.
/// - Silence: a faint ripple keeps the object alive without faking speech.
/// - Thinking: a travelling wave while recognition/formatting runs.
///
/// Reduce Motion turns the springs into direct jumps and stops the idle and
/// thinking motion; the bars still follow the microphone.
struct LiquidWaveformView: View {
    enum Mode: Equatable {
        case listening
        case thinking
    }

    @ObservedObject var history: RecordingWaveformHistory
    var mode: Mode = .listening
    var palette: WaveformPalette = WaveformPalette(
        rawValue: UserDefaults.standard.string(forKey: WaveformPalette.preferenceKey) ?? "") ?? .default

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var physics = BarPhysics()

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60.0)) { timeline in
            Canvas { context, size in
                draw(in: &context, size: size, time: timeline.date.timeIntervalSinceReferenceDate)
            }
        }
        .accessibilityHidden(true)
    }

    // MARK: - Drawing

    private func draw(in context: inout GraphicsContext, size: CGSize, time: Double) {
        let pitch: CGFloat = 4.6        // bar + gap
        let barWidth: CGFloat = 2.2
        let fit = Int(size.width / pitch)
        let count = max(5, fit % 2 == 0 ? fit - 1 : fit)  // odd, so one bar sits on center
        let mid = size.height / 2
        let maxHalf = size.height / 2
        let minHalf: CGFloat = barWidth / 2

        let targets = (0..<count).map { target(index: $0, count: count, time: time) }
        let heights = physics.step(targets: targets, now: time, snap: reduceMotion)

        let used = CGFloat(count) * pitch
        let originX = (size.width - used) / 2 + (pitch - barWidth) / 2
        var bars = Path()
        for (i, h) in heights.enumerated() {
            let half = minHalf + (maxHalf - minHalf) * CGFloat(min(1, h))
            let rect = CGRect(x: originX + CGFloat(i) * pitch, y: mid - half, width: barWidth, height: half * 2)
            bars.addRoundedRect(in: rect, cornerSize: CGSize(width: barWidth / 2, height: barWidth / 2))
        }

        let (edge, center) = palette.stops
        let shading = GraphicsContext.Shading.linearGradient(
            Gradient(stops: [
                .init(color: edge, location: 0),
                .init(color: center, location: 0.5),
                .init(color: edge, location: 1),
            ]),
            startPoint: CGPoint(x: originX, y: mid),
            endPoint: CGPoint(x: originX + used - pitch + barWidth, y: mid))

        context.fill(bars, with: shading)
    }

    /// Target height 0…1 for bar `index`.
    private func target(index: Int, count: Int, time: Double) -> Double {
        let center = Double(count - 1) / 2
        let distance = abs(Double(index) - center) / max(center, 1)   // 0 center … 1 edge
        let envelope = 1 - 0.45 * distance * distance                  // taller in the middle
        let motion = reduceMotion ? 0.0 : 1.0

        switch mode {
        case .listening:
            // Newest level at the center, ~0.3 s old at the edges.
            let speech = Self.smoothedSample(history.levels, age: distance * 0.38)
            let idle = 0.04 + 0.02 * motion * sin(time * 3.0 - distance * 5)
            return min(1, max(idle, speech * envelope))
        case .thinking:
            let wave = 0.5 + 0.5 * sin(time * 4.5 * motion - distance * 5)
            return 0.06 + 0.30 * wave * envelope
        }
    }

    /// Perceptual mapping: quiet speech still moves the bars.
    static func energy(_ level: Float) -> Double {
        guard level.isFinite else { return 0 }
        return pow(Double(min(1, max(0, level) * 3.6)), 0.72)
    }

    /// Energy averaged over neighbouring readings, so single noisy ~21 ms
    /// slices don't make bars twitch.
    private static func smoothedSample(_ levels: [Float], age: Double) -> Double {
        let step = levels.count > 1 ? 1.0 / Double(levels.count - 1) : 0
        let taps = [-2.0, -1.0, 0.0, 1.0, 2.0], weights = [1.0, 2.0, 3.0, 2.0, 1.0]
        var total = 0.0
        for (tap, weight) in zip(taps, weights) {
            total += weight * sample(levels, age: min(1, max(0, age + tap * step)))
        }
        return total / weights.reduce(0, +)
    }

    /// Interpolated energy by age (0 = newest, 1 = oldest in the history).
    private static func sample(_ levels: [Float], age: Double) -> Double {
        guard levels.count > 1 else { return energy(levels.first ?? 0) }
        let position = (1 - age) * Double(levels.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = min(levels.count - 1, lower + 1)
        let fraction = position - Double(lower)
        let a = energy(levels[lower]), b = energy(levels[upper])
        return a + (b - a) * fraction
    }
}

/// Per-bar springs, advanced once per rendered frame. A reference type so the
/// Canvas closure can keep state between frames.
final class BarPhysics {
    private var heights: [Double] = []
    private var velocities: [Double] = []
    private var lastTime: Double?

    func step(targets: [Double], now: Double, snap: Bool) -> [Double] {
        if heights.count != targets.count {
            heights = targets
            velocities = Array(repeating: 0, count: targets.count)
        }
        let dt = min(1.0 / 30.0, max(0, now - (lastTime ?? now)))
        lastTime = now
        if snap || dt == 0 {
            heights = targets
            return heights
        }
        // Critically damped: quick but no bounce, so motion reads as fluid.
        let stiffness = 170.0, damping = 2 * stiffness.squareRoot()
        for i in heights.indices {
            let force = stiffness * (targets[i] - heights[i]) - damping * velocities[i]
            velocities[i] += force * dt
            heights[i] = min(1, max(0, heights[i] + velocities[i] * dt))
        }
        return heights
    }
}
