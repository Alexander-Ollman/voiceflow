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
        case listening   // driven by the microphone
        case thinking    // key released: a quick "heard you" flare, then a gliding highlight
        case success     // text landed: one green ripple outward
        case failure     // nothing inserted: one amber ripple
    }

    /// Outcome colours (edge, center).
    static let successStops = (Color(red: 0.18, green: 0.74, blue: 0.44), Color(red: 0.66, green: 1.00, blue: 0.74))
    static let failureStops = (Color(red: 0.98, green: 0.56, blue: 0.20), Color(red: 1.00, green: 0.84, blue: 0.48))

    @ObservedObject var history: RecordingWaveformHistory
    var mode: Mode = .listening
    var palette: WaveformPalette = WaveformPalette(
        rawValue: UserDefaults.standard.string(forKey: WaveformPalette.preferenceKey) ?? "") ?? .default

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var physics = BarPhysics()
    @State private var clock = ModeClock()

    var body: some View {
        // .animation with no interval: display rate (up to 120 Hz on ProMotion).
        TimelineView(.animation) { timeline in
            Canvas { context, size in
                draw(in: &context, size: size, time: timeline.date.timeIntervalSinceReferenceDate)
            }
        }
        .accessibilityHidden(true)
    }

    // MARK: - Drawing

    private func draw(in context: inout GraphicsContext, size: CGSize, time: Double) {
        let pitch: CGFloat = 5.75       // bar + gap
        let barWidth: CGFloat = 2.75
        let fit = Int(size.width / pitch)
        let count = max(5, fit % 2 == 0 ? fit - 1 : fit)  // odd, so one bar sits on center
        let mid = size.height / 2
        let maxHalf = size.height / 2
        let minHalf: CGFloat = barWidth / 2

        let elapsed = clock.elapsed(in: mode, now: time)
        let targets = (0..<count).map { target(index: $0, count: count, time: time, elapsed: elapsed) }
        let heights = physics.step(targets: targets, now: time, snap: reduceMotion)

        let used = CGFloat(count) * pitch
        let originX = (size.width - used) / 2 + (pitch - barWidth) / 2
        var bars = Path()
        var barRects: [CGRect] = []
        for (i, h) in heights.enumerated() {
            let half = minHalf + (maxHalf - minHalf) * CGFloat(min(1, h))
            let rect = CGRect(x: originX + CGFloat(i) * pitch, y: mid - half, width: barWidth, height: half * 2)
            barRects.append(rect)
            bars.addRoundedRect(in: rect, cornerSize: CGSize(width: barWidth / 2, height: barWidth / 2))
        }

        func shading(_ stops: (Color, Color)) -> GraphicsContext.Shading {
            .linearGradient(
                Gradient(stops: [
                    .init(color: stops.0, location: 0),
                    .init(color: stops.1, location: 0.5),
                    .init(color: stops.0, location: 1),
                ]),
                startPoint: CGPoint(x: originX, y: mid),
                endPoint: CGPoint(x: originX + used - pitch + barWidth, y: mid))
        }

        switch mode {
        case .listening, .thinking:
            context.fill(bars, with: shading(palette.stops))
        case .success, .failure:
            // Cross-fade from the palette into the outcome colour (~100 ms).
            let blend = reduceMotion ? 1 : min(1, elapsed / 0.1)
            var base = context
            base.opacity = 1 - blend
            base.fill(bars, with: shading(palette.stops))
            var outcome = context
            outcome.opacity = blend
            outcome.fill(bars, with: shading(mode == .success ? Self.successStops : Self.failureStops))
        }

        // While thinking, a soft white highlight rides the gliding bump.
        if mode == .thinking, !reduceMotion {
            let center = Double(count - 1) / 2
            for (i, rect) in barRects.enumerated() {
                let x = (Double(i) - center) / max(center, 1)
                let glow = Self.thinkingBump(x: x, elapsed: elapsed)
                guard glow > 0.05 else { continue }
                var highlight = context
                highlight.opacity = 0.55 * glow
                highlight.fill(Path(roundedRect: rect, cornerRadius: barWidth / 2), with: .color(.white))
            }
        }
    }

    /// The thinking highlight: a soft bump gliding back and forth (x in -1…1).
    private static func thinkingBump(x: Double, elapsed: Double) -> Double {
        let position = 0.78 * sin(elapsed * 2.8 - .pi / 2)   // starts at the left edge
        return exp(-pow((x - position) / 0.3, 2))
    }

    /// Target height 0…1 for bar `index`.
    private func target(index: Int, count: Int, time: Double, elapsed: Double) -> Double {
        let center = Double(count - 1) / 2
        let signed = (Double(index) - center) / max(center, 1)         // -1 left … 1 right
        let distance = abs(signed)                                     // 0 center … 1 edge
        let envelope = 1 - 0.45 * distance * distance                  // taller in the middle
        let motion = reduceMotion ? 0.0 : 1.0

        switch mode {
        case .listening:
            // Newest level at the center, ~0.3 s old at the edges. Ease into
            // each new reading over its ~21 ms gap rather than stepping to it:
            // shift the read position back by the unelapsed part of the gap.
            let levels = history.levels
            let step = levels.count > 1 ? 1.0 / Double(levels.count - 1) : 0
            let eased = min(1, max(0, (time - history.lastAppend) / Self.levelInterval))
            let age = distance * 0.38 + (1 - eased) * step
            let speech = Self.smoothedSample(levels, age: min(1, age))
            let idle = 0.04 + 0.02 * motion * sin(time * 3.0 - distance * 5)
            return min(1, max(idle, speech * envelope))
        case .thinking:
            if reduceMotion { return 0.10 + 0.18 * envelope }
            // "Heard you": a quick flare that gathers into the center (~320 ms)…
            let flareProgress = min(1, elapsed / 0.32)
            let flare = (1 - flareProgress) * (1 - flareProgress)
                * 1.0 * exp(-distance * (0.8 + 5 * flareProgress))
            // …then a soft bump gliding back and forth while we work.
            let glide = min(1, max(0, (elapsed - 0.12) / 0.2))
            let bump = 0.06 + 0.40 * glide * Self.thinkingBump(x: signed, elapsed: elapsed) * envelope
            return max(flare, bump)
        case .success, .failure:
            if reduceMotion { return 0.14 * envelope }
            // One ripple running from the center to the edges (~380 ms).
            let radius = elapsed / 0.38
            let ring = exp(-pow((distance - radius) / 0.2, 2))
            let peak = mode == .success ? 0.85 : 0.6
            return 0.05 + peak * ring * (1 - 0.35 * distance)
        }
    }

    /// Spacing of AudioRecorder.meterLevels readings (1024 frames at 48 kHz).
    static let levelInterval = 1024.0 / 48_000.0

    /// Perceptual mapping: quiet speech still moves the bars.
    static func energy(_ level: Float) -> Double {
        guard level.isFinite else { return 0 }
        return pow(Double(min(1, max(0, level) * 3.6)), 0.72)
    }

    /// Energy averaged over neighbouring readings, so single noisy ~21 ms
    /// slices don't make bars twitch.
    private static func smoothedSample(_ levels: [Float], age: Double) -> Double {
        let step = levels.count > 1 ? 1.0 / Double(levels.count - 1) : 0
        let taps = [-1.0, 0.0, 1.0], weights = [1.0, 2.0, 1.0]
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
        let dt = min(1.0 / 20.0, max(0, now - (lastTime ?? now)))
        lastTime = now
        if snap || dt == 0 {
            heights = targets
            return heights
        }
        // Critically damped (no bounce): a fast rise so syllables register
        // immediately, a softer fall so the release reads as fluid. Integrated
        // in small substeps so the stiff spring stays stable at any frame rate.
        let substeps = max(1, Int((dt / (1.0 / 240.0)).rounded(.up)))
        let h = dt / Double(substeps)
        for i in heights.indices {
            let rising = targets[i] > heights[i]
            let stiffness = rising ? 900.0 : 260.0
            let damping = 2 * stiffness.squareRoot()
            for _ in 0..<substeps {
                let force = stiffness * (targets[i] - heights[i]) - damping * velocities[i]
                velocities[i] += force * h
                heights[i] += velocities[i] * h
            }
            heights[i] = min(1, max(0, heights[i]))
        }
        return heights
    }
}

/// Seconds since the waveform entered its current mode.
final class ModeClock {
    private var mode: LiquidWaveformView.Mode?
    private var since: Double = 0
    func elapsed(in mode: LiquidWaveformView.Mode, now: Double) -> Double {
        if mode != self.mode {
            self.mode = mode
            since = now
        }
        return now - since
    }
}
