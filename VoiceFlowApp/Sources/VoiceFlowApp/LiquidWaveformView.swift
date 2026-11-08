import SwiftUI

/// A fluid, layered waveform for the voice overlay.
///
/// The shape is driven by real microphone history: the newest level sits in
/// the center and older levels spread toward both edges, so speech visibly
/// ripples outward. Three translucent lens layers with slowly drifting phase
/// give it a liquid feel. Silence settles to a calm, thin line.
///
/// With Reduce Motion the phase drift stops; the shape still follows the
/// microphone (that is information, not decoration).
struct LiquidWaveformView: View {
    enum Mode: Equatable {
        case listening   // driven by the microphone
        case thinking    // recognizer/formatter busy: low, gently shimmering
    }

    @ObservedObject var history: RecordingWaveformHistory
    var mode: Mode = .listening

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var smoother = LevelSmoother()

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60.0, paused: reduceMotion)) { timeline in
            Canvas { context, size in
                let t = reduceMotion ? 0 : timeline.date.timeIntervalSinceReferenceDate
                draw(in: &context, size: size, time: t)
            }
        }
        .accessibilityHidden(true)
    }

    // MARK: - Drawing

    private struct Layer {
        let color: Color
        let frequency: Double   // wave cycles across the width
        let speed: Double       // radians per second
        let gain: Double        // relative height
        let offset: Double      // phase offset
    }

    private static let layers: [Layer] = [
        Layer(color: VF.Palette.iris,  frequency: 1.15, speed: 1.9, gain: 0.85, offset: 0.0),
        Layer(color: VF.Palette.lagoon, frequency: 1.65, speed: -2.6, gain: 1.0, offset: 1.7),
        Layer(color: VF.Palette.mint,  frequency: 2.20, speed: 3.3, gain: 0.7, offset: 3.1),
    ]

    private func draw(in context: inout GraphicsContext, size: CGSize, time: Double) {
        let levels = history.levels
        let mid = size.height / 2
        let halfHeight = Double(size.height / 2) - 0.5

        let level: Double
        switch mode {
        case .listening:
            level = smoother.next(Double(Self.energy(levels.last ?? 0)))
        case .thinking:
            level = smoother.next(0.24 + 0.06 * sin(time * 2.2))
        }

        let samples = max(24, Int(size.width / 3))
        context.blendMode = .plusLighter

        for layer in Self.layers {
            var top: [CGPoint] = []
            top.reserveCapacity(samples + 1)
            for i in 0...samples {
                let x = Double(i) / Double(samples)
                // Taper to zero at both ends so each layer reads as a liquid lens.
                let window = pow(sin(.pi * x), 1.6)
                let local: Double
                switch mode {
                case .listening:
                    // Mirror history around the center: newest in the middle.
                    let distance = abs(x - 0.5) * 2               // 0 center … 1 edge
                    local = min(1, 0.45 * level + 0.75 * Self.sample(levels, age: distance))
                case .thinking:
                    // A soft highlight travelling across the line.
                    let sweep = (sin(time * 1.6) + 1) / 2
                    local = level * (0.6 + 0.8 * exp(-pow((x - sweep) * 5, 2)))
                }
                let carrier = sin(2 * .pi * layer.frequency * x + layer.speed * time + layer.offset)
                let amplitude = max(0.05, local) * layer.gain * window
                let y = Double(mid) - halfHeight * amplitude * (0.55 + 0.45 * carrier)
                top.append(CGPoint(x: x * Double(size.width), y: y))
            }

            var path = Path()
            path.move(to: top[0])
            for p in top.dropFirst() { path.addLine(to: p) }
            // Mirror below the center line for a symmetric lens.
            for p in top.reversed() { path.addLine(to: CGPoint(x: p.x, y: 2 * mid - p.y)) }
            path.closeSubpath()

            context.fill(path, with: .linearGradient(
                Gradient(colors: [layer.color.opacity(0.15), layer.color.opacity(0.95), layer.color.opacity(0.15)]),
                startPoint: CGPoint(x: 0, y: mid), endPoint: CGPoint(x: size.width, y: mid)))
        }

        // A bright core line keeps the object legible during silence.
        context.blendMode = .normal
        var core = Path()
        core.move(to: CGPoint(x: size.width * 0.06, y: mid))
        core.addLine(to: CGPoint(x: size.width * 0.94, y: mid))
        context.stroke(core, with: .linearGradient(
            Gradient(colors: [.white.opacity(0), .white.opacity(0.55 - 0.35 * min(1, level * 2)), .white.opacity(0)]),
            startPoint: CGPoint(x: 0, y: mid), endPoint: CGPoint(x: size.width, y: mid)),
            lineWidth: 1.2)
    }

    /// Perceptual mapping shared with the bar waveform: quiet speech still moves.
    static func energy(_ level: Float) -> Float {
        guard level.isFinite else { return 0 }
        return pow(min(1, max(0, level) * 3.6), 0.72)
    }

    /// Interpolated level by age (0 = newest, 1 = oldest in the history).
    private static func sample(_ levels: [Float], age: Double) -> Double {
        guard levels.count > 1 else { return Double(energy(levels.first ?? 0)) }
        let position = (1 - age) * Double(levels.count - 1)
        let lower = Int(position.rounded(.down))
        let upper = min(levels.count - 1, lower + 1)
        let fraction = position - Double(lower)
        let a = Double(energy(levels[lower])), b = Double(energy(levels[upper]))
        return a + (b - a) * fraction
    }
}

/// Fast attack, slow release, so syllables pop and pauses settle smoothly.
/// A reference type so the Canvas closure can advance it per frame.
final class LevelSmoother {
    private var value: Double = 0
    func next(_ target: Double) -> Double {
        let rate = target > value ? 0.45 : 0.08
        value += (target - value) * rate
        return value
    }
}
