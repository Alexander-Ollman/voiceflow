import Foundation

/// Phase 8 — Whisper-volume mode.
///
/// Detects low-RMS dictation (the user is speaking quietly in an open office /
/// coffee shop / late at night) and applies a gain boost so the STT model
/// doesn't have to fight the noise floor.
///
/// Returns the boosted buffer and a flag indicating whether the boost fired so
/// the prompt builder can append a [LOW_VOLUME] hint to encourage the LLM to
/// be more forgiving on uncertain transcripts.
enum AudioGain {

    /// RMS threshold below which we consider the speech "whispered."
    /// Empirically: comfortable voice ~0.08–0.20 RMS, whisper ~0.005–0.03.
    static let whisperThreshold: Float = 0.025

    /// Maximum gain to apply (prevents clipping unmute-yelling).
    static let maxGain: Float = 6.0

    /// Target post-boost RMS — keeps signal in the model's preferred range.
    static let targetRMS: Float = 0.08

    struct Result {
        let samples: [Float]
        let wasBoosted: Bool
        let originalRMS: Float
        let appliedGain: Float
    }

    /// Compute RMS and apply gain boost when the signal is whisper-level.
    /// Always returns a non-empty buffer (the input unchanged when boost was
    /// not needed). Soft-clips at ±1.0 to avoid hard distortion.
    static func boostIfWhispered(_ samples: [Float]) -> Result {
        guard !samples.isEmpty else {
            return Result(samples: samples, wasBoosted: false, originalRMS: 0, appliedGain: 1)
        }

        let rms = computeRMS(samples)
        if rms >= whisperThreshold || rms < 1e-6 {
            // Either loud enough OR effectively silent — don't touch.
            return Result(samples: samples, wasBoosted: false, originalRMS: rms, appliedGain: 1)
        }

        let gain = min(targetRMS / rms, maxGain)
        let boosted = samples.map { softClip($0 * gain) }
        return Result(samples: boosted, wasBoosted: true, originalRMS: rms, appliedGain: gain)
    }

    private static func computeRMS(_ samples: [Float]) -> Float {
        var sum: Float = 0
        for s in samples {
            sum += s * s
        }
        return (sum / Float(samples.count)).squareRoot()
    }

    /// tanh soft-clip — preserves dynamics better than hard clip when boosting.
    private static func softClip(_ x: Float) -> Float {
        let t = tanhf(x * 0.9)
        return t
    }
}
