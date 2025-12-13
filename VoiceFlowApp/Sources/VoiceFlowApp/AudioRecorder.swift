import AVFoundation
import Accelerate

/// Audio recorder that captures microphone input and resamples to 16kHz
final class AudioRecorder: ObservableObject {
    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?

    @Published var isRecording = false
    @Published var audioLevel: Float = 0

    private var audioBuffer: [Float] = []
    private let targetSampleRate: Double = 16000

    /// Start recording from microphone
    func startRecording() throws {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        audioBuffer.removeAll()

        // Install tap on input node
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer, inputSampleRate: inputFormat.sampleRate)
        }

        try engine.start()

        self.audioEngine = engine
        self.inputNode = inputNode
        self.isRecording = true
    }

    /// Stop recording and return audio samples at 16kHz
    func stopRecording() -> [Float] {
        inputNode?.removeTap(onBus: 0)
        audioEngine?.stop()

        isRecording = false
        audioLevel = 0

        let samples = audioBuffer
        audioBuffer.removeAll()
        return samples
    }

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer, inputSampleRate: Double) {
        guard let channelData = buffer.floatChannelData else { return }

        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)

        // Convert to mono if stereo
        var monoSamples: [Float]
        if channelCount > 1 {
            monoSamples = [Float](repeating: 0, count: frameCount)
            for i in 0..<frameCount {
                var sum: Float = 0
                for ch in 0..<channelCount {
                    sum += channelData[ch][i]
                }
                monoSamples[i] = sum / Float(channelCount)
            }
        } else {
            monoSamples = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        }

        // Calculate audio level for visualization
        var rms: Float = 0
        vDSP_rmsqv(monoSamples, 1, &rms, vDSP_Length(monoSamples.count))
        DispatchQueue.main.async {
            self.audioLevel = min(1.0, rms * 10)
        }

        // Resample to 16kHz if needed
        let resampledSamples: [Float]
        if abs(inputSampleRate - targetSampleRate) > 1 {
            resampledSamples = resample(monoSamples, from: inputSampleRate, to: targetSampleRate)
        } else {
            resampledSamples = monoSamples
        }

        audioBuffer.append(contentsOf: resampledSamples)
    }

    /// Simple linear interpolation resampling
    private func resample(_ samples: [Float], from inputRate: Double, to outputRate: Double) -> [Float] {
        let ratio = inputRate / outputRate
        let outputLength = Int(Double(samples.count) / ratio)

        var output = [Float](repeating: 0, count: outputLength)

        for i in 0..<outputLength {
            let inputIndex = Double(i) * ratio
            let index0 = Int(inputIndex)
            let index1 = min(index0 + 1, samples.count - 1)
            let fraction = Float(inputIndex - Double(index0))

            output[i] = samples[index0] * (1 - fraction) + samples[index1] * fraction
        }

        return output
    }
}
