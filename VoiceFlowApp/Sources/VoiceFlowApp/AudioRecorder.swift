import AVFoundation
import Accelerate
import AudioToolbox

/// Audio recorder that captures microphone input and resamples to 16kHz
final class AudioRecorder: ObservableObject {
    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?

    @Published var isRecording = false
    @Published var audioLevel: Float = 0

    private var audioBuffer: [Float] = []
    private let targetSampleRate: Double = 16000

    /// Optional callback fired for each resampled audio chunk (16kHz mono).
    /// Used by streaming transcription to feed chunks in real-time.
    var onAudioChunk: (([Float]) -> Void)?

    /// Start recording from microphone
    func startRecording() throws {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode

        // Avoid degrading Bluetooth output. If the system default input is a
        // Bluetooth device (AirPods/headset), opening its mic forces the whole
        // device into HFP/SCO — a mono 8–16 kHz link — which collapses its A2DP
        // output to phone quality for *every* app. Redirect capture to the
        // built-in mic in that case so the user's audio stays high quality.
        // Must happen before reading the input format (the format follows the
        // selected device) and before installing the tap.
        redirectAwayFromBluetoothIfNeeded(inputNode)

        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Guard against invalid audio format (can happen after sleep, device switch, etc.)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            throw NSError(
                domain: "AudioRecorder",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Audio input format is invalid (sampleRate=\(inputFormat.sampleRate), channels=\(inputFormat.channelCount)). Try again."]
            )
        }

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

        // Fully release the engine so the OS reclaims the input device promptly.
        // Leaving a stopped engine alive can keep the HAL input unit claimed,
        // pinning a Bluetooth device in HFP longer than the dictation lasts.
        audioEngine = nil
        inputNode = nil

        let samples = audioBuffer
        audioBuffer.removeAll()
        return samples
    }

    /// When the "Avoid Bluetooth microphone" setting is on and the system default
    /// input is a Bluetooth device, point the engine's input node at the built-in
    /// mic. No-op when the setting is off, the default input isn't Bluetooth, or
    /// the Mac has no built-in mic (falls back to the system default).
    private func redirectAwayFromBluetoothIfNeeded(_ inputNode: AVAudioInputNode) {
        let avoidBluetooth = UserDefaults.standard.object(forKey: "avoidBluetoothMic") as? Bool ?? true
        guard let deviceID = AudioDevices.preferredCaptureDeviceID(avoidBluetooth: avoidBluetooth),
              let audioUnit = inputNode.audioUnit
        else { return }

        var mutableID = deviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &mutableID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        if status == noErr {
            NSLog(
                "[VoiceFlow] Mic: default input is Bluetooth; capturing via built-in mic '%@' to preserve output quality",
                AudioDevices.name(of: deviceID)
            )
        } else {
            NSLog("[VoiceFlow] Mic: failed to redirect to built-in mic (err %d); using system default", status)
        }
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

        // Forward chunk to streaming callback if set
        onAudioChunk?(resampledSamples)
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
