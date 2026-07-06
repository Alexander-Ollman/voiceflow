import CoreAudio
import Foundation

/// Core Audio device introspection helpers.
///
/// Used to keep Bluetooth output devices (AirPods, BT headphones) in their
/// high-quality A2DP profile while dictating. Opening a Bluetooth microphone
/// forces the whole device into the Hands-Free Profile (HFP/SCO) — a mono,
/// 8–16 kHz bidirectional link — which collapses its *output* quality to phone
/// quality for every app on the system. Preferring the built-in mic when a
/// Bluetooth device is the default input avoids triggering that switch.
enum AudioDevices {
    private static let systemObject = AudioObjectID(kAudioObjectSystemObject)

    private static func address(
        _ selector: AudioObjectPropertySelector,
        scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal
    ) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain
        )
    }

    /// System default input device, or nil if none is set.
    static func defaultInputDeviceID() -> AudioDeviceID? {
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var addr = address(kAudioHardwarePropertyDefaultInputDevice)
        let status = AudioObjectGetPropertyData(systemObject, &addr, 0, nil, &size, &deviceID)
        guard status == noErr, deviceID != 0 else { return nil }
        return deviceID
    }

    /// Transport type (`kAudioDeviceTransportType*`) of a device, or nil on error.
    static func transportType(of deviceID: AudioDeviceID) -> UInt32? {
        var transport = UInt32(0)
        var size = UInt32(MemoryLayout<UInt32>.size)
        var addr = address(kAudioDevicePropertyTransportType)
        let status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &transport)
        guard status == noErr else { return nil }
        return transport
    }

    /// True when the device connects over Bluetooth (classic or LE).
    static func isBluetooth(_ deviceID: AudioDeviceID) -> Bool {
        guard let t = transportType(of: deviceID) else { return false }
        return t == kAudioDeviceTransportTypeBluetooth || t == kAudioDeviceTransportTypeBluetoothLE
    }

    /// Number of input channels a device exposes (0 == not an input device).
    static func inputChannelCount(_ deviceID: AudioDeviceID) -> Int {
        var addr = address(kAudioDevicePropertyStreamConfiguration, scope: kAudioObjectPropertyScopeInput)
        var size = UInt32(0)
        guard AudioObjectGetPropertyDataSize(deviceID, &addr, 0, nil, &size) == noErr, size > 0 else {
            return 0
        }
        let raw = UnsafeMutableRawPointer.allocate(
            byteCount: Int(size),
            alignment: MemoryLayout<AudioBufferList>.alignment
        )
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, raw) == noErr else { return 0 }
        let bufferList = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return bufferList.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    /// All audio device IDs known to Core Audio.
    static func allDeviceIDs() -> [AudioDeviceID] {
        var addr = address(kAudioHardwarePropertyDevices)
        var size = UInt32(0)
        guard AudioObjectGetPropertyDataSize(systemObject, &addr, 0, nil, &size) == noErr, size > 0 else {
            return []
        }
        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(systemObject, &addr, 0, nil, &size, &ids) == noErr else { return [] }
        return ids
    }

    /// First built-in input device, or nil if the Mac has none (e.g. Mac mini/Studio).
    static func builtInInputDeviceID() -> AudioDeviceID? {
        for id in allDeviceIDs() where inputChannelCount(id) > 0 {
            if transportType(of: id) == kAudioDeviceTransportTypeBuiltIn { return id }
        }
        return nil
    }

    /// Human-readable device name, for logging.
    static func name(of deviceID: AudioDeviceID) -> String {
        // The property yields a +1-retained CFStringRef; marshal via Unmanaged so
        // ARC balances the retain (takeRetainedValue) instead of leaking it.
        var name: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        var addr = address(kAudioObjectPropertyName)
        let status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &name)
        guard status == noErr, let name else { return "unknown" }
        return name.takeRetainedValue() as String
    }

    /// Decide which device to force capture onto, or nil to use the system default.
    ///
    /// Returns the built-in mic only when `avoidBluetooth` is set, the current
    /// default input is a Bluetooth device, and a built-in input exists. A
    /// non-Bluetooth default (built-in, USB, aggregate) is left untouched — those
    /// have no HFP problem.
    static func preferredCaptureDeviceID(avoidBluetooth: Bool) -> AudioDeviceID? {
        guard avoidBluetooth,
              let def = defaultInputDeviceID(),
              isBluetooth(def),
              let builtIn = builtInInputDeviceID()
        else { return nil }
        return builtIn
    }
}
