import AppKit
import CoreGraphics
import ScreenCaptureKit

/// Captures the active window as a compressed JPEG for VLM analysis.
/// Uses ScreenCaptureKit (now always — the app's deployment target is macOS 15+).
final class ScreenshotCapture {

    /// Capture the frontmost window as JPEG data (resized to fit 1280x720, quality 0.7).
    func captureActiveWindow() async throws -> Data {
        return try await captureWithScreenCaptureKit()
    }

    /// Check if screen recording permission is granted.
    static func hasPermission() -> Bool {
        return CGPreflightScreenCaptureAccess()
    }

    /// Request screen recording permission from the user.
    static func requestPermission() {
        CGRequestScreenCaptureAccess()
    }

    // MARK: - ScreenCaptureKit

    private func captureWithScreenCaptureKit() async throws -> Data {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)

        guard let frontApp = NSWorkspace.shared.frontmostApplication else {
            throw ScreenshotError.noActiveWindow
        }

        // Find the frontmost window for the active app
        guard let window = content.windows.first(where: {
            $0.owningApplication?.processID == frontApp.processIdentifier && $0.isOnScreen
        }) else {
            throw ScreenshotError.noActiveWindow
        }

        let filter = SCContentFilter(desktopIndependentWindow: window)

        let config = SCStreamConfiguration()
        // Scale to fit within 1280x720
        let scale = min(1280.0 / CGFloat(window.frame.width), 720.0 / CGFloat(window.frame.height), 1.0)
        config.width = Int(CGFloat(window.frame.width) * scale)
        config.height = Int(CGFloat(window.frame.height) * scale)
        config.showsCursor = false

        let cgImage = try await SCScreenshotManager.captureImage(
            contentFilter: filter,
            configuration: config
        )

        return try compressToJPEG(cgImage)
    }

    // MARK: - Private Helpers

    private func compressToJPEG(_ cgImage: CGImage) throws -> Data {
        let nsImage = NSImage(cgImage: cgImage, size: NSSize(
            width: CGFloat(cgImage.width),
            height: CGFloat(cgImage.height)
        ))
        guard let tiffData = nsImage.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let jpegData = bitmap.representation(
                  using: .jpeg,
                  properties: [.compressionFactor: 0.7]
              ) else {
            throw ScreenshotError.compressionFailed
        }
        return jpegData
    }
}

enum ScreenshotError: LocalizedError {
    case noActiveWindow
    case captureFailed
    case compressionFailed

    var errorDescription: String? {
        switch self {
        case .noActiveWindow:
            return "No active window found for screenshot"
        case .captureFailed:
            return "Failed to capture window image"
        case .compressionFailed:
            return "Failed to compress screenshot to JPEG"
        }
    }
}
