import AppKit
import CoreGraphics
import ScreenCaptureKit

/// Captures the active window as a compressed JPEG for VLM analysis.
/// Uses ScreenCaptureKit (macOS 14+) with CGWindowListCreateImage fallback (macOS 13).
final class ScreenshotCapture {

    /// Capture the frontmost window as JPEG data (resized to fit 1280x720, quality 0.7).
    func captureActiveWindow() async throws -> Data {
        if #available(macOS 14.0, *) {
            return try await captureWithScreenCaptureKit()
        } else {
            return try captureWithCGWindowList()
        }
    }

    /// Check if screen recording permission is granted.
    static func hasPermission() -> Bool {
        return CGPreflightScreenCaptureAccess()
    }

    /// Request screen recording permission from the user.
    static func requestPermission() {
        CGRequestScreenCaptureAccess()
    }

    // MARK: - ScreenCaptureKit (macOS 14+)

    @available(macOS 14.0, *)
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

    // MARK: - CGWindowList fallback (macOS 13)

    private func captureWithCGWindowList() throws -> Data {
        guard let frontApp = NSWorkspace.shared.frontmostApplication else {
            throw ScreenshotError.noActiveWindow
        }

        let pid = frontApp.processIdentifier

        guard let windowList = CGWindowListCopyWindowInfo(
            [.optionOnScreenOnly, .excludeDesktopElements],
            kCGNullWindowID
        ) as? [[CFString: Any]] else {
            throw ScreenshotError.noActiveWindow
        }

        // Find the frontmost window belonging to the active app
        var targetWindowID: CGWindowID?
        for windowInfo in windowList {
            guard let ownerPID = windowInfo[kCGWindowOwnerPID] as? Int32,
                  ownerPID == pid,
                  let layer = windowInfo[kCGWindowLayer] as? Int,
                  layer == 0
            else { continue }

            if let windowID = windowInfo[kCGWindowNumber] as? CGWindowID {
                targetWindowID = windowID
                break
            }
        }

        guard let windowID = targetWindowID else {
            throw ScreenshotError.noActiveWindow
        }

        guard let cgImage = CGWindowListCreateImage(
            .null,
            .optionIncludingWindow,
            windowID,
            [.boundsIgnoreFraming, .bestResolution]
        ) else {
            throw ScreenshotError.captureFailed
        }

        let resized = resizeImage(cgImage, maxWidth: 1280, maxHeight: 720)
        return try compressToJPEG(resized)
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

    private func resizeImage(_ image: CGImage, maxWidth: Int, maxHeight: Int) -> CGImage {
        let width = image.width
        let height = image.height

        if width <= maxWidth && height <= maxHeight {
            return image
        }

        let widthRatio = CGFloat(maxWidth) / CGFloat(width)
        let heightRatio = CGFloat(maxHeight) / CGFloat(height)
        let scale = min(widthRatio, heightRatio)

        let newWidth = Int(CGFloat(width) * scale)
        let newHeight = Int(CGFloat(height) * scale)

        guard let context = CGContext(
            data: nil,
            width: newWidth,
            height: newHeight,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            return image
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))

        return context.makeImage() ?? image
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
