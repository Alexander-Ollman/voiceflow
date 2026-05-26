import AppKit
import CoreGraphics
import ScreenCaptureKit
import Vision

/// Last-resort context source: OCR the focused window when AX, browser bridge,
/// and shadow buffer all fail or are stale.
///
/// Cost is real (~250–400ms on Apple Silicon with .accurate), so this is gated
/// behind every other layer in `FieldContext.resolve()`. Requires Screen
/// Recording permission; returns nil if denied.
enum VisionOCR {

    struct Result {
        let text: String
        let lineCount: Int
    }

    /// OCR the frontmost window. Returns nil if capture or recognition fails.
    /// `timeoutMs` caps the entire pipeline; 500ms is a sane default.
    static func ocrFrontWindow(timeoutMs: Int = 500) async -> Result? {
        guard CGPreflightScreenCaptureAccess() else { return nil }
        guard #available(macOS 14.0, *) else { return nil }
        guard let cg = await captureFrontWindowSCK() else { return nil }
        return await runOCR(cg: cg, timeoutMs: timeoutMs)
    }

    // Vision request handlers aren't Sendable; isolate the perform call inside
    // a non-Sendable scope by running it on a serial queue we own.
    private static let ocrQueue = DispatchQueue(label: "voiceflow.visionocr", qos: .userInitiated)

    private static func runOCR(cg: CGImage, timeoutMs: Int) async -> Result? {
        await withCheckedContinuation { (continuation: CheckedContinuation<Result?, Never>) in
            // Box the continuation so the timeout and completion paths can both
            // resume it safely (only one wins).
            final class ContinuationBox: @unchecked Sendable {
                var cont: CheckedContinuation<Result?, Never>?
                let lock = NSLock()
                init(_ c: CheckedContinuation<Result?, Never>) { self.cont = c }
                func resume(_ value: Result?) {
                    lock.lock(); defer { lock.unlock() }
                    guard let c = cont else { return }
                    cont = nil
                    c.resume(returning: value)
                }
            }
            let box = ContinuationBox(continuation)

            ocrQueue.asyncAfter(deadline: .now() + .milliseconds(timeoutMs)) {
                box.resume(nil)
            }

            ocrQueue.async {
                let request = VNRecognizeTextRequest { req, _ in
                    guard let observations = req.results as? [VNRecognizedTextObservation] else {
                        box.resume(nil); return
                    }
                    let lines = observations.compactMap { $0.topCandidates(1).first?.string }
                    box.resume(Result(text: lines.joined(separator: "\n"), lineCount: lines.count))
                }
                request.recognitionLevel = .accurate
                request.usesLanguageCorrection = false

                do {
                    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
                    try handler.perform([request])
                } catch {
                    NSLog("[VisionOCR] perform failed: %@", error.localizedDescription)
                    box.resume(nil)
                }
            }
        }
    }

    @available(macOS 14.0, *)
    private static func captureFrontWindowSCK() async -> CGImage? {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
            guard let frontApp = NSWorkspace.shared.frontmostApplication,
                  let window = content.windows.first(where: {
                      $0.owningApplication?.processID == frontApp.processIdentifier && $0.isOnScreen
                  })
            else { return nil }

            let filter = SCContentFilter(desktopIndependentWindow: window)
            let config = SCStreamConfiguration()
            // Keep native resolution — OCR accuracy drops sharply when text gets downscaled.
            config.width = Int(window.frame.width)
            config.height = Int(window.frame.height)
            config.showsCursor = false
            return try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        } catch {
            NSLog("[VisionOCR] SCK capture failed: %@", error.localizedDescription)
            return nil
        }
    }
}
