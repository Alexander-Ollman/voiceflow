import AppKit
import SwiftUI

/// Where the voice overlay appears.
enum OverlayPlacement: String, CaseIterable, Identifiable {
    case bottom   // floating glass capsule above the Dock (works everywhere)
    case island   // grows out of the camera housing as a floating pill
    case notch    // extends the camera housing down from the top edge

    var id: String { rawValue }
    static let preferenceKey = "overlayPlacement"

    var title: String {
        switch self {
        case .bottom: return "Bottom of screen"
        case .island: return "Dynamic Island"
        case .notch:  return "Notch"
        }
    }

    static var current: OverlayPlacement {
        OverlayPlacement(rawValue: UserDefaults.standard.string(forKey: preferenceKey) ?? "") ?? .bottom
    }
}

extension NSScreen {
    /// The camera housing (notch or Dynamic Island) in screen coordinates,
    /// derived from the usable menu-bar areas on either side of it. nil on
    /// displays without one.
    var cameraHousingFrame: CGRect? {
        guard let left = auxiliaryTopLeftArea, let right = auxiliaryTopRightArea,
              right.minX > left.maxX else { return nil }
        let height = safeAreaInsets.top > 0 ? safeAreaInsets.top : left.height
        return CGRect(x: left.maxX, y: frame.maxY - height, width: right.minX - left.maxX, height: height)
    }
}

/// The voice overlay attached to the camera housing.
///
/// The housing band itself is never drawn over: on Dynamic Island Macs the
/// system owns the island (it expands for the microphone indicator and other
/// activity), and on notched Macs the notch is fixed hardware with menu-bar
/// items and indicators on both sides. So everything here hangs *below* it:
/// - Island: a separate black pill just under the island.
/// - Notch: the shape is exactly notch-wide inside the menu-bar band (black
///   hardware there anyway), then widens only below the menu bar.
/// Waveform on the left, status on the right; live text expands below.
struct IslandOverlayView: View {
    @ObservedObject var state: OverlayState
    let style: OverlayPlacement          // .island or .notch
    /// Housing size in points (from NSScreen.cameraHousingFrame).
    let housing: CGSize

    static let compactWidth: CGFloat = 196
    static let rowHeight: CGFloat = 34
    static let expandedWidth: CGFloat = 420
    static let maxTextHeight: CGFloat = 180
    static let islandGap: CGFloat = 6
    /// Panel size: room for the widest/tallest state plus a shadow margin.
    static func panelSize(housing: CGSize) -> CGSize {
        CGSize(width: expandedWidth + 40,
               height: housing.height + islandGap + rowHeight + maxTextHeight + 70)
    }

    private var text: String {
        switch state.state {
        case .streaming(let t), .formatting(let t): return t
        default: return ""
        }
    }
    private var expanded: Bool { !text.isEmpty }
    private var bodyWidth: CGFloat { expanded ? Self.expandedWidth : Self.compactWidth }

    private var waveMode: LiquidWaveformView.Mode {
        switch state.state {
        case .recording, .streaming: return .listening
        case .succeeded: return .success
        case .failed: return .failure
        default: return .thinking
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            if style == .island {
                Color.clear.frame(height: housing.height + Self.islandGap)
                content
                    .background(
                        RoundedRectangle(cornerRadius: expanded ? 22 : Self.rowHeight / 2, style: .continuous)
                            .fill(Color.black)
                            .shadow(color: .black.opacity(0.3), radius: 10, y: 4))
            } else {
                VStack(spacing: 0) {
                    Color.clear.frame(width: bodyWidth, height: housing.height)
                    content
                }
                .fixedSize()
                .background(
                    NotchDropShape(neckWidth: housing.width, neckHeight: housing.height,
                                   bottomRadius: expanded ? 22 : 16)
                        .fill(Color.black)
                        .shadow(color: .black.opacity(expanded ? 0.3 : 0), radius: 10, y: 4))
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .environment(\.colorScheme, .dark)
        .animation(VF.Motion.morph, value: expanded)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityText)
    }

    private var content: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                LiquidWaveformView(history: state.waveform, mode: waveMode)
                    .frame(width: 110, height: 18)
                IslandStatus(state: state)
                    .frame(width: 34, alignment: .trailing)
            }
            .frame(height: Self.rowHeight)
            if expanded {
                transcript.transition(.opacity)
            }
        }
        .frame(width: bodyWidth)
        .fixedSize(horizontal: false, vertical: true)
    }

    private var transcript: some View {
        BottomClampLayout(maxHeight: Self.maxTextHeight) {
            Text(text)
                .font(.system(size: 14, weight: .regular, design: .rounded))
                .foregroundStyle(.white.opacity(0.95))
                .lineSpacing(2)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.top, 10)
        .clipped()
        .mask(VStack(spacing: 0) {
            LinearGradient(colors: [.clear, .black], startPoint: .top, endPoint: .bottom).frame(height: 10)
            Color.black
        })
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
    }

    private var accessibilityText: String {
        switch state.state {
        case .idle: return ""
        case .recording: return "VoiceFlow is listening"
        case .streaming(let t): return t.isEmpty ? "VoiceFlow is listening" : "Listening: \(t)"
        case .processing: return "Working"
        case .formatting(let t): return t
        case .aiProcessing(let l): return l
        case .succeeded: return "Done"
        case .failed: return "Nothing was inserted"
        }
    }
}

/// Status beside the waveform: a recording timer, a sparkle while working, then a
/// check or warning for the outcome.
private struct IslandStatus: View {
    @ObservedObject var state: OverlayState

    var body: some View {
        Group {
            switch state.state {
            case .recording, .streaming:
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    Text(elapsed(at: context.date))
                        .font(.system(size: 12, weight: .medium, design: .rounded).monospacedDigit())
                        .foregroundStyle(.white.opacity(0.7))
                }
            case .processing, .formatting, .aiProcessing:
                Image(systemName: "sparkles")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(VF.Palette.aqua)
            case .succeeded:
                Image(systemName: "checkmark")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(LiquidWaveformView.successStops.1)
                    .transition(.scale.combined(with: .opacity))
            case .failed:
                Image(systemName: "exclamationmark")
                    .font(.system(size: 12, weight: .bold))
                    .foregroundStyle(LiquidWaveformView.failureStops.1)
                    .transition(.scale.combined(with: .opacity))
            case .idle:
                EmptyView()
            }
        }
        .animation(VF.Motion.quick, value: stateKey)
    }

    private var stateKey: String { String(describing: state.state).prefix(12).description }

    private func elapsed(at date: Date) -> String {
        let seconds = max(0, Int(date.timeIntervalSince(state.recordingStarted ?? date)))
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }
}

/// Notch extension: exactly notch-wide through the menu-bar band (the
/// "neck"), then a wider body below it joined by concave fillets, with
/// rounded bottom corners. Nothing is drawn beside the notch in the band.
struct NotchDropShape: Shape {
    var neckWidth: CGFloat
    var neckHeight: CGFloat
    var bottomRadius: CGFloat

    var animatableData: CGFloat {
        get { bottomRadius }
        set { bottomRadius = newValue }
    }

    func path(in rect: CGRect) -> Path {
        let neckMinX = rect.midX - neckWidth / 2, neckMaxX = rect.midX + neckWidth / 2
        let top = rect.minY, shoulder = rect.minY + neckHeight
        let fillet = max(0, min(8, (rect.width - neckWidth) / 2, neckHeight))
        let corner = min(6, (rect.width - neckWidth) / 2)        // body's top outer corners
        let r = min(bottomRadius, (rect.maxY - shoulder) / 2, rect.width / 2)
        var p = Path()
        p.move(to: CGPoint(x: neckMinX, y: top))
        p.addLine(to: CGPoint(x: neckMaxX, y: top))
        p.addLine(to: CGPoint(x: neckMaxX, y: shoulder - fillet))
        p.addQuadCurve(to: CGPoint(x: neckMaxX + fillet, y: shoulder), control: CGPoint(x: neckMaxX, y: shoulder))
        p.addLine(to: CGPoint(x: rect.maxX - corner, y: shoulder))
        p.addQuadCurve(to: CGPoint(x: rect.maxX, y: shoulder + corner), control: CGPoint(x: rect.maxX, y: shoulder))
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - r))
        p.addQuadCurve(to: CGPoint(x: rect.maxX - r, y: rect.maxY), control: CGPoint(x: rect.maxX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX + r, y: rect.maxY))
        p.addQuadCurve(to: CGPoint(x: rect.minX, y: rect.maxY - r), control: CGPoint(x: rect.minX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX, y: shoulder + corner))
        p.addQuadCurve(to: CGPoint(x: rect.minX + corner, y: shoulder), control: CGPoint(x: rect.minX, y: shoulder))
        p.addLine(to: CGPoint(x: neckMinX - fillet, y: shoulder))
        p.addQuadCurve(to: CGPoint(x: neckMinX, y: shoulder - fillet), control: CGPoint(x: neckMinX, y: shoulder))
        p.closeSubpath()
        return p
    }
}
