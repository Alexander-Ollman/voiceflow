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

/// The voice overlay drawn around the camera housing. Pure black so it merges
/// with the housing on OLED and mini-LED panels; nothing is drawn over the
/// housing itself. Waveform in the leading wing, status in the trailing wing,
/// live text expands below.
struct IslandOverlayView: View {
    @ObservedObject var state: OverlayState
    let style: OverlayPlacement          // .island or .notch
    /// Housing size in points (from NSScreen.cameraHousingFrame).
    let housing: CGSize

    static let wing: CGFloat = 76
    static let expandedWidth: CGFloat = 420
    static let maxTextHeight: CGFloat = 180
    /// Panel size: room for the widest/tallest state plus a shadow margin.
    static func panelSize(housing: CGSize) -> CGSize {
        CGSize(width: max(expandedWidth, housing.width + wing * 2) + 40,
               height: housing.height + maxTextHeight + 70)
    }

    private var text: String {
        switch state.state {
        case .streaming(let t), .formatting(let t): return t
        default: return ""
        }
    }
    private var expanded: Bool { !text.isEmpty }

    /// Island floats just below the top edge and reaches slightly past the
    /// housing band so it fully covers the system's own island shape; notch is
    /// flush to the top.
    private var topInset: CGFloat { style == .island ? 3 : 0 }
    private var bandHeight: CGFloat { style == .island ? housing.height + 3 : housing.height }
    private var compactWidth: CGFloat { housing.width + Self.wing * 2 }

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
            ZStack(alignment: .top) {
                shape
                    .fill(Color.black)
                    .shadow(color: .black.opacity(expanded ? 0.35 : 0), radius: 12, y: 6)
                VStack(spacing: 0) {
                    wings.frame(height: bandHeight)
                    if expanded {
                        transcript.transition(.opacity)
                    }
                }
            }
            .frame(width: expanded ? Self.expandedWidth : compactWidth)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.top, topInset)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .environment(\.colorScheme, .dark)
        .animation(VF.Motion.morph, value: expanded)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityText)
    }

    private var shape: some Shape {
        style == .notch
            ? AnyShape(NotchShape(bottomRadius: expanded ? 22 : 14, earRadius: 6))
            : AnyShape(RoundedRectangle(cornerRadius: expanded ? 24 : bandHeight / 2, style: .continuous))
    }

    /// Left wing: waveform. Middle: the housing (left empty). Right wing: status.
    private var wings: some View {
        HStack(spacing: 0) {
            LiquidWaveformView(history: state.waveform, mode: waveMode)
                .frame(width: Self.wing - 22, height: 14)
                .frame(width: Self.wing)
            Color.clear.frame(width: housing.width)
            IslandStatus(state: state)
                .frame(width: Self.wing)
        }
        .frame(maxWidth: .infinity)
    }

    private var transcript: some View {
        BottomClampLayout(maxHeight: Self.maxTextHeight) {
            Text(text)
                .font(.system(size: 14, weight: .regular, design: .rounded))
                .foregroundStyle(.white.opacity(0.95))
                .lineSpacing(2)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.top, 14)
        .clipped()
        .mask(VStack(spacing: 0) {
            LinearGradient(colors: [.clear, .black], startPoint: .top, endPoint: .bottom).frame(height: 14)
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

/// Trailing-wing status: a recording timer, a sparkle while working, then a
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

/// Flush-top shape that extends the notch: square top meeting the screen edge
/// with small concave "ears", rounded bottom corners.
struct NotchShape: Shape {
    var bottomRadius: CGFloat
    var earRadius: CGFloat

    var animatableData: AnimatablePair<CGFloat, CGFloat> {
        get { AnimatablePair(bottomRadius, earRadius) }
        set { bottomRadius = newValue.first; earRadius = newValue.second }
    }

    func path(in rect: CGRect) -> Path {
        let e = earRadius, r = min(bottomRadius, rect.height / 2)
        var p = Path()
        p.move(to: CGPoint(x: rect.minX - e, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX + e, y: rect.minY))
        p.addQuadCurve(to: CGPoint(x: rect.maxX, y: rect.minY + e), control: CGPoint(x: rect.maxX, y: rect.minY))
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY - r))
        p.addQuadCurve(to: CGPoint(x: rect.maxX - r, y: rect.maxY), control: CGPoint(x: rect.maxX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX + r, y: rect.maxY))
        p.addQuadCurve(to: CGPoint(x: rect.minX, y: rect.maxY - r), control: CGPoint(x: rect.minX, y: rect.maxY))
        p.addLine(to: CGPoint(x: rect.minX, y: rect.minY + e))
        p.addQuadCurve(to: CGPoint(x: rect.minX - e, y: rect.minY), control: CGPoint(x: rect.minX, y: rect.minY))
        p.closeSubpath()
        return p
    }
}
