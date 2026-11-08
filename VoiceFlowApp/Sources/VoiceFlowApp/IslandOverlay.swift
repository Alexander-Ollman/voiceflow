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

/// Space the system is using around the camera housing right now, as extra
/// width beyond the housing on each side. macOS draws privacy indicators
/// (e.g. the microphone dot while VoiceFlow records) as small Window Server
/// windows beside the Dynamic Island; we keep our wings outside them.
final class SystemIslandGeometry: ObservableObject {
    @Published var leftExtra: CGFloat = 0
    @Published var rightExtra: CGFloat = 0

    /// Measured expansion of the system island while a privacy indicator shows.
    static let expandedLeft: CGFloat = 38
    static let expandedRight: CGFloat = 49
    static let trailingPastIndicator: CGFloat = 25

    /// Re-measure against `housing` (screen coordinates) on `screen`.
    func refresh(housing: CGRect, screen: NSScreen) {
        let primaryHeight = NSScreen.screens.first?.frame.height ?? screen.frame.height
        let screenTopCG = primaryHeight - screen.frame.maxY          // CG coords: origin top-left
        let search = housing.insetBy(dx: -90, dy: 0)
        var minX = housing.minX, maxX = housing.maxX
        var indicatorMaxX: CGFloat?
        let windows = CGWindowListCopyWindowInfo([.optionOnScreenOnly], kCGNullWindowID) as? [[String: Any]] ?? []
        for window in windows {
            guard (window[kCGWindowOwnerName as String] as? String) == "Window Server",
                  let layer = window[kCGWindowLayer as String] as? Int, layer > 1000,
                  let b = window[kCGWindowBounds as String] as? [String: CGFloat],
                  let x = b["X"], let y = b["Y"], let w = b["Width"], let h = b["Height"],
                  w > 2, w < 200, h > 2,
                  y - screenTopCG < housing.height + 4,                 // within the band
                  x < search.maxX, x + w > search.minX else { continue }
            minX = min(minX, x)
            maxX = max(maxX, x + w)
            indicatorMaxX = max(indicatorMaxX ?? 0, x + w)
        }
        // The indicator window is only the dot. Around it, the system expands
        // the island itself (drawn by the Window Server, not enumerable). On a
        // Mac17,13 / macOS 27.2 with the mic in use it spans 38 pt left of the
        // cutout and ~25 pt right of the dot (49 pt right of the cutout).
        if let indicatorMaxX {
            minX = min(minX, housing.minX - Self.expandedLeft)
            maxX = max(maxX, indicatorMaxX + Self.trailingPastIndicator, housing.maxX + Self.expandedRight)
        }
        // A little air between the system pill and our wing content.
        let margin: CGFloat = indicatorMaxX == nil ? 0 : 6
        let left = (housing.minX - minX + margin).rounded(), right = (maxX - housing.maxX + margin).rounded()
        if left != leftExtra { leftExtra = left }
        if right != rightExtra { rightExtra = right }

        // Menu-bar room beside what the system occupies, so wings never cover
        // menus (left) or status items (right).
        let occupiedMin = housing.minX - left, occupiedMax = housing.maxX + right
        var statusMinX = screen.frame.maxX
        for window in windows {
            guard let layer = window[kCGWindowLayer as String] as? Int,
                  layer == Int(CGWindowLevelForKey(.statusWindow)),
                  (window[kCGWindowOwnerName as String] as? String) != "Window Server",
                  let b = window[kCGWindowBounds as String] as? [String: CGFloat],
                  let x = b["X"], let w = b["Width"], let y = b["Y"],
                  y - screenTopCG < housing.height, w < 300, x >= housing.maxX - 1 else { continue }
            statusMinX = min(statusMinX, x)
        }
        let room = (left: occupiedMin - Self.appMenusMaxX(screen: screen, primaryHeight: primaryHeight),
                    right: statusMinX - occupiedMax)
        if room.left != leftRoom { leftRoom = room.left }
        if room.right != rightRoom { rightRoom = room.right }
    }

    /// Room in the menu bar left and right of what the system occupies.
    @Published var leftRoom: CGFloat = .greatestFiniteMagnitude
    @Published var rightRoom: CGFloat = .greatestFiniteMagnitude

    private static var menuCache: (pid: pid_t, at: Date, maxX: CGFloat)?

    /// Right edge of the frontmost app's menu titles, read through
    /// Accessibility (cached per app for a second). Without Accessibility,
    /// assume the left half of the menu bar is in use.
    private static func appMenusMaxX(screen: NSScreen, primaryHeight: CGFloat) -> CGFloat {
        guard let app = NSWorkspace.shared.frontmostApplication else { return screen.frame.minX }
        if let cache = menuCache, cache.pid == app.processIdentifier, Date().timeIntervalSince(cache.at) < 1 {
            return cache.maxX
        }
        var maxX = screen.frame.midX - 80
        if AXIsProcessTrusted() {
            maxX = screen.frame.minX
            let element = AXUIElementCreateApplication(app.processIdentifier)
            var bar: CFTypeRef?
            if AXUIElementCopyAttributeValue(element, kAXMenuBarAttribute as CFString, &bar) == .success,
               let bar, CFGetTypeID(bar) == AXUIElementGetTypeID() {
                var children: CFTypeRef?
                AXUIElementCopyAttributeValue(bar as! AXUIElement, kAXChildrenAttribute as CFString, &children)
                for item in (children as? [AXUIElement]) ?? [] {
                    var posRef: CFTypeRef?, sizeRef: CFTypeRef?
                    var pos = CGPoint.zero, size = CGSize.zero
                    if AXUIElementCopyAttributeValue(item, kAXPositionAttribute as CFString, &posRef) == .success,
                       AXUIElementCopyAttributeValue(item, kAXSizeAttribute as CFString, &sizeRef) == .success,
                       let posRef, let sizeRef {
                        AXValueGetValue(posRef as! AXValue, .cgPoint, &pos)
                        AXValueGetValue(sizeRef as! AXValue, .cgSize, &size)
                        // Only titles on this screen, left of centre.
                        if pos.x >= screen.frame.minX, pos.x < screen.frame.midX { maxX = max(maxX, pos.x + size.width) }
                    }
                }
            }
        }
        menuCache = (app.processIdentifier, Date(), maxX)
        return maxX
    }
}

/// The voice overlay attached to the camera housing.
///
/// - Island: wings either side of the Dynamic Island, sized around the live
///   extent of the system's own island content (SystemIslandGeometry), so the
///   microphone privacy indicator is never covered; one black pill sits
///   behind the system island and the system draws above it.
/// - Notch: the notch is fixed hardware with menu-bar items beside it, so the
///   shape is exactly notch-wide inside the menu-bar band, then widens below.
/// Waveform left, status right; live text expands below.
struct IslandOverlayView: View {
    @ObservedObject var state: OverlayState
    let style: OverlayPlacement          // .island or .notch
    /// Housing size in points (from NSScreen.cameraHousingFrame).
    let housing: CGSize
    /// Live extent of the system's own island content (island style only).
    @ObservedObject var system: SystemIslandGeometry

    /// Island style: the system island band sits ~5–43 pt below the top edge.
    static let islandTop: CGFloat = 5
    static let islandWing: CGFloat = 72

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
            if wingsFit {
                besideBody
            } else if style == .island {
                islandBelowBody
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

    // MARK: Island: wings beside the system island

    /// Width the system is using: the housing plus any indicator beside it.
    private var occupiedWidth: CGFloat { housing.width + system.leftExtra + system.rightExtra }
    private var bandHeight: CGFloat { housing.height - 2 }

    /// One black pill behind the system island with our wings on either side;
    /// the system draws its indicators above it, so they are never hidden.
    /// Wings go beside the housing only when the menu bar has room for them
    /// on both sides; otherwise the overlay hangs below instead.
    private var wingsFit: Bool {
        system.leftRoom >= Self.islandWing + 4 && system.rightRoom >= Self.islandWing + 4
    }

    /// Wings beside what the system occupies. Island: one black pill behind
    /// the system island. Notch: flush with the top edge, with ears, like the
    /// notch grown sideways. The system draws its indicators above ours.
    private var besideBody: some View {
        let compactWidth = occupiedWidth + Self.islandWing * 2
        let rowHeight = style == .island ? bandHeight : housing.height
        return VStack(spacing: 6) {
            HStack(spacing: 0) {
                LiquidWaveformView(history: state.waveform, mode: waveMode)
                    .frame(width: Self.islandWing - 20, height: 16)
                    .frame(width: Self.islandWing)
                Color.clear.frame(width: occupiedWidth)
                IslandStatus(state: state)
                    .frame(width: Self.islandWing)
            }
            .frame(width: compactWidth, height: rowHeight)
            .background(wingsBackground(rowHeight: rowHeight))
            // Keep the gap centred on what the system occupies.
            .offset(x: (system.rightExtra - system.leftExtra) / 2)

            // Live text drops into its own card below the menu bar.
            if expanded {
                transcript
                    .frame(width: Self.expandedWidth)
                    .background(
                        RoundedRectangle(cornerRadius: 20, style: .continuous)
                            .fill(Color.black)
                            .shadow(color: .black.opacity(0.3), radius: 10, y: 4))
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(.top, style == .island ? Self.islandTop : 0)
        .animation(VF.Motion.morph, value: system.leftExtra)
        .animation(VF.Motion.morph, value: system.rightExtra)
    }

    @ViewBuilder
    private func wingsBackground(rowHeight: CGFloat) -> some View {
        if style == .island {
            Capsule(style: .continuous).fill(Color.black)
        } else {
            NotchWingsShape(bottomRadius: 12, earRadius: 6).fill(Color.black)
        }
    }

    /// Fallback when the menu bar is too full: a separate pill under the island.
    private var islandBelowBody: some View {
        VStack(spacing: 0) {
            Color.clear.frame(height: housing.height + Self.islandGap)
            content
                .background(
                    RoundedRectangle(cornerRadius: expanded ? 22 : Self.rowHeight / 2, style: .continuous)
                        .fill(Color.black)
                        .shadow(color: .black.opacity(0.3), radius: 10, y: 4))
        }
    }

    // MARK: Fallback: body below the housing

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

/// Notch grown sideways: flush with the top edge (square top with small
/// concave ears where it meets the screen edge), rounded bottom corners.
struct NotchWingsShape: Shape {
    var bottomRadius: CGFloat
    var earRadius: CGFloat

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

/// Panel for the camera-housing overlay. AppKit normally constrains windows
/// to stay below the menu bar, which pushed the overlay ~45 pt down in
/// full-screen spaces; this one must sit exactly at the top edge.
final class TopEdgePanel: NSPanel {
    override func constrainFrameRect(_ frameRect: NSRect, to screen: NSScreen?) -> NSRect { frameRect }
}
