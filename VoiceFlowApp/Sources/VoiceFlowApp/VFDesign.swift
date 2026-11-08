import SwiftUI
import AppKit

/// VoiceFlow's shared visual language. Every surface (overlay, settings,
/// setup, panels) draws colors, radii and glass from here so the app reads
/// as one object family instead of per-screen one-offs.
enum VF {
    // MARK: Palette — derived from the logo's cyan/teal, plus one cool violet
    // used only inside the voice waveform for depth.
    enum Palette {
        static let aqua   = Color(red: 0.24, green: 0.78, blue: 0.93)
        static let lagoon = Color(red: 0.30, green: 0.55, blue: 1.00)
        static let mint   = Color(red: 0.50, green: 0.96, blue: 0.88)
        static let iris   = Color(red: 0.60, green: 0.48, blue: 1.00)
        /// Deep accent for filled controls carrying white text.
        static let ocean  = Color(red: 0.07, green: 0.50, blue: 0.78)

        static let success = Color(red: 0.36, green: 0.84, blue: 0.56)
        static let warning = Color(red: 1.00, green: 0.75, blue: 0.35)
        static let danger  = Color(red: 1.00, green: 0.47, blue: 0.43)
    }

    static let accent = Palette.aqua

    /// Fill for primary buttons and selected states. Deep enough that white
    /// label text keeps contrast.
    static let accentGradient = LinearGradient(
        colors: [Color(red: 0.05, green: 0.56, blue: 0.76), Color(red: 0.20, green: 0.38, blue: 0.92)],
        startPoint: .topLeading, endPoint: .bottomTrailing)

    /// Bright luminous gradient for icons and accents on dark glass.
    static let glowGradient = LinearGradient(
        colors: [Palette.mint, Palette.aqua, Palette.lagoon],
        startPoint: .topLeading, endPoint: .bottomTrailing)

    // MARK: Surfaces (dark vibrant windows)
    static let cardCorner: CGFloat = Radius.card
    static let pillCorner: CGFloat = 18
    /// Fill for grouped rows and option tiles inside dark vibrant windows.
    static let surface = Color.white.opacity(0.06)
    static let surfaceStroke = Color.white.opacity(0.10)
    static let sidebarHover = Color.white.opacity(0.05)
    static let sidebarSelected = Color.white.opacity(0.10)

    /// Specular highlight stroke for glass surfaces — brighter at top.
    static let glassStroke = LinearGradient(
        colors: [Color.white.opacity(0.28), Color.white.opacity(0.06)],
        startPoint: .top, endPoint: .bottom)

    /// Inner top highlight — a subtle gleam at the top edge of cards.
    static let glassGleam = LinearGradient(
        colors: [Color.white.opacity(0.10), Color.white.opacity(0.0)],
        startPoint: .top, endPoint: .center)

    // MARK: Geometry
    enum Radius {
        static let control: CGFloat = 8
        static let field: CGFloat = 10
        static let card: CGFloat = 16
        static let panel: CGFloat = 22
    }

    enum Space {
        static let xs: CGFloat = 4
        static let s: CGFloat = 8
        static let m: CGFloat = 12
        static let l: CGFloat = 16
        static let xl: CGFloat = 24
    }

    // MARK: Type
    enum Font {
        static let title = SwiftUI.Font.system(size: 20, weight: .semibold, design: .rounded)
        static let heading = SwiftUI.Font.system(size: 13, weight: .semibold)
        static let body = SwiftUI.Font.system(size: 13)
        static let caption = SwiftUI.Font.system(size: 11)
        static let mono = SwiftUI.Font.system(size: 11, weight: .medium, design: .monospaced)
        static let transcript = SwiftUI.Font.system(size: 15, weight: .regular, design: .rounded)
    }

    // MARK: Motion
    enum Motion {
        static let morph = Animation.spring(response: 0.42, dampingFraction: 0.82)
        static let quick = Animation.easeOut(duration: 0.15)
    }
}

// MARK: - Glass surfaces

extension View {
    /// Native Liquid Glass on macOS 26+, a translucent material with a hairline
    /// edge on earlier systems. Use for floating surfaces (overlay, panels, HUDs).
    @ViewBuilder
    func vfGlass<S: Shape>(in shape: S, tint: Color? = nil) -> some View {
        if #available(macOS 26, *) {
            self.glassEffect(tint.map { Glass.regular.tint($0) } ?? .regular, in: shape)
        } else {
            self
                .background(shape.fill(.ultraThinMaterial))
                .background(shape.fill((tint ?? .clear).opacity(0.5)))
                .overlay(shape.stroke(.white.opacity(0.18), lineWidth: 1))
                .shadow(color: .black.opacity(0.25), radius: 18, y: 8)
        }
    }
}

extension View {
    /// The app's primary action: native glass-prominent on macOS 26+, tinted
    /// bordered-prominent before. Native styles keep focus, default-button and
    /// disabled behavior.
    @ViewBuilder
    func vfPrimaryButton() -> some View {
        if #available(macOS 26, *) {
            self.buttonStyle(.glassProminent).tint(VF.Palette.ocean)
        } else {
            self.buttonStyle(.borderedProminent).tint(VF.Palette.ocean)
        }
    }

    /// Window chrome shared by Settings, Setup, Logs and Tips: dark vibrant
    /// backdrop, dark scheme, brand tint.
    func vfWindowChrome(material: NSVisualEffectView.Material = .hudWindow) -> some View {
        self
            .background(VisualEffectBackground(material: material, blending: .behindWindow).ignoresSafeArea())
            .preferredColorScheme(.dark)
            .tint(VF.accent)
    }
}

/// Small keycap label, e.g. ⌥ Space, used for shortcut hints everywhere.
struct VFKeycap: View {
    let key: String
    var body: some View {
        Text(key)
            .font(VF.Font.mono)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(RoundedRectangle(cornerRadius: 5, style: .continuous).fill(Color.primary.opacity(0.10)))
            .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).strokeBorder(Color.primary.opacity(0.12)))
    }
}

// MARK: - Components
/// NSVisualEffectView wrapper. Gives windows a vibrant dark backdrop.
struct VisualEffectBackground: NSViewRepresentable {
    var material: NSVisualEffectView.Material = .hudWindow
    var blending: NSVisualEffectView.BlendingMode = .behindWindow
    var emphasized: Bool = false

    func makeNSView(context: Context) -> NSVisualEffectView {
        let v = NSVisualEffectView()
        v.material = material
        v.blendingMode = blending
        v.state = .active
        v.isEmphasized = emphasized
        v.appearance = NSAppearance(named: .vibrantDark)
        return v
    }

    func updateNSView(_ nsView: NSVisualEffectView, context: Context) {
        nsView.material = material
        nsView.blendingMode = blending
        nsView.isEmphasized = emphasized
    }
}

/// Frosted "Liquid Glass" rounded card surface.
/// Layers: ultraThinMaterial base + faint white tint + top-edge gleam + gradient highlight stroke.
struct VFCard<Content: View>: View {
    var padding: CGFloat = 18
    var corner: CGFloat = VF.cardCorner
    @ViewBuilder var content: () -> Content

    var body: some View {
        content()
            .padding(padding)
            .background {
                ZStack {
                    RoundedRectangle(cornerRadius: corner, style: .continuous)
                        .fill(.ultraThinMaterial)
                    RoundedRectangle(cornerRadius: corner, style: .continuous)
                        .fill(Color.white.opacity(0.04))
                    RoundedRectangle(cornerRadius: corner, style: .continuous)
                        .fill(VF.glassGleam)
                        .blendMode(.plusLighter)
                        .opacity(0.6)
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: corner, style: .continuous)
                    .strokeBorder(VF.glassStroke, lineWidth: 1)
            }
            .shadow(color: Color.black.opacity(0.25), radius: 18, x: 0, y: 8)
    }
}

/// Page header: large title + caption subtitle, optional trailing slot.
struct VFPageHeader<Trailing: View>: View {
    let title: String
    let subtitle: String?
    @ViewBuilder var trailing: () -> Trailing

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 22, weight: .bold))
                if let s = subtitle, !s.isEmpty {
                    Text(s)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
            Spacer()
            trailing()
        }
    }
}

extension VFPageHeader where Trailing == EmptyView {
    init(title: String, subtitle: String? = nil) {
        self.title = title
        self.subtitle = subtitle
        self.trailing = { EmptyView() }
    }
}

/// Section header within a page — caption-sized, breaks up cards.
struct VFSectionHeader: View {
    let title: String
    var icon: String? = nil

    var body: some View {
        HStack(spacing: 6) {
            if let icon {
                Image(systemName: icon)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.secondary)
            }
            Text(title.uppercased())
                .font(.system(size: 11, weight: .semibold))
                .tracking(0.8)
                .foregroundColor(.secondary)
        }
    }
}

/// Filled pill button — primary action on a row or in a header.
struct VFFilledPillStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .foregroundColor(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 7)
            .background(
                Capsule()
                    .fill(VF.accentGradient)
                    .opacity(configuration.isPressed ? 0.85 : 1.0)
            )
    }
}

/// Outlined pill button — secondary action ("Open", "Install").
struct VFOutlinePillStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12, weight: .semibold))
            .foregroundColor(.white.opacity(0.92))
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .stroke(Color.white.opacity(0.22), lineWidth: 1)
                    .background(
                        Capsule().fill(Color.white.opacity(configuration.isPressed ? 0.10 : 0.04))
                    )
            )
    }
}

/// Small ghost pill — caption sized, used inline with body text.
struct VFGhostPillStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 11, weight: .medium))
            .foregroundColor(.white.opacity(0.75))
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(
                Capsule().fill(Color.white.opacity(configuration.isPressed ? 0.12 : 0.06))
            )
    }
}
