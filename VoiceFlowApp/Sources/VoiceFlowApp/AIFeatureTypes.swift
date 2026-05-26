import Foundation

// MARK: - Intent classifier types
//
// Mirror of the Rust types in voiceflow-core/src/text_normalize/intent_classifier.rs.
// Decoded from JSON returned by `voiceflow_classify_intent`.

enum CommandKind: String, Codable {
    case rewrite
    case proofread
    case shorten
    case bullet
    case `continue`
    case summarize
    case reply
    case explain
    case draft
    case question
}

enum IntentKind: Equatable {
    case verbatim
    case inlineCorrection
    case retroactiveCorrection
    case command(CommandKind)
}

extension IntentKind: Codable {
    private enum CodingKeys: String, CodingKey { case kind, command }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch try c.decode(String.self, forKey: .kind) {
        case "verbatim": self = .verbatim
        case "inline_correction": self = .inlineCorrection
        case "retroactive_correction": self = .retroactiveCorrection
        case "command":
            let cmd = try c.decode(CommandKind.self, forKey: .command)
            self = .command(cmd)
        case let other:
            throw DecodingError.dataCorruptedError(
                forKey: .kind, in: c, debugDescription: "Unknown intent kind: \(other)"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .verbatim: try c.encode("verbatim", forKey: .kind)
        case .inlineCorrection: try c.encode("inline_correction", forKey: .kind)
        case .retroactiveCorrection: try c.encode("retroactive_correction", forKey: .kind)
        case .command(let k):
            try c.encode("command", forKey: .kind)
            try c.encode(k, forKey: .command)
        }
    }
}

struct AnchorHint: Codable {
    let find: String
    let replace: String
}

struct IntentResult: Codable {
    let kind: IntentKind
    let residual: String
    let anchorHint: AnchorHint?

    enum CodingKeys: String, CodingKey {
        case kind, residual, anchorHint = "anchor_hint"
    }
}

// MARK: - Retroactive edit types
//
// Mirror of voiceflow-core/src/llm/structured_edit.rs.

enum EditAction: String, Codable {
    case replaceRange = "replace_range"
    case insert
    case delete
    case noOp = "no_op"
}

enum Occurrence: String, Codable {
    case first
    case last
    case only
}

struct Edit: Codable {
    let action: EditAction
    let anchor: String
    let occurrence: Occurrence
    let replacement: String
    let confidence: Float
    let explanation: String
}

struct RetroactiveInput: Codable {
    let fieldText: String
    let fieldSource: String
    let recentInsertions: [String]
    let userUtterance: String

    enum CodingKeys: String, CodingKey {
        case fieldText = "field_text"
        case fieldSource = "field_source"
        case recentInsertions = "recent_insertions"
        case userUtterance = "user_utterance"
    }
}

// MARK: - AI voice command types

struct CommandInput: Codable {
    let command: CommandKind
    let parameter: String
    let selection: String
    let fieldText: String
    let fieldSource: String

    enum CodingKeys: String, CodingKey {
        case command, parameter, selection
        case fieldText = "field_text"
        case fieldSource = "field_source"
    }
}

struct CommandOutput: Codable {
    let output: String
    let abstained: Bool
}
