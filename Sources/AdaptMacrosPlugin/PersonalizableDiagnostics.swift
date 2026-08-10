import SwiftDiagnostics
import SwiftSyntax

/// Diagnostic messages emitted by ``PersonalizableMacro``.
enum PersonalizableDiagnostic: String, DiagnosticMessage {
    case notAStruct
    case missingTaskArgument
    case missingCompletion
    case multipleCompletions
    case missingPrompt
    case unsupportedPropertyType
    case missingTypeAnnotation

    var severity: DiagnosticSeverity { .error }

    var diagnosticID: MessageID {
        MessageID(domain: "AdaptMacros", id: rawValue)
    }

    var message: String {
        // Fallback; prefer the contextual overloads below.
        switch self {
        case .notAStruct:
            return "@Personalizable can only be applied to a struct."
        case .missingTaskArgument:
            return #"@Personalizable requires a task identifier, e.g. @Personalizable(task: "email-style")."#
        case .missingCompletion:
            return "@Personalizable requires exactly one @Completion property; found none. Mark the property that holds the expected model output with @Completion."
        case .multipleCompletions:
            return "@Personalizable requires exactly one @Completion property. Keep a single @Completion and mark inputs with @Prompt."
        case .missingPrompt:
            return "@Personalizable requires at least one @Prompt property; found none. Mark context/input fields with @Prompt."
        case .unsupportedPropertyType:
            return "Property has a type that @Personalizable cannot serialise. Change it to String."
        case .missingTypeAnnotation:
            return "Property must have an explicit type annotation; @Personalizable only serialises String."
        }
    }

    static func notAStruct(typeName: String, kind: String) -> String {
        "@Personalizable can only be applied to a struct; '\(typeName)' is a \(kind). Change '\(kind)' to 'struct'."
    }

    static func missingCompletion(typeName: String) -> String {
        "@Personalizable requires exactly one @Completion property; found none on '\(typeName)'. Mark the property that holds the expected model output with @Completion."
    }

    static func multipleCompletions(typeName: String, count: Int, names: [String]) -> String {
        let listed = names.map { "'\($0)'" }.joined(separator: ", ")
        return "@Personalizable requires exactly one @Completion property; found \(count) on '\(typeName)' (\(listed)). Keep a single @Completion and mark inputs with @Prompt."
    }

    static func missingPrompt(typeName: String) -> String {
        "@Personalizable requires at least one @Prompt property; found none on '\(typeName)'. Mark context/input fields with @Prompt."
    }

    static func unsupportedPropertyType(property: String, typeName: String) -> String {
        "Property '\(property)' has type '\(typeName)', which @Personalizable cannot serialise. Change it to String (or convert the value into a String property before capture)."
    }

    static func missingTypeAnnotation(property: String) -> String {
        "Property '\(property)' must have an explicit type annotation; @Personalizable only serialises String."
    }
}

/// Fix-it messages for ``PersonalizableMacro``.
enum PersonalizableFixIt: String, FixItMessage {
    case changeToStruct

    var fixItID: MessageID {
        MessageID(domain: "AdaptMacros", id: "fixit.\(rawValue)")
    }

    var message: String {
        switch self {
        case .changeToStruct:
            return "change to 'struct'"
        }
    }
}

extension Diagnostic {
    /// Builds a diagnostic whose message text is fully expanded (for assertMacroExpansion).
    static func personalizable(
        node: some SyntaxProtocol,
        id: PersonalizableDiagnostic,
        message: String,
        fixIts: [FixIt] = []
    ) -> Diagnostic {
        Diagnostic(
            node: Syntax(node),
            message: ContextualMessage(id: id, message: message),
            fixIts: fixIts
        )
    }
}

/// Diagnostic message with a pre-formatted string (not the enum's generic fallback).
private struct ContextualMessage: DiagnosticMessage {
    let id: PersonalizableDiagnostic
    let message: String

    var severity: DiagnosticSeverity { id.severity }
    var diagnosticID: MessageID { id.diagnosticID }
}
