import SwiftSyntax
import SwiftSyntaxMacros

/// Marker peer macro for `@Prompt` — expansion is empty; ``PersonalizableMacro`` reads the attribute.
public struct PromptMacro: PeerMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        []
    }
}

/// Marker peer macro for `@Completion` — expansion is empty; ``PersonalizableMacro`` reads the attribute.
public struct CompletionMacro: PeerMacro {
    public static func expansion(
        of node: AttributeSyntax,
        providingPeersOf declaration: some DeclSyntaxProtocol,
        in context: some MacroExpansionContext
    ) throws -> [DeclSyntax] {
        []
    }
}
