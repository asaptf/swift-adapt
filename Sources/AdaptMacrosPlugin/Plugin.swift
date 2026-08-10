import SwiftCompilerPlugin
import SwiftSyntaxMacros

@main
struct AdaptMacrosPlugin: CompilerPlugin {
    let providingMacros: [Macro.Type] = [
        PersonalizableMacro.self,
        PromptMacro.self,
        CompletionMacro.self,
    ]
}
