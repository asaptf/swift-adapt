import AdaptMacrosPlugin
import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

final class PersonalizableDiagnosticTests: XCTestCase {
    private let macros: [String: Macro.Type] = [
        "Personalizable": PersonalizableMacro.self,
        "Prompt": PromptMacro.self,
        "Completion": CompletionMacro.self,
    ]

    func testZeroCompletionsEmitsDistinctMessage() {
        assertMacroExpansion(
            """
            @Personalizable(task: "email-style")
            struct EmailDraft {
                @Prompt var context: String
            }
            """,
            expandedSource: """
                struct EmailDraft {
                    var context: String
                }
                """,
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "@Personalizable requires exactly one @Completion property; found none on 'EmailDraft'. Mark the property that holds the expected model output with @Completion.",
                    line: 1,
                    column: 1,
                    severity: .error
                )
            ],
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }

    func testTwoCompletionsEmitsDistinctMessage() {
        assertMacroExpansion(
            """
            @Personalizable(task: "email-style")
            struct EmailDraft {
                @Prompt var context: String
                @Completion var body: String
                @Completion var subject: String
            }
            """,
            expandedSource: """
                struct EmailDraft {
                    var context: String
                    var body: String
                    var subject: String
                }
                """,
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "@Personalizable requires exactly one @Completion property; found 2 on 'EmailDraft' ('body', 'subject'). Keep a single @Completion and mark inputs with @Prompt.",
                    line: 1,
                    column: 1,
                    severity: .error
                )
            ],
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }

    func testZeroPromptsEmitsMessage() {
        assertMacroExpansion(
            """
            @Personalizable(task: "email-style")
            struct EmailDraft {
                @Completion var body: String
            }
            """,
            expandedSource: """
                struct EmailDraft {
                    var body: String
                }
                """,
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "@Personalizable requires at least one @Prompt property; found none on 'EmailDraft'. Mark context/input fields with @Prompt.",
                    line: 1,
                    column: 1,
                    severity: .error
                )
            ],
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }

    func testAppliedToClassEmitsMessageWithFixIt() {
        assertMacroExpansion(
            """
            @Personalizable(task: "email-style")
            class EmailDraft {
                @Prompt var context: String
                @Completion var body: String
            }
            """,
            expandedSource: """
                class EmailDraft {
                    var context: String
                    var body: String
                }
                """,
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "@Personalizable can only be applied to a struct; 'EmailDraft' is a class. Change 'class' to 'struct'.",
                    line: 1,
                    column: 1,
                    severity: .error,
                    fixIts: [
                        FixItSpec(message: "change to 'struct'")
                    ]
                )
            ],
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }

    func testAppliedToEnumEmitsMessageWithFixIt() {
        assertMacroExpansion(
            """
            @Personalizable(task: "email-style")
            enum EmailDraft {
                @Prompt var context: String
                @Completion var body: String
            }
            """,
            expandedSource: """
                enum EmailDraft {
                    var context: String
                    var body: String
                }
                """,
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "@Personalizable can only be applied to a struct; 'EmailDraft' is a enum. Change 'enum' to 'struct'.",
                    line: 1,
                    column: 1,
                    severity: .error,
                    fixIts: [
                        FixItSpec(message: "change to 'struct'")
                    ]
                )
            ],
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }

    func testClassFixItApplies() {
        assertMacroExpansion(
            """
            @Personalizable(task: "email-style")
            class EmailDraft {
                @Prompt var context: String
                @Completion var body: String
            }
            """,
            expandedSource: """
                class EmailDraft {
                    var context: String
                    var body: String
                }
                """,
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "@Personalizable can only be applied to a struct; 'EmailDraft' is a class. Change 'class' to 'struct'.",
                    line: 1,
                    column: 1,
                    severity: .error,
                    fixIts: [
                        FixItSpec(message: "change to 'struct'")
                    ]
                )
            ],
            macros: macros,
            applyFixIts: ["change to 'struct'"],
            fixedSource: """
                @Personalizable(task: "email-style")
                struct EmailDraft {
                    @Prompt var context: String
                    @Completion var body: String
                }
                """,
            indentationWidth: .spaces(4)
        )
    }

    func testUnsupportedTypeNamesProperty() {
        assertMacroExpansion(
            """
            @Personalizable(task: "email-style")
            struct EmailDraft {
                @Prompt var context: String
                @Completion var wordCount: Int
            }
            """,
            expandedSource: """
                struct EmailDraft {
                    var context: String
                    var wordCount: Int
                }
                """,
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "Property 'wordCount' has type 'Int', which @Personalizable cannot serialise. Change it to String (or convert the value into a String property before capture).",
                    line: 4,
                    column: 32,
                    severity: .error
                )
            ],
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }

    func testMissingTaskArgument() {
        assertMacroExpansion(
            """
            @Personalizable
            struct EmailDraft {
                @Prompt var context: String
                @Completion var body: String
            }
            """,
            expandedSource: """
                struct EmailDraft {
                    var context: String
                    var body: String
                }
                """,
            diagnostics: [
                DiagnosticSpec(
                    message:
                        #"@Personalizable requires a task identifier, e.g. @Personalizable(task: "email-style")."#,
                    line: 1,
                    column: 1,
                    severity: .error
                )
            ],
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }

    func testMissingTypeAnnotation() {
        assertMacroExpansion(
            """
            @Personalizable(task: "email-style")
            struct EmailDraft {
                @Prompt var context = "x"
                @Completion var body: String
            }
            """,
            expandedSource: """
                struct EmailDraft {
                    var context = "x"
                    var body: String
                }
                """,
            diagnostics: [
                DiagnosticSpec(
                    message:
                        "Property 'context' must have an explicit type annotation; @Personalizable only serialises String.",
                    line: 3,
                    column: 17,
                    severity: .error
                )
            ],
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }
}
