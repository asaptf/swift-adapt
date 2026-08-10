import AdaptMacrosPlugin
import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest

final class PersonalizableExpansionTests: XCTestCase {
    private let macros: [String: Macro.Type] = [
        "Personalizable": PersonalizableMacro.self,
        "Prompt": PromptMacro.self,
        "Completion": CompletionMacro.self,
    ]

    func testHappyPathExpansion() {
        assertMacroExpansion(
            """
            @Personalizable(task: "email-style")
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

                extension EmailDraft: PersonalizationSignal {
                    static var personalizationTaskID: String {
                        "email-style"
                    }

                    func makeTrainingExample(
                        source: SignalSource,
                        weight: Double?,
                        id: UUID,
                        capturedAt: Date
                    ) -> TrainingExample {
                        TrainingExample(
                            id: id,
                            prompt: self.context,
                            completion: self.body,
                            weight: weight,
                            capturedAt: capturedAt,
                            source: source
                        )
                    }
                }
                """,
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }

    func testPublicStructPreservesAccess() {
        assertMacroExpansion(
            """
            @Personalizable(task: "email-style")
            public struct EmailDraft {
                @Prompt public var context: String
                @Completion public var body: String
            }
            """,
            expandedSource: """
                public struct EmailDraft {
                    public var context: String
                    public var body: String
                }

                extension EmailDraft: PersonalizationSignal {
                    public static var personalizationTaskID: String {
                        "email-style"
                    }

                    public func makeTrainingExample(
                        source: SignalSource,
                        weight: Double?,
                        id: UUID,
                        capturedAt: Date
                    ) -> TrainingExample {
                        TrainingExample(
                            id: id,
                            prompt: self.context,
                            completion: self.body,
                            weight: weight,
                            capturedAt: capturedAt,
                            source: source
                        )
                    }
                }
                """,
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }

    func testMultiplePromptsJoinWithBlankLine() {
        assertMacroExpansion(
            """
            @Personalizable(task: "reply")
            struct ReplyDraft {
                @Prompt var thread: String
                @Prompt var recipient: String
                @Completion var body: String
            }
            """,
            expandedSource: """
                struct ReplyDraft {
                    var thread: String
                    var recipient: String
                    var body: String
                }

                extension ReplyDraft: PersonalizationSignal {
                    static var personalizationTaskID: String {
                        "reply"
                    }

                    func makeTrainingExample(
                        source: SignalSource,
                        weight: Double?,
                        id: UUID,
                        capturedAt: Date
                    ) -> TrainingExample {
                        TrainingExample(
                            id: id,
                            prompt: [self.thread, self.recipient].joined(separator: "\\n\\n"),
                            completion: self.body,
                            weight: weight,
                            capturedAt: capturedAt,
                            source: source
                        )
                    }
                }
                """,
            macros: macros,
            indentationWidth: .spaces(4)
        )
    }
}
