import AdaptCore
import Testing

/// Stub with a synthetic chat template for offline formatter tests.
///
/// Records the last `additionalContext` so tests can assert template variables
/// (e.g. `enable_thinking`) reach the renderer.
private final class TemplateStubTokenizer: SFTTokenizing, @unchecked Sendable {
    var hasChatTemplate: Bool
    /// Last `additionalContext` passed to `applyChatTemplate` (for assertions).
    private(set) var lastAdditionalContext: [String: any Sendable]?
    /// Call count for `applyChatTemplate`.
    private(set) var applyChatTemplateCount = 0

    init(hasChatTemplate: Bool = true) {
        self.hasChatTemplate = hasChatTemplate
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        if addSpecialTokens { ids.append(1) }
        for scalar in text.unicodeScalars {
            ids.append(Int(scalar.value % 200) + 10)
        }
        return ids
    }

    func applyChatTemplate(
        messages: [[String: String]],
        addGenerationPrompt: Bool,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        applyChatTemplateCount += 1
        lastAdditionalContext = additionalContext
        guard hasChatTemplate else { throw SFTFormattingError.missingChatTemplate }
        // Markers: 200 bos, 201 user, 202 assistant-open, 203 turn-end.
        // Thinking-mode markers (when enable_thinking is present):
        //   true  → 210 (think-open after assistant-open)
        //   false → 211 (no-think marker after assistant-open)
        // Omitted enable_thinking → no extra marker (template default).
        var ids: [Int] = [200]
        for message in messages {
            let role = message["role"] ?? ""
            let content = message["content"] ?? ""
            ids.append(role == "assistant" ? 202 : 201)
            ids.append(contentsOf: encode(text: content, addSpecialTokens: false))
            ids.append(203)
        }
        if addGenerationPrompt {
            ids.append(202)
            if let enableThinking = additionalContext?["enable_thinking"] as? Bool {
                ids.append(enableThinking ? 210 : 211)
            }
        }
        return ids
    }
}

@Suite("SFTPromptFormatter")
struct SFTPromptFormatterTests {
    @Test("detectConvention uses chatTemplate when tokenizer has one")
    func detectWithTemplate() {
        let tok = TemplateStubTokenizer(hasChatTemplate: true)
        #expect(SFTPromptFormatter.detectConvention(tokenizer: tok) == .chatTemplate)
    }

    @Test("detectConvention falls back to raw when tokenizer has none")
    func detectWithoutTemplate() {
        let tok = TemplateStubTokenizer(hasChatTemplate: false)
        #expect(SFTPromptFormatter.detectConvention(tokenizer: tok) == .rawConcatenation)
    }

    @Test("templated training masks loss to assistant span only")
    func templatedTrainingMaskBoundary() throws {
        let tok = TemplateStubTokenizer(hasChatTemplate: true)
        let formatted = try SFTPromptFormatter.formatTraining(
            prompt: "Hi",
            completion: "Yo",
            tokenizer: tok,
            convention: .chatTemplate
        )
        // Prefix: [200, 201, H, i, 203, 202]
        // Full:   [200, 201, H, i, 203, 202, Y, o, 203]
        let prefix = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: "Hi",
            tokenizer: tok,
            convention: .chatTemplate
        )
        #expect(formatted.promptTokenCount == prefix.count)
        #expect(formatted.tokens.count > formatted.promptTokenCount)
        // Scaffold markers before the assistant content must not be past the boundary
        // in a way that would supervise them: every supervised index is >= promptTokenCount.
        let supervised = Array(formatted.tokens[formatted.promptTokenCount...])
        #expect(!supervised.isEmpty)
        // Assistant content "Yo" ids appear in the supervised span.
        let y = Int(("Y" as UnicodeScalar).value % 200) + 10
        let o = Int(("o" as UnicodeScalar).value % 200) + 10
        #expect(supervised.contains(y))
        #expect(supervised.contains(o))
        // User content must not be in the supervised span.
        let h = Int(("H" as UnicodeScalar).value % 200) + 10
        #expect(!supervised.contains(h))
    }

    @Test("train and generate produce byte-identical prefixes for the same prompt")
    func trainGeneratePrefixInvariant() throws {
        let tok = TemplateStubTokenizer(hasChatTemplate: true)
        let prompt = "Decline a meeting."
        let completion = "Can't make it."

        let trained = try SFTPromptFormatter.formatTraining(
            prompt: prompt,
            completion: completion,
            tokenizer: tok,
            convention: .chatTemplate
        )
        let genPrefix = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: prompt,
            tokenizer: tok,
            convention: .chatTemplate
        )

        // Real assertion: not a comment. Train prompt span == generation prefix.
        #expect(Array(trained.tokens.prefix(trained.promptTokenCount)) == genPrefix)
        #expect(genPrefix == Array(trained.tokens[0..<trained.promptTokenCount]))
    }

    @Test("raw fallback is identical on train and generate when no template")
    func rawFallbackBothSides() throws {
        let tok = TemplateStubTokenizer(hasChatTemplate: false)
        #expect(SFTPromptFormatter.detectConvention(tokenizer: tok) == .rawConcatenation)

        let prompt = "Hello"
        let completion = "World"
        let trained = try SFTPromptFormatter.formatTraining(
            prompt: prompt,
            completion: completion,
            tokenizer: tok,
            convention: .rawConcatenation
        )
        let genPrefix = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: prompt,
            tokenizer: tok,
            convention: .rawConcatenation
        )
        #expect(Array(trained.tokens.prefix(trained.promptTokenCount)) == genPrefix)
        #expect(trained.convention == .rawConcatenation)
    }

    @Test("legacy stored nil promptFormat resolves to rawConcatenation")
    func legacyStoredNil() {
        #expect(SFTPromptFormatter.convention(fromStored: nil) == .rawConcatenation)
        #expect(
            SFTPromptFormatter.convention(fromStored: .chatTemplate) == .chatTemplate
        )
    }

    // MARK: - chatTemplateEnableThinking

    @Test("nil chatTemplateEnableThinking leaves template context empty (default)")
    func enableThinkingNilLeavesContextEmpty() throws {
        let tok = TemplateStubTokenizer(hasChatTemplate: true)
        let defaultPrefix = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: "Hi",
            tokenizer: tok,
            convention: .chatTemplate
        )
        #expect(tok.lastAdditionalContext == nil)
        // No think / no-think marker (210/211) when context is omitted.
        #expect(!defaultPrefix.contains(210))
        #expect(!defaultPrefix.contains(211))

        let again = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: "Hi",
            tokenizer: tok,
            convention: .chatTemplate,
            chatTemplateEnableThinking: nil
        )
        #expect(tok.lastAdditionalContext == nil)
        #expect(again == defaultPrefix)
    }

    @Test("chatTemplateEnableThinking false reaches template as enable_thinking")
    func enableThinkingFalseReachesTemplate() throws {
        let tok = TemplateStubTokenizer(hasChatTemplate: true)
        let ids = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: "Hi",
            tokenizer: tok,
            convention: .chatTemplate,
            chatTemplateEnableThinking: false
        )
        #expect(tok.lastAdditionalContext?["enable_thinking"] as? Bool == false)
        // Stub appends 211 when enable_thinking == false.
        #expect(ids.last == 211)
        #expect(ids.contains(211))
        #expect(!ids.contains(210))
    }

    @Test("chatTemplateEnableThinking true reaches template as enable_thinking")
    func enableThinkingTrueReachesTemplate() throws {
        let tok = TemplateStubTokenizer(hasChatTemplate: true)
        let ids = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: "Hi",
            tokenizer: tok,
            convention: .chatTemplate,
            chatTemplateEnableThinking: true
        )
        #expect(tok.lastAdditionalContext?["enable_thinking"] as? Bool == true)
        #expect(ids.last == 210)
        #expect(!ids.contains(211))
    }

    @Test("true/false enable_thinking changes rendered prefix vs default")
    func enableThinkingChangesRenderedPrefix() throws {
        let tok = TemplateStubTokenizer(hasChatTemplate: true)
        let defaultIDs = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: "Hi",
            tokenizer: tok,
            convention: .chatTemplate,
            chatTemplateEnableThinking: nil
        )
        let offIDs = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: "Hi",
            tokenizer: tok,
            convention: .chatTemplate,
            chatTemplateEnableThinking: false
        )
        let onIDs = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: "Hi",
            tokenizer: tok,
            convention: .chatTemplate,
            chatTemplateEnableThinking: true
        )
        #expect(defaultIDs != offIDs)
        #expect(defaultIDs != onIDs)
        #expect(offIDs != onIDs)
    }

    @Test("chatTemplateEnableThinking under rawConcatenation throws explicitly")
    func enableThinkingUnderRawThrows() {
        let tok = TemplateStubTokenizer(hasChatTemplate: false)
        do {
            _ = try SFTPromptFormatter.formatGenerationPrefix(
                prompt: "Hi",
                tokenizer: tok,
                convention: .rawConcatenation,
                chatTemplateEnableThinking: false
            )
            Issue.record("expected chatTemplateOptionNotApplicable")
        } catch let error as SFTFormattingError {
            guard case .chatTemplateOptionNotApplicable(let message) = error else {
                Issue.record("wrong error case: \(error)")
                return
            }
            #expect(message.contains("chatTemplateEnableThinking"))
            #expect(message.contains("rawConcatenation"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        // Encoder must not have been asked to render a template.
        #expect(tok.applyChatTemplateCount == 0)
    }

    @Test("raw path with nil chatTemplateEnableThinking is unchanged")
    func rawPathNilThinkingUnchanged() throws {
        let tok = TemplateStubTokenizer(hasChatTemplate: false)
        let ids = try SFTPromptFormatter.formatGenerationPrefix(
            prompt: "Hi",
            tokenizer: tok,
            convention: .rawConcatenation,
            chatTemplateEnableThinking: nil
        )
        let expected = tok.encode(text: "Hi", addSpecialTokens: true)
        #expect(ids == expected)
        #expect(tok.applyChatTemplateCount == 0)
    }
}
