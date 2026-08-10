import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

/// Implements `@Personalizable(task:)` — extension macro synthesizing `PersonalizationSignal`.
public enum PersonalizableMacro: ExtensionMacro {
    public static func expansion(
        of node: AttributeSyntax,
        attachedTo declaration: some DeclGroupSyntax,
        providingExtensionsOf type: some TypeSyntaxProtocol,
        conformingTo protocols: [TypeSyntax],
        in context: some MacroExpansionContext
    ) throws -> [ExtensionDeclSyntax] {
        guard let taskLiteral = extractTaskLiteral(from: node) else {
            context.diagnose(
                .personalizable(
                    node: node,
                    id: .missingTaskArgument,
                    message: PersonalizableDiagnostic.missingTaskArgument.message
                )
            )
            return []
        }

        let typeName: String
        let accessPrefix: String
        let annotatedProperties: [AnnotatedProperty]

        if let structDecl = declaration.as(StructDeclSyntax.self) {
            typeName = structDecl.name.text
            accessPrefix = accessModifierPrefix(from: structDecl.modifiers)
            annotatedProperties = collectAnnotatedProperties(from: structDecl.memberBlock)
        } else if let classDecl = declaration.as(ClassDeclSyntax.self) {
            diagnoseNotAStruct(
                node: node,
                typeName: classDecl.name.text,
                kind: "class",
                keyword: classDecl.classKeyword,
                context: context
            )
            return []
        } else if let enumDecl = declaration.as(EnumDeclSyntax.self) {
            diagnoseNotAStruct(
                node: node,
                typeName: enumDecl.name.text,
                kind: "enum",
                keyword: enumDecl.enumKeyword,
                context: context
            )
            return []
        } else if let actorDecl = declaration.as(ActorDeclSyntax.self) {
            diagnoseNotAStruct(
                node: node,
                typeName: actorDecl.name.text,
                kind: "actor",
                keyword: actorDecl.actorKeyword,
                context: context
            )
            return []
        } else {
            context.diagnose(
                .personalizable(
                    node: node,
                    id: .notAStruct,
                    message: PersonalizableDiagnostic.notAStruct.message
                )
            )
            return []
        }

        let prompts = annotatedProperties.filter { $0.kind == .prompt }
        let completions = annotatedProperties.filter { $0.kind == .completion }

        var hasError = false

        if completions.isEmpty {
            context.diagnose(
                .personalizable(
                    node: node,
                    id: .missingCompletion,
                    message: PersonalizableDiagnostic.missingCompletion(typeName: typeName)
                )
            )
            hasError = true
        } else if completions.count > 1 {
            let names = completions.map(\.name)
            context.diagnose(
                .personalizable(
                    node: node,
                    id: .multipleCompletions,
                    message: PersonalizableDiagnostic.multipleCompletions(
                        typeName: typeName,
                        count: completions.count,
                        names: names
                    )
                )
            )
            hasError = true
        }

        if prompts.isEmpty {
            context.diagnose(
                .personalizable(
                    node: node,
                    id: .missingPrompt,
                    message: PersonalizableDiagnostic.missingPrompt(typeName: typeName)
                )
            )
            hasError = true
        }

        for property in annotatedProperties {
            if property.typeAnnotation == nil {
                context.diagnose(
                    .personalizable(
                        node: property.diagnosticNode,
                        id: .missingTypeAnnotation,
                        message: PersonalizableDiagnostic.missingTypeAnnotation(property: property.name)
                    )
                )
                hasError = true
                continue
            }
            if !property.isSupportedStringType {
                let typeText = property.typeAnnotation?.trimmedDescription ?? "?"
                context.diagnose(
                    .personalizable(
                        node: property.diagnosticNode,
                        id: .unsupportedPropertyType,
                        message: PersonalizableDiagnostic.unsupportedPropertyType(
                            property: property.name,
                            typeName: typeText
                        )
                    )
                )
                hasError = true
            }
        }

        guard !hasError, let completion = completions.first else {
            return []
        }

        let promptExpression = makePromptExpression(prompts: prompts)
        let completionName = TokenSyntax.identifier(completion.name)
        let escapedTask = escapeStringLiteral(taskLiteral)

        let extensionDecl: DeclSyntax = """
            extension \(type.trimmed): PersonalizationSignal {
                \(raw: accessPrefix)static var personalizationTaskID: String {
                    "\(raw: escapedTask)"
                }

                \(raw: accessPrefix)func makeTrainingExample(
                    source: SignalSource,
                    weight: Double?,
                    id: UUID,
                    capturedAt: Date
                ) -> TrainingExample {
                    TrainingExample(
                        id: id,
                        prompt: \(raw: promptExpression),
                        completion: self.\(completionName),
                        weight: weight,
                        capturedAt: capturedAt,
                        source: source
                    )
                }
            }
            """

        guard let ext = extensionDecl.as(ExtensionDeclSyntax.self) else {
            return []
        }
        return [ext]
    }
}

// MARK: - Helpers

private enum AnnotationKind {
    case prompt
    case completion
}

private struct AnnotatedProperty {
    let name: String
    let kind: AnnotationKind
    let typeAnnotation: TypeSyntax?
    let diagnosticNode: Syntax

    var isSupportedStringType: Bool {
        guard let typeAnnotation else { return false }
        let text = typeAnnotation.trimmedDescription
        return text == "String" || text == "Swift.String"
    }
}

private func extractTaskLiteral(from node: AttributeSyntax) -> String? {
    guard case .argumentList(let arguments) = node.arguments else {
        return nil
    }
    // Prefer labeled `task:`; accept a single unlabeled string as well.
    let taskArg =
        arguments.first(where: { $0.label?.text == "task" })
        ?? (arguments.count == 1 ? arguments.first : nil)
    guard let taskArg else { return nil }
    guard let literal = taskArg.expression.as(StringLiteralExprSyntax.self) else {
        return nil
    }
    return literal.representedLiteralValue
}

private func accessModifierPrefix(from modifiers: DeclModifierListSyntax) -> String {
    for modifier in modifiers {
        switch modifier.name.tokenKind {
        case .keyword(.public):
            return "public "
        case .keyword(.package):
            return "package "
        case .keyword(.open):
            // Structs cannot be open; still honour if present on an unusual decl.
            return "public "
        default:
            continue
        }
    }
    return ""
}

private func collectAnnotatedProperties(from memberBlock: MemberBlockSyntax) -> [AnnotatedProperty] {
    var result: [AnnotatedProperty] = []
    for member in memberBlock.members {
        guard let varDecl = member.decl.as(VariableDeclSyntax.self) else { continue }
        // Instance properties only — statics are not capture fields.
        if varDecl.modifiers.contains(where: {
            $0.name.tokenKind == .keyword(.static) || $0.name.tokenKind == .keyword(.class)
        }) {
            continue
        }

        let kind: AnnotationKind?
        if hasAttribute(varDecl.attributes, named: "Prompt") {
            kind = .prompt
        } else if hasAttribute(varDecl.attributes, named: "Completion") {
            kind = .completion
        } else {
            kind = nil
        }
        guard let kind else { continue }

        for binding in varDecl.bindings {
            guard let pattern = binding.pattern.as(IdentifierPatternSyntax.self) else {
                continue
            }
            let diagnosticNode: Syntax
            if let typeNode = binding.typeAnnotation?.type {
                diagnosticNode = Syntax(typeNode)
            } else {
                diagnosticNode = Syntax(pattern)
            }
            result.append(
                AnnotatedProperty(
                    name: pattern.identifier.text,
                    kind: kind,
                    typeAnnotation: binding.typeAnnotation?.type,
                    diagnosticNode: diagnosticNode
                )
            )
        }
    }
    return result
}

private func hasAttribute(_ attributes: AttributeListSyntax, named name: String) -> Bool {
    for element in attributes {
        guard let attr = element.as(AttributeSyntax.self) else { continue }
        if attributeName(attr) == name {
            return true
        }
    }
    return false
}

private func attributeName(_ attr: AttributeSyntax) -> String {
    if let ident = attr.attributeName.as(IdentifierTypeSyntax.self) {
        return ident.name.text
    }
    if let member = attr.attributeName.as(MemberTypeSyntax.self) {
        return member.name.text
    }
    return attr.attributeName.trimmedDescription
}

private func makePromptExpression(prompts: [AnnotatedProperty]) -> String {
    if prompts.count == 1 {
        return "self.\(prompts[0].name)"
    }
    let parts = prompts.map { "self.\($0.name)" }.joined(separator: ", ")
    return "[\(parts)].joined(separator: \"\\n\\n\")"
}

private func escapeStringLiteral(_ value: String) -> String {
    var result = ""
    for ch in value {
        switch ch {
        case "\\": result += "\\\\"
        case "\"": result += "\\\""
        case "\n": result += "\\n"
        case "\r": result += "\\r"
        case "\t": result += "\\t"
        default: result.append(ch)
        }
    }
    return result
}

private func diagnoseNotAStruct(
    node: AttributeSyntax,
    typeName: String,
    kind: String,
    keyword: TokenSyntax,
    context: some MacroExpansionContext
) {
    let message = PersonalizableDiagnostic.notAStruct(typeName: typeName, kind: kind)
    // Preserve leading trivia (including the newline after the attribute line) so
    // the fixed source keeps layout.
    let structKeyword = TokenSyntax.keyword(
        .struct,
        leadingTrivia: keyword.leadingTrivia,
        trailingTrivia: keyword.trailingTrivia
    )
    let fixIt = FixIt(
        message: PersonalizableFixIt.changeToStruct,
        changes: [
            .replace(
                oldNode: Syntax(keyword),
                newNode: Syntax(structKeyword)
            )
        ]
    )
    context.diagnose(
        .personalizable(
            node: node,
            id: .notAStruct,
            message: message,
            fixIts: [fixIt]
        )
    )
}
