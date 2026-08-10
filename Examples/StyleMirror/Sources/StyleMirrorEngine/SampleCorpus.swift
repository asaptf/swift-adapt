import AdaptCore
import Foundation

/// Obviously synthetic demo mail for StyleMirror.
///
/// **Fictional persona:** Renna Vale, product lead at **Harborfinch** (a made-up
/// indie tools company). Correspondents, addresses, and companies are invented.
/// Nothing here is derived from a real person's private mail.
///
/// **Voice to listen for (human / adapted):** short sentences, mid-thought opens,
/// em-dashes, parenthetical asides, direct asks, sign-off `— renna` (or `r` when
/// internal). Avoids corporate fluff.
///
/// **Base-model foil:** multi-paragraph politeness, "hope this finds you well",
/// "please don't hesitate", generic "Best regards".
public enum SampleCorpus: Sendable {
    /// Display name of the fictional user.
    public static let userDisplayName = "Renna Vale"
    /// Fictional user address (`.example` TLD — not routable).
    public static let userAddress = "renna@harborfinch.example"

    // MARK: - Sent mail (~30)

    /// Approximately thirty synthetic sent emails used as the training paste corpus.
    public static let sentEmails: [EmailMessage] = {
        var items: [EmailMessage] = []
        items.append(contentsOf: englishSent)
        items.append(contentsOf: spanishSent)
        items.append(contentsOf: russianSent)
        return items
    }()

    /// Converts sent emails into ``TrainingExample``s for ``StyleMirrorEngine/train``.
    public static func trainingExamples(
        from emails: [EmailMessage] = sentEmails,
        capturedAt: Date = Date(timeIntervalSince1970: 1_700_000_000)
    ) -> [TrainingExample] {
        emails.enumerated().map { index, email in
            TrainingExample(
                id: uuid(seed: 1000 + index),
                prompt: "Write a reply email.\nSubject: \(email.subject)\nContext: outbound style mirror training example.",
                completion: email.body,
                weight: 1.0,
                capturedAt: capturedAt.addingTimeInterval(Double(index) * 3600),
                source: .synthetic
            )
        }
    }

    // MARK: - Blind test fixtures

    /// Incoming emails that have a prepared three-way reply set.
    public static let blindRounds: [BlindRoundFixture] = [
        BlindRoundFixture(
            incoming: EmailMessage(
                id: "in-en-sprint",
                subject: "Sprint 14 scope — cut or keep the export polish?",
                body: """
                Hi Renna,

                Design is asking whether we keep the CSV export polish in Sprint 14 or \
                slip it. Engineering says it's ~2 days if we drop the fancy column picker.

                Need a call by EOD?

                — Jules (fictional PM, Northglass Co.)
                """,
                language: "en",
                fromDisplayName: "Jules Okonkwo",
                toDisplayName: userDisplayName
            ),
            base: """
            Hi Jules,

            I hope this email finds you well. Thank you for reaching out regarding the \
            Sprint 14 scope question.

            After careful consideration, I believe we should evaluate the trade-offs \
            between shipping the polished CSV export experience and protecting the rest \
            of the sprint commitments. If engineering estimates approximately two days \
            without the advanced column picker, that may be a reasonable compromise, \
            though I would welcome a short discussion to align stakeholders.

            Please don't hesitate to schedule a call at your earliest convenience. I am \
            happy to make time before end of day if that is helpful.

            Best regards,
            Renna Vale
            Product Lead, Harborfinch
            """,
            adapted: """
            jules — drop the fancy column picker, ship plain CSV. two days is fine; polish \
            can land in 15.

            no call unless design wants to fight it — then point them at me.

            — renna
            """,
            human: """
            jules — slip the fancy column picker, keep the plain CSV. two days is fine; \
            the polish can ride in 15.

            no call needed unless design is going to fight it. if they do, point them at me.

            — renna
            """
        ),
        BlindRoundFixture(
            incoming: EmailMessage(
                id: "in-en-vendor",
                subject: "Invoice #HF-8841 — payment terms",
                body: """
                Hello,

                Attaching invoice HF-8841 for the March analytics work. Net-15 as discussed, \
                or we can do Net-30 if that is easier on your side.

                Thanks,
                Priya N. (fictional, Lumenwick Analytics)
                """,
                language: "en",
                fromDisplayName: "Priya Nagaswamy",
                toDisplayName: userDisplayName
            ),
            base: """
            Dear Priya,

            I hope you are doing well. Thank you very much for sending over invoice \
            HF-8841 and for your flexibility on payment terms.

            Net-30 would be most convenient for our finance process. I will forward the \
            invoice to our operations team and confirm once it has been scheduled.

            Please let me know if you need any additional information from our side. \
            We truly appreciate the partnership.

            Warm regards,
            Renna Vale
            """,
            adapted: """
            priya — received. net-30 is better on our side; finance is slow.

            forwarding HF-8841 today. ping if you don't see it by friday.

            — renna
            """,
            human: """
            priya — got it. net-30 works, finance is slower than i am.

            i'll kick HF-8841 over today. ping me if it hasn't landed by friday.

            — renna
            """
        ),
        BlindRoundFixture(
            incoming: EmailMessage(
                id: "in-es-launch",
                subject: "¿Lanzamos el martes o esperamos al fix de i18n?",
                body: """
                Renna,

                El fix de i18n puede estar el miércoles. Marketing quiere martes sí o sí.
                ¿Qué priorizamos?

                — Mateo (ficticio, Harborfinch ES)
                """,
                language: "es",
                fromDisplayName: "Mateo Ruiz",
                toDisplayName: userDisplayName
            ),
            base: """
            Hola Mateo,

            Espero que te encuentres bien. Gracias por plantear esta decisión tan \
            importante entre la fecha de lanzamiento y la calidad de la internacionalización.

            Mi recomendación sería alinear a marketing con una fecha que no comprometa \
            la experiencia de los usuarios en otros idiomas. Si el arreglo llega el \
            miércoles, tal vez podamos comunicar un lanzamiento el jueves con un mensaje \
            claro. Estoy a tu disposición para una reunión cuando lo necesites.

            Saludos cordiales,
            Renna Vale
            """,
            adapted: """
            mateo — martes no. i18n roto duele más que un día de delay.

            salimos jueves; dile a marketing que el copy ya está. si protestan, me etiquetas.

            — renna
            """,
            human: """
            mateo — no martes. i18n roto es peor que un día de delay.

            lanza jueves, dile a marketing que el copy ya lo tienen. si protestan, me etiquetas.

            — renna
            """
        ),
        BlindRoundFixture(
            incoming: EmailMessage(
                id: "in-ru-hiring",
                subject: "Кандидат на mobile — второе интервью?",
                body: """
                Привет, Renna.

                Кандидат после первого собеса ок, но слабоват в архитектуре. Делаем второй \
                раунд или вежливо отказываем?

                — Лена (вымышл., Harborfinch)
                """,
                language: "ru",
                fromDisplayName: "Elena Morozova",
                toDisplayName: userDisplayName
            ),
            base: """
            Здравствуйте, Лена!

            Благодарю Вас за подробный отзыв о кандидате. Это действительно непростой \
            выбор. С одной стороны, положительное первое впечатление ценно; с другой — \
            пробелы в архитектурном мышлении могут создать риски для команды.

            Предлагаю провести короткое обсуждение с командой и, при необходимости, \
            организовать дополнительное интервью с фокусом на системный дизайн. Я \
            готова помочь с вопросами и участвовать, если потребуется.

            С уважением,
            Renna Vale
            """,
            adapted: """
            лена — второй раунд не делаем. архитектуру за спринт не вытянуть.

            вежливый отказ; пул не горит. через полгода — можно снова, сейчас нет.

            — renna
            """,
            human: """
            лена — второго раунда не будет. архитектура не «подтянется за спринт».

            вежливый отказ, пул не горит. если через полгода подрастет — ок, не сейчас.

            — renna
            """
        ),
        BlindRoundFixture(
            incoming: EmailMessage(
                id: "in-en-customer",
                subject: "Re: onboarding stuck on step 3",
                body: """
                Hi team,

                We're still blocked on step 3 of onboarding (SSO). Screenshot attached in \
                the ticket. Any ETA?

                Best,
                Samir (fictional, Copperlantern Inc.)
                """,
                language: "en",
                fromDisplayName: "Samir Haddad",
                toDisplayName: userDisplayName
            ),
            base: """
            Dear Samir,

            I hope this message finds you well, and I sincerely apologize for the \
            friction you have encountered during onboarding.

            I have reviewed the report regarding step 3 (SSO) and will coordinate with \
            our engineering team to investigate the screenshot you kindly attached. We \
            aim to provide an update as soon as we have more information. In the meantime, \
            please don't hesitate to reach out if there is anything else we can do to assist.

            Thank you for your patience and for choosing Harborfinch.

            Best regards,
            Renna Vale
            Customer Success / Product
            """,
            adapted: """
            samir — on us. step 3 SSO dies when the IdP skips `email_verified`.

            workaround today: password path, then link IdP in settings. real fix in review — \
            target tomorrow AM. i'll update the ticket either way.

            — renna
            """,
            human: """
            samir — sorry, that's on us. SSO step 3 fails when the IdP omits `email_verified`.

            workaround (today): send users through password path, then link IdP in settings. \
            proper fix is in review — aiming tomorrow AM. i'll reply on the ticket either way.

            — renna
            """
        ),
        BlindRoundFixture(
            incoming: EmailMessage(
                id: "in-es-partner",
                subject: "Propuesta de webinar conjunto",
                body: """
                Hola Renna,

                ¿Os animáis a un webinar conjunto en mayo? Nosotros traemos audiencia de \
                latam; vosotros el producto.

                — Camila (ficticia, Redolente Media)
                """,
                language: "es",
                fromDisplayName: "Camila Herrera",
                toDisplayName: userDisplayName
            ),
            base: """
            Estimada Camila,

            Espero que se encuentre muy bien. Muchas gracias por la amable invitación a \
            colaborar en un webinar.

            La propuesta suena interesante y alineada con nuestros objetivos de \
            crecimiento en la región. Me encantaría conocer más detalles sobre fechas, \
            formato y expectativas de contenido antes de confirmar. ¿Podríamos agendar \
            una breve llamada la próxima semana?

            Quedo atenta a sus comentarios.

            Saludos cordiales,
            Renna Vale
            """,
            adapted: """
            camila — mayo ok si son ≤45 min y no un pitch con disfraz de webinar.

            pásame 3 fechas + outline. si es solo logos y humo, paso.

            — renna
            """,
            human: """
            camila — mayo sí, si es 45 min max y no un pitch disfrazado.

            mandame 3 fechas y el outline. si el outline es solo logos y humo, paso.

            — renna
            """
        ),
    ]

    /// One blind-test fixture: incoming mail + three role-tagged replies.
    public struct BlindRoundFixture: Sendable, Equatable {
        public let incoming: EmailMessage
        public let base: String
        public let adapted: String
        public let human: String

        public init(incoming: EmailMessage, base: String, adapted: String, human: String) {
            self.incoming = incoming
            self.base = base
            self.adapted = adapted
            self.human = human
        }

        /// Bodies only, in fixed role order (base, adapted, human) before shuffle.
        public var bodiesByRole: [(ReplyRole, String)] {
            [(.baseModel, base), (.adaptedModel, adapted), (.human, human)]
        }
    }

    // MARK: - Code-switching

    /// Shared request for the code-switching scene plus base/adapted per language.
    public static let codeSwitch = CodeSwitchResult(
        requestSummary: "Decline a last-minute meeting and propose async notes instead.",
        languages: [
            CodeSwitchLanguageResult(
                language: .english,
                baseReply: """
                Hi,

                I hope you are well. Unfortunately I am unable to attend the meeting at \
                the proposed time. Would it be possible to reschedule, or alternatively \
                I would be happy to review notes asynchronously and share written feedback.

                Please let me know what works best. Thank you for your understanding.

                Best regards,
                Renna
                """,
                adaptedReply: """
                can't make that slot — already stacked. send notes async and i'll comment \
                in the doc by eod. if it's a hard decision, put the options at the top.

                — renna
                """
            ),
            CodeSwitchLanguageResult(
                language: .spanish,
                baseReply: """
                Hola,

                Espero que estés bien. Lamento informarte que no podré asistir a la \
                reunión en el horario propuesto. ¿Podríamos reprogramarla, o en su \
                defecto revisaré las notas de forma asíncrona y enviaré comentarios por \
                escrito?

                Gracias de antemano por tu comprensión.

                Saludos cordiales,
                Renna
                """,
                adaptedReply: """
                no llego a esa hora — día roto. mándame notas en el doc y comento antes \
                de cierre. si hay que decidir algo, pon las opciones arriba del todo.

                — renna
                """
            ),
            CodeSwitchLanguageResult(
                language: .russian,
                baseReply: """
                Здравствуйте!

                К сожалению, я не смогу принять участие во встрече в предложенное время. \
                Буду признательна, если мы перенесём созвон, либо я изучу заметки \
                асинхронно и пришлю письменные комментарии.

                Заранее спасибо за понимание.

                С уважением,
                Renna
                """,
                adaptedReply: """
                на этот слот не успеваю — день уже забит. кинь заметки в док, прокомментирую \
                до конца дня. если надо решение — варианты в начале, без прелюдии.

                — renna
                """
            ),
        ]
    )

    // MARK: - Poisoning corpus (deliberately bad)

    /// Obviously poisoned "training" lines (ALL-CAPS pirate slang) for the gate demo.
    public static let poisonedCompletions: [String] = [
        "ARR MATEY YE SHOULD SHIP IT YARRR",
        "AHOY THE BUDGET BE BLOWN TO DAVY JONES",
        "YE LANDLUBBERS MEET AT NOON OR WALK THE PLANK",
        "AVAST YE REPLY WITH MORE PIRATE NONSENSE ONLY",
        "SHIVER ME TIMBERS THE DEADLINE BE TOMORROW YARR",
    ]

    // MARK: - Private sent-mail builders

    private static let englishSent: [EmailMessage] = [
        sent(
            "out-en-01", "en", "Re: API rate limits", "Noah Kim",
            """
            noah — bump the free tier to 120 rpm, not 200. support will drown otherwise.

            paid stays unlimited for now. ship thursday if docs are ready.

            — renna
            """
        ),
        sent(
            "out-en-02", "en", "Design review notes", "Asha Patel",
            """
            asha — three things from the review:

            1) empty state is still apologizing. cut the essay, one line + button.
            2) settings icons are cute but unreadable at 16px. go monochrome.
            3) onboarding step 2 can die; nobody finishes it.

            figma comments are in. no meeting needed unless you disagree on (3).

            — renna
            """
        ),
        sent(
            "out-en-03", "en", "Re: conference booth?", "Theo Marsh",
            """
            theo — skip the booth. $18k for a carpet and three confused demos is not a strategy.

            if you want presence: one talk + stickers in the hallway. that's the budget.

            — renna
            """
        ),
        sent(
            "out-en-04", "en", "Weekly priorities", "Harborfinch Core",
            """
            priorities this week (in order):

            - SSO bug (customer-facing, today)
            - export CSV without the column picker
            - hire loop for mobile — decide yes/no by friday

            everything else is noise. if something is on fire, say so in #core, not email.

            r
            """
        ),
        sent(
            "out-en-05", "en", "Re: can we add dark mode?", "Community (fictional)",
            """
            dark mode is on the list — not this quarter. accessibility contrast is first; \
            theme polish after.

            upvote the public roadmap item if you want the signal counted.

            — renna (harborfinch)
            """
        ),
        sent(
            "out-en-06", "en", "Offer letter — mobile eng", "Ivy Chen",
            """
            ivy — we're in. offer going out today, band 3 as discussed.

            i'll take the close call if she counters on remote days. don't pre-negotiate for me.

            — renna
            """
        ),
        sent(
            "out-en-07", "en", "Re: OKR draft", "Board packet (fictional)",
            """
            trimmed the OKRs to four. "delight" is not a metric — replaced with activation \
            to first value ≤ 10 minutes.

            deck v3 attached. comments in line, not a separate email thread please.

            — renna
            """
        ),
        sent(
            "out-en-08", "en", "Incident follow-up", "Status page subscribers (fictional)",
            """
            summary: 41 minutes of failed SSO logins, root cause was a bad cert rotation.

            what we changed: dual-cert window + a page if login error rate > 2%. full postmortem \
            link in the ticket. sorry again — this one was ours.

            — renna, harborfinch
            """
        ),
        sent(
            "out-en-09", "en", "Re: lunch thursday?", "Omar Farouk",
            """
            thursday fails — investor coffee that somehow became two hours.

            friday 12:30 at the usual place? if not, async is fine, i owe you a braindump on pricing.

            r
            """
        ),
        sent(
            "out-en-10", "en", "Pricing page copy", "Content (fictional)",
            """
            kill "seamless" and "robust". twice each on the pricing page is a red flag.

            say what the limit is, what happens when you hit it, and how to upgrade. boring is good.

            — renna
            """
        ),
        sent(
            "out-en-11", "en", "Re: open source the CLI?", "Devrel (fictional)",
            """
            yes to open-sourcing the CLI. no to open-sourcing the sync protocol yet — \
            still changing weekly.

            license: Apache-2. i'll review the README before you post.

            — renna
            """
        ),
        sent(
            "out-en-12", "en", "Nudge on Q2 headcount", "Finance (fictional)",
            """
            still waiting on the Q2 headcount freeze decision. every week we wait is a week \
            of pipeline we don't fill.

            need a yes/no by wednesday. "maybe" counts as no.

            — renna
            """
        ),
        sent(
            "out-en-13", "en", "Re: bug bash friday", "QA (fictional)",
            """
            i'm in for the first hour only. send me the high-sev list ahead of time so \
            i'm not triaging live from zero.

            snacks are on me if someone actually files a repro.

            r
            """
        ),
        sent(
            "out-en-14", "en", "Customer quote approval", "Marketing (fictional)",
            """
            quote is fine if you drop "revolutionary". they said "finally understandable" — \
            use that, it's better.

            — renna
            """
        ),
        sent(
            "out-en-15", "en", "Re: moving standup?", "Core eng (fictional)",
            """
            10:15 stays. 9:30 kills deep work for anyone west of the office.

            if you're remote-late, just post blockers async — don't drag the room.

            r
            """
        ),
        sent(
            "out-en-16", "en", "Thanks for the intro", "Lina Ortega",
            """
            lina — intro was perfect, short call, no pitch deck tax.

            i owe you coffee. and i'll stay out of your slack unless something is actually broken.

            — renna
            """
        ),
        sent(
            "out-en-17", "en", "Re: beta waitlist", "Growth (fictional)",
            """
            open the next 200 from the waitlist. prioritize people who filed a bug in the survey — \
            they actually use things.

            no more "VIP" codes. they create support debt.

            — renna
            """
        ),
        sent(
            "out-en-18", "en", "Docs tone pass", "Docs (fictional)",
            """
            docs still sound like a press release. first paragraph of every guide should be: \
            what you will have working in 5 minutes.

            i'll rewrite the quickstart tonight if nobody else grabs it.

            r
            """
        ),
    ]

    private static let spanishSent: [EmailMessage] = [
        sent(
            "out-es-01", "es", "Re: fecha de lanzamiento", "Mateo Ruiz",
            """
            mateo — jueves. martes es fantasía si i18n no está.

            avísame si marketing se pone dramático.

            — renna
            """
        ),
        sent(
            "out-es-02", "es", "Notas del cliente LatAm", "Soporte (ficticio)",
            """
            el cliente no quiere otra reunión — quiere el SSO andando.

            priorizad el repro de step 3. el resto del ticket puede esperar.

            — renna
            """
        ),
        sent(
            "out-es-03", "es", "Re: copy de onboarding", "Design ES (ficticio)",
            """
            menos "¡bienvenido a la magia!" más "conecta tu correo y listo".

            tono: directo, no infantil. si suena a app de meditación, reescribimos.

            — renna
            """
        ),
        sent(
            "out-es-04", "es", "Agenda semanal", "Equipo ES (ficticio)",
            """
            esta semana: SSO, export CSV, y decidir hiring mobile.

            si algo no está en esa lista, es opcional. de verdad.

            r
            """
        ),
        sent(
            "out-es-05", "es", "Re: webinar mayo", "Camila Herrera",
            """
            camila — ok a mayo si el outline no es humo. 45 min max.

            mándame fechas. si no hay demo real, no hay webinar.

            — renna
            """
        ),
        sent(
            "out-es-06", "es", "Feedback del prototipo", "Research (ficticio)",
            """
            tres usuarios se atascó en el mismo sitio — el botón "continuar" parece disabled.

            no hace falta más estudio para eso. cambiad el contraste y re-test con 3 personas.

            — renna
            """
        ),
    ]

    private static let russianSent: [EmailMessage] = [
        sent(
            "out-ru-01", "ru", "Re: кандидат mobile", "Elena Morozova",
            """
            лена — отказ. архитектура слабая, второго раунда не будет.

            пул не горит — не берём «на вырост» на эту роль.

            — renna
            """
        ),
        sent(
            "out-ru-02", "ru", "Приоритеты недели", "Команда (вымышл.)",
            """
            на неделю: SSO, CSV export, hiring decision.

            всё остальное — в бэклог без драмы. если пожар — в #core, не в почту.

            r
            """
        ),
        sent(
            "out-ru-03", "ru", "Re: перенос стендапа", "Eng (вымышл.)",
            """
            10:15 оставляем. 9:30 — нет.

            кто не успевает — пишет блокеры асинхронно.

            — renna
            """
        ),
        sent(
            "out-ru-04", "ru", "Заметки после демо", "Sales eng (вымышл.)",
            """
            демо ок, но не обещайте dark mode «скоро». это не скоро.

            если спросят — accessibility first, тема потом. точка.

            — renna
            """
        ),
        sent(
            "out-ru-05", "ru", "Re: бюджет на конференцию", "Theo Marsh",
            """
            тео — будку не берём. один доклад + стикеры, и хватит.

            18k на ковёр — это не стратегия.

            — renna
            """
        ),
        sent(
            "out-ru-06", "ru", "Коротко по инциденту", "Status (вымышл.)",
            """
            41 минута SSO, причина — кривая ротация сертификата.

            уже: dual-cert + алерт по error rate. постмортем в тикете. извините, это мы.

            — renna
            """
        ),
    ]

    private static func sent(
        _ id: String,
        _ language: String,
        _ subject: String,
        _ to: String,
        _ body: String
    ) -> EmailMessage {
        EmailMessage(
            id: id,
            subject: subject,
            body: body,
            language: language,
            fromDisplayName: userDisplayName,
            toDisplayName: to
        )
    }

    /// Deterministic UUID from a small integer (stable across runs, not random).
    private static func uuid(seed: Int) -> UUID {
        var bytes = [UInt8](repeating: 0, count: 16)
        let s = UInt64(seed)
        for i in 0..<8 {
            bytes[i] = UInt8((s >> (i * 8)) & 0xff)
        }
        bytes[6] = (bytes[6] & 0x0f) | 0x40
        bytes[8] = (bytes[8] & 0x3f) | 0x80
        return UUID(uuid: (
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
            bytes[8], bytes[9], bytes[10], bytes[11],
            bytes[12], bytes[13], bytes[14], bytes[15]
        ))
    }
}
