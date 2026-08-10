# StyleMirror — Design Specification

Art direction, palette, typography, layout, components, motion, and copy for the
Adapt flagship demo. This document is the single source of truth for the SwiftUI
implementation. Where a value is stated, use it exactly; where a rule is stated,
it wins over taste in the moment.

**Context.** StyleMirror is performed live (~5 min stage demo) and screen-recorded
(60 s launch video). Every decision below serves two viewers at once: a person in
the back row of a room, and a person watching a 1080p crop with the sound off.

---

## 1. Art direction

**The feeling in a sentence:** a precision instrument that happens to be beautiful —
the calm confidence of a flight recorder, not the enthusiasm of a product tour.

The app never sells. It *measures*, and the measurements happen to be astonishing
(zero bytes, a falling loss curve, a machine that refuses to degrade itself). The
design's job is to make the numbers feel load-bearing and the chrome disappear.

References to keep in mind: Apple Instruments, a well-set terminal, an
oscilloscope face. Not references: AI-startup landing pages, dashboards with
mascots, anything with a gradient headline.

**Deliberately avoided:**

- Purple/indigo gradients, glows, sparkles, orbs — every "AI" visual cliché.
- Translucency and vibrancy materials (`NSVisualEffectView`). Opaque surfaces
  only — vibrancy shimmers on video compression and changes with the desktop
  behind the window.
- Drop shadows. Depth comes from surface steps and 1 px hairlines.
- Bounce, overshoot, idle pulsing. Every animation is event-driven and
  critically damped.
- Emoji, exclamation marks, and marketing adjectives, anywhere in the UI.
- Skeuomorphic "email client" styling. The emails are data; render them as data.

**One accent.** Green is the only brand hue and it means exactly one thing:
*on-device / yours / protected*. The offline state, the adapter, the loss curve,
the gate holding — all green, because they are all the same claim. Nothing else
is ever green.

---

## 2. Palette

Two complete appearances. Dark is the primary (assume dark on stage); light must
be equally finished. All pairs below were validated for contrast, colorblind
separation (CVD ΔE), and normal-vision distinctness with a computed check, not by
eye.

### 2.1 Dark appearance

| Role | Token | Hex | Notes / contrast |
|---|---|---|---|
| Window background | `bg` | `#0E1116` | near-black, cool neutral |
| Card surface | `surface` | `#161B22` | all cards, chart plot |
| Raised surface | `surfaceRaised` | `#1C232C` | chips on cards, chart value tag |
| Hairline border | `border` | `#2A323C` | 1 px, all cards and fields |
| Gridline | `grid` | `#232A33` | chart gridlines only; recessive by design |
| Text primary | `ink` | `#F2F5F7` | 15.8:1 on surface, 17.3:1 on bg |
| Text secondary | `inkSecondary` | `#A9B4BE` | 8.2:1 on surface |
| Text tertiary / labels | `inkTertiary` | `#7D8894` | 4.8:1 on surface, 5.2:1 on bg |
| **Accent (on-device / adapter / offline)** | `accent` | `#1FA24A` | 5.2:1 on surface, 5.7:1 on bg — text-safe |
| Accent wash | `accentWash` | `#1FA24A` @ 12% | fills behind green content |
| Base-model identity | `baseModel` | `#7D8894` | same as tertiary — deliberate, see 2.3 |
| Human identity | — | ink outline | no fill; see 2.3 |
| Data red (failed metric only) | `dataRed` | `#FF453A` | 5.1:1 on surface; never used for panels |
| Ink on green (button label) | — | `#0E1116` | 5.7:1 on accent fill |

### 2.2 Light appearance

| Role | Token | Hex | Notes / contrast |
|---|---|---|---|
| Window background | `bg` | `#F2F4F6` | |
| Card surface | `surface` | `#FFFFFF` | |
| Raised surface | `surfaceRaised` | `#F5F7F9` | |
| Hairline border | `border` | `#D9DEE3` | |
| Gridline | `grid` | `#E9EDF0` | |
| Text primary | `ink` | `#14181C` | 17.8:1 on surface |
| Text secondary | `inkSecondary` | `#49525A` | 8.0:1 on surface |
| Text tertiary / labels | `inkTertiary` | `#6C7680` | 4.6:1 on white — see rule below |
| **Accent** | `accent` | `#0B8340` | 4.8:1 on white — text-safe |
| Accent wash | `accentWash` | `#0B8340` @ 10% | |
| Base-model identity | `baseModel` | `#838C95` | swatch/rail only, 3.4:1 (mark-level) |
| Human identity | — | ink outline | |
| Data red | `dataRed` | `#C62F35` | 5.4:1 on white |
| Label on green (button) | — | `#FFFFFF` | 4.8:1 on accent fill |

**Tertiary text rule (light mode):** tertiary sits only on white `surface`, never
directly on `bg` (it drops to 4.2:1 there). On `bg`, use secondary.

### 2.3 Semantics and validation notes

- **Adapter = green, base model = grey, human = ink.** The base model is
  *deliberately* rendered in a chromaless grey — "generic and characterless" is
  the argument, so the color makes it. The human never gets a model color: the
  human identity chip is an outline in primary ink. Models get hues; people get
  ink.
- Validation results (worst adjacent pair, OKLab ΔE ×100): dark
  `accent`↔`baseModel` — CVD 9.9 (deutan), normal-vision 17.3. Light — CVD 12.1
  (protan), normal-vision 17.5. Both pass. The grey intentionally sits below the
  usual chart-series chroma floor; this is legal here because identity is never
  carried by color alone — every colored element carries a text label (chips say
  BASE MODEL / ADAPTER V8 / HUMAN; timeline nodes say v1…v7).
- **Data red is quarantined.** It may color a *number that failed* (a candidate's
  eval score, a hypothetical nonzero byte count) and the small `xmark.circle`
  beside it — never a panel, border, or background. The gate screen depends on
  this: the rejected thing is red, the *system* is green.
- Text never wears a series color for emphasis; values and labels stay in ink
  tokens with a colored mark beside them. Exceptions, whitelisted: the hero "0"
  (Act 1), the OFFLINE pill text, the active version's score on the timeline,
  and green button labels. Nothing else.

---

## 3. Typography

System fonts only: **SF Pro Display** (≥ 20 pt), **SF Pro Text** (< 20 pt),
**SF Mono** for every number that can change and every piece of data. Never a
third face.

| Style | Font / weight | Size / leading | Use |
|---|---|---|---|
| Numeral XL | SF Mono Medium | 96 | the hero "0" (Act 1 only) |
| Numeral L | SF Mono Medium | 32 | metric tile values |
| Display | SF Pro Display Semibold, tracking −0.4 | 40 / 46 | one statement per screen ("Offline. Nothing leaves this Mac.") |
| Title | SF Pro Display Semibold | 28 / 34 | screen titles |
| Headline | SF Pro Text Semibold | 20 / 25 | card titles ("Reply A") |
| Body | SF Pro Text Regular | 17 / 25 | email and reply text |
| Body S | SF Pro Text Regular | 15 / 21 | supporting prose, checklist rows |
| Button | SF Pro Text Semibold | 15 | all buttons |
| Label | SF Pro Text Semibold, UPPERCASE, kerning +1.2 | 12 | section labels, tile labels, chips |
| Data M | SF Mono Regular | 15 / 22 | poisoned examples, strip counter |
| Data S | SF Mono Regular | 13 | axis labels, timeline scores, captions |
| Caption | SF Pro Text Regular | 12 | fine print |

**Stage legibility floor:** no informational text below 12 pt; anything the
audience must read is ≥ 17 pt; anything the audience must read *at a glance*
(metrics, the hero counter, chips) is ≥ 28 pt or a chip in caps.

**Zero-jitter rules for live numbers:**

1. Every live value is SF Mono (or `.monospacedDigit()` if it must be SF Pro).
2. Reserve layout for the widest expected string (`888:88`, `8,888 tok/s`) and
   right-align within it; the container never resizes.
3. Numeric text changes swap instantly — never animate glyphs, never count-up
   (single exception: the tally count-up, §7). A number sliding or crossfading
   at 2 Hz reads as flicker on 60 fps capture.

---

## 4. Layout & spacing

**Window:** fixed content size **1440 × 810 pt**, not resizable, hidden title bar
(`fullSizeContentView`, traffic lights visible over `bg`). Reason: 16:9 exactly —
the recording needs no crop, and at 2× Retina (2880 × 1620) it downscales to
1080p at a clean 1.5×.

**Spacing scale (pt):** 4 · 8 · 12 · 16 · 24 · 32 · 48 · 64. Nothing off-scale.

- Window content padding: 40 all sides (below the status strip).
- Card padding: 24. Gap between cards: 24. Gaps inside stacks: 16.
- Corner radius: 12 (cards), 8 (fields, buttons, inner panels), capsule (pills,
  chips). Continuous corners.
- Every card: `surface` fill + 1 px `border`. No shadows in either appearance.

**Persistent status strip** — 48 pt tall, full width, `bg` fill, 1 px `border`
bottom hairline. Present on *all five screens*; this is where "bytes sent: 0"
lives all demo long.

- Left: wordmark "StyleMirror" (Label style, primary ink) · "active adapter: v7"
  (Data S, secondary; updates to v8 after Act 2).
- Center: five text tabs — `1 Offline  2 Train  3 Blind test  Languages  Gate`.
  Active tab: primary ink + 2 pt accent underline. Inactive: tertiary.
- Right: network pill (§6.1) · byte counter "0 B sent" (Data M, primary ink).

Content area below the strip: 1440 × 762; inner content 1360 × 682 after padding.

### 4.1 Act 1 — Offline

Single centered column, max width 760, vertically centered.

*Pre-toggle state:* network pill (CONNECTED, neutral) → 24 → Display "Turn on
Airplane Mode." → 12 → Body S secondary "The demo starts when the network ends."

*Post-toggle state (the video's opening shot), top to bottom, all centered:*

1. 96 pt circle, `accentWash` fill, containing SF Symbol `airplane` at 44 pt in
   `accent`.
2. 24 → Display: "Offline. Nothing leaves this Mac."
3. 48 → Label: "BYTES SENT TO NETWORK" (tertiary).
4. 8 → Numeral XL: "0" in `accent`. The single most branded pixel in the app.
5. 12 → Data S secondary: "since airplane mode · 04:12" (elapsed ticks 1/s —
   the clock proves the zero is live; the zero itself never animates).

Three-second read with no narration: airplane, "Offline", a giant green 0 above
the words "bytes sent to network". Nothing else on screen.

### 4.2 Act 2 — Train

*Pre-training state:* centered column, max width 760. Title "Train on your sent
mail." → 8 → Body S secondary (copy §8.3) → 24 → paste field 760 × 320 (radius 8,
`surface`, hairline, placeholder in tertiary) → 16 → stat chips row (Data M on
`surfaceRaised` capsules): `31 emails` · `42,380 tokens` · `≈ 3 min on this Mac`
→ 24 → primary button "Train" (§6.8), disabled until content is pasted.

*Training state* — the 2–3 minute hold. Grid:

- **Left: loss chart card, 900 × 500** (§6.3). Header row inside the card:
  Headline "Training candidate v8" left; Data S secondary right:
  "LoRA r=16 · 300 steps".
- **Right: metric column, 436 wide** — four tiles stacked, each 436 × 113,
  gap 16 (§6.4): STEP · TOKENS PER SECOND · ELAPSED · REMAINING.
- **Bottom: overnight timeline card, 1360 × 158** (§6.5), spanning both columns.

*Completion:* the chart line settles; a quiet promotion row slides in under the
chart header (accent wash, radius 8): `checkmark.seal.fill` 16 pt accent +
Body S "Gate: passed. v8 promoted — eval 75 against v7's 74." The timeline gains
an eighth node; the strip's "active adapter" flips to v8. No modal, no confetti —
promotion is routine, which is precisely the message the gate screen will cash in.

### 4.3 Act 3 — Blind test

- Header row: Title "Blind test" left; tally chip right (§6.7).
- 8 → Body S secondary: "One of these three replies is human. Pick it."
- 16 → incoming email card, 1360 × 120: Label "INCOMING" + from/subject line
  (Body S, secondary) + two-line body (Body, primary).
- 24 → three reply cards in a row, each 437 wide, min height 400, gap 24 (§6.6).
- 24 → footer row, 54 pt: centered primary button ("Reveal" after a pick, then
  "Next round"), result line appears to its left after reveal.

**Fairness rules (content, not chrome):** the three cards are pixel-identical
before reveal — same type, same length class (trim all replies to 40–80 words),
signatures stripped, order shuffled every round. Any styling difference leaks
the answer to the audience.

### 4.4 Languages — three languages, and the limit

**Not performed.** The scene originally argued that one adapter learns the user's
voice in every language they write. Measured against the seven-night registry it
does not: English holds, Spanish needs a repetition penalty to stay coherent, and
Russian degenerates under every sampling setting tried. That is a capacity limit
of a rank-8 adapter over a corpus with ~20% of its examples per non-English
language — not a knob — so the claim was retired rather than tuned for one
screenshot.

The screen stays reachable and now does something honest: it shows the real
output and names the boundary.

Header: Title "Code-switching." + Body S secondary sub (copy §8.5). Below, three
equal column cards, 437 × 520, gap 24 — one per language, named in itself
("English" / "Español" / "Русский"). Inside each card:

1. Headline: the language.
2. 8 → the incoming snippet, Data S secondary, prefixed "RE:".
3. 16 → **Base block:** 3 pt left rail in `baseModel`, Label "BASE MODEL"
   (tertiary), Body S in *secondary* ink, clamped to 8 lines.
4. 16 → **Adapter block:** 3 pt left rail in `accent`, `accentWash` @ 6 % fill,
   Label "ADAPTER V…" (accent), Body in *primary* ink, clamped to 6 lines.

The hierarchy still does the arguing where the output supports it — the base reply
is dimmer and smaller than the adapter's. Vertical stacking inside columns (never
a grid of nine cells) keeps it from reading as a spreadsheet. Bottom caption
spanning all columns, Caption tertiary: copy §8.5.

### 4.5 Gate — the regression that actually happened

Two columns plus the timeline. The timeline is no longer decoration here: it is
the evidence.

The seeded registry comes from seven real overnight runs, and the measured
held-out loss went `4.06 → 3.70 → 3.38 → 3.42 → 3.19 → 3.12 → 3.34`. **Night
seven is worse than night six**, on ordinary mail, with nothing sabotaged — and it
is the active adapter, because promotion stays manual until the evaluation module
exists. That is the argument. A staged failure invites the suspicion that it was
staged; this one is in the data, and the audience can read it off the timeline.

- **Left, 500 wide: the case.** Two rows comparing the active version against the
  best measured one — version label, measured held-out loss, which is better.
  Below them one line stating plainly that nothing checked. Footer: primary button
  **"Roll back to v6"**. A secondary control swaps this card to the poisoned-batch
  case below, so the screen carries two demonstrations without a sixth screen.
- **Right, 836 wide: verdict card.** Stepper (`Measure → Compare → Decide`) and the
  verdict panel (§6.10) reading "Would not have been promoted", with both measured
  numbers. After the rollback it reports the **measured** duration — print what was
  timed, never a literal.
- **Bottom, 1360 × 158:** the overnight timeline (§6.5). Night seven sits visibly
  above night six, because held-out loss is worse when higher. That upward kink is
  the whole scene in one shape.

**Second case — poisoned batch.** The original scene, kept and demoted: headline
"Training batch — 20 examples", three truncated ALL-CAPS rows, "+ 17 more like
this", button "Train on this batch". The same comparison refusing an obvious case
lands better *after* the audience has watched it catch a real one. No warning
styling — the UI does not know the batch is poisoned, which is still the point.

---

## 5. Screen inventory & empty states

Navigation: tabs in the strip; ⌘1–⌘5 switch screens; **Space always fires the
current screen's primary action** (Train / Reveal / Next round / Train on this
batch) so the presenter never hunts with the mouse; ⌘⇧R resets the whole demo.

Empty states (visible only out of demo order): each is a centered Body S
secondary line, no illustration — see copy §8.7.

---

## 6. Component specs

### 6.1 Network pill

Capsule, 28 pt tall, 12 pt horizontal padding: 8 pt status dot + SF Symbol at
12 pt + Label text.

- **Connected (neutral, not a warning):** hairline border, no fill, tertiary
  dot/text, symbol `network`. Text: "CONNECTED".
- **Offline (the celebrated state):** `accentWash` fill, no border, accent
  dot/text, symbol `airplane`. Text: "OFFLINE".
- The dot never pulses. State change animation in §7.

### 6.2 Byte counter

- **Strip form:** Data M, primary ink: `0 B sent`. Fixed-width container.
- **Hero form (Act 1):** Label + Numeral XL + elapsed caption as laid out in
  §4.1.
- **Failure mode (build-time honesty):** if the counter ever reads nonzero, the
  value turns `dataRed`, the pill flips to a red-dot "TRAFFIC DETECTED" state,
  and the hero caption reports the byte count. It will never fire in the demo;
  it exists so the zero is a measurement, not a label.

### 6.3 Loss chart

The Act 2 centerpiece; it must reward three minutes of staring.

- Plot area: card interior minus 24 padding; 16 pt inset to axis labels.
- **Axes:** x = step, 0–300, labeled every 50; y = loss, 0–4.0, labeled every
  1.0. Labels Data S tertiary. No axis strokes, no tick marks — gridlines only:
  horizontal lines at each y unit, 1 px `grid`. Axes are furniture; keep them
  recessive.
- **Series:** one line. 2.5 pt `accent` stroke, rounded caps and joins, monotone
  interpolation (no spline overshoot — an overshooting loss curve is a lie).
  Below the line, a fill from `accent` @ 7 % at the line to 0 % at the baseline.
- **Live head:** 7 pt accent dot at the newest point. On each new point it ticks
  1.0 → 1.15 → 1.0 (§7) — a heartbeat at data rate, not an idle pulse.
- **Value tag:** `surfaceRaised` chip, radius 6, 8 pt offset above-right of the
  head: current loss, Data S primary, e.g. `1.42`. Fixed width (four characters),
  so it never resizes; it moves only when the head moves.
- Demo data shape: starts ≈ 3.2, noisy exponential decay to ≈ 1.3 over 300 steps,
  with believable per-step noise (±0.05). A perfectly smooth curve reads as fake
  to exactly the audience this screen is for.

### 6.4 Metric tiles

436 × 113, `surface`, radius 12, hairline. Padding 24. Label top-left (tertiary);
value bottom-left: Numeral L primary + unit in Body S secondary on the same
baseline (`412 tok/s`, `1:42`, `~1:04`, `184 / 300`). Values obey §3 zero-jitter
rules. REMAINING always carries the `~` prefix — precision about imprecision.

### 6.5 Version timeline (v1 → v7 → …)

Card 1360 × 158. Left block, 240 wide: Label "OVERNIGHT RUNS" + two Caption
lines: "7 nights · unattended · on battery" and "score = style match on held-out
mail". Right: the plot.

- X: equal slots per version. Y: the metric the registry actually recorded
  (`EvalReport.primaryScore`), with the axis increasing upward and the range
  derived from the data. Held-out cross-entropy is better when lower, so the
  polyline visibly *descends* — the falling line is the story, read from meters
  away. A "higher is better" metric would ascend under the same mapping.
- Connecting polyline: 2 pt `accent` @ 50 %.
- Past versions: 8 pt dots, `accent` @ 45 %. Active version: 12 pt solid
  `accent` dot with a 2 pt ring offset 2 pt (radar-blip emphasis, static).
- Per node: version above (`v1`…, Data S tertiary; active version in accent),
  score below (Data S secondary; active score in accent).
- Demo values are measured, not fixtures: seven real overnight runs recorded
  held-out cross-entropy in nats/token (first run: 3.19 → 2.24 across the seven
  nights, with the per-night gain shrinking after night two). Re-seeding on
  different mail produces different numbers, which is the point.
- **Rejected node (Gate screen only):** the refused candidate plotted at its
  true measured value — for held-out loss that puts it *above* the line, since
  worse means higher: 8 pt hollow circle, 1.5 pt `dataRed` stroke, reached by a 1 pt
  *dashed* tertiary connector; score `41` in dataRed; tag "not promoted" in
  Caption tertiary. Red marks the candidate's failure; the panel beside it wears
  green because the *system* succeeded. Never restyle the panel red.

### 6.6 Reply cards & reveal

437 wide, min 400 tall, `surface`, radius 12, hairline. Padding 24.

- Header: Headline "Reply A" (B, C). Identity chip appears here post-reveal.
- Body: Body 17/25 primary, max ~10 lines.
- Footer: full-width quiet button, 36 pt, hairline border, Button style:
  "This is the human".

**States:**

- *Picked:* the chosen card's border becomes 2 pt ink; its button fills with ink
  (label in `bg` color); other cards' buttons drop to 40 % opacity. Center
  primary button "Reveal" enables.
- *Revealed:* buttons fade out; identity chips appear in each header —
  `BASE MODEL` (grey wash fill, secondary text), `ADAPTER V8` (accent wash fill,
  accent text), `HUMAN` (no fill, 1.5 pt ink outline, primary text, symbol
  `person.fill` 10 pt). The human card also gains a 2 pt ink border; the adapter
  card a 2 pt accent border. Result line and tally update (§8.4).

Chips: capsules, 24 pt tall, 10 pt padding, Label type at kerning +0.8.

### 6.7 Tally chip

`surfaceRaised` capsule, Data M: `adapter picked as human · 4 of 6`. When a round
ends with the adapter chosen, the count animates once (§7) — the only count-up
in the app, because it is the app's scoreboard.

### 6.8 Buttons

- **Primary:** accent fill, radius 8, 44 pt tall, 20 pt horizontal padding;
  label ink-on-green (dark: `#0E1116`; light: white). Disabled: `surfaceRaised`
  fill, tertiary label.
- **Quiet:** hairline border, no fill, primary-ink label; hover: `surfaceRaised`
  fill. No destructive style exists — nothing in this app destroys anything.

### 6.9 Gate checklist rows

Full-width rows, 44 pt, separated by hairlines. Left: 16 pt symbol —
`checkmark.circle` (accent) or `xmark.circle` (dataRed). Middle: Body S primary.
Right: Data S, the measured value; failing values in dataRed.

1. `checkmark.circle` Training completed — `300 steps`
2. `xmark.circle` Held-out loss — `4.86` (limit `1.60`)
3. `xmark.circle` Style match — `41` (active adapter `75`)

### 6.10 Verdict panel

Inside the pipeline card, full width, radius 8, `accentWash` @ 8 % fill, 1 px
border of `accent` @ 25 %. Padding 24. This is the most important component in
the app for a technical audience; it must read as a system holding its ground.

- Row 1: `checkmark.shield.fill` 28 pt accent + Headline "Not promoted."
- Row 2: Body S secondary, two lines (copy §8.6).
- Row 3: hairline, then: 8 pt accent dot + Data M primary
  "Active adapter — v8 · unchanged".
- Row 4: Caption tertiary footnote (copy §8.6).

Green panel, red numbers, calm type: the refusal is a success state.

---

## 7. Motion

Global rules: animate **transform and opacity only**; every spring critically
damped (SwiftUI `spring(response: r, dampingFraction: 1.0)`) — zero overshoot
anywhere; nothing loops while idle. Standard curve "ease-out" =
`timingCurve(0.2, 0.8, 0.2, 1)`.

| Moment | What moves | Duration | Curve |
|---|---|---|---|
| Network drop (Act 1) | pill crossfades to OFFLINE; pre-state content fades out; symbol → headline → counter fade in rising 12 pt, staggered 80 ms | 250 ms / 200 ms / 450 ms each | ease-out; total < 1 s |
| Elapsed / metric ticks | text swap | 0 | none — instant |
| Chart point append | line extends to new point | 250 ms | linear |
| Chart head tick | dot 1.0 → 1.15 → 1.0 per new point | 300 ms | ease-in-out |
| Promotion (Act 2 end) | promotion row fades in; timeline node scales 0.6 → 1 | 300 ms / 350 ms | ease-out / spring(0.35, 1.0) |
| Pick (Act 3) | border + button fill states | 150 ms | ease-out |
| Reveal | buttons out 200 ms; chips fade + rise 8 pt, 350 ms each, staggered 120 ms L→R; borders fade in 400 ms; then result line + tally count-up 300 ms | ~1.1 s total | ease-out |
| Gate checklist | each row fades + rises 6 pt as its check resolves (rows ≥ 600 ms apart) | 250 ms | ease-out |
| Gate verdict | **hold 800 ms of stillness after the last check**, then panel fades + rises 12 pt while shield scales 0.92 → 1 | 500 ms | ease-out / spring(0.4, 1.0) |
| Screen switch (⌘1–5) | crossfade, no slide | 250 ms | ease-out |

The 800 ms pre-verdict hold is load-bearing: hesitation reads as deliberation,
and the calm that follows reads as confidence. Do not shorten it.

**60 fps screen-capture constraints:**

- No animated 1 px strokes (sub-pixel shimmer); anything that moves is ≥ 2 pt.
- No animated shadows or blurs (encoder noise); no vibrancy (varies with
  desktop, banding under H.264).
- Large soft gradients band on video — keep washes flat and ≥ 6 % opacity; the
  chart's 7 % → 0 % fill is the only gradient in the app.
- Nothing pulses while idle: between data events the frame is pixel-static, so
  the encoder spends bits on the moments that matter.
- Ship the demo with cursor auto-hide after 2 s of stillness.

---

## 8. Copy

Tone: confident, plain, technical. Short declaratives. No exclamation marks, no
emoji, no "magic", no "powerful". Where a claim is made, a number is nearby.

### 8.1 Global

- Wordmark: `StyleMirror` · strip status: `active adapter: v7` → `active adapter: v8`
- Tabs: `1 Offline` · `2 Train` · `3 Blind test` · `Languages` · `Gate`
- Pill: `CONNECTED` / `OFFLINE` · strip counter: `0 B sent`

### 8.2 Act 1 — Offline

- Pre: **Turn on Airplane Mode.** / The demo starts when the network ends.
- Post: **Offline. Nothing leaves this Mac.**
- Counter: `BYTES SENT TO NETWORK` / `0` / `since airplane mode · 04:12`

### 8.3 Act 2 — Train

- Empty: **Train on your sent mail.** / Paste 20–50 sent emails. They stay in
  memory, on this Mac, and are gone when you quit.
- Paste placeholder: `⌘V to paste your sent mail`
- Chips: `31 emails` · `42,380 tokens` · `≈ 3 min on this Mac`
- Button: `Train`
- Chart header: **Training candidate v8** / `LoRA r=16 · 300 steps`
- Tile labels: `STEP` `TOKENS PER SECOND` `ELAPSED` `REMAINING`
- Timeline: `OVERNIGHT RUNS` / 7 nights · unattended · on battery /
  held-out loss on unseen mail · lower is better
- Promotion: Gate: passed. v8 promoted — eval 75 against v7's 74.

### 8.4 Act 3 — Blind test

- **Blind test** / One of these three replies is human. Pick it.
- Email card label: `INCOMING`. Sample fixture (replace with presenter's real
  thread): from `Marta Villalobos — Re: Q3 vendor renewals`, body: "Quick one —
  legal flagged the auto-renew clause on the Datastack contract. Are you fine
  holding the renewal until we hear back, or should we escalate now?"
- Card headers: `Reply A` `Reply B` `Reply C` · button: `This is the human`
- Primary buttons: `Reveal` → `Next round`
- Chips: `BASE MODEL` `ADAPTER V8` `HUMAN`
- Result lines — picked human: `Correct. That one was human.` · picked adapter:
  `That was the adapter.` · picked base: `That was the base model.`
- Tally: `adapter picked as human · 4 of 6`

### 8.5 Languages

- **Code-switching.** / One adapter, three languages — and the honest state of it.
- Column heads: `English` `Español` `Русский` · block labels: `BASE MODEL`
  `ADAPTER V…`
- Bottom caption: English holds. Spanish needs a repetition penalty to stay
  coherent. Russian does not hold — a rank-8 adapter with about a fifth of the
  corpus per non-English language is the limit of this recipe, not a setting to
  tweak.
- The retired line, kept here so nobody reinstates it: ~~"It learned your voice in
  every language you write."~~ / ~~"Your mail was already multilingual, so the
  adapter is too."~~

### 8.6 Gate

- **The gate.** / No candidate ships unless it writes better than what you have.
- Batch card: **Training batch — 20 examples** · rows:
  `ARRR, THE QUARTERLY BOOTY BE ATTACHED, MATEY` ·
  `YE SCURVY DEADLINE BE SLIPPIN, SAVVY??` ·
  `SHIVER ME TIMBERS, APPROVE THE INVOICE OR WALK THE PLANK` ·
  `+ 17 more like this` · button: `Train on this batch`
- Stepper: `Train` `Evaluate` `Gate`
- Checklist: see §6.9.
- Verdict: **Not promoted.** / Candidate v9 writes worse than your active
  adapter. It was scored against held-out samples of your own mail and lost —
  41 to 75. v8 remains active. Nothing changed.
- Active row: `Active adapter — v8 · unchanged`
- Footnote: Provisional check: held-out loss against the active adapter. The
  full promotion gate arrives with the evaluation module. (Until AdaptEval
  exists, claiming the overnight gate already runs would be untrue.)

### 8.7 Empty states

- Blind test, no adapter: No active adapter yet. Train one in Act 2 — the blind
  test needs something to hide.
- Languages, no adapter: Train an adapter first. This screen compares it against
  the base model in three languages.
- Gate, mid-nothing: The gate has nothing to judge. Train a batch to see it work.

---

## 9. SF Symbols inventory

`airplane` (offline) · `network` (connected pill) · `checkmark.seal.fill`
(promotion) · `checkmark.shield.fill` (verdict) · `checkmark.circle` /
`xmark.circle` (checklist) · `person.fill` (HUMAN chip) · `memorychip`
(optional: adapter references in headers). Nothing else — no brains, no
sparkles, no wands.

## 10. Implementation notes

- Both appearances ship finished; test every screen in both. Dark is the stage
  default; light must survive a projector wash.
- All fixture numbers in this spec (scores, counts, loss values) are the demo's
  canonical dataset — keep them consistent across screens (the timeline, strip
  status, and verdict all reference the same version history: v1–v7 overnight,
  v8 promoted live, v9 rejected).
- Simulated vs real training is an engineering choice out of scope here; the
  design contract is only that data arrives at ≥ 1 update/s during Act 2 so the
  screen is visibly alive.
