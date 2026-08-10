# StyleMirror: the demo, screen by screen

`Examples/StyleMirror` is a macOS SwiftUI app built to be performed live in about five minutes. It currently runs on a scripted engine, not real training — the library's training loop is not wired in yet. What the demo shows is the product argument: what an app built on Adapt looks like when it works, and which screens carry that argument. Four of them do.

## Act 1: airplane mode

![Offline as the feature](images/01-offline.png)

The presenter switches on airplane mode in front of the audience before anything else happens. The app treats offline as the feature, not a degraded state, and a "0 B sent" counter stays on screen for the rest of the demo. That counter is a real measurement, not a printed zero: the app routes all outbound traffic through a single chokepoint, and it never calls it. The screen's argument: you don't have to trust a privacy policy when there is no network path to police.

## The gate says no

![The eval gate refusing a bad adapter](images/02-gate.png)

An adapter trained on deliberately corrupted examples arrives at the evaluation gate, and the gate refuses to promote it. The panel is green while the failed numbers are red, because the refusal is the system working as designed. The active adapter is unchanged; the user's experience never got worse. This is the screen that distinguishes a library from a script — anyone can train an adapter, the hard part is declining to ship a bad one.

## The blind test

![Base model, adapter, and the human — unlabeled](images/03-blind-test.png)

One incoming email, three candidate replies: the base model, the adapter, and the user's real archived reply. The audience guesses which is human. All three cards are deliberately identical in size and length class, so nothing but the writing distinguishes them. The argument: if the room hesitates, the adapter has captured something real about the voice.

## Same voice, three languages

![One adapter answering in English, Spanish and Russian](images/04-languages.png)

The same adapter answers in English, Spanish and Russian. The base model's reply is rendered dimmer and smaller than the adapter's on purpose — the layout is making the argument. What the adapter learned is not a set of stock phrases in one language; it is a voice, and the voice survives translation.
