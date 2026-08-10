import AdaptCore
import Foundation

/// Shared blind-round role assignment, seeded shuffle, and tally scoring.
///
/// Used by ``ScriptedEngine`` and ``AdaptEngine`` so shuffle / scoring behaviour
/// stays identical regardless of how the three bodies were produced.
public enum BlindRoundSupport: Sendable {
    /// One open round waiting for a guess.
    public struct OpenRound: Sendable {
        public let roleByCandidate: [UUID: ReplyRole]
        public let humanCandidateID: UUID
        public var resolved: Bool

        public init(
            roleByCandidate: [UUID: ReplyRole],
            humanCandidateID: UUID,
            resolved: Bool = false
        ) {
            self.roleByCandidate = roleByCandidate
            self.humanCandidateID = humanCandidateID
            self.resolved = resolved
        }
    }

    /// Builds a ``BlindTestRound`` from three role-tagged bodies and shuffles
    /// display order with `seed &+ roundIndex`.
    public static func prepareRound(
        incomingEmailID: String,
        incoming: EmailMessage,
        bodiesByRole: [(ReplyRole, String)],
        seed: UInt64,
        roundIndex: UInt64
    ) -> (round: BlindTestRound, open: OpenRound) {
        var tagged = bodiesByRole
        var rng = SeededGenerator(seed: seed &+ roundIndex &* 0xA5A5)
        // Fisher–Yates
        for i in stride(from: tagged.count - 1, through: 1, by: -1) {
            let j = Int(rng.next() % UInt64(i + 1))
            tagged.swapAt(i, j)
        }

        var roleByCandidate: [UUID: ReplyRole] = [:]
        var humanID: UUID?
        let candidates: [BlindCandidate] = tagged.map { role, body in
            let id = deterministicID(seed: seed, round: roundIndex, role: role)
            roleByCandidate[id] = role
            if role == .human { humanID = id }
            return BlindCandidate(id: id, body: body)
        }
        let resolvedHumanID = humanID ?? candidates[0].id
        let roundID = deterministicID(seed: seed, round: roundIndex, role: nil)
        let round = BlindTestRound(
            id: roundID,
            incomingEmailID: incomingEmailID,
            incoming: incoming,
            candidates: candidates
        )
        let open = OpenRound(
            roleByCandidate: roleByCandidate,
            humanCandidateID: resolvedHumanID,
            resolved: false
        )
        return (round, open)
    }

    /// Scores a guess and returns the updated tally.
    public static func scoreGuess(
        open: OpenRound,
        roundID: UUID,
        candidateID: UUID,
        previousTally: BlindTestTally
    ) throws -> (result: BlindTestGuessResult, tally: BlindTestTally) {
        guard !open.resolved else {
            throw StyleMirrorError.invalidState("round \(roundID) already scored")
        }
        guard let guessedRole = open.roleByCandidate[candidateID] else {
            throw StyleMirrorError.unknownCandidate(candidateID.uuidString)
        }

        let identifiedHuman = guessedRole == .human
        let adapterMistaken = guessedRole == .adaptedModel
        let baseMistaken = guessedRole == .baseModel

        let tally = BlindTestTally(
            roundsPlayed: previousTally.roundsPlayed + 1,
            humanCorrectlyIdentified: previousTally.humanCorrectlyIdentified
                + (identifiedHuman ? 1 : 0),
            adapterMistakenForHuman: previousTally.adapterMistakenForHuman
                + (adapterMistaken ? 1 : 0),
            baseMistakenForHuman: previousTally.baseMistakenForHuman
                + (baseMistaken ? 1 : 0)
        )

        let result = BlindTestGuessResult(
            roundID: roundID,
            guessedCandidateID: candidateID,
            guessedRole: guessedRole,
            humanCandidateID: open.humanCandidateID,
            identifiedHuman: identifiedHuman,
            adapterMistakenForHuman: adapterMistaken,
            reveal: open.roleByCandidate,
            tally: tally
        )
        return (result, tally)
    }

    /// Deterministic UUID from seed + round + optional role (reproducible shuffles).
    public static func deterministicID(
        seed: UInt64,
        round: UInt64,
        role: ReplyRole?
    ) -> UUID {
        var bytes = [UInt8](repeating: 0, count: 16)
        var x = seed ^ (round &* 0x9E3779B97F4A7C15)
        if let role {
            x ^= UInt64(role.rawValue.utf8.reduce(0) { ($0 &<< 5) &+ $0 &+ UInt64($1) })
        } else {
            x ^= 0xDEADBEEF
        }
        for i in 0..<8 {
            bytes[i] = UInt8((x >> (i * 8)) & 0xff)
        }
        let y = x &* 0xBF58476D1CE4E5B9
        for i in 0..<8 {
            bytes[8 + i] = UInt8((y >> (i * 8)) & 0xff)
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
