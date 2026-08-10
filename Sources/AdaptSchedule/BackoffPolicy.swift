import Foundation

/// Exponential backoff after repeated promotion-gate **refusals** (§4.6).
///
/// ## Refusal vs abstention (do not collapse)
///
/// AdaptEval's `GateDecision` distinguishes:
/// - **refuse** — candidate failed the gate → **feeds backoff** (`feedsBackoff`)
/// - **abstain** — not enough evidence → **never feeds backoff**
/// - **promote** — success → **resets** backoff
///
/// A broken held-out pin (`.pinBroken`) is neither refuse nor abstain; it does
/// **not** feed backoff. Plateau detection only tracks refused candidates so
/// the device stops retraining a flat line without punishing data scarcity.
public struct BackoffPolicy: Sendable, Equatable, Hashable {
    /// Delay after the first consecutive refusal.
    public var baseDelay: Duration
    /// Cap on the exponential delay.
    public var maxDelay: Duration
    /// When false, backoff is never applied (tests / force-run).
    public var enabled: Bool

    public init(
        baseDelay: Duration = .seconds(24 * 60 * 60),
        maxDelay: Duration = .seconds(16 * 24 * 60 * 60),
        enabled: Bool = true
    ) {
        self.baseDelay = baseDelay
        self.maxDelay = maxDelay
        self.enabled = enabled
    }

    /// Validates delays; throws ``AdaptScheduleError/invalidConfiguration``.
    public func validate() throws {
        guard baseDelay > .zero else {
            throw AdaptScheduleError.invalidConfiguration("baseDelay must be positive")
        }
        guard maxDelay >= baseDelay else {
            throw AdaptScheduleError.invalidConfiguration("maxDelay must be ≥ baseDelay")
        }
    }

    /// Delay to apply after `consecutiveRefusals` refusals (1-based count).
    public func delay(afterConsecutiveRefusals consecutiveRefusals: Int) -> Duration {
        guard consecutiveRefusals > 0 else { return .zero }
        // base * 2^(n-1), capped at maxDelay (whole seconds).
        let shift = min(consecutiveRefusals - 1, 30)
        let multiplier = Int64(1 << shift)
        let baseSeconds = baseDelay.components.seconds
        let maxSeconds = maxDelay.components.seconds
        let scaled = baseSeconds * multiplier
        let capped = scaled < maxSeconds ? scaled : maxSeconds
        return .seconds(capped)
    }
}

/// Durable backoff counter for one lineage.
public struct BackoffState: Codable, Sendable, Equatable, Hashable {
    /// Number of consecutive gate **refusals** since the last reset.
    public var consecutiveRefusals: Int
    /// Wall-clock time when training may run again; `nil` if not deferred.
    public var nextEligibleAt: Date?

    public init(consecutiveRefusals: Int = 0, nextEligibleAt: Date? = nil) {
        self.consecutiveRefusals = consecutiveRefusals
        self.nextEligibleAt = nextEligibleAt
    }

    /// Whether training is currently deferred.
    public func isDeferred(at now: Date = Date()) -> Bool {
        guard let next = nextEligibleAt else { return false }
        return now < next
    }

    /// Applies a refusal: increments counter and schedules next eligibility.
    public mutating func recordRefusal(policy: BackoffPolicy, now: Date = Date()) {
        guard policy.enabled else { return }
        consecutiveRefusals += 1
        let delay = policy.delay(afterConsecutiveRefusals: consecutiveRefusals)
        nextEligibleAt = now.addingTimeInterval(delay.timeInterval)
    }

    /// Clears backoff (promote, or explicit reset). Abstention must call
    /// neither this nor ``recordRefusal`` — leave state unchanged.
    public mutating func reset() {
        consecutiveRefusals = 0
        nextEligibleAt = nil
    }
}

/// Loads and stores ``BackoffState`` beside a lineage (`backoff_state.json`).
public enum BackoffStore: Sendable {
    public static let fileName = "backoff_state.json"

    public static func url(in lineageDirectory: URL) -> URL {
        lineageDirectory.appendingPathComponent(fileName)
    }

    public static func load(from lineageDirectory: URL) throws -> BackoffState {
        let url = url(in: lineageDirectory)
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else {
            return BackoffState()
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw AdaptScheduleError.stageFailed(
                stage: .train,
                message: "Failed to read \(fileName): \(error.localizedDescription)"
            )
        }
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode(BackoffState.self, from: data)
        } catch {
            throw AdaptScheduleError.stageFailed(
                stage: .train,
                message: "Failed to decode \(fileName): \(error.localizedDescription)"
            )
        }
    }

    public static func save(_ state: BackoffState, to lineageDirectory: URL) throws {
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: lineageDirectory, withIntermediateDirectories: true)
        } catch {
            throw AdaptScheduleError.stageFailed(
                stage: .train,
                message: "Failed to create lineage directory for backoff: \(error.localizedDescription)"
            )
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data: Data
        do {
            data = try encoder.encode(state)
        } catch {
            throw AdaptScheduleError.stageFailed(
                stage: .train,
                message: "Failed to encode backoff state: \(error.localizedDescription)"
            )
        }
        let destination = url(in: lineageDirectory)
        let temp = lineageDirectory.appendingPathComponent(".tmp-\(UUID().uuidString)-\(fileName)")
        do {
            try data.write(to: temp, options: .atomic)
            if fm.fileExists(atPath: destination.path) {
                _ = try fm.replaceItemAt(
                    destination,
                    withItemAt: temp,
                    backupItemName: nil,
                    options: .usingNewMetadataOnly
                )
            } else {
                try fm.moveItem(at: temp, to: destination)
            }
        } catch {
            try? fm.removeItem(at: temp)
            throw AdaptScheduleError.stageFailed(
                stage: .train,
                message: "Failed to write \(fileName): \(error.localizedDescription)"
            )
        }
    }
}

extension Duration {
    /// Whole + fractional seconds as `TimeInterval` for `Date` arithmetic.
    fileprivate var timeInterval: TimeInterval {
        let c = components
        return TimeInterval(c.seconds) + TimeInterval(c.attoseconds) / 1e18
    }
}
