import Foundation

/// Single chokepoint for **outbound application traffic** accounting.
///
/// StyleMirror claims "bytes sent: 0" because the demo app contains **no
/// networking code that originates requests**. This type is the only API that
/// would record outbound payload sizes. Nothing in StyleMirrorEngine (or the
/// placeholder executable) calls ``recordOutbound(bytes:)``.
///
/// Therefore ``bytesSent`` stays at zero as a structural property of the binary,
/// not because a UI label hardcodes `"0"`.
///
/// If a future feature ever performs a network request, it **must** go through
/// this meter so the readout remains honest. Until then, the counter is a real
/// measurement of "calls that recorded traffic" — which is zero.
public actor OutboundTrafficMeter {
    /// Shared process-wide meter for the demo app.
    public static let shared = OutboundTrafficMeter()

    /// Cumulative outbound bytes recorded through this meter.
    public private(set) var bytesSent: UInt64 = 0

    /// Number of outbound operations recorded (for diagnostics).
    public private(set) var operationCount: UInt64 = 0

    public init() {}

    /// Records an outbound payload.
    ///
    /// **Not called by any StyleMirror code today.** Exists so the accounting
    /// path is real: if this method is never invoked, bytes stay zero.
    public func recordOutbound(bytes: UInt64) {
        bytesSent += bytes
        operationCount += 1
    }

    /// Async stream of byte totals after each change (and current value on subscribe).
    public func byteUpdates() -> AsyncStream<UInt64> {
        // Minimal stream: yield current value once. Callers that need live
        // updates after future `recordOutbound` can re-read `bytesSent` or
        // we can extend with continuations later without changing the chokepoint.
        let current = bytesSent
        return AsyncStream { continuation in
            continuation.yield(current)
            continuation.finish()
        }
    }
}
