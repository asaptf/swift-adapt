import Foundation
import Network

/// Observes path reachability for the airplane-mode demo scene.
///
/// Uses `NWPathMonitor` and exposes status as an `AsyncStream` so the UI can
/// `for await` and flip chrome the instant the device goes offline.
///
/// This type **does not send traffic**. It only reads local path status from the
/// Network framework. Each subscription owns its own monitor instance.
public final class NetworkReachability: Sendable {
    /// Creates a reachability observer (stateless factory; each stream is independent).
    public init() {}

    /// Continuous stream of ``NetworkStatus`` values.
    ///
    /// Yields the current status as soon as the path handler fires, then every
    /// change. Cancelling the consumer cancels that subscription's monitor.
    public var updates: AsyncStream<NetworkStatus> {
        AsyncStream { continuation in
            let monitor = NWPathMonitor()
            let queue = DispatchQueue(label: "StyleMirror.NetworkReachability.\(UUID().uuidString)")
            monitor.pathUpdateHandler = { path in
                let status: NetworkStatus = path.status == .satisfied ? .online : .offline
                continuation.yield(status)
            }
            monitor.start(queue: queue)

            continuation.onTermination = { @Sendable _ in
                monitor.cancel()
            }
        }
    }

    /// One-shot snapshot of the current path.
    public static func currentStatus() async -> NetworkStatus {
        await withCheckedContinuation { (continuation: CheckedContinuation<NetworkStatus, Never>) in
            let monitor = NWPathMonitor()
            let queue = DispatchQueue(label: "StyleMirror.NetworkReachability.snapshot")
            let lock = NSLock()
            let box = ResumeBox()
            monitor.pathUpdateHandler = { path in
                let status: NetworkStatus = path.status == .satisfied ? .online : .offline
                monitor.cancel()
                lock.lock()
                let already = box.resumed
                box.resumed = true
                lock.unlock()
                guard !already else { return }
                continuation.resume(returning: status)
            }
            monitor.start(queue: queue)
        }
    }
}

/// Non-Sendable box used only behind a lock for one-shot resume.
private final class ResumeBox: @unchecked Sendable {
    var resumed = false
}
