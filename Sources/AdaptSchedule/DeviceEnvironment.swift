import Foundation

#if canImport(UIKit)
import UIKit
#endif

/// Snapshot of host conditions the night pipeline consults before training.
///
/// Production code reads `ProcessInfo` / `UIDevice`; tests inject fixed values
/// so thermal and battery policies are exercised without hardware.
public struct DeviceEnvironmentSnapshot: Sendable, Equatable, Hashable {
    /// Current thermal pressure.
    public var thermalState: ProcessInfo.ThermalState
    /// Battery fraction in `0...1`. On macOS without a battery this is `1.0`.
    public var batteryLevel: Double
    /// Whether the device is on external power (or battery state is unknown/full).
    public var isCharging: Bool
    /// Low Power Mode (iOS) — informational; not a hard gate by default.
    public var isLowPowerModeEnabled: Bool

    public init(
        thermalState: ProcessInfo.ThermalState = .nominal,
        batteryLevel: Double = 1.0,
        isCharging: Bool = true,
        isLowPowerModeEnabled: Bool = false
    ) {
        self.thermalState = thermalState
        self.batteryLevel = batteryLevel
        self.isCharging = isCharging
        self.isLowPowerModeEnabled = isLowPowerModeEnabled
    }
}

/// Reads live device conditions for policy checks.
public protocol DeviceEnvironmentReading: Sendable {
    /// Captures a consistent snapshot for one policy evaluation.
    func snapshot() -> DeviceEnvironmentSnapshot
}

/// Default reader: `ProcessInfo` thermal + (on iOS) `UIDevice` battery.
public struct SystemDeviceEnvironment: DeviceEnvironmentReading {
    public init() {}

    public func snapshot() -> DeviceEnvironmentSnapshot {
        #if os(iOS)
        let device = UIDevice.current
        let wasMonitoring = device.isBatteryMonitoringEnabled
        if !wasMonitoring {
            device.isBatteryMonitoringEnabled = true
        }
        defer {
            if !wasMonitoring {
                device.isBatteryMonitoringEnabled = false
            }
        }
        let level = Double(device.batteryLevel)
        // `-1` means battery monitoring failed / simulator quirk — treat as full
        // and charging so simulator hosts are not falsely blocked.
        let normalizedLevel = level < 0 ? 1.0 : min(1.0, max(0.0, level))
        let state = device.batteryState
        let charging =
            state == .charging || state == .full || level < 0
        return DeviceEnvironmentSnapshot(
            thermalState: ProcessInfo.processInfo.thermalState,
            batteryLevel: normalizedLevel,
            isCharging: charging,
            isLowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
        #else
        // macOS: no UIDevice battery API in this path. Desktops are treated as
        // "charging" with full battery so the iOS-oriented gates do not fire.
        // Laptop battery could be added later via IOKit if product needs it.
        return DeviceEnvironmentSnapshot(
            thermalState: ProcessInfo.processInfo.thermalState,
            batteryLevel: 1.0,
            isCharging: true,
            isLowPowerModeEnabled: ProcessInfo.processInfo.isLowPowerModeEnabled
        )
        #endif
    }
}

/// Fixed snapshot for offline tests.
public struct FixedDeviceEnvironment: DeviceEnvironmentReading {
    public var value: DeviceEnvironmentSnapshot

    public init(_ value: DeviceEnvironmentSnapshot) {
        self.value = value
    }

    public func snapshot() -> DeviceEnvironmentSnapshot { value }
}
