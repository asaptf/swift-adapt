import Foundation

/// When the night pipeline is allowed to burn energy (architecture §4.6).
///
/// ## Defaults
///
/// | Constraint | Default | Basis |
/// |---|---|---|
/// | Charging required | **iOS: yes / macOS: no** | §4.6 "Only when charging … (iOS)" |
/// | Min battery | **0.40** | §4.6 "battery > 40% (iOS)" |
/// | Abort thermal | **`.serious`** | §4.6 "Abort on thermal `.serious`" |
///
/// Battery and charging gates apply only when ``enforceBatteryAndCharging`` is
/// true (default on iOS). macOS defaults leave those soft so desktops are not
/// blocked by the absence of a UIDevice battery API.
public struct DevicePolicy: Sendable, Equatable, Hashable {
    /// When true, refuse to run unless `isCharging` is true.
    public var requireCharging: Bool
    /// Minimum battery fraction in `0...1` (exclusive lower bound in prose
    /// "battery > 40%" is implemented as `level >= minimumBatteryLevel`).
    public var minimumBatteryLevel: Double
    /// Abort when thermal state is **at or above** this value.
    public var abortOnThermal: ProcessInfo.ThermalState
    /// When false, skip charging/battery checks (typical macOS desktop path).
    public var enforceBatteryAndCharging: Bool

    public init(
        requireCharging: Bool,
        minimumBatteryLevel: Double,
        abortOnThermal: ProcessInfo.ThermalState = .serious,
        enforceBatteryAndCharging: Bool
    ) {
        self.requireCharging = requireCharging
        self.minimumBatteryLevel = minimumBatteryLevel
        self.abortOnThermal = abortOnThermal
        self.enforceBatteryAndCharging = enforceBatteryAndCharging
    }

    /// §4.6 iOS defaults: charging, battery ≥ 40%, abort on `.serious`.
    public static var iOS: DevicePolicy {
        DevicePolicy(
            requireCharging: true,
            minimumBatteryLevel: 0.40,
            abortOnThermal: .serious,
            enforceBatteryAndCharging: true
        )
    }

    /// macOS defaults: thermal abort only; no battery/charging gate.
    public static var macOS: DevicePolicy {
        DevicePolicy(
            requireCharging: false,
            minimumBatteryLevel: 0.0,
            abortOnThermal: .serious,
            enforceBatteryAndCharging: false
        )
    }

    /// Platform-appropriate default.
    public static var current: DevicePolicy {
        #if os(iOS)
        return .iOS
        #else
        return .macOS
        #endif
    }

    /// Validates ranges; throws ``AdaptScheduleError/invalidConfiguration``.
    public func validate() throws {
        guard minimumBatteryLevel >= 0, minimumBatteryLevel <= 1, minimumBatteryLevel.isFinite
        else {
            throw AdaptScheduleError.invalidConfiguration(
                "minimumBatteryLevel must be in [0, 1]"
            )
        }
    }

    /// Throws when `environment` violates this policy.
    public func enforce(_ environment: DeviceEnvironmentSnapshot) throws {
        try validate()
        if environment.thermalState >= abortOnThermal {
            throw AdaptScheduleError.thermalStateTooHigh(environment.thermalState)
        }
        guard enforceBatteryAndCharging else { return }
        if requireCharging, !environment.isCharging {
            throw AdaptScheduleError.notCharging
        }
        if environment.batteryLevel < minimumBatteryLevel {
            throw AdaptScheduleError.batteryTooLow(
                level: environment.batteryLevel,
                minimum: minimumBatteryLevel
            )
        }
    }
}
