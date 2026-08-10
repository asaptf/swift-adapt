import AdaptCore
import AdaptData
import AdaptEval
import AdaptRegistry
import AdaptSchedule
import AdaptTrain
import Foundation
import Testing

@Suite("Device policy & capability")
struct PolicyAndCapabilityTests {

    @Test("thermal .serious gates the run with typed error")
    func thermalGates() async throws {
        let root = try ScheduleTestSupport.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let lineage = ScheduleTestSupport.lineage("thermal")
        let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
        let buffer = try await ScheduleTestSupport.makeBuffer(
            lineage: lineage,
            root: root.appendingPathComponent("buf")
        )
        let pipeline = ScheduleTestSupport.makePipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            device: DeviceEnvironmentSnapshot(thermalState: .serious, isCharging: true)
        )

        var config = PipelineConfiguration(
            devicePolicy: .macOS,
            enforceCapabilityGate: false
        )
        // macOS policy still aborts on .serious
        config.devicePolicy.abortOnThermal = .serious

        let outcome = try await pipeline.run(configuration: config)
        guard case .policyBlocked(let error) = outcome.stopReason else {
            Issue.record("expected policyBlocked, got \(outcome.stopReason)")
            return
        }
        guard case .thermalStateTooHigh(let state) = error else {
            Issue.record("expected thermalStateTooHigh, got \(error)")
            return
        }
        #expect(state == .serious)
        #expect(outcome.record(for: .prune)?.executed == false)
    }

    @Test("battery and charging gates fire under iOS policy")
    func batteryAndCharging() async throws {
        let root = try ScheduleTestSupport.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let lineage = ScheduleTestSupport.lineage("battery")
        let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
        let buffer = try await ScheduleTestSupport.makeBuffer(
            lineage: lineage,
            root: root.appendingPathComponent("buf")
        )

        // Not charging
        let pipeline1 = ScheduleTestSupport.makePipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            device: DeviceEnvironmentSnapshot(
                thermalState: .nominal,
                batteryLevel: 0.9,
                isCharging: false
            )
        )
        let config = PipelineConfiguration(
            devicePolicy: .iOS,
            enforceCapabilityGate: false
        )
        let o1 = try await pipeline1.run(configuration: config)
        guard case .policyBlocked(let e1) = o1.stopReason else {
            Issue.record("expected notCharging block")
            return
        }
        #expect(e1 == .notCharging)

        // Low battery while charging
        let pipeline2 = ScheduleTestSupport.makePipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer,
            device: DeviceEnvironmentSnapshot(
                thermalState: .nominal,
                batteryLevel: 0.25,
                isCharging: true
            )
        )
        let o2 = try await pipeline2.run(configuration: config)
        guard case .policyBlocked(let e2) = o2.stopReason else {
            Issue.record("expected batteryTooLow block")
            return
        }
        guard case .batteryTooLow(let level, let minimum) = e2 else {
            Issue.record("expected batteryTooLow, got \(e2)")
            return
        }
        #expect(level == 0.25)
        #expect(minimum == 0.40)
    }

    @Test("capability gate refuses under-provisioned device with typed error")
    func capabilityGate() async throws {
        let root = try ScheduleTestSupport.tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let lineage = ScheduleTestSupport.lineage("capability")
        let registry = try AdapterRegistry(rootURL: root.appendingPathComponent("reg"))
        let buffer = try await ScheduleTestSupport.makeBuffer(
            lineage: lineage,
            root: root.appendingPathComponent("buf")
        )
        let pipeline = ScheduleTestSupport.makePipeline(
            lineage: lineage,
            registry: registry,
            buffer: buffer
        )

        let fourGB: UInt64 = 4 * 1_024 * 1_024 * 1_024
        let config = PipelineConfiguration(
            devicePolicy: .macOS,
            enforceCapabilityGate: true,
            physicalMemoryOverride: fourGB
        )
        let outcome = try await pipeline.run(configuration: config)
        guard case .policyBlocked(let error) = outcome.stopReason else {
            Issue.record("expected policyBlocked, got \(outcome.stopReason)")
            return
        }
        guard case .insufficientMemory(let have, let need) = error else {
            Issue.record("expected insufficientMemory, got \(error)")
            return
        }
        #expect(have == fourGB)
        #expect(need == AdaptCapability.minimumTrainingMemoryBytes)
        #expect(!AdaptCapability.canTrain(physicalMemoryBytes: fourGB))
        #expect(AdaptCapability.canTrain(physicalMemoryBytes: need))
    }

    @Test("DevicePolicy.enforce pure checks")
    func devicePolicyDirect() throws {
        try DevicePolicy.iOS.enforce(
            DeviceEnvironmentSnapshot(
                thermalState: .nominal,
                batteryLevel: 0.5,
                isCharging: true
            )
        )
        #expect(throws: AdaptScheduleError.self) {
            try DevicePolicy.iOS.enforce(
                DeviceEnvironmentSnapshot(
                    thermalState: .critical,
                    batteryLevel: 0.9,
                    isCharging: true
                )
            )
        }
    }
}
