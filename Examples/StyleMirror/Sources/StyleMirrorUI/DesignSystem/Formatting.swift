import Foundation

/// Number formatting for on-screen data.
///
/// The interface is English, so numbers are formatted in a fixed English locale
/// rather than the machine's. Left to `.formatted()`'s default, a Mac set to a
/// European locale renders "14,80 (limit 3,16)" beside English copy.
extension Double {
    /// Fixed-locale decimal string with `fractionDigits` after the point.
    func demoNumber(_ fractionDigits: Int) -> String {
        formatted(
            .number
                .precision(.fractionLength(fractionDigits))
                .locale(Locale(identifier: "en_US"))
        )
    }
}

extension Int {
    /// Fixed-locale grouped integer, e.g. `42,380`.
    var demoNumber: String {
        formatted(.number.locale(Locale(identifier: "en_US")))
    }
}
