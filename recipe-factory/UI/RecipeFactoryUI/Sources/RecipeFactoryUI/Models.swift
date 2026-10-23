import Foundation

// MARK: - Serve manifest (manifest.json) — see UI/JSON-SCHEMA.md §1

struct ServeManifest: Codable, Identifiable {
    struct Recipe: Codable { let repo_sha: String; let dirty: Bool }
    struct Engine: Codable {
        let name: String
        let base: String
        let overlay_sha: String
        let image_digest: String
        let container_id: String?
    }
    struct Model: Codable { let repo: String; let snapshot: String }
    struct Config: Codable {
        let profile: String
        let overrides: [String]
        let resolved: String
        let knobs: [String: String]
    }
    struct HSMP: Codable { let fclk_mhz: Int; let mclk_mhz: Int }
    struct Box: Codable {
        let bios_slot: String
        let hsmp: HSMP
        let gpu_power_cap_w: Int
        let gpus: Int
        let meminfo: [String: Double]
        let engram_residency: [String: Double]
    }
    let recipe: Recipe
    let engine: Engine
    let model: Model
    let config: Config
    let box: Box
    let created_utc: String

    var id: String { created_utc }
    /// Non-loopback exposure without a key is a flag worth surfacing in the UI.
    var isVerified: Bool { true }
}

// MARK: - Headline rung — see UI/JSON-SCHEMA.md §3

struct HeadlineRung: Codable, Identifiable {
    struct Rate: Codable { let median_tok_s: Double?; let wall_tok_s: Double }
    struct Prefill: Codable { let cold_tok_s: Double?; let cold_ttft_s: Double }
    struct Needles: Codable { let found: Int; let of: Int; let prompt_tokens: Int; let ttft_s: Double }
    struct Stress: Codable { let wall_s: Double; let shape: String }
    struct NLL: Codable {
        let delta: Double; let sd: Double; let z: Double
        let moved: Int; let total: Int; let baseline: String
    }
    let rung: String
    let launcher_default: Bool
    let c1: Rate
    let c4: Rate
    let prefill_16k: Prefill
    let prefill_32k: Prefill
    let needles: Needles
    let stress: Stress
    let nll: NLL
    let evidence: String

    var id: String { rung }
}

struct HeadlineDocument: Codable {
    let generated_by: String
    let rungs: [HeadlineRung]
}

// MARK: - Knob with its measured trade-off (KNOBS-AND-PROFILES §2, as data)

struct KnobTradeoff: Identifiable, Hashable {
    let name: String
    let family: String
    let defaultValue: String
    let measured: String      // the trade-off sentence, with numbers
    let evidence: String

    var id: String { name }
}

// MARK: - Ledger entry (TRIED-OPTIONS-LEDGER, as data)

enum LedgerVerdict: String, Codable, CaseIterable {
    case adopt = "ADOPT"
    case rejected = "REJ"
    case closed = "CLOSED"
    case parked = "PARKED"
}

struct LedgerEntry: Identifiable {
    let id: String            // experiment id, e.g. "E23"
    let family: String        // engine / speculation / placement / ...
    let option: String
    let outcome: String
    let verdict: LedgerVerdict
    let reopen: String?       // the only legitimate way back

    static let families = ["engine", "speculation", "placement", "deferral",
                           "chunk", "engram", "kernels", "threads", "bios"]
}

// MARK: - Profile preset (KNOBS-AND-PROFILES §3, as data)

struct ProfilePreset: Identifiable {
    let name: String
    let summary: String
    let knobs: [(key: String, value: String, why: String)]

    var id: String { name }
}
