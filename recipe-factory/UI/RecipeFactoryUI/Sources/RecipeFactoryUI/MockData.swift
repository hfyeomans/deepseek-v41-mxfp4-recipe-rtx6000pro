import Foundation

/// Mock data mirroring the shapes in UI/JSON-SCHEMA.md, using this project's measured
/// numbers (evidence/t352/default-flip and the TRIED-OPTIONS-LEDGER). In the real app these
/// come from `rsync`-pulled JSON over ssh — see UI/macOS-UI-PLAN.md §3.
enum MockData {
    static let manifest = ServeManifest(
        recipe: .init(repo_sha: "8d02f85", dirty: false),
        engine: .init(name: "sglang-hybrid",
                      base: "pinned-upstream-commit",
                      overlay_sha: "deec7a6f",
                      image_digest: "sha256:314749b5...",
                      container_id: nil),
        model: .init(repo: "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                     snapshot: "1bae0e1088993b9b82de08f59de5be5a34e42170"),
        config: .init(profile: "qualified",
                      overrides: [],
                      resolved: "resolved: experts gpu=219 cpu=165 deferral D=4 chunk=8192 "
                                + "context=262144 slots=4 profile=qualified overrides=[]",
                      knobs: ["KT_NUM_GPU_EXPERTS": "219", "CHUNKED_PREFILL_SIZE": "8192",
                              "KT_MAX_DEFERRED": "4"]),
        box: .init(bios_slot: "3",
                   hsmp: .init(fclk_mhz: 2100, mclk_mhz: 3200),
                   gpu_power_cap_w: 325,
                   gpus: 2,
                   meminfo: ["MemTotal_kB": 258_800_000, "MemAvailable_kB": 40_000_000],
                   engram_residency: ["layer1": 0.42, "layer14": 0.44]),
        created_utc: "2026-10-23T14:57:16Z"
    )

    static let headline = HeadlineDocument(
        generated_by: "benchmarks_headline.py",
        rungs: [
            HeadlineRung(
                rung: "262K × 4 (launcher default)", launcher_default: true,
                c1: .init(median_tok_s: 60.8, wall_tok_s: 58.8),
                c4: .init(median_tok_s: nil, wall_tok_s: 118.9),
                prefill_16k: .init(cold_tok_s: 4344.0, cold_ttft_s: 3.7),
                prefill_32k: .init(cold_tok_s: 4495.0, cold_ttft_s: 7.1),
                needles: .init(found: 6, of: 6, prompt_tokens: 255_388, ttft_s: 59.0),
                stress: .init(wall_s: 278.0, shape: "C4 × 253,428 tokens"),
                nll: .init(delta: -0.00033, sd: 0.0013, z: 0.26,
                           moved: 65_226, total: 65_653, baseline: "evidence/t328/.../corpus"),
                evidence: "evidence/t352/default-flip"),
            HeadlineRung(
                rung: "1M × 1 (per-rung override)", launcher_default: false,
                c1: .init(median_tok_s: 41.3, wall_tok_s: 39.4),
                c4: .init(median_tok_s: nil, wall_tok_s: 0),
                prefill_16k: .init(cold_tok_s: 1659.0, cold_ttft_s: 9.6),
                prefill_32k: .init(cold_tok_s: 1657.0, cold_ttft_s: 19.3),
                needles: .init(found: 6, of: 6, prompt_tokens: 1_022_456, ttft_s: 447.8),
                stress: .init(wall_s: 496.6, shape: "C1 × 1,022,456 tokens"),
                nll: .init(delta: 0.0, sd: 0.0, z: 0.0,
                           moved: 0, total: 65_653, baseline: "evidence/t1017/nll/baseline"),
                evidence: "evidence/t1017/"),
        ]
    )

    static let knobs: [KnobTradeoff] = [
        .init(name: "KT_EXPERT_MAP", family: "placement", defaultValue: "el8 mix map",
              measured: "CPU route share 42 → 15%; C1 +37%, C4 +120%; static-map floor 10–15%",
              evidence: "evidence/t38b/"),
        .init(name: "KT_NUM_GPU_EXPERTS", family: "placement", defaultValue: "219",
              measured: "230 fails the NLL gate; 226 was the pre-chunk-8192 fit ceiling",
              evidence: "evidence/e29/, evidence/t328/"),
        .init(name: "KT_MAX_DEFERRED", family: "deferral", defaultValue: "4",
              measured: "D=4 +22% C1; D=5 fails NLL (+0.00287) and greedy consistency",
              evidence: "evidence/e21-deferral/, evidence/t315-d5/"),
        .init(name: "CHUNKED_PREFILL_SIZE", family: "chunk", defaultValue: "8192",
              measured: "16384: +22% prefill at −4.4% C1; 2048: −40% prefill",
              evidence: "evidence/e34-indexer-memory/"),
        .init(name: "HOST_THREAD_CPUS", family: "threads", defaultValue: "auto",
              measured: "ends the C4 106/123 bimodality; prefill −0.7…−1.9%",
              evidence: "evidence/t352/"),
        .init(name: "ENGRAM_ADMISSION_PREFETCH", family: "engram", defaultValue: "0",
              measured: "−0.2…−0.4 s (3–9%) on a cold 16K TTFT admitted during decode",
              evidence: "evidence/t349/"),
    ]

    static let ledger: [LedgerEntry] = [
        .init(id: "E23", family: "speculation", option: "DSpark/MTP K=5",
              outcome: "C1 −16%, C3/C4 −69…−77%; verify rows multiply CPU-expert traffic",
              verdict: .rejected, reopen: "a CPU-tier change (third GPU / more CCDs)"),
        .init(id: "T3.9h-b", family: "speculation", option: "n-gram (chain-only verify)",
              outcome: "+60/+76% C1 on copy, −26% review, −63% C4; decode-state defect under rejected drafts",
              verdict: .rejected, reopen: "single-user copy-heavy deployment, passing spec-consistency.py"),
        .init(id: "E30", family: "engine", option: "B2 PCIe expert streaming",
              outcome: "+2% C1 at 40 experts/layer, −3% at 60 (0.21 ms/expert vs kt 0.146)",
              verdict: .rejected, reopen: nil),
        .init(id: "E29", family: "placement", option: "230 GPU experts",
              outcome: "+2.5% C1 wall for −40% prefill; NLL +0.0088 fails the gate",
              verdict: .rejected, reopen: nil),
        .init(id: "T3.8c", family: "placement", option: "own-counts map refit",
              outcome: "loses to the el8 map at every blend on held-out families; content-specific hot sets",
              verdict: .closed, reopen: "≥3 new prompt families × ≥20K decode tokens each"),
        .init(id: "E40", family: "engram", option: "re-quantized Engram tables (FP4/q3_K)",
              outcome: "cannot shrink the page-cache footprint (~190 KiB per first-touched token)",
              verdict: .rejected, reopen: nil),
        .init(id: "T3.32", family: "engine", option: "TensorFold as engine",
              outcome: "native registry is Nemotron-only; no CPU/DRAM tier; 2.9–3.0 bpw forks 12–24% behind",
              verdict: .rejected, reopen: "all six triggers in TENSORFOLD-ASSESSMENT §5"),
        .init(id: "T3.8b", family: "placement", option: "el8 mix hot-expert map",
              outcome: "C1 42.4 → 58.0 (+37%), C4 +120%; transfers unchanged to the 1017 weights",
              verdict: .adopt, reopen: nil),
        .init(id: "T3.52", family: "threads", option: "HOST_THREAD_CPUS=auto",
              outcome: "C4 126.7/123.0 with no 106 mode; NLL bit-identical",
              verdict: .adopt, reopen: nil),
    ]

    static let presets: [ProfilePreset] = [
        .init(name: "qualified", summary: "262K × 4 — what every headline number was measured on",
              knobs: [("CONTEXT_LENGTH", "262144", "the launcher default rung"),
                      ("MAX_RUNNING_REQUESTS", "4", "qualified ceiling at 262K")]),
        .init(name: "coding-agent", summary: "C1 / TTFT-sensitive: one agent, long sessions",
              knobs: [("MAX_RUNNING_REQUESTS", "1–2", "single streaming agent"),
                      ("ENGRAM_ADMISSION_PREFETCH", "1", "chunk-0 prefetch: −0.2…−0.4 s per cold TTFT"),
                      ("KERNEL_CACHE_DIR", "persistent", "no +14–19 s Triton-compile tax")]),
        .init(name: "multi-user", summary: "C4: several concurrent sessions",
              knobs: [("HOST_THREAD_CPUS", "auto", "non-negotiable: the C4 bimodality fix"),
                      ("CUDA_GRAPH_BS", "1 2 3 4", "explicit bs=3 capture: +28% at C3")]),
        .init(name: "long-context-1m", summary: "1M × 1 single giant context",
              knobs: [("CONTEXT_LENGTH", "1048576", "top of the ladder"),
                      ("MAX_RUNNING_REQUESTS", "1", "multi-slot above 262K never ran"),
                      ("MEM_FRACTION_STATIC", "0.962", "0.96 silently caps the pool at 505,856 tokens")]),
        .init(name: "quality-first", summary: "benchmark-grade reasoning + determinism",
              knobs: [("REASONING_EFFORT", "max", "benchmark reasoning budget (100)"),
                      ("KT_MAX_DEFERRED", "4 (never 5)", "D=5 fails NLL — the gate is the floor")]),
    ]
}
