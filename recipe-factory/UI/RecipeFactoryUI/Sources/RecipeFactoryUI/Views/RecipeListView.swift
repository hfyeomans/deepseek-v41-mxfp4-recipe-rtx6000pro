import SwiftUI

/// Recipe Builder Wizard (P0→P4) — v0 placeholder that shows the step structure and, per step,
/// the command the wizard would run over ssh and the file it would write. The wizard never hides
/// the methodology: every step renders its command.
struct RecipeListView: View {
    private let steps: [(id: String, title: String, detail: String, command: String, writes: String)] = [
        ("p0", "P0 — Model + feasibility", "Checkpoint byte budget from safetensors headers, then the placement calculator across layouts × GPU counts.",
         "checkpoint-budget.py <repo> && feasibility-calc.py budget.json --gpus 2,3,4 --layout hybrid",
         "evidence/budget.json, docs/FEASIBILITY.md"),
        ("p1", "P1 — Host microbenchmarks", "DRAM read, PCIe H2D + additivity, NVMe random gather, GPU copy: every vendor spec becomes a measurement.",
         "bench-dram.py && bench-pcie-h2d.py && bench-engram-gather.py",
         "evidence/e1-*.json → docs/HARDWARE.md (zero estimate rows)"),
        ("p2", "P2 — Baseline", "Re-measure the incumbent deployment with THIS harness so every later number compares.",
         "bench-decode.py --concurrency 1,4 && bench-prefill.py --warm-pass",
         "evidence/baseline/"),
        ("p3", "P3 — Build + gates", "Declare knobs in the typed registry, then NLL identity → verify/needles → stress → batch consistency → health.",
         "serve_config.py shell && serve.sh && nll-score.py --identity-dir ...",
         "scripts/serve_config.py, config/profiles/, evidence/<task>/"),
        ("p4", "P4 — Scale", "The 3/4-GPU or upgraded-CPU columns on the same fixture and power condition.",
         "feasibility-calc.py --gpus 3,4 ...",
         "one comparison table in docs/BENCHMARKS.md"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recipe Builder Wizard").font(.title2.bold())
            Text("Each step runs the recipe's own scripts over ssh — the UI executes the same code path CI trusts, never a re-implementation.")
                .font(.callout).foregroundStyle(.secondary)

            ForEach(steps, id: \.id) { step in
                GroupBox {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(step.title).font(.headline)
                        Text(step.detail).font(.callout).foregroundStyle(.secondary)
                        Text(step.command)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .padding(6)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
                        HStack {
                            Text("writes: \(step.writes)").font(.caption2).foregroundStyle(.tertiary)
                            Spacer()
                            Button("Run") { /* ssh exec hook */ }
                        }
                    }
                    .padding(6)
                }
            }
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
