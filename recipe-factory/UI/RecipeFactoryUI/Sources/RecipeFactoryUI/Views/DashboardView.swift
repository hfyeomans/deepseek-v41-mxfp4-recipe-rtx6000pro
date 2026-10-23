import SwiftUI

/// Home screen: the serve manifest strip, the headline cards, and the evidence picker.
/// In the real app the data is pulled over ssh (rsync of evidence/ + config/); the model types
/// are the same — see UI/JSON-SCHEMA.md.
struct DashboardView: View {
    @State private var selectedRung: String = MockData.headline.rungs[0].id

    private var rung: HeadlineRung {
        MockData.headline.rungs.first { $0.id == selectedRung }
            ?? MockData.headline.rungs[0]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            manifestStrip

            Picker("Rung", selection: $selectedRung) {
                ForEach(MockData.headline.rungs) { r in
                    Text(r.rung).tag(r.id)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 420)

            headlineGrid(rung)

            HStack(spacing: 12) {
                Label("NLL gate: \(rung.nll.delta, specifier: "%+.5f") (z \(rung.nll.z, specifier: "%.2f"))",
                      systemImage: "checkmark.seal")
                Label("Health: 0 NaN / 0 Xid", systemImage: "heart.text.square")
                Label(rung.evidence, systemImage: "doc.text.magnifyingglass")
            }
            .font(.callout)
            .foregroundStyle(.secondary)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var manifestStrip: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(MockData.manifest.model.repo).font(.headline)
                    Text("@\(String(MockData.manifest.model.snapshot.prefix(12)))")
                        .font(.system(.callout, design: .monospaced))
                    Text("profile: \(MockData.manifest.config.profile)").font(.callout)
                    if !MockData.manifest.config.overrides.isEmpty {
                        Text("overrides=[\(MockData.manifest.config.overrides.joined(separator: " "))]")
                            .font(.callout).foregroundStyle(.orange)
                    }
                    Spacer()
                    Button("Pull evidence now") { /* ssh rsync hook */ }
                }
                HStack(spacing: 16) {
                    Label("overlay \(MockData.manifest.engine.overlay_sha)", systemImage: "square.stack.3d.up")
                    Label("BIOS slot \(MockData.manifest.box.bios_slot) · FCLK \(MockData.manifest.box.hsmp.fclk_mhz)",
                          systemImage: "cpu")
                    Label("\(MockData.manifest.box.gpu_power_cap_w) W cap", systemImage: "bolt")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            .padding(6)
        }
    }

    private func headlineGrid(_ r: HeadlineRung) -> some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 170), spacing: 12)], spacing: 12) {
            metricCard("C1 decode", value: "\(r.c1.median_tok_s ?? 0, specifier: "%.1f") tok/s",
                       sub: "wall \(r.c1.wall_tok_s, specifier: "%.1f")")
            metricCard("C4 wall", value: "\(r.c4.wall_tok_s, specifier: "%.1f") tok/s", sub: "distinct prompts")
            metricCard("Prefill 16K", value: "\(r.prefill_16k.cold_tok_s ?? 0, specifier: "%.0f") tok/s",
                       sub: "TTFT \(r.prefill_16k.cold_ttft_s, specifier: "%.1f") s cold")
            metricCard("Prefill 32K", value: "\(r.prefill_32k.cold_tok_s ?? 0, specifier: "%.0f") tok/s",
                       sub: "TTFT \(r.prefill_32k.cold_ttft_s, specifier: "%.1f") s cold")
            metricCard("Needles", value: "\(r.needles.found)/\(r.needles.of)",
                       sub: "\(r.needles.prompt_tokens.formatted()) tokens · TTFT \(r.needles.ttft_s, specifier: "%.0f") s")
            metricCard("Stress", value: "\(r.stress.wall_s, specifier: "%.0f") s", sub: r.stress.shape)
        }
    }

    private func metricCard(_ title: String, value: String, sub: String) -> some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.caption).foregroundStyle(.secondary)
                Text(value).font(.title2.monospacedDigit())
                Text(sub).font(.caption2).foregroundStyle(.tertiary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(4)
        }
    }
}
