import SwiftUI

/// Trade-off Explorer: the knob families with their measured trade-offs. The explorer never
/// extrapolates — each row is a measured datum with its evidence directory; guard boundaries
/// (regions the registry refuses) are drawn as warnings, not curves.
struct TradeoffExplorerView: View {
    private let families = Dictionary(grouping: MockData.knobs, by: \.family)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Trade-off Explorer").font(.title2.bold())
            Text("Every row is a measured number with its evidence directory; the guard column names the registry refusal that keeps a condemned setting from coming back.")
                .font(.callout).foregroundStyle(.secondary)

            ForEach(families.keys.sorted(), id: \.self) { family in
                GroupBox {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(family).font(.headline).padding(.bottom, 2)
                        ForEach(families[family] ?? [], id: \.id) { knob in
                            HStack(alignment: .firstTextBaseline) {
                                Text(knob.name)
                                    .font(.system(.callout, design: .monospaced))
                                    .frame(width: 260, alignment: .leading)
                                Text("default: \(knob.defaultValue)")
                                    .font(.callout.bold())
                                    .frame(width: 200, alignment: .leading)
                                Text(knob.measured).font(.callout).foregroundStyle(.secondary)
                                Spacer()
                                Text(knob.evidence).font(.caption2).foregroundStyle(.tertiary)
                            }
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
