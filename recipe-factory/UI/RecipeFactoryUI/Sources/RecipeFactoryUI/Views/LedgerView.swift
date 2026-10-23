import SwiftUI

/// Ledger Browser: the tried-options ledger as structured data. Filter by family and verdict;
/// the reopen trigger is the only legitimate way back to a rejected option.
struct LedgerView: View {
    @State private var familyFilter: String? = nil
    @State private var verdictFilter: LedgerVerdict? = nil

    private var entries: [LedgerEntry] {
        MockData.ledger.filter { entry in
            (familyFilter == nil || entry.family == familyFilter)
                && (verdictFilter == nil || entry.verdict == verdictFilter)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Tried-Options Ledger").font(.title2.bold())
            Text("A rejected option is a recorded dead end. Re-measure only when a listed reopen trigger holds on your hardware.")
                .font(.callout).foregroundStyle(.secondary)

            HStack {
                Picker("Family", selection: $familyFilter) {
                    Text("all families").tag(String?.none)
                    ForEach(LedgerEntry.families, id: \.self) { f in Text(f).tag(String?.some(f)) }
                }
                Picker("Verdict", selection: $verdictFilter) {
                    Text("all verdicts").tag(LedgerVerdict?.none)
                    ForEach(LedgerVerdict.allCases, id: \.self) { v in Text(v.rawValue).tag(LedgerVerdict?.some(v)) }
                }
                Spacer()
                Text("\(entries.count) entries").font(.caption).foregroundStyle(.secondary)
            }
            .frame(maxWidth: 640)

            Table(entries) {
                TableColumn("ID", value: \.id).width(70)
                TableColumn("Family", value: \.family).width(110)
                TableColumn("Option") { entry in Text(entry.option) }
                TableColumn("Outcome (measured)") { entry in Text(entry.outcome) }
                TableColumn("Verdict") { entry in
                    Text(entry.verdict.rawValue).bold().foregroundStyle(color(for: entry.verdict))
                }
                .width(80)
                TableColumn("Reopen trigger") { entry in
                    Text(entry.reopen ?? "—").foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func color(for verdict: LedgerVerdict) -> Color {
        switch verdict {
        case .adopt: .green
        case .rejected: .red
        case .closed: .orange
        case .parked: .secondary
        }
    }
}
