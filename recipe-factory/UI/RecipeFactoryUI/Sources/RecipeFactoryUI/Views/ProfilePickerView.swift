import SwiftUI

/// Profile Picker: each preset rendered with its knob deltas and why. Switching in the real app =
/// dry-run the typed registry over ssh first (show the refusal verbatim), then edit the systemd
/// drop-in and restart.
struct ProfilePickerView: View {
    @State private var selected: String? = nil

    private var preset: ProfilePreset {
        MockData.presets.first { $0.id == selected } ?? MockData.presets[0]
    }

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            List(MockData.presets, selection: $selected) { p in
                VStack(alignment: .leading, spacing: 2) {
                    Text(p.name).font(.headline)
                    Text(p.summary).font(.caption).foregroundStyle(.secondary)
                }
                .tag(Optional(p.id))
            }
            .frame(width: 300)

            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Text(preset.name).font(.title3.bold())
                    Text(preset.summary).font(.callout).foregroundStyle(.secondary)
                    Divider()
                    ForEach(Array(preset.knobs.enumerated()), id: \.offset) { _, k in
                        HStack(alignment: .firstTextBaseline) {
                            Text(k.key)
                                .font(.system(.callout, design: .monospaced))
                                .frame(width: 240, alignment: .leading)
                            Text(k.value).font(.callout.bold())
                            Text(k.why).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                    HStack {
                        Button("Dry-run registry") { /* ssh: serve_config.py shell, show refusals */ }
                        Button("Apply via systemd drop-in") { /* gated on dry-run success */ }
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}
