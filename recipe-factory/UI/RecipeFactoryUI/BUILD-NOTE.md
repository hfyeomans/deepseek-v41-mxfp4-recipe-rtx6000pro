# RecipeFactoryUI — build note

This is a minimal SwiftUI skeleton (data models + five tab views + sample evidence JSON) for the app
described in [../macOS-UI-PLAN.md](../macOS-UI-PLAN.md). It is a proof of concept for the *data
contract*, not the app.

Verification status at commit time:

- `Models.swift` + `MockData.swift` (the entire data layer the UI consumes) **typecheck cleanly**
  with the macOS 27.2 / Swift 6.4 toolchain (`swiftc -typecheck`).
- The SwiftUI view files could not be fully build-verified in the authoring sandbox: SwiftPM's
  macro-plugin server (used by `@State`/`@Observable`) is blocked there (`swift-plugin-server
  produced malformed response`). Run `swift build` on an unsandboxed Mac with Xcode ≥ 16 to verify;
  the views are intentionally plain SwiftUI (no external dependencies).

Real-app wiring left for later (per the plan):

- replace `MockData` with a loader that decodes the rsync-pulled JSON (schema in
  [../JSON-SCHEMA.md](../JSON-SCHEMA.md)) into the same model types;
- the ssh transport (`Process` + `/usr/bin/ssh`, ssh-config alias, or SwiftNIO SSH);
- migration from Swift Package to an `.xcodeproj` app target for signing/notarization
  (Developer ID + `notarytool` + Sparkle — checklist in the plan §4).
