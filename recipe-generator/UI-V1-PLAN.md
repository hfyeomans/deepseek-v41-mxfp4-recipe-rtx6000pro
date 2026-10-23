# UI-V1-PLAN.md — the Recipe Factory app, screen by screen (Phase 2 of PROJECT-PLAN.md)

Concrete build plan for UI v1: a macOS SwiftUI app (the "lab table" from
[UI-THEME.md](UI-THEME.md)), built in a **git worktree** (`ui-app` branch, the worktree keeps the
app outside the session workspace so the filesystem guard cannot roll it back). Built with
MobileBuildMCP (`scaffold-macos` → `macos build-and-run` → `screencapture` → vision-model review →
iterate). Three milestones, each ending in a runnable, screenshot-reviewed build. Data contracts come
only from files that already exist — the app never invents numbers.

## Screen map (v1 scope)

| Screen | Reads | Renders | Acceptance |
|---|---|---|---|
| 1. **Plan viewer** | `placement_calc.py --json` output | tier plan table (component/GiB/tier/reason), resolved line, margins, projections, **trust banner** (calibrated → color-lit; provisional → monochrome "sheet on") | a real example plan renders; the trust switch changes the whole screen's color state |
| 2. **Tuning panel** | `levers.json` + `registry-surface.json` + live `placement_calc.py` runs | every lever as a physical throw: expert split (master breaker), map, D (ratchet, hard stop at 4), chunk (rotary), KV context×slots, quant, engine | turning a lever re-runs the pass and re-lights the plan; guards render in blood with the measured reason |
| 3. **Profile diff** | `profiles/*.json` | baseline vs each profile as lever deltas with the measured justification | switching profiles restates the deltas; drift from `levers.json` is flagged |
| 4. **Emit + publish** | `emit_recipe.py` | the emitted tree preview (README/resolved line/launcher/profiles), an **optional GitHub repo ask** (owner/name, private toggle) with the publish commands shown; run-with-consent only | emitting writes the tree; the repo ask never pushes without an explicit answer |
| 5. **Evidence witness** | stamped evidence JSON + `manifest.json` (UI/JSON-SCHEMA.md shapes) | gate results (NLL, needles, stress, consistency, health); numbers without a manifest render as glass (unreadable) | a stamped file renders with its manifest; an unstamped file refuses to render |

Non-goals for v1 (the discernment boundary, RECIPE-EMISSION.md §4): the app never serves, benchmarks,
gates, censuses kernels, downloads checkpoints, builds images, or touches BIOS. Each is a hand-off: the
app renders the exact commands the owner runs on the host.

## Milestones

| Milestone | Contents | Exit criterion |
|---|---|---|
| **M3a — skeleton** ✅ | scaffolded macOS target, theme tokens (`Theme`: storm palette, green reserved as `qualified`), window chrome, empty-state headline | builds clean, runs; done in the worktree (`ui-app`) |
| **M3b — plan viewer** ✅ | screen 1 with a bundled real plan (the dsv41 example plan, generated, committed beside the app) | the 219/165 plan renders with the calibrated trust banner; reviewed via screencapture + vision-model review |
| **M3c — tuning** ✅ / **emit** ⏳ | screens 2–5 wired to the CLI (`placement_calc.py --json` executed live; `emit_recipe.py` subprocess next) | the loop plan → tune works from the app on the 1017/dsv41 examples; emit + GitHub-ask pending |

Iteration loop per screen: `mobilebuildmcp macos build-and-run` → `screencapture` → review (the owner
live in Xcode, plus a V4.1-Flash vision subagent on the screenshots) → next screen. Each screen lands as
one commit on `ui-app`; the branch merges to master (and quorra) when the owner says.

## Theme decisions locked in (from UI-THEME.md)

- Palette: storm `#0B0E14` background, cyan `#5AD1FF` GPU tier, violet `#A78BFA` CPU tier, ember
  `#FF6B4A` prefill/power/NVMe, ivory `#E8E4DA` weights/tables, green `#3DD68C` **only** for
  qualified/fits (the code enforces it: `Theme.qualified` is the only green token).
- The trust switch is the one interaction the whole app hangs on: provisional = monochrome + sheet;
  calibrated = color-lit. Implemented in M3a/M3b.
- Type: system monospaced for all measurements/knobs, rounded display for prose. Motion ≤150 ms,
  mechanical.
- Code style: **modern SwiftUI + Swift 6.4 only** — `@MainActor @Observable`, Sendable model types,
  off-actor `Process` execution, `Grid`/`HSplitView`, no legacy AppKit bridging.

## File layout (builds where the skeleton already lives)

```
recipe-factory/UI/RecipeFactoryUI/           # the Swift package (data layer, stays)
recipe-factory/UI/RecipeFactoryApp/          # the macOS app target (this plan)
recipe-factory/UI/RecipeFactoryAppPackage/   # the feature package: Theme, PlanModels, TuningState, ContentView
recipe-generator/examples/dsv41-flash/plan.json   # bundled sample plan for the viewer (generated)
```

The app is built inside the recipe repo (in the `ui-app` worktree) for now; moving it to its own repo
(notarization/releases) is owner decision D1, executed at PROJECT-PLAN milestone M5.
