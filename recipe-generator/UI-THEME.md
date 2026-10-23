# UI-THEME.md — cyberpunk × AI tropes × Frankenstein (green kept minimal)

Owner direction for the generator's UI/UX (2026-10-23): *"a cross between cyberpunk, AI tropes, and
frankenstein (but not a lot of green)."* This is the theme spec; it is design input for the macOS UI planned in
[`recipe-factory/UI/macOS-UI-PLAN.md`](../recipe-factory/UI/macOS-UI-PLAN.md) — whether the generator ships its
own panel or feeds that app, both consume this file. It maps the visual language onto the generator's actual
concepts, because the theme has to *mean* something per screen, not just look the part.

## 1. The governing metaphor

The generator is the **lab table, not the creature**. The model checkpoint is the raw material; the placement
plan is the assembly; the first serve is the moment the creature opens its eyes; the gate suite is the
restraint table (it must pass every strap before it's allowed out). Cyberpunk supplies the *aesthetic* (electric
city, glass and phosphor, "the machine is hungry for bandwidth"), AI tropes supply the *subject matter*
(neural architecture as anatomy, weights as tissue), Frankenstein supplies the *mood* (we built this thing
from parts that were never meant to fit together — 188 GiB of lookup table stitched to a 96 GiB card — and the
moment it works is both triumph and alarm).

**The one-line visual brief:** a storm-lit laboratory rack in a neon city, where every cable is a bandwidth
and every glow is a tier.

## 2. Palette (the "not a lot of green" rule)

Green is *reserved*. It appears only as the **qualified signal** (the creature passed the gates) and the
**"fits with margin"** bar — never as decoration. Everything else:

| Role | Color | Meaning |
|---|---|---|
| Background (dark storm) | `#0B0E14` near-black blue | the lab at night |
| Panel / card | `#121826` | glass + steel |
| Primary electric | `#5AD1FF` cyan | bandwidth, active current, the GPU tier |
| Secondary arc | `#A78BFA` violet | the CPU tier, the deferred work, the quiet hum |
| Heat / alarm | `#FF6B4A` ember orange | prefill pressure, power caps, the needle that must not drop |
| Bone / ivory | `#E8E4DA` | weights, tables, the anatomical parts (engram tables = bone) |
| Phosphor text | `#C9D4E3` | readable body |
| **Green (reserved)** | `#3DD68C` | qualified / gate-passed / fits — and nothing else |
| Blood warning | `#E5484D` | NaN row, NLL gate fail, OOM — the failed experiment |

Sparse accents of gold (`#D9A441`) for provenance/pins (revision locks). No gradients except the "storm"
backdrop; borders are 1px "seams".

## 3. Concept mapping (theme → screen)

| Generator concept | Visual treatment |
|---|---|
| Budget JSON (the inventory) | **The organ tray**: components as specimens — experts as muscle fiber (dense, striated), Engram tables as bone (dense, ivory), attention as nerve (cyan filaments). Header-only reads = "x-ray" mode (no weights downloaded = see through the body). |
| Tier assignment | **The assembly slab**: three illuminated troughs — VRAM (cyan, shallow, precious), DRAM (violet, deep), NVMe (ember, cold storage). Components slide into troughs; a misfit (overflow) sparks and shakes. The bar chart of VRAM/DRAM/NVMe bandwidths from `recipe-factory/METHODOLOGY.md` §4 is drawn as **three lightning rods** of different heights. |
| Hot-expert map | **The brain's wiring**: per-layer routing drawn as filaments from the router to the resident experts; CPU-routed experts flicker violet, GPU ones burn cyan. The 15 % CPU share is a "leak" meter. |
| Deferral D | **The stitched queue**: tokens march through the layer stack like stitches; deferred CPU work trails behind on a violet thread. D>4 shows the thread *fraying* (the NLL gate fails). |
| Calibration rule | **"The creature must open its eyes."** A projection is an **animated diagram on the slab** (monochrome, labeled ESTIMATE); it only renders in full color — cyan and violet lit — after it has been validated against a measured datapoint. The trust label is the switch: provisional = sheet over the creature; calibrated = sheet off. |
| Gate suite (NLL, needles, stress, consistency, health) | **The restraint table**: five straps, each stamped PASS/FAIL; the creature is not "alive" (qualified) until every strap is green — the only place green is allowed to flood. |
| Tried-options ledger | **The specimen jars** on the shelf behind the table: every rejected experiment preserved with its number ("DSpark K5, −69 % C4") — they are trophies of rigor, not failures. |
| Profiles | **Personalities of the creature**: coding-agent = wired for one master (single bright stream); multi-user = four streams braided; long-context = one vast slow river; power-capped = dimmed lab, amber economizer; quality-first = all straps tightened. |
| Lever tuning mode | **The console**: each lever a physical throw — expert split is the master breaker (the big one), chunk size a rotary, D a ratchet with a hard stop at 4 (the guard), speculation a covered switch with a "DO NOT THROW — reopen trigger: CPU-tier change" tag. Turning a lever re-lights the projections instantly; a lever that trips a guard refuses with a red flash and the measured reason. |
| Numbers | Every number carries its **provenance tag** like a sample label: evidence id, manifest hash (short), power condition. A number without a tag renders as glass — unreadable. |
| UI chrome | Terminal-flavored type (mono for numbers, humanist for prose), scanline shimmer on load, a subtle heartbeat pulse on the "resolved line". |

## 4. Typography and motion

- Display/mono: a square grotesque (JetBrains Mono / IBM Plex Mono) for all measurements, knob names, and the
  resolved line. Prose: Inter or a neutral humanist.
- Motion: fast (≤150 ms) mechanical transitions — levers click, troughs fill with a "current" shimmer. No
  bouncy easing; this is lab equipment, not a toy.
- Sound (optional, off by default): a soft contactor *clack* on lever throws, a low hum on the qualified
  screen.

## 5. Copy voice

The UI speaks like the lab notes of a methodical chef: short imperative lines ("Place the experts.", "Check the
straps.", "The creature opens its eyes."), every claim with a number and a source, no marketing adjectives.
The owner's phrase — **"mise en place for models"** — is the product's tagline and the empty-state headline.

## 6. Implementation notes

- Implement the palette as a single `ThemeTokens` (SwiftUI `Color` extensions / CSS variables), with
  `green` behind a semantic alias (`Color.qualified`) so it *cannot* be used decoratively by accident — the
  "not a lot of green" rule enforced in code, poteto-style ("the best rules are in your code").
- The trust-label switch (§3, calibration row) is the one interaction the whole app hangs on; prototype it
  first.
- This file is design input; it changes nothing the serving recipe reads.
