# Recipe Factory — macOS UI Plan (SwiftUI)

A macOS app the owner could ship: build new serving recipes with the recipe-factory methodology, watch a
remote GPU box serve, pick profiles, explore trade-offs — all backed by the evidence JSONs the recipes
already produce. This document is the design; the full app is out of scope today. A compilable SwiftUI
skeleton with the data models and a sample evidence JSON lives in [RecipeFactoryUI/](RecipeFactoryUI/).

## 1. Product framing

**One-line:** "Recipes with receipts." A native macOS window onto the methodology in [../METHODOLOGY.md](../METHODOLOGY.md),
the knob taxonomy in [../KNOBS-AND-PROFILES.md](../KNOBS-AND-PROFILES.md) and the ledger in
[../TRIED-OPTIONS-LEDGER.md](../TRIED-OPTIONS-LEDGER.md). The differentiator vs everything in the landscape
(recipes.vllm.ai, Local AI Registry, cookbooks): every number in this UI is backed by manifest-stamped
evidence from a real serve, and every knob comes with its measured trade-off and its rejection history.

**Users:** (1) the owner — one GPU box, one or two recipes, wants TTFT/tok-s visibility and safe profile
switches; (2) later, anyone following a recipe — read-only dashboard + wizard.

## 2. Screens

### 2.1 Live Dashboard (the home screen)
Reads the evidence JSONs + the serve manifest (see [JSON-SCHEMA.md](JSON-SCHEMA.md)) pulled from the GPU box.
- **Top strip:** serve identity — model + snapshot, engine + overlay SHA, image digest, profile name +
  `overrides=[...]`, BIOS slot, power cap. Straight from `manifest.json`.
- **Headline cards:** C1 median/wall tok/s, C4 wall, prefill 16K/32K (cold/warm pair, TTFT), needles
  (n/6 at N tokens), NLL Δ vs baseline, stress wall, health (0 NaN / 0 Xid).
- **Trend sparkline:** last N runs per metric from the stamped JSONs; a number whose sibling manifest
  doesn't match is shown as *unverified* (greyed) — the UI enforces the same stamping rule CI does.
- **Sessions:** a picker over `evidence/<id>/` directories; each shows its README-styled question +
  result table.
- **Actions:** "Pull evidence now" (rsync over ssh), "Open benchmarks doc", "Show manifest JSON".

### 2.2 Recipe Builder Wizard (new recipes)
A step-through of P0→P4, each step generating files that land in the template layout:
1. **Model** — HF repo + snapshot (or local dir) → runs `checkpoint-budget.py` remotely (ssh) and shows
   the component byte table + the "what must be GPU-resident" split.
2. **Hardware** — inventory JSON (or the Local AI Registry record for the GPU) + measured bandwidths
   (pre-filled from a hardware library, flagged *estimate* until the P1 microbenchmarks run).
3. **Feasibility** — the placement calculator table across layouts × GPU counts (fits/does-not-fit,
   margins, decode ceiling), with the calibration-point check surfaced as a warning.
4. **Profile** — pick a use case (coding-agent / multi-user / long-context ladder / power-capped /
   quality-first) → a draft `qualified.env` with every knob annotated by its measured trade-off from
   the ledger; unknown knobs are declared in the typed registry before the wizard will write the file.
5. **Gates** — a checklist of the gate suite with one-click "run next gate" (ssh), each result stamped
   and filed under `evidence/`.
6. **Ship** — generates the repo skeleton (TEMPLATE/), the systemd unit, and the registry/CI files.

The wizard is deliberately *not* a black box: every step shows the command it runs and the file it
writes, so the methodology remains the product.

### 2.3 Profile Picker (safe switching)
Lists the profiles (`config/profiles/*.env`), each rendered as a diff against the qualified default
with the measured cost of each delta (e.g. `CONTEXT_LENGTH=1048576` → "+rung 1M×1, fraction must be
0.962, single slot"). Switching = editing the systemd drop-in + a serve restart, gated by a dry-run of
the typed registry (the UI runs `serve_config.py shell` over ssh and shows the refusal, verbatim,
before anything restarts).

### 2.4 Trade-off Explorer
The interactive version of KNOBS-AND-PROFILES §2: one axis per knob family (experts resident, chunk
size, deferral, slots, fraction…), the measured numbers as data points, and the guard boundaries drawn
on top (the regions the registry refuses). Hovering a point shows its evidence directory. The explorer
never extrapolates: outside the measured range it shows the guard, not a curve.

### 2.5 Ledger Browser
TRIED-OPTIONS-LEDGER.md as structured data: filter by family (engine / speculation / placement /
deferral / chunk / Engram / kernels / threads / BIOS), status (ADOPT/REJ/CLOSED/PARKED), and reopen
trigger. A "would I re-measure this?" panel answers: only if a listed reopen trigger holds on the
current hardware.

## 3. How it talks to the remote GPU box

The app never requires anything on the box beyond ssh + the recipe repo. Transport, in preference order:

1. **ssh exec (v1, chosen).** `Process` + `/usr/bin/ssh <alias>` (ssh config alias, no passwords in the
   app), or SwiftNIO SSH for a proper connection pool. Commands are the recipe's own scripts
   (`serve_config.py resolved`, `benchmarks_headline.py markdown`, `evidence_manifest.py show`), so the
   app executes the *same code path CI trusts* — no second implementation of the logic.
2. **File pull.** `rsync -az --exclude '*.pt'` of `evidence/` + `config/` into a local cache
   (`~/Library/Application Support/RecipeFactory/`), then read-only parsing. Evidence dirs are small
   (stamped JSONs); the `.pt` recorder blobs are excluded. Poll on app focus + a 5-minute timer +
   explicit "Pull now".
3. **Never:** a custom agent/daemon on the box in v1 (the box is busy serving; a daemon is v2 and only
   as a read-only `/healthz`-style evidence endpoint if polling latency ever matters).

Secrets: the API key stays a *path* on the box (the recipe's own `API_KEY_FILE` convention); the UI
never handles the token. Tailscale is the assumed network layer — the app's "Add box" asks for the ssh
alias only.

## 4. Distribution: DMG + notarization path (the shippable-later checklist)

1. **App structure:** Swift Package (skeleton shipped) → migrate to an `.xcodeproj` app target when the
   real UI work starts. Sandboxed app, network-client entitlement, no privileged helpers.
2. **Code signing:** Developer ID Application certificate; hardened runtime on.
3. **Notarization:** `notarytool submit` with an App Store Connect API key (Developer ID profile);
   staple with `xcrun stapler`. CI: GitHub Actions macOS runner does `xcodebuild archive → notarytool
   submit → stapler → hdiutil create`.
4. **Updates:** Sparkle 2 (EdDSA) with the DMG feed, or ad-hoc updates until the app has >1 user. The
   release branch policy mirrors the recipes': release artifacts carry a PROVENANCE entry (build commit,
   toolchain, notarization ticket hash).
5. **First milestone (v0.1):** Dashboard + Profile Picker, read-only, over one ssh box, DMG unsigned for
   personal use. **v0.2:** Recipe Builder wizard + Trade-off Explorer, signed + notarized. **v1.0:**
   Ledger browser + multi-box + Sparkle.

## 5. Data contract

The UI consumes exactly three JSON shapes — the serve manifest, the stamped evidence files, and the
generated headline block (schema in [JSON-SCHEMA.md](JSON-SCHEMA.md)). Rule: **the UI adds no schema the
recipes don't already emit.** If a screen needs a number the evidence discipline doesn't produce, the
answer is to extend the recipe's `evidence_manifest.py`/headline generator (which CI verifies), not to
invent a parallel format.

## 6. Risks and open questions (owner decisions)

- **Scope:** v0.1 (dashboard only) vs going straight to the wizard — the wizard needs the most recipe
  changes (registry-driven file generation), so it is the last screen to build.
- **Transport:** SwiftNIO SSH dependency vs plain `Process`+ssh — Process+ssh is fewer moving parts and
  reuses the user's ssh config/keys; SwiftNIO SSH enables streaming logs later.
- **App vs companion:** a menu-bar live status (C1 tok/s, TTFT p99, health dot) is cheap on top of the
  dashboard and likely the daily-driver surface; propose it in v0.1.
- **Registry publishing:** whether Trade-off Explorer data should also publish to the Local AI Registry
  (their API accepts normalized recipe records) — a "publish" button is a natural v1 feature.
