# Recipe Factory — JSON Schema the UI consumes

The UI reads exactly the JSON the recipe discipline already produces — no parallel formats. Three shapes:

1. **`manifest.json`** — the per-serve evidence manifest, written by `scripts/evidence_manifest.py`
   beside every stamped evidence file.
2. **Stamped evidence files** — the benchmark outputs (`decode.json`, `prefill.json`, `verify.json`,
   `stress-kv.json`, `nll.json`), each carrying `evidence.manifest_hash` and sitting beside the
   manifest it was produced under.
3. **The generated headline block** — the machine-readable form of the top table in
   `docs/BENCHMARKS.md` (emitted by `scripts/benchmarks_headline.py markdown`; the commit-time JSON is
   the source the UI should read).

Below is the v1 draft. Field names match the reference implementation in the parent repo
(`evidence_manifest.py`, the `bench-*` scripts); where a field is recipe-specific it is marked
`(extension)`. The rule: a UI screen that needs a number not in this schema extends the *generator*,
and CI keeps the generator honest — the UI never invents data.

---

## 1. Serve manifest (`manifest.json`)

```json
{
  "$schema": "https://example.invalid/recipe-factory/manifest-v1.json",
  "recipe": { "repo_sha": "40-hex", "dirty": false },
  "engine": {
    "name": "sglang-hybrid",
    "base": "pinned-upstream-commit",
    "overlay_sha": "40-hex",
    "image_digest": "sha256:...",
    "container_id": "12-hex or null"
  },
  "model": { "repo": "org/model", "snapshot": "40-hex" },
  "config": {
    "profile": "qualified",
    "overrides": ["KNOB", "..."],
    "resolved": "resolved: experts gpu=219 cpu=165 deferral D=4 ... profile=qualified overrides=[...]",
    "knobs": { "KNOB_NAME": "value" }
  },
  "box": {
    "bios_slot": "3",
    "hsmp": { "fclk_mhz": 2100, "mclk_mhz": 3200 },
    "gpu_power_cap_w": 325,
    "gpus": 2,
    "meminfo": { "MemTotal_kB": 0, "MemAvailable_kB": 0 },
    "engram_residency": { "layer1": 0.0, "layer14": 0.0 }
  },
  "created_utc": "2026-10-23T14:57:16Z"
}
```

UI mapping: the Dashboard's top strip is `config.profile + overrides` and `engine.overlay_sha`; the
cards' *serve identity* column is this whole object. `box.hsmp.fclk_mhz` vs `box.bios_slot` is shown as
a mismatch warning when they disagree with the declared profile.

## 2. Stamped evidence files (common envelope)

```json
{
  "evidence": { "manifest_hash": "sha256-hex-of-manifest.json", "manifest": "manifest.json" },
  "benchmark": "decode | prefill | verify | stress | nll | humaneval | power",
  "args": { "...": "the exact invocation" },
  "results": [ ]
}
```

### 2.1 `bench-decode.py` output (C1/C4 contract)

```json
{ "evidence": { "manifest_hash": "..." },
  "benchmark": "decode",
  "args": { "concurrency": 1, "output_tokens": 512, "warmups": 1, "runs": 6,
            "distinct_prompts": false, "same_prompt": false, "seed": 42 },
  "results": [
    { "run": 0, "per_seq_tok_s": [58.4], "aggregate_tok_s": 58.4, "wall_tok_s": 56.2,
      "ttft_s": 3.7, "spec_acceptance": null, "power_busy_median_w": [205.0, 218.0] }
  ],
  "summary": { "c1_median_tok_s": 58.4, "c1_wall_tok_s": 56.2 }
}
```

`spec_acceptance` is the Prometheus counter delta per run; `null` when speculation is off (the recipe's
default). `--same-prompt` runs carry `"same_prompt": true` and the UI labels them "consistency gate, not
throughput" wherever they appear.

### 2.2 `bench-prefill.py` output (cold/warm pair)

```json
{ "evidence": { "manifest_hash": "..." },
  "benchmark": "prefill",
  "args": { "prompt_tokens": [8192, 16384, 32768], "runs": 3, "warm_pass": true, "seed": 777 },
  "passes": [
    { "pass": "cold", "residency_before": { "layer1": 0.001, "layer14": 0.001 },
      "sizes": [ { "prompt_tokens": 8192,
                   "runs": [ { "ttft_s": 3.71, "effective_tok_s": 4344.0 } ] } ] },
    { "pass": "warm", "residency_before": { "layer1": 0.42, "layer14": 0.44 },
      "sizes": [ { "prompt_tokens": 8192,
                   "runs": [ { "ttft_s": 3.56, "effective_tok_s": 4504.0 } ] } ] }
  ]
}
```

The UI renders cold and warm as paired bars with the residency recorded beside them; a pass without
`residency_before` is rendered as *unverified cache state* (the reference project measured a 1,471 vs
1,820 tok/s drift on unrecorded state).

### 2.3 `verify.py` output (functional + needles)

```json
{ "evidence": { "manifest_hash": "..." },
  "benchmark": "verify",
  "args": { "context": 262144 },
  "results": {
    "alias_ok": true, "max_model_len": 262144, "arithmetic_ok": true,
    "thinking_toggle_ok": true, "tool_call_ok": true,
    "needles": { "found": 6, "placed": 6, "prompt_tokens": 255388,
                 "context": 262144, "finish_reason_stop": true, "ttft_s": 59.0 }
  }
}
```

### 2.4 `stress-kv-full.py` output

```json
{ "evidence": { "manifest_hash": "..." },
  "benchmark": "stress",
  "args": { "slots": 4, "context": 262144 },
  "results": { "wall_s": 278.0, "ttft_max_s": 271.0, "all_finished": true,
               "peak_vram_mib": [97333, 97214], "min_free_mib": [2533, 2468] }
}
```

### 2.5 `nll-score.py` output (identity gate)

```json
{ "evidence": { "manifest_hash": "..." },
  "benchmark": "nll",
  "args": { "corpus_tokens": 65653, "baseline": "evidence/t328/default-219-8192/corpus" },
  "results": { "delta_nll": -0.00033, "delta_sd": 0.0013, "z": 0.26,
               "moved": 65226, "total": 65653, "gate": "pass" }
}
```

## 3. Headline block (machine-readable form)

```json
{ "generated_by": "benchmarks_headline.py",
  "rungs": [
    { "rung": "262K x 4", "launcher_default": true,
      "c1": { "median_tok_s": 60.8, "wall_tok_s": 58.8 },
      "c4": { "wall_tok_s": 118.9 },
      "prefill_16k": { "cold_tok_s": 4344.0, "cold_ttft_s": 3.7 },
      "prefill_32k": { "cold_tok_s": 4495.0, "cold_ttft_s": 7.1 },
      "needles": { "found": 6, "of": 6, "prompt_tokens": 255388, "ttft_s": 59.0 },
      "stress": { "wall_s": 278.0, "shape": "C4 x 253428 tokens" },
      "nll": { "delta": -0.00033, "sd": 0.0013, "z": 0.26, "moved": 65226, "total": 65653,
               "baseline": "evidence/t328/default-219-8192/corpus" },
      "evidence": "evidence/t352/default-flip" }
  ]
}
```

## 4. UI-side extensions (kept out of the recipe's core files)

The app may *derive* (never require) two local indexes, cached under Application Support:

```json
{ "kind": "ledger-index",
  "entries": [ { "id": "E23", "family": "speculation", "option": "DSpark K5",
                 "outcome": "C1 -16%, C3/C4 -69..-77%", "verdict": "REJ", "reopen": "CPU-tier change" } ] }

{ "kind": "profile-diff",
  "profile": "long-context-1m", "vs": "qualified",
  "deltas": [ { "knob": "CONTEXT_LENGTH", "from": "262144", "to": "1048576",
                "cost": "single slot; fraction must be 0.962" } ] }
```

If these ever need to be authoritative (multi-user sync), they move into the recipe repo as generated
files with their own staleness check — same discipline as the headline block.
