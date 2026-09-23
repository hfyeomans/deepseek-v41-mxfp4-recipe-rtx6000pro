# Research notes — index

Durable write-ups of design discussions that happened in Amp threads rather than in experiments. They preserve the numbers, tables, diagrams, assumptions and decision criteria so nothing lives only in a chat log. Written 2026-09-22 from the recipe repository's evidence plus the discussion thread; where a thread figure could not be matched to a file under `evidence/` it is labelled accordingly.

Every number in these notes carries one of five labels:

| Label | Meaning |
|---|---|
| **measured** | appears in `evidence/`, `docs/BENCHMARKS.md` or an experiment result in `plan/EXPERIMENTS.md` |
| **projected** | arithmetic from measured inputs; no run |
| **inferred** | a conclusion drawn from documentation or code reading, not from a run on this box |
| **pending** | an experiment that is registered but has no result |
| **failed** | a run that ended in an error; the error is the result |

Nothing here is a quotation of a person. Where the discussion thread's own wording is reproduced (the open-source plan), it is marked as the thread's assessment.

| Document | Covers |
|---|---|
| [HETEROGENEOUS-ROUTING-AND-OPTIMIZATION-NOTES.md](HETEROGENEOUS-ROUTING-AND-OPTIMIZATION-NOTES.md) | Exo / MLX / macOS 128 GB with the 2-GPU Linux box (transport, RTT break-even, Metal experts, Mac roles, bounded experiments with stop/go); third RTX PRO 6000 back-of-envelope (why not TP3, TP2 + asymmetric EP rank, projected C1/C3/C4/prefill); the interactive-vs-local router; the speculative-decoding (DSpark/MTP) roadmap; risks and follow-ups |
| [CURRENT-ARCHITECTURE-AND-ENGRAM.md](CURRENT-ARCHITECTURE-AND-ENGRAM.md) | What runs today: hardware, formats, TP2 placement, GPU/CPU experts, rank-0 kt-kernel path, file-backed Engram, attention/KV, breakable CUDA graphs, streamed prefill, deferral, request flow, bottlenecks, persistent vs ephemeral state; the Engram placement decision with its RAM accounting |
| [BENCHMARK-EVOLUTION-AND-E20-FP8.md](BENCHMARK-EVOLUTION-AND-E20-FP8.md) | E19 → E20 → E21 → E22 → E23 R0 decode and prefill tables, aggregate / per-stream / wall distinctions, TTFT by context length, current qualified numbers, speculation status; the E20 dense W8A8 FP8 tuning in detail (module, config selection, fallback, tuner gap, seven shapes, search, gate, validation, before/after) |
| [OPEN-SOURCE-PLAN.md](OPEN-SOURCE-PLAN.md) | Preserving the SGLang overlay reproducibly (base SHA, commit series, build/run, licences); candidate upstream PRs in reviewable units with evidence and scope limits; what is not upstream-ready; the current reproducibility gap |
| [MIGRATION-ROADMAP-PYTHON-TO-NATIVE-HYBRID.md](MIGRATION-ROADMAP-PYTHON-TO-NATIVE-HYBRID.md) | Staged migration from the current Python-coordinated hybrid to rank-balanced production ownership: low-coupling native work before multi-rank, Python semantic prototype, post-validation native consolidation, optional topology, production gates, rollback, and milestone metrics |
| [BIOS-TIER3-OVERCLOCK-RESEARCH.md](BIOS-TIER3-OVERCLOCK-RESEARCH.md) | Written by a research sub-thread 2026-09-23 from the BIOS guide, HARDWARE.md and the Boot 1/Boot 2 inventories: Shimada Peak (Zen 5) clock domains (MCLK/UCLK/FCLK/GMI), what is and is not sourced for the 9955WX (Zen 4 lore and single-system community reports labelled as such), the discriminating FCLK-vs-MEMCLK cold-boot plan with falsifiable predictions, Linux clock-reading options, the from-Auto cleanup matrix (verdict: leave nearly everything Auto), voltage boundaries (AM5-community context only, not authorisation), rollback on this board, payoff ceiling. **Integration note:** its T3-0 "measurement-only boot" was satisfied the same day without a boot — `scripts/hsmp-clocks.py` reads FCLK 2000 / MCLK 3200 live (BIOS guide §6, E27); its §5 payoff table's 28.6 tok/s baseline is the rounded 35 ms step, not the qualified C1 34.0 wall / 39.2 summed — carry the percentages, not the absolute tok/s |
| [MAINTAINABILITY-REVIEW-2026-09.md](MAINTAINABILITY-REVIEW-2026-09.md) | Maintainability review of the hybrid's graph/buffer ownership, shape invariants, correctness debt, release gates, overlay traceability, and staged refactoring boundaries |

Related living documents: [plan/ROADMAP.md](../../plan/ROADMAP.md) (readout and change log), [plan/EXPERIMENTS.md](../../plan/EXPERIMENTS.md) (E18–E24 protocols and results), [docs/BENCHMARKS.md](../BENCHMARKS.md) (tables), [docs/TROUBLESHOOTING.md](../TROUBLESHOOTING.md) (failure signatures), [docs/LEARNING.md](../LEARNING.md) §11 (state of evidence).
