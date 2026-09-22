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

Related living documents: [plan/ROADMAP.md](../../plan/ROADMAP.md) (readout and change log), [plan/EXPERIMENTS.md](../../plan/EXPERIMENTS.md) (E18–E24 protocols and results), [docs/BENCHMARKS.md](../BENCHMARKS.md) (tables), [docs/TROUBLESHOOTING.md](../TROUBLESHOOTING.md) (failure signatures), [docs/LEARNING.md](../LEARNING.md) §11 (state of evidence).
